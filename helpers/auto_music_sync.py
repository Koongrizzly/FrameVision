
"""
Music Clip Creator / Auto Music Sync for FrameVision

Integration from tools_tab.py:

    from helpers.auto_music_sync import install_auto_music_sync_tool
    sec_musicclip = CollapsibleSection("Music Clip Creator", expanded=False)
    install_auto_music_sync_tool(self, sec_musicclip)
"""

from __future__ import annotations

import os
import sys
import math
import json
import random
import shutil
import subprocess
import tempfile
import re
from datetime import datetime
from dataclasses import dataclass, replace, field
from typing import List, Optional, Tuple, Dict

from PySide6.QtCore import Qt, QThread, Signal, QSettings, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QComboBox,
    QCheckBox,
    QProgressBar,
    QMessageBox,
    QGraphicsOpacityEffect,
    QSizePolicy,
    QGroupBox,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QDialog,
    QDialogButtonBox,
    QTabWidget,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QFrame,
    QTextEdit,
)
from PySide6.QtGui import QIcon
from .visual_thumbs import VisualThumbManager




# Optional timeline panel (separate module to keep this file smaller)
try:
    from .timeline import TimelinePanel
except Exception:  # pragma: no cover - fallback if helpers package/module not available
    TimelinePanel = None  # type: ignore
# ---------------------------- small helpers --------------------------------


def _find_ffmpeg_from_env() -> str:
    env_ffmpeg = os.environ.get("FV_FFMPEG")
    if env_ffmpeg and os.path.exists(env_ffmpeg):
        return env_ffmpeg

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "presets", "bin", "ffmpeg.exe"),
        os.path.join(here, "..", "presets", "bin", "ffmpeg"),
        "ffmpeg.exe",
        "ffmpeg",
    ]
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            return c

    return "ffmpeg"


def _find_ffprobe_from_env() -> str:
    env_ffprobe = os.environ.get("FV_FFPROBE")
    if env_ffprobe and os.path.exists(env_ffprobe):
        return env_ffprobe

    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "..", "presets", "bin", "ffprobe.exe"),
        os.path.join(here, "..", "presets", "bin", "ffprobe"),
        "ffprobe.exe",
        "ffprobe",
    ]
    for c in candidates:
        if shutil.which(c) or os.path.exists(c):
            return c

    return "ffprobe"




def _cine_grid_dims(screen_count: int) -> tuple[int, int]:
    """Return (cols, rows) for 2–9 tiles so the grid cleanly fills the frame.

    We keep it simple and deterministic: every value 2–9 maps to a fixed
    (cols, rows) pair with cols * rows == screen_count, so there are no
    unused "empty" cells in the xstack layout.
    """
    try:
        n = int(screen_count)
    except Exception:
        n = 4
    if n < 2:
        n = 2
    if n > 9:
        n = 9

    layout_map = {
        2: (2, 1),  # two vertical panels
        3: (3, 1),  # three vertical panels
        4: (2, 2),
        5: (5, 1),
        6: (3, 2),
        7: (7, 1),
        8: (4, 2),
        9: (3, 3),
    }
    return layout_map.get(n, (n, 1))


def _cine_grid_layout(screen_count: int, target_w: int, target_h: int):
    """Return per-tile (w, h, x, y) for 2–9 tiles.

    Layouts are tuned so that, for example, 5 ⇒ 3+2 and 7 ⇒ 4+3 rows,
    which keeps the overall grid closer to a cinematic 16:9 aspect ratio
    and avoids the ultra‑wide strip look while still filling the frame.
    """
    try:
        n = int(screen_count)
    except Exception:
        n = 4
    if n < 1:
        n = 1
    if n > 9:
        n = 9

    # Keep targets encoder-friendly (yuv420p / nvenc can be picky),
    # and avoid edge cases where padding ends up 1px smaller than the
    # scaled input due to rounding / SAR.
    try:
        target_w = int(target_w)
        target_h = int(target_h)
    except Exception:
        pass
    if target_w % 2:
        target_w += 1
    if target_h % 2:
        target_h += 1

    def _evenize_sizes(sizes: list[int], total: int) -> list[int]:
        """Make a list of sizes that sum to total, with each entry >= 2 and even."""
        if total % 2:
            total += 1
        if not sizes:
            return sizes
        if len(sizes) == 1:
            v = max(2, int(total))
            if v % 2:
                v += 1
            return [v]

        out = []
        for s in sizes[:-1]:
            v = max(2, int(s))
            if v % 2:
                v -= 1
            v = max(2, v)
            out.append(v)

        last = int(total) - sum(out)
        last = max(2, last)
        # Since total and sum(out) are both even, last should already be even.
        if last % 2:
            last -= 1
            if last < 2:
                last = 2
        out.append(last)

        # Final guard: fix any drift by adjusting the last cell.
        drift = int(total) - sum(out)
        if drift != 0:
            out[-1] = max(2, out[-1] + drift)
            if out[-1] % 2:
                out[-1] += 1
        return out

    # Rows are described as "how many tiles in this row", top to bottom.
    row_patterns = {
        1: [1],
        2: [2],
        3: [3],
        4: [2, 2],
        5: [3, 2],
        6: [3, 3],
        7: [4, 3],
        8: [4, 4],
        9: [3, 3, 3],
    }
    rows_spec = row_patterns.get(n, [n])
    total_rows = len(rows_spec)

    # Split the height evenly between rows, put any remainder on the last row
    # so the stacked rows always fill the frame top‑to‑bottom.
    base_h = max(1, target_h // max(1, total_rows))
    heights = [base_h for _ in range(total_rows)]
    used_h = base_h * total_rows
    remainder_h = target_h - used_h
    if remainder_h != 0 and heights:
        heights[-1] = max(1, heights[-1] + remainder_h)

    heights = _evenize_sizes(heights, target_h)

    layout = []
    y = 0
    idx_tile = 0
    for row_idx, cols_in_row in enumerate(rows_spec):
        if idx_tile >= n:
            break
        h = heights[row_idx]
        cols_in_row = max(1, int(cols_in_row))
        base_w = max(1, target_w // cols_in_row)
        widths = [base_w for _ in range(cols_in_row)]
        used_w = base_w * cols_in_row
        remainder_w = target_w - used_w
        if remainder_w != 0 and widths:
            widths[-1] = max(1, widths[-1] + remainder_w)

        widths = _evenize_sizes(widths, target_w)

        x = 0
        for col in range(cols_in_row):
            if idx_tile >= n:
                break
            w = widths[col]
            layout.append((w, h, x, y))
            x += w
            idx_tile += 1
        y += h

    return layout

def _run_ffmpeg(cmd: List[str]) -> Tuple[int, str]:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        out, _ = proc.communicate()
        return proc.returncode, out or ""
    except Exception as e:
        return 1, f"Failed to run ffmpeg: {e!r}"


def _ffprobe_duration(ffprobe: str, path: str) -> Optional[float]:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return float(out.strip())
    except Exception:
        return None



def _ffprobe_resolution(ffprobe: str, path: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) for the first video stream, or None."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if not out or "x" not in out:
            return None
        w_s, h_s = out.split("x", 1)
        w = int(float(w_s.strip()))
        h = int(float(h_s.strip()))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        return None
    return None


def _ffprobe_video_bitrate(ffprobe: str, path: str) -> Optional[int]:
    """Return approximate *video* bitrate in bits/sec, or None.

    Tries stream bit_rate, then format bit_rate minus audio, then size/duration estimate.
    """
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate,size:stream=index,codec_type,bit_rate",
        "-of",
        "json",
        path,
    ]
    try:
        raw = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        data = json.loads(raw or "{}")
    except Exception:
        return None

    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    def _to_int(v) -> int:
        try:
            if v is None:
                return 0
            s = str(v).strip()
            if not s or s.lower() in ("n/a", "nan"):
                return 0
            return int(float(s))
        except Exception:
            return 0

    def _to_float(v) -> float:
        try:
            if v is None:
                return 0.0
            s = str(v).strip()
            if not s or s.lower() in ("n/a", "nan"):
                return 0.0
            return float(s)
        except Exception:
            return 0.0

    v_br = 0
    a_br = 0
    for st in streams:
        ct = (st.get("codec_type") or "").lower()
        if ct == "video" and v_br <= 0:
            v_br = _to_int(st.get("bit_rate"))
        elif ct == "audio" and a_br <= 0:
            a_br = _to_int(st.get("bit_rate"))

    if v_br > 0:
        return v_br

    fmt_br = _to_int(fmt.get("bit_rate"))
    if fmt_br > 0:
        if a_br > 0 and fmt_br > a_br:
            return int(fmt_br - a_br)
        return fmt_br

    dur = _to_float(fmt.get("duration"))
    size = _to_int(fmt.get("size"))
    if dur > 0.01 and size > 0:
        est_total = int((size * 8) / dur)
        if a_br > 0 and est_total > a_br:
            return int(est_total - a_br)
        return est_total

    return None


def _ffprobe_fps_expr(ffprobe: str, path: str) -> Optional[str]:
    """Return an ffmpeg-friendly FPS expression (e.g. '30000/1001') for v:0, or None."""
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except Exception:
        return None

    avg = None
    rfr = None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("avg_frame_rate="):
            avg = line.split("=", 1)[1].strip()
        elif line.startswith("r_frame_rate="):
            rfr = line.split("=", 1)[1].strip()

    def _clean(expr: Optional[str]) -> Optional[str]:
        if not expr:
            return None
        expr = expr.strip()
        if expr in ("0/0", "0", "N/A", "nan"):
            return None
        return expr

    avg = _clean(avg)
    rfr = _clean(rfr)
    return avg or rfr

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)




# ------------------------------------------------------------------
# Filename sanitizers (Windows-safe)
# ------------------------------------------------------------------

_INVALID_FILENAME_CHARS = set('<>:"/\\|?*')

def _sanitize_stem(stem: str, fallback: str = "output") -> str:
    """Return a filesystem-safe filename stem (no extension)."""
    try:
        s = str(stem)
    except Exception:
        s = fallback
    s = s.strip().replace(" ", "_")
    # Replace control chars and Windows-forbidden characters.
    out = []
    for ch in s:
        o = ord(ch)
        if o < 32 or ch in _INVALID_FILENAME_CHARS:
            out.append("_")
        else:
            out.append(ch)
    s = "".join(out)
    s = re.sub(r"_+", "_", s).strip(" ._")
    if not s:
        s = fallback
    # Keep it reasonably short to avoid path length issues.
    if len(s) > 120:
        s = s[:120].rstrip(" ._")
        if not s:
            s = fallback
    return s

def _sanitize_filename(name: str, fallback: str = "output.mp4") -> str:
    """Return a filesystem-safe filename (keeps extension if present)."""
    try:
        n = str(name)
    except Exception:
        return fallback
    n = n.strip()
    stem, ext = os.path.splitext(n)
    safe_stem = _sanitize_stem(stem, fallback=os.path.splitext(fallback)[0] or "output")
    safe_ext = ext
    # If extension has forbidden chars, drop it.
    if any((ch in _INVALID_FILENAME_CHARS or ord(ch) < 32) for ch in safe_ext):
        safe_ext = ""
    out = safe_stem + safe_ext
    # Avoid empty result
    if not out or out.strip(" ._") == "":
        out = fallback
    return out


def _load_clip_presets() -> List[Tuple[str, str, str]]:
    """Load 1-click clip presets from JSON, with minimal built-in fallback.

    JSON location (relative to app root):
        /presets/setsave/clip_presets.json

    Schema:
        {
            "version": 1,
            "presets": [
                {
                    "id": "clean",
                    "name": "Clean cuts (no FX)",
                    "description": "...",
                    "settings": { ... }
                },
                ...
            ]
        }
    """
    # Only keep the two "safe" built-ins as a fallback so the dialog
    # still works even if the JSON file is missing or broken.
    default_presets: List[Tuple[str, str, str]] = [
        ("clean", "Clean cuts (no FX)", "Very clean edit, slower cuts, no extra FX."),
        ("chill", "Chill / slow cinematic", "Soft cuts with gentle slow-motion and cinematic moves."),
    ]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(here, "..", "presets", "setsave", "clip_presets.json")
        if not os.path.exists(json_path):
            return default_presets

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("presets", [])
        if not isinstance(items, list):
            return default_presets

        presets: List[Tuple[str, str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or "").strip()
            if not pid:
                continue
            name = str(item.get("name") or "").strip() or pid
            desc = str(item.get("description") or "").strip()
            presets.append((pid, name, desc))

        # If JSON is empty or invalid, fall back to the two built-ins.
        return presets or default_presets
    except Exception:
        # Any error: fall back to the two built-in presets so the UI keeps working.
        return default_presets


def _get_clip_preset_settings(preset_id: str) -> dict | None:
    """Return the settings dict for a given preset id from clip_presets.json.

    This is used by _apply_preset so that all 1-click presets (including
    user-defined ones) live in JSON instead of being hardcoded here.

    Returns None if the file is missing, broken or the preset is not found.
    """
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(here, "..", "presets", "setsave", "clip_presets.json")
        if not os.path.exists(json_path):
            return None

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        items = data.get("presets", [])
        if not isinstance(items, list):
            return None

        for item in items:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("id") or "").strip()
            if pid != preset_id:
                continue
            settings = item.get("settings")
            if isinstance(settings, dict):
                return settings
            return None
    except Exception:
        return None
    return None




def _clip_presets_json_path() -> str:
    """Return absolute path to presets/setsave/clip_presets.json."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "presets", "setsave", "clip_presets.json")


def _read_clip_presets_json() -> dict:
    """Read clip_presets.json. If missing/broken, return an empty structure."""
    path = _clip_presets_json_path()
    try:
        if not os.path.exists(path):
            return {"version": 1, "presets": []}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "presets": []}
        if "version" not in data:
            data["version"] = 1
        if not isinstance(data.get("presets"), list):
            data["presets"] = []
        return data
    except Exception:
        return {"version": 1, "presets": []}


def _atomic_write_json(path: str, data: dict) -> None:
    """Atomic JSON write with .tmp and .bak (best-effort)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    bak = path + ".bak"
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass
    # backup old
    try:
        if os.path.exists(path):
            try:
                if os.path.exists(bak):
                    os.remove(bak)
            except Exception:
                pass
            try:
                os.replace(path, bak)
            except Exception:
                # fallback: copy
                try:
                    shutil.copy2(path, bak)
                except Exception:
                    pass
    except Exception:
        pass
    os.replace(tmp, path)


def _write_clip_presets_json(data: dict) -> Tuple[bool, str]:
    """Validate and write clip_presets.json. Returns (ok, message)."""
    try:
        if not isinstance(data, dict):
            return False, "Invalid data (not a dict)."
        presets = data.get("presets")
        if not isinstance(presets, list):
            return False, "Invalid presets list."
        seen = set()
        for item in presets:
            if not isinstance(item, dict):
                return False, "Preset item is not an object."
            pid = str(item.get("id") or "").strip()
            if not pid:
                return False, "Preset id cannot be empty."
            if pid in seen:
                return False, f"Duplicate preset id: {pid}"
            seen.add(pid)
            if not isinstance(item.get("settings", {}), dict):
                return False, f"Preset '{pid}' settings must be an object."
        if "version" not in data:
            data["version"] = 1
        path = _clip_presets_json_path()
        _atomic_write_json(path, data)
        return True, "Saved."
    except Exception as e:
        return False, f"Failed to save: {e}"


class ClipPresetManagerDialog(QDialog):
    """Edit clip_presets.json without touching the 1-click 'run videoclip' flow."""

    def __init__(self, parent: QWidget, capture_fx_cb, apply_fx_cb):
        super().__init__(parent)
        self.setWindowTitle("Preset manager")
        self.resize(860, 520)
        self._capture_fx_cb = capture_fx_cb
        self._apply_fx_cb = apply_fx_cb
        self._data = _read_clip_presets_json()
        self._current_index: int | None = None

        root = QHBoxLayout(self)

        # left: list + actions
        left = QVBoxLayout()
        self.list = QListWidget(self)
        left.addWidget(self.list, 1)

        row_btns = QHBoxLayout()
        self.btn_new = QPushButton("New", self)
        self.btn_dup = QPushButton("Duplicate", self)
        self.btn_del = QPushButton("Delete", self)
        row_btns.addWidget(self.btn_new)
        row_btns.addWidget(self.btn_dup)
        row_btns.addWidget(self.btn_del)
        row_btns.addStretch(1)
        left.addLayout(row_btns)

        root.addLayout(left, 1)

        # right: editor
        right = QVBoxLayout()

        form = QFormLayout()
        self.edit_id = QLineEdit(self)
        self.edit_name = QLineEdit(self)
        self.edit_desc = QLineEdit(self)
        form.addRow("ID:", self.edit_id)
        form.addRow("Name:", self.edit_name)
        form.addRow("Description:", self.edit_desc)
        right.addLayout(form)

        # capture + edit controls
        row_caps = QHBoxLayout()
        self.btn_capture = QPushButton("Capture current Options + Advanced", self)
        self.btn_capture.setToolTip("Save current FX/transitions/effects settings into this preset.")
        self.btn_edit = QPushButton("Edit", self)
        self.btn_edit.setToolTip("Unlock Settings JSON for manual editing (advanced).")
        row_caps.addWidget(self.btn_capture, 1)
        row_caps.addWidget(self.btn_edit, 0)
        right.addLayout(row_caps)

        right.addWidget(QLabel("Settings (FX-only):", self))
        self.text_settings = QTextEdit(self)
        self.text_settings.setReadOnly(True)
        right.addWidget(self.text_settings, 1)

        self.btn_save_item = QPushButton("Apply edits to this preset", self)
        right.addWidget(self.btn_save_item)

        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)
        right.addWidget(buttons)

        try:
            btn_use = buttons.button(QDialogButtonBox.Apply)
            if btn_use is not None:
                btn_use.setText("Use this preset")
                btn_use.setToolTip("Apply this preset to the main UI (does not save the JSON file).")
        except Exception:
            pass

        root.addLayout(right, 2)

        # signals
        self.list.currentRowChanged.connect(self._on_select)
        self.btn_new.clicked.connect(self._on_new)
        self.btn_dup.clicked.connect(self._on_dup)
        self.btn_del.clicked.connect(self._on_del)
        self.btn_capture.clicked.connect(self._on_capture)
        self.btn_edit.clicked.connect(self._on_toggle_edit)
        self.btn_save_item.clicked.connect(self._on_save_item)
        buttons.accepted.connect(self._on_save_all)
        buttons.rejected.connect(self.reject)
        try:
            btn_use = buttons.button(QDialogButtonBox.Apply)
            if btn_use is not None:
                btn_use.clicked.connect(self._on_use_preset)
        except Exception:
            pass

        self._refresh_list(select_first=True)

    def _set_settings_editable(self, editable: bool) -> None:
        try:
            self.text_settings.setReadOnly(not editable)
        except Exception:
            pass
        try:
            self.btn_edit.setText("Lock" if editable else "Edit")
            self.btn_edit.setToolTip(
                "Lock Settings JSON (recommended)." if editable else "Unlock Settings JSON for manual editing (advanced)."
            )
        except Exception:
            pass

    def _on_toggle_edit(self) -> None:
        try:
            self._set_settings_editable(self.text_settings.isReadOnly())
        except Exception:
            # safest fallback
            try:
                self.text_settings.setReadOnly(False)
                self.btn_edit.setText("Lock")
            except Exception:
                pass

    def _presets(self) -> list:
        items = self._data.get("presets")
        return items if isinstance(items, list) else []

    def _refresh_list(self, select_first: bool = False) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for item in self._presets():
            pid = str(item.get("id") or "")
            name = str(item.get("name") or pid)
            it = QListWidgetItem(f"{name}  [{pid}]", self.list)
            it.setData(Qt.UserRole, pid)
        self.list.blockSignals(False)
        if select_first and self.list.count() > 0:
            self.list.setCurrentRow(0)
        elif self.list.count() == 0:
            self._current_index = None
            self.edit_id.setText("")
            self.edit_name.setText("")
            self.edit_desc.setText("")
            self.text_settings.setPlainText("{}")

    def _load_into_editor(self, idx: int) -> None:
        presets = self._presets()
        if idx < 0 or idx >= len(presets):
            return
        self._current_index = idx
        item = presets[idx]
        self.edit_id.setText(str(item.get("id") or ""))
        self.edit_name.setText(str(item.get("name") or ""))
        self.edit_desc.setText(str(item.get("description") or ""))
        settings = item.get("settings") if isinstance(item.get("settings"), dict) else {}
        self.text_settings.setPlainText(json.dumps(settings, ensure_ascii=False, indent=2))
        # default to locked when switching presets
        self._set_settings_editable(False)

    def _on_select(self, row: int) -> None:
        self._load_into_editor(row)

    def _message(self, title: str, text: str, icon=QMessageBox.Information) -> None:
        m = QMessageBox(self)
        m.setIcon(icon)
        m.setWindowTitle(title)
        m.setText(text)
        m.exec()

    def _on_new(self) -> None:
        presets = self._presets()
        base_id = "new_preset"
        n = 1
        ids = {str(p.get('id') or '') for p in presets}
        pid = base_id
        while pid in ids:
            n += 1
            pid = f"{base_id}_{n}"
        presets.append({"id": pid, "name": "New preset", "description": "", "settings": {}})
        self._refresh_list()
        self.list.setCurrentRow(len(presets) - 1)

    def _on_dup(self) -> None:
        idx = self.list.currentRow()
        presets = self._presets()
        if idx < 0 or idx >= len(presets):
            return
        src = presets[idx]
        ids = {str(p.get('id') or '') for p in presets}
        pid0 = str(src.get('id') or 'preset').strip() or 'preset'
        pid = pid0 + "_copy"
        n = 1
        while pid in ids:
            n += 1
            pid = f"{pid0}_copy{n}"
        presets.append({
            "id": pid,
            "name": str(src.get('name') or pid) + " (copy)",
            "description": str(src.get('description') or ""),
            "settings": dict(src.get('settings') or {}),
        })
        self._refresh_list()
        self.list.setCurrentRow(len(presets) - 1)

    def _on_del(self) -> None:
        idx = self.list.currentRow()
        presets = self._presets()
        if idx < 0 or idx >= len(presets):
            return
        pid = str(presets[idx].get('id') or '')
        resp = QMessageBox.question(self, "Delete preset", f"Delete preset '{pid}'?", QMessageBox.Yes | QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        presets.pop(idx)
        self._refresh_list(select_first=True)

    def _on_capture(self) -> None:
        idx = self.list.currentRow()
        presets = self._presets()
        if idx < 0 or idx >= len(presets):
            return
        try:
            settings = self._capture_fx_cb() or {}
            if not isinstance(settings, dict):
                settings = {}
            presets[idx]["settings"] = settings
            self.text_settings.setPlainText(json.dumps(settings, ensure_ascii=False, indent=2))
        except Exception as e:
            self._message("Capture failed", str(e), QMessageBox.Warning)

    
    def _on_use_preset(self) -> None:
        """Apply the selected preset to the main UI without saving JSON."""
        idx = self.list.currentRow()
        presets = self._presets()
        if idx < 0 or idx >= len(presets):
            return
        # keep in-memory edits (id/name/desc) in sync
        try:
            self._on_save_item()
        except Exception:
            pass
        settings = presets[idx].get("settings") or {}
        if not isinstance(settings, dict):
            settings = {}
        try:
            if callable(self._apply_fx_cb):
                self._apply_fx_cb(settings)
        except Exception as e:
            try:
                self._message("Use preset failed", str(e), QMessageBox.Warning)
            except Exception:
                pass

    def _on_save_item(self) -> None:
        idx = self.list.currentRow()
        presets = self._presets()
        if idx < 0 or idx >= len(presets):
            return

        # validate id
        new_id = self.edit_id.text().strip()
        if not new_id:
            self._message("Invalid preset", "Preset ID cannot be empty.", QMessageBox.Warning)
            return

        # prevent duplicate ids (except for the current row)
        for j, p in enumerate(presets):
            if j == idx:
                continue
            if str(p.get("id") or "").strip() == new_id:
                self._message("Invalid preset", f"Preset ID '{new_id}' already exists.", QMessageBox.Warning)
                return

        # parse settings JSON (always, so edits actually persist)
        raw = self.text_settings.toPlainText().strip() or "{}"
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Settings must be a JSON object (dictionary).")
        except Exception as e:
            self._message("Invalid settings JSON", f"Could not parse Settings JSON:\n{e}", QMessageBox.Warning)
            return

        presets[idx]["id"] = new_id
        presets[idx]["name"] = self.edit_name.text().strip()
        presets[idx]["description"] = self.edit_desc.text().strip()
        presets[idx]["settings"] = parsed
        self._refresh_list()
        self.list.setCurrentRow(idx)

    def _on_save_all(self) -> None:
        # apply current edits to selected first
        try:
            self._on_save_item()
        except Exception:
            pass
        ok, msg = _write_clip_presets_json(self._data)
        if not ok:
            self._message("Save failed", msg, QMessageBox.Warning)
            return
        self.accept()

# --------------------------- data classes ----------------------------------


@dataclass
class Beat:
    time: float
    strength: float
    kind: str  # "major" / "minor"


@dataclass
class Section:
    start: float
    end: float
    kind: str  # "intro", "verse", "chorus", "drop", "break"


@dataclass
class ClipSource:
    path: str
    duration: float
    is_image: bool = False


@dataclass
class TimelineSegment:
    clip_path: str
    clip_start: float
    duration: float
    effect: str
    energy_class: str  # "low", "mid", "high"
    transition: str    # "none", "fade", "flashcut"
    slow_factor: float = 1.0  # 1.0 = normal speed, <1.0 = slow motion (video only)
    # Cinematic one-off effects (freeze, stutter, reverse, speedups, ramp)
    cine_freeze: bool = False
    cine_stutter: bool = False
    cine_reverse: bool = False
    # Prism whip (horizontal drift)
    cine_tear_v: bool = False
    # Prism whip (vertical drift)
    cine_tear_h: bool = False
    # Speedup hits (play the source forward/backwards faster and repeat if needed)
    cine_speedup_forward: bool = False
    cine_speedup_backward: bool = False
    cine_speedup_forward_factor: float = 1.0
    cine_speedup_backward_factor: float = 1.0
    cine_speed_ramp: bool = False
    cine_freeze_len: float = 0.0
    cine_freeze_zoom: float = 0.0
    cine_tear_v_strength: float = 0.0
    cine_tear_h_strength: float = 0.0
    # Color-cycle glitch (cinematic)
    cine_color_cycle: bool = False
    cine_color_cycle_speed_ms: int = 400
    cine_stutter_repeats: int = 0
    cine_reverse_len: float = 0.0
    cine_reverse_window: float = 0.0
    cine_ramp_in: float = 0.0
    cine_ramp_out: float = 0.0
    cine_boomerang: bool = False
    cine_boomerang_bounces: int = 0
    # Dimension portal shapes (rect / trapezoid / diamond / portrait)
    cine_dimension: bool = False
    cine_dimension_kind: str = ""
    # Slice reveal / 1-3 window pan (boomerang across 3 positions)
    cine_pan916: bool = False
    cine_pan916_speed_ms: int = 400
    cine_pan916_parts: int = 3
    cine_pan916_transparent: bool = False
    cine_pan916_random: bool = False
    cine_pan916_phase: int = 0
    cine_pan916_flip_lr: bool = False
    # Mosaic multi-screen effect
    cine_mosaic: bool = False
    cine_mosaic_screens: int = 0
    # Multiply same-clip multi-screen effect
    cine_multiply: bool = False
    cine_multiply_screens: int = 0
    # Upside-down flip and rotating screen
    cine_flip: bool = False
    cine_rotate: bool = False
    cine_rotate_degrees: float = 0.0
    # Camera motion hits (dolly / Ken Burns)
    cine_dolly: bool = False
    cine_dolly_strength: float = 0.0
    cine_kenburns: bool = False
    cine_kenburns_strength: float = 0.0
    # Shared camera motion direction: 0=random, 1=zoom in, 2=zoom out,
    # 3=pan left, 4=pan right, 5=pan up, 6=pan down
    cine_motion_dir: int = 0
    # Break impact FX (first beat after a break)
    is_break_impact: bool = False
    impact_flash_strength: float = 0.0
    impact_flash_speed_ms: int = 250
    impact_shock_strength: float = 0.0
    impact_echo_trail_strength: float = 0.0
    impact_confetti_density: float = 0.0
    impact_zoom_amount: float = 0.0
    impact_shake_strength: float = 0.0
    impact_fog_density: float = 0.0
    impact_fire_gold_intensity: float = 0.0
    impact_fire_multi_intensity: float = 0.0
    impact_color_cycle_speed: float = 0.0
    # Timed strobe: list of offsets (seconds within this segment) where Flash strobe should fire.
    strobe_on_time_offsets: List[float] = field(default_factory=list)
    # Absolute position on the music timeline (seconds)
    timeline_start: float = 0.0
    timeline_end: float = 0.0
    is_image: bool = False  # Mark if this is an image (for special effects)


@dataclass
class MusicAnalysisConfig:
    sensitivity: int = 5


@dataclass
class MusicAnalysisResult:
    beats: List[Beat]
    sections: List[Section]
    duration: float




# --------------------------- queue/headless helpers ------------------------

def _analysis_to_dict(a: "MusicAnalysisResult") -> dict:
    try:
        from dataclasses import asdict
        return asdict(a)
    except Exception:
        # Fallback: best-effort manual
        return {
            "beats": [getattr(b, "__dict__", {}) for b in (getattr(a, "beats", []) or [])],
            "sections": [getattr(s, "__dict__", {}) for s in (getattr(a, "sections", []) or [])],
            "duration": float(getattr(a, "duration", 0.0) or 0.0),
        }

def _analysis_from_dict(d: dict) -> "MusicAnalysisResult":
    beats = [Beat(**b) for b in (d.get("beats") or [])]
    sections = [Section(**s) for s in (d.get("sections") or [])]
    return MusicAnalysisResult(beats=beats, sections=sections, duration=float(d.get("duration") or 0.0))

def _segments_to_list(segments: list) -> list:
    try:
        from dataclasses import asdict
        return [asdict(s) for s in (segments or [])]
    except Exception:
        out = []
        for s in (segments or []):
            try:
                out.append(dict(getattr(s, "__dict__", {})))
            except Exception:
                pass
        return out

def _segments_from_list(items: list) -> list:
    out = []
    for d in (items or []):
        try:
            out.append(TimelineSegment(**d))
        except Exception:
            try:
                # tolerate missing keys
                out.append(TimelineSegment(
                    clip_path=str(d.get("clip_path","")),
                    clip_start=float(d.get("clip_start") or 0.0),
                    duration=float(d.get("duration") or 0.0),
                    effect=str(d.get("effect") or "none"),
                    energy_class=str(d.get("energy_class") or "mid"),
                    transition=str(d.get("transition") or "none"),
                    slow_factor=float(d.get("slow_factor") or 1.0),
                ))
            except Exception:
                pass
    return out

def run_queue_payload(payload_path: str) -> int:
    """Headless entrypoint used by queued jobs.

    This is intentionally dependency-light: it reuses RenderWorker's pipeline
    but swaps Qt signals for simple stdout progress lines so worker.tools_ffmpeg
    can parse percentages.
    """
    try:
        with open(payload_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        try:
            print(f"ERROR: failed to read payload: {e}")
        except Exception:
            pass
        return 2

    try:
        analysis = _analysis_from_dict(payload.get("analysis") or {})
        segments = _segments_from_list(payload.get("segments") or [])
        audio_path = str(payload.get("audio_path") or "")
        output_dir = str(payload.get("output_dir") or "")
        ffmpeg = str(payload.get("ffmpeg") or ffmpeg_path())
        ffprobe = str(payload.get("ffprobe") or ffprobe_path())
        tr = payload.get("target_resolution")
        target_resolution = None
        if isinstance(tr, (list, tuple)) and len(tr) == 2:
            try:
                target_resolution = (int(tr[0]), int(tr[1]))
            except Exception:
                target_resolution = None
        fit_mode = int(payload.get("fit_mode") or 0)
        transition_mode = int(payload.get("transition_mode") or 0)
        intro_fade = bool(payload.get("intro_fade", True))
        outro_fade = bool(payload.get("outro_fade", True))
        keep_source_bitrate = bool(payload.get("keep_source_bitrate", False))
        use_visual_overlay = bool(payload.get("use_visual_overlay", False))
        visual_strategy = int(payload.get("visual_strategy") or 0)
        visual_section_overrides = payload.get("visual_section_overrides") or None
        visual_overlay_opacity = float(payload.get("visual_overlay_opacity") or 0.25)
        out_name_override = payload.get("out_name_override") or None
        strobe_on_time_times = payload.get("strobe_on_time_times") or None
        try:
            strobe_flash_strength = float(payload.get("strobe_flash_strength") or 0.0)
        except Exception:
            strobe_flash_strength = 0.0
        try:
            strobe_flash_speed_ms = int(payload.get("strobe_flash_speed_ms") or 250)
        except Exception:
            strobe_flash_speed_ms = 250
    except Exception as e:
        try:
            print(f"ERROR: invalid payload fields: {e}")
        except Exception:
            pass
        return 3

    # Create worker and monkeypatch progress signal to stdout
    rw = RenderWorker(
        audio_path=audio_path,
        output_dir=output_dir,
        analysis=analysis,
        segments=segments,
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        target_resolution=target_resolution,
        fit_mode=fit_mode,
        transition_mode=transition_mode,
        intro_fade=intro_fade,
        outro_fade=outro_fade,
        keep_source_bitrate=keep_source_bitrate,
        use_visual_overlay=use_visual_overlay,
        visual_strategy=visual_strategy,
        visual_section_overrides=visual_section_overrides,
        visual_overlay_opacity=visual_overlay_opacity,
        out_name_override=out_name_override,
        strobe_on_time_times=strobe_on_time_times,
        strobe_flash_strength=strobe_flash_strength,
        strobe_flash_speed_ms=strobe_flash_speed_ms,
    )

    class _DummySig:
        def emit(self, pct, msg=""):
            try:
                ip = int(pct)
            except Exception:
                ip = pct
            try:
                print(f"{ip}% {msg}".strip())
            except Exception:
                try:
                    print(f"{ip}%")
                except Exception:
                    pass

    try:
        rw.progress = _DummySig()
    except Exception:
        pass

    try:
        rw._run_impl()
        try:
            print("100% Done.")
        except Exception:
            pass
        return 0
    except Exception as e:
        try:
            print(f"ERROR: {e}")
        except Exception:
            pass
        return 1


# --------------------------- music analysis --------------------------------


def analyze_music(audio_path: str, ffmpeg: str, config: Optional[MusicAnalysisConfig] = None) -> MusicAnalysisResult:
    """Very lightweight beat + energy analysis using ffmpeg + PCM RMS."""
    tmpdir = tempfile.mkdtemp(prefix="fv_mclip_an_")
    wav_path = os.path.join(tmpdir, "mono.wav")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        audio_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-f",
        "wav",
        wav_path,
    ]
    code, out = _run_ffmpeg(cmd)
    if code != 0 or not os.path.exists(wav_path):
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("Failed to convert audio for analysis:\n" + out)

    import wave
    import struct

    with wave.open(wav_path, "rb") as wf:
        fr = wf.getframerate()
        total_frames = wf.getnframes()
        duration = total_frames / float(fr)

        win_size = int(fr * 0.05)  # 50ms windows
        values = []
        times = []
        read = 0
        while read < total_frames:
            n = min(win_size, total_frames - read)
            raw = wf.readframes(n)
            read += n
            if not raw:
                break
            count = len(raw) // 2  # 16-bit
            if count == 0:
                rms = 0.0
            else:
                fmt = "<" + "h" * count
                samples = struct.unpack(fmt, raw)
                acc = 0.0
                for s in samples:
                    acc += (s / 32768.0) ** 2
                rms = math.sqrt(acc / count)
            values.append(rms)
            times.append(len(values) * (n / float(fr)))

    shutil.rmtree(tmpdir, ignore_errors=True)

    if not values:
        raise RuntimeError("Audio analysis produced no samples.")

    max_v = max(values) or 1.0
    norm = [v / max_v for v in values]

    mean = sum(norm) / len(norm)
    var = sum((v - mean) ** 2 for v in norm) / len(norm)
    std = math.sqrt(var)

    # Apply beat sensitivity from config (1 = fewer beats, 10 = more beats)
    if config is not None:
        try:
            raw_val = float(config.sensitivity)
        except Exception:
            raw_val = 10.0
    else:
        raw_val = 10.0
    # Slider uses 2..20 -> map to 1.0..10.0 in 0.5 steps
    sens = max(1.0, min(10.0, raw_val / 2.0))
    # Map sensitivity linearly: 1 -> ~1.24x thresholds (fewer beats), 10 -> ~0.70x (more beats)
    scale = 1.0 - (sens - 5.0) * 0.06
    if scale < 0.6:
        scale = 0.6
    elif scale > 1.4:
        scale = 1.4
    beat_thr = mean + std * 0.7 * scale
    major_thr = mean + std * 1.4 * scale

    beats: List[Beat] = []
    last_peak = -9999
    min_dist = int(0.15 / 0.05)  # 150ms

    for i in range(1, len(norm) - 1):
        v = norm[i]
        if v < beat_thr:
            continue
        if v >= norm[i - 1] and v >= norm[i + 1]:
            if i - last_peak < min_dist:
                if beats and v > beats[-1].strength:
                    beats[-1] = Beat(
                        time=times[i],
                        strength=v,
                        kind="major" if v >= major_thr else "minor",
                    )
                    last_peak = i
                continue
            kind = "major" if v >= major_thr else "minor"
            beats.append(Beat(time=times[i], strength=v, kind=kind))
            last_peak = i

    # If we have beats but there's a long quiet tail with no peaks,
    # synthesize gentle extra beats in the tail so quiet / outro parts
    # still have material for the timeline.
    if beats:
        tail = duration - beats[-1].time
        if tail > 5.0:
            # Place minor beats at regular spacing in the tail.
            # Use a stable pattern so results remain predictable.
            num_virtual = max(2, min(16, int(tail / 1.0)))
            if num_virtual > 0:
                spacing = tail / (num_virtual + 1)
                t = beats[-1].time + spacing
                while t < duration - 0.25:
                    beats.append(
                        Beat(
                            time=t,
                            strength=mean,
                            kind="minor",
                        )
                    )
                    t += spacing

    # Ensure beats are sorted in time in case virtual beats were appended
    beats.sort(key=lambda b: b.time)

    # Energy profile in 1s windows
    win1 = int(1.0 / 0.05)
    e_vals = []
    e_times = []
    for i in range(0, len(norm), win1):
        chunk = norm[i : i + win1]
        if not chunk:
            continue
        e_vals.append(sum(chunk) / len(chunk))
        e_times.append(i * 0.05)

    if not e_vals:
        e_vals = [mean]
        e_times = [0.0]

    e_mean = sum(e_vals) / len(e_vals)
    e_var = sum((v - e_mean) ** 2 for v in e_vals) / len(e_vals)
    e_std = math.sqrt(e_var)

    low_thr = e_mean - 0.4 * e_std
    high_thr = e_mean + 0.4 * e_std

    sections: List[Section] = []
    cur_kind = None
    cur_start = 0.0

    def flush(end_t: float):
        nonlocal cur_kind, cur_start
        if cur_kind is None:
            return
        sections.append(Section(start=cur_start, end=end_t, kind=cur_kind))

    for i, v in enumerate(e_vals):
        t = e_times[i]
        if v < low_thr:
            kind = "intro_or_break"
        elif v > high_thr:
            kind = "chorus_or_drop"
        else:
            kind = "verse_or_mid"
        if cur_kind is None:
            cur_kind = kind
            cur_start = t
        elif kind != cur_kind:
            flush(t)
            cur_kind = kind
            cur_start = t
    flush(duration)

    labelled: List[Section] = []
    chorus_seen = False
    for idx, s in enumerate(sections):
        if idx == 0:
            k = "intro"
        elif s.kind == "chorus_or_drop":
            k = "chorus" if not chorus_seen else "drop"
            chorus_seen = True
        elif s.kind == "intro_or_break":
            k = "break"
        else:
            k = "verse"
        labelled.append(Section(start=s.start, end=s.end, kind=k))

    # Ensure we always end on an explicit "outro" section.
    if labelled:
        last = labelled[-1]
        last_duration = max(0.0, last.end - last.start)
        # If the final segment is extremely short, merge it with the
        # previous one so the outro feels like a real section instead
        # of a tiny fragment.
        if last_duration < 1.25 and len(labelled) >= 2:
            prev = labelled[-2]
            merged = Section(
                start=prev.start,
                end=last.end,
                kind="outro",
            )
            labelled[-2:] = [merged]
        else:
            labelled[-1] = Section(
                start=last.start,
                end=last.end,
                kind="outro",
            )

    return MusicAnalysisResult(beats=beats, sections=labelled, duration=duration)




def discover_video_sources(video_input: str, ffprobe: str) -> List[ClipSource]:
    """Return a list of ClipSource from a file, directory, or a '|' separated list of files."""
    exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}
    sources: List[ClipSource] = []

    # Multiple explicit files
    if "|" in video_input:
        parts = [p.strip() for p in video_input.split("|") if p.strip()]
        for raw in parts:
            path = os.path.abspath(raw)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in exts:
                continue
            dur = _ffprobe_duration(ffprobe, path)
            if dur and dur > 0:
                sources.append(ClipSource(path=path, duration=dur))
        return sources

    # Single file or directory
    video_input = os.path.abspath(video_input)

    if os.path.isdir(video_input):
        for name in sorted(os.listdir(video_input)):
            path = os.path.join(video_input, name)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(path)
            if ext.lower() not in exts:
                continue
            dur = _ffprobe_duration(ffprobe, path)
            if dur and dur > 0:
                sources.append(ClipSource(path=path, duration=dur))
    elif os.path.isfile(video_input):
        dur = _ffprobe_duration(ffprobe, video_input)
        if dur and dur > 0:
            sources.append(ClipSource(path=video_input, duration=dur))

    return sources

# --------------------------- timeline builder ------------------------------

def build_timeline(
    analysis: MusicAnalysisResult,
    sources: List[ClipSource],
    fx_level: str,
    microclip_mode: int,
    beats_per_segment: int,
    transition_mode: int,
    clip_order_mode: int,
    force_full_length: bool,
    seed_enabled: bool,
    seed_value: int,
    transition_random: bool,
    transition_modes_enabled: Optional[List[int]],
    intro_transitions_only: bool = False,
    slow_motion_enabled: bool = False,
    slow_motion_factor: float = 1.0,
    slow_motion_sections: Optional[List[str]] = None,
    slow_motion_random: bool = False,
    cine_enable: bool = False,
    cine_freeze: bool = False,
    cine_stutter: bool = False,
    cine_tear_v: bool = False,
    cine_tear_v_strength: float = 0.7,
    cine_tear_h: bool = False,
    cine_tear_h_strength: float = 0.7,
    cine_color_cycle: bool = False,
    cine_color_cycle_speed_ms: int = 400,
    cine_reverse: bool = False,
    cine_speedup_forward: bool = False,
    cine_speedup_forward_factor: float = 1.5,
    cine_speedup_backward: bool = False,
    cine_speedup_backward_factor: float = 1.5,
    cine_speed_ramp: bool = False,
    cine_freeze_len: float = 0.5,
    cine_freeze_zoom: float = 0.15,
    cine_stutter_repeats: int = 3,
    cine_reverse_len: float = 0.5,
    cine_ramp_in: float = 0.25,
    cine_ramp_out: float = 0.25,
    cine_boomerang: bool = False,
    cine_boomerang_bounces: int = 2,
    cine_dimension: bool = False,
    cine_pan916: bool = False,
    cine_pan916_speed_ms: int = 400,
    cine_pan916_parts: int = 3,
    cine_pan916_transparent: bool = False,
    cine_pan916_random: bool = False,
    cine_mosaic: bool = False,
    cine_mosaic_screens: int = 4,
    cine_mosaic_random: bool = False,
    cine_flip: bool = False,
    cine_rotate: bool = False,
    cine_rotate_max_degrees: float = 20.0,
    cine_multiply: bool = False,
    cine_multiply_screens: int = 4,
    cine_multiply_random: bool = False,
    cine_dolly: bool = False,
    cine_dolly_strength: float = 0.4,
    cine_kenburns: bool = False,
    cine_kenburns_strength: float = 0.35,
    cine_motion_dir: int = 0,
    audio_duration: float | None = None,
    impact_enable: bool = False,
    impact_flash: bool = False,
    impact_shock: bool = False,
    impact_echo_trail: bool = False,
    impact_confetti: bool = False,
    impact_zoom: bool = False,
    impact_shake: bool = False,
    impact_fog: bool = False,
    impact_fire_gold: bool = False,
    impact_fire_multi: bool = False,
    impact_color_cycle: bool = False,
    impact_random: bool = False,
    impact_flash_strength: float = 0.8,
    impact_flash_speed_ms: int = 250,
    impact_shock_strength: float = 0.75,
    impact_echo_trail_strength: float = 0.7,
    impact_confetti_density: float = 0.7,
    impact_zoom_amount: float = 0.2,
    impact_shake_strength: float = 0.6,
    impact_fog_density: float = 0.65,
    impact_fire_gold_intensity: float = 0.75,
    impact_fire_multi_intensity: float = 0.8,
    impact_color_cycle_speed: float = 0.7,
    strobe_on_time: bool = False,
    strobe_on_time_times: Optional[List[float]] = None,
    image_sources: Optional[List[ClipSource]] = None,
    section_overrides: Optional[Dict[int, ClipSource]] = None,
    image_segment_interval: int = 0,
) -> List[TimelineSegment]:


    """Build a musical timeline from beats, sections and sources.

    fx_level:
        "minimal", "moderate", "high"

    microclip_mode:
        0 = off
        1 = only in high-energy sections (chorus / drops)
        2 = whole track
        3 = verses only (microclips inside verse sections; longer cuts elsewhere)

    transition_mode:
        0 = soft fades
        1 = hard cuts
        2 = mixed (fades + flash cuts sometimes)

    clip_order_mode:
        0 = random (default)
        1 = sequential
        2 = shuffle (no repeats until all clips used)
    """
    if not sources:
        return []

    if image_sources is None:
        image_sources = []

    if seed_enabled:
        random.seed(int(seed_value) & 0xFFFFFFFF)

    # Helper: map a logical time (seconds) to a section label, including 'outro' for the final section.
    def _section_label_at(time_t: float) -> str:
        sections = analysis.sections
        if not sections:
            return "verse"
        last_idx = len(sections) - 1
        for idx, s in enumerate(sections):
            # Include the tail of the last section
            if s.start <= time_t < s.end or (idx == last_idx and time_t >= s.start):
                if idx == last_idx and s.kind not in ("intro",):
                    return "outro"
                return s.kind
        # Fallback: treat as part of the last section
        return sections[-1].kind

    def _section_index_at(time_t: float) -> Optional[int]:
        """Return the index of the section that contains the given time."""
        sections = analysis.sections
        if not sections:
            return None
        last_idx = len(sections) - 1
        for idx, s in enumerate(sections):
            if s.start <= time_t < s.end or (idx == last_idx and time_t >= s.start):
                return idx
        return last_idx

    def _apply_slow_motion(segments: List[TimelineSegment],
                           centers: List[float],
                           section_labels: List[str]) -> None:
        if not slow_motion_enabled or not segments:
            return

        # Normalize configuration
        selected_sections = set()
        if slow_motion_sections:
            selected_sections = {str(k).lower() for k in slow_motion_sections}

        # 1) Direct section-based slow motion
        slow_flags = [False] * len(segments)
        for idx, label in enumerate(section_labels):
            lbl = (label or "").lower()
            # Ignore intro when transitions-only is enabled
            if intro_transitions_only and lbl == "intro":
                continue
            if lbl in selected_sections:
                slow_flags[idx] = True

        # 2) Random slow motion: at most one event per 60 seconds of timeline
        if slow_motion_random:
            cumulative = 0.0
            centers_timeline: List[float] = []
            for seg in segments:
                center = cumulative + max(seg.duration, 0.0) * 0.5
                centers_timeline.append(center)
                cumulative += max(seg.duration, 0.0)

            blocks = {}
            for idx, t in enumerate(centers_timeline):
                if slow_flags[idx]:
                    continue  # already slowed via section selection
                # Skip intro when transitions-only is enabled
                try:
                    lbl = (section_labels[idx] or "").lower() if idx < len(section_labels) else ""
                except Exception:
                    lbl = ""
                if intro_transitions_only and lbl == "intro":
                    continue
                block = int(t // 60.0)
                blocks.setdefault(block, []).append(idx)

            for idx_list in blocks.values():
                if not idx_list:
                    continue
                choice = random.choice(idx_list)
                slow_flags[choice] = True

        # Apply final slow factors
        for idx, seg in enumerate(segments):
            if slow_flags[idx]:
                seg.slow_factor = float(slow_motion_factor or 1.0)
            else:
                seg.slow_factor = 1.0


    def _apply_cinematic_effects(segments: List[TimelineSegment], allow_mosaic: bool, section_labels: Optional[List[str]] = None) -> None:
        """Mark segments for rare cinematic effects (freeze, stutter, reverse, speedups, ramps).

        The logic is intentionally conservative: for the freeze / stutter / reverse
        effects we pick at most one segment per 5 seconds of video timeline, and
        only when the global cinematic toggle plus the corresponding effect toggle
        are enabled. Speed ramps are applied to all slow-motion segments when the
        option is on, but they currently piggy-back on the existing slow-factor
        logic rather than splitting segments.
        """
        if not cine_enable or not segments:
            return

        # Collect which cinematic effects are globally enabled.
        enabled_effects: List[str] = []
        # Dimension (portal shapes) uses a shuffle-bag so all shapes play once
        # before repeating.
        dimension_modes: List[str] = ["rect", "trapezoid", "diamond", "portrait"]
        dimension_bag: List[str] = []

        def _pick_dimension_kind() -> str:
            nonlocal dimension_bag
            if not dimension_bag:
                dimension_bag = list(dimension_modes)
                random.shuffle(dimension_bag)
            try:
                return str(dimension_bag.pop())
            except Exception:
                return "rect"

        boomerang_indices: List[int] = []
        used_segments = set()
        if cine_freeze:
            enabled_effects.append("freeze")
        if cine_tear_v:
            enabled_effects.append("tear_v")
        if cine_tear_h:
            enabled_effects.append("tear_h")

        if cine_color_cycle:
            enabled_effects.append("color_cycle")

        if cine_stutter:
            enabled_effects.append("stutter")
        if cine_reverse:
            enabled_effects.append("reverse")
        if cine_speedup_forward:
            enabled_effects.append("speedup_forward")
        if cine_speedup_backward:
            enabled_effects.append("speedup_backward")
        if cine_boomerang:
            enabled_effects.append("boomerang")
        if cine_dimension:
            enabled_effects.append("dimension")
        if cine_pan916:
            enabled_effects.append("pan916")
        # Mosaic uses the same 'one event per 5s' rule as the other cinematic effects.
        # Safety: only enable Mosaic when there are enough video clips loaded.
        if cine_mosaic and allow_mosaic:
            enabled_effects.append("mosaic")
        if cine_flip:
            enabled_effects.append("flip")
        if cine_rotate:
            enabled_effects.append("rotate")
        # Multiply (same-clip multi-screen) follows the same cadence as Mosaic.
        if cine_multiply:
            enabled_effects.append("multiply")

        # If nothing except ramps is enabled, we only tag slow-motion segments below.
        # (Still honour cine_enable so the user can quickly disable everything.)
        # Compute segment centres on the *video* timeline so we can respect the
        # "about once every 5 seconds" requirement.
        centers: List[float] = []
        cumulative = 0.0
        for seg in segments:
            duration = max(0.0, float(getattr(seg, "duration", 0.0)))
            centers.append(cumulative + duration * 0.5)
            cumulative += duration


        # Eligible indices (skip intro when transitions-only is enabled)
        eligible_idx: List[int] = list(range(len(segments)))
        if intro_transitions_only and segment_labels:
            eligible_idx = [i for i in eligible_idx if (section_labels[i] or "").lower() != "intro"]

        if enabled_effects and cumulative > 0.0:
            buckets: dict[int, List[int]] = {}
            for idx in eligible_idx:
                t = centers[idx]
                block = int(t // 5.0) # change speed of change
                buckets.setdefault(block, []).append(idx)

            for block_idx in sorted(buckets.keys()):
                idx_list = buckets[block_idx]

                if not idx_list:
                    continue
                # One cinematic event per 5-second block at most.
                seg_idx = random.choice(idx_list)
                seg = segments[seg_idx]
                effect_name = random.choice(enabled_effects)
                if effect_name == "freeze":
                    seg.cine_freeze = True
                    seg.cine_freeze_len = float(max(0.10, min(1.0, cine_freeze_len)))
                    # Slider stays the same (0.0–0.5), but now drives the Shutter-pop intensity.
                    seg.cine_freeze_zoom = float(max(0.0, min(0.5, cine_freeze_zoom)))
                    # Random left/right chroma direction so repeats feel less samey.
                    seg.cine_freeze_dir = random.choice([-1, 1])
                elif effect_name == "stutter":
                    seg.cine_stutter = True
                    seg.cine_stutter_repeats = int(max(2, min(5, cine_stutter_repeats)))
                elif effect_name == "tear_v":
                    seg.cine_tear_v = True

                    # Base strength from slider (0.1–1.0)
                    base = float(max(0.1, min(1.0, cine_tear_v_strength)))

                    # Per-event strength jitter (keeps it from feeling identical)
                    jitter = random.uniform(0.85, 1.15)
                    seg.cine_tear_v_strength = float(max(0.1, min(1.0, base * jitter)))

                    # Per-event direction (left slice goes left OR right, right slice mirrors)
                    # We store it on the segment; no dataclass change required.
                    seg.cine_tear_v_dir = random.choice([-1, 1])
                elif effect_name == "tear_h":
                    seg.cine_tear_h = True

                    # Base strength from slider (0.1–1.0)
                    base = float(max(0.1, min(1.0, cine_tear_h_strength)))

                    # Per-event strength jitter (keeps it from feeling identical)
                    jitter = random.uniform(0.85, 1.15)
                    seg.cine_tear_h_strength = float(max(0.1, min(1.0, base * jitter)))

                    # Per-event direction (top slice goes up OR down, bottom slice mirrors)
                    # We store it on the segment; no dataclass change required.
                    seg.cine_tear_h_dir = random.choice([-1, 1])
                elif effect_name == "color_cycle":
                    seg.cine_color_cycle = True
                    try:
                        ms = int(cine_color_cycle_speed_ms)
                    except Exception:
                        ms = 400
                    ms = max(100, min(1000, ms))
                    # Snap to 50ms steps like the other speed sliders.
                    ms = int(round(ms / 50.0) * 50)
                    seg.cine_color_cycle_speed_ms = ms
                elif effect_name == "reverse":
                    seg.cine_reverse = True
                    seg.cine_reverse_len = float(max(0.10, min(1.5, cine_reverse_len)))
                    # Preserve the musical segment duration.
                    # Store a dedicated short window for the reverse-bounce renderer.
                    try:
                        win = float(seg.cine_reverse_len or 0.0)
                    except Exception:
                        win = 0.0
                    if win <= 0.0:
                        win = 0.5
                    try:
                        base_len = float(getattr(seg, "duration", 0.0) or 0.0)
                    except Exception:
                        base_len = 0.0
                    if base_len > 0.0:
                        win = min(win, base_len)
                    seg.cine_reverse_window = win
                elif effect_name == "speedup_forward":
                    seg.cine_speedup_forward = True
                    try:
                        factor = float(cine_speedup_forward_factor)
                    except Exception:
                        factor = 1.5
                    factor = max(1.25, min(4.0, factor))
                    seg.cine_speedup_forward_factor = factor
                    # Reuse the shared speed pipeline.
                    seg.slow_factor = factor
                elif effect_name == "speedup_backward":
                    seg.cine_speedup_backward = True
                    try:
                        factor = float(cine_speedup_backward_factor)
                    except Exception:
                        factor = 1.5
                    factor = max(1.25, min(4.0, factor))
                    seg.cine_speedup_backward_factor = factor
                    seg.cine_reverse = True
                    seg.slow_factor = factor
                elif effect_name == "boomerang":
                    # Mark this segment as a boomerang loop                    boomerang_indices.append(seg_idx)
                    seg.cine_boomerang = True
                    try:
                        bounces = int(cine_boomerang_bounces)
                    except Exception:
                        bounces = 2
                    bounces = max(1, min(9, bounces))
                    seg.cine_boomerang_bounces = bounces
                    # Ensure the clip region used for this segment lives fully
                    # inside a short punchy window of the source clip, but do NOT
                    # shrink the musical segment duration. We store a dedicated
                    # boomerang window length for the renderer.
                    try:
                        max_window = 0.25  # use only the first ~0.25 second of the source clip for boomerang
                        target_len = float(seg_trim_dur or 0.0)
                        if target_len <= 0.0:
                            target_len = 0.1
                        cur_dur = min(target_len, max_window)
                        if cur_dur <= 0.0:
                            cur_dur = 0.1
                        setattr(seg, "cine_boomerang_window", cur_dur)
                        cur_start = float(getattr(seg, "clip_start", 0.0) or 0.0)
                        if cur_start < 0.0:
                            cur_start = 0.0
                        if cur_start + cur_dur > max_window:
                            cur_start = max(0.0, max_window - cur_dur)
                        seg.clip_start = cur_start
                    except Exception:
                        # As a safety fallback, just pin to the absolute start.
                        try:
                            setattr(seg, "cine_boomerang_window", 0.25)
                            seg.clip_start = 0.0
                        except Exception:
                            pass
                elif effect_name == "dimension":
                    # Dimension portal: show the segment inside a randomly chosen portal shape.
                    # Uses a shuffle-bag so each shape is used once before repeating.
                    seg.cine_dimension = True
                    seg.cine_dimension_kind = _pick_dimension_kind()

                elif effect_name == "pan916":
                    # Slice reveal (boomerang across 3 positions).
                    seg.cine_pan916 = True
                    try:
                        ms = int(cine_pan916_speed_ms)
                    except Exception:
                        ms = 400
                    ms = max(200, min(1000, ms))
                    # Snap to the 50ms steps used by the UI slider.
                    ms = int(round(ms / 50.0) * 50)
                    seg.cine_pan916_speed_ms = ms
                    # Extra options: number of slices and background transparency.
                    seg.cine_pan916_random = bool(cine_pan916_random)
                    if seg.cine_pan916_random:
                        # Random slice count (deterministic when the user enables seeding).
                        parts = int(random.randint(2, 6))
                    else:
                        try:
                            parts = int(cine_pan916_parts)
                        except Exception:
                            parts = 3
                        parts = max(2, min(6, parts))
                    seg.cine_pan916_parts = parts
                    seg.cine_pan916_transparent = bool(cine_pan916_transparent)
                    if seg.cine_pan916_random:
                        # Deterministic if the user enabled seeding above.
                        period = max(2, (2 * parts - 2))
                        seg.cine_pan916_phase = int(random.randint(0, max(0, period - 1)))
                        seg.cine_pan916_flip_lr = bool(random.getrandbits(1))
                    else:
                        seg.cine_pan916_phase = 0
                        seg.cine_pan916_flip_lr = False

                elif effect_name == "mosaic":
                    seg.cine_mosaic = True
                    if cine_mosaic_random:
                        # Pick any layout between 2 and 9 screens each time Mosaic is assigned.
                        screens = random.randint(2, 9)
                    else:
                        try:
                            screens = int(cine_mosaic_screens)
                        except Exception:
                            screens = 4
                        # Clamp to the supported 2–9 range
                        screens = max(2, min(9, screens))
                    seg.cine_mosaic_screens = screens

                elif effect_name == "flip":
                    # Mark this segment for a 180° upside-down hit, and also
                    # apply a rotating-screen spin so it feels intentional.
                    seg.cine_flip = True
                    seg.cine_rotate = True
                    try:
                        max_deg = float(cine_rotate_max_degrees)
                    except Exception:
                        max_deg = 20.0
                    # Clamp to a safe range to avoid nausea.
                    max_deg = max(5.0, min(90.0, max_deg))
                    seg.cine_rotate_degrees = max_deg

                elif effect_name == "rotate":
                    # Apply a short rotating-screen hit with a random angle
                    # up to the user-specified maximum.
                    seg.cine_rotate = True
                    try:
                        max_deg = float(cine_rotate_max_degrees)
                    except Exception:
                        max_deg = 20.0
                    # Clamp to a safe range to avoid nausea.
                    max_deg = max(5.0, min(90.0, max_deg))
                    seg.cine_rotate_degrees = max_deg

                elif effect_name == "multiply":
                    # Multiply uses the same layouts as Mosaic, but
                    # shows multiple copies of *the same* clip instead of
                    # filling the grid with different sources.
                    seg.cine_multiply = True
                    if cine_multiply_random:
                        # Pick any layout between 2 and 9 screens each time Multiply is assigned.
                        screens = random.randint(2, 9)
                    else:
                        try:
                            screens = int(cine_multiply_screens)
                        except Exception:
                            screens = 4
                        # Clamp to the supported 2–9 range
                        screens = max(2, min(9, screens))
                    seg.cine_multiply_screens = screens

        # Camera motion hits (dolly-zoom / Ken Burns) – rarer, about one per 15s block.
        camera_enabled = (cine_dolly or cine_kenburns)
        if camera_enabled and cumulative > 0.0:
            cam_buckets: dict[int, List[int]] = {}
            for idx in eligible_idx:
                t = centers[idx]
                block = int(t // 15.0)
                cam_buckets.setdefault(block, []).append(idx)

            def _pick_motion_dir(global_dir: int, allowed_dirs) -> int:
                try:
                    g = int(global_dir)
                except Exception:
                    g = 0
                allowed_list = list(allowed_dirs)
                if not allowed_list:
                    return 0
                # If the user picked a specific direction and it's allowed, honour it.
                if g in allowed_list:
                    return g
                # Otherwise pick a random direction from the allowed set.
                return random.choice(allowed_list)

            available_cams: List[str] = []
            if cine_dolly:
                available_cams.append("dolly")
            if cine_kenburns:
                available_cams.append("kenburns")

            for block_idx, idx_list in cam_buckets.items():
                if not idx_list or not available_cams:
                    continue
                seg_idx = random.choice(idx_list)
                seg = segments[seg_idx]
                which = random.choice(available_cams)
                if which == "dolly":
                    seg.cine_dolly = True
                    seg.cine_dolly_strength = float(max(0.10, min(1.0, cine_dolly_strength)))
                    seg.cine_motion_dir = int(_pick_motion_dir(cine_motion_dir, (1, 2)))
                elif which == "kenburns":
                    seg.cine_kenburns = True
                    seg.cine_kenburns_strength = float(max(0.10, min(1.0, cine_kenburns_strength)))
                    seg.cine_motion_dir = int(_pick_motion_dir(cine_motion_dir, (3, 4, 5, 6)))

        # Speed ramps: tag all slow-motion segments (slow_factor != 1.0) when enabled.
        if cine_speed_ramp and slow_motion_enabled:
            for seg in segments:
                if float(getattr(seg, "slow_factor", 1.0)) != 1.0:
                    seg.cine_speed_ramp = True
                    seg.cine_ramp_in = float(max(0.05, min(0.60, cine_ramp_in)))
                    seg.cine_ramp_out = float(max(0.05, min(0.60, cine_ramp_out)))

    


    def _apply_break_impact_fx(segments: List[TimelineSegment],
                               centers: List[float],
                               segment_labels: Optional[List[str]] = None) -> None:
        """Mark segments that should receive 'break impact' FX (first beat after a break).

        We scan music sections for 'break' regions, then for each break we find the
        first strong beat that follows the end of the break (within a short window).
        The segment whose logical centre is closest to that beat (or whose
        approximate coverage includes it) is tagged as an impact segment and
        receives break impact parameters according to the UI configuration.

        If no explicit 'break' sections are found, we fall back to section boundaries
        into high–energy parts (chorus / drop). As a last resort, we pick the first
        beat after the largest silence / gap between beats so that the feature still
        does something musical on simpler tracks.
        """
        if not impact_enable or not segments:
            return

        beats = analysis.beats
        if not beats:
            return

        impact_times: List[float] = []

        # --- Primary: merged 'break' clusters --------------------------------
        raw_breaks: List[Section] = [sec for sec in analysis.sections if sec.kind == "break"]
        if raw_breaks:
            raw_breaks.sort(key=lambda s: s.start)
            merged: List[Section] = []
            cur = raw_breaks[0]
            gap_limit = 4.0  # seconds; small gaps are treated as one big build‑up
            for sec in raw_breaks[1:]:
                if sec.start - cur.end <= gap_limit:
                    # Extend current build‑up cluster.
                    cur = Section(start=cur.start, end=max(cur.end, sec.end), kind="break")
                else:
                    merged.append(cur)
                    cur = sec
            merged.append(cur)

            for mb in merged:
                end_t = float(mb.end)
                # Consider beats up to a few seconds after the *merged* break.
                window_end = min(analysis.duration, end_t + 4.0)
                cand = [b for b in beats if end_t <= b.time <= window_end]
                if not cand:
                    continue
                majors = [b for b in cand if b.kind == "major"]
                if majors:
                    chosen = min(majors, key=lambda b: b.time)
                else:
                    chosen = min(cand, key=lambda b: b.time)
                impact_times.append(chosen.time)

        # --- Fallback 1: boundaries into chorus / drop -----------------------
        if not impact_times and analysis.sections:
            for prev, cur in zip(analysis.sections, analysis.sections[1:]):
                if cur.kind in ("chorus", "drop") and cur.start > prev.end:
                    start_t = float(prev.end)
                    window_end = min(analysis.duration, float(cur.start) + 4.0)
                    cand = [b for b in beats if start_t <= b.time <= window_end]
                    if not cand:
                        continue
                    majors = [b for b in cand if b.kind == "major"]
                    if majors:
                        chosen = min(majors, key=lambda b: b.time)
                    else:
                        chosen = min(cand, key=lambda b: b.time)
                    impact_times.append(chosen.time)

        # --- Fallback 2: largest beat gap (main drop guess) ------------------
        if not impact_times and len(beats) >= 2:
            max_gap = 0.0
            drop_time = None
            for i in range(len(beats) - 1):
                gap = float(beats[i + 1].time - beats[i].time)
                if gap > max_gap:
                    max_gap = gap
                    drop_time = float(beats[i + 1].time)
            # Require a reasonably long gap so we don't fire on micro‑pauses.
            if drop_time is not None and max_gap >= 1.0:
                impact_times.append(drop_time)

        if not impact_times:
            return

        # Prepare list of enabled effects for selection.
        enabled_effect_names: List[str] = []
        if impact_flash:
            enabled_effect_names.append("flash")
        if impact_shock:
            enabled_effect_names.append("shock")
        if impact_echo_trail:
            enabled_effect_names.append("echo_trail")
        if impact_confetti:
            enabled_effect_names.append("confetti")
        if impact_zoom:
            enabled_effect_names.append("zoom")
        if impact_shake:
            enabled_effect_names.append("shake")
        if impact_fog:
            enabled_effect_names.append("fog")
        if impact_fire_gold:
            enabled_effect_names.append("fire_gold")
        if impact_fire_multi:
            enabled_effect_names.append("fire_multi")
        if impact_color_cycle:
            enabled_effect_names.append("color_cycle")

        if not enabled_effect_names:
            # Nothing to do even if master toggle is on.
            return

        # Helper to assign parameters for a chosen set of effect names on a segment.
        def _tag_segment(seg: TimelineSegment, chosen_effects: List[str]) -> None:
            seg.is_break_impact = True

            seg.impact_flash_strength = impact_flash_strength if "flash" in chosen_effects else 0.0
            seg.impact_flash_speed_ms = impact_flash_speed_ms if "flash" in chosen_effects else 0
            seg.impact_shock_strength = impact_shock_strength if "shock" in chosen_effects else 0.0
            seg.impact_echo_trail_strength = impact_echo_trail_strength if "echo_trail" in chosen_effects else 0.0
            seg.impact_confetti_density = impact_confetti_density if "confetti" in chosen_effects else 0.0
            seg.impact_zoom_amount = impact_zoom_amount if "zoom" in chosen_effects else 0.0
            seg.impact_shake_strength = impact_shake_strength if "shake" in chosen_effects else 0.0
            seg.impact_fog_density = impact_fog_density if "fog" in chosen_effects else 0.0
            seg.impact_fire_gold_intensity = impact_fire_gold_intensity if "fire_gold" in chosen_effects else 0.0
            seg.impact_fire_multi_intensity = impact_fire_multi_intensity if "fire_multi" in chosen_effects else 0.0
            seg.impact_color_cycle_speed = impact_color_cycle_speed if "color_cycle" in chosen_effects else 0.0

        # Map each impact time to the segment whose centre best matches it.
        impact_candidates: List[int] = []
        for t_imp in impact_times:
            best_idx: Optional[int] = None
            best_dist: Optional[float] = None

            for idx, seg in enumerate(segments):
                seg_len = max(0.1, float(getattr(seg, "duration", 0.0)))
                center_t = centers[idx] if idx < len(centers) else 0.0
                start_t = center_t - 0.5 * seg_len
                end_t = center_t + 0.5 * seg_len

                if start_t <= t_imp <= end_t:
                    best_idx = idx
                    break

                # Fallback: pick segment whose centre is closest to the impact time.
                dist = abs(center_t - t_imp)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx

            if best_idx is not None:
                # Skip intro when transitions-only is enabled
                if intro_transitions_only and segment_labels:
                    try:
                        lbl = (section_labels[best_idx] or "").lower()
                    except Exception:
                        lbl = ""
                    if lbl == "intro":
                        continue
                impact_candidates.append(best_idx)

        if not impact_candidates:
            return

        # Group candidates into ~5 second buckets along the video timeline,
        # mirroring the cadence of cinematic effects.
        bucket_map: Dict[int, List[int]] = {}
        for idx in impact_candidates:
            center_t = centers[idx] if idx < len(centers) else 0.0
            bucket = int(center_t // 2.7) # second place to change speed of change
            bucket_map.setdefault(bucket, []).append(idx)

        chosen_indices: List[int] = []
        for bucket_idx in sorted(bucket_map.keys()):
            idx_list = bucket_map[bucket_idx]
            if not idx_list:
                continue
            # Avoid duplicates inside a bucket, then pick a single segment.
            unique = sorted(set(idx_list))
            chosen_indices.append(random.choice(unique))

        if not chosen_indices:
            return

        # Assign break-impact FX. We always apply exactly one effect per chosen
        # segment so they are spaced out like cinematic FX instead of all
        # firing at once. With "Random" enabled, the chosen effect is random
        # per impact; otherwise we cycle deterministically through the enabled
        # list.
        num_effects = len(enabled_effect_names)
        eff_cursor = 0

        for seg_idx in sorted(set(chosen_indices)):
            seg = segments[seg_idx]

            # Per‑segment enabled list: colour strobe ("confetti") is only used
            # on segments that are at most 2 seconds long so the strobe stays snappy.
            per_seg_enabled = list(enabled_effect_names)
            try:
                seg_len = float(getattr(seg, "duration", 0.0) or 0.0)
            except Exception:
                seg_len = 0.0
            if seg_len > 2.0 and "confetti" in per_seg_enabled:
                per_seg_enabled = [e for e in per_seg_enabled if e != "confetti"]

            if not per_seg_enabled:
                continue

            if impact_random:
                eff_name = random.choice(per_seg_enabled)
            else:
                # Cycle deterministically through the global list, but fall back
                # to the first allowed effect if the chosen one was filtered out.
                base_name = enabled_effect_names[eff_cursor % num_effects]
                eff_cursor += 1
                eff_name = base_name if base_name in per_seg_enabled else per_seg_enabled[0]

            _tag_segment(seg, [eff_name])



    def _rebuild_timeline_positions(segments: List[TimelineSegment]) -> None:
        """Rebuild absolute timeline_start / timeline_end for all segments.

        We treat each segment's ``duration`` as the *base* amount of source
        material (real-time seconds). When a segment is slowed down
        (slow_factor < 1.0), it needs *more* room on the musical timeline to
        play the same material, so we stretch its logical length by dividing
        by the slow factor. For all other cases we keep the base duration.

        This mirrors the intent of:

            base_duration = beats_per_segment * avg_beat_interval
            if seg.slow_factor < 1.0 and seg.slow_factor > 0.01:
                seg_duration = base_duration / seg.slow_factor
            else:
                seg_duration = base_duration

        and ensures that cumulative timeline time always uses the stretched
        duration, not the base duration.
        """
        current_time = 0.0
        for seg in segments:
            base_duration = float(getattr(seg, "duration", 0.0))
            try:
                sf = float(getattr(seg, "slow_factor", 1.0) or 1.0)
            except Exception:
                sf = 1.0
            if 0.01 < sf < 1.0 and base_duration > 0.0:
                seg_duration = base_duration / sf
            else:
                seg_duration = base_duration
            seg.timeline_start = current_time
            current_time += seg_duration
            seg.timeline_end = current_time


# Determine which transition modes are allowed for randomization
    allowed_modes: List[int] = []
    if transition_modes_enabled:
        for m in transition_modes_enabled:
            if 0 <= m <= 14 and m not in allowed_modes:
                allowed_modes.append(m)
    if not allowed_modes:
        allowed_modes.append(transition_mode)

    beats = analysis.beats
    if len(beats) < 4:
        # Single-segment fallback for very short tracks.
        seg = TimelineSegment(
            clip_path=sources[0].path,
            clip_start=0.0,
            duration=analysis.duration,
            effect="none",
            energy_class="mid",
            transition="none" if transition_mode == 1 else "fade",
        )
        segments_tmp = [seg]
        center_time = max(0.0, analysis.duration * 0.5)
        label = _section_label_at(center_time)
        _apply_slow_motion(segments_tmp, [center_time], [label])
        return segments_tmp

    def energy_class(time_t: float) -> str:
        for s in analysis.sections:
            if s.start <= time_t < s.end:
                if s.kind in ("chorus", "drop"):
                    return "high"
                if s.kind in ("intro", "break"):
                    return "low"
                return "mid"
        return "mid"

    # Beat intervals
    # We build intervals between consecutive beats (plus a tail segment up to the
    # end of the track), then group those intervals into base segments according
    # to beats_per_segment. For each grouped segment we also track how many beats
    # it spans so we can detect "calm" regions (large gaps between beats).
    intervals = []
    for i in range(len(beats) - 1):
        t0 = beats[i].time
        t1 = beats[i + 1].time
        if t1 > t0:
            intervals.append((t0, t1))
    if beats[-1].time < analysis.duration:
        intervals.append((beats[-1].time, analysis.duration))

    grouped = []  # list of (start_time, end_time, beat_count)
    cur_start = None
    cur_count = 0
    for (t0, t1) in intervals:
        if cur_start is None:
            cur_start = t0
            cur_count = 1
        else:
            cur_count += 1
        if cur_count >= max(1, beats_per_segment):
            grouped.append((cur_start, t1, cur_count))
            cur_start = None
            cur_count = 0
    if cur_start is not None:
        # Tail segment that didn't reach beats_per_segment; still keep its count.
        grouped.append((cur_start, analysis.duration, max(1, cur_count)))

    num_sources = len(sources)
    allow_mosaic = num_sources >= 10
    last_source_idx: Optional[int] = None
    shuffle_pool = list(range(num_sources))
    random.shuffle(shuffle_pool)
    shuffle_pos = 0
    clip_offsets = {i: 0.0 for i in range(num_sources)}
    section_overrides = section_overrides or {}
    section_override_offsets: Dict[int, float] = {}

    def _pick_source_index() -> int:
        nonlocal last_source_idx, shuffle_pool, shuffle_pos
        if num_sources == 1:
            idx = 0
        elif clip_order_mode == 1:  # sequential
            if last_source_idx is None:
                idx = 0
            else:
                idx = (last_source_idx + 1) % num_sources
        elif clip_order_mode == 2:  # shuffle, no repeats until all used
            if shuffle_pos >= len(shuffle_pool):
                shuffle_pool = list(range(num_sources))
                random.shuffle(shuffle_pool)
                shuffle_pos = 0
            idx = shuffle_pool[shuffle_pos]
            shuffle_pos += 1
        else:  # 0 = random (no immediate repeat if possible)
            candidates = list(range(num_sources))
            if last_source_idx is not None and num_sources > 1:
                try:
                    candidates.remove(last_source_idx)
                except ValueError:
                    pass
            idx = random.choice(candidates)
        last_source_idx = idx
        return idx

    def pick_region(length: float) -> Tuple[str, float]:
        idx = _pick_source_index()
        src = sources[idx]
        offset = clip_offsets.get(idx, 0.0)
        if offset + length > src.duration:
            offset = 0.0
        start = offset
        clip_offsets[idx] = offset + length
        return src.path, max(0.0, start)

    segments: List[TimelineSegment] = []

    # User-controlled pacing for still images: how many segments between image inserts.
    try:
        image_interval = int(image_segment_interval or 0)
    except Exception:
        image_interval = 0
    if image_interval < 0:
        image_interval = 0
    image_index = 0
    segments_since_image = 0


    # Smart energy-aware pacing:
    # In calm regions (few / widely spaced beats or low-energy sections),
    # we prefer longer shots so the video can breathe instead of cutting
    # every beat. In active regions we keep the existing microclip logic.
    CALM_GAP_THRESHOLD = 1.4   # seconds between beats to consider section "calm"
    CALM_MIN_LEN = 3.0         # preferred minimum length for calm shots
    CALM_MAX_LEN = 7.0         # preferred maximum length for calm shots

    previous_was_calm = False

    # Track segment centers and section labels for slow-motion targeting
    segment_centers: List[float] = []
    segment_labels: List[str] = []

    for (t0, t1, beat_count) in grouped:
        dur = max(0.35, t1 - t0)
        center = 0.5 * (t0 + t1)
        energy = energy_class(center)
        section_label = _section_label_at(center)
        # Intro transitions-only: strip FX early (keep transitions)
        if intro_transitions_only and (section_label or '').lower() == 'intro':
            effect = 'none'

        # Average spacing between beats in this segment (proxy for beat density)
        avg_gap = dur / max(1, beat_count)
        is_calm = (avg_gap >= CALM_GAP_THRESHOLD) or (energy == "low")
        just_after_calm = (not is_calm) and previous_was_calm

        # Base segment length
        if is_calm:
            # Calm / break-like section: use longer, more relaxed shots.
            if dur <= CALM_MIN_LEN:
                seg_len = dur
            else:
                max_len = min(CALM_MAX_LEN, dur)
                seg_len = random.uniform(CALM_MIN_LEN, max_len)
        elif just_after_calm:
            # First segment right after a calm/break: emphasise the beat
            # by forcing a short accent shot so the new beat is noticeable.
            if dur <= 1.0:
                seg_len = dur
            else:
                seg_len = max(0.3, min(1.0, dur, random.uniform(0.4, 1.0)))
        else:
            # Microclip logic for active sections.
            if microclip_mode == 0:
                seg_len = min(dur, 4.0)
            elif microclip_mode == 1:
                if energy == "high":
                    seg_len = max(0.3, min(0.8, dur, random.uniform(0.3, 0.8)))
                else:
                    seg_len = min(dur, 4.0)
            elif microclip_mode == 2:
                if energy == "high":
                    base_min, base_max = 0.25, 0.7
                elif energy == "mid":
                    base_min, base_max = 0.4, 1.0
                else:
                    base_min, base_max = 0.6, 1.3
                seg_len = max(
                    base_min,
                    min(base_max, dur, random.uniform(base_min, base_max)),
                )
            else:
                # Verses only: microclips are used only during verse sections.
                if section_label == "verse":
                    if energy == "high":
                        base_min, base_max = 0.25, 0.7
                    elif energy == "mid":
                        base_min, base_max = 0.4, 1.0
                    else:
                        base_min, base_max = 0.6, 1.3
                    seg_len = max(
                        base_min,
                        min(base_max, dur, random.uniform(base_min, base_max)),
                    )
                else:
                    seg_len = min(dur, 4.0)

        # Safety: never exceed the beat-group window duration.
        seg_len = min(seg_len, dur)

        use_image = False

        # Prefer explicit per-section media overrides when present.
        sec_idx = _section_index_at(center)
        override_src: Optional[ClipSource] = None
        if section_overrides and sec_idx is not None:
            override_src = section_overrides.get(sec_idx)

        if override_src is not None:
            if bool(getattr(override_src, "is_image", False)):
                # For still images, always use the full frame.
                clip_path = override_src.path
                clip_start = 0.0
                use_image = True
            else:
                # For override videos, walk through the file across segments.
                total = float(getattr(override_src, "duration", 0.0) or 0.0)
                offset = section_override_offsets.get(sec_idx, 0.0)
                if total > 0.0 and offset + seg_len > total:
                    offset = 0.0
                clip_path = override_src.path
                clip_start = max(0.0, offset)
                section_override_offsets[sec_idx] = offset + seg_len
        elif image_sources:
            # Deterministic pacing for still images based on the user slider.
            # If image_interval <= 0, fall back to the previous random behaviour.
            if image_interval > 0 and image_sources:
                if segments_since_image >= max(1, image_interval) - 1:
                    source = image_sources[image_index % len(image_sources)]
                    image_index += 1
                    clip_path = source.path
                    clip_start = 0.0  # Images have no "start" in file
                    use_image = True
                    segments_since_image = 0
                else:
                    clip_path, clip_start = pick_region(seg_len)
                    segments_since_image += 1
            else:
                # Legacy ~15%% random image insert behaviour.
                if image_sources and random.random() < 0.15:
                    source = random.choice(image_sources)
                    clip_path = source.path
                    clip_start = 0.0  # Images have no "start" in file
                    use_image = True
                else:
                    clip_path, clip_start = pick_region(seg_len)
        else:
            clip_path, clip_start = pick_region(seg_len)

        previous_was_calm = is_calm

        # FX selection by level and energy
        # Now supports a small library of segment FX:
        #   - zoom        : gentle punch-in
        #   - flash       : small brightness pop
        #   - rgb_split   : chromatic aberration
        #   - vhs         : VHS-style noise + scanlines
        #   - motion_blur : mild global blend
        effect = "none"
        r = random.random()

        if fx_level == "minimal":
            # Very subtle: only occasional zoom on strong peaks
            if energy == "high" and r < 0.18:
                effect = "zoom"

        elif fx_level == "moderate":
            if energy == "high":
                if r < 0.25:
                    effect = "zoom"
                elif r < 0.45:
                    effect = "flash"
                elif r < 0.65:
                    effect = "rgb_split"
                elif r < 0.78:
                    effect = "vhs"
                elif r < 0.90:
                    effect = "motion_blur"
            elif energy == "mid":
                if r < 0.20:
                    effect = random.choice(["zoom", "rgb_split"])

        elif fx_level == "high":  # high
            if energy == "high":
                if r < 0.25:
                    effect = "zoom"
                elif r < 0.25:
                    effect = "flash"
                elif r < 0.85:
                    effect = "rgb_split"
                elif r < 0.80:
                    effect = "vhs"
                else:
                    effect = "motion_blur"
            elif energy == "mid":
                if r < 0.30:
                    effect = random.choice(["zoom", "rgb_split", "flash"])
                elif r < 0.45:
                    effect = "vhs"
                elif r < 0.60:
                    effect = "motion_blur"
            else:
                if r < 0.22:
                    effect = random.choice(["zoom", "rgb_split"])

        # Transition choice (fade-safe, using a small transition effect library)
        if transition_random and allowed_modes:
            mode_for_segment = random.choice(allowed_modes)
        else:
            mode_for_segment = transition_mode

        # Transition style mapping (must match the UI order in "Transitions" + "Manage transitions")
        # 0  Soft film dissolves           -> real between-clip dissolve (stitched later)
        # 1  Hard cuts                     -> no transition
        # 2  Scale punch (zoom)            -> real xfade punch (stitched later)
        # 3  Shimmer blur (shiny)           -> real blurred crossfade (stitched later)

        # 4  Iris reveal (circle)          -> real stitched iris transition (xfade)
        # 5  Motion blur whip-cuts         -> in-clip motion blur boost
        # 6  Slit-scan smear push          -> real modern smear + push (stitched later)
        # 7  Radial burst reveal           -> real stitched burst reveal (circleopen + flash/blur overlap)
        # 8  Directional push (slide)      -> real between-clip push/slide (stitched later)
        # 9  Wipe                          -> real between-clip wipe (stitched later)
        # 10 Smooth zoom crossfade         -> in-clip smooth zoom
        # 11 Curtain open (doors)         -> real between-clip curtain/doors open (stitched later)
        # 12 Pixelize                      -> real between-clip pixelize (stitched later)
        # 13 Distance (liquid blend)          -> real xfade distance (stitched later)
        # 14 Wind smears                    -> real xfade wind smears (stitched later)
        if mode_for_segment == 1:  # Hard cuts
            transition = "none"

        elif mode_for_segment == 0:  # Soft film dissolves (real dissolve between clips)
            transition = "t_exposure_dissolve"

        elif mode_for_segment == 2:  # Scale punch (zoom)
            transition = "t_scale_punch"

        elif mode_for_segment == 3:  # Shimmer blur (shiny)
            transition = "t_shimmer_blur"

        elif mode_for_segment == 4:  # Iris reveal (circle)
            transition = "t_iris"

        elif mode_for_segment == 5:  # Motion blur whip-cuts
            transition = "t_motion_blur"

        elif mode_for_segment == 6:  # Slit-scan smear push (real between-clip transition)
            transition = "t_slitscan_push"


        elif mode_for_segment == 7:  # Radial burst reveal (real stitched transition)
            transition = "t_radial_burst"

        elif mode_for_segment == 8:  # Directional push (slide)
            transition = "t_push"

        elif mode_for_segment == 9:  # Wipe
            transition = "t_wipe"

        elif mode_for_segment == 10:  # Smooth zoom crossfade
            transition = "t_smooth_zoom"

        elif mode_for_segment == 11:  # Curtain open (doors)
            transition = "t_curtain_open"

        elif mode_for_segment == 12:  # Pixelize
            transition = "t_pixelize"

        elif mode_for_segment == 13:  # Distance (liquid blend)
            transition = "t_distance"

        elif mode_for_segment == 14:  # Wind smears
            transition = "t_wind_smears"

        else:
            transition = "none"
        segments.append(
            TimelineSegment(
                clip_path=clip_path,
                clip_start=clip_start,
                duration=seg_len,
                effect=effect,
                energy_class=energy,
                transition=transition,
                is_image=use_image,
            )
        )
        # Track center time and section label for slow-motion logic
        segment_centers.append(center)
        segment_labels.append(section_label)


    # Apply slow-motion decisions on built segments
    _apply_slow_motion(segments, segment_centers, segment_labels)

    # Rebuild timeline positions so slow-motion segments
    # take up more room on the musical timeline.
    _rebuild_timeline_positions(segments)



    # REAL DROP DETECTION – THE ONE THAT ACTUALLY FEELS PERFECT
    # For each detected 'break' section, look ahead a few seconds and find the
    # strongest major beat that follows. The segment that actually *contains*
    # that beat becomes the break-impact moment, and its FX strength scales
    # with the beat power so hard drops feel noticeably bigger.
    if impact_enable and analysis.sections and analysis.beats and segments:
        break_sections = [s for s in analysis.sections if s.kind == "break"]

        for break_sec in break_sections:
            window_start = break_sec.end
            window_end = min(analysis.duration, window_start + 6.0)  # look up to 6 seconds after break

            # Get all major beats in this window
            candidates = [
                b for b in analysis.beats
                if window_start <= b.time < window_end and b.kind == "major"
            ]

            if not candidates:
                continue

            # THIS IS THE MAGIC LINE
            strongest_beat = max(candidates, key=lambda b: b.strength)

            # Find the segment that contains this absolute god-tier beat
            for seg in segments:
                if seg.timeline_start <= strongest_beat.time < seg.timeline_end:
                    seg.is_break_impact = True

                    # Optional: make the impact strength follow the actual beat power
                    power = min(1.0, strongest_beat.strength * 1.8)  # your strength is ~0.3–1.0

                    if impact_flash:
                        seg.impact_flash_strength = max(seg.impact_flash_strength, power * 1.0)
                    if impact_shock:
                        seg.impact_shock_strength = max(seg.impact_shock_strength, power * 0.9)
                    if impact_zoom:
                        seg.impact_zoom_amount = max(seg.impact_zoom_amount, power * 0.4)
                    if impact_shake:
                        seg.impact_shake_strength = max(seg.impact_shake_strength, power * 0.8)
                    if impact_confetti:
                        seg.impact_confetti_density = max(seg.impact_confetti_density, power * 0.7)
                    if impact_color_cycle:
                        seg.impact_color_cycle_speed = max(seg.impact_color_cycle_speed, power * 1.0)
                    break

    # Extra: ensure the main drop (loudest beat inside any 'drop' section)
    # is always tagged as an impact segment based on its true beat strength.
    if impact_enable and analysis.sections and analysis.beats:
        drop_ranges = [(s.start, s.end) for s in analysis.sections if s.kind == "drop"]
        if drop_ranges:
            drop_beats: List[Beat] = []
            for b in analysis.beats:
                for start_t, end_t in drop_ranges:
                    if start_t <= b.time < end_t:
                        drop_beats.append(b)
                        break
            if drop_beats:
                chosen_beat = max(drop_beats, key=lambda b: b.strength)
                intensity = min(1.0, chosen_beat.strength * 2.0)
                for seg in segments:
                    if seg.timeline_start <= chosen_beat.time < seg.timeline_end:
                        seg.is_break_impact = True
                        # Optionally boost the effect strength based on how hard the beat actually is.
                        if impact_flash:
                            seg.impact_flash_strength = max(seg.impact_flash_strength, intensity * 1.0)
                        if impact_zoom:
                            seg.impact_zoom_amount = max(seg.impact_zoom_amount, intensity * 0.3)
                        if impact_shake:
                            seg.impact_shake_strength = max(seg.impact_shake_strength, intensity * 0.8)
                        if impact_color_cycle:
                            seg.impact_color_cycle_speed = max(seg.impact_color_cycle_speed, intensity * 1.0)
                        break

    # Optionally extend timeline so total video duration matches full music length
    if force_full_length and segments:
        total_video = sum(seg.duration for seg in segments)
        target = analysis.duration
        if audio_duration is not None and audio_duration > 0:
            try:
                target = max(target, float(audio_duration))
            except Exception:
                pass
        # Only extend (never shrink); allow a small tolerance before extending.
        if total_video < target * 0.998 and target > 0:
            remaining = target - total_video
            # Use a conservative average length for filler segments
            avg_len = max(0.5, min(2.5, total_video / len(segments)))
            def _make_filler_segment(start_fraction: float, length: float) -> TimelineSegment:
                # Clamp logical time to the track duration
                t_center = target * min(max(start_fraction, 0.0), 1.0)
                energy = energy_class(t_center)
                clip_path, clip_start = pick_region(length)
                # Use similar transition logic for fillers, but a bit softer
                if transition_random and allowed_modes:
                    mode_for_segment = random.choice(allowed_modes)
                else:
                    mode_for_segment = transition_mode

                if mode_for_segment == 1:  # Hard cuts
                    transition = "none"
                elif mode_for_segment == 0:  # Soft film dissolves (stitched later)
                    transition = "t_exposure_dissolve"
                elif mode_for_segment == 2:  # Scale punch (zoom)
                    # Keep a tiny chance of a flashcut on high energy to avoid "too smooth" late filler.
                    if energy == "high" and random.random() < 0.22:
                        transition = "flashcut"
                    else:
                        transition = "t_scale_punch"
                elif mode_for_segment == 3:  # Shimmer blur (shiny)
                    transition = "t_shimmer_blur"
                elif mode_for_segment == 4:  # Iris reveal (circle)
                    transition = "t_iris"
                elif mode_for_segment == 5:  # Motion blur whip-cuts
                    transition = "t_motion_blur"
                elif mode_for_segment == 6:  # Slit-scan smear push (stitched later)
                    transition = "t_slitscan_push"
                elif mode_for_segment == 7:  # Radial burst reveal (stitched later)
                    transition = "t_radial_burst"
                elif mode_for_segment == 8:  # Directional push (stitched later)
                    transition = "t_push"
                elif mode_for_segment == 9:  # Wipe (stitched later)
                    transition = "t_wipe"
                elif mode_for_segment == 10:  # Smooth zoom crossfade
                    transition = "t_smooth_zoom"
                elif mode_for_segment == 11:  # Curtain open (stitched later)
                    transition = "t_curtain_open"
                elif mode_for_segment == 12:  # Pixelize (stitched later)
                    transition = "t_pixelize"
                elif mode_for_segment == 13:  # Distance (stitched later)
                    transition = "t_distance"
                elif mode_for_segment == 14:  # Wind smears (stitched later)
                    transition = "t_wind_smears"
                else:
                    transition = "none"
                seg = TimelineSegment(
                    clip_path=clip_path,
                    clip_start=clip_start,
                    duration=length,
                    effect="none",
                    energy_class=energy,
                    transition=transition,
                )
                # Track filler center time and label as well
                segment_centers.append(t_center)
                segment_labels.append(_section_label_at(t_center))
                return seg

            # Add filler segments towards the end of the track until we cover the gap
            while remaining > 0.05:
                seg_len = min(avg_len, remaining)
                start_frac = (target - remaining + seg_len * 0.5) / target
                filler = _make_filler_segment(start_frac, seg_len)
                segments.append(filler)
                remaining -= seg_len

    # Final pass: also rebuild after any optional extension so filler
    # segments receive proper timeline_start / timeline_end values.
    _rebuild_timeline_positions(segments)

    # After extension, apply cinematic and break-impact FX across the
    # full timeline so late filler segments also get hits. We then
    # rebuild positions again before constructing the calm outro, so
    # the last few seconds can still avoid hard cuts.
    if segments:
        segment_centers_final: List[float] = []
        for seg in segments:
            start_t = float(getattr(seg, 'timeline_start', 0.0) or 0.0)
            end_t = float(getattr(seg, 'timeline_end', 0.0) or 0.0)
            center_t = start_t + max(0.0, (end_t - start_t)) * 0.5
            segment_centers_final.append(center_t)

        _apply_cinematic_effects(segments, allow_mosaic, segment_labels)
        _apply_break_impact_fx(segments, segment_centers_final, segment_labels)
        _rebuild_timeline_positions(segments)

    # --- CINEMATIC OUTRO: reserve the final few seconds for 1–2 calm shots ---
    OUTRO_DURATION = 8.0

    # Prefer an explicit audio duration when available, otherwise fall back to
    # the analysed track length.
    track_len = None
    try:
        if audio_duration is not None and float(audio_duration) > 0.0:
            track_len = float(audio_duration)
        else:
            track_len = float(analysis.duration)
    except Exception:
        track_len = None

    if segments and track_len is not None and track_len > 10.0:
        # We want the last ~OUTRO_DURATION seconds to be a calm outro.
        target_outro_start = max(0.0, track_len - OUTRO_DURATION)

        # Keep all segments that clearly end before the planned outro window.
        main_segments: List[TimelineSegment] = []
        for seg in segments:
            if getattr(seg, "timeline_end", 0.0) <= target_outro_start:
                main_segments.append(seg)
            else:
                break

        if main_segments:
            # Remaining logical time we want to cover with the outro.
            outro_start_time = getattr(main_segments[-1], "timeline_end", 0.0)
            remaining = max(0.5, track_len - outro_start_time)

            # Pick the longest (or two longest) clips that can reasonably cover the outro.
            eligible = [s for s in sources if s.duration >= remaining * 0.7]
            if eligible:
                k = min(8, len(eligible))
                candidates = random.sample(eligible, k=k)
            else:
                candidates = sorted(sources, key=lambda s: s.duration, reverse=True)[:8]
            if not candidates:
                candidates = sorted(sources, key=lambda s: s.duration, reverse=True)[:2]

            outro_segments: List[TimelineSegment] = []
            time_left = remaining
            for clip in candidates:
                if time_left <= 0.0:
                    break
                seg_dur = min(clip.duration, time_left)
                if seg_dur < 2.0:
                    continue

                seg = TimelineSegment(
                    clip_path=clip.path,
                    clip_start=0.0,
                    duration=seg_dur,
                    effect="minimal",
                    energy_class="low",
                    transition="fade",
                    slow_factor=0.85,   # gentle slow‑mo
                    cine_speed_ramp=True,
                    cine_ramp_out=0.15,
                )
                outro_segments.append(seg)
                time_left -= seg_dur

            if outro_segments:
                # Adjust the final outro segment so that, after slow‑motion is applied,
                # the overall logical length is close to the track length.
                def _effective_len(seg: TimelineSegment) -> float:
                    try:
                        sf = float(getattr(seg, "slow_factor", 1.0) or 1.0)
                    except Exception:
                        sf = 1.0
                    base = float(getattr(seg, "duration", 0.0))
                    if 0.01 < sf < 1.0 and base > 0.0:
                        return base / sf
                    return base

                combined = main_segments + outro_segments
                total_eff = sum(_effective_len(s) for s in combined)
                if total_eff > 0.0:
                    last = outro_segments[-1]
                    last_eff = _effective_len(last)
                    delta = track_len - total_eff
                    # Convert the desired change in effective length back to base duration.
                    try:
                        sf_last = float(getattr(last, "slow_factor", 1.0) or 1.0)
                    except Exception:
                        sf_last = 1.0
                    adjust = delta * (sf_last if sf_last < 1.0 else 1.0)
                    if last.duration + adjust > 0.2:
                        last.duration += adjust

                segments = main_segments + outro_segments

                # Rebuild positions one more time so the new outro snaps cleanly
                # to the end of the track.
                _rebuild_timeline_positions(segments)

    # Final safety: when intro transitions-only is enabled, ensure no FX are applied during intro.
    if intro_transitions_only and segments:
        # Ensure timeline positions exist so we can detect intro segments reliably.
        try:
            _rebuild_timeline_positions(segments)
        except Exception:
            pass

        def _reset_fx_on_segment(seg: TimelineSegment) -> None:
            # Core toggles
            try:
                seg.effect = "none"
            except Exception:
                pass
            try:
                seg.slow_factor = 1.0
            except Exception:
                pass

            # Clear known / dynamic cinematic + impact attributes
            try:
                items = list(vars(seg).items())
            except Exception:
                items = []
            for k, v in items:
                if not (k.startswith("cine_") or k.startswith("impact_") or k == "is_break_impact"):
                    continue
                try:
                    if isinstance(v, bool):
                        setattr(seg, k, False)
                    elif isinstance(v, int):
                        setattr(seg, k, 0)
                    elif isinstance(v, float):
                        if k.endswith("_factor"):
                            setattr(seg, k, 1.0)
                        else:
                            setattr(seg, k, 0.0)
                    elif isinstance(v, str):
                        setattr(seg, k, "")
                    else:
                        try:
                            delattr(seg, k)
                        except Exception:
                            pass
                except Exception:
                    pass

        for seg in segments:
            try:
                st = float(getattr(seg, "timeline_start", 0.0) or 0.0)
                en = float(getattr(seg, "timeline_end", st) or st)
                center_t = st + max(0.0, en - st) * 0.5
            except Exception:
                center_t = 0.0
            lbl = (_section_label_at(center_t) or "").lower()
            if lbl == "intro":
                _reset_fx_on_segment(seg)

    # Timed strobe: map user timestamps to segment-local offsets so the Flash strobe can fire on time.
    if strobe_on_time and strobe_on_time_times and segments:
        times: List[float] = []
        for v in strobe_on_time_times:
            try:
                fv = float(v)
            except Exception:
                continue
            if fv < 0.0:
                continue
            times.append(fv)
        # De-dup + sort
        uniq: List[float] = []
        for t in sorted(times):
            if not uniq or abs(uniq[-1] - t) >= 0.01:
                uniq.append(t)

        for t in uniq:
            # Find the segment that covers this timestamp (output timeline seconds).
            best_idx: Optional[int] = None
            for i, seg in enumerate(segments):
                try:
                    st = float(getattr(seg, "timeline_start", 0.0) or 0.0)
                    en = float(getattr(seg, "timeline_end", st) or st)
                except Exception:
                    continue
                if st <= t < en or (i == len(segments) - 1 and st <= t <= en):
                    best_idx = i
                    break
            if best_idx is None:
                continue

            seg = segments[best_idx]
            try:
                st = float(getattr(seg, "timeline_start", 0.0) or 0.0)
                en = float(getattr(seg, "timeline_end", st) or st)
            except Exception:
                st, en = 0.0, 0.0
            seg_len = max(0.0, en - st)
            if seg_len <= 0.0:
                continue

            off = max(0.0, min(seg_len, t - st))
            try:
                seg.strobe_on_time_offsets.append(off)
            except Exception:
                try:
                    setattr(seg, "strobe_on_time_offsets", [off])
                except Exception:
                    pass

            # Reuse the Flash strobe settings (strength/speed) even if the break-impact toggles are off.
            try:
                cur = float(getattr(seg, "impact_flash_strength", 0.0) or 0.0)
            except Exception:
                cur = 0.0
            try:
                setattr(seg, "impact_flash_strength", max(cur, float(impact_flash_strength)))
            except Exception:
                pass
            try:
                setattr(seg, "impact_flash_speed_ms", int(impact_flash_speed_ms))
            except Exception:
                pass


    return segments


# ---------------------------- render worker --------------------------------



class SourceScanWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(list)
    failed = Signal(str)

    def __init__(
        self,
        video_input: str,
        ffprobe: str,
        min_clip_length: float | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.video_input = video_input
        self.ffprobe = ffprobe
        self.min_clip_length = min_clip_length

    def run(self) -> None:
        """Scan video sources in a background thread with caching + progress.

        This worker mirrors discover_video_sources() but:
        - Emits incremental progress updates so the UI can show scan status.
        - Caches per‑file durations under the app's temp directory so
          repeated scans of the same folder are much faster.
        """
        try:
            exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg"}

            video_input = self.video_input
            ffprobe = self.ffprobe

            # Build list of candidate files (single file, folder, or '|' list).
            paths: list[str] = []
            if "|" in video_input:
                parts = [p.strip() for p in str(video_input).split("|") if p.strip()]
                for raw in parts:
                    path = os.path.abspath(raw)
                    if not os.path.isfile(path):
                        continue
                    _, ext = os.path.splitext(path)
                    if ext.lower() not in exts:
                        continue
                    paths.append(path)
            else:
                base = os.path.abspath(video_input)
                if os.path.isdir(base):
                    try:
                        names = sorted(os.listdir(base))
                    except Exception:
                        names = []
                    for name in names:
                        path = os.path.join(base, name)
                        if not os.path.isfile(path):
                            continue
                        _, ext = os.path.splitext(path)
                        if ext.lower() not in exts:
                            continue
                        paths.append(path)
                elif os.path.isfile(base):
                    _, ext = os.path.splitext(base)
                    if ext.lower() in exts:
                        paths.append(base)

            if not paths:
                self.failed.emit("__no_clips__")
                return

            # Cache disabled: do not read or write scan_cache.json; always probe durations fresh.
            cache: dict[str, dict] = {}
            cache_file: str | None = None
            dirty = False

            sources: list[ClipSource] = []
            total = len(paths)

            for idx, path in enumerate(paths):
                try:
                    st = os.stat(path)
                    key = os.path.abspath(path)
                    entry = cache.get(key) if isinstance(cache, dict) else None
                    dur: float | None = None

                    if isinstance(entry, dict):
                        try:
                            if (
                                entry.get("mtime") == st.st_mtime
                                and entry.get("size") == st.st_size
                            ):
                                d_val = float(entry.get("duration", 0.0) or 0.0)
                                if d_val > 0.0:
                                    dur = d_val
                        except Exception:
                            dur = None

                    if dur is None:
                        dur = _ffprobe_duration(ffprobe, path)
                        if dur and dur > 0:
                            if isinstance(cache, dict):
                                cache[key] = {
                                    "duration": float(dur),
                                    "mtime": st.st_mtime,
                                    "size": st.st_size,
                                }
                                dirty = True

                    if dur and dur > 0:
                        sources.append(ClipSource(path=path, duration=float(dur)))
                except Exception:
                    # Ignore individual file failures; we only care about usable clips.
                    pass

                # Update scan progress (0–10%% reserved for scanning).
                try:
                    pct = int(1 + 9 * ((idx + 1) / max(1, total)))
                    self.progress.emit(pct, f"Scanning clips {idx+1}/{total}...")
                except Exception:
                    pass

            if not sources:
                self.failed.emit("__no_clips__")
                return

            # Apply optional minimum clip length filter.
            if self.min_clip_length is not None and self.min_clip_length > 0:
                usable = [s for s in sources if s.duration >= self.min_clip_length]
                if not usable:
                    # Signal a specific "all too short" condition back to the UI.
                    self.failed.emit(f"__all_short__:{self.min_clip_length:.1f}")
                    return
                sources = usable

            # Cache persistence disabled: no scan_cache.json will be written.
            # if dirty and cache_file and isinstance(cache, dict):
            #     try:
            #         with open(cache_file, "w", encoding="utf-8") as f:
            #             json.dump(cache, f)
            #     except Exception:
            #         # Cache write failure is non-fatal.
            #         pass

            self.finished_ok.emit(sources)
        except Exception as e:
            self.failed.emit(str(e))


class RenderWorker(QThread):
    progress = Signal(int, str)
    finished_ok = Signal(str)
    failed = Signal(str)

    def __init__(
        self,
        audio_path: str,
        output_dir: str,
        analysis: MusicAnalysisResult,
        segments: List[TimelineSegment],
        ffmpeg: str,
        ffprobe: str,
        target_resolution: Optional[Tuple[int, int]],
        fit_mode: int,
        transition_mode: int,
        intro_fade: bool,
        outro_fade: bool,
        keep_source_bitrate: bool = False,
        use_visual_overlay: bool = False,
        visual_strategy: int = 0,
        visual_section_overrides: Optional[Dict[str, Optional[str]]] = None,
        visual_overlay_opacity: float = 0.25,
        out_name_override: Optional[str] = None,
        strobe_on_time_times: Optional[List[float]] = None,
        strobe_flash_strength: float = 0.0,
        strobe_flash_speed_ms: int = 250,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.analysis = analysis
        self.segments = segments
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.target_resolution = target_resolution
        self.fit_mode = fit_mode
        self.transition_mode = transition_mode
        self.intro_fade = intro_fade
        self.outro_fade = outro_fade
        self.keep_source_bitrate = bool(keep_source_bitrate)
        # When True, render a music-player visual track and overlay it.
        self.use_visual_overlay = bool(use_visual_overlay)
        # 0 = single visual, 1 = per segment, 2 = per section type
        try:
            self.visual_strategy = int(visual_strategy)
        except Exception:
            self.visual_strategy = 0
        self.visual_section_overrides = visual_section_overrides or {}
        try:
            self.visual_overlay_opacity = float(visual_overlay_opacity)
        except Exception:
            self.visual_overlay_opacity = 0.25
        self.out_name_override = out_name_override

        # Timed strobe (global): keep these on the worker so we can apply the strobe
        # after any duration correction / overlays, guaranteeing accurate timestamps.
        try:
            self.strobe_on_time_times = list(strobe_on_time_times) if strobe_on_time_times else []
        except Exception:
            self.strobe_on_time_times = []
        try:
            self.strobe_flash_strength = float(strobe_flash_strength or 0.0)
        except Exception:
            self.strobe_flash_strength = 0.0
        try:
            self.strobe_flash_speed_ms = int(strobe_flash_speed_ms or 250)
        except Exception:
            self.strobe_flash_speed_ms = 250


    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as e:
            self.failed.emit(str(e))

    def _run_impl(self) -> None:
        if not self.segments:
            raise RuntimeError("Timeline is empty.")

        _ensure_dir(self.output_dir)
        tmpdir = tempfile.mkdtemp(prefix="fv_mclip_render_")
        # If enabled, probe an approximate source bitrate so we can preserve quality.
        # For multi-clip projects we take the max bitrate across a small sample of unique clips.
        self._keep_bitrate_kbps = 0
        if getattr(self, "keep_source_bitrate", False):
            try:
                seen = []
                for seg in (self.segments or []):
                    pth = getattr(seg, "clip_path", None)
                    if not pth or not isinstance(pth, str):
                        continue
                    if getattr(seg, "is_image", False):
                        continue
                    if pth not in seen:
                        seen.append(pth)
                    if len(seen) >= 20:
                        break

                best_bps = 0
                for pth in seen:
                    br = _ffprobe_video_bitrate(self.ffprobe, pth)
                    if br and br > best_bps:
                        best_bps = int(br)

                kbps = int(round(best_bps / 1000.0)) if best_bps > 0 else 0
                # Clamp to a sensible range; ignore obviously wrong values.
                if kbps < 250:
                    kbps = 0
                if kbps > 200000:
                    kbps = 200000
                self._keep_bitrate_kbps = kbps
            except Exception:
                self._keep_bitrate_kbps = 0


        # Default render FPS is 30 for multi-clip projects (normalizes mixed sources).
        # If the project is effectively a single-source video (one unique clip path),
        # keep the source FPS to avoid forcing 30fps on export.
        self._target_fps_expr = "30"
        self._target_fps_float = 30.0
        try:
            uniq_paths = []
            for seg in (self.segments or []):
                pth = getattr(seg, "clip_path", None)
                if not pth or not isinstance(pth, str):
                    continue
                if getattr(seg, "is_image", False):
                    continue
                if pth not in uniq_paths:
                    uniq_paths.append(pth)
                if len(uniq_paths) > 1:
                    break

            if len(uniq_paths) == 1:
                fps_src = _ffprobe_fps_expr(self.ffprobe, uniq_paths[0])
                if fps_src:
                    self._target_fps_expr = str(fps_src)
                    try:
                        if "/" in fps_src:
                            a, b = fps_src.split("/", 1)
                            self._target_fps_float = float(a) / max(1.0, float(b))
                        else:
                            self._target_fps_float = float(fps_src)
                    except Exception:
                        self._target_fps_float = 30.0
        except Exception:
            pass

        fps_expr = self._target_fps_expr
        fps_float = float(getattr(self, "_target_fps_float", 30.0) or 30.0)

        def _x264_bitrate_args() -> list[str]:
            kb = int(getattr(self, "_keep_bitrate_kbps", 0) or 0)
            if kb <= 0:
                return []
            # Constrained VBR around the source bitrate.
            return [
                "-b:v", f"{kb}k",
                "-maxrate", f"{kb}k",
                "-bufsize", f"{max(1, kb*2)}k",
            ]

        self._x264_bitrate_args = _x264_bitrate_args


        def _x264_encode_args() -> list[str]:
            kb = int(getattr(self, "_keep_bitrate_kbps", 0) or 0)
            if kb <= 0:
                return ["-crf", "18"]
            maxr = int(max(1, round(kb * 1.05)))
            buf = int(max(1, kb * 2))
            return ["-b:v", f"{kb}k", "-maxrate", f"{maxr}k", "-bufsize", f"{buf}k"]

        self._x264_encode_args = _x264_encode_args


        try:
            # Collect all available video source paths once for Mosaic effects.
            all_video_paths = sorted(
                {
                    getattr(s, "clip_path", None)
                    for s in self.segments
                    if getattr(s, "clip_path", None)
                    and not getattr(s, "is_image", False)
                }
            )
            duration_cache: dict[str, float] = {}

            # Precompute overlap durations for "real transitions" (xfade-based).
            # We store the duration on the *incoming* segment (segment j), so we can:
            # - extend that segment by exactly the overlap at render time (prevents end-freeze padding)
            # - reuse the same duration during stitching (stable timing)
            dissolve_ids = {"t_exposure_dissolve", "t_luma_fade"}
            scale_punch_ids = {"t_scale_punch"}
            push_ids = {"t_push"}
            slitscan_ids = {"t_slitscan_push"}
            wipe_ids = {"t_wipe"}
            curtain_ids = {"t_curtain_open"}
            pixelize_ids = {"t_pixelize"}
            iris_ids = {"t_iris"}
            radial_ids = {"t_radial_burst"}
            shimmer_ids = {"t_shimmer_blur"}
            distance_ids = {"t_distance"}
            wind_ids = {"t_wind_smears"}
            stitch_ids = dissolve_ids | push_ids | slitscan_ids | wipe_ids | curtain_ids | pixelize_ids | iris_ids | radial_ids | shimmer_ids | distance_ids | wind_ids | scale_punch_ids

            def _seg_expected_out_len(s: TimelineSegment) -> float:
                try:
                    base = float(getattr(s, "duration", 0.0) or 0.0)
                except Exception:
                    base = 0.0
                try:
                    sf = float(getattr(s, "slow_factor", 1.0) or 1.0)
                except Exception:
                    sf = 1.0
                if sf == 0.0:
                    sf = 1.0
                # Render applies setpts=PTS/sf when sf != 1.0, so duration scales by 1/sf.
                return (base / sf) if sf != 1.0 else base

            # Default to no overlap everywhere.
            for _s in self.segments:
                try:
                    setattr(_s, "_stitch_dur", 0.0)
                except Exception:
                    pass

            for j in range(1, len(self.segments)):
                try:
                    trans = getattr(self.segments[j], "transition", None) or "none"
                except Exception:
                    trans = "none"
                if trans not in stitch_ids:
                    continue

                prev_out = _seg_expected_out_len(self.segments[j - 1])
                cur_out = _seg_expected_out_len(self.segments[j])
                base = min(prev_out, cur_out)

                if base <= 0.0:
                    d = 0.0
                else:
                    # Visible, editor-like overlap (clamped to clip lengths).
                    d = base * 0.28
                    d = max(0.18, min(0.85, d))
                    d = min(d, max(0.12, base * 0.40))

                try:
                    setattr(self.segments[j], "_stitch_dur", float(d))
                except Exception:
                    pass

            parts = []
            n = len(self.segments)
            for i, seg in enumerate(self.segments):
                pct = int(5 + 80 * (i / max(1, n)))
                self.progress.emit(pct, f"Rendering segment {i+1}/{n}...")
                out_part = os.path.join(tmpdir, f"part_{i:04d}.mp4")

                vf_parts: List[str] = []
                base_vf_parts: List[str] = []

                # Extend the incoming segment by its real-transition overlap so the final stitched
                # video keeps the original timeline length without padding frozen frames at the end.
                try:
                    seg_base_dur = float(getattr(seg, "duration", 0.0) or 0.0)
                except Exception:
                    seg_base_dur = 0.0
                try:
                    seg_sf = float(getattr(seg, "slow_factor", 1.0) or 1.0)
                except Exception:
                    seg_sf = 1.0
                if seg_sf == 0.0:
                    seg_sf = 1.0
                try:
                    seg_stitch_dur_out = float(getattr(seg, "_stitch_dur", 0.0) or 0.0)
                except Exception:
                    seg_stitch_dur_out = 0.0
                seg_extra_src = seg_stitch_dur_out * seg_sf if seg_stitch_dur_out > 0.0 else 0.0
                seg_trim_dur = max(0.01, seg_base_dur + seg_extra_src)
                seg_out_dur = seg_trim_dur / seg_sf if seg_sf != 0.0 else seg_trim_dur

                # If the extended trim window would run off the end of the source, slide the start back.
                if (not getattr(seg, "is_image", False)) and getattr(seg, "clip_path", None):
                    try:
                        src_dur = duration_cache.get(seg.clip_path)
                        if src_dur is None:
                            src_dur = _ffprobe_duration(self.ffprobe, seg.clip_path)
                            if src_dur is not None:
                                duration_cache[seg.clip_path] = float(src_dur)
                        if src_dur and float(src_dur) > 0.0:
                            try:
                                cs = float(getattr(seg, "clip_start", 0.0) or 0.0)
                            except Exception:
                                cs = 0.0
                            if cs + seg_trim_dur > float(src_dur) - 0.02:
                                seg.clip_start = max(0.0, float(src_dur) - seg_trim_dur)
                    except Exception:
                        pass

                if self.target_resolution is not None:
                    w, h = self.target_resolution
                    try:
                        fit_mode = int(getattr(self, "fit_mode", 0))
                    except Exception:
                        fit_mode = 0

                    if fit_mode == 1:  # Fill (crop to fill)
                        base_vf_parts = [
                            f"scale={w}:{h}:force_original_aspect_ratio=increase",
                            f"crop={w}:{h}",
                        ]
                    elif fit_mode == 2:  # Stretch (distort to fill)
                        base_vf_parts = [
                            f"scale={w}:{h}",
                        ]
                    else:  # Original (letterbox / pillarbox)
                        base_vf_parts = [
                            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
                            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black",
                        ]
                    vf_parts.extend(base_vf_parts)

                # Ken Burns zoom for images
                if getattr(seg, "is_image", False) and self.target_resolution is not None:
                    try:
                        w, h = self.target_resolution
                    except Exception:
                        w, h = 1920, 1080
                    # Match zoom duration to the (possibly extended) segment length so the image animates for the full segment.
                    seg_dur = float(seg_out_dur or 0.0)
                    if seg_dur <= 0.0:
                        seg_dur = 2.5  # sensible fallback
                    kb_frames = max(1, int(seg_dur * 30.0))
                    zoom_dir = "zoom+0.0025" if random.random() < 0.5 else "zoom-0.0025"
                    vf_parts.append(
                        f"zoompan=z='{zoom_dir}':d={kb_frames}:s={w}x{h}:fps={fps_expr}"
                    )

                # Segment-level FX (per-segment visual styles)
                if seg.effect == "zoom":
                    # Gentle punch-in (about 8%)
                    vf_parts.append("scale=iw*1.08:ih*1.08,crop=iw/1.08:ih/1.08")
                elif seg.effect == "flash":
                    # Small global brightness lift for the whole segment
                    vf_parts.append("eq=brightness=0.18")
                elif seg.effect == "rgb_split":
                    # Chromatic aberration / RGB split
                    vf_parts.append("chromashift=u=2:v=-2")
                elif seg.effect == "vhs":
                    # VHS noise + scanlines (downscale + upscale with neighbor sampling)
                    vf_parts.append("noise=alls=15:allf=t+u,scale=iw:ih/2:flags=neighbor,scale=iw:ih:flags=neighbor")
                elif seg.effect == "motion_blur":
                    # Mild motion blur via temporal blend
                    vf_parts.append("tblend=all_mode=average,framestep=1")

                cine_start = len(vf_parts)
                # Cinematic one-off effects                # Cinematic one-off effects
                if getattr(seg, "cine_freeze", False):
                    # Shutter-pop: quick punch + glossy smear (no speed change).
                    # (This replaces the old freeze-frame effect; it never pauses or slows time.)

                    try:
                        punch = float(getattr(seg, "cine_freeze_zoom", 0.0) or 0.0)
                    except Exception:
                        punch = 0.0

                    # Map the old "zoom %" slider (stored as 0.0–0.5) into a 0.0–1.0 intensity.
                    intensity = max(0.0, min(1.0, punch / 0.30 if 0.30 > 0 else punch))

                    try:
                        hit_d = float(getattr(seg, "cine_freeze_len", 0.0) or 0.0)
                    except Exception:
                        hit_d = 0.0
                    if hit_d <= 0.0:
                        hit_d = 0.35
                    hit_d = max(0.08, min(1.0, hit_d))

                    # Smooth envelope: 0 -> 1 -> 0 over the hit window.
                    env_expr = f"if(lt(t,{hit_d:.3f}),sin(PI*t/{hit_d:.3f}),0)"
                    z_amp = max(0.02, min(0.18, 0.02 + 0.16 * intensity))
                    z_expr = f"(1+{z_amp:.4f}*({env_expr}))"

                    try:
                        dir_sign = int(getattr(seg, "cine_freeze_dir", 1) or 1)
                    except Exception:
                        dir_sign = 1
                    dir_sign = -1 if dir_sign < 0 else 1

                    # Pixel-level chroma offset.
                    px = int(round(2 + 8 * intensity)) * dir_sign
                    px = max(-18, min(18, px))

                    alpha = max(0.35, min(0.88, 0.48 + 0.40 * intensity))
                    con = 1.02 + 0.16 * intensity
                    sat = 1.00 + 0.10 * intensity
                    sharp = 0.25 + 0.70 * intensity

                    vf_parts.append(
                        "split=2[psrc][pbase];"
                        f"[psrc]tmix=frames=5:weights='1 0.90 0.72 0.52 0.35',"
                        f"rgbashift=rh={px}:bh={-px}:edge=smear,"
                        f"eq=contrast={con:.3f}:saturation={sat:.3f},"
                        f"scale=w='iw*({z_expr})':h='ih*({z_expr})':eval=frame,"
                        f"crop=w='iw/({z_expr})':h='ih/({z_expr})',"
                        f"unsharp=5:5:{sharp:.3f}:5:5:0,"
                        f"format=rgba,colorchannelmixer=aa={alpha:.3f}[pfx];"
                        f"[pbase][pfx]overlay=shortest=1:enable='lt(t,{hit_d:.3f})',format=yuv420p"
                    )

                if getattr(seg, "cine_stutter", False):

                    # Stutter / triple-frame style slice: drop interim frames so motion
                    # feels jittery. We keep duration the same, only the cadence changes.
                    repeats = int(getattr(seg, "cine_stutter_repeats", 3) or 3)
                    # For now we map repeats -> a simple framestep factor in a safe range.
                    step = max(2, min(5, repeats + 1))
                    vf_parts.append(f"framestep={step}")

                if getattr(seg, "cine_tear_v", False):
                    # Prism whip (horizontal drift): a short chroma-smeared whip + micro-zoom.
                    # (Replaces the old 3-slice "screen tear" which looked harsh.)
                    try:
                        s = float(getattr(seg, "cine_tear_v_strength", 0.7) or 0.7)
                    except Exception:
                        s = 0.7
                    s = max(0.1, min(1.0, s))

                    # Direction picked per event in _apply_cinematic_effects
                    try:
                        dir_sign = int(getattr(seg, "cine_tear_v_dir", 1) or 1)
                    except Exception:
                        dir_sign = 1
                    dir_sign = -1 if dir_sign < 0 else 1

                    # Short hit window (seconds)
                    hit_d = 0.22 + 0.28 * s

                    # Envelope (0 -> 1 -> 0) for the hit.
                    env_expr = f"if(lt(t,{hit_d:.3f}),sin(PI*t/{hit_d:.3f}),0)"

                    # Micro zoom + drift
                    z_amp = 0.030 + 0.070 * s
                    z_expr = f"(1+{z_amp:.4f}*({env_expr}))"
                    drift_x = (0.012 + 0.030 * s) * dir_sign
                    drift_y = (0.004 + 0.010 * s) * (-dir_sign)

                    x_expr = f"iw*{drift_x:.5f}*({env_expr})"
                    y_expr = f"ih*{drift_y:.5f}*({env_expr})"

                    # Chromatic offset (pixels)
                    try:
                        px = int(round(2 + 7 * s)) * dir_sign
                    except Exception:
                        px = 5 * dir_sign
                    px = max(-18, min(18, px))

                    alpha = max(0.35, min(0.95, 0.55 + 0.35 * s))

                    vf_parts.append(
                        "split=2[tvsrc][tvbase];"
                        f"[tvsrc]tmix=frames=3:weights='1 2 1',"
                        f"rgbashift=rh={px}:bh={-px}:edge=smear,"
                        f"eq=contrast={1.00 + 0.18*s:.3f}:saturation={1.00 + 0.10*s:.3f},"
                        f"scale=w='iw*({z_expr})':h='ih*({z_expr})':eval=frame,"
                        f"crop=w='iw/({z_expr})':h='ih/({z_expr})':x='{x_expr}':y='{y_expr}',"
                        f"unsharp=5:5:{0.35 + 0.55*s:.3f}:5:5:0,"
                        f"format=rgba,colorchannelmixer=aa={alpha:.3f}[tvfx];"
                        f"[tvbase][tvfx]overlay=shortest=1:enable='lt(t,{hit_d:.3f})',format=yuv420p"
                    )


                if getattr(seg, "cine_tear_h", False):
                    # Prism whip (vertical drift): same as above but drifting vertically.
                    try:
                        s = float(getattr(seg, "cine_tear_h_strength", 0.7) or 0.7)
                    except Exception:
                        s = 0.7
                    s = max(0.1, min(1.0, s))

                    # Direction picked per event in _apply_cinematic_effects
                    try:
                        dir_sign = int(getattr(seg, "cine_tear_h_dir", 1) or 1)
                    except Exception:
                        dir_sign = 1
                    dir_sign = -1 if dir_sign < 0 else 1

                    hit_d = 0.18 + 0.26 * s
                    env_expr = f"if(lt(t,{hit_d:.3f}),sin(PI*t/{hit_d:.3f}),0)"

                    z_amp = 0.028 + 0.060 * s
                    z_expr = f"(1+{z_amp:.4f}*({env_expr}))"

                    drift_x = (0.004 + 0.012 * s) * (dir_sign)
                    drift_y = (0.014 + 0.034 * s) * (dir_sign)

                    x_expr = f"iw*{drift_x:.5f}*({env_expr})"
                    y_expr = f"ih*{drift_y:.5f}*({env_expr})"

                    try:
                        py = int(round(2 + 7 * s)) * dir_sign
                    except Exception:
                        py = 5 * dir_sign
                    py = max(-18, min(18, py))

                    alpha = max(0.35, min(0.95, 0.55 + 0.35 * s))

                    vf_parts.append(
                        "split=2[thsrc][thbase];"
                        f"[thsrc]tmix=frames=3:weights='1 2 1',"
                        f"rgbashift=rv={py}:bv={-py}:edge=smear,"
                        f"eq=contrast={1.00 + 0.18*s:.3f}:saturation={1.00 + 0.10*s:.3f},"
                        f"scale=w='iw*({z_expr})':h='ih*({z_expr})':eval=frame,"
                        f"crop=w='iw/({z_expr})':h='ih/({z_expr})':x='{x_expr}':y='{y_expr}',"
                        f"unsharp=5:5:{0.35 + 0.55*s:.3f}:5:5:0,"
                        f"format=rgba,colorchannelmixer=aa={alpha:.3f}[thfx];"
                        f"[thbase][thfx]overlay=shortest=1:enable='lt(t,{hit_d:.3f})',format=yuv420p"
                    )

                if getattr(seg, "cine_color_cycle", False):
                    # Color-cycle glitch (cinematic): full-spectrum hue rotation.
                    # Speed is interpreted as milliseconds per full 360° cycle.
                    try:
                        ms = int(getattr(seg, "cine_color_cycle_speed_ms", 400) or 400)
                    except Exception:
                        ms = 400
                    ms = max(100, min(1000, ms))
                    # Convert ms-per-cycle into degrees-per-second.
                    rate = 360000.0 / float(ms)
                    # Use mod() to keep hue stable on all ffmpeg builds.
                    vf_parts.append(f"hue=h='mod(t*{rate:.3f},360)':s=1.35")
                    vf_parts.append("eq=contrast=1.06:saturation=1.18")


                if getattr(seg, "cine_reverse", False):
                    # Reverse-bounce: optionally repeat a short reversed window
                    # to fill long segments, using a finite loop for safety.
                    try:
                        seg_dur = float(getattr(seg, "duration", 0.0) or 0.0)
                    except Exception:
                        seg_dur = 0.0
                    try:
                        win = float(getattr(seg, "cine_reverse_window", 0.0) or 0.0)
                    except Exception:
                        win = 0.0
                    if win <= 0.0:
                        try:
                            win = float(getattr(seg, "cine_reverse_len", 0.0) or 0.0)
                        except Exception:
                            win = 0.0
                    if win <= 0.0:
                        win = min(0.5, seg_dur) if seg_dur > 0 else 0.5
                    if seg_dur > 0.0:
                        win = min(win, seg_dur)

                    # If the window is shorter than the segment, loop the reversed slice.
                    if seg_dur > 0.0 and win > 0.0 and seg_dur > (win + 0.01):
                        fps_guess = 30.0
                        try:
                            frames = max(1, int(round(win * fps_guess)))
                        except Exception:
                            frames = 1
                        try:
                            needed = int(math.ceil(seg_dur / win))
                        except Exception:
                            needed = 1
                        loops = max(0, needed - 1)
                        # Safety cap to avoid runaway memory usage.
                        loops = min(50, loops)
                        vf_parts.append(
                            f"trim=duration={win:.3f},reverse,loop=loop={loops}:size={frames}:start=0,trim=duration={seg_dur:.3f}"
                        )
                    else:
                        # Simple full-segment reverse.
                        vf_parts.append("reverse")

                if getattr(seg, "cine_pan916", False):
                    # N-slice window reveal on a black (or dim) canvas:
                    # show only one slice of the frame at a time, stepping
                    # left → ... → right → ... → left (boomerang).
                    # Uses an FFmpeg expression so it naturally repeats for long segments.
                    try:
                        ms = int(getattr(seg, "cine_pan916_speed_ms", 400) or 400)
                    except Exception:
                        ms = 400
                    ms = max(200, min(1000, ms))
                    ms = int(round(ms / 50.0) * 50)
                    step_s = ms / 1000.0

                    try:
                        slice_parts = int(getattr(seg, "cine_pan916_parts", 3) or 3)
                    except Exception:
                        slice_parts = 3
                    slice_parts = max(2, min(6, slice_parts))

                    # Boomerang period: 0..slice_parts-1..1 (repeat).
                    period = max(2, (2 * slice_parts - 2))

                    # Optional per-segment randomization (assigned in build_timeline when enabled).
                    try:
                        phase = int(getattr(seg, "cine_pan916_phase", 0) or 0)
                    except Exception:
                        phase = 0
                    try:
                        flip_lr = bool(getattr(seg, "cine_pan916_flip_lr", False))
                    except Exception:
                        flip_lr = False
                    if period > 0:
                        phase = phase % period
                    else:
                        phase = 0

                    k_expr = f"mod(floor(t/{step_s:.3f})+{phase},{period})"
                    pos_base = f"if(lte({k_expr},{slice_parts-1}),{k_expr},{period}-{k_expr})"
                    pos_expr = f"({slice_parts-1})-({pos_base})" if flip_lr else pos_base

                    # Build a full-resolution output by compositing the cropped window
                    # over a black background (or dimmed video when transparent is enabled),
                    # so it works perfectly for 16:9 output too.
                    transparent_bg = bool(getattr(seg, "cine_pan916_transparent", False))
                    if transparent_bg:
                        bg_chain = "[base]lutrgb=r=val*0.5:g=val*0.5:b=val*0.5[bg];"
                    else:
                        bg_chain = "[base]drawbox=x=0:y=0:w=iw:h=ih:color=black@1.0:t=fill[bg];"

                    vf_parts.append(
                        "split=2[src][base];"
                        + bg_chain
                        + f"[src]crop=w=trunc(iw/{slice_parts}/2)*2:h=ih:x='({pos_expr})*iw/{slice_parts}':y=0[win];"
                        + f"[bg][win]overlay=x='({pos_expr})*W/{slice_parts}':y=0:shortest=1"
                    )

                if getattr(seg, "cine_dimension", False):

                    # Dimension portal: render the segment inside a randomly chosen
                    # portal shape (rect / trapezoid / diamond / portrait 9:16).
                    # Lightweight geometry-only effect (no PNG masks).
                    try:
                        kind = str(getattr(seg, "cine_dimension_kind", "rect") or "rect").strip().lower()
                    except Exception:
                        kind = "rect"

                    # Background: keep full frame but dim/desaturate slightly.
                    bg_chain = "eq=brightness=-0.12:saturation=0.80"

                    # Foreground: shrink into a portal window.
                    fg_base = "scale=trunc(iw*0.70/2)*2:trunc(ih*0.70/2)*2:flags=bicubic"
                    fg_punch = "eq=contrast=1.08:saturation=1.10"

                    if kind in ("portrait", "9:16", "9x16", "916"):
                        # Center crop to 9:16 first, then shrink.
                        fg_chain = (
                            "crop=w=trunc(min(iw,ih*9/16)/2)*2:h=ih:x=(iw-ow)/2:y=0,"
                            + fg_base
                            + ","
                            + fg_punch
                        )
                    elif kind in ("trapezoid", "trapezium"):
                        fg_chain = (
                            fg_base
                            + ",perspective=sense=destination:"
                              "x0=W*0.22:y0=H*0.10:x1=W*0.78:y1=H*0.10:"
                              "x2=W*0.05:y2=H*0.95:x3=W*0.95:y3=H*0.95:eval=init,"
                            + fg_punch
                        )
                    elif kind in ("diamond", "rhombus", "ruit"):
                        fg_chain = (
                            fg_base
                            + ",perspective=sense=destination:"
                              "x0=W*0.50:y0=H*0.02:x1=W*0.98:y1=H*0.50:"
                              "x2=W*0.02:y2=H*0.50:x3=W*0.50:y3=H*0.98:eval=init,"
                            + fg_punch
                        )
                    else:
                        fg_chain = fg_base + "," + fg_punch

                    # Faint border so the portal reads clearly.
                    fg_chain = fg_chain + ",drawbox=x=0:y=0:w=iw:h=ih:color=white@0.20:t=4"

                    vf_parts.append(
                        "split=2[bg][fg];"
                        f"[bg]{bg_chain}[bg0];"
                        f"[fg]{fg_chain}[fg0];"
                        "[bg0][fg0]overlay=(W-w)/2:(H-h)/2"
                    )
                if getattr(seg, "cine_flip", False):
                    # Flip the frame 180° (upside-down) by combining vertical and horizontal flips.
                    vf_parts.append("vflip,hflip")
                if getattr(seg, "cine_rotate", False):
                    # Apply a short rotating-screen pulse: 0° → max → 0° over the segment.
                    try:
                        max_deg = float(getattr(seg, "cine_rotate_degrees", 20.0) or 20.0)
                    except Exception:
                        max_deg = 20.0
                    max_deg = max(5.0, min(90.0, max_deg))
                    try:
                        seg_dur = float(getattr(seg, "duration", 0.0) or 0.0)
                    except Exception:
                        seg_dur = 0.0
                    if seg_dur <= 0.1:
                        seg_dur = 0.1
                    max_rad = math.radians(max_deg)
                    # Use ffmpeg expression: angle(t) = max_rad * sin(PI * t / seg_dur)
                    # This yields a single pulse: 0 → +max → 0 over the lifetime of the segment.
                    vf_parts.append(f"rotate={max_rad:.6f}*sin(PI*t/{seg_dur:.3f}):c=black")
                if getattr(seg, "cine_speed_ramp", False) and not getattr(seg, "is_image", False):
                    # Use the configured ramp-in / ramp-out times to drive how strong
                    # the motion blend should be. Longer ramps -> stronger smoothing.
                    try:
                        base_len = float(getattr(seg, "duration", 0.0) or 0.0)
                    except Exception:
                        base_len = 0.0
                    try:
                        ramp_in = float(getattr(seg, "cine_ramp_in", 0.0) or 0.0)
                    except Exception:
                        ramp_in = 0.0
                    try:
                        ramp_out = float(getattr(seg, "cine_ramp_out", 0.0) or 0.0)
                    except Exception:
                        ramp_out = 0.0

                    ramp_span = max(0.05, ramp_in + ramp_out)
                    if base_len > 0.0:
                        frac = max(0.1, min(1.0, ramp_span / base_len))
                    else:
                        frac = 0.5

                    # Map the fraction of the segment covered by ramps into
                    # 1–3 passes of tblend: short ramps = subtle, long ramps = strong.
                    if frac > 0.6:
                        passes = 3
                    elif frac > 0.3:
                        passes = 2
                    else:
                        passes = 1

                    ramp_filters = ",".join(["tblend=all_mode=average,framestep=1"] * passes)
                    vf_parts.append(ramp_filters)

                # Camera motion hits: dolly-zoom / Ken Burns pan/zoom
                if getattr(seg, "cine_dolly", False) or getattr(seg, "cine_kenburns", False):
                    # Use whichever strength is active; both are in the 0.10–1.0 range.
                    if getattr(seg, "cine_dolly", False):
                        base = float(getattr(seg, "cine_dolly_strength", 0.4) or 0.4)
                    else:
                        base = float(getattr(seg, "cine_kenburns_strength", 0.35) or 0.35)
                    # Map to a subtle zoom amount between ~1.02x and ~1.35x.
                    zoom_amount = 1.0 + max(0.02, min(0.35, base * 0.35))
                    motion_dir = int(getattr(seg, "cine_motion_dir", 0) or 0)

                    if motion_dir in (1, 2):
                        # Pure zoom in/out.
                        if motion_dir == 1:
                            z_expr = zoom_amount
                        else:
                            z_expr = 1.0 / zoom_amount
                        vf_parts.append(
                            f"scale=iw*{z_expr:.3f}:ih*{z_expr:.3f},crop=iw/{z_expr:.3f}:ih/{z_expr:.3f}"
                        )
                    else:
                        # Pan across the clip with a slight zoom (Ken Burns style).
                        pan_zoom = zoom_amount
                        inv = 1.0 / pan_zoom
                        if motion_dir == 3:   # pan left
                            x_expr = "0"
                            y_expr = f"(ih-ih/{pan_zoom:.3f})/2"
                        elif motion_dir == 4: # pan right
                            x_expr = f"(iw-iw/{pan_zoom:.3f})"
                            y_expr = f"(ih-ih/{pan_zoom:.3f})/2"
                        elif motion_dir == 5: # pan up
                            x_expr = f"(iw-iw/{pan_zoom:.3f})/2"
                            y_expr = "0"
                        elif motion_dir == 6: # pan down
                            x_expr = f"(iw-iw/{pan_zoom:.3f})/2"
                            y_expr = f"(ih-ih/{pan_zoom:.3f})"
                        else:
                            # Fallback: centred slight zoom.
                            x_expr = f"(iw-iw/{pan_zoom:.3f})/2"
                            y_expr = f"(ih-ih/{pan_zoom:.3f})/2"
                        vf_parts.append(
                            f"scale=iw*{pan_zoom:.3f}:ih*{pan_zoom:.3f},"
                            f"crop=iw/{pan_zoom:.3f}:ih/{pan_zoom:.3f}:{x_expr}:{y_expr}"
                        )

                cine_end = len(vf_parts)

                impact_start = len(vf_parts)

                # Break impact FX (first beat after a break)
                if getattr(seg, "is_break_impact", False):
                    # Flash strobe: short white flashes at the very start (club strobe style).

                    # - Strength slider controls how hard it hits.

                    # - Speed slider controls cadence (ms per flash period).

                    flash_s = float(getattr(seg, "impact_flash_strength", 0.0) or 0.0)

                    if flash_s > 0.0:

                        s = max(0.1, min(1.0, flash_s))


                        # Speed: ms per flash period (100–1000ms).

                        try:

                            speed_ms = int(getattr(seg, "impact_flash_speed_ms", 250) or 250)

                        except Exception:

                            speed_ms = 250

                        speed_ms = max(100, min(1000, speed_ms))

                        period_s = speed_ms / 1000.0


                        # Flash length: keep it short even at slow speeds.

                        pulse_s = max(0.02, min(0.08, period_s * 0.35))


                        # Time window: only strobe near the start of the segment so it feels like an impact hit.

                        try:

                            seg_len = float(getattr(seg, "duration", 0.0) or 0.0)

                        except Exception:

                            seg_len = 0.0

                        max_window = min(0.9, seg_len if seg_len > 0.0 else 0.9)


                        # Strength -> alpha (opacity) of the white flash overlay.

                        flash_a = 0.30 + 0.55 * s  # 0.30–0.85


                        # Expression-driven strobe (single filter) to ensure it actually flashes.

                        vf_parts.append(

                            "drawbox=x=0:y=0:w=iw:h=ih:"

                            f"color=white@{flash_a:.2f}:t=fill:"

                            f"enable='between(t,0,{max_window:.3f})*lt(mod(t,{period_s:.3f}),{pulse_s:.3f})'"

                        )


                    # Shockwave burst: a brief zoom pulse with mild contrast lift.
                    shock_s = float(getattr(seg, "impact_shock_strength", 0.0) or 0.0)
                    if shock_s > 0.0:
                        zoom_amt = 0.05 + 0.20 * max(0.0, min(1.0, shock_s))
                        z_expr = 1.0 + zoom_amt
                        vf_parts.append(
                            f"scale=iw*{z_expr:.3f}:ih*{z_expr:.3f},crop=iw/{z_expr:.3f}:ih/{z_expr:.3f}"
                        )

                    # Beat ripple: radial shockwave distortion (6–12 frames, peak around frames 3–4 @ 30fps).
                    ripple_s = float(getattr(seg, "impact_confetti_density", 0.0) or 0.0)
                    if ripple_s > 0.0:
                        s = max(0.0, min(1.0, ripple_s))
                        # Duration: 0.20–0.40s (~6–12 frames @ 30fps)
                        ripple_d = 0.20 + 0.20 * s
                        t1 = ripple_d * 0.33
                        t2 = ripple_d * 0.55

                        # Lens warp pulse (barrel distortion). Strongest in the middle (frame ~3–4).
                        k1a = -0.12 * s
                        k1b = -0.32 * s
                        k1c = -0.18 * s
                        vf_parts.append(
                            f"lenscorrection=k1={k1a:.3f}:k2=0:enable='between(t,0,{t1:.3f})'"
                        )
                        vf_parts.append(
                            f"lenscorrection=k1={k1b:.3f}:k2=0:enable='between(t,{t1:.3f},{t2:.3f})'"
                        )
                        vf_parts.append(
                            f"lenscorrection=k1={k1c:.3f}:k2=0:enable='between(t,{t2:.3f},{ripple_d:.3f})'"
                        )

                        # Subtle chroma split during the ripple (no full-frame flash).
                        px = max(1, min(5, int(round(1 + 4 * s))))
                        vf_parts.append(
                            f"rgbashift=rh={px}:bh=-{px}:edge=smear:enable='between(t,0,{ripple_d:.3f})'"
                        )

                    # Time Echo Trail: blend a short trail of previous frames on impact.
                    echo_s = float(getattr(seg, "impact_echo_trail_strength", 0.0) or 0.0)
                    if echo_s > 0.0:
                        echo_d = 0.60
                        mid = 0.25  # heavy hit then wane out

                        # Heavy punch at the start
                        vf_parts.append(
                            f"tmix=frames=5:weights='1 0.90 0.75 0.55 0.35':enable='between(t,0,{mid:.3f})'"
                        )

                        # Lighter tail
                        vf_parts.append(
                            f"tmix=frames=5:weights='1 0.60 0.35 0.20 0.10':enable='between(t,{mid:.3f},{echo_d:.3f})'"
                            
                        )
                        vf_parts.append(
                            f"eq=brightness=0.08:enable='between(t,0,{echo_d:.3f})'"
                        )


                    # Zoom punch-in: segment-wide punch for extra impact.
                    zoom_s = float(getattr(seg, "impact_zoom_amount", 0.0) or 0.0)
                    if zoom_s > 0.0:
                        zoom_amt = max(0.02, min(0.40, zoom_s))
                        z_expr = 1.0 + zoom_amt
                        vf_parts.append(
                            f"scale=iw*{z_expr:.3f}:ih*{z_expr:.3f},crop=iw/{z_expr:.3f}:ih/{z_expr:.3f}"
                        )

                    # Camera shake: tiny rotation jitter.
                    shake_s = float(getattr(seg, "impact_shake_strength", 0.0) or 0.0)
                    if shake_s > 0.0:
                        # Amplitude scaled by strength; frequency is fixed so it stays readable.
                        amp = 0.02 * max(0.2, min(1.0, shake_s))
                        vf_parts.append(
                            f"rotate={amp:.3f}*sin(40*PI*t):bilinear=0"
                        )

                    # Fog blast: a soft haze using brightness + a touch of noise.
                    fog_s = float(getattr(seg, "impact_fog_density", 0.0) or 0.0)
                    if fog_s > 0.0:
                        fog_b = 0.15 * max(0.2, min(1.0, fog_s))
                        fog_d = max(0.10, min(0.50, seg.duration * 0.5))
                        vf_parts.append(
                            f"eq=brightness={fog_b:.2f}:enable='between(t,0,{fog_d:.3f})'"
                        )

                    # Color-cycle glitch (segment-wide): continuous hue cycling for the whole impact segment.
                    cycle_s = float(getattr(seg, 'impact_color_cycle_speed', 0.0) or 0.0)
                    if cycle_s > 0.0:
                        # Map 0.1–2.0 to a readable hue speed (degrees/sec).
                        rate = max(120.0, min(1200.0, 540.0 * cycle_s))
                        vf_parts.append(
                            f"hue=h='mod(t*{rate:.1f},360)':s=1.35"
                        )
                        vf_parts.append(
                            "eq=contrast=1.06:saturation=1.18"
                        )

                # Timed strobe: user-specified timestamps (within this segment).
                strobe_offsets = getattr(seg, "strobe_on_time_offsets", None) if not getattr(self, "strobe_on_time_times", None) else None
                if strobe_offsets:
                    try:
                        flash_s = float(getattr(seg, "impact_flash_strength", 0.0) or 0.0)
                    except Exception:
                        flash_s = 0.0
                    if flash_s > 0.0:
                        s = max(0.1, min(1.0, flash_s))
                        try:
                            speed_ms = int(getattr(seg, "impact_flash_speed_ms", 250) or 250)
                        except Exception:
                            speed_ms = 250
                        speed_ms = max(100, min(1000, speed_ms))
                        period_s = speed_ms / 1000.0
                        pulse_s = max(0.02, min(0.08, period_s * 0.35))

                        # Use output-time segment length when available, so offsets stay correct even with slow motion.
                        try:
                            st = float(getattr(seg, "timeline_start", 0.0) or 0.0)
                            en = float(getattr(seg, "timeline_end", st) or st)
                            seg_len = max(0.0, en - st)
                        except Exception:
                            seg_len = 0.0
                        if seg_len <= 0.0:
                            try:
                                seg_len = float(getattr(seg, "duration", 0.0) or 0.0)
                            except Exception:
                                seg_len = 0.0
                        seg_len = max(0.0, seg_len)

                        flash_a = 0.30 + 0.55 * s  # 0.30–0.85

                        # Add one drawbox strobe per timestamp (offset).
                        try:
                            offs = sorted(set([float(x) for x in strobe_offsets]))
                        except Exception:
                            offs = []
                        for off in offs:
                            try:
                                off_f = float(off)
                            except Exception:
                                continue
                            if off_f < 0.0:
                                continue
                            if seg_len > 0.0:
                                off_f = max(0.0, min(seg_len, off_f))
                            # Strobe window: up to 0.9s starting from the offset (clamped to segment).
                            max_window = 0.9
                            if seg_len > 0.0:
                                max_window = min(0.9, max(0.02, seg_len - off_f))
                            end_t = off_f + max_window
                            if seg_len > 0.0:
                                end_t = min(seg_len, end_t)

                            vf_parts.append(
                                "drawbox=x=0:y=0:w=iw:h=ih:"
                                f"color=white@{flash_a:.2f}:t=fill:"
                                f"enable='between(t,{off_f:.3f},{end_t:.3f})*lt(mod(t-{off_f:.3f},{period_s:.3f}),{pulse_s:.3f})'"
                            )


                impact_end = len(vf_parts)

                # Transitions using a small effect library (all short and fade-safe)
                if seg.transition == "flashcut":
                    # White flash: short brightness pop at the start (~30–60ms).
                    flash_d = max(0.06, min(0.06, seg.duration / 6.0))
                    vf_parts.append(
                        f"eq=brightness=0.45:enable='between(t,0,{flash_d:.3f})'"
                    )
                elif seg.transition == "flashcolor":
                    # Colored flash: short hue-shifted flash at the start.
                    flash_d   = max(0.4, min(1.2, seg.duration * 0.8))
                    # Rotate hue by a fixed offset per segment index for reproducible variety.
                    hue_shift = (i * 37) % 360  # pseudo "16-ish" cycle
                    vf_parts.append(
                        f"eq=brightness=0.35:enable='between(t,0,{flash_d:.3f})',"
                        f"hue=h={hue_shift}:enable='between(t,0,{flash_d:.3f})'"
                    )
                elif seg.transition == "whip":
                    # Zoom pulse / scale pulse: beat-style zoom using scale+crop.
                    # This is a visual effect only (no fades), safe across ffmpeg builds.
                    # Use a fixed gentle pulse period (~0.5s) since we don't have beat_period here.
                    base_period = 0.3
                    zoom_amount = 0.02
                    zoom_expr = f"1+{zoom_amount}*abs(sin(2*3.14159*t/{base_period:.3f}))"
                    vf_parts.append(
                        f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,crop=iw:ih"
                    )
                elif seg.transition == "t_zoom_pulse":
                    # Legacy zoom pulse / scale pulse from the old Auto Music Sync tool
                    base_period = 0.5
                    zoom_amount = 0.03
                    zoom_expr = f"1+{zoom_amount}*abs(sin(2*3.14159*t/{base_period:.3f}))"
                    vf_parts.append(
                        f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,crop=iw:ih"
                    )
                
                elif seg.transition == "t_color_cycle":
                    # Rapid hue cycling so the transition "changes colors" (not just a single tint shift).
                    try:
                        seg_len = float(getattr(seg, "duration", 0.0) or 0.0)
                    except Exception:
                        seg_len = 0.0
                    # Keep it short so it reads as a transition, not a permanent recolor.
                    d = 0.60
                    if seg_len > 0.0:
                        d = max(0.25, min(0.85, seg_len * 0.60))
                    # Two-ish full hue spins per second gives a clear multi-color sweep.
                    vf_parts.append(
                        f"hue=h='mod(t*540,360)':s=1.25:enable='between(t,0,{d:.3f})'"
                    )
                    vf_parts.append(
                        f"eq=contrast=1.05:saturation=1.12:enable='between(t,0,{d:.3f})'"
                    )
                    # Tiny time-mix at the start to add a subtle glitch smear.
                    smear = min(d, 0.30)
                    vf_parts.append(
                        f"tmix=frames=3:weights='1 0.75 0.45':enable='between(t,0,{smear:.3f})'"
                    )
                elif seg.transition == "t_rgb_split":
                    # Legacy RGB split / chromatic aberration transition
                    vf_parts.append("chromashift=u=2:v=-2")
                elif seg.transition == "t_vhs":
                    # Legacy VHS noise + scanlines transition
                    vf_parts.append("noise=alls=15:allf=t+u,scale=iw:ih/2:flags=neighbor,scale=iw:ih:flags=neighbor")
                elif seg.transition == "t_motion_blur":
                    # Legacy motion blur boost transition
                    vf_parts.append("tblend=all_mode=average,framestep=1")
                elif seg.transition in ("t_push", "t_slitscan_push"):
                    # Directional push / slit-scan smear push are real between-clip transitions.
                    # It is applied later during stitching, so we do nothing here.
                    pass
                elif seg.transition in ("t_luma_fade", "t_exposure_dissolve", "t_scale_punch", "t_wipe", "t_curtain_open", "t_pixelize", "t_distance", "t_wind_smears"):
                    # Wipe / exposure dissolve are real between-clip transitions.
                    # It is applied later during the stitching step so we do NOT fade to black here.
                    pass
                elif seg.transition == "t_smooth_zoom":
                    # Smooth zoom crossfade: continuous zoom over the whole clip.
                    # Zoom amount is unchanged; we only randomize the drift direction (incl diagonals).
                    total = max(0.20, float(seg.duration or 1.0))
                    zoom_expr = "1+0.15*t/{:.3f}".format(total)

                    if self.target_resolution is None:
                        # No known output size => keep the legacy centered zoom.
                        vf_parts.append(
                            f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,crop=iw:ih"
                        )
                    else:
                        # Shuffle-bag so directions rotate instead of feeling repetitive.
                        bag = getattr(self, "_smooth_zoom_dir_bag", None)
                        if not bag:
                            bag = ["left","right","up","down","upleft","upright","downleft","downright"]
                            random.shuffle(bag)
                            self._smooth_zoom_dir_bag = bag
                        dsel = bag.pop()
                        if not bag:
                            # Refill for next time.
                            bag[:] = ["left","right","up","down","upleft","upright","downleft","downright"]
                            random.shuffle(bag)

                        # Progress for crop drift: crop exprs don't reliably expose 't' on all ffmpeg builds,
                        # so we use frame index 'n' (segments are rendered at 30fps).
                        frames = max(1, int(round(total * 30.0)))
                        p_expr = f"min(1,n/{frames})"

                        dx = f"(iw-{w})"
                        dy = f"(ih-{h})"
                        cx = f"({dx})/2"
                        cy = f"({dy})/2"

                        if dsel == "left":
                            ex, ey = "0", cy
                        elif dsel == "right":
                            ex, ey = dx, cy
                        elif dsel == "up":
                            ex, ey = cx, "0"
                        elif dsel == "down":
                            ex, ey = cx, dy
                        elif dsel == "upleft":
                            ex, ey = "0", "0"
                        elif dsel == "upright":
                            ex, ey = dx, "0"
                        elif dsel == "downleft":
                            ex, ey = "0", dy
                        else:  # downright
                            ex, ey = dx, dy

                        x_raw = f"({cx}) + (({ex})-({cx}))*({p_expr})"
                        y_raw = f"({cy}) + (({ey})-({cy}))*({p_expr})"
                        x_expr = f"max(0,min({dx},{x_raw}))"
                        y_expr = f"max(0,min({dy},{y_raw}))"

                        vf_parts.append(
                            f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,"
                            f"crop={w}:{h}:x={x_expr}:y={y_expr}"
                        )
                else:
                    # "none" and anything else fall back to a hard cut (no extra filters here)
                    pass

                # Apply slow-motion filter if requested for this segment
                if getattr(seg, "slow_factor", 1.0) != 1.0:
                    try:
                        sf = float(seg.slow_factor)
                    except Exception:
                        sf = 1.0
                    if sf != 1.0:
                        vf_parts.append(f"setpts=PTS/{sf}")
                        if base_vf_parts:
                            base_vf_parts = base_vf_parts + [f"setpts=PTS/{sf}"]

                def _vf_join(parts):
                    if not parts:
                        return None
                    p = list(parts)
                    # Make NVENC happy / keep stitching stable.
                    # Only add a final format if the chain doesn't already end in yuv420p.
                    tail = ",".join(p[-3:]) if len(p) >= 3 else ",".join(p)
                    if "format=yuv420p" not in tail:
                        p.append("format=yuv420p")
                    return ",".join(p)

                vf_arg = _vf_join(vf_parts)
                safe_vf_arg = _vf_join(base_vf_parts)

                vf_no_cine_arg = None
                vf_no_impact_arg = None
                vf_no_cine_no_impact_arg = None
                try:
                    if "cine_start" in locals() and "cine_end" in locals():
                        if isinstance(cine_start, int) and isinstance(cine_end, int) and cine_end > cine_start:
                            parts_no_cine = vf_parts[:cine_start] + vf_parts[cine_end:]
                            vf_no_cine_arg = _vf_join(parts_no_cine)
                except Exception:
                    vf_no_cine_arg = None
                try:
                    if "impact_start" in locals() and "impact_end" in locals():
                        if isinstance(impact_start, int) and isinstance(impact_end, int) and impact_end > impact_start:
                            parts_no_impact = vf_parts[:impact_start] + vf_parts[impact_end:]
                            vf_no_impact_arg = _vf_join(parts_no_impact)

                            if "cine_start" in locals() and "cine_end" in locals():
                                if isinstance(cine_start, int) and isinstance(cine_end, int) and cine_end >= cine_start:
                                    # Impact section comes after cinematic section in this filter build.
                                    a = vf_parts[:cine_start]
                                    b = vf_parts[cine_end:impact_start]
                                    c = vf_parts[impact_end:]
                                    vf_no_cine_no_impact_arg = _vf_join(a + b + c)
                except Exception:
                    vf_no_impact_arg = None
                    vf_no_cine_no_impact_arg = None


                # Base ffmpeg command for this segment (input, trim, no audio)
                # Special-case: cinematic "boomerang" effect (short forward-back loop).
                if getattr(seg, "cine_boomerang", False) and not getattr(seg, "is_image", False):
                    # Use a short source window for the forward/reverse bounce,
                    # but keep the musical segment length as the target output duration.
                    try:
                        target_len = float(getattr(seg, "duration", 0.0) or 0.0)
                    except Exception:
                        target_len = 0.0
                    if target_len <= 0.0:
                        target_len = 0.2

                    try:
                        boom_window = float(getattr(seg, "cine_boomerang_window", 0.0) or 0.0)
                    except Exception:
                        boom_window = 0.0
                    if boom_window <= 0.0:
                        boom_window = min(0.25, target_len)
                        if boom_window <= 0.0:
                            boom_window = 0.2

                    try:
                        start = float(getattr(seg, "clip_start", 0.0) or 0.0)
                    except Exception:
                        start = 0.0

                    input_args = [
                        "-ss",
                        f"{start:.3f}",
                        "-t",
                        f"{boom_window:.3f}",
                        "-i",
                        seg.clip_path,
                    ]

                    # Fallback resolution if none was requested.
                    if self.target_resolution:
                        target_w, target_h = self.target_resolution
                    else:
                        target_w, target_h = 1920, 1080

                    # Number of boomerang cycles requested (1–9 from the UI).
                    try:
                        user_cycles = int(getattr(seg, "cine_boomerang_bounces", 1) or 1)
                    except Exception:
                        user_cycles = 1
                    if user_cycles < 1:
                        user_cycles = 1
                    if user_cycles > 9:
                        user_cycles = 9

                    # Compute how many cycles are needed to fill the segment time.
                    pair_len = max(0.01, boom_window * 2.0)
                    try:
                        needed_cycles = int(math.ceil(target_len / pair_len))
                    except Exception:
                        needed_cycles = 1
                    if needed_cycles < 1:
                        needed_cycles = 1

                    cycles = max(user_cycles, needed_cycles)
                    # Safety cap to avoid runaway memory usage.
                    if cycles > 50:
                        cycles = 50

                    # The loop filter needs a concrete frame count.
                    pair_frames = max(1, int(round(pair_len * 30)))

                    filter_graph_parts = [
                        "[0:v]split[fwd][tmp]",
                        "[tmp]reverse[rev]",
                        "[fwd][rev]concat=n=2:v=1:a=0[pair]",
                        f"[pair]loop=loop={max(0, cycles-1)}:size={pair_frames}:start=0[boom]",
                        f"[boom]trim=duration={target_len:.3f},setpts=PTS-STARTPTS[boomtrim]",
                        f"[boomtrim]setpts=PTS/{seg_sf},fps=fps={fps_expr},scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                        f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black[vout]",
                    ]
                    filter_complex = ";".join(filter_graph_parts)

                    cmd = [
                        self.ffmpeg,
                        "-y",
                        *input_args,
                        "-an",
                        "-filter_complex",
                        filter_complex,
                        "-map",
                        "[vout]",
                        "-c:v",
                        "h264_nvenc",
                        "-preset",
                        "p2",
                        "-rc",
                        "vbr_hq",
                        "-cq",
                        "19",
                        "-b:v",
                        "0",
                        "-r",
                        "30",
                        "-pix_fmt",
                    "yuv420p",
                    out_part,
                    ]
                    code, out = _run_ffmpeg(cmd)
                    if code != 0 or not os.path.exists(out_part):
                        raise RuntimeError(f"ffmpeg failed for boomerang segment {i+1}:\n{out}")
                    parts.append(out_part)
                    continue

                # Multiply cinematic effect: same clip in a grid of multiple tiles.
                if getattr(seg, "cine_multiply", False) and getattr(seg, "clip_path", None) and not getattr(seg, "is_image", False):
                    # Number of screens requested (clamped 2–9).
                    try:
                        screens = int(getattr(seg, "cine_multiply_screens", 4))
                    except Exception:
                        screens = 4
                    screens = max(2, min(9, screens))

                    path = getattr(seg, "clip_path")
                    # Target resolution for the grid (fall back to 1920x1080 if not specified).
                    if self.target_resolution:
                        target_w, target_h = self.target_resolution
                    else:
                        target_w, target_h = 1920, 1080

                    # Make targets stable for yuv420p / NVENC (even sizes),
                    # and keep tile math consistent with pad/scale.
                    try:
                        target_w = int(target_w)
                        target_h = int(target_h)
                    except Exception:
                        pass
                    if target_w % 2:
                        target_w += 1
                    if target_h % 2:
                        target_h += 1

                    # Compute a deterministic layout and per-tile placement that fully covers the frame.
                    cols, rows = _cine_grid_dims(screens)
                    tile_w = max(1, target_w // cols)
                    tile_h = max(1, target_h // rows)

                    # Build input list with per-tile seek & duration, using cached durations.
                    seg_duration = float(seg_trim_dur) or 1.0
                    input_args = []
                    valid_inputs = 0
                    for idx_m in range(screens):
                        dur = duration_cache.get(path)
                        if dur is None:
                            dur = _ffprobe_duration(self.ffprobe, path)
                            if dur is None or dur <= 0.0:
                                break
                            duration_cache[path] = dur
                        max_start = max(0.0, dur - seg_duration)
                        start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
                        input_args.extend(
                            [
                                "-ss",
                                f"{start:.3f}",
                                "-t",
                                f"{seg_duration:.3f}",
                                "-i",
                                path,
                            ]
                        )
                        valid_inputs += 1

                    if valid_inputs > 0:
                        # Build filter_complex: scale/pad each tile, stack with xstack, then fit to target.
                        # Use the same smart layout helper as Mosaic so rows fill the frame cleanly.
                        layout = _cine_grid_layout(valid_inputs, target_w, target_h)
                        per_inputs = []
                        layout_entries = []
                        for idx_m in range(valid_inputs):
                            tile_w_i, tile_h_i, pos_x, pos_y = layout[idx_m]
                            per_inputs.append(
                                f"[{idx_m}:v]scale={tile_w_i}:{tile_h_i}:force_original_aspect_ratio=decrease,"
                                f"pad={tile_w_i}:{tile_h_i}:(ow-iw)/2:(oh-ih)/2:black[v{idx_m}]"
                            )
                            layout_entries.append(f"{pos_x}_{pos_y}")

                        xstack = (
                            "".join(f"[v{idx_m}]" for idx_m in range(valid_inputs))
                            + f"xstack=inputs={valid_inputs}:layout="
                            + "|".join(layout_entries)
                            + ":fill=black[vs]"
                        )

                        # Optional slow-motion factor: keep behaviour consistent with other segments.
                        slow_factor = float(getattr(seg, "slow_factor", 1.0) or 1.0)
                        if slow_factor != 1.0:
                            final = (
                                f"[vs]setpts=PTS/{slow_factor},fps=fps={fps_expr},"
                                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black[vout]"
                            )
                        else:
                            final = (
                                f"[vs]fps=fps={fps_expr},scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black[vout]"
                            )

                        filter_complex = ";".join(per_inputs + [xstack, final])

                        out_part = os.path.join(tmpdir, f"part_{i:04d}.mp4")
                        cmd = [
                            self.ffmpeg,
                            "-y",
                            *input_args,
                            "-an",
                            "-filter_complex",
                            filter_complex,
                            "-map",
                            "[vout]",
                            "-c:v",
                            "h264_nvenc",
                            "-preset",
                            "fast",
                            "-pix_fmt",
                            "yuv420p",
                            out_part,
                        ]
                        code, out = _run_ffmpeg(cmd)
                        if code != 0 or not os.path.exists(out_part):
                            raise RuntimeError(f"ffmpeg failed for multiply segment {i+1}:\n{out}")
                        parts.append(out_part)
                        continue


                # Mosaic cinematic effect: replace this segment with a grid of multiple clips.
                if getattr(seg, "cine_mosaic", False):
                    # Number of screens requested (clamped 2–9).
                    try:
                        screens = int(getattr(seg, "cine_mosaic_screens", 4))
                    except Exception:
                        screens = 4
                    screens = max(2, min(9, screens))

                    if all_video_paths:
                        # Choose random source clips for the mosaic.
                        if len(all_video_paths) >= screens:
                            chosen_paths = random.sample(all_video_paths, screens)
                        else:
                            # Sample with replacement when there are fewer unique clips than screens.
                            chosen_paths = [random.choice(all_video_paths) for _ in range(screens)]

                        # Target resolution for the grid (fall back to 1920x1080 if not specified).
                    if self.target_resolution:
                        target_w, target_h = self.target_resolution
                    else:
                        target_w, target_h = 1920, 1080

                    # Make targets stable for yuv420p / NVENC (even sizes),
                    # and keep tile math consistent with pad/scale.
                    try:
                        target_w = int(target_w)
                        target_h = int(target_h)
                    except Exception:
                        pass
                    if target_w % 2:
                        target_w += 1
                    if target_h % 2:
                        target_h += 1

                        # Compute a deterministic (cols, rows) layout that fully covers the frame.
                        cols, rows = _cine_grid_dims(screens)
                        tile_w = max(1, target_w // cols)
                        tile_h = max(1, target_h // rows)

                        # Build input list with per-source seek & duration, using cached durations.
                        seg_duration = float(seg_trim_dur) or 1.0
                        input_args = []
                        valid_inputs = 0
                        for path in chosen_paths:
                            dur = duration_cache.get(path)
                            if dur is None:
                                dur = _ffprobe_duration(self.ffprobe, path)
                                if dur is None or dur <= 0.0:
                                    continue
                                duration_cache[path] = dur
                            max_start = max(0.0, dur - seg_duration)
                            start = random.uniform(0.0, max_start) if max_start > 0 else 0.0
                            input_args.extend(
                                [
                                    "-ss",
                                    f"{start:.3f}",
                                    "-t",
                                    f"{seg_duration:.3f}",
                                    "-i",
                                    path,
                                ]
                            )
                            valid_inputs += 1

                        if valid_inputs > 0:
                            # Build filter_complex: scale/pad each tile, stack with xstack, then fit to target.
                            # Use a smarter layout so that 5 ⇒ 3+2 and 7 ⇒ 4+3 etc.,
                            # and rows always fill the full frame height.
                            layout = _cine_grid_layout(valid_inputs, target_w, target_h)
                            per_inputs = []
                            layout_entries = []
                            for idx_m in range(valid_inputs):
                                tile_w_i, tile_h_i, pos_x, pos_y = layout[idx_m]
                                per_inputs.append(
                                    f"[{idx_m}:v]scale={tile_w_i}:{tile_h_i}:force_original_aspect_ratio=decrease,"
                                    f"pad={tile_w_i}:{tile_h_i}:(ow-iw)/2:(oh-ih)/2:black[v{idx_m}]"
                                )
                                layout_entries.append(f"{pos_x}_{pos_y}")

                            
                            xstack = (
                                "".join(f"[v{idx_m}]" for idx_m in range(valid_inputs))
                                + f"xstack=inputs={valid_inputs}:layout="
                                + "|".join(layout_entries)
                                + ":fill=black[vs]"
                            )

                            # Optional slow-motion factor: keep behaviour consistent with other segments.
                            slow_factor = float(getattr(seg, "slow_factor", 1.0) or 1.0)
                            if slow_factor != 1.0:
                                final = (
                                    f"[vs]setpts=PTS/{slow_factor},fps=fps={fps_expr},"
                                    f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                                    f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black[vout]"
                                )
                            else:
                                final = (
                                    f"[vs]fps=fps={fps_expr},scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                                    f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black[vout]"
                                )

                            filter_complex = ";".join(per_inputs + [xstack, final])

                            out_part = os.path.join(tmpdir, f"part_{i:04d}.mp4")
                            cmd = [
                                self.ffmpeg,
                                "-y",
                                *input_args,
                                "-an",
                                "-filter_complex",
                                filter_complex,
                                "-map",
                                "[vout]",
                                "-c:v",
                                "h264_nvenc",
                                "-preset",
                                "fast",
                                "-pix_fmt",
                                "yuv420p",
                                out_part,
                            ]
                            code, out = _run_ffmpeg(cmd)
                            if code != 0 or not os.path.exists(out_part):
                                raise RuntimeError(f"ffmpeg failed for mosaic segment {i+1}:\n{out}")
                            parts.append(out_part)
                            continue
                        # If we couldn't build a valid mosaic, fall back to standard rendering.

                # For still images, loop the single frame so it lasts the full segment duration.
                if getattr(seg, "is_image", False):
                    base_cmd = [
                        self.ffmpeg,
                        "-y",
                        "-loop",
                        "1",
                        "-t",
                        f"{seg_trim_dur:.3f}",
                        "-i",
                        seg.clip_path,
                        "-an",
                    ]
                else:
                    # Speedup hits may need repeating if the usable clip window is shorter
                    # than the segment duration. We use a finite stream_loop count to avoid
                    # runaway memory usage from infinite loops.
                    if getattr(seg, "cine_speedup_forward", False) or getattr(seg, "cine_speedup_backward", False):
                        # Determine active factor
                        try:
                            if getattr(seg, "cine_speedup_forward", False):
                                factor = float(getattr(seg, "cine_speedup_forward_factor", 1.5) or 1.5)
                            else:
                                factor = float(getattr(seg, "cine_speedup_backward_factor", 1.5) or 1.5)
                        except Exception:
                            factor = 1.5
                        factor = max(1.25, min(4.0, factor))

                        # Estimate how many repeats are needed based on probed source duration.
                        usable = None
                        try:
                            src_dur = _ffprobe_duration(self.ffprobe, seg.clip_path)
                            if src_dur is not None:
                                usable = max(0.0, float(src_dur) - float(seg.clip_start))
                        except Exception:
                            usable = None

                        loop_args = []
                        if usable and usable > 0:
                            effective_window = usable / factor if factor > 0 else usable
                            if effective_window > 0:
                                needed = int(math.ceil(float(seg.duration) / effective_window))
                                # -stream_loop is "number of additional times" after first play.
                                loops = max(0, needed - 1)
                                # Hard cap for safety.
                                loops = min(50, loops)
                                if loops > 0:
                                    loop_args = ["-stream_loop", str(loops)]

                        base_cmd = [
                            self.ffmpeg,
                            "-y",
                            *loop_args,
                            "-i",
                            seg.clip_path,
                            "-ss",
                            f"{seg.clip_start:.3f}",
                            "-t",
                            f"{seg_trim_dur:.3f}",
                            "-an",
                        ]
                    else:
                        base_cmd = [
                            self.ffmpeg,
                            "-y",
                            "-i",
                            seg.clip_path,
                            "-ss",
                            f"{seg.clip_start:.3f}",
                            "-t",
                            f"{seg_trim_dur:.3f}",
                            "-an",
                        ]

                # Common encoding settings
                encode_args = [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p2",
                    "-rc",
                    "vbr_hq",
                    "-cq",
                    "19",
                    "-b:v",
                    "0",
                    "-r",
                    "30",
                    out_part,
                ]

                                # Attempt filter chains in a safe order:
                # 1) full chain
                # 2) without cinematic one-offs (keeps other FX)
                # 3) without cinematic + impact FX
                # 4) safe chain (resolution-only)
                attempts = []
                if vf_arg:
                    attempts.append(("full", vf_arg))
                if vf_no_cine_arg and vf_no_cine_arg != vf_arg:
                    attempts.append(("no_cine", vf_no_cine_arg))
                if vf_no_cine_no_impact_arg and vf_no_cine_no_impact_arg not in (vf_arg, vf_no_cine_arg):
                    attempts.append(("no_cine_no_impact", vf_no_cine_no_impact_arg))
                if safe_vf_arg and safe_vf_arg not in (vf_arg, vf_no_cine_arg, vf_no_cine_no_impact_arg):
                    attempts.append(("safe", safe_vf_arg))
                if not attempts:
                    attempts.append(("none", None))

                code = 1
                out = ""
                for ai, (aname, avf) in enumerate(attempts):
                    cmd = list(base_cmd)
                    if avf:
                        cmd += ["-vf", avf]
                    cmd += encode_args
                    code, out = _run_ffmpeg(cmd)

                    if code == 0 and os.path.exists(out_part):
                        break

                    if ai < len(attempts) - 1:
                        # Re-try with a safer chain, but keep the segment alive.
                        try:
                            tail = (out or "").splitlines()[-1] if out else ""
                            print(f"[Videoclip_creator] Segment {i+1}: FX effect filter failed; defaulting to safe chain, error is harmless. {tail}")
                        except Exception:
                            pass
                        try:
                            if os.path.exists(out_part):
                                os.remove(out_part)
                        except Exception:
                            pass

                if code != 0 or not os.path.exists(out_part):
                    try:
                        if os.path.exists(out_part):
                            os.remove(out_part)
                    except Exception:
                        pass

                    cmd = list(base_cmd)

                    fallback_vf = None
                    if self.target_resolution is not None:
                        try:
                            tw, th = self.target_resolution
                        except Exception:
                            tw, th = 1920, 1080
                        # Use the simplest and most compatible scaling chain,
                        # ignoring fit_mode. This is only used in rare error
                        # cases where the more advanced chains failed.
                        fallback_vf = (
                            f"scale={int(tw)}:{int(th)}:force_original_aspect_ratio=decrease,"
                            f"pad={int(tw)}:{int(th)}:(ow-iw)/2:(oh-ih)/2:black"
                        )

                    if fallback_vf:
                        cmd += ["-vf", fallback_vf]

                    cmd += encode_args
                    code, out = _run_ffmpeg(cmd)

                if code != 0 or not os.path.exists(out_part):
                    raise RuntimeError(f"ffmpeg failed for segment {i+1} ({seg.clip_path}):\n{out}")
                parts.append(out_part)
            self.progress.emit(90, "Concatenating segments...")
            # Stitch segments. For most transition styles we can safely concat, but
            # Wipe (mode 9) needs overlap between clips, so we apply
            # it here as a real crossfade (no black frames).
            concat_video = os.path.join(tmpdir, "video_concat.mp4")

            # NOTE: xfade/concat filters require both inputs to have identical
            # dimensions + pixel format. Some sources end up 1px off (e.g. 720 vs 719)
            # due to scaling rounding, which makes transitions fail and fall back to cuts.
            stitch_w = None
            stitch_h = None
            if self.target_resolution is not None:
                try:
                    stitch_w, stitch_h = self.target_resolution
                except Exception:
                    stitch_w, stitch_h = None, None
            if (stitch_w is None or stitch_h is None) and parts:
                r0 = _ffprobe_resolution(self.ffprobe, parts[0])
                if r0:
                    stitch_w, stitch_h = r0
            if stitch_w is None or stitch_h is None:
                stitch_w, stitch_h = 1280, 720
            stitch_w = int(stitch_w)
            stitch_h = int(stitch_h)
            if stitch_w % 2:
                stitch_w += 1
            if stitch_h % 2:
                stitch_h += 1
            stitch_norm = (
                f"settb=AVTB,setpts=PTS-STARTPTS,fps={fps_expr},"
                f"scale={stitch_w}:{stitch_h}:force_original_aspect_ratio=decrease,"
                f"pad={stitch_w}:{stitch_h}:(ow-iw)/2:(oh-ih)/2:black,"
                f"setsar=1,"
                f"format=yuv420p"
            )


            def _concat_demuxer(out_path: str) -> None:
                concat_list = os.path.join(tmpdir, "concat.txt")
                with open(concat_list, "w", encoding="utf-8") as f:
                    for p in parts:
                        safe = p.replace("\\", "/")
                        f.write(f"file '{safe}'\n")
                cmd = [
                    self.ffmpeg,
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    concat_list,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    *self._x264_encode_args(),
                    "-pix_fmt",
                    "yuv420p",
                    out_path,
                ]
                code, out = _run_ffmpeg(cmd)
                if code != 0 or not os.path.exists(out_path):
                    raise RuntimeError("Failed to concat segments:\n" + out)

            # Fast path: single part (no stitching needed)
            if len(parts) == 1:
                try:
                    if parts[0] != concat_video:
                        try:
                            shutil.copy(parts[0], concat_video)
                        except Exception:
                            # If copy fails for any reason, just use the original part.
                            concat_video = parts[0]
                except Exception:
                    concat_video = parts[0]
            else:
                dissolve_ids = {"t_exposure_dissolve", "t_luma_fade"}
                scale_punch_ids = {"t_scale_punch"}
                push_ids = {"t_push"}
                slitscan_ids = {"t_slitscan_push"}
                wipe_ids = {"t_wipe"}
                curtain_ids = {"t_curtain_open"}
                pixelize_ids = {"t_pixelize"}
                iris_ids = {"t_iris"}
                radial_ids = {"t_radial_burst"}
                shimmer_ids = {"t_shimmer_blur"}
                distance_ids = {"t_distance"}
                wind_ids = {"t_wind_smears"}
                stitch_ids = dissolve_ids | push_ids | slitscan_ids | wipe_ids | curtain_ids | pixelize_ids | iris_ids | radial_ids | shimmer_ids | distance_ids | wind_ids | scale_punch_ids
                use_stitch = False
                total_overlap = 0.0

                def _save_stitch_log(prefix, cmd, out):
                    """Save full ffmpeg output to a readable log file outside tmpdir (tmpdir is deleted on exit)."""
                    try:
                        base_dir = tempfile.gettempdir()
                    except Exception:
                        base_dir = os.getcwd()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    log_path = os.path.join(base_dir, f"fv_mclip_{prefix}_{ts}.txt")
                    try:
                        with open(log_path, "w", encoding="utf-8", errors="ignore") as f:
                            if cmd:
                                f.write("COMMAND:\n")
                                f.write(" ".join(str(x) for x in cmd))
                                f.write("\n\n")
                            f.write(out or "")
                    except Exception:
                        pass
                    return log_path

                try:
                    # We treat the *incoming* segment's transition as the cut style.
                    for j in range(1, min(len(parts), len(self.segments))):
                        if getattr(self.segments[j], "transition", None) in stitch_ids:
                            use_stitch = True
                            break
                except Exception:
                    use_stitch = False

                if use_stitch:
                    self.progress.emit(90, "Stitching transitions (xfade)...")
                    try:
                        cur = parts[0]
                        for j in range(1, len(parts)):
                            trans = "none"
                            try:
                                if j < len(self.segments):
                                    trans = getattr(self.segments[j], "transition", "none") or "none"
                            except Exception:
                                trans = "none"

                            out_tmp = os.path.join(tmpdir, f"stitch_{j:04d}.mp4")

                            if trans in push_ids:
                                # Real directional push (slide) between clips using xfade.
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                # Visible, editor-like transition length (0.25–0.6s), clamped to clip lengths.
                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.25, 0.6)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.10, min(dur_a, dur_b) * 0.25)
                                    d = min(d, max_allowed)
                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                slide_modes = ["slideleft", "slideright", "slideup", "slidedown"]
                                mode = random.choice(slide_modes)

                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                # Last resort: if slide transitions aren't supported in this ffmpeg build,
                                # try a normal fade dissolve so the run still completes with a *real* overlap.
                                if code != 0 or not os.path.exists(out_tmp):
                                    fc2 = (
                                        f"[0:v]{stitch_norm}[a];"
                                        f"[1:v]{stitch_norm}[b];"
                                        f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                    )
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc2,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)


                            elif trans in scale_punch_ids:

                                # "Scale punch" (zoom-burst) transition:
                                # a punchy zoom-in at the cut, with a short flash + motion smear,
                                # while still being a *real* xfade overlap (editor-style).
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0

                                if d <= 0.0:
                                    d = random.uniform(0.22, 0.48)

                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.12, min(dur_a, dur_b) * 0.28)
                                    d = min(d, max_allowed)

                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                # Strength varies slightly so it doesn't feel copy/pasted.
                                z0 = random.uniform(1.12, 1.22)
                                z_amp = max(0.01, z0 - 1.0)

                                # Apply the "punch" ONLY inside the overlap window by trimming.
                                t0 = offset
                                t1 = offset + d

                                z_expr = f"max(1.0\\, {z0:.4f} - ({z_amp:.4f})*t/{d:.4f})"

                                mid_fx = (
                                    # Pop + tiny flash at the start
                                    "eq=contrast=1.06:brightness=0.02:saturation=1.10,"
                                    "eq=brightness=0.14:contrast=1.12:enable='lt(t,0.060)',"
                                    # Motion smear for the first fraction of the overlap (adds energy)
                                    "tmix=frames=5:weights='1 0.85 0.65 0.45 0.25':enable='between(t,0,0.180)',"
                                    # Zoom-burst (scale up then settle) + center crop back to output size
                                    f"scale=iw*({z_expr}):ih*({z_expr}):eval=frame,"
                                    f"crop={stitch_w}:{stitch_h}:(iw-{stitch_w})/2:(ih-{stitch_h})/2"
                                )

                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f}[x];"
                                    f"[x]split=3[x0][x1][x2];"
                                    f"[x0]trim=0:{t0:.3f},setpts=PTS-STARTPTS[pre];"
                                    f"[x1]trim={t0:.3f}:{t1:.3f},setpts=PTS-STARTPTS,{mid_fx}[mid];"
                                    f"[x2]trim={t1:.3f},setpts=PTS-STARTPTS[post];"
                                    f"[pre][mid][post]concat=n=3:v=1:a=0,format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)

                                if code != 0 or not os.path.exists(out_tmp):
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                # Last resort: if this punch filter chain isn't supported in this ffmpeg build,
                                # fall back to a normal fade dissolve so the run still completes.
                                if code != 0 or not os.path.exists(out_tmp):
                                    fc2 = (
                                        f"[0:v]{stitch_norm}[a];"
                                        f"[1:v]{stitch_norm}[b];"
                                        f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                    )
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc2,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)
                            elif trans in slitscan_ids:

                                # Slit-scan smear push: real stitched push with a modern smear over the overlap window.
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.35, 0.75)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.10, min(dur_a, dur_b) * 0.32)
                                    d = min(d, max_allowed)
                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                # Horizontal pushes read most like slit-scan. Randomize direction to avoid repetition.
                                slide_modes = ["slideleft", "slideright"]
                                mode = random.choice(slide_modes)

                                t0 = offset
                                t1 = offset + d

                                # Build xfade first, then selectively smear ONLY the overlap window by trimming.
                                # This avoids relying on per-filter timeline support across different ffmpeg builds.
                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f}[x];"
                                    f"[x]split=3[x0][x1][x2];"
                                    f"[x0]trim=0:{t0:.3f},setpts=PTS-STARTPTS[pre];"
                                    f"[x1]trim={t0:.3f}:{t1:.3f},setpts=PTS-STARTPTS,"
                                    f"tmix=frames=6:weights='1 0.90 0.75 0.55 0.35 0.18',"
                                    f"boxblur=12:1[mid];"
                                    f"[x2]trim={t1:.3f},setpts=PTS-STARTPTS[post];"
                                    f"[pre][mid][post]concat=n=3:v=1:a=0,format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                # Fallback 1: plain slide push.
                                if code != 0 or not os.path.exists(out_tmp):
                                    fc2 = (
                                        f"[0:v]{stitch_norm}[a];"
                                        f"[1:v]{stitch_norm}[b];"
                                        f"[a][b]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                    )
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc2,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                # Fallback 2: normal fade dissolve.
                                if code != 0 or not os.path.exists(out_tmp):
                                    fc3 = (
                                        f"[0:v]{stitch_norm}[a];"
                                        f"[1:v]{stitch_norm}[b];"
                                        f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                    )
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc3,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)


                            elif trans in wipe_ids:

                                # Real wipe transition using xfade (visible, editor-style).
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.25, 0.6)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.12, min(dur_a, dur_b) * 0.25)
                                    d = min(d, max_allowed)
                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                # Alternate wipe directions so it doesn't feel repetitive.
                                wipe_modes = ["wipeleft", "wiperight", "wipeup", "wipedown", "diagtl", "diagbr"]
                                mode = wipe_modes[int(j) % len(wipe_modes)]

                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    # Fallback encoder (CPU) if NVENC or filter path fails.
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                # Last resort: if wipe transitions are not supported in this ffmpeg build,
                                # try a normal fade dissolve so the run still completes.
                                if code != 0 or not os.path.exists(out_tmp):
                                    fc2 = (
                                        f"[0:v]{stitch_norm}[a];"
                                        f"[1:v]{stitch_norm}[b];"
                                        f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                    )
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc2,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)
                            elif trans in radial_ids:

                                # Real radial burst reveal (stitched): circleopen xfade + short burst flash/blur in the overlap.
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.28, 0.55)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.10, min(dur_a, dur_b) * 0.28)
                                    d = min(d, max_allowed)
                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                mode = "circleopen"
                                t0 = offset
                                t1 = offset + d

                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f}[x];"
                                    f"[x]split=3[x0][x1][x2];"
                                    f"[x0]trim=0:{t0:.3f},setpts=PTS-STARTPTS[pre];"
                                    f"[x1]trim={t0:.3f}:{t1:.3f},setpts=PTS-STARTPTS,"
                                    f"eq=contrast=1.05:brightness=0.06:saturation=1.08,"
                                    f"tmix=frames=5:weights='1 0.85 0.65 0.45 0.25',"
                                    f"boxblur=10:1[mid];"
                                    f"[x2]trim={t1:.3f},setpts=PTS-STARTPTS[post];"
                                    f"[pre][mid][post]concat=n=3:v=1:a=0,format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0:
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)


                            elif trans in curtain_ids:

                                # Curtain/doors open transition using xfade (center opening).
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.25, 0.6)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.12, min(dur_a, dur_b) * 0.25)
                                    d = min(d, max_allowed)
                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                # Alternate between horizontal and vertical opening to avoid repetition.
                                curtain_modes = ["horzopen", "vertopen"]
                                mode = curtain_modes[int(j) % len(curtain_modes)]

                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)

                            elif trans in pixelize_ids:

                                # Pixelize transition using xfade (chunky pixel blocks).
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.22, 0.55)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.12, min(dur_a, dur_b) * 0.25)
                                    d = min(d, max_allowed)
                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition=pixelize:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)

                            elif trans in iris_ids:

                                # Real iris (circle) transition using xfade (visible, editor-style).
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.35, 0.7)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.10, min(dur_a, dur_b) * 0.30)
                                    d = min(d, max_allowed)
                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                # Alternate open/close so it doesn't feel repetitive.
                                iris_modes = ["circleopen", "circleclose"]
                                mode = iris_modes[int(j) % len(iris_modes)]

                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    # Fallback encoder (CPU) if NVENC or filter path fails.
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                # Last resort: if iris transitions aren't supported in this ffmpeg build,
                                # fall back to a normal fade dissolve so the run still completes.
                                if code != 0 or not os.path.exists(out_tmp):
                                    fc2 = (
                                        f"[0:v]{stitch_norm}[a];"
                                        f"[1:v]{stitch_norm}[b];"
                                        f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                    )
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc2,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)
                            elif trans in shimmer_ids:
                                # Shimmer blur crossfade: fast, glossy blur during the overlap window.
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.18, 0.35)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.10, min(dur_a, dur_b) * 0.22)
                                    d = min(d, max_allowed)
                                d = max(0.10, min(0.60, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                t0 = offset
                                t1 = offset + d

                                # Blur/boost only during the overlap so the rest stays crisp.
                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p,"
                                    f"gblur=sigma=18:steps=2:enable='between(t,{t0:.3f},{t1:.3f})',"
                                    f"eq=contrast=1.03:saturation=1.06:enable='between(t,{t0:.3f},{t1:.3f})'[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                # Last resort: if blur filters aren't supported, fall back to a normal fade.
                                if code != 0 or not os.path.exists(out_tmp):
                                    fc2 = (
                                        f"[0:v]{stitch_norm}[a];"
                                        f"[1:v]{stitch_norm}[b];"
                                        f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                    )
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc2,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)

                            elif trans in distance_ids:

                                # "Distance" transition (pseudo-morph / liquid blend feel) using xfade.

                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0

                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0


                                try:

                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)

                                except Exception:

                                    d = 0.0

                                if d <= 0.0:

                                    d = random.uniform(0.25, 0.55)

                                if dur_a > 0.0 and dur_b > 0.0:

                                    max_allowed = max(0.12, min(dur_a, dur_b) * 0.25)

                                    d = min(d, max_allowed)

                                d = max(0.12, min(1.0, float(d)))

                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)


                                fc = (

                                    f"[0:v]{stitch_norm}[a];"

                                    f"[1:v]{stitch_norm}[b];"

                                    f"[a][b]xfade=transition=distance:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"

                                )


                                cmd = [

                                    self.ffmpeg,

                                    "-y",

                                    "-i",

                                    cur,

                                    "-i",

                                    parts[j],

                                    "-filter_complex",

                                    fc,

                                    "-map",

                                    "[v]",

                                    "-c:v",

                                    "h264_nvenc",

                                    "-preset",

                                    "p3",

                                    "-cq",

                                    "19",

                                    "-pix_fmt",

                                    "yuv420p",

                                    out_tmp,

                                ]

                                code, out = _run_ffmpeg(cmd)

                                if code != 0 or not os.path.exists(out_tmp):

                                    cmd = [

                                        self.ffmpeg,

                                        "-y",

                                        "-i",

                                        cur,

                                        "-i",

                                        parts[j],

                                        "-filter_complex",

                                        fc,

                                        "-map",

                                        "[v]",

                                        "-c:v",

                                        "libx264",

                                        *self._x264_encode_args(),
                                        "-pix_fmt",

                                        "yuv420p",

                                        out_tmp,

                                    ]

                                    code, out = _run_ffmpeg(cmd)


                                if code == 0 and os.path.exists(out_tmp):

                                    use_stitch = True

                                    total_overlap += float(d)


                            elif trans in wind_ids:

                                # "Wind smears" transition (directional streaky blend) using xfade wind modes.

                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0

                                if d <= 0.0:
                                    d = random.uniform(0.25, 0.60)

                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.12, min(dur_a, dur_b) * 0.25)
                                    d = min(d, max_allowed)

                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                wind_modes = ["hlwind", "hrwind", "vuwind", "vdwind"]
                                mode = random.choice(wind_modes)

                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition={mode}:duration={d:.3f}:offset={offset:.3f},format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)

                                if code != 0 or not os.path.exists(out_tmp):
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)


                            elif trans in dissolve_ids:
                                # Real dissolve using xfade (no black frames).
                                dur_a = _ffprobe_duration(self.ffprobe, cur) or 0.0
                                dur_b = _ffprobe_duration(self.ffprobe, parts[j]) or 0.0

                                try:
                                    d = float(getattr(self.segments[j], "_stitch_dur", 0.0) or 0.0)
                                except Exception:
                                    d = 0.0
                                if d <= 0.0:
                                    d = random.uniform(0.30, 0.6)
                                if dur_a > 0.0 and dur_b > 0.0:
                                    max_allowed = max(0.12, min(dur_a, dur_b) * 0.25)
                                    d = min(d, max_allowed)
                                d = max(0.12, min(1.0, float(d)))
                                offset = max(0.0, (dur_a if dur_a > 0.0 else 0.0) - d)

                                # Clean dissolve using xfade (no exposure/brightness tricks).
                                # NOTE: xfade can still lift perceived brightness slightly on some clips/codecs.
                                # To keep the original colors stable, apply a tiny negative brightness compensation
                                # ONLY during the overlap window (offset .. offset+duration).
                                overlap_end = offset + d
                                dissolve_b = -0.030  # tune this if dissolves still look "washed"
                                
                                fc = (
                                    f"[0:v]{stitch_norm}[a];"
                                    f"[1:v]{stitch_norm}[b];"
                                    f"[a][b]xfade=transition=fade:duration={d:.3f}:offset={offset:.3f},"
                                    f"eq=brightness={dissolve_b:.3f}:enable='between(t,{offset:.3f},{overlap_end:.3f})',"
                                    f"format=yuv420p[v]"
                                )

                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    # Fallback encoder
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)
                                if code == 0 and os.path.exists(out_tmp):
                                    use_stitch = True
                                    total_overlap += float(d)
                            else:
                                # Hard cut (concat two videos).
                                fc = f"[0:v]{stitch_norm}[a];[1:v]{stitch_norm}[b];[a][b]concat=n=2:v=1:a=0[v]"
                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    cur,
                                    "-i",
                                    parts[j],
                                    "-filter_complex",
                                    fc,
                                    "-map",
                                    "[v]",
                                    "-c:v",
                                    "h264_nvenc",
                                    "-preset",
                                    "p3",
                                    "-cq",
                                    "19",
                                    "-pix_fmt",
                                    "yuv420p",
                                    out_tmp,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code != 0 or not os.path.exists(out_tmp):
                                    cmd = [
                                        self.ffmpeg,
                                        "-y",
                                        "-i",
                                        cur,
                                        "-i",
                                        parts[j],
                                        "-filter_complex",
                                        fc,
                                        "-map",
                                        "[v]",
                                        "-c:v",
                                        "libx264",
                                        "-preset",
                                        "veryfast",
                                        *self._x264_encode_args(),
                                        "-pix_fmt",
                                        "yuv420p",
                                        out_tmp,
                                    ]
                                    code, out = _run_ffmpeg(cmd)

                            if code != 0 or not os.path.exists(out_tmp):
                                log_path = _save_stitch_log("stitch_fail", cmd, out)
                                tail = "\n".join((out or "").splitlines()[-80:])
                                raise RuntimeError("Failed to stitch transitions. Log saved to:\n" + log_path + "\n\n" + tail)

                            # Remove previous intermediate to keep temp size sane.
                            try:
                                if cur not in parts and os.path.exists(cur):
                                    os.remove(cur)
                            except Exception:
                                pass

                            cur = out_tmp
                            try:
                                # keep progress moving between 90–92
                                pct = 90 + int(2 * (j / max(1, (len(parts) - 1))))
                                self.progress.emit(min(92, pct), f"Stitching transitions ({j}/{len(parts)-1})...")
                            except Exception:
                                pass


                        # No end-padding here: overlap compensation is handled by extending the incoming
                        # segment trims during render (prevents long frozen tail).

                        # Final stitched output
                        try:
                            if cur != concat_video:
                                try:
                                    os.replace(cur, concat_video)
                                except Exception:
                                    shutil.copy(cur, concat_video)
                        except Exception:
                            # If moving fails, keep using current file
                            concat_video = cur
                    except Exception:
                        # Fallback: hard concat (no transitions)
                        self.progress.emit(90, "Concatenating segments...")
                        _concat_demuxer(concat_video)
                else:
                    _concat_demuxer(concat_video)

            # Optional intro/outro styling (fade to/from black)
            video_for_mux = concat_video
            if self.intro_fade or self.outro_fade:
                self.progress.emit(93, "Applying intro/outro styling...")
                dur = _ffprobe_duration(self.ffprobe, concat_video)
                if dur and dur > 0.2:
                    styled_video = os.path.join(tmpdir, "video_styled.mp4")
                    vf_parts = []

                    # Fade-in duration: up to 15%% of track, clamped between 0.2s and 1.0s
                    if self.intro_fade:
                        fade_in_d = 0.8
                        fade_in_d = min(fade_in_d, max(0.2, dur * 0.15))
                        vf_parts.append(f"fade=t=in:st=0:d={fade_in_d:.3f}")
                    # Fade-out duration: up to 15%% of track, clamped between 0.2s and 1.2s
                    if self.outro_fade:
                        fade_out_d = 0.8
                        fade_out_d = min(fade_out_d, max(0.2, dur * 0.15))
                        start_out = max(0.0, dur - fade_out_d)
                        vf_parts.append(f"fade=t=out:st={start_out:.3f}:d={fade_out_d:.3f}")
                    if vf_parts:
                        vf_chain = ",".join(vf_parts)
                        cmd = [
                            self.ffmpeg,
                            "-y",
                            "-i",
                            concat_video,
                            "-vf",
                            vf_chain,
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            *self._x264_encode_args(),
                            "-pix_fmt",
                            "yuv420p",
                            styled_video,
                        ]
                        code, out = _run_ffmpeg(cmd)
                        if code == 0 and os.path.exists(styled_video):
                            video_for_mux = styled_video


            # After you have video_for_mux (either styled_video or concat_video)
            video_dur = _ffprobe_duration(self.ffprobe, video_for_mux)
            if video_dur is None:
                raise RuntimeError("Could not probe final video duration")

            audio_dur = self.analysis.duration

            if abs(video_dur - audio_dur) > max(0.02, (1.0 / max(1.0, fps_float))):  # more than ~1 frame off
                adjusted_video = os.path.join(tmpdir, "video_exact_duration.mp4")
                multiplier = audio_dur / video_dur
                vf = f"setpts={multiplier:.10f}*PTS,fps=fps={fps_expr}"
                
                cmd = [
                    self.ffmpeg, "-y",
                    "-i", video_for_mux,
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "veryfast", *self._x264_encode_args(),
                    adjusted_video
                ]
                code, out = _run_ffmpeg(cmd)
                if code != 0:
                    raise RuntimeError(f"Duration fix failed: {out}")
                video_for_mux = adjusted_video

            
            # Optional impact overlays using MP4 assets (fog/confetti/fireworks)
            try:
                assets_dir = os.path.join(os.path.dirname(__file__), "assets")
                overlay_path = None

                # Scan all segments to see which impact styles are actually used
                max_fog = max((float(getattr(s, "impact_fog_density", 0.0) or 0.0) for s in self.segments), default=0.0)
                max_conf = max((float(getattr(s, "impact_confetti_density", 0.0) or 0.0) for s in self.segments), default=0.0)
                max_fire_gold = max((float(getattr(s, "impact_fire_gold_intensity", 0.0) or 0.0) for s in self.segments), default=0.0)
                max_fire_multi = max((float(getattr(s, "impact_fire_multi_intensity", 0.0) or 0.0) for s in self.segments), default=0.0)

                # Pick the first non-zero overlay in a simple priority order
                cand = None
                if max_fog > 0.0:
                    cand = os.path.join(assets_dir, "smoke.mp4")
                elif max_conf > 0.0:
                    cand = os.path.join(assets_dir, "confetti.mp4")
                elif max_fire_gold > 0.0:
                    cand = os.path.join(assets_dir, "fireworks.mp4")
                elif max_fire_multi > 0.0:
                    cand = os.path.join(assets_dir, "fireworks2.mp4")

                if cand and os.path.exists(cand):
                    overlay_path = cand

                if overlay_path:
                    safe_p = overlay_path.replace("\\", "/")
                    overlayed_video = os.path.join(tmpdir, "video_with_overlay.mp4")
                    vf = (
                        f"movie='{safe_p}'[ov];"
                        f"[0:v][ov]overlay=(W-w)/2:(H-h)/2:shortest=1:alpha=0.6"
                    )
                    cmd = [
                        self.ffmpeg,
                        "-y",
                        "-i",
                        video_for_mux,
                        "-vf",
                        vf,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        *self._x264_encode_args(),
                        "-pix_fmt",
                        "yuv420p",
                        overlayed_video,
                    ]
                    code, out = _run_ffmpeg(cmd)
                    if code == 0 and os.path.exists(overlayed_video):
                        video_for_mux = overlayed_video
            except Exception:
                # Overlay effects are purely cosmetic; ignore failures.
                pass

            # Optional music-player visual overlay using the same presets as the live player
            if getattr(self, "use_visual_overlay", False):
                try:
                    from .viz_offline import render_visual_track

                    self.progress.emit(92, "Rendering music-player visuals...")
                    visuals_path = os.path.join(tmpdir, "visuals_track.mp4")
                    # Use the target resolution if available; otherwise fall back to 1920x1080.
                    if self.target_resolution is not None:
                        try:
                            vw, vh = self.target_resolution
                        except Exception:
                            vw, vh = 1920, 1080
                    else:
                        vw, vh = 1920, 1080

                    # Build simple visual schedules based on strategy.
                    strategy = int(getattr(self, "visual_strategy", 0))
                    segment_boundaries = None
                    section_map = None
                    if strategy == 1:
                        # New random visual every segment: use segment timeline starts as boundaries.
                        try:
                            starts = sorted(
                                set(
                                    float(getattr(seg, "timeline_start", 0.0) or 0.0)
                                    for seg in (self.segments or [])
                                )
                            )
                            if not starts or starts[0] > 0.01:
                                starts.insert(0, 0.0)
                            segment_boundaries = starts
                        except Exception:
                            segment_boundaries = None
                    elif strategy == 2:
                        # One visual per section type: intro/verse/chorus/break/drop/outro.
                        try:
                            section_map = [
                                (float(sec.start), float(sec.end), str(sec.kind))
                                for sec in (getattr(self, "analysis", None).sections or [])
                            ]
                        except Exception:
                            section_map = None

                    ok_viz = render_visual_track(
                        audio_path=self.audio_path,
                        out_video=visuals_path,
                        ffmpeg_bin=self.ffmpeg,
                        resolution=(int(vw), int(vh)),
                        fps=int(round(fps_float)),
                        strategy=strategy,
                        segment_boundaries=segment_boundaries,
                        section_map=section_map,
                        section_visual_overrides=getattr(self, "visual_section_overrides", None),
                    )
                    if ok_viz and os.path.exists(visuals_path):
                        overlayed_video_viz = os.path.join(tmpdir, "video_with_visuals.mp4")
                        # Use visuals track as a second input and overlay it directly with transparency.
                        # This avoids relying on the ffmpeg 'movie' source filter and keeps the base
                        # clips visible underneath.
                        # Build filter string with user-selected opacity (visual_overlay_opacity).
                        try:
                            alpha = float(getattr(self, "visual_overlay_opacity", 0.25))
                        except Exception:
                            alpha = 0.25
                        if alpha < 0.0:
                            alpha = 0.0
                        if alpha > 1.0:
                            alpha = 1.0

                        # Build ffmpeg filter string and optionally disable the
                        # music-player visuals for sections where the user
                        # explicitly chose "No beat-synced visual" in the UI.
                        disable_ranges = []
                        try:
                            # Only meaningful for per-section strategy.
                            overrides = getattr(self, "visual_section_overrides", {}) or {}
                            if strategy == 2 and section_map and isinstance(overrides, dict):
                                for start_sec, end_sec, kind in section_map:
                                    key = str(kind).lower().strip()
                                    if key in overrides and overrides.get(key) is None:
                                        try:
                                            st = float(start_sec)
                                            en = float(end_sec)
                                        except Exception:
                                            continue
                                        if en <= st:
                                            continue
                                        if st < 0.0:
                                            st = 0.0
                                        disable_ranges.append((st, en))
                        except Exception:
                            disable_ranges = []

                        enable_clause = ""
                        if disable_ranges:
                            parts = [
                                f"between(t,{st:.3f},{en:.3f})" for st, en in disable_ranges
                            ]
                            joined = "+".join(parts)
                            enable_expr = f"not({joined})"
                            # Use ffmpeg's generic 'enable' expression on the overlay
                            # so visuals are skipped entirely during disabled ranges.
                            enable_clause = f":enable='{enable_expr}'"
                        filter_str = (
                            "[1:v]"
                            "scale=%(w)d:%(h)d:force_original_aspect_ratio=decrease,"
                            "pad=%(w)d:%(h)d:(ow-iw)/2:(oh-ih)/2:black,"
                            "format=rgba,colorchannelmixer=aa=%(alpha)0.2f[viz];"
                            "[0:v][viz]overlay=0:0:shortest=1%(enable)s"
                            % {
                                "w": int(vw),
                                "h": int(vh),
                                "alpha": alpha,
                                "enable": enable_clause,
                            }
                        )

                        cmd = [
                            self.ffmpeg,
                            "-y",
                            "-i",
                            video_for_mux,
                            "-i",
                            visuals_path,
                            "-filter_complex",
                            filter_str,
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            *self._x264_encode_args(),
                            "-pix_fmt",
                            "yuv420p",
                            overlayed_video_viz,
                        ]
                        code, out = _run_ffmpeg(cmd)
                        if code == 0 and os.path.exists(overlayed_video_viz):
                            video_for_mux = overlayed_video_viz
                except Exception:
                    # Visual overlay is cosmetic; ignore failures.
                    pass


            # Timed strobe (global): apply AFTER duration correction and overlays so timestamps stay accurate.
            if getattr(self, "strobe_on_time_times", None):
                try:
                    raw_times = self.strobe_on_time_times or []
                    times: List[float] = []
                    for v in raw_times:
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if fv < 0.0:
                            continue
                        times.append(fv)

                    # De-dup + sort
                    uniq: List[float] = []
                    for t in sorted(times):
                        if not uniq or abs(uniq[-1] - t) >= 0.01:
                            uniq.append(t)

                    if uniq:
                        try:
                            s_val = float(getattr(self, "strobe_flash_strength", 0.0) or 0.0)
                        except Exception:
                            s_val = 0.0
                        if s_val > 0.0:
                            s = max(0.1, min(1.0, s_val))

                            try:
                                speed_ms = int(getattr(self, "strobe_flash_speed_ms", 250) or 250)
                            except Exception:
                                speed_ms = 250
                            speed_ms = max(100, min(1000, speed_ms))
                            period_s = speed_ms / 1000.0
                            pulse_s = max(0.02, min(0.08, period_s * 0.35))

                            flash_a = 0.30 + 0.55 * s  # 0.30–0.85

                            try:
                                total_dur = float(getattr(self.analysis, "duration", 0.0) or 0.0)
                            except Exception:
                                total_dur = 0.0

                            vf_parts: List[str] = []
                            for t0 in uniq:
                                if total_dur > 0.0 and t0 > total_dur + 0.01:
                                    continue
                                end_t = t0 + 0.9
                                if total_dur > 0.0:
                                    end_t = min(total_dur, end_t)
                                if end_t <= t0 + 0.01:
                                    continue

                                vf_parts.append(
                                    "drawbox=x=0:y=0:w=iw:h=ih:"
                                    f"color=white@{flash_a:.2f}:t=fill:"
                                    f"enable='between(t,{t0:.3f},{end_t:.3f})*lt(mod(t-{t0:.3f},{period_s:.3f}),{pulse_s:.3f})'"
                                )

                            if vf_parts:
                                self.progress.emit(94, "Applying timed strobe...")
                                strobe_video = os.path.join(tmpdir, "video_timed_strobe.mp4")
                                vf = ",".join(vf_parts)
                                cmd = [
                                    self.ffmpeg,
                                    "-y",
                                    "-i",
                                    video_for_mux,
                                    "-vf",
                                    vf,
                                    "-c:v",
                                    "libx264",
                                    "-preset",
                                    "veryfast",
                                    *self._x264_encode_args(),
                                    "-pix_fmt",
                                    "yuv420p",
                                    strobe_video,
                                ]
                                code, out = _run_ffmpeg(cmd)
                                if code == 0 and os.path.exists(strobe_video):
                                    video_for_mux = strobe_video
                except Exception:
                    # Cosmetic only — ignore failures
                    pass

            # mux with audio
            self.progress.emit(95, "Merging with audio...")
            if self.out_name_override:
                out_final = os.path.join(self.output_dir, _sanitize_filename(str(self.out_name_override)))
            else:
                base = os.path.splitext(os.path.basename(self.audio_path))[0]
                ts = datetime.now().strftime("%d%m%H%M")  # ddmmhhmm to avoid overwrites
                safe_base = _sanitize_stem(base)
                out_name = f"{safe_base}_clip_{ts}.mp4"
                out_final = os.path.join(self.output_dir, out_name)
            cmd = [
                self.ffmpeg,
                "-y",
                "-i",
                video_for_mux,
                "-i",
                self.audio_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                out_final,
            ]
            code, out = _run_ffmpeg(cmd)
            if code != 0 or not os.path.exists(out_final):
                raise RuntimeError("Failed to mux video and audio:\n" + out)

            self.progress.emit(100, "Done.")
            self.finished_ok.emit(out_final)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------- main widget ----------------------------------


class AutoMusicSyncWidget(QWidget):
    def __init__(self, parent=None, sticky_footer: bool = False) -> None:
        super().__init__(parent)
        self._sticky_footer = bool(sticky_footer)
        self._queue_requested = False
        self._direct_run_active = False
        self._ffmpeg = _find_ffmpeg_from_env()
        self._ffprobe = _find_ffprobe_from_env()
        self._analysis: Optional[MusicAnalysisResult] = None
        self._analysis_config = MusicAnalysisConfig()
        self._worker: Optional[RenderWorker] = None
        self._scan_worker: Optional[SourceScanWorker] = None
        self._pending_audio: Optional[str] = None
        self._pending_video: Optional[str] = None
        self._pending_out_dir: Optional[str] = None
        self.clip_sources = []
        self.image_sources = []  # List for loaded images
        # User-controlled interval (number of segments) between still images
        self.image_segment_interval = 4
        # Optional per-section media overrides chosen from the timeline tab
        self._section_media: Dict[int, ClipSource] = {}
        # Enabled transition styles for randomization (indices of combo_transitions)
        self._enabled_transition_modes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
        # Guard flag used when toggling No FX programmatically so we don't immediately undo it
        self._nofx_guard = False

        # Timed strobe (flash strobe at user-specified timeline seconds)
        self._strobe_on_time_times: List[float] = []

        # Per-section music-player visual overrides (intro/verse/chorus/break/drop/outro)
        self._visual_section_overrides = {}
        # Default opacity for music-player visuals overlay (0.0–1.0)
        self.visual_overlay_opacity = 0.25
        # Settings for remembering last paths & options
        self._settings = QSettings("FrameVision", "MusicClipCreator")
        # Manager for per-visual thumbnails (lazy-generated real previews)
        self._visual_thumbs = VisualThumbManager(self, ffmpeg=self._ffmpeg)

        self._build_ui()
        self._load_settings()

        # Hide still-image UI block (images in generator)
        self._hide_image_sources_block()


    # ----------------------------- UI toast ---------------------------------
    def _show_toast(self, message: str, duration_ms: int = 2200, fade_ms: int = 650) -> None:
        """Show a small non-blocking toast near the bottom of the window."""
        try:
            parent = self.window() if self.window() is not None else self
        except Exception:
            parent = self

        # Close previous toast if still visible
        try:
            old = getattr(self, "_active_toast", None)
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
                try:
                    old.deleteLater()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            toast = QFrame(parent)
            toast.setObjectName("FVToast")
            toast.setWindowFlags(Qt.FramelessWindowHint | Qt.ToolTip)
            toast.setAttribute(Qt.WA_ShowWithoutActivating, True)
            toast.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            lay = QVBoxLayout(toast)
            lay.setContentsMargins(14, 10, 14, 10)
            lay.setSpacing(0)
            lbl = QLabel(str(message), toast)
            lbl.setWordWrap(True)
            lbl.setAlignment(Qt.AlignCenter)
            lay.addWidget(lbl)

            # Minimal styling (keeps working even if no app stylesheet is loaded)
            toast.setStyleSheet(
                "#FVToast {"
                " background: rgba(0,0,0,190);"
                " color: white;"
                " border: 1px solid rgba(255,255,255,70);"
                " border-radius: 10px;"
                " }"
            )

            toast.adjustSize()

            # Position: bottom-center of the parent window
            try:
                g = parent.frameGeometry()
            except Exception:
                g = parent.geometry()
            x = int(g.x() + (g.width() - toast.width()) * 0.5)
            y = int(g.y() + g.height() - toast.height() - 80)
            toast.move(x, y)

            eff = QGraphicsOpacityEffect(toast)
            eff.setOpacity(1.0)
            toast.setGraphicsEffect(eff)

            anim = QPropertyAnimation(eff, b"opacity", toast)
            anim.setDuration(int(max(120, fade_ms)))
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.setEasingCurve(QEasingCurve.InOutQuad)

            def _finish():
                try:
                    toast.close()
                except Exception:
                    pass
                try:
                    toast.deleteLater()
                except Exception:
                    pass
                try:
                    if getattr(self, "_active_toast", None) is toast:
                        self._active_toast = None
                except Exception:
                    pass

            anim.finished.connect(_finish)

            # Keep references alive
            toast._toast_anim = anim  # type: ignore[attr-defined]
            self._active_toast = toast

            toast.show()
            toast.raise_()

            # Start fade after the on-screen duration
            QTimer.singleShot(int(max(300, duration_ms)), anim.start)
        except Exception:
            # Toast is best-effort; never break workflows if it fails.
            pass


    def _build_ui(self) -> None:
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        # Outer layout holds a tab widget so we can have a dedicated timeline tab
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 6)
        outer.setSpacing(6)

        self.tabs = QTabWidget(self)
        outer.addWidget(self.tabs)

        # Main generator tab
        page_main = QWidget(self)
        main = QVBoxLayout(page_main)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(6)
        form = QFormLayout()
        self._form_layout = form
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setSpacing(4)

        # audio
        self.edit_audio = QLineEdit(self)
        btn_a = QPushButton("Browse...", self)
        row_a = QHBoxLayout()
        row_a.addWidget(self.edit_audio, 1)
        row_a.addWidget(btn_a)
        form.addRow("Music / Audio:", row_a)

        # video (file or folder)
        self.edit_video = QLineEdit(self)
        btn_vf = QPushButton("Video file...", self)
        btn_vd = QPushButton("Clip folder...", self)
        btn_vmf = QPushButton("Clip files...", self)

        # Put the pickers on the first row, and the resulting path(s) on the next row
        # so the selected input can use the full width.
        row_v_btns = QHBoxLayout()
        row_v_btns.setContentsMargins(0, 0, 0, 0)
        row_v_btns.setSpacing(6)
        row_v_btns.addWidget(btn_vf)
        row_v_btns.addWidget(btn_vd)
        row_v_btns.addWidget(btn_vmf)
        row_v_btns.addStretch(1)
        form.addRow("", row_v_btns)
        form.addRow("", self.edit_video)

        # Optional loader for image sources (hidden by default)
        self.image_sources_row = QWidget(self)
        row_clips = QHBoxLayout()
        row_clips.setContentsMargins(0, 0, 0, 0)
        row_clips.setSpacing(6)
        self.btn_load_images = QPushButton("Load Images...", self.image_sources_row)
        self.btn_load_images_folder = QPushButton("Image folder...", self.image_sources_row)
        self.btn_load_images.setToolTip("Pick one or more image files to use as still-image inserts.")
        self.btn_load_images_folder.setToolTip("Pick a folder and load all supported images inside.")
        row_clips.addWidget(self.btn_load_images)
        row_clips.addWidget(self.btn_load_images_folder)
        self.image_sources_row.setLayout(row_clips)
        self.btn_load_images.clicked.connect(self._on_load_images)
        self.btn_load_images_folder.clicked.connect(self._on_load_images_folder)
        form.addRow("Loaded sources:", self.image_sources_row)

        # Slider: how many segments between still images
        self.image_interval_row = QWidget(self)
        row_img_interval = QHBoxLayout()
        row_img_interval.setContentsMargins(0, 0, 0, 0)
        row_img_interval.setSpacing(6)
        self.label_image_interval = QLabel("New image every 4 segments", self.image_interval_row)
        self.slider_image_interval = QSlider(Qt.Horizontal, self.image_interval_row)
        self.slider_image_interval.setMinimum(0)
        self.slider_image_interval.setMaximum(20)
        self.slider_image_interval.setSingleStep(1)
        self.slider_image_interval.setPageStep(2)
        self.slider_image_interval.setValue(4)
        self.slider_image_interval.setToolTip(
            "How many segments to wait before switching to a new still image.\n"
            "2 = very frequent image changes, 20 = very rare."
        )
        self.slider_image_interval.valueChanged.connect(self._on_image_interval_changed)
        row_img_interval.addWidget(self.label_image_interval)
        row_img_interval.addWidget(self.slider_image_interval, 1)
        self.image_interval_row.setLayout(row_img_interval)
        form.addRow("", self.image_interval_row)

        self.list_sources = QListWidget(self)
        form.addRow("", self.list_sources)

        # output
        self.edit_output = QLineEdit(self)
        btn_o = QPushButton("Browse...", self)
        row_o = QHBoxLayout()
        row_o.addWidget(self.edit_output, 1)
        row_o.addWidget(btn_o)
        form.addRow("Output folder:", row_o)

        # clip order
        row_order = QHBoxLayout()
        row_order.addWidget(QLabel("Clip order:", self))
        self.combo_clip_order = QComboBox(self)
        self.combo_clip_order.addItems(
            [
                "Random (default)",
                "Sequential",
                "Shuffle (no repeats)",
            ]
        )
        self.combo_clip_order.setToolTip(
            "How clips from a folder are picked:\n"
            "- Random: random clips, avoid using the same clip twice in a row.\n"
            "- Sequential: go through the folder in order, then loop.\n"
            "- Shuffle: each round uses all clips once in random order before repeating."
        )
        row_order.addWidget(self.combo_clip_order, 1)
        row_order.addStretch(1)
        form.addRow(row_order)

        # minimum clip length filter
        row_minclip = QHBoxLayout()
        self.check_min_clip = QCheckBox("Ignore clips shorter than", self)
        self.check_min_clip.setChecked(False)
        self.spin_min_clip = QDoubleSpinBox(self)
        self.spin_min_clip.setDecimals(1)
        self.spin_min_clip.setSingleStep(0.5)
        self.spin_min_clip.setRange(0.5, 10.0)
        self.spin_min_clip.setValue(1.5)
        self.spin_min_clip.setSuffix(" s")
        self.check_min_clip.setToolTip(
            "When enabled, video clips shorter than this duration will be ignored\n"
            "during clip discovery. Useful to avoid ultra-short fragments that can\n"
            "cause jittery pacing or visual glitches."
        )
        row_minclip.addWidget(self.check_min_clip)
        row_minclip.addWidget(self.spin_min_clip)
        row_minclip.addStretch(1)
        form.addRow(row_minclip)

        # random seed
        row_seed = QHBoxLayout()
        row_seed.addWidget(QLabel("Random seed:", self))
        self.spin_seed = QSpinBox(self)
        self.spin_seed.setRange(0, 999999999)
        self.spin_seed.setValue(1337)
        self.spin_seed.setEnabled(False)
        self.spin_seed.setToolTip(
            "Seed value used when 'Use fixed seed' is enabled.\n"
            "Same seed + same inputs = repeatable clip order and FX.\n"
            "Disable to get a fresh random result each run."
        )
        row_seed.addWidget(self.spin_seed)
        self.check_use_seed = QCheckBox("Use fixed seed", self)
        self.check_use_seed.setToolTip(
            "When enabled, the random generator is seeded with the value above,\n"
            "so you can reproduce the same edit again. When disabled, each run\n"
            "will be different."
        )
        row_seed.addWidget(self.check_use_seed)
        row_seed.addStretch(1)
        form.addRow(row_seed)

        # resolution
        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("Output resolution:", self))
        self.combo_res = QComboBox(self)
        self.combo_res.addItems(
            [
                "Auto (single video: keep source)",
                "Multi clips: highest clip resolution",
                "Multi clips: lowest clip resolution",
                "Fixed: 1080p (1920×1080) 16:9",
                "Fixed: 720p (1280×720) 16:9",
                "Fixed: 480p (854×480) 16:9",
                "Fixed: 1080p (1080×1920) 9:16",
                "Fixed: 720p (720×1280) 9:16",
                "Fixed: 480p (480×854) 9:16",
            ]
        )
        self.combo_res.setToolTip(
            "Resolution strategy when working with multiple clips.\n"
            "- Auto: single video -> keep its resolution.\n"
            "- Highest/Lowest clip resolution: unify to that size.\n"
            "- Fixed 16:9: scale everything to 480p / 720p / 1080p landscape.\n"
            "- Fixed 9:16: scale everything to 480p / 720p / 1080p vertical."
        )
        # Default output resolution: Fixed 720p (1280×720)
        # (Only used when there are no saved settings yet.)
        try:
            self.combo_res.setCurrentIndex(4)
        except Exception:
            pass
        row_res.addWidget(self.combo_res, 1)
        form.addRow(row_res)

        # bitrate
        row_br = QHBoxLayout()
        self.check_keep_source_bitrate = QCheckBox("Keep Source Bitrate", self)
        self.check_keep_source_bitrate.setChecked(False)
        self.check_keep_source_bitrate.setToolTip(
            "When enabled, the final render will try to keep the source video bitrate\n"
            "(or the highest bitrate among a sample of your clips) to preserve quality.\n"
            "This can increase output file size."
        )
        row_br.addWidget(self.check_keep_source_bitrate)
        row_br.addStretch(1)
        form.addRow(row_br)

        # frame fit mode
        row_fit = QHBoxLayout()
        row_fit.addWidget(QLabel("Frame fit:", self))
        self.combo_fit = QComboBox(self)
        self.combo_fit.addItems(
            [
                "Original (letterbox/pillarbox)",
                "Fill (crop to fill)",
                "Stretch (distort to fill)",
            ]
        )
        self.combo_fit.setToolTip(
            "How clips are fitted into the target resolution.\\n"\
            "- Original: keep aspect ratio, add black bars if needed.\\n"\
            "- Fill: keep aspect ratio but crop edges to fill the frame.\\n"\
            "- Stretch: ignore aspect ratio and stretch to fill the frame."
        )
        row_fit.addWidget(self.combo_fit, 1)
        form.addRow(row_fit)

        main.addLayout(form)

        # options
        self.box_opts = QGroupBox("Options", self)
        opts = QVBoxLayout(self.box_opts)

        # FX level
        row_fx = QHBoxLayout()
        row_fx.addWidget(QLabel("FX Level:", self))
        self.combo_fx = QComboBox(self)
        self.combo_fx.addItems(["Minimal", "Moderate", "High"])
        self.combo_fx.setToolTip(
            "Choose how active the visual effects should be:\n"
            "- Minimal: clean cuts with subtle accents.\n"
            "- Moderate: more zoom/flash on peaks.\n"
            "- High: strongest visual activity, best with many short clips."
        )
        row_fx.addWidget(self.combo_fx, 1)
        row_fx.addStretch(1)
        opts.addLayout(row_fx)

        # microclip toggles
        row_micro = QHBoxLayout()
        self.check_micro_chorus = QCheckBox("Microclips in chorus/drops only", self)
        self.check_micro_chorus.setToolTip(
            "Short energetic microclips only during high-energy sections\n"
            "(chorus / drops). Verses and intros stay calmer."
        )
        row_micro.addWidget(self.check_micro_chorus)

        self.check_micro_all = QCheckBox("Microclips for the whole track", self)
        self.check_micro_all.setToolTip(
            "Microclips are used throughout the entire song. Great with many\n"
            "short clips; can feel hyperactive on a single long video."
        )
        row_micro.addWidget(self.check_micro_all)

        self.check_micro_verses = QCheckBox("Verses only", self)
        self.check_micro_verses.setToolTip(
            "Microclips are used only during verse sections. Choruses, drops\n"
            "and other parts of the song use longer, calmer shots."
        )
        row_micro.addWidget(self.check_micro_verses)

        row_micro.addStretch(1)
        opts.addLayout(row_micro)

        # full-length mode (hidden, always enabled internally)
        self.check_full_length = QCheckBox("Always fill full music length", self)
        self.check_full_length.setToolTip(
            "When enabled, the generated video will be extended with extra segments\n"
            "so that its duration matches the full music track. Useful when microclips\n"
            "or sparse beats would otherwise create a shorter video."
        )
        self.check_full_length.setChecked(True)
        self.check_full_length.hide()

        # intro / outro fades
        row_fade = QHBoxLayout()
        self.check_intro_fade = QCheckBox("Fade in from black at start         ", self)
        self.check_intro_fade.setChecked(True)
        self.check_intro_fade.setToolTip(
            "Add a short fade-in from black at the very beginning of the music clip."
        )
        row_fade.addWidget(self.check_intro_fade)

        self.check_outro_fade = QCheckBox("Fade out to black at end", self)
        self.check_outro_fade.setChecked(True)
        self.check_outro_fade.setToolTip(
            "Add a short fade-out to black at the very end of the music clip."
        )
        row_fade.addWidget(self.check_outro_fade)
        row_fade.addStretch(1)
        opts.addLayout(row_fade)

        # Intro: transitions-only mode (no FX)
        row_intro_only = QHBoxLayout()
        self.check_intro_transitions_only = QCheckBox("Intro: transitions only (disable FX during intro)", self)
        self.check_intro_transitions_only.setChecked(False)
        self.check_intro_transitions_only.setToolTip(
            "When enabled, the intro section uses transitions only: disables video FX, "
            "slow motion, cinematic FX and break/drop impact FX during the intro. "
            "Useful when the intro has little bass/energy."
        )
        row_intro_only.addWidget(self.check_intro_transitions_only)
        row_intro_only.addStretch(1)
        opts.addLayout(row_intro_only)

        # slow motion options
        row_slow_master = QHBoxLayout()
        self.check_slow_enable = QCheckBox("Enable slow-motion effect", self)
        self.check_slow_enable.setToolTip(
            "Enable to slow down the video (visuals only) during selected song sections.\n"
            "Audio stays at normal speed."
        )
        row_slow_master.addWidget(self.check_slow_enable)
        row_slow_master.addStretch(1)
        opts.addLayout(row_slow_master)

        self.slow_options = QWidget(self)
        slow_layout = QVBoxLayout(self.slow_options)
        slow_layout.setContentsMargins(26, 0, 0, 0)
        slow_layout.setSpacing(2)

        row_slow_sections1 = QHBoxLayout()
        self.check_slow_intro = QCheckBox("Intro", self.slow_options)
        self.check_slow_break = QCheckBox("Break", self.slow_options)
        self.check_slow_chorus = QCheckBox("Chorus", self.slow_options)
        row_slow_sections1.addWidget(self.check_slow_intro)
        row_slow_sections1.addWidget(self.check_slow_break)
        row_slow_sections1.addWidget(self.check_slow_chorus)
        row_slow_sections1.addStretch(1)
        slow_layout.addLayout(row_slow_sections1)

        row_slow_sections2 = QHBoxLayout()
        self.check_slow_drop = QCheckBox("Drop", self.slow_options)
        self.check_slow_outro = QCheckBox("Outro", self.slow_options)
        self.check_slow_random = QCheckBox("Random (1× every 60s max)", self.slow_options)
        row_slow_sections2.addWidget(self.check_slow_drop)
        row_slow_sections2.addWidget(self.check_slow_outro)
        row_slow_sections2.addWidget(self.check_slow_random)
        row_slow_sections2.addStretch(1)
        slow_layout.addLayout(row_slow_sections2)

        row_slow_factor = QHBoxLayout()
        label_slow = QLabel("Slow-motion factor:", self.slow_options)
        row_slow_factor.addWidget(label_slow)
        self.slider_slow_factor = QSlider(Qt.Horizontal, self.slow_options)
        self.slider_slow_factor.setMinimum(10)
        self.slider_slow_factor.setMaximum(100)
        self.slider_slow_factor.setSingleStep(1)
        self.slider_slow_factor.setPageStep(5)
        self.slider_slow_factor.setValue(50)
        self.slider_slow_factor.setToolTip(
            "Video speed factor for slow motion.\n"
            "1.00 = normal speed, 0.50 = 2× slower, 0.25 = 4× slower."
        )
        row_slow_factor.addWidget(self.slider_slow_factor, 1)
        self.label_slow_factor_value = QLabel("0.50x", self.slow_options)
        self.label_slow_factor_value.setMinimumWidth(40)
        row_slow_factor.addWidget(self.label_slow_factor_value)
        slow_layout.addLayout(row_slow_factor)

        opts.addWidget(self.slow_options)
        self.slow_options.setVisible(False)

        # cinematic effects options
        row_cine_master = QHBoxLayout()
        self.check_cine_enable = QCheckBox("Enable cinematic effects", self)
        self.check_cine_enable.setToolTip(
            "Enable a few rare, high-impact visual effects (shutter-pop, stutter, "
            "reverse-bounce, ramps). Effects are placed automatically "
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_enable.setChecked(True)
        except Exception:
            pass
        row_cine_master.addWidget(self.check_cine_enable)
        row_cine_master.addStretch(1)
        opts.addLayout(row_cine_master)

        self.cine_options = QWidget(self)
        cine_layout = QVBoxLayout(self.cine_options)
        cine_layout.setContentsMargins(26, 0, 0, 0)
        cine_layout.setSpacing(2)

        # Quick toggle: turn all cinematic toggles on/off at once
        row_cine_all = QHBoxLayout()
        self.btn_cine_all = QPushButton("All on/off", self.cine_options)
        self.btn_cine_all.setToolTip("Turn ON/OFF all toggles in the cinematic section.")
        row_cine_all.addWidget(self.btn_cine_all)
        row_cine_all.addStretch(1)
        cine_layout.addLayout(row_cine_all)

        # Shutter-pop effect
        row_cine_freeze = QHBoxLayout()
        self.check_cine_freeze = QCheckBox("Shutter-pop (punch + smear)", self.cine_options)
        self.check_cine_freeze.setToolTip(
            "Occasionally add a quick camera-shutter style punch: micro-zoom + glossy motion smear.\n"
            "No speed change, no freezing."
        )
        row_cine_freeze.addWidget(self.check_cine_freeze)
        label_freeze_len = QLabel("Hit:", self.cine_options)
        row_cine_freeze.addWidget(label_freeze_len)
        self.slider_cine_freeze_len = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_freeze_len.setMinimum(10)   # 0.10 s
        self.slider_cine_freeze_len.setMaximum(100)  # 1.00 s
        self.slider_cine_freeze_len.setSingleStep(1)
        self.slider_cine_freeze_len.setValue(50)     # default 0.50 s
        row_cine_freeze.addWidget(self.slider_cine_freeze_len, 1)
        self.label_cine_freeze_len = QLabel("0.50 s", self.cine_options)
        self.label_cine_freeze_len.setMinimumWidth(50)
        row_cine_freeze.addWidget(self.label_cine_freeze_len)
        cine_layout.addLayout(row_cine_freeze)

        row_cine_freeze_zoom = QHBoxLayout()
        label_freeze_zoom = QLabel("Punch:", self.cine_options)
        row_cine_freeze_zoom.addWidget(label_freeze_zoom)
        self.slider_cine_freeze_zoom = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_freeze_zoom.setMinimum(5)    # 5%%
        self.slider_cine_freeze_zoom.setMaximum(30)   # 30%%
        self.slider_cine_freeze_zoom.setSingleStep(1)
        self.slider_cine_freeze_zoom.setValue(15)     # default 15%%
        row_cine_freeze_zoom.addWidget(self.slider_cine_freeze_zoom, 1)
        self.label_cine_freeze_zoom = QLabel("+15%%", self.cine_options)
        self.label_cine_freeze_zoom.setMinimumWidth(50)
        row_cine_freeze_zoom.addWidget(self.label_cine_freeze_zoom)
        cine_layout.addLayout(row_cine_freeze_zoom)

        # Prism whip (horizontal drift)
        row_cine_tear_v = QHBoxLayout()
        self.check_cine_tear_v = QCheckBox("Prism whip (horizontal drift)", self.cine_options)
        self.check_cine_tear_v.setToolTip(
            "A short, eye-catching whip: micro-zoom + chroma smear + soft motion trail (horizontal drift)."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_tear_v.setChecked(True)
        except Exception:
            pass
        row_cine_tear_v.addWidget(self.check_cine_tear_v)
        label_cine_tear_v = QLabel("Strength:", self.cine_options)
        row_cine_tear_v.addWidget(label_cine_tear_v)
        self.slider_cine_tear_v_strength = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_tear_v_strength.setMinimum(10)
        self.slider_cine_tear_v_strength.setMaximum(100)
        self.slider_cine_tear_v_strength.setSingleStep(1)
        self.slider_cine_tear_v_strength.setValue(70)
        row_cine_tear_v.addWidget(self.slider_cine_tear_v_strength, 1)
        self.label_cine_tear_v_strength = QLabel("0.70", self.cine_options)
        self.label_cine_tear_v_strength.setMinimumWidth(40)
        row_cine_tear_v.addWidget(self.label_cine_tear_v_strength)
        cine_layout.addLayout(row_cine_tear_v)

        # Prism whip (vertical drift)
        row_cine_tear_h = QHBoxLayout()
        self.check_cine_tear_h = QCheckBox("Prism whip (vertical drift)", self.cine_options)
        self.check_cine_tear_h.setToolTip(
            "A short, eye-catching whip: micro-zoom + chroma smear + soft motion trail (vertical drift)."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_tear_h.setChecked(True)
        except Exception:
            pass
        row_cine_tear_h.addWidget(self.check_cine_tear_h)
        label_cine_tear_h = QLabel("Strength:", self.cine_options)
        row_cine_tear_h.addWidget(label_cine_tear_h)
        self.slider_cine_tear_h_strength = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_tear_h_strength.setMinimum(10)
        self.slider_cine_tear_h_strength.setMaximum(100)
        self.slider_cine_tear_h_strength.setSingleStep(1)
        self.slider_cine_tear_h_strength.setValue(70)
        row_cine_tear_h.addWidget(self.slider_cine_tear_h_strength, 1)
        self.label_cine_tear_h_strength = QLabel("0.70", self.cine_options)
        self.label_cine_tear_h_strength.setMinimumWidth(40)
        row_cine_tear_h.addWidget(self.label_cine_tear_h_strength)
        cine_layout.addLayout(row_cine_tear_h)

        # Color-cycle glitch (cinematic)
        row_cine_color_cycle = QHBoxLayout()
        self.check_cine_color_cycle = QCheckBox("Color-cycle glitch", self.cine_options)
        self.check_cine_color_cycle.setToolTip(
            "Occasionally add a short hue-cycle glitch (like the break/drop Color-cycle),\n"
            "but as a cinematic one-off effect.\n"
            "Speed controls how fast the hue cycles: 100ms = very fast, 1000ms = slow."
        )
        row_cine_color_cycle.addWidget(self.check_cine_color_cycle)
        label_cine_color_cycle = QLabel("Speed:", self.cine_options)
        row_cine_color_cycle.addWidget(label_cine_color_cycle)
        self.slider_cine_color_cycle_speed = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_color_cycle_speed.setMinimum(100)
        self.slider_cine_color_cycle_speed.setMaximum(1000)
        self.slider_cine_color_cycle_speed.setSingleStep(50)
        self.slider_cine_color_cycle_speed.setPageStep(100)
        self.slider_cine_color_cycle_speed.setValue(400)
        row_cine_color_cycle.addWidget(self.slider_cine_color_cycle_speed, 1)
        self.label_cine_color_cycle_speed = QLabel("400 ms", self.cine_options)
        self.label_cine_color_cycle_speed.setMinimumWidth(55)
        row_cine_color_cycle.addWidget(self.label_cine_color_cycle_speed)
        cine_layout.addLayout(row_cine_color_cycle)

        # Stutter slice
        row_cine_stutter = QHBoxLayout()
        self.check_cine_stutter = QCheckBox("Stutter slice", self.cine_options)
        self.check_cine_stutter.setToolTip(
            "Occasionally create a short triple-frame style stutter hit for extra impact."
        )
        row_cine_stutter.addWidget(self.check_cine_stutter)
        label_stutter = QLabel("Repeats:", self.cine_options)
        row_cine_stutter.addWidget(label_stutter)
        self.spin_cine_stutter_repeats = QSpinBox(self.cine_options)
        self.spin_cine_stutter_repeats.setRange(2, 5)
        self.spin_cine_stutter_repeats.setValue(3)
        row_cine_stutter.addWidget(self.spin_cine_stutter_repeats)
        row_cine_stutter.addStretch(1)
        cine_layout.addLayout(row_cine_stutter)

        # Reverse-bounce
        row_cine_reverse = QHBoxLayout()
        self.check_cine_reverse = QCheckBox("Reverse-bounce", self.cine_options)
        self.check_cine_reverse.setToolTip(
            "Occasionally play a very short fragment backwards for a bounce-like feel."
        )
        row_cine_reverse.addWidget(self.check_cine_reverse)
        label_reverse_len = QLabel("Length:", self.cine_options)
        row_cine_reverse.addWidget(label_reverse_len)
        self.slider_cine_reverse_len = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_reverse_len.setMinimum(10)   # 0.10 s
        self.slider_cine_reverse_len.setMaximum(200)  # 2.00 s
        self.slider_cine_reverse_len.setSingleStep(1)
        self.slider_cine_reverse_len.setValue(50)     # default 0.50 s
        row_cine_reverse.addWidget(self.slider_cine_reverse_len, 1)
        self.label_cine_reverse_len = QLabel("0.50 s", self.cine_options)
        self.label_cine_reverse_len.setMinimumWidth(50)
        row_cine_reverse.addWidget(self.label_cine_reverse_len)
        cine_layout.addLayout(row_cine_reverse)

        # Boomerang loop
        row_cine_boom = QHBoxLayout()
        self.check_cine_boomerang = QCheckBox("Boomerang loop", self.cine_options)
        self.check_cine_boomerang.setToolTip(
            "Occasionally turn a segment into a short forward/back 'boomerang' bounce.\n"
            "Balanced presets use a single bounce; club/rave can use faster mini-loops."
        )
        row_cine_boom.addWidget(self.check_cine_boomerang)
        label_cine_boom = QLabel("Bounces:", self.cine_options)
        row_cine_boom.addWidget(label_cine_boom)
        self.slider_cine_boomerang_bounces = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_boomerang_bounces.setMinimum(1)
        self.slider_cine_boomerang_bounces.setMaximum(9)
        self.slider_cine_boomerang_bounces.setSingleStep(1)
        self.slider_cine_boomerang_bounces.setValue(2)
        row_cine_boom.addWidget(self.slider_cine_boomerang_bounces, 1)
        self.label_cine_boomerang_bounces = QLabel("x2", self.cine_options)
        self.label_cine_boomerang_bounces.setMinimumWidth(40)
        row_cine_boom.addWidget(self.label_cine_boomerang_bounces)
        cine_layout.addLayout(row_cine_boom)

        # Dimension portal shapes
        row_cine_dim = QHBoxLayout()
        self.check_cine_dimension = QCheckBox("Different dimension (portal shapes)", self.cine_options)
        self.check_cine_dimension.setToolTip(
            "Occasionally show the segment inside a 'portal' window: rectangle / trapezoid / diamond / 9:16 portrait.\n"
            "Lightweight geometry-only effect (no PNG masks)."
        )
        row_cine_dim.addWidget(self.check_cine_dimension)
        row_cine_dim.addStretch(1)
        cine_layout.addLayout(row_cine_dim)

        # 9:16 pan crop (boomerang)
        # Main row: enable + speed
        row_cine_pan916 = QHBoxLayout()
        self.check_cine_pan916 = QCheckBox("Slice reveal (boomerang)", self.cine_options)
        self.check_cine_pan916.setToolTip(
            "Shows only a part of the frame at a time on a black/transparent canvas: left → middle → right → middle → ...\n"
            "Repeats automatically for long segments. Looks great on 16:9 output."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_pan916.setChecked(True)
        except Exception:
            pass
        row_cine_pan916.addWidget(self.check_cine_pan916)
        label_cine_pan916 = QLabel("Step:", self.cine_options)
        row_cine_pan916.addWidget(label_cine_pan916)
        self.slider_cine_pan916_speed = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_pan916_speed.setRange(150, 1000)
        self.slider_cine_pan916_speed.setSingleStep(50)
        self.slider_cine_pan916_speed.setPageStep(100)
        self.slider_cine_pan916_speed.setValue(400)
        row_cine_pan916.addWidget(self.slider_cine_pan916_speed, 1)
        self.label_cine_pan916_speed = QLabel("400 ms", self.cine_options)
        self.label_cine_pan916_speed.setMinimumWidth(70)
        row_cine_pan916.addWidget(self.label_cine_pan916_speed)
        row_cine_pan916.addStretch(1)
        cine_layout.addLayout(row_cine_pan916)

        # Options row (moved under the setting): parts slider + transparent + random
        row_cine_pan916_opts = QHBoxLayout()
        row_cine_pan916_opts.addSpacing(22)  # indent under the checkbox label
        label_cine_pan916_parts = QLabel("Parts:", self.cine_options)
        row_cine_pan916_opts.addWidget(label_cine_pan916_parts)

        self.slider_cine_pan916_parts = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_pan916_parts.setRange(2, 6)
        self.slider_cine_pan916_parts.setSingleStep(1)
        self.slider_cine_pan916_parts.setPageStep(1)
        self.slider_cine_pan916_parts.setFixedWidth(90)
        self.slider_cine_pan916_parts.setValue(3)
        self.slider_cine_pan916_parts.valueChanged.connect(self._on_cine_pan916_parts_changed)
        row_cine_pan916_opts.addWidget(self.slider_cine_pan916_parts)

        self.label_cine_pan916_parts = QLabel("3", self.cine_options)
        self.label_cine_pan916_parts.setMinimumWidth(85)
        row_cine_pan916_opts.addWidget(self.label_cine_pan916_parts)

        self.check_cine_pan916_transparent = QCheckBox("Transparent", self.cine_options)
        self.check_cine_pan916_transparent.setToolTip(
            "When enabled, the hidden area shows the same video at 50% opacity instead of solid black."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_pan916_transparent.setChecked(True)
        except Exception:
            pass
        row_cine_pan916_opts.addWidget(self.check_cine_pan916_transparent)

        self.check_cine_pan916_random = QCheckBox("Random", self.cine_options)
        self.check_cine_pan916_random.setToolTip(
            "Randomizes the slice count (2–6), starting slice, and direction each time this effect is used.\n"
            "(If you use a seed, it will stay reproducible.)"
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_pan916_random.setChecked(True)
        except Exception:
            pass
        self.check_cine_pan916_random.stateChanged.connect(self._on_cine_pan916_random_changed)
        row_cine_pan916_opts.addWidget(self.check_cine_pan916_random)
        row_cine_pan916_opts.addStretch(1)
        cine_layout.addLayout(row_cine_pan916_opts)

        # Mosaic multi-screen effect
        row_cine_mosaic = QHBoxLayout()
        self.check_cine_mosaic = QCheckBox("Mosaic multi-screen", self.cine_options)
        self.check_cine_mosaic.setToolTip(
            "Occasionally replace a segment with a grid of multiple clips from the source folder."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_mosaic.setChecked(True)
        except Exception:
            pass
        row_cine_mosaic.addWidget(self.check_cine_mosaic)
        self.check_cine_mosaic_random = QCheckBox("Random", self.cine_options)
        self.check_cine_mosaic_random.setToolTip(
            "Pick a random layout (2–9 screens) each time Mosaic is used."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_mosaic_random.setChecked(True)
        except Exception:
            pass
        row_cine_mosaic.addWidget(self.check_cine_mosaic_random)
        label_cine_mosaic = QLabel("Screens:", self.cine_options)
        row_cine_mosaic.addWidget(label_cine_mosaic)
        self.slider_cine_mosaic_screens = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_mosaic_screens.setMinimum(2)
        self.slider_cine_mosaic_screens.setMaximum(9)
        self.slider_cine_mosaic_screens.setSingleStep(1)
        self.slider_cine_mosaic_screens.setValue(4)
        row_cine_mosaic.addWidget(self.slider_cine_mosaic_screens, 1)
        self.label_cine_mosaic_screens = QLabel("4 screens", self.cine_options)
        self.label_cine_mosaic_screens.setMinimumWidth(60)
        row_cine_mosaic.addWidget(self.label_cine_mosaic_screens)
        cine_layout.addLayout(row_cine_mosaic)

        # Upside-down flip
        row_cine_flip = QHBoxLayout()
        self.check_cine_flip = QCheckBox("Upside-down flip", self.cine_options)
        self.check_cine_flip.setToolTip(
            "Occasionally flip the frame 180° for a disorienting upside-down hit."
        )
        row_cine_flip.addWidget(self.check_cine_flip)
        row_cine_flip.addStretch(1)
        cine_layout.addLayout(row_cine_flip)

        # Rotating screen hit
        row_cine_rotate = QHBoxLayout()
        self.check_cine_rotate = QCheckBox("Rotating screen hit", self.cine_options)
        self.check_cine_rotate.setToolTip(
            "Occasionally apply a short rotating-screen hit (small spin) on strong musical moments."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_rotate.setChecked(True)
        except Exception:
            pass
        row_cine_rotate.addWidget(self.check_cine_rotate)
        label_cine_rotate = QLabel("Max angle:", self.cine_options)
        row_cine_rotate.addWidget(label_cine_rotate)
        self.slider_cine_rotate_degrees = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_rotate_degrees.setMinimum(5)
        self.slider_cine_rotate_degrees.setMaximum(90)
        self.slider_cine_rotate_degrees.setSingleStep(1)
        self.slider_cine_rotate_degrees.setValue(20)
        row_cine_rotate.addWidget(self.slider_cine_rotate_degrees, 1)
        self.label_cine_rotate_degrees = QLabel("±20°", self.cine_options)
        self.label_cine_rotate_degrees.setMinimumWidth(50)
        row_cine_rotate.addWidget(self.label_cine_rotate_degrees)
        cine_layout.addLayout(row_cine_rotate)

        # Multiply: same-clip multi-screen effect
        row_cine_multiply = QHBoxLayout()
        self.check_cine_multiply = QCheckBox("Multiply (same clip)", self.cine_options)
        self.check_cine_multiply.setToolTip(
            "Occasionally split the screen into a grid that shows multiple copies of the *same* clip."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_multiply.setChecked(True)
        except Exception:
            pass
        row_cine_multiply.addWidget(self.check_cine_multiply)
        self.check_cine_multiply_random = QCheckBox("Random", self.cine_options)
        self.check_cine_multiply_random.setToolTip(
            "Pick a random layout (2–9 screens) each time Multiply is used."
        )
        # Default ON (matches UI defaults in the screenshot)
        try:
            self.check_cine_multiply_random.setChecked(True)
        except Exception:
            pass
        row_cine_multiply.addWidget(self.check_cine_multiply_random)
        label_cine_multiply = QLabel("Copies:", self.cine_options)
        row_cine_multiply.addWidget(label_cine_multiply)
        self.slider_cine_multiply_screens = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_multiply_screens.setMinimum(2)
        self.slider_cine_multiply_screens.setMaximum(9)
        self.slider_cine_multiply_screens.setSingleStep(1)
        self.slider_cine_multiply_screens.setValue(4)
        row_cine_multiply.addWidget(self.slider_cine_multiply_screens, 1)
        self.label_cine_multiply_screens = QLabel("4 copies", self.cine_options)
        self.label_cine_multiply_screens.setMinimumWidth(60)
        row_cine_multiply.addWidget(self.label_cine_multiply_screens)
        cine_layout.addLayout(row_cine_multiply)

        # Dolly-zoom / Ken Burns camera motion (backend-only; hidden from UI)
        self.check_cine_dolly = QCheckBox("Dolly-zoom hit", self.cine_options)
        self.check_cine_dolly.setToolTip(
            "Rare dolly-zoom style camera move (push in/out while zooming) for a dramatic hit.\n"
            "Hits are placed at most about once every 30 seconds."
        )
        self.slider_cine_dolly_strength = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_dolly_strength.setMinimum(10)
        self.slider_cine_dolly_strength.setMaximum(200)
        self.slider_cine_dolly_strength.setSingleStep(5)
        self.slider_cine_dolly_strength.setValue(50)
        self.label_cine_dolly_strength = QLabel("50%", self.cine_options)
        self.label_cine_dolly_strength.setMinimumWidth(40)

        self.check_cine_kenburns = QCheckBox("Ken Burns pan/zoom", self.cine_options)
        self.check_cine_kenburns.setToolTip(
            "Rare, gentle pan/zoom across a clip (Ken Burns style).\n"
            "Also limited to about one hit every 30 seconds."
        )
        self.slider_cine_kenburns_strength = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_kenburns_strength.setMinimum(10)
        self.slider_cine_kenburns_strength.setMaximum(200)
        self.slider_cine_kenburns_strength.setSingleStep(5)
        self.slider_cine_kenburns_strength.setValue(40)
        self.label_cine_kenburns_strength = QLabel("40%", self.cine_options)
        self.label_cine_kenburns_strength.setMinimumWidth(40)

        # Shared motion direction dropdown (used internally, hidden)
        self.combo_cine_motion_dir = QComboBox(self.cine_options)
        self.combo_cine_motion_dir.addItems(
            [
                "Random each hit",
                "Zoom in",
                "Zoom out",
                "Pan left",
                "Pan right",
                "Pan up",
                "Pan down",
            ]
        )
        self.combo_cine_motion_dir.setToolTip(
            "Controls the direction for dolly / Ken Burns camera moves.\n"
            "Leave on 'Random each hit' to vary direction automatically."
        )

        # Speedup (forward)
        row_cine_speedup_forward = QHBoxLayout()
        self.check_cine_speedup_forward = QCheckBox("Speedup (forward)", self.cine_options)
        self.check_cine_speedup_forward.setToolTip(
            "Speed up a segment while playing it forward. If the source clip is shorter than the segment, it will be repeated as needed."
        )
        row_cine_speedup_forward.addWidget(self.check_cine_speedup_forward)
        label_cine_speedup_forward = QLabel("Rate:", self.cine_options)
        row_cine_speedup_forward.addWidget(label_cine_speedup_forward)
        self.spin_cine_speedup_forward = QDoubleSpinBox(self.cine_options)
        self.spin_cine_speedup_forward.setDecimals(2)
        self.spin_cine_speedup_forward.setSingleStep(0.05)
        self.spin_cine_speedup_forward.setRange(1.25, 4.0)
        self.spin_cine_speedup_forward.setValue(1.5)
        self.spin_cine_speedup_forward.setSuffix("x")
        row_cine_speedup_forward.addWidget(self.spin_cine_speedup_forward)
        row_cine_speedup_forward.addStretch(1)
        cine_layout.addLayout(row_cine_speedup_forward)

        # Speedup (backwards)
        row_cine_speedup_backward = QHBoxLayout()
        self.check_cine_speedup_backward = QCheckBox("Speedup (backwards)", self.cine_options)
        self.check_cine_speedup_backward.setToolTip(
            "Speed up a segment while playing it backwards. If the source clip is shorter than the segment, it will be repeated as needed."
        )
        row_cine_speedup_backward.addWidget(self.check_cine_speedup_backward)
        label_cine_speedup_backward = QLabel("Rate:", self.cine_options)
        row_cine_speedup_backward.addWidget(label_cine_speedup_backward)
        self.spin_cine_speedup_backward = QDoubleSpinBox(self.cine_options)
        self.spin_cine_speedup_backward.setDecimals(2)
        self.spin_cine_speedup_backward.setSingleStep(0.05)
        self.spin_cine_speedup_backward.setRange(1.25, 4.0)
        self.spin_cine_speedup_backward.setValue(1.5)
        self.spin_cine_speedup_backward.setSuffix("x")
        row_cine_speedup_backward.addWidget(self.spin_cine_speedup_backward)
        row_cine_speedup_backward.addStretch(1)
        cine_layout.addLayout(row_cine_speedup_backward)

        # Speed ramps (only useful when slow-motion is enabled)
        row_cine_ramp = QHBoxLayout()
        self.check_cine_speed_ramp = QCheckBox("Use cinematic speed ramps", self.cine_options)
        self.check_cine_speed_ramp.setToolTip(
            "When slow-motion is enabled, smooth the change into and out of "
            "slow segments instead of an abrupt jump."
        )
        row_cine_ramp.addWidget(self.check_cine_speed_ramp)
        row_cine_ramp.addStretch(1)
        cine_layout.addLayout(row_cine_ramp)

        row_cine_ramp_times = QHBoxLayout()
        label_ramp_in = QLabel("Ramp in:", self.cine_options)
        row_cine_ramp_times.addWidget(label_ramp_in)
        self.slider_cine_ramp_in = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_ramp_in.setMinimum(15)    # 0.15 s
        self.slider_cine_ramp_in.setMaximum(100)   # 1.00 s
        self.slider_cine_ramp_in.setSingleStep(1)
        self.slider_cine_ramp_in.setValue(25)     # 0.25 s
        row_cine_ramp_times.addWidget(self.slider_cine_ramp_in, 1)
        self.label_cine_ramp_in = QLabel("0.25 s", self.cine_options)
        self.label_cine_ramp_in.setMinimumWidth(50)
        row_cine_ramp_times.addWidget(self.label_cine_ramp_in)

        label_ramp_out = QLabel("Ramp out:", self.cine_options)
        row_cine_ramp_times.addWidget(label_ramp_out)
        self.slider_cine_ramp_out = QSlider(Qt.Horizontal, self.cine_options)
        self.slider_cine_ramp_out.setMinimum(15)   # 0.15 s
        self.slider_cine_ramp_out.setMaximum(100)  # 0.50 s
        self.slider_cine_ramp_out.setSingleStep(1)
        self.slider_cine_ramp_out.setValue(25)    # 0.25 s
        row_cine_ramp_times.addWidget(self.slider_cine_ramp_out, 1)
        self.label_cine_ramp_out = QLabel("0.25 s", self.cine_options)
        self.label_cine_ramp_out.setMinimumWidth(50)
        row_cine_ramp_times.addWidget(self.label_cine_ramp_out)
        cine_layout.addLayout(row_cine_ramp_times)

        opts.addWidget(self.cine_options)
        # Defaults to visible because "Enable cinematic effects" defaults to ON.
        # (Saved settings can still override this during _load_settings.)
        self.cine_options.setVisible(True)

        # break impact FX options (first beat after a break)
        row_impact_master = QHBoxLayout()
        self.check_impact_enable = QCheckBox("Enhance break / drop impact", self)
        self.check_impact_enable.setToolTip(
            "Trigger strong club-style FX (flash, shockwave, colour strobe, color-cycle glitch, zoom, shake, fog, "
            "fireworks) on the very first beat after a breakdown."
        )
        row_impact_master.addWidget(self.check_impact_enable)
        row_impact_master.addStretch(1)
        opts.addLayout(row_impact_master)

        self.impact_options = QWidget(self)
        impact_layout = QVBoxLayout(self.impact_options)
        impact_layout.setContentsMargins(26, 0, 0, 0)
        impact_layout.setSpacing(2)

        # Quick toggle: turn all break/drop impact toggles on/off at once
        row_impact_all = QHBoxLayout()
        self.btn_impact_all = QPushButton("All on/off", self.impact_options)
        self.btn_impact_all.setToolTip("Turn ON/OFF all toggles in the break/drop impact section.")
        row_impact_all.addWidget(self.btn_impact_all)
        row_impact_all.addStretch(1)
        impact_layout.addLayout(row_impact_all)

        # Flash strobe
        row_impact_flash = QHBoxLayout()
        self.check_impact_flash = QCheckBox("Flash strobe", self.impact_options)
        row_impact_flash.addWidget(self.check_impact_flash)
        label_impact_flash = QLabel("Strength:", self.impact_options)
        row_impact_flash.addWidget(label_impact_flash)
        self.slider_impact_flash = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_flash.setMinimum(10)
        self.slider_impact_flash.setMaximum(150)
        self.slider_impact_flash.setSingleStep(1)
        self.slider_impact_flash.setValue(80)
        row_impact_flash.addWidget(self.slider_impact_flash, 1)
        self.label_impact_flash = QLabel("0.80", self.impact_options)
        self.label_impact_flash.setMinimumWidth(40)
        row_impact_flash.addWidget(self.label_impact_flash)
        impact_layout.addLayout(row_impact_flash)

        # Flash strobe speed (ms per flash)
        row_impact_flash_speed = QHBoxLayout()
        row_impact_flash_speed.addSpacing(22)  # indent under the checkbox label
        label_impact_flash_speed = QLabel("Speed (ms):", self.impact_options)
        row_impact_flash_speed.addWidget(label_impact_flash_speed)
        self.slider_impact_flash_speed = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_flash_speed.setMinimum(100)
        self.slider_impact_flash_speed.setMaximum(1000)
        self.slider_impact_flash_speed.setSingleStep(50)
        self.slider_impact_flash_speed.setPageStep(100)
        self.slider_impact_flash_speed.setValue(250)
        row_impact_flash_speed.addWidget(self.slider_impact_flash_speed, 1)
        self.label_impact_flash_speed = QLabel("250 ms", self.impact_options)
        self.label_impact_flash_speed.setMinimumWidth(60)
        row_impact_flash_speed.addWidget(self.label_impact_flash_speed)
        impact_layout.addLayout(row_impact_flash_speed)

        # Shockwave
        row_impact_shock = QHBoxLayout()
        self.check_impact_shock = QCheckBox("Shockwave burst", self.impact_options)
        row_impact_shock.addWidget(self.check_impact_shock)
        label_impact_shock = QLabel("Strength:", self.impact_options)
        row_impact_shock.addWidget(label_impact_shock)
        self.slider_impact_shock = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_shock.setMinimum(10)
        self.slider_impact_shock.setMaximum(150)
        self.slider_impact_shock.setSingleStep(1)
        self.slider_impact_shock.setValue(75)
        row_impact_shock.addWidget(self.slider_impact_shock, 1)
        self.label_impact_shock = QLabel("0.75", self.impact_options)
        self.label_impact_shock.setMinimumWidth(40)
        row_impact_shock.addWidget(self.label_impact_shock)
        impact_layout.addLayout(row_impact_shock)

        
        # Echo trail (impact smear)
        row_impact_echo = QHBoxLayout()
        self.check_impact_echo = QCheckBox("Echo trail (smear)", self.impact_options)
        self.check_impact_echo.setToolTip(
            "Blend a short trail of previous frames on the first beat after a breakdown."
        )
        row_impact_echo.addWidget(self.check_impact_echo)
        label_impact_echo = QLabel("Strength:", self.impact_options)
        row_impact_echo.addWidget(label_impact_echo)
        self.slider_impact_echo = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_echo.setMinimum(10)
        self.slider_impact_echo.setMaximum(100)
        self.slider_impact_echo.setSingleStep(1)
        self.slider_impact_echo.setValue(70)
        row_impact_echo.addWidget(self.slider_impact_echo, 1)
        self.label_impact_echo = QLabel("0.70", self.impact_options)
        self.label_impact_echo.setMinimumWidth(40)
        row_impact_echo.addWidget(self.label_impact_echo)
        impact_layout.addLayout(row_impact_echo)

# Beat ripple (radial shockwave distortion for drop/break hits)
        row_impact_confetti = QHBoxLayout()
        self.check_impact_confetti = QCheckBox("Beat ripple", self.impact_options)
        self.check_impact_confetti.setToolTip(
            "Radial shockwave distortion for drop/break hits.\n"
            "Short pulse: ~6–12 frames (peak around frames 3–4 @ 30fps)."
        )

        row_impact_confetti.addWidget(self.check_impact_confetti)
        label_impact_confetti = QLabel("Strength:", self.impact_options)
        row_impact_confetti.addWidget(label_impact_confetti)
        self.slider_impact_confetti = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_confetti.setMinimum(10)
        self.slider_impact_confetti.setMaximum(100)
        self.slider_impact_confetti.setSingleStep(1)
        self.slider_impact_confetti.setValue(70)
        row_impact_confetti.addWidget(self.slider_impact_confetti, 1)
        self.label_impact_confetti = QLabel("0.70", self.impact_options)
        self.label_impact_confetti.setMinimumWidth(40)
        row_impact_confetti.addWidget(self.label_impact_confetti)
        impact_layout.addLayout(row_impact_confetti)

        # Color-cycle glitch (segment-wide hue cycling)
        row_impact_colorcycle = QHBoxLayout()
        self.check_impact_colorcycle = QCheckBox("Color-cycle glitch (segment)", self.impact_options)
        self.check_impact_colorcycle.setToolTip(
            "Applies a continuous hue-cycle across the *entire* impact segment.\n"
            "This is meant for drop/chorus moments (not as a transition)."
        )
        row_impact_colorcycle.addWidget(self.check_impact_colorcycle)
        label_impact_colorcycle = QLabel("Speed:", self.impact_options)
        row_impact_colorcycle.addWidget(label_impact_colorcycle)
        self.slider_impact_colorcycle = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_colorcycle.setMinimum(10)
        self.slider_impact_colorcycle.setMaximum(200)
        self.slider_impact_colorcycle.setSingleStep(1)
        self.slider_impact_colorcycle.setValue(70)
        row_impact_colorcycle.addWidget(self.slider_impact_colorcycle, 1)
        self.label_impact_colorcycle = QLabel("0.70", self.impact_options)
        self.label_impact_colorcycle.setMinimumWidth(40)
        row_impact_colorcycle.addWidget(self.label_impact_colorcycle)
        impact_layout.addLayout(row_impact_colorcycle)

        # Zoom punch
        row_impact_zoom = QHBoxLayout()
        self.check_impact_zoom = QCheckBox("Zoom punch-in", self.impact_options)
        row_impact_zoom.addWidget(self.check_impact_zoom)
        label_impact_zoom = QLabel("Amount:", self.impact_options)
        row_impact_zoom.addWidget(label_impact_zoom)
        self.slider_impact_zoom = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_zoom.setMinimum(5)
        self.slider_impact_zoom.setMaximum(50)
        self.slider_impact_zoom.setSingleStep(1)
        self.slider_impact_zoom.setValue(20)
        row_impact_zoom.addWidget(self.slider_impact_zoom, 1)
        self.label_impact_zoom = QLabel("20%", self.impact_options)
        self.label_impact_zoom.setMinimumWidth(40)
        row_impact_zoom.addWidget(self.label_impact_zoom)
        impact_layout.addLayout(row_impact_zoom)

        # Camera shake
        row_impact_shake = QHBoxLayout()
        self.check_impact_shake = QCheckBox("Camera shake", self.impact_options)
        row_impact_shake.addWidget(self.check_impact_shake)
        label_impact_shake = QLabel("Strength:", self.impact_options)
        row_impact_shake.addWidget(label_impact_shake)
        self.slider_impact_shake = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_shake.setMinimum(10)
        self.slider_impact_shake.setMaximum(150)
        self.slider_impact_shake.setSingleStep(1)
        self.slider_impact_shake.setValue(60)
        row_impact_shake.addWidget(self.slider_impact_shake, 1)
        self.label_impact_shake = QLabel("0.60", self.impact_options)
        self.label_impact_shake.setMinimumWidth(40)
        row_impact_shake.addWidget(self.label_impact_shake)
        impact_layout.addLayout(row_impact_shake)

        # Fog blast
        row_impact_fog = QHBoxLayout()
        self.check_impact_fog = QCheckBox("Fog / CO₂ blast", self.impact_options)
        row_impact_fog.addWidget(self.check_impact_fog)
        label_impact_fog = QLabel("Density:", self.impact_options)
        row_impact_fog.addWidget(label_impact_fog)
        self.slider_impact_fog = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_fog.setMinimum(10)
        self.slider_impact_fog.setMaximum(100)
        self.slider_impact_fog.setSingleStep(1)
        self.slider_impact_fog.setValue(65)
        row_impact_fog.addWidget(self.slider_impact_fog, 1)
        self.label_impact_fog = QLabel("0.65", self.impact_options)
        self.label_impact_fog.setMinimumWidth(40)
        row_impact_fog.addWidget(self.label_impact_fog)
        impact_layout.addLayout(row_impact_fog)

        # Hide legacy break-impact FX (fog blast) while keeping state + settings support.
        self.check_impact_fog.setChecked(False)
        for w in (
            self.check_impact_fog,
            label_impact_fog,
            self.slider_impact_fog,
            self.label_impact_fog,
        ):
            w.hide()

        # Fireworks (gold)
        row_impact_fire_gold = QHBoxLayout()
        self.check_impact_fire_gold = QCheckBox("Fireworks (gold)", self.impact_options)
        row_impact_fire_gold.addWidget(self.check_impact_fire_gold)
        label_impact_fire_gold = QLabel("Intensity:", self.impact_options)
        row_impact_fire_gold.addWidget(label_impact_fire_gold)
        self.slider_impact_fire_gold = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_fire_gold.setMinimum(10)
        self.slider_impact_fire_gold.setMaximum(100)
        self.slider_impact_fire_gold.setSingleStep(1)
        self.slider_impact_fire_gold.setValue(75)
        row_impact_fire_gold.addWidget(self.slider_impact_fire_gold, 1)
        self.label_impact_fire_gold = QLabel("0.75", self.impact_options)
        self.label_impact_fire_gold.setMinimumWidth(40)
        row_impact_fire_gold.addWidget(self.label_impact_fire_gold)
        impact_layout.addLayout(row_impact_fire_gold)

        # Hide legacy break-impact FX (gold fireworks) while keeping state + settings support.
        self.check_impact_fire_gold.setChecked(False)
        for w in (
            self.check_impact_fire_gold,
            label_impact_fire_gold,
            self.slider_impact_fire_gold,
            self.label_impact_fire_gold,
        ):
            w.hide()

        # Fireworks (multicolor)
        row_impact_fire_multi = QHBoxLayout()
        self.check_impact_fire_multi = QCheckBox("Fireworks (multicolor)", self.impact_options)
        row_impact_fire_multi.addWidget(self.check_impact_fire_multi)
        label_impact_fire_multi = QLabel("Intensity:", self.impact_options)
        row_impact_fire_multi.addWidget(label_impact_fire_multi)
        self.slider_impact_fire_multi = QSlider(Qt.Horizontal, self.impact_options)
        self.slider_impact_fire_multi.setMinimum(10)
        self.slider_impact_fire_multi.setMaximum(100)
        self.slider_impact_fire_multi.setSingleStep(1)
        self.slider_impact_fire_multi.setValue(80)
        row_impact_fire_multi.addWidget(self.slider_impact_fire_multi, 1)
        self.label_impact_fire_multi = QLabel("0.80", self.impact_options)
        self.label_impact_fire_multi.setMinimumWidth(40)
        row_impact_fire_multi.addWidget(self.label_impact_fire_multi)
        impact_layout.addLayout(row_impact_fire_multi)

        # Hide legacy break-impact FX (multicolor fireworks) while keeping state + settings support.
        self.check_impact_fire_multi.setChecked(False)
        for w in (
            self.check_impact_fire_multi,
            label_impact_fire_multi,
            self.slider_impact_fire_multi,
            self.label_impact_fire_multi,
        ):
            w.hide()

        # Random choice mode
        row_impact_random = QHBoxLayout()
        self.check_impact_random = QCheckBox("Random (choose 1 per break)", self.impact_options)
        row_impact_random.addWidget(self.check_impact_random)
        row_impact_random.addStretch(1)
        impact_layout.addLayout(row_impact_random)

        opts.addWidget(self.impact_options)
        self.impact_options.setVisible(False)

        # Timed strobe: trigger the Flash strobe at custom time(s) in the song/video timeline.
        row_strobe_on_time = QHBoxLayout()
        self.check_strobe_on_time = QCheckBox("Strobe on time", self)
        self.check_strobe_on_time.setToolTip(
            "When enabled, you can add one or more timestamps (seconds) where the Flash strobe should fire."
        )
        row_strobe_on_time.addWidget(self.check_strobe_on_time)
        row_strobe_on_time.addStretch(1)
        opts.addLayout(row_strobe_on_time)

        self.strobe_time_options = QWidget(self)
        strobe_layout = QVBoxLayout(self.strobe_time_options)
        strobe_layout.setContentsMargins(26, 0, 0, 0)
        strobe_layout.setSpacing(4)

        row_strobe_add = QHBoxLayout()
        label_strobe_time = QLabel("Time (s):", self.strobe_time_options)
        row_strobe_add.addWidget(label_strobe_time)

        self.spin_strobe_time = QDoubleSpinBox(self.strobe_time_options)
        self.spin_strobe_time.setDecimals(2)
        self.spin_strobe_time.setSingleStep(0.25)
        self.spin_strobe_time.setRange(0.0, 99999.0)
        self.spin_strobe_time.setValue(0.0)
        row_strobe_add.addWidget(self.spin_strobe_time)

        self.btn_strobe_time_add = QPushButton("Add", self.strobe_time_options)
        row_strobe_add.addWidget(self.btn_strobe_time_add)

        self.btn_strobe_time_remove = QPushButton("Remove selected", self.strobe_time_options)
        row_strobe_add.addWidget(self.btn_strobe_time_remove)

        self.btn_strobe_time_clear = QPushButton("Remove all", self.strobe_time_options)
        row_strobe_add.addWidget(self.btn_strobe_time_clear)

        row_strobe_add.addStretch(1)
        strobe_layout.addLayout(row_strobe_add)

        self.list_strobe_times = QListWidget(self.strobe_time_options)
        self.list_strobe_times.setToolTip("Strobe timestamps (seconds).")
        self.list_strobe_times.setMaximumHeight(110)
        strobe_layout.addWidget(self.list_strobe_times)

        opts.addWidget(self.strobe_time_options)
        self.strobe_time_options.setVisible(False)

        # Master FX kill-switch
        row_nofx = QHBoxLayout()
        self.check_nofx = QCheckBox("Disable ALL video FX (No FX mode)", self)
        self.check_nofx.setToolTip(
            "When enabled, all video effects are disabled: no slow motion, no cinematic FX, "
            "no break-impact FX and only clean hard cuts between clips."
        )
        row_nofx.addWidget(self.check_nofx)
        row_nofx.addStretch(1)
        opts.addLayout(row_nofx)


        # Music-player visuals overlay (optional)
        row_viz = QHBoxLayout()
        self.check_visual_overlay = QCheckBox("Add music-player visuals overlay", self)
        self.check_visual_overlay.setToolTip(
            "Render one of your beat synced visuals as an overlay layer on the final clip. "
             "Warning : Every toggle in here will MORE then double the generation time of your finished video !"
        )
        row_viz.addWidget(self.check_visual_overlay)
        row_viz.addStretch(1)
        opts.addLayout(row_viz)

        # Music-player visuals strategies (shown only when overlay is enabled)
        self.viz_options_container = QWidget(self)
        viz_layout = QVBoxLayout(self.viz_options_container)
        viz_layout.setContentsMargins(32, 0, 0, 0)

        self.check_visual_strategy_segment = QCheckBox(
            "New random visual every segment", self.viz_options_container
        )
        self.check_visual_strategy_section = QCheckBox(
            "Different visual per section type (intro / verse / chorus / break / drop / outro)",
            self.viz_options_container,
        )

        viz_layout.addWidget(self.check_visual_strategy_segment)
        viz_layout.addWidget(self.check_visual_strategy_section)

        row_viz_section_select = QHBoxLayout()
        self.btn_visual_section_select = QPushButton("Select my own…", self.viz_options_container)
        self.btn_visual_section_select.setEnabled(False)
        row_viz_section_select.addWidget(self.btn_visual_section_select)
        row_viz_section_select.addStretch(1)
        viz_layout.addLayout(row_viz_section_select)

        self.label_visual_section_choices = QLabel("", self.viz_options_container)
        self.label_visual_section_choices.setWordWrap(True)
        viz_layout.addWidget(self.label_visual_section_choices)

        # Visual overlay opacity controls
        row_viz_opacity = QHBoxLayout()
        lbl_viz_opacity = QLabel("Visual overlay opacity:", self.viz_options_container)
        row_viz_opacity.addWidget(lbl_viz_opacity)

        self.slider_visual_opacity = QSlider(Qt.Horizontal, self.viz_options_container)
        self.slider_visual_opacity.setMinimum(10)   # 0.10
        self.slider_visual_opacity.setMaximum(100)  # 1.00
        self.slider_visual_opacity.setSingleStep(5)
        self.slider_visual_opacity.setPageStep(10)
        self.slider_visual_opacity.setValue(int(self.visual_overlay_opacity * 100.0))
        self.slider_visual_opacity.setToolTip(
            "Controls how strong the music-player visuals appear on top of the video.\n"
            "Lower = more transparent, Higher = more opaque."
        )
        row_viz_opacity.addWidget(self.slider_visual_opacity, 1)

        self.spin_visual_opacity = QDoubleSpinBox(self.viz_options_container)
        self.spin_visual_opacity.setRange(0.10, 1.00)
        self.spin_visual_opacity.setSingleStep(0.05)
        self.spin_visual_opacity.setDecimals(2)
        self.spin_visual_opacity.setValue(float(self.visual_overlay_opacity))
        self.spin_visual_opacity.setToolTip(
            "Controls how strong the music-player visuals appear on top of the video.\n"
            "Lower = more transparent, Higher = more opaque."
        )
        row_viz_opacity.addWidget(self.spin_visual_opacity)

        viz_layout.addLayout(row_viz_opacity)

        self.slider_visual_opacity.valueChanged.connect(self._on_visual_opacity_slider_changed)
        self.spin_visual_opacity.valueChanged.connect(self._on_visual_opacity_spin_changed)

        self.viz_options_container.setVisible(False)
        opts.addWidget(self.viz_options_container)

        # transitions
        row_trans = QHBoxLayout()
        self.label_transitions = QLabel("Transitions:", self)
        row_trans.addWidget(self.label_transitions)
        self.combo_transitions = QComboBox(self)
        # High-level transition presets (indices must stay stable)
        self.combo_transitions.addItems(
            [
                "Soft film dissolves",
                "Hard cuts",
                "Scale punch (zoom)",
                "Shimmer blur (shiny)",
                "Iris reveal (circle)",
                "Motion blur whip-cuts",
                "Slit-scan smear push",
                "Radial burst reveal",
                "Directional push (slide)",
                "Wipe",
                "Smooth zoom crossfade",
                "Curtain open (doors)",
                "Pixelize",
                "Distance (liquid blend)",
                "Wind smears",
            ]
        )
        # Default transition preset: Hard cuts
        self.combo_transitions.setCurrentIndex(1)
        self.combo_transitions.setToolTip(
            "High-level transition look:\n"
            "\n"
            "- Soft film dissolves: gentle, low-key transitions (currently clean cuts).\n"
            "- Hard cuts: straight, no-frills cuts.\n"
            "- Scale punch (zoom): punchy zoom-burst crossfade with a tiny flash.\n"
            "- Shimmer blur (shiny): fast blurred crossfade (glossy, editor-style).\n"
            "- Iris reveal (circle): real stitched iris transition between clips.\n"
            "- Motion blur whip-cuts: extra blur on fast cuts.\n"
            "- Slit-scan smear push: modern smear + push (real stitched transition).\n"
            "- Radial burst reveal: quick expanding burst reveal (real stitched transition).\n"
            "- Directional push: smooth slide-style motion across the cut.\n"
            "- Wipe: real between-clip wipe (stitched transition).\n"
            "- Smooth zoom crossfade: continuous, gentle zoom across the cut.\n"
            "- Curtain open (doors): center-opening curtain/doors (real stitched transition).\n"
            "- Pixelize: chunky pixel-block transition (real stitched transition).\n"
            "- Distance (liquid blend): pseudo-morph / liquid blend feel (real stitched transition).\n"
            "- Wind smears: wind-like streaky crossfade (real stitched transition)."
        )
        row_trans.addWidget(self.combo_transitions, 1)
        row_trans.addStretch(1)
        opts.addLayout(row_trans)

        
        # random transitions toggle + manager
        row_trans_ctrl = QHBoxLayout()
        self.check_trans_random = QCheckBox("Random transitions", self)
        self.check_trans_random.setToolTip(
            "When enabled, each segment will pick a transition style at random\n"
            "from the enabled list below. When disabled, the selected style\n"
            "in the dropdown is used for all segments."
        )
        row_trans_ctrl.addWidget(self.check_trans_random)
        btn_manage_trans = QPushButton("Manage transitions...", self)
        btn_manage_trans.setToolTip(
            "Choose which transition styles are allowed when 'Random transitions'\n"
            "is enabled (Slide, Hard cuts, Zoom pulse, Iris, Motion blur, Slit-scan), Creative mix."
        )
        row_trans_ctrl.addWidget(btn_manage_trans)
        row_trans_ctrl.addStretch(1)
        opts.addLayout(row_trans_ctrl)


        main.addWidget(self.box_opts)

        # advanced
        self.box_adv = QGroupBox("Advanced", self)
        self.box_adv.setCheckable(True)
        self.box_adv.setChecked(True)
        adv = QFormLayout(self.box_adv)
        adv.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        adv.setSpacing(4)

        self.slider_sens = QSlider(Qt.Horizontal, self)
        self.slider_sens.setMinimum(10)   # represents 1.0
        self.slider_sens.setMaximum(200)  # represents 20.0 (i.e. 2.0 steps *10)
        self.slider_sens.setValue(50)     # default 5.0
        self.slider_sens.setSingleStep(5) # 0.5 represented as +5
        
        self.slider_sens.setMinimum(2)
        self.slider_sens.setMaximum(20)
        self.slider_sens.setValue(10)
        self.slider_sens.setToolTip(
            "Beat sensitivity for the detector (0.5–20.0 internal scale).\n"
            "Lower = fewer beats (stricter), higher = more beats (looser)."
        )
        adv.addRow("Beat sensitivity:", self.slider_sens)

        self.spin_beats_per_seg = QSpinBox(self)
        self.spin_beats_per_seg.setRange(1, 256)
        self.spin_beats_per_seg.setValue(8)
        self.spin_beats_per_seg.setToolTip(
            "Number of beats grouped into one base video segment.\n"
            "1 = very fast cuts, 4 = default, 8+ = slower cuts. You can go up to 256 when you want long clips in very fast music (eg. DnB 170bpm"
        )
        adv.addRow("Beats per base segment:", self.spin_beats_per_seg)

        main.addWidget(self.box_adv)

        # progress + buttons
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("Ready.")
        main.addWidget(self.progress)

        # compact analysis summary
        self.label_summary = QLabel("", self)
        self.label_summary.setWordWrap(True)
        main.addWidget(self.label_summary)

        # shortcut button to jump directly to the Music timeline tab
        self.btn_check_timeline = QPushButton("Check timeline", self)
        self.btn_check_timeline.setVisible(False)
        main.addWidget(self.btn_check_timeline)

        # footer button bar (embedded by default; can be 'sticky' when wrapped in a scroll area)
        self.footer_bar = QWidget(self)
        self.footer_bar.setObjectName("MusicClipCreatorFooter")
        footer_outer = QVBoxLayout(self.footer_bar)
        footer_outer.setContentsMargins(0, 0, 0, 0)
        footer_outer.setSpacing(4)

        footer_row_top = QHBoxLayout()
        footer_row_top.setContentsMargins(0, 0, 0, 0)
        footer_row_top.setSpacing(6)

        footer_row_bottom = QHBoxLayout()
        footer_row_bottom.setContentsMargins(0, 0, 0, 0)
        footer_row_bottom.setSpacing(6)

        self.btn_analyze = QPushButton("Analyze", self.footer_bar)
        self.btn_generate = QPushButton("Generate Clip", self.footer_bar)

        self.btn_queue = QPushButton("Queue Clip", self.footer_bar)
        try:
            self.btn_queue.setToolTip("Queue this Music Clip Creator job to the Queue tab instead of rendering immediately.")
        except Exception:
            pass

        # Hover style for the "Generate Clip" button: match the top banner gradient.
        try:
            self.btn_generate.setObjectName("mvGenerate")
            self.btn_generate.setStyleSheet(
                "#mvGenerate:hover {"
                " color: white;"
                " border: 1px solid rgba(255, 255, 255, 70);"
                " background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                "   stop:0 #cd28ff, stop:0.5 #9f4df2, stop:1 #28ffbb);"
                "}"
                "#mvGenerate:pressed {"
                " color: white;"
                " border: 1px solid rgba(255, 255, 255, 90);"
                " background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
                "   stop:0 #b020e6, stop:0.5 #8440d6, stop:1 #20d6a0);"
                "}"
            )
        except Exception:
            pass

        # New: View results (open Media Explorer on output folder)
        self.btn_view_results = QPushButton("View results", self.footer_bar)
        try:
            self.btn_view_results.setToolTip("Open Media Explorer on the output folder and scan for generated clips.")
        except Exception:
            pass

        # New: 1-click presets button
        self.btn_presets = QPushButton("1-click presets", self.footer_bar)
        self.btn_edit_presets = QPushButton("Edit presets", self.footer_bar)
        self.btn_edit_presets.setToolTip("Open preset manager (edit clip_presets.json).\nThis never starts a videoclip run.")

        
        self.btn_presets.setToolTip(
            "Open 1-click presets for FX and timing.\n"
            "Presets do NOT change which clips, music track, output folder\n"
            "or resolution you selected – set those first."
        )

        self.btn_cancel = QPushButton("Cancel", self.footer_bar)
        # Hidden in UI: keep handler wired but don't show the button.
        self.btn_cancel.hide()
        self.btn_reset_all = QPushButton("Reset all", self.footer_bar)
        self.btn_reset_all.setToolTip(
            "Stop any running job, clear loaded sources and reset all settings to defaults."
        )

        # Row 1: main actions
        footer_row_top.addWidget(self.btn_analyze)
        footer_row_top.addWidget(self.btn_generate)
        try:
            footer_row_top.addWidget(self.btn_queue)
        except Exception:
            pass
        footer_row_top.addStretch(1)

        # Row 2: secondary actions
        footer_row_bottom.addWidget(self.btn_view_results)
        footer_row_bottom.addWidget(self.btn_presets)
        footer_row_bottom.addWidget(self.btn_edit_presets)
        footer_row_bottom.addWidget(self.btn_cancel)
        footer_row_bottom.addWidget(self.btn_reset_all)
        footer_row_bottom.addStretch(1)

        footer_outer.addLayout(footer_row_top)
        footer_outer.addLayout(footer_row_bottom)


        if not self._sticky_footer:
            main.addWidget(self.footer_bar)
        main.addStretch(1)


        # Install tabs
        self.tabs.addTab(page_main, "Generator")

        # Music timeline tab lives in a separate helper module so it can grow
        # without bloating this file too much.
        if TimelinePanel is not None:
            self.timeline_panel = TimelinePanel(self)
        else:
            self.timeline_panel = QWidget(self)
            tl = QVBoxLayout(self.timeline_panel)
            lbl = QLabel(
                "Timeline view will show the analyzed music structure here once available.",
                self.timeline_panel,
            )
            lbl.setWordWrap(True)
            tl.addWidget(lbl)
            tl.addStretch(1)
        self.tabs.addTab(self.timeline_panel, "Music timeline")

        # Connect mini-timeline actions to this widget (if available)
        if TimelinePanel is not None and isinstance(self.timeline_panel, TimelinePanel):
            try:
                self.timeline_panel.segmentAddRequested.connect(self._on_timeline_add)  # type: ignore[attr-defined]
                self.timeline_panel.segmentRemoveRequested.connect(self._on_timeline_remove)  # type: ignore[attr-defined]
                self.timeline_panel.segmentSelected.connect(self._on_timeline_select)  # type: ignore[attr-defined]
            except Exception:
                # Never let optional timeline wiring break the main tool.
                pass

        # connections
        btn_a.clicked.connect(self._browse_audio)
        btn_vf.clicked.connect(self._browse_video_file)
        btn_vd.clicked.connect(self._browse_video_dir)
        btn_vmf.clicked.connect(self._browse_video_files)
        btn_o.clicked.connect(self._browse_output)
        self.btn_analyze.clicked.connect(self._on_analyze)
        self.btn_generate.clicked.connect(self._on_generate)
        try:
            self.btn_queue.clicked.connect(self._on_queue_generate)
        except Exception:
            pass
        try:
            self.btn_view_results.clicked.connect(self._open_output_in_media_explorer)
        except Exception:
            pass
        self.btn_presets.clicked.connect(self._on_presets_clicked)
        try:
            self.btn_edit_presets.clicked.connect(self._on_edit_presets_clicked)
        except Exception:
            pass
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_reset_all.clicked.connect(self._on_reset_all)
        self.btn_check_timeline.clicked.connect(self._on_check_timeline)
        btn_manage_trans.clicked.connect(self._on_manage_transitions)

        self.check_micro_chorus.stateChanged.connect(self._on_micro_mode_changed)
        self.check_micro_all.stateChanged.connect(self._on_micro_mode_changed)
        if hasattr(self, "check_micro_verses"):
            self.check_micro_verses.stateChanged.connect(self._on_micro_mode_changed)
        self.check_use_seed.stateChanged.connect(self._on_seed_toggle)
        self.check_trans_random.stateChanged.connect(self._on_trans_random_toggled)
        self.check_slow_enable.stateChanged.connect(self._on_slow_toggle)
        self.slider_slow_factor.valueChanged.connect(self._on_slow_factor_changed)

        # Cinematic effects panel
        self.check_cine_enable.stateChanged.connect(self._on_cine_toggle)
        self.check_cine_enable.toggled.connect(self.cine_options.setVisible)
        self.slider_cine_freeze_len.valueChanged.connect(self._on_cine_freeze_len_changed)
        self.slider_cine_freeze_zoom.valueChanged.connect(self._on_cine_freeze_zoom_changed)
        self.slider_cine_tear_v_strength.valueChanged.connect(self._on_cine_tear_v_strength_changed)
        self.slider_cine_tear_h_strength.valueChanged.connect(self._on_cine_tear_h_strength_changed)
        self.slider_cine_color_cycle_speed.valueChanged.connect(self._on_cine_color_cycle_speed_changed)
        self.slider_cine_reverse_len.valueChanged.connect(self._on_cine_reverse_len_changed)
        self.slider_cine_ramp_in.valueChanged.connect(self._on_cine_ramp_in_changed)
        self.slider_cine_ramp_out.valueChanged.connect(self._on_cine_ramp_out_changed)
        self.slider_cine_boomerang_bounces.valueChanged.connect(self._on_cine_boomerang_bounces_changed)
        self.slider_cine_pan916_speed.valueChanged.connect(self._on_cine_pan916_speed_changed)
        self.slider_cine_mosaic_screens.valueChanged.connect(self._on_cine_mosaic_screens_changed)
        self.check_cine_mosaic_random.stateChanged.connect(self._on_cine_mosaic_random_changed)
        self.slider_cine_rotate_degrees.valueChanged.connect(self._on_cine_rotate_degrees_changed)
        self.slider_cine_multiply_screens.valueChanged.connect(self._on_cine_multiply_screens_changed)
        self.check_cine_multiply_random.stateChanged.connect(self._on_cine_multiply_random_changed)
        self.slider_cine_dolly_strength.valueChanged.connect(self._on_cine_dolly_strength_changed)
        self.slider_cine_kenburns_strength.valueChanged.connect(self._on_cine_kenburns_strength_changed)

        # Break impact FX panel
        self.check_impact_enable.toggled.connect(self.impact_options.setVisible)
        self.slider_impact_flash.valueChanged.connect(self._on_impact_flash_changed)
        self.slider_impact_flash_speed.valueChanged.connect(self._on_impact_flash_speed_changed)
        self.slider_impact_shock.valueChanged.connect(self._on_impact_shock_changed)
        self.slider_impact_echo.valueChanged.connect(self._on_impact_echo_changed)
        self.slider_impact_confetti.valueChanged.connect(self._on_impact_confetti_changed)
        self.slider_impact_colorcycle.valueChanged.connect(self._on_impact_colorcycle_changed)
        self.slider_impact_zoom.valueChanged.connect(self._on_impact_zoom_changed)
        self.slider_impact_shake.valueChanged.connect(self._on_impact_shake_changed)
        self.slider_impact_fog.valueChanged.connect(self._on_impact_fog_changed)
        self.slider_impact_fire_gold.valueChanged.connect(self._on_impact_fire_gold_changed)
        self.slider_impact_fire_multi.valueChanged.connect(self._on_impact_fire_multi_changed)

        

        # All on/off buttons for the two FX sub-sections
        try:
            self.btn_cine_all.clicked.connect(self._on_cine_all_onoff)
        except Exception:
            pass
        try:
            self.btn_impact_all.clicked.connect(self._on_impact_all_onoff)
        except Exception:
            pass

        # When the user explicitly enables any FX-related option, automatically
        # release the master 'No FX' switch so effects become visible again.
        self.combo_fx.currentIndexChanged.connect(lambda _i: self._maybe_release_nofx())
        self.combo_transitions.currentIndexChanged.connect(lambda _i: self._maybe_release_nofx())
        self.check_slow_enable.stateChanged.connect(lambda s: self._maybe_release_nofx() if s else None)
        self.check_cine_enable.stateChanged.connect(lambda s: self._maybe_release_nofx() if s else None)
        self.check_impact_enable.stateChanged.connect(lambda s: self._maybe_release_nofx() if s else None)
# Global 'No FX' master switch
        self.check_nofx.stateChanged.connect(self._on_nofx_toggled)
        # Timed strobe signals
        try:
            self.check_strobe_on_time.stateChanged.connect(self._on_strobe_on_time_toggle)
        except Exception:
            pass
        try:
            self.btn_strobe_time_add.clicked.connect(self._on_strobe_time_add)
            self.btn_strobe_time_remove.clicked.connect(self._on_strobe_time_remove_selected)
            self.btn_strobe_time_clear.clicked.connect(self._on_strobe_time_clear_all)
        except Exception:
            pass

        # If visuals overlay or its strategies are enabled, automatically release No FX.
        if hasattr(self, "check_visual_overlay"):
            self.check_visual_overlay.stateChanged.connect(lambda s: self._maybe_release_nofx() if s else None)
        if hasattr(self, "check_visual_strategy_segment"):
            self.check_visual_strategy_segment.stateChanged.connect(lambda s: self._maybe_release_nofx() if s else None)
        if hasattr(self, "check_visual_strategy_section"):
            self.check_visual_strategy_section.stateChanged.connect(lambda s: self._maybe_release_nofx() if s else None)

        # Keep visual strategy options mutually exclusive and only visible when overlay is enabled.
        if hasattr(self, "check_visual_overlay") and hasattr(self, "viz_options_container"):

            def _update_visual_strategy_visibility(checked: bool) -> None:
                self.viz_options_container.setVisible(bool(checked))
                if not checked:
                    # When overlay is turned off, clear strategy toggles.
                    try:
                        self.check_visual_strategy_segment.setChecked(False)
                    except Exception:
                        pass
                    try:
                        self.check_visual_strategy_section.setChecked(False)
                    except Exception:
                        pass

            self.check_visual_overlay.toggled.connect(_update_visual_strategy_visibility)

        if hasattr(self, "check_visual_strategy_segment") and hasattr(self, "check_visual_strategy_section"):

            def _on_segment_strategy_toggled(checked: bool) -> None:
                if checked:
                    try:
                        self.check_visual_strategy_section.setChecked(False)
                    except Exception:
                        pass
                if hasattr(self, "btn_visual_section_select"):
                    try:
                        self.btn_visual_section_select.setEnabled(
                            bool(getattr(self, "check_visual_strategy_section", None))
                            and bool(self.check_visual_strategy_section.isChecked())
                        )
                    except Exception:
                        self.btn_visual_section_select.setEnabled(False)

            def _on_section_strategy_toggled(checked: bool) -> None:
                if checked:
                    try:
                        self.check_visual_strategy_segment.setChecked(False)
                    except Exception:
                        pass
                if hasattr(self, "btn_visual_section_select"):
                    try:
                        self.btn_visual_section_select.setEnabled(bool(checked))
                    except Exception:
                        self.btn_visual_section_select.setEnabled(False)

            self.check_visual_strategy_segment.toggled.connect(_on_segment_strategy_toggled)
            self.check_visual_strategy_section.toggled.connect(_on_section_strategy_toggled)

            if hasattr(self, "btn_visual_section_select"):
                try:
                    self.btn_visual_section_select.clicked.connect(self._on_select_visuals_per_section)
                except Exception:
                    pass

            # Ensure initial enabled state of the button matches current strategy.
            try:
                _on_section_strategy_toggled(self.check_visual_strategy_section.isChecked())
            except Exception:
                pass


    def _update_visual_section_summary(self) -> None:
        label = getattr(self, "label_visual_section_choices", None)
        overrides = getattr(self, "_visual_section_overrides", None)
        if label is None or overrides is None:
            return
        if not overrides:
            label.setText("")
            return
        order = [
            ("intro", "Intro"),
            ("verse", "Verse"),
            ("chorus", "Chorus"),
            ("break", "Break"),
            ("drop", "Drop"),
            ("outro", "Outro"),
        ]
        parts = []
        for key, title in order:
            if key in overrides:
                mode = overrides.get(key)
                if mode is None:
                    desc = "no visual"
                else:
                    pretty = str(mode)
                    if pretty.startswith("viz:"):
                        pretty = pretty[4:]
                    desc = pretty
                parts.append(f"{title}: {desc}")
        label.setText(" / ".join(parts))

    def _on_select_visuals_per_section(self) -> None:
        try:
            from .viz_offline import _list_visual_modes
            from .music import VisualEngine
        except Exception:
            QMessageBox.warning(
                self,
                "Visual presets not available",
                                "Could not load visual presets for the music-player visuals.\n"
                "Please ensure presets/viz is available.",

                QMessageBox.Ok,
            )
            return

        try:
            engine = VisualEngine(parent=None)
            modes = _list_visual_modes(engine)
        except Exception:
            modes = []

        if not modes:
            QMessageBox.warning(
                self,
                "No visuals found",
                                "No music-player visual presets were found.\n"
                "Please add presets in presets/viz and try again.",

                QMessageBox.Ok,
            )
            return

        # Map section keys to human-readable labels
        sections = [
            ("intro", "Intro"),
            ("verse", "Verse"),
            ("chorus", "Chorus"),
            ("break", "Break"),
            ("drop", "Drop"),
            ("outro", "Outro"),
        ]

        current = getattr(self, "_visual_section_overrides", {}) or {}
        manager = getattr(self, "_visual_thumbs", None)

        dlg = QDialog(self)
        dlg.setWindowTitle("Select visuals per section type")
        layout = QVBoxLayout(dlg)

        info = QLabel(
            "Choose which music-player visual preset to use for each section type.\n"
            "You can also disable visuals for specific sections."
        , dlg)
        info.setWordWrap(True)
        layout.addWidget(info)

        # Main content area: section rows on the left, large preview on the right.
        main_row = QHBoxLayout()
        layout.addLayout(main_row)

        left_col = QVBoxLayout()
        main_row.addLayout(left_col, 1)

        right_col = QVBoxLayout()
        main_row.addLayout(right_col)

        preview_title = QLabel("Preview:", dlg)
        right_col.addWidget(preview_title)

        preview_label = QLabel(
            "Hover over a visual in the list\n"
            "to see a larger preview here.",
            dlg,
        )
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setWordWrap(True)
        preview_label.setMinimumSize(320, 180)
        preview_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 80);"
            "border: 1px solid rgba(255, 255, 255, 60);"
        )
        right_col.addWidget(preview_label)

        def _make_preview_updater(combo: QComboBox):
            """Create a slot that updates the large preview when the combo changes.

            We rely on VisualThumbManager.preview_pixmap_for_mode to reuse the
            existing tiny thumbnail on disk and simply scale it up. No new
            thumbnails are written.
            """
            def _update(index: int) -> None:
                if preview_label is None:
                    return

                # Guard against weird indices or missing items.
                try:
                    data = combo.itemData(index)
                    text = combo.itemText(index)
                except Exception:
                    data = None
                    text = ""
                if not text:
                    text = "Random visual"

                # Random / none entry: text-only hint.
                if not isinstance(data, str) or not data:
                    preview_label.clear()
                    preview_label.setText(
                        f"{text}\n(uses any installed preset)."
                    )
                    return

                # Ask the manager for a larger pixmap based on the existing thumb.
                pm = None
                if manager is not None:
                    try:
                        pm = manager.preview_pixmap_for_mode(str(data))
                    except Exception:
                        pm = None

                if pm is None or getattr(pm, "isNull", lambda: True)():
                    preview_label.clear()
                    preview_label.setText(text)
                    return

                preview_label.setPixmap(pm)
                preview_label.setText("")

            return _update

        row_widgets = []  # (section_key, combo, checkbox)
        for key, label_text in sections:
            row = QHBoxLayout()
            lbl = QLabel(f"{label_text}:", dlg)
            row.addWidget(lbl)

            combo = QComboBox(dlg)
            combo.addItem("Random visual", "")
            for m in modes:
                if not m:
                    continue
                pretty = str(m)
                if pretty.startswith("viz:"):
                    pretty = pretty[4:]
                icon = QIcon()
                if manager is not None:
                    try:
                        icon = manager.icon_for_mode(str(m))
                    except Exception:
                        icon = QIcon()
                if icon.isNull():
                    combo.addItem(pretty, m)
                else:
                    combo.addItem(icon, pretty, m)

            override = current.get(key)
            if isinstance(override, str) and override:
                # Try to select matching preset
                for i in range(combo.count()):
                    if combo.itemData(i) == override:
                        combo.setCurrentIndex(i)
                        break

            row.addWidget(combo, 1)

            chk = QCheckBox("No beat-synced visual", dlg)
            row.addWidget(chk)

            if key in current and current.get(key) is None:
                chk.setChecked(True)
                combo.setEnabled(False)

            def _make_toggle(c: QComboBox, cb: QCheckBox):
                def _on_toggled(state: bool) -> None:
                    c.setEnabled(not state)
                return _on_toggled

            chk.toggled.connect(_make_toggle(combo, chk))

            update_preview = _make_preview_updater(combo)
            combo.currentIndexChanged.connect(update_preview)
            try:
                combo.highlighted.connect(update_preview)
            except Exception:
                # Some very old Qt builds may not support highlighted().
                pass

            left_col.addLayout(row)
            row_widgets.append((key, combo, chk))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.Accepted:
            return

        overrides = {}
        for key, combo, chk in row_widgets:
            if chk.isChecked():
                overrides[key] = None
            else:
                data = combo.currentData()
                if isinstance(data, str) and data:
                    overrides[key] = data

        self._visual_section_overrides = overrides
        self._update_visual_section_summary()



    def _on_image_interval_changed(self, value: int) -> None:
        """Update label + internal setting for the still-image segment interval slider."""
        try:
            v = int(value)
        except Exception:
            v = int(getattr(self, "image_segment_interval", 4) or 4)
        if v < 0:
            v = 0
        if v > 20:
            v = 20
        self.image_segment_interval = v
        label = getattr(self, "label_image_interval", None)
        if label is not None:
            try:
                label.setText(f"New image every {v} segments")
            except Exception:
                pass


    def _on_visual_opacity_slider_changed(self, value: int) -> None:
        """Sync spinbox + internal opacity value when the slider changes."""
        try:
            alpha = float(value) / 100.0
        except Exception:
            alpha = float(getattr(self, "visual_overlay_opacity", 0.25))
        alpha = max(0.10, min(1.0, alpha))
        self.visual_overlay_opacity = alpha
        spin = getattr(self, "spin_visual_opacity", None)
        if spin is not None:
            try:
                spin.blockSignals(True)
                spin.setValue(alpha)
            finally:
                spin.blockSignals(False)
    
    def _on_visual_opacity_spin_changed(self, value: float) -> None:
        """Sync slider + internal opacity value when the spinbox changes."""
        try:
            alpha = float(value)
        except Exception:
            alpha = float(getattr(self, "visual_overlay_opacity", 0.25))
        alpha = max(0.10, min(1.0, alpha))
        self.visual_overlay_opacity = alpha
        slider = getattr(self, "slider_visual_opacity", None)
        if slider is not None:
            try:
                slider.blockSignals(True)
                slider.setValue(int(round(alpha * 100.0)))
            finally:
                slider.blockSignals(False)

    def _on_manage_transitions(self) -> None:
        """Open a dialog to choose which transition styles are allowed for random mode."""
        names = [
            "Soft film dissolves",
            "Hard cuts",
            "Scale punch (zoom)",
            "Shimmer blur (shiny)",
            "Iris reveal (circle)",
            "Motion blur whip-cuts",
            "Slit-scan smear push",
            "Radial burst reveal",
            "Directional push (slide)",
            "Wipe",
            "Smooth zoom crossfade",
            "Curtain open (doors)",
            "Pixelize",
            "Distance (liquid blend)",
            "Wind smears",
        ]
        dlg = QDialog(self)
        dlg.setWindowTitle("Manage transitions")
        layout = QVBoxLayout(dlg)
        info = QLabel(
            "Select which transition styles can be used when\n"
            "'Random transitions' is enabled."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        checkboxes = []
        for i, name in enumerate(names):
            cb = QCheckBox(name, dlg)
            cb.setChecked(i in self._enabled_transition_modes)
            layout.addWidget(cb)
            checkboxes.append(cb)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        if dlg.exec() != QDialog.Accepted:
            return
        selected = {i for i, cb in enumerate(checkboxes) if cb.isChecked()}
        if not selected:
            QMessageBox.warning(
                self,
                "No transitions selected",
                "At least one transition style must be enabled for random mode.",
                QMessageBox.Ok,
            )
            return
        self._enabled_transition_modes = selected




        # Persist immediately so it survives app restart even if the user doesn't render right away.
        try:
            self._save_settings()
        except Exception:
            pass
    def _update_transition_visibility(self) -> None:
        """Show or hide the transitions dropdown row based on random mode."""
        # If random transitions are enabled, hide the label + dropdown since
        # they are not used. When disabled, show them again.
        try:
            is_random = self.check_trans_random.isChecked()
        except Exception:
            return
        visible = not is_random
        try:
            self.label_transitions.setVisible(visible)
        except Exception:
            pass
        try:
            self.combo_transitions.setVisible(visible)
        except Exception:
            pass

    def _on_trans_random_toggled(self, state: int) -> None:
        """Keep the transitions row in sync with the 'Random transitions' toggle."""
        self._update_transition_visibility()

    def _on_cancel(self) -> None:
        """Cancel current render (if any), reset progress and clean temp folder."""
        # If a worker exists and is running, ask it to stop.
        if getattr(self, "_worker", None) is not None:
            try:
                self._worker.requestInterruption()
            except Exception:
                pass
        # Update UI
        self.progress.setValue(0)
        self.progress.setFormat("Cancelled.")
        self.label_summary.setText(self.label_summary.text() + "\nRender cancelled.")
        # Best-effort temp cleanup by rerunning cleanup helper.
        try:
            cleanup_temp_dir(self.edit_output.text().strip())
        except Exception:
            pass




        # Reset UI state
        try:
            self._queue_requested = False
        except Exception:
            pass
        try:
            if getattr(self, "_direct_run_active", False):
                self._set_direct_run_active(False)
        except Exception:
            pass
    def _on_reset_all(self) -> None:
        """Fully reset the Music Clip Creator state and clear loaded sources."""
        # Stop any running render or background scan.
        try:
            if getattr(self, "_worker", None) is not None:
                try:
                    self._worker.requestInterruption()
                except Exception:
                    pass
                self._worker = None
        except Exception:
            pass
        try:
            if getattr(self, "_scan_worker", None) is not None:
                try:
                    self._scan_worker.requestInterruption()
                except Exception:
                    pass
                self._scan_worker = None
        except Exception:
            pass


        # Reset UI state (DIRECT RUN label / queued flag)
        try:
            self._queue_requested = False
        except Exception:
            pass
        try:
            if getattr(self, "_direct_run_active", False):
                self._set_direct_run_active(False)
        except Exception:
            pass

        # Clear progress + summary text.
        try:
            self.progress.setValue(0)
            self.progress.setFormat("Ready.")
        except Exception:
            pass
        try:
            self.label_summary.setText("")
        except Exception:
            pass

        # Reset in-memory analysis and configuration.
        try:
            self._analysis = None
            self._analysis_config = MusicAnalysisConfig()
        except Exception:
            pass

        # Clear any per-section overrides and cached sources.
        try:
            self._section_media.clear()
        except Exception:
            pass
        try:
            self.clip_sources = []
            self.image_sources = []
        except Exception:
            pass
        try:
            self._pending_audio = None
            self._pending_video = None
            self._pending_out_dir = None
        except Exception:
            pass

        # Clear the visible sources list.
        try:
            if hasattr(self, "list_sources"):
                self.list_sources.clear()
        except Exception:
            pass

        # Clear basic path fields (keep output path).
        try:
            self.edit_audio.clear()
            self.edit_video.clear()
        except Exception:
            pass

        # Turn off cinematic + break/drop FX using the section all-on/off helpers.
        try:
            if hasattr(self, "check_cine_enable"):
                self.check_cine_enable.setChecked(False)
        except Exception:
            pass
        try:
            self._set_all_cinematic_toggles(False)
        except Exception:
            pass
        try:
            if hasattr(self, "check_impact_enable"):
                self.check_impact_enable.setChecked(False)
        except Exception:
            pass
        try:
            self._set_all_impact_toggles(False)
        except Exception:
            pass

        # Ask the timeline panel (if present) to forget any previous analysis.
        try:
            if hasattr(self, "timeline_panel"):
                for name in ("clear_timeline", "reset_view", "clear"):
                    fn = getattr(self.timeline_panel, name, None)
                    if callable(fn):
                        fn()
                        break
        except Exception:
            pass

        # Finally, forget persisted settings so a fresh session starts from defaults.
        # Keep the output folder so 'Reset all' doesn't wipe it.
        try:
            out_keep = ""
            try:
                out_keep = self.edit_output.text().strip()
            except Exception:
                out_keep = ""
            self._settings.clear()
            if out_keep:
                self._settings.setValue("output_path", out_keep)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 1-click preset helper
    # ------------------------------------------------------------------

    def _on_presets_clicked(self) -> None:
        """
        Show a small dialog with the built-in presets, apply the choice and
        immediately queue a render.

        Important: presets ONLY touch visual / FX options. They do not
        change which audio/clip paths are selected, the output folder or
        the resolution.
        """
        # Avoid starting while something else is running.
        if getattr(self, "_worker", None) is not None and self._worker.isRunning():
            self._error("Busy", "A render is already running.")
            return
        if getattr(self, "_scan_worker", None) is not None and self._scan_worker.isRunning():
            self._error("Busy", "Clip scanning is already running.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("1-click presets")
        layout = QVBoxLayout(dlg)

        info = QLabel(
            "Choose a preset for how intense the visuals should be.\n\n"
            "Presets only change visual / FX options:\n"
            "  • FX level and transitions\n"
            "  • microclips + beats per base segment\n"
            "  • slow-motion, cinematic and break-impact FX\n\n"
            "They do NOT change:\n"
            "  • which music track is selected\n"
            "  • which clip(s) or folder are selected\n"
            "  • the output folder or output resolution.\n\n"
            "Set those first, then pick a preset and press OK.\n"
            "After you confirm, the preset is applied and the normal\n"
            "'Queue Clip' flow is started (check the Queue tab).",
            dlg,
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        list_widget = QListWidget(dlg)
        # Load presets from JSON (if present) merged with built-ins.
        presets = _load_clip_presets()
        for _, name, desc in presets:
            item = QListWidgetItem(name)
            item.setToolTip(desc)
            list_widget.addItem(item)
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
        layout.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.Accepted:
            return

        row = list_widget.currentRow()
        if row < 0 or row >= len(presets):
            return

        preset_id, preset_name, _ = presets[row]
        self._apply_preset(preset_id, preset_name)

        # Queue the clip with the chosen preset.
        self._on_queue_generate()

    def _apply_preset(self, preset_id: str, preset_name: str) -> None:
        """
        Apply one of the 1-click FX presets to the current UI.

        The actual preset values are loaded from clip_presets.json so that
        presets live in data instead of code. Only the two safe presets
        ("clean" / NoFX and "chill" / slow cinematic) keep a hard-coded
        fallback so the dialog still works if the JSON file is missing.
        """
        # ---------- global baseline ----------
        # No-FX off by default (individual presets may turn it back on).
        self.check_nofx.setChecked(False)

        # Microclips
        self.check_micro_chorus.setChecked(False)
        self.check_micro_all.setChecked(False)
        try:
            self.check_micro_verses.setChecked(False)
        except Exception:
            pass

        # Fades: all presets keep gentle fades in/out.
        self.check_intro_fade.setChecked(True)
        self.check_outro_fade.setChecked(True)
        if hasattr(self, "check_intro_transitions_only"):
            self.check_intro_transitions_only.setChecked(False)

        # Slow-motion base
        self.check_slow_enable.setChecked(False)
        for cb in (
            self.check_slow_intro,
            self.check_slow_break,
            self.check_slow_chorus,
            self.check_slow_drop,
            self.check_slow_outro,
            self.check_slow_random,
        ):
            cb.setChecked(False)
        self.slider_slow_factor.setValue(50)  # 0.50x baseline (only used when enabled)

        # Cinematic base
        self.check_cine_enable.setChecked(False)
        self.check_cine_freeze.setChecked(False)
        self.check_cine_tear_v.setChecked(False)
        self.check_cine_stutter.setChecked(False)
        self.check_cine_reverse.setChecked(False)
        self.check_cine_speed_ramp.setChecked(False)
        self.check_cine_boomerang.setChecked(False)
        # New camera motion toggles
        self.check_cine_dolly.setChecked(False)
        self.check_cine_kenburns.setChecked(False)
        try:
            self.slider_cine_dolly_strength.setValue(50)
            self.slider_cine_kenburns_strength.setValue(40)
            self.combo_cine_motion_dir.setCurrentIndex(0)
        except Exception:
            pass

        # Break-impact base
        self.check_impact_enable.setChecked(False)
        for cb in (
            self.check_impact_flash,
            self.check_impact_shock,
            self.check_impact_confetti,
            self.check_impact_zoom,
            self.check_impact_shake,
            self.check_impact_fog,
            self.check_impact_fire_gold,
            self.check_impact_fire_multi,
            self.check_impact_random,
        ):
            cb.setChecked(False)

        # Transitions + beats baseline
        self.combo_transitions.setCurrentIndex(1)  # Hard cuts
        self.check_trans_random.setChecked(False)
        self.spin_beats_per_seg.setValue(8)        # mid-speed by default
        self.combo_fx.setCurrentIndex(1)           # Moderate by default

        # ---------- JSON‑driven preset body ----------

        settings = _get_clip_preset_settings(preset_id)

        if isinstance(settings, dict):
            # Apply JSON settings on top of the baseline.
            self._apply_preset_from_settings_dict(settings)
        else:
            # If JSON is missing or this preset id is not defined, fall back
            # to the two safe built-ins so the classic behaviour still works.
            if preset_id == "clean":
                self.check_nofx.setChecked(True)
                self.combo_fx.setCurrentIndex(0)   # Minimal
                self.spin_beats_per_seg.setValue(8)
                self.combo_transitions.setCurrentIndex(1)  # Hard cuts only

            elif preset_id == "chill":
                self.combo_fx.setCurrentIndex(0)   # Minimal
                self.spin_beats_per_seg.setValue(16)
                self.check_slow_enable.setChecked(True)
                try:
                    self.check_slow_break.setChecked(True)
                    self.check_slow_chorus.setChecked(True)
                except Exception:
                    pass
                self.slider_slow_factor.setValue(60)  # 0.60x
                self.check_cine_enable.setChecked(True)
                self.check_cine_freeze.setChecked(True)
                self.slider_cine_freeze_len.setValue(70)   # 0.70 s
                self.slider_cine_freeze_zoom.setValue(10)  # subtle zoom
                self.check_cine_speed_ramp.setChecked(True)
                self.slider_cine_ramp_in.setValue(25)      # 0.25 s
                self.slider_cine_ramp_out.setValue(25)     # 0.25 s
                # Chill: gentle Ken Burns only, no dolly-zoom by default.
                self.check_cine_dolly.setChecked(False)
                self.check_cine_kenburns.setChecked(True)
                try:
                    self.slider_cine_dolly_strength.setValue(35)
                    self.slider_cine_kenburns_strength.setValue(40)
                    self.combo_cine_motion_dir.setCurrentIndex(0)
                except Exception:
                    pass
                self.combo_transitions.setCurrentIndex(0)  # Soft film dissolves
            else:
                # Unknown preset without JSON data – leave the baseline in place
                # and show a friendly message instead of crashing.
                try:
                    self._info(
                        "Preset not found",
                        f"Preset '{preset_name}' (id: {preset_id}) is not defined in "
                        "clip_presets.json and has no built-in fallback.\n\n"
                        "Please re-save your presets or update the presets file."
                    )
                except Exception:
                    pass

        # Make sure hidden camera-motion FX stay off even if a preset tries to enable them.
        try:
            self._disable_camera_motion_fx()
        except Exception:
            pass

        try:
            self.label_summary.setText(f"Preset applied: {preset_name}")
        except Exception:
            pass


    # ------------------------------------------------------------------
    # Preset manager (edits clip_presets.json only; never starts a run)
    # ------------------------------------------------------------------

    def _on_edit_presets_clicked(self) -> None:
        """Open the preset manager dialog.

        This is intentionally separated from the 1-click preset flow so we
        don't risk breaking the "pick preset -> immediately make videoclip"
        pipeline.
        """
        try:
            dlg = ClipPresetManagerDialog(self, self._capture_options_advanced_settings, self._apply_preset_from_settings_dict)
            dlg.exec()
        except Exception as e:
            try:
                self._error("Preset manager", f"Failed to open preset manager:\n{e}")
            except Exception:
                pass

    def _capture_options_advanced_settings(self) -> dict:
        """Capture ALL settings inside the Options + Advanced UI sections.

        The resulting dict is compatible with _apply_preset_from_settings_dict,
        i.e. it uses widget attribute names as keys.
        """
        out: dict = {}

        box_opts = getattr(self, "box_opts", None)
        box_adv = getattr(self, "box_adv", None)
        if box_opts is None and box_adv is None:
            return out

        # Map widget object id -> attribute name on this instance.
        id_to_attr: dict[int, str] = {}
        for k, v in getattr(self, "__dict__", {}).items():
            try:
                if isinstance(v, QWidget):
                    id_to_attr[id(v)] = k
            except Exception:
                continue

        def add_widget(w: QWidget) -> None:
            key = id_to_attr.get(id(w))
            if not key:
                return
            # Skip obvious non-settings widgets.
            try:
                if isinstance(w, (QLabel, QPushButton, QTextEdit, QListWidget, QProgressBar)):
                    return
            except Exception:
                pass

            try:
                # QGroupBox can be checkable (Advanced)
                if hasattr(w, "isChecked") and hasattr(w, "setChecked"):
                    out[key] = bool(w.isChecked())
                    return
                # Combos store index (matches _apply_preset_from_settings_dict)
                if isinstance(w, QComboBox):
                    out[key] = int(w.currentIndex())
                    return
                # Sliders / spinboxes etc.
                if hasattr(w, "value") and hasattr(w, "setValue"):
                    out[key] = w.value()
                    return
            except Exception:
                return

        for root in (box_opts, box_adv):
            if root is None:
                continue
            try:
                if isinstance(root, QWidget):
                    add_widget(root)
                    for child in root.findChildren(QWidget):
                        add_widget(child)
            except Exception:
                continue

        return out


    def _apply_preset_from_settings_dict(self, settings: dict) -> None:
        """Apply a preset settings dict (from clip_presets.json) onto the UI.

        The dict uses widget attribute names as keys (for example
        "check_nofx", "combo_fx", "spin_beats_per_seg", "slider_slow_factor"...)
        and the method does a best-effort mapping to the actual Qt widgets.

        Any unknown keys or missing widgets are silently ignored so that
        presets stay forwards‑compatible with future UI tweaks.
        """
        for key, value in settings.items():
            attr = getattr(self, key, None)
            if attr is None:
                continue
            try:
                # QCheckBox / QAbstractButton‑style
                if hasattr(attr, "setChecked"):
                    attr.setChecked(bool(value))
                    continue
                # QComboBox‑style
                if hasattr(attr, "setCurrentIndex"):
                    try:
                        idx = int(value)
                    except Exception:
                        continue
                    attr.setCurrentIndex(idx)
                    continue
                # Sliders / spinboxes etc. – anything that exposes setValue().
                if hasattr(attr, "setValue"):
                    try:
                        attr.setValue(value)
                    except TypeError:
                        try:
                            # Fallback conversions for json numbers.
                            if isinstance(value, bool):
                                attr.setValue(int(value))
                            else:
                                attr.setValue(float(value))
                        except Exception:
                            # Last resort, try int() and ignore on failure.
                            try:
                                attr.setValue(int(value))
                            except Exception:
                                pass
            except Exception:
                # Never crash if a single field fails – just skip it.
                continue
        # Ensure hidden camera-motion FX are always off, even if present in the preset dict.
        try:
            self._disable_camera_motion_fx()
        except Exception:
            pass

    def _load_settings(self) -> None:
        """Load last-used paths and options from QSettings."""
        s = self._settings
        self.edit_audio.setText(s.value("audio_path", "", str))
        self.edit_video.setText(s.value("video_path", "", str))
        self.edit_output.setText(s.value("output_path", "output/videoclips", str))
        try:
            if not self.edit_output.text().strip():
                self.edit_output.setText("output/videoclips")
        except Exception:
            pass

        self.combo_fx.setCurrentIndex(int(s.value("fx_level", self.combo_fx.currentIndex())))
        self.check_nofx.setChecked(bool(int(s.value("nofx", int(self.check_nofx.isChecked() if hasattr(self, "check_nofx") else 0)))))
        if hasattr(self, "check_visual_overlay"):
            self.check_visual_overlay.setChecked(bool(int(s.value("visual_overlay", int(self.check_visual_overlay.isChecked())))))
        if hasattr(self, "check_visual_strategy_segment"):
            self.check_visual_strategy_segment.setChecked(bool(int(s.value("visual_strategy_segment", int(self.check_visual_strategy_segment.isChecked())))))
        if hasattr(self, "check_visual_strategy_section"):
            self.check_visual_strategy_section.setChecked(bool(int(s.value("visual_strategy_section", int(self.check_visual_strategy_section.isChecked())))))
        self.check_micro_chorus.setChecked(bool(int(s.value("micro_chorus", int(self.check_micro_chorus.isChecked())))))
        self.check_micro_all.setChecked(bool(int(s.value("micro_all", int(self.check_micro_all.isChecked())))))
        if hasattr(self, "check_micro_verses"):
            self.check_micro_verses.setChecked(bool(int(s.value("micro_verses", int(self.check_micro_verses.isChecked())))))
        self.check_full_length.setChecked(True)
        self.check_intro_fade.setChecked(bool(int(s.value("intro_fade", int(self.check_intro_fade.isChecked())))))
        self.check_outro_fade.setChecked(bool(int(s.value("outro_fade", int(self.check_outro_fade.isChecked())))))
        if hasattr(self, "check_intro_transitions_only"):
            self.check_intro_transitions_only.setChecked(bool(int(s.value("intro_transitions_only", int(self.check_intro_transitions_only.isChecked())))))
        self.combo_clip_order.setCurrentIndex(int(s.value("clip_order", self.combo_clip_order.currentIndex())))
        self.combo_transitions.setCurrentIndex(int(s.value("transitions_mode", self.combo_transitions.currentIndex())))
        self.check_trans_random.setChecked(bool(int(s.value("transitions_random", int(self.check_trans_random.isChecked())))))
        # Restore the allowed transition styles for Random transitions
        try:
            raw = s.value("transitions_random_enabled_modes", "", str)
        except Exception:
            raw = ""
        modes = []
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, (list, tuple)):
                    modes = [int(x) for x in data]
                else:
                    modes = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
            except Exception:
                try:
                    modes = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
                except Exception:
                    modes = []
        cleaned = []
        for x in modes:
            try:
                xi = int(x)
            except Exception:
                continue
            if 0 <= xi <= 14 and xi not in cleaned:
                cleaned.append(xi)
        if cleaned:
            self._enabled_transition_modes = set(cleaned)
        else:
            # If nothing was saved yet, keep the current default (all enabled)
            if not getattr(self, "_enabled_transition_modes", None):
                self._enabled_transition_modes = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}

        self.spin_seed.setValue(int(s.value("seed_value", self.spin_seed.value())))
        self.check_use_seed.setChecked(bool(int(s.value("use_seed", int(self.check_use_seed.isChecked())))))
        self.combo_res.setCurrentIndex(int(s.value("res_mode", self.combo_res.currentIndex())))
        self.combo_fit.setCurrentIndex(int(s.value("fit_mode", self.combo_fit.currentIndex())))
        if hasattr(self, "check_keep_source_bitrate"):
            self.check_keep_source_bitrate.setChecked(bool(int(s.value("keep_source_bitrate", int(self.check_keep_source_bitrate.isChecked())))))

        self.slider_sens.setValue(int(s.value("beat_sensitivity", self.slider_sens.value())))
        self.spin_beats_per_seg.setValue(int(s.value("beats_per_segment", self.spin_beats_per_seg.value())))

        # Restore still-image segment interval (segments per image)
        try:
            interval = int(s.value("image_segment_interval", int(getattr(self, "image_segment_interval", 4))))
        except Exception:
            interval = int(getattr(self, "image_segment_interval", 4))
        if interval < 0:
            interval = 0
        if interval > 20:
            interval = 20
        self.image_segment_interval = interval
        slider = getattr(self, "slider_image_interval", None)
        if slider is not None:
            try:
                slider.blockSignals(True)
                slider.setMinimum(0)
                slider.setMaximum(20)
                slider.setValue(interval)
            finally:
                slider.blockSignals(False)
        self._on_image_interval_changed(interval)

        self.check_min_clip.setChecked(bool(int(s.value("min_clip_enabled", int(self.check_min_clip.isChecked())))))
        self.spin_min_clip.setValue(float(s.value("min_clip_seconds", self.spin_min_clip.value())))

        # Slow motion options
        self.check_slow_enable.setChecked(bool(int(s.value("slow_enable", int(self.check_slow_enable.isChecked())))))
        self.check_slow_intro.setChecked(bool(int(s.value("slow_intro", int(self.check_slow_intro.isChecked())))))
        self.check_slow_break.setChecked(bool(int(s.value("slow_break", int(self.check_slow_break.isChecked())))))
        self.check_slow_chorus.setChecked(bool(int(s.value("slow_chorus", int(self.check_slow_chorus.isChecked())))))
        self.check_slow_drop.setChecked(bool(int(s.value("slow_drop", int(self.check_slow_drop.isChecked())))))
        self.check_slow_outro.setChecked(bool(int(s.value("slow_outro", int(self.check_slow_outro.isChecked())))))
        self.check_slow_random.setChecked(bool(int(s.value("slow_random", int(self.check_slow_random.isChecked())))))
        self.slider_slow_factor.setValue(int(s.value("slow_factor_slider", self.slider_slow_factor.value())))

        # Cinematic effects options
        self.check_cine_enable.setChecked(bool(int(s.value("cine_enable", int(self.check_cine_enable.isChecked())))))
        self.check_cine_freeze.setChecked(bool(int(s.value("cine_freeze", int(self.check_cine_freeze.isChecked())))))
        self.slider_cine_freeze_len.setValue(int(s.value("cine_freeze_len", self.slider_cine_freeze_len.value())))
        self.slider_cine_freeze_zoom.setValue(int(s.value("cine_freeze_zoom", self.slider_cine_freeze_zoom.value())))
        self.check_cine_tear_v.setChecked(bool(int(s.value("cine_tear_v", int(self.check_cine_tear_v.isChecked())))))
        self.slider_cine_tear_v_strength.setValue(int(s.value("cine_tear_v_strength", self.slider_cine_tear_v_strength.value())))
        self.check_cine_tear_h.setChecked(bool(int(s.value("cine_tear_h", int(self.check_cine_tear_h.isChecked())))))
        self.slider_cine_tear_h_strength.setValue(int(s.value("cine_tear_h_strength", self.slider_cine_tear_h_strength.value())))
        self.check_cine_color_cycle.setChecked(bool(int(s.value("cine_color_cycle", int(self.check_cine_color_cycle.isChecked())))))
        self.slider_cine_color_cycle_speed.setValue(int(s.value("cine_color_cycle_speed_ms", int(self.slider_cine_color_cycle_speed.value()))))

        self.check_cine_stutter.setChecked(bool(int(s.value("cine_stutter", int(self.check_cine_stutter.isChecked())))))
        self.spin_cine_stutter_repeats.setValue(int(s.value("cine_stutter_repeats", self.spin_cine_stutter_repeats.value())))
        self.check_cine_reverse.setChecked(bool(int(s.value("cine_reverse", int(self.check_cine_reverse.isChecked())))))
        self.slider_cine_reverse_len.setValue(int(s.value("cine_reverse_len", self.slider_cine_reverse_len.value())))
        self.check_cine_speedup_forward.setChecked(bool(int(s.value("cine_speedup_forward", int(self.check_cine_speedup_forward.isChecked())))))
        try:
            self.spin_cine_speedup_forward.setValue(float(s.value("cine_speedup_forward_factor", self.spin_cine_speedup_forward.value())))
        except Exception:
            pass
        self.check_cine_speedup_backward.setChecked(bool(int(s.value("cine_speedup_backward", int(self.check_cine_speedup_backward.isChecked())))))
        try:
            self.spin_cine_speedup_backward.setValue(float(s.value("cine_speedup_backward_factor", self.spin_cine_speedup_backward.value())))
        except Exception:
            pass
        self.check_cine_speed_ramp.setChecked(bool(int(s.value("cine_speed_ramp", int(self.check_cine_speed_ramp.isChecked())))))
        self.slider_cine_ramp_in.setValue(int(s.value("cine_ramp_in", self.slider_cine_ramp_in.value())))
        self.slider_cine_ramp_out.setValue(int(s.value("cine_ramp_out", self.slider_cine_ramp_out.value())))
        self.check_cine_boomerang.setChecked(bool(int(s.value("cine_boomerang", int(self.check_cine_boomerang.isChecked())))))
        self.slider_cine_boomerang_bounces.setValue(int(s.value("cine_boomerang_bounces", self.slider_cine_boomerang_bounces.value())))
        if hasattr(self, "check_cine_dimension"):
            self.check_cine_dimension.setChecked(bool(int(s.value("cine_dimension", int(self.check_cine_dimension.isChecked())))))

        if hasattr(self, "check_cine_pan916"):
            self.check_cine_pan916.setChecked(bool(int(s.value("cine_pan916", int(self.check_cine_pan916.isChecked())))))
            self.slider_cine_pan916_speed.setValue(int(s.value("cine_pan916_speed_ms", self.slider_cine_pan916_speed.value())))
            try:
                self._on_cine_pan916_speed_changed(int(self.slider_cine_pan916_speed.value()))
            except Exception:
                pass
            self.slider_cine_pan916_parts.setValue(int(s.value("cine_pan916_parts", self.slider_cine_pan916_parts.value())))
            try:
                self._on_cine_pan916_parts_changed(int(self.slider_cine_pan916_parts.value()))
            except Exception:
                pass
            self.check_cine_pan916_transparent.setChecked(bool(int(s.value("cine_pan916_transparent", int(self.check_cine_pan916_transparent.isChecked())))))
            if hasattr(self, "check_cine_pan916_random"):
                self.check_cine_pan916_random.setChecked(bool(int(s.value("cine_pan916_random", int(self.check_cine_pan916_random.isChecked())))))
                try:
                    self._on_cine_pan916_random_changed(self.check_cine_pan916_random.isChecked())
                except Exception:
                    pass



        self.check_cine_mosaic.setChecked(bool(int(s.value("cine_mosaic", int(self.check_cine_mosaic.isChecked())))))
        self.slider_cine_mosaic_screens.setValue(int(s.value("cine_mosaic_screens", self.slider_cine_mosaic_screens.value())))
        self.check_cine_mosaic_random.setChecked(bool(int(s.value("cine_mosaic_random", int(self.check_cine_mosaic_random.isChecked())))))
        self.check_cine_flip.setChecked(bool(int(s.value("cine_flip", int(self.check_cine_flip.isChecked())))))
        self.check_cine_rotate.setChecked(bool(int(s.value("cine_rotate", int(self.check_cine_rotate.isChecked())))))
        self.slider_cine_rotate_degrees.setValue(int(s.value("cine_rotate_degrees", self.slider_cine_rotate_degrees.value())))
        self.check_cine_multiply.setChecked(bool(int(s.value("cine_multiply", int(self.check_cine_multiply.isChecked())))))
        self.slider_cine_multiply_screens.setValue(int(s.value("cine_multiply_screens", self.slider_cine_multiply_screens.value())))
        self.check_cine_multiply_random.setChecked(bool(int(s.value("cine_multiply_random", int(self.check_cine_multiply_random.isChecked())))))

        # Dolly / Ken Burns camera moves are hidden from the UI and always disabled.
        try:
            self._disable_camera_motion_fx()
        except Exception:
            pass

        # Break impact FX options
        self.check_impact_enable.setChecked(bool(int(s.value("impact_enable", int(getattr(self, "check_impact_enable").isChecked())))))
        self.check_impact_flash.setChecked(bool(int(s.value("impact_flash", int(self.check_impact_flash.isChecked())))))
        self.slider_impact_flash.setValue(int(s.value("impact_flash_strength", self.slider_impact_flash.value())))
        if hasattr(self, "slider_impact_flash_speed"):
            self.slider_impact_flash_speed.setValue(int(s.value("impact_flash_speed_ms", self.slider_impact_flash_speed.value())))
        self.check_impact_shock.setChecked(bool(int(s.value("impact_shock", int(self.check_impact_shock.isChecked())))))
        self.slider_impact_shock.setValue(int(s.value("impact_shock_strength", self.slider_impact_shock.value())))
        self.check_impact_echo.setChecked(bool(int(s.value("impact_echo_trail", int(self.check_impact_echo.isChecked())))))
        self.slider_impact_echo.setValue(int(s.value("impact_echo_trail_strength", self.slider_impact_echo.value())))
        self.check_impact_confetti.setChecked(bool(int(s.value("impact_confetti", int(self.check_impact_confetti.isChecked())))))
        self.check_impact_colorcycle.setChecked(bool(int(s.value("impact_colorcycle", int(self.check_impact_colorcycle.isChecked())))))
        self.slider_impact_confetti.setValue(int(s.value("impact_confetti_density", self.slider_impact_confetti.value())))
        self.slider_impact_colorcycle.setValue(int(s.value("impact_colorcycle_speed", self.slider_impact_colorcycle.value())))
        self.check_impact_zoom.setChecked(bool(int(s.value("impact_zoom", int(self.check_impact_zoom.isChecked())))))
        self.slider_impact_zoom.setValue(int(s.value("impact_zoom_amount", self.slider_impact_zoom.value())))
        self.check_impact_shake.setChecked(bool(int(s.value("impact_shake", int(self.check_impact_shake.isChecked())))))
        self.slider_impact_shake.setValue(int(s.value("impact_shake_strength", self.slider_impact_shake.value())))
        self.check_impact_fog.setChecked(bool(int(s.value("impact_fog", int(self.check_impact_fog.isChecked())))))
        self.slider_impact_fog.setValue(int(s.value("impact_fog_density", self.slider_impact_fog.value())))
        self.check_impact_fire_gold.setChecked(bool(int(s.value("impact_fire_gold", int(self.check_impact_fire_gold.isChecked())))))
        self.slider_impact_fire_gold.setValue(int(s.value("impact_fire_gold_intensity", self.slider_impact_fire_gold.value())))
        self.check_impact_fire_multi.setChecked(bool(int(s.value("impact_fire_multi", int(self.check_impact_fire_multi.isChecked())))))
        self.slider_impact_fire_multi.setValue(int(s.value("impact_fire_multi_intensity", self.slider_impact_fire_multi.value())))
        self.check_impact_random.setChecked(bool(int(s.value("impact_random", int(self.check_impact_random.isChecked())))))
        # Timed strobe (flash on time)
        try:
            self.check_strobe_on_time.setChecked(bool(int(s.value("strobe_on_time", int(self.check_strobe_on_time.isChecked())))))
        except Exception:
            try:
                self.check_strobe_on_time.setChecked(False)
            except Exception:
                pass

        times: List[float] = []
        try:
            raw = s.value("strobe_on_time_times", "", str)
        except Exception:
            raw = ""
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    for v in data:
                        try:
                            fv = float(v)
                        except Exception:
                            continue
                        if fv >= 0.0:
                            times.append(fv)
            except Exception:
                times = []
        try:
            self._set_strobe_times(times)
        except Exception:
            pass
        try:
            self._on_strobe_on_time_toggle(int(self.check_strobe_on_time.isChecked()))
        except Exception:
            pass

        # Hidden break-impact FX rows are always disabled, even if older settings had them on.
        for cb in (
            getattr(self, "check_impact_fog", None),
            getattr(self, "check_impact_fire_gold", None),
            getattr(self, "check_impact_fire_multi", None),
        ):
            if cb is not None:
                try:
                    cb.setChecked(False)
                except Exception:
                    pass

        # Ensure transitions dropdown visibility matches restored random toggle
        self._update_transition_visibility()
        # Ensure slow-motion controls reflect restored toggle
        self._on_slow_toggle(int(self.check_slow_enable.isChecked()))
        self._on_slow_factor_changed(self.slider_slow_factor.value())
        # Ensure cinematic controls visibility + labels
        self._on_cine_toggle(int(self.check_cine_enable.isChecked()))
        self._on_cine_freeze_len_changed(self.slider_cine_freeze_len.value())
        self._on_cine_freeze_zoom_changed(self.slider_cine_freeze_zoom.value())
        self._on_cine_color_cycle_speed_changed(self.slider_cine_color_cycle_speed.value())
        self._on_cine_reverse_len_changed(self.slider_cine_reverse_len.value())
        self._on_cine_ramp_in_changed(self.slider_cine_ramp_in.value())
        self._on_cine_ramp_out_changed(self.slider_cine_ramp_out.value())
        self._on_cine_boomerang_bounces_changed(self.slider_cine_boomerang_bounces.value())
        self._on_cine_pan916_speed_changed(self.slider_cine_pan916_speed.value())
        self._on_cine_pan916_parts_changed(self.slider_cine_pan916_parts.value())
        try:
            self._on_cine_pan916_random_changed(self.check_cine_pan916_random.isChecked())
        except Exception:
            pass
        self._on_cine_mosaic_screens_changed(self.slider_cine_mosaic_screens.value())
        self._on_cine_mosaic_random_changed(self.check_cine_mosaic_random.isChecked())
        self._on_cine_dolly_strength_changed(self.slider_cine_dolly_strength.value())
        self._on_cine_kenburns_strength_changed(self.slider_cine_kenburns_strength.value())
        # Ensure break impact FX panel visibility + labels
        try:
            self.impact_options.setVisible(self.check_impact_enable.isChecked())
        except Exception:
            pass
        self._on_impact_flash_changed(self.slider_impact_flash.value())
        try:
            self._on_impact_flash_speed_changed(self.slider_impact_flash_speed.value())
        except Exception:
            pass
        self._on_impact_shock_changed(self.slider_impact_shock.value())
        self._on_impact_confetti_changed(self.slider_impact_confetti.value())
        self._on_impact_zoom_changed(self.slider_impact_zoom.value())
        self._on_impact_shake_changed(self.slider_impact_shake.value())
        self._on_impact_fog_changed(self.slider_impact_fog.value())
        self._on_impact_fire_gold_changed(self.slider_impact_fire_gold.value())
        self._on_impact_fire_multi_changed(self.slider_impact_fire_multi.value())

        # Load per-section visual overrides for music-player overlay (if present).
        try:
            raw_viz_sections = s.value("visual_section_overrides", "", str)
        except Exception:
            raw_viz_sections = ""
        overrides = {}
        if raw_viz_sections:
            try:
                data = json.loads(raw_viz_sections)
                if isinstance(data, dict):
                    for k, v in data.items():
                        key = str(k).lower().strip()
                        if not key:
                            continue
                        if v in (None, ""):
                            overrides[key] = None
                        else:
                            overrides[key] = str(v)
            except Exception:
                overrides = {}
        self._visual_section_overrides = overrides
        try:
            self._update_visual_section_summary()
        except Exception:
            pass

        # Load visual overlay opacity (music-player visuals alpha).
        try:
            alpha = float(s.value("visual_overlay_opacity", self.visual_overlay_opacity))
        except Exception:
            alpha = float(getattr(self, "visual_overlay_opacity", 0.25))
        alpha = max(0.10, min(1.0, alpha))
        self.visual_overlay_opacity = alpha
        
        slider = getattr(self, "slider_visual_opacity", None)
        if slider is not None:
            try:
                slider.blockSignals(True)
                slider.setValue(int(round(alpha * 100.0)))
            finally:
                slider.blockSignals(False)
        
        spin = getattr(self, "spin_visual_opacity", None)
        if spin is not None:
            try:
                spin.blockSignals(True)
                spin.setValue(alpha)
            finally:
                spin.blockSignals(False)
        # Keep still-image UI block hidden, even after restoring settings.
        try:
            self._hide_image_sources_block()
        except Exception:
            pass



    def _save_settings(self) -> None:
        """Save last-used paths and options to QSettings."""
        s = self._settings
        s.setValue("audio_path", self.edit_audio.text().strip())
        s.setValue("video_path", self.edit_video.text().strip())
        s.setValue("output_path", self.edit_output.text().strip())

        s.setValue("fx_level", self.combo_fx.currentIndex())
        s.setValue("nofx", int(self.check_nofx.isChecked()))
        if hasattr(self, "check_visual_overlay"):
            s.setValue("visual_overlay", int(self.check_visual_overlay.isChecked()))
        if hasattr(self, "check_visual_strategy_segment"):
            s.setValue("visual_strategy_segment", int(self.check_visual_strategy_segment.isChecked()))
        if hasattr(self, "check_visual_strategy_section"):
            s.setValue("visual_strategy_section", int(self.check_visual_strategy_section.isChecked()))
        s.setValue("visual_section_overrides", json.dumps(getattr(self, "_visual_section_overrides", {})))
        s.setValue("visual_overlay_opacity", float(getattr(self, "visual_overlay_opacity", 0.25)))
        s.setValue("micro_chorus", int(self.check_micro_chorus.isChecked()))
        s.setValue("micro_all", int(self.check_micro_all.isChecked()))
        if hasattr(self, "check_micro_verses"):
            s.setValue("micro_verses", int(self.check_micro_verses.isChecked()))
        s.setValue("full_length", int(self.check_full_length.isChecked()))
        s.setValue("intro_fade", int(self.check_intro_fade.isChecked()))
        s.setValue("outro_fade", int(self.check_outro_fade.isChecked()))
        if hasattr(self, "check_intro_transitions_only"):
            s.setValue("intro_transitions_only", int(self.check_intro_transitions_only.isChecked()))
        s.setValue("clip_order", self.combo_clip_order.currentIndex())
        s.setValue("transitions_mode", self.combo_transitions.currentIndex())
        s.setValue("transitions_random", int(self.check_trans_random.isChecked()))
        # Remember which transition styles are enabled for Random transitions
        try:
            s.setValue("transitions_random_enabled_modes", json.dumps(sorted(getattr(self, "_enabled_transition_modes", []))))
        except Exception:
            pass
        s.setValue("seed_value", self.spin_seed.value())
        s.setValue("use_seed", int(self.check_use_seed.isChecked()))
        s.setValue("res_mode", self.combo_res.currentIndex())
        s.setValue("fit_mode", self.combo_fit.currentIndex())
        if hasattr(self, "check_keep_source_bitrate"):
            s.setValue("keep_source_bitrate", int(self.check_keep_source_bitrate.isChecked()))

        s.setValue("beat_sensitivity", self.slider_sens.value())
        s.setValue("beats_per_segment", self.spin_beats_per_seg.value())
        s.setValue("image_segment_interval", int(getattr(self, "image_segment_interval", 4)))


        s.setValue("min_clip_enabled", int(self.check_min_clip.isChecked()))
        s.setValue("min_clip_seconds", float(self.spin_min_clip.value()))

        # Slow motion options
        s.setValue("slow_enable", int(self.check_slow_enable.isChecked()))
        s.setValue("slow_intro", int(self.check_slow_intro.isChecked()))
        s.setValue("slow_break", int(self.check_slow_break.isChecked()))
        s.setValue("slow_chorus", int(self.check_slow_chorus.isChecked()))
        s.setValue("slow_drop", int(self.check_slow_drop.isChecked()))
        s.setValue("slow_outro", int(self.check_slow_outro.isChecked()))
        s.setValue("slow_random", int(self.check_slow_random.isChecked()))
        s.setValue("slow_factor_slider", int(self.slider_slow_factor.value()))

        # Cinematic effects options
        s.setValue("cine_enable", int(self.check_cine_enable.isChecked()))
        s.setValue("cine_freeze", int(self.check_cine_freeze.isChecked()))
        s.setValue("cine_freeze_len", int(self.slider_cine_freeze_len.value()))
        s.setValue("cine_freeze_zoom", int(self.slider_cine_freeze_zoom.value()))
        s.setValue("cine_tear_v", int(self.check_cine_tear_v.isChecked()))
        s.setValue("cine_tear_v_strength", int(self.slider_cine_tear_v_strength.value()))
        s.setValue("cine_tear_h", int(self.check_cine_tear_h.isChecked()))
        s.setValue("cine_tear_h_strength", int(self.slider_cine_tear_h_strength.value()))
        s.setValue("cine_color_cycle", int(self.check_cine_color_cycle.isChecked()))
        s.setValue("cine_color_cycle_speed_ms", int(self.slider_cine_color_cycle_speed.value()))
        s.setValue("cine_stutter", int(self.check_cine_stutter.isChecked()))
        s.setValue("cine_stutter_repeats", int(self.spin_cine_stutter_repeats.value()))
        s.setValue("cine_reverse", int(self.check_cine_reverse.isChecked()))
        s.setValue("cine_reverse_len", int(self.slider_cine_reverse_len.value()))
        s.setValue("cine_speedup_forward", int(self.check_cine_speedup_forward.isChecked()))
        s.setValue("cine_speedup_forward_factor", float(self.spin_cine_speedup_forward.value()))
        s.setValue("cine_speedup_backward", int(self.check_cine_speedup_backward.isChecked()))
        s.setValue("cine_speedup_backward_factor", float(self.spin_cine_speedup_backward.value()))
        s.setValue("cine_speed_ramp", int(self.check_cine_speed_ramp.isChecked()))
        s.setValue("cine_ramp_in", int(self.slider_cine_ramp_in.value()))
        s.setValue("cine_ramp_out", int(self.slider_cine_ramp_out.value()))
        s.setValue("cine_boomerang", int(self.check_cine_boomerang.isChecked()))
        s.setValue("cine_boomerang_bounces", int(self.slider_cine_boomerang_bounces.value()))
        if hasattr(self, "check_cine_dimension"):
            s.setValue("cine_dimension", int(self.check_cine_dimension.isChecked()))
        if hasattr(self, "check_cine_pan916"):
            s.setValue("cine_pan916", int(self.check_cine_pan916.isChecked()))
            s.setValue("cine_pan916_speed_ms", int(self.slider_cine_pan916_speed.value()))
            s.setValue("cine_pan916_parts", int(self.slider_cine_pan916_parts.value()))
            s.setValue("cine_pan916_transparent", int(self.check_cine_pan916_transparent.isChecked()))
            if hasattr(self, "check_cine_pan916_random"):
                s.setValue("cine_pan916_random", int(self.check_cine_pan916_random.isChecked()))
        s.setValue("cine_mosaic", int(self.check_cine_mosaic.isChecked()))
        s.setValue("cine_mosaic_screens", int(self.slider_cine_mosaic_screens.value()))
        s.setValue("cine_mosaic_random", int(self.check_cine_mosaic_random.isChecked()))
        s.setValue("cine_flip", int(self.check_cine_flip.isChecked()))
        s.setValue("cine_rotate", int(self.check_cine_rotate.isChecked()))
        s.setValue("cine_rotate_degrees", int(self.slider_cine_rotate_degrees.value()))
        s.setValue("cine_multiply", int(self.check_cine_multiply.isChecked()))
        s.setValue("cine_multiply_screens", int(self.slider_cine_multiply_screens.value()))
        s.setValue("cine_multiply_random", int(self.check_cine_multiply_random.isChecked()))
        s.setValue("cine_dolly", int(self.check_cine_dolly.isChecked()))
        s.setValue("cine_dolly_strength", int(self.slider_cine_dolly_strength.value()))
        s.setValue("cine_kenburns", int(self.check_cine_kenburns.isChecked()))
        s.setValue("cine_kenburns_strength", int(self.slider_cine_kenburns_strength.value()))
        s.setValue("cine_motion_dir", int(self.combo_cine_motion_dir.currentIndex()))
        # Break impact FX options
        s.setValue("impact_enable", int(self.check_impact_enable.isChecked()))
        s.setValue("impact_flash", int(self.check_impact_flash.isChecked()))
        s.setValue("impact_flash_strength", int(self.slider_impact_flash.value()))
        if hasattr(self, "slider_impact_flash_speed"):
            s.setValue("impact_flash_speed_ms", int(self.slider_impact_flash_speed.value()))
        s.setValue("impact_shock", int(self.check_impact_shock.isChecked()))
        s.setValue("impact_shock_strength", int(self.slider_impact_shock.value()))
        s.setValue("impact_echo_trail", int(self.check_impact_echo.isChecked()))
        s.setValue("impact_echo_trail_strength", int(self.slider_impact_echo.value()))
        s.setValue("impact_confetti", int(self.check_impact_confetti.isChecked()))
        s.setValue("impact_colorcycle", int(self.check_impact_colorcycle.isChecked()))
        s.setValue("impact_confetti_density", int(self.slider_impact_confetti.value()))
        s.setValue("impact_colorcycle_speed", int(self.slider_impact_colorcycle.value()))
        s.setValue("impact_zoom", int(self.check_impact_zoom.isChecked()))
        s.setValue("impact_zoom_amount", int(self.slider_impact_zoom.value()))
        s.setValue("impact_shake", int(self.check_impact_shake.isChecked()))
        s.setValue("impact_shake_strength", int(self.slider_impact_shake.value()))
        s.setValue("impact_fog", int(self.check_impact_fog.isChecked()))
        s.setValue("impact_fog_density", int(self.slider_impact_fog.value()))
        s.setValue("impact_fire_gold", int(self.check_impact_fire_gold.isChecked()))
        s.setValue("impact_fire_gold_intensity", int(self.slider_impact_fire_gold.value()))
        s.setValue("impact_fire_multi", int(self.check_impact_fire_multi.isChecked()))
        s.setValue("impact_fire_multi_intensity", int(self.slider_impact_fire_multi.value()))
        s.setValue("impact_random", int(self.check_impact_random.isChecked()))

        # Timed strobe (flash on time)
        if hasattr(self, "check_strobe_on_time"):
            s.setValue("strobe_on_time", int(self.check_strobe_on_time.isChecked()))
            try:
                s.setValue("strobe_on_time_times", json.dumps(self._get_strobe_times()))
            except Exception:
                s.setValue("strobe_on_time_times", "[]")

    # dialogs / helpers

    def _browse_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select music/audio file",
            "",
            "Audio files (*.mp3 *.wav *.flac *.m4a *.ogg);;All files (*.*)",
        )
        if path:
            self.edit_audio.setText(path)

    def _browse_video_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video file",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.mpg *.mpeg);;All files (*.*)",
        )
        if path:
            self.edit_video.setText(path)

    def _browse_video_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder with clips", "")
        if path:
            self.edit_video.setText(path)

    def _browse_video_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one or more video clips",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.mpg *.mpeg);;All files (*.*)",
        )
        if paths:
            # Join multiple paths with '|' so discovery can treat them as a list
            self.edit_video.setText("|".join(paths))

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select output folder", "")
        if path:
            self.edit_output.setText(path)

    def _open_output_in_media_explorer(self) -> None:
        """Jump to Media Explorer tab and scan this tool's output folder."""
        # Resolve output folder (default: output/videoclips)
        try:
            out_txt = self.edit_output.text().strip()
        except Exception:
            out_txt = ""
        if not out_txt:
            out_txt = os.path.join("output", "videoclips")

        out_dir = None
        try:
            from pathlib import Path as _P
            out_dir = _P(str(out_txt)).expanduser()
            try:
                out_dir = out_dir.resolve()
            except Exception:
                pass
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        except Exception:
            out_dir = None

        if out_dir is None:
            return

        # Preferred: route through the main window helper so all tabs share one implementation.
        mw = None
        try:
            mw = self.window()
        except Exception:
            mw = None

        try:
            if mw is not None and hasattr(mw, "open_media_explorer_folder"):
                mw.open_media_explorer_folder(str(out_dir), preset="videos", include_subfolders=False)
                return
        except Exception:
            pass

        # Fallback: open in OS explorer.
        try:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
            return
        except Exception:
            pass

        try:
            if os.name == "nt":
                os.startfile(str(out_dir))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out_dir)])
            else:
                subprocess.Popen(["xdg-open", str(out_dir)])
        except Exception:
            pass

    def _on_load_clips(self) -> None:
        """Load one or more video clips (or a folder) into the sources list."""
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Clips",
            "",
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.mpg *.mpeg);;All Files (*)",
            options=options,
        )
        video_input = ""
        if files:
            video_input = "|".join(files)
            # Update text field for consistency with the existing video input logic
            self.edit_video.setText(video_input)
        else:
            folder = QFileDialog.getExistingDirectory(self, "Select Clips Folder")
            if not folder:
                return
            video_input = folder
            self.edit_video.setText(video_input)

        # Discover clip sources using the same helper as the background scanner.
        try:
            sources = discover_video_sources(video_input, self._ffprobe)
        except Exception:
            sources = []

        if not sources:
            self._error("No clips", "Could not find any usable video clips in the selection.")
            return

        self.clip_sources = sources
        self._update_sources_list()

    def _add_images_to_sources(self, files) -> None:
        """Add image files as still-image sources."""
        if not files:
            return

        # Convert images to short clips with random duration
        for img_path in files:
            duration = random.uniform(1.5, 3.0)
            clip = ClipSource(path=img_path, duration=duration)
            clip.is_image = True  # Mark as image
            self.image_sources.append(clip)

        self._update_sources_list()

    def _on_load_images(self) -> None:
        """Load one or more image files and add as still-image inserts."""
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;All Files (*)",
            options=options,
        )
        self._add_images_to_sources(files)

    def _on_load_images_folder(self) -> None:
        """Load all supported images from a folder and add as still-image inserts."""
        folder = QFileDialog.getExistingDirectory(self, "Select Images Folder")
        if not folder:
            return

        import glob

        extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif", "*.webp"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(folder, ext)))
            files.extend(glob.glob(os.path.join(folder, ext.upper())))

        self._add_images_to_sources(files)


    def _error(self, title: str, msg: str) -> None:
        QMessageBox.critical(self, title, msg, QMessageBox.Ok)

    def _info(self, title: str, msg: str) -> None:
        QMessageBox.information(self, title, msg, QMessageBox.Ok)

    def _resolve_paths(self) -> Optional[Tuple[str, str, str]]:
        audio = self.edit_audio.text().strip()
        video = self.edit_video.text().strip()
        out_dir = self.edit_output.text().strip()

        if not audio or not os.path.isfile(audio):
            self._error("Missing audio", "Please select a valid music/audio file.")
            return None

        # video can be:
        # - a single file
        # - a folder
        # - or a '|' separated list of files (from Clip files... picker)
        if not video:
            self._error(
                "Missing video input",
                "Please select a valid video file, a folder containing clips, "
                "or use the Clip files... button to pick multiple clips.",
            )
            return None

        if "|" in video:
            # basic validation: at least one existing file in the list
            parts = [p.strip() for p in video.split("|") if p.strip()]
            if not parts or not any(os.path.isfile(p) for p in parts):
                self._error(
                    "Missing video input",
                    "None of the selected clip files could be found. Please re-select your clips.",
                )
                return None
        elif not (os.path.isfile(video) or os.path.isdir(video)):
            self._error(
                "Missing video input",
                "Please select a valid video file or a folder containing clips.",
            )
            return None
        if not out_dir:
            out_dir = os.path.join(os.path.dirname(audio), "output")
            self.edit_output.setText(out_dir)
        _ensure_dir(out_dir)
        return audio, video, out_dir

    # microclip mode mutual exclusivity

    def _on_micro_mode_changed(self, state: int) -> None:
        if not state:
            return
        sender = self.sender()
        toggles = [
            getattr(self, "check_micro_chorus", None),
            getattr(self, "check_micro_all", None),
            getattr(self, "check_micro_verses", None),
        ]
        for cb in toggles:
            if cb is None or cb is sender:
                continue
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)

    def _on_seed_toggle(self, state: int) -> None:
        enabled = state != 0
        self.spin_seed.setEnabled(enabled)

    def _on_slow_toggle(self, state: int) -> None:
        # Show or hide slow-motion controls when the main toggle changes.
        enabled = state != 0
        try:
            self.slow_options.setVisible(enabled)
        except Exception:
            pass

    def _on_slow_factor_changed(self, value: int) -> None:
        # Update the numeric label next to the slow-motion factor slider.
        try:
            factor = max(0.10, min(1.0, value / 100.0))
        except Exception:
            factor = 1.0
        try:
            self.label_slow_factor_value.setText(f"{factor:.2f}x")
        except Exception:
            pass
        # Also refresh cinematic controls that depend on slow-motion being enabled.
        try:
            self._update_cine_controls_enabled()
        except Exception:
            pass

    # --- cinematic effects helpers -------------------------------------

    def _on_cine_toggle(self, state: int) -> None:
        # Accept either Qt.CheckState values (0/1/2) or plain bool/int.
        enabled = bool(state)
        try:
            self.cine_options.setVisible(enabled)
        except Exception:
            pass
        try:
            self._update_cine_controls_enabled()
        except Exception:
            pass

    def _update_cine_controls_enabled(self) -> None:
        cine_enabled = getattr(self, "check_cine_enable", None)
        slow_enabled = getattr(self, "check_slow_enable", None)
        cine_on = bool(cine_enabled and cine_enabled.isChecked())
        slow_on = bool(slow_enabled and slow_enabled.isChecked())
        ramp_ok = cine_on and slow_on
        for w in (
            getattr(self, "check_cine_speed_ramp", None),
            getattr(self, "slider_cine_ramp_in", None),
            getattr(self, "slider_cine_ramp_out", None),
            getattr(self, "label_cine_ramp_in", None),
            getattr(self, "label_cine_ramp_out", None),
        ):
            if w is not None:
                try:
                    w.setEnabled(ramp_ok)
                except Exception:
                    pass


    def _disable_camera_motion_fx(self) -> None:
        """Hide and force-disable dolly-zoom / Ken Burns camera motion FX.

        The underlying widgets stay alive for backwards compatibility with
        saved presets/settings, but while this helper is active they are
        always hidden and turned off.
        """
        for name in (
            "check_cine_dolly",
            "check_cine_kenburns",
        ):
            cb = getattr(self, name, None)
            if cb is not None:
                try:
                    cb.setChecked(False)
                    cb.setEnabled(False)
                    cb.hide()
                except Exception:
                    pass

        for name, default in (
            ("slider_cine_dolly_strength", 50),
            ("slider_cine_kenburns_strength", 40),
        ):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.setValue(default)
                    w.setEnabled(False)
                    w.hide()
                except Exception:
                    pass

        for name in (
            "label_cine_dolly_strength",
            "label_cine_kenburns_strength",
        ):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.hide()
                except Exception:
                    pass

        combo = getattr(self, "combo_cine_motion_dir", None)
        if combo is not None:
            try:
                combo.setCurrentIndex(0)
                combo.setEnabled(False)
                combo.hide()
            except Exception:
                pass

    def _hide_image_sources_block(self) -> None:
        """Hide the still-image UI block (Load Images + segment interval slider).

        This keeps the underlying code paths intact (so older presets/settings
        won't crash), but removes the UI that lets users add still images.
        """
        # Force-disable still images for generator runs.
        try:
            self.image_sources = []
        except Exception:
            pass
        try:
            self.image_segment_interval = 0
        except Exception:
            pass

        # If the slider exists, force it to 0 (disabled) without firing signals.
        slider = getattr(self, "slider_image_interval", None)
        if slider is not None:
            try:
                slider.blockSignals(True)
                slider.setValue(0)
            except Exception:
                pass
            finally:
                try:
                    slider.blockSignals(False)
                except Exception:
                    pass

        # Hide the row widgets and their labels in the form.
        form = getattr(self, "_form_layout", None)
        for field_name in ("image_sources_row", "image_interval_row"):
            w = getattr(self, field_name, None)
            if w is None:
                continue
            try:
                w.setEnabled(False)
                w.hide()
            except Exception:
                pass
            if form is not None:
                try:
                    lbl = form.labelForField(w)
                    if lbl is not None:
                        lbl.hide()
                except Exception:
                    pass

        # Also hide any individual widgets if they exist (belt-and-suspenders).
        for name in ("btn_load_images", "btn_load_images_folder", "label_image_interval", "slider_image_interval"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.setEnabled(False)
                    w.hide()
                except Exception:
                    pass

        # Refresh the sources list so it won't show any lingering images.
        try:
            self._update_sources_list()
        except Exception:
            pass



    def _on_cine_freeze_len_changed(self, value: int) -> None:
        seconds = value / 100.0
        try:
            self.label_cine_freeze_len.setText(f"{seconds:.2f} s")
        except Exception:
            pass

    def _on_cine_freeze_zoom_changed(self, value: int) -> None:
        pct = value
        try:
            self.label_cine_freeze_zoom.setText(f"+{pct:d}%")
        except Exception:
            pass
    
    def _on_cine_tear_v_strength_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_cine_tear_v_strength.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_cine_tear_h_strength_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_cine_tear_h_strength.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_cine_color_cycle_speed_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 400
        # Clamp and snap to 50ms steps
        v = max(100, min(1000, v))
        v = int(round(v / 50.0) * 50)
        try:
            self.label_cine_color_cycle_speed.setText(f"{v:d} ms")
        except Exception:
            pass

    def _on_cine_reverse_len_changed(self, value: int) -> None:
        seconds = value / 100.0
        try:
            self.label_cine_reverse_len.setText(f"{seconds:.2f} s")
        except Exception:
            pass

    def _on_cine_ramp_in_changed(self, value: int) -> None:
        seconds = value / 100.0
        try:
            self.label_cine_ramp_in.setText(f"{seconds:.2f} s")
        except Exception:
            pass

    def _on_cine_ramp_out_changed(self, value: int) -> None:
        seconds = value / 100.0
        try:
            self.label_cine_ramp_out.setText(f"{seconds:.2f} s")
        except Exception:
            pass

    def _on_cine_boomerang_bounces_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 1
        v = max(1, min(9, v))
        try:
            self.label_cine_boomerang_bounces.setText(f"x{v:d}")
        except Exception:
            pass

    def _on_cine_pan916_speed_changed(self, value: int) -> None:
        """Snap the 9:16 pan-crop speed slider to 50ms steps and update the label."""
        try:
            v = int(value)
        except Exception:
            v = 400
        v = max(150, min(1000, v))
        v = int(round(v / 50.0) * 50)
        try:
            if getattr(self, "slider_cine_pan916_speed").value() != v:
                self.slider_cine_pan916_speed.blockSignals(True)
                self.slider_cine_pan916_speed.setValue(v)
                self.slider_cine_pan916_speed.blockSignals(False)
        except Exception:
            try:
                self.slider_cine_pan916_speed.blockSignals(False)
            except Exception:
                pass
        try:
            self.label_cine_pan916_speed.setText(f"{v:d} ms")
        except Exception:
            pass

    def _on_cine_pan916_parts_changed(self, value: int) -> None:
        """Update the slice-count slider label and clamp to the supported 2–6 range."""
        try:
            v = int(value)
        except Exception:
            v = 3
        v = max(2, min(6, v))
        try:
            if getattr(self, "slider_cine_pan916_parts").value() != v:
                self.slider_cine_pan916_parts.blockSignals(True)
                self.slider_cine_pan916_parts.setValue(v)
                self.slider_cine_pan916_parts.blockSignals(False)
        except Exception:
            try:
                self.slider_cine_pan916_parts.blockSignals(False)
            except Exception:
                pass
        # When random mode is enabled, this label shows a generic range instead of a fixed number.
        try:
            if hasattr(self, "check_cine_pan916_random") and self.check_cine_pan916_random.isChecked():
                self.label_cine_pan916_parts.setText("Random 2–6")
            else:
                self.label_cine_pan916_parts.setText(f"{v:d}")
        except Exception:
            pass

    def _on_cine_pan916_random_changed(self, state: int) -> None:
        """Enable/disable the Parts slider and adjust label when random is toggled."""
        try:
            is_random = bool(state)
        except Exception:
            try:
                is_random = self.check_cine_pan916_random.isChecked()
            except Exception:
                is_random = False

        # Disable the slider when random is on, since each hit will pick its own slice count.
        try:
            self.slider_cine_pan916_parts.setEnabled(not is_random)
        except Exception:
            pass

        # Update label to reflect mode.
        try:
            if is_random:
                self.label_cine_pan916_parts.setText("Random 2–6")
            else:
                self._on_cine_pan916_parts_changed(self.slider_cine_pan916_parts.value())
        except Exception:
            pass

    def _on_cine_mosaic_screens_changed(self, value: int) -> None:
        """Clamp Mosaic slider to 2–9 screens and refresh the label."""
        try:
            v = int(value)
        except Exception:
            v = 4
        # Clamp to supported range.
        if v < 2:
            v = 2
        if v > 9:
            v = 9
        # Keep the slider's value in sync with the clamped number.
        try:
            if self.slider_cine_mosaic_screens.value() != v:
                self.slider_cine_mosaic_screens.blockSignals(True)
                self.slider_cine_mosaic_screens.setValue(v)
                self.slider_cine_mosaic_screens.blockSignals(False)
        except Exception:
            pass
        # Update label; when random is enabled, show a generic message instead.
        try:
            if hasattr(self, "check_cine_mosaic_random") and self.check_cine_mosaic_random.isChecked():
                self.label_cine_mosaic_screens.setText("Random 2–9")
            else:
                self.label_cine_mosaic_screens.setText(f"{v} screens")
        except Exception:
            pass

    def _on_cine_mosaic_random_changed(self, state: int) -> None:
        """Enable/disable the Mosaic screens slider and adjust label when random is toggled."""
        try:
            is_random = bool(state)
        except Exception:
            try:
                is_random = self.check_cine_mosaic_random.isChecked()
            except Exception:
                is_random = False
        # Disable the slider when random is on, since each Mosaic hit will pick its own layout.
        try:
            self.slider_cine_mosaic_screens.setEnabled(not is_random)
        except Exception:
            pass
        # Update label to reflect mode.
        try:
            if is_random:
                self.label_cine_mosaic_screens.setText("Random 2–9")
            else:
                # Re-apply current slider value to refresh the label with a concrete number.
                self._on_cine_mosaic_screens_changed(self.slider_cine_mosaic_screens.value())
        except Exception:
            pass



    def _on_cine_multiply_screens_changed(self, value: int) -> None:
        """Clamp Multiply slider to 2–9 copies and refresh the label."""
        try:
            v = int(value)
        except Exception:
            v = 4
        # Clamp to supported range.
        if v < 2:
            v = 2
        if v > 9:
            v = 9
        try:
            if self.slider_cine_multiply_screens.value() != v:
                self.slider_cine_multiply_screens.blockSignals(True)
                self.slider_cine_multiply_screens.setValue(v)
                self.slider_cine_multiply_screens.blockSignals(False)
        except Exception:
            pass
        try:
            if hasattr(self, "check_cine_multiply_random") and self.check_cine_multiply_random.isChecked():
                self.label_cine_multiply_screens.setText("Random 2–9")
            else:
                self.label_cine_multiply_screens.setText(f"{v} copies")
        except Exception:
            pass

    def _on_cine_multiply_random_changed(self, state: int) -> None:
        """Enable/disable Multiply slider and adjust label when random is toggled."""
        try:
            is_random = bool(state)
        except Exception:
            try:
                is_random = self.check_cine_multiply_random.isChecked()
            except Exception:
                is_random = False
        try:
            self.slider_cine_multiply_screens.setEnabled(not is_random)
        except Exception:
            pass
        try:
            if is_random:
                self.label_cine_multiply_screens.setText("Random 2–9")
            else:
                self._on_cine_multiply_screens_changed(self.slider_cine_multiply_screens.value())
        except Exception:
            pass
    def _on_cine_rotate_degrees_changed(self, value: int) -> None:
        """Update label for the rotating screen max angle slider."""
        try:
            v = int(value)
        except Exception:
            v = 20
        v = max(5, min(90, v))
        try:
            self.label_cine_rotate_degrees.setText(f"±{v:d}°")
        except Exception:
            pass



    def _on_cine_dolly_strength_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 40
        v = max(10, min(100, v))
        try:
            self.label_cine_dolly_strength.setText(f"{v:d}%")
        except Exception:
            pass

    def _on_cine_kenburns_strength_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 10
        try:
            self.label_cine_kenburns_strength.setText(f"{v:d}%")
        except Exception:
            pass


    # --- section all on/off helpers -------------------------------------

    def _set_all_cinematic_toggles(self, enabled: bool) -> None:
        """Set all visible cinematic toggles to the given state."""
        targets = [
            "check_cine_freeze",
            "check_cine_tear_v",
            "check_cine_tear_h",
            "check_cine_color_cycle",
            "check_cine_stutter",
            "check_cine_reverse",
            "check_cine_speedup_forward",
            "check_cine_speedup_backward",
            "check_cine_speed_ramp",
            "check_cine_boomerang",
            "check_cine_pan916",
            "check_cine_pan916_random",
            "check_cine_mosaic",
            "check_cine_mosaic_random",
            "check_cine_flip",
            "check_cine_rotate",
            "check_cine_multiply",
            "check_cine_multiply_random",
        ]
        for name in targets:
            cb = getattr(self, name, None)
            if cb is None:
                continue
            try:
                cb.setChecked(bool(enabled))
            except Exception:
                pass

        # Keep hidden camera-motion FX forced off.
        try:
            self._disable_camera_motion_fx()
        except Exception:
            pass

    def _on_cine_all_onoff(self) -> None:
        """Toggle all cinematic checkboxes in this section."""
        targets = [
            getattr(self, "check_cine_freeze", None),
            getattr(self, "check_cine_tear_v", None),
            getattr(self, "check_cine_tear_h", None),
            getattr(self, "check_cine_color_cycle", None),
            getattr(self, "check_cine_stutter", None),
            getattr(self, "check_cine_reverse", None),
            getattr(self, "check_cine_speedup_forward", None),
            getattr(self, "check_cine_speedup_backward", None),
            getattr(self, "check_cine_speed_ramp", None),
            getattr(self, "check_cine_boomerang", None),
            getattr(self, "check_cine_pan916", None),
            getattr(self, "check_cine_pan916_random", None),
            getattr(self, "check_cine_mosaic", None),
            getattr(self, "check_cine_mosaic_random", None),
            getattr(self, "check_cine_flip", None),
            getattr(self, "check_cine_rotate", None),
            getattr(self, "check_cine_multiply", None),
            getattr(self, "check_cine_multiply_random", None),
        ]
        all_on = True
        any_cb = False
        for cb in targets:
            if cb is None:
                continue
            any_cb = True
            try:
                if not cb.isChecked():
                    all_on = False
                    break
            except Exception:
                all_on = False
                break
        if not any_cb:
            return
        self._set_all_cinematic_toggles(not all_on)

    def _set_all_impact_toggles(self, enabled: bool) -> None:
        """Set all visible break/drop impact toggles to the given state."""
        targets = [
            "check_impact_flash",
            "check_impact_shock",
            "check_impact_echo",
            "check_impact_confetti",
            "check_impact_colorcycle",
            "check_impact_zoom",
            "check_impact_shake",
            "check_impact_random",
        ]
        for name in targets:
            cb = getattr(self, name, None)
            if cb is None:
                continue
            try:
                cb.setChecked(bool(enabled))
            except Exception:
                pass

        # Ensure hidden legacy impact FX stay off.
        for name in ("check_impact_fog", "check_impact_fire_gold", "check_impact_fire_multi"):
            cb = getattr(self, name, None)
            if cb is None:
                continue
            try:
                cb.setChecked(False)
            except Exception:
                pass

    def _on_impact_all_onoff(self) -> None:
        """Toggle all break/drop impact checkboxes in this section."""
        targets = [
            getattr(self, "check_impact_flash", None),
            getattr(self, "check_impact_shock", None),
            getattr(self, "check_impact_echo", None),
            getattr(self, "check_impact_confetti", None),
            getattr(self, "check_impact_colorcycle", None),
            getattr(self, "check_impact_zoom", None),
            getattr(self, "check_impact_shake", None),
            getattr(self, "check_impact_random", None),
        ]
        all_on = True
        any_cb = False
        for cb in targets:
            if cb is None:
                continue
            any_cb = True
            try:
                if not cb.isChecked():
                    all_on = False
                    break
            except Exception:
                all_on = False
                break
        if not any_cb:
            return
        self._set_all_impact_toggles(not all_on)


    # --- timed strobe (flash on time) helpers -----------------------------
    def _strobe_times_sorted_unique(self, values: List[float]) -> List[float]:
        out: List[float] = []
        for v in values:
            try:
                fv = float(v)
            except Exception:
                continue
            if fv < 0.0:
                continue
            # de-dup with small tolerance
            dup = False
            for ex in out:
                if abs(ex - fv) < 0.01:
                    dup = True
                    break
            if not dup:
                out.append(fv)
        out.sort()
        return out

    def _set_strobe_times(self, times: List[float]) -> None:
        self._strobe_on_time_times = self._strobe_times_sorted_unique(times)
        try:
            self.list_strobe_times.clear()
        except Exception:
            return
        for t in self._strobe_on_time_times:
            item = QListWidgetItem(f"{t:.2f} s")
            item.setData(Qt.UserRole, float(t))
            self.list_strobe_times.addItem(item)

    def _get_strobe_times(self) -> List[float]:
        # Source of truth: internal list (kept in sync with the list widget).
        try:
            return list(self._strobe_on_time_times)
        except Exception:
            return []

    def _on_strobe_on_time_toggle(self, state: int) -> None:
        enabled = bool(state)
        try:
            self.strobe_time_options.setVisible(enabled)
        except Exception:
            pass
        if enabled:
            # Turning this on implies FX, so auto-release No FX.
            try:
                self._maybe_release_nofx()
            except Exception:
                pass

    def _on_strobe_time_add(self) -> None:
        try:
            t = float(self.spin_strobe_time.value())
        except Exception:
            t = 0.0
        times = self._get_strobe_times()
        times.append(t)
        self._set_strobe_times(times)
        try:
            self._maybe_release_nofx()
        except Exception:
            pass

    def _on_strobe_time_remove_selected(self) -> None:
        try:
            sel = self.list_strobe_times.selectedItems()
        except Exception:
            sel = []
        if not sel:
            return
        cur = self._get_strobe_times()
        rem = set()
        for it in sel:
            try:
                rem.add(float(it.data(Qt.UserRole)))
            except Exception:
                pass
        new = [t for t in cur if all(abs(t - r) >= 0.01 for r in rem)]
        self._set_strobe_times(new)

    def _on_strobe_time_clear_all(self) -> None:
        self._set_strobe_times([])




    # --- break impact FX helpers ----------------------------------------

    def _on_impact_flash_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_flash.setText(f"{v/100.0:.2f}")
        except Exception:
            pass
    def _on_impact_flash_speed_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 250
        # Snap to 50ms steps for nicer UX.
        v = int(round(v / 50.0) * 50)
        v = max(100, min(1000, v))
        try:
            if getattr(self, "slider_impact_flash_speed", None) is not None and self.slider_impact_flash_speed.value() != v:
                self.slider_impact_flash_speed.blockSignals(True)
                self.slider_impact_flash_speed.setValue(v)
        except Exception:
            pass
        finally:
            try:
                if getattr(self, "slider_impact_flash_speed", None) is not None:
                    self.slider_impact_flash_speed.blockSignals(False)
            except Exception:
                pass
        try:
            self.label_impact_flash_speed.setText(f"{v} ms")
        except Exception:
            pass


    def _on_impact_shock_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_shock.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_impact_confetti_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_confetti.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_impact_colorcycle_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_colorcycle.setText(f"{v/100.0:.2f}")
        except Exception:
            pass



    def _on_impact_echo_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_echo.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_impact_zoom_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_zoom.setText(f"{v:d}%")
        except Exception:
            pass

    def _on_impact_shake_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_shake.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_impact_fog_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_fog.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_impact_fire_gold_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_fire_gold.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_impact_fire_multi_changed(self, value: int) -> None:
        try:
            v = int(value)
        except Exception:
            v = 0
        try:
            self.label_impact_fire_multi.setText(f"{v/100.0:.2f}")
        except Exception:
            pass

    def _on_nofx_toggled(self, state: int) -> None:
        """Master kill switch for all FX. When enabled, turn off all other FX toggles."""
        enabled = bool(state)
        if not enabled:
            # When turning No FX off, we do not restore previous FX states; user can re-enable manually.
            return

        # While we are programmatically disabling other FX checkboxes, suppress the auto-release helper.
        self._nofx_guard = True
        try:
            # Turn off slow motion
            try:
                self.check_slow_enable.setChecked(False)
            except Exception:
                pass
            # Turn off cinematic effects
            try:
                self.check_cine_enable.setChecked(False)
            except Exception:
                pass
            # Turn off break impact FX
            try:
                self.check_impact_enable.setChecked(False)
            except Exception:
                pass
            # Turn off timed strobe
            try:
                if hasattr(self, "check_strobe_on_time"):
                    self.check_strobe_on_time.setChecked(False)
            except Exception:
                pass
            # Disable random transitions to avoid surprise flash / creative modes.
            try:
                self.check_trans_random.setChecked(False)
            except Exception:
                pass
            # Turn off music-player visuals overlay and strategies
            try:
                if hasattr(self, "check_visual_overlay"):
                    self.check_visual_overlay.setChecked(False)
            except Exception:
                pass
            try:
                if hasattr(self, "check_visual_strategy_segment"):
                    self.check_visual_strategy_segment.setChecked(False)
            except Exception:
                pass
            try:
                if hasattr(self, "check_visual_strategy_section"):
                    self.check_visual_strategy_section.setChecked(False)
            except Exception:
                pass
        finally:
            self._nofx_guard = False
    def _maybe_release_nofx(self) -> None:
        """If No FX is currently enabled and the user explicitly turns some FX back on,
        automatically release the master switch.

        This is gated by ``_nofx_guard`` so internal changes (like when enabling
        No FX itself or restoring settings) do not immediately flip it off again.
        """
        if getattr(self, "check_nofx", None) is None:
            return
        if self._nofx_guard:
            return
        if self.check_nofx.isChecked():
            # Avoid recursive signal storms
            self.check_nofx.blockSignals(True)
            try:
                self.check_nofx.setChecked(False)
            finally:
                self.check_nofx.blockSignals(False)

    def _on_check_timeline(self) -> None:
        """Switch to the Music timeline tab."""
        try:
            index = self.tabs.indexOf(self.timeline_panel)
            if index != -1:
                self.tabs.setCurrentIndex(index)
        except Exception:
            # Never let timeline navigation issues break the main tool.
            pass

    # --- Mini-timeline callbacks (per-section media links) -----------------

    def _find_section_index(self, section) -> Optional[int]:
        """Map a timeline section object back to its index in the analysis."""
        if self._analysis is None or not getattr(self._analysis, "sections", None):
            return None
        sections = list(self._analysis.sections)
        try:
            return sections.index(section)
        except ValueError:
            target = (
                float(getattr(section, "start", 0.0) or 0.0),
                float(getattr(section, "end", 0.0) or 0.0),
                str(getattr(section, "kind", "section") or "section"),
            )
            for idx, sec in enumerate(sections):
                cur = (
                    float(getattr(sec, "start", 0.0) or 0.0),
                    float(getattr(sec, "end", 0.0) or 0.0),
                    str(getattr(sec, "kind", "section") or "section"),
                )
                if cur == target:
                    return idx
        return None

    def _update_timeline_media_label(self, section, path: Optional[str]) -> None:
        """Ask the timeline panel to show the attached media name, if available."""
        if not hasattr(self, "timeline_panel"):
            return
        try:
            label = os.path.basename(path) if path else ""
            if hasattr(self.timeline_panel, "set_section_media_label"):
                # type: ignore[attr-defined]
                self.timeline_panel.set_section_media_label(section, label)
        except Exception:
            # Never let UI decoration issues break the main tool.
            pass

    def _on_timeline_add(self, section) -> None:
        """Right-click 'Add' on the mini-timeline: attach a clip or image."""
        idx = self._find_section_index(section)
        if idx is None:
            return

        filters = (
            "Video files (*.mp4 *.mov *.mkv *.avi *.webm *.mpg *.mpeg *.m4v);;"
            "Image files (*.png *.jpg *.jpeg *.bmp *.gif *.webp);;"
            "All files (*.*)"
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video or image for this section",
            "",
            filters,
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        is_image = ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")

        if is_image:
            # Nominal duration for stills; real pacing follows the music.
            duration = 2.5
        else:
            dur = _ffprobe_duration(self._ffprobe, path)
            if dur is None or dur <= 0:
                try:
                    duration = float(getattr(self._analysis, "duration", 0.0) or 0.0)
                except Exception:
                    duration = 10.0
                if duration <= 0:
                    duration = 10.0
            else:
                duration = float(dur)

        override = ClipSource(path=path, duration=duration)
        # Remember whether this override came from an image or a video.
        try:
            override.is_image = bool(is_image)  # type: ignore[attr-defined]
        except Exception:
            pass

        self._section_media[idx] = override
        self._update_timeline_media_label(section, path)

    def _on_timeline_remove(self, section) -> None:
        """Right-click 'Remove' on the mini-timeline: clear attached media."""
        idx = self._find_section_index(section)
        if idx is None:
            return
        self._section_media.pop(idx, None)
        self._update_timeline_media_label(section, None)

    def _on_timeline_select(self, section) -> None:
        """Section selection from the mini-timeline (hook kept for future use)."""
        # TimelinePanel already highlights the matching row in its own table.
        return

    # analysis

    def _on_analyze(self) -> None:
        # If clips or images have been attached to the music timeline,
        # warn that a fresh analysis will wipe those overrides.
        try:
            has_overrides = bool(getattr(self, "_section_media", {}))
        except Exception:
            has_overrides = False
        if has_overrides:
            from PySide6.QtWidgets import QMessageBox
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Warning")
            box.setText(
                "Analyzing will remove the video/image clips you added to the timeline.\n\n"
                "Do you want to continue?"
            )
            btn_yes = box.addButton("Yes", QMessageBox.YesRole)
            btn_no = box.addButton("No", QMessageBox.NoRole)
            btn_keep = box.addButton("Use existing analysis", QMessageBox.RejectRole)
            box.setDefaultButton(btn_yes)
            box.exec()
            clicked = box.clickedButton()
            if clicked is btn_no:
                return
            if clicked is btn_keep:
                # Skip re-analysis and keep the existing analysis and timeline clips.
                return

        resolved = self._resolve_paths()
        if not resolved:
            return
        audio, _, _ = resolved
        self.progress.setValue(0)
        self.progress.setFormat("Analyzing music...")
        # Sync analysis config with current UI
        self._analysis_config.sensitivity = self.slider_sens.value()
        try:
            self._analysis = analyze_music(audio, self._ffmpeg, self._analysis_config)
        except Exception as e:
            self._analysis = None
            self.progress.setValue(0)
            self.progress.setFormat("Ready.")
            self._error("Analysis failed", str(e))
            return

        # Reset any section-specific media overrides for this new analysis
        self._section_media.clear()

        beats = len(self._analysis.beats)
        secs = len(self._analysis.sections)
        self.progress.setValue(30)
        self.progress.setFormat(f"Analysis complete: {beats} beats, {secs} sections.")

        # Save current paths and analysis-related options
        self._save_settings()

        # Compact analysis summary under the progress bar
        try:
            total_dur = self._analysis.duration
        except Exception:
            total_dur = None

        parts = []
        if total_dur is not None:
            parts.append(f"Total duration: {total_dur:.1f}s")
        parts.append(f"Beats detected: {beats}")
        parts.append(f"Sections: {secs}")
        # Single‑line summary only; full structure lives in the Music timeline tab.
        self.label_summary.setText(" \u2022 ".join(parts))
        try:
            # Reveal the shortcut once we have something meaningful to show.
            self.btn_check_timeline.setVisible(True)
        except Exception:
            pass

        # Update timeline tab with the latest analysis, if available.
        try:
            if hasattr(self, "timeline_panel") and hasattr(self.timeline_panel, "set_analysis") and self._analysis is not None:
                self.timeline_panel.set_analysis(self._analysis)
        except Exception:
            # Never let timeline issues break the main tool.
            pass
    def _target_resolution(
        self, sources: List[ClipSource], video_input: str
    ) -> Optional[Tuple[int, int]]:
        mode = self.combo_res.currentIndex()

        # Mode 0: single video keep source
        if os.path.isfile(video_input) and mode == 0:
            return None

        if not sources:
            return None

        def get_res(path: str) -> Optional[Tuple[int, int]]:
            cmd = [
                self._ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                path,
            ]
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                data = json.loads(out)
                streams = data.get("streams") or []
                if not streams:
                    return None
                s = streams[0]
                return int(s.get("width", 0)), int(s.get("height", 0))
            except Exception:
                return None

        res_list = []
        for s in sources:
            r = get_res(s.path)
            if r and r[0] > 0 and r[1] > 0:
                res_list.append(r)

        if not res_list:
            return None

        if mode == 1:  # highest
            w = max(r[0] for r in res_list)
            h = max(r[1] for r in res_list)
            return w, h
        if mode == 2:  # lowest
            w = min(r[0] for r in res_list)
            h = min(r[1] for r in res_list)
            return w, h
        if mode == 3:
            return 854, 480
        if mode == 4:
            return 1280, 720
        if mode == 5:
            return 1920, 1080
        return None


    def _set_direct_run_active(self, active: bool) -> None:
        """Update the Generate button label while a DIRECT RUN is in progress."""
        try:
            self._direct_run_active = bool(active)
        except Exception:
            pass
        try:
            btn = getattr(self, "btn_generate", None)
            if btn is not None:
                btn.setText("Generating Clip" if active else "Generate Clip")
        except Exception:
            pass


    def _get_fixed_resolution(self) -> Tuple[int, int] | None:
        text = self.combo_res.currentText()
        if "1080×1920" in text or "1080p (1080×1920)" in text:
            return 1080, 1920
        if "720×1280" in text or "720p (720×1280)" in text:
            return 720, 1280
        if "480×854" in text or "480p (480×854)" in text:
            return 480, 854
        if "1080p (1920×1080)" in text or "Fixed: 1080p" in text:
            return 1920, 1080
        if "720p (1280×720)" in text or "Fixed: 720p" in text:
            return 1280, 720
        if "480p (854×480)" in text or "Fixed: 480p" in text:
            return 854, 480
        return None  # Not a fixed resolution

    def _on_generate(self) -> None:
        # Prevent starting a new render or scan while one is already running.
        if self._worker is not None and self._worker.isRunning():
            self._error("Busy", "A render is already running.")
            return
        if self._scan_worker is not None and self._scan_worker.isRunning():
            self._error("Busy", "Clip scanning is already running.")
            return

        resolved = self._resolve_paths()
        if not resolved:
            return
        audio, video, out_dir = resolved


        # DIRECT RUN UX: toast + button label while render is running.
        direct_run = not bool(getattr(self, "_queue_requested", False))
        if direct_run:
            try:
                self._set_direct_run_active(True)
            except Exception:
                pass
            try:
                self._show_toast("DIRECT RUN started, don't start other heavy CPU jobs while it is running.")
            except Exception:
                pass

        # Always analyze fresh for every creation so we never reuse
        # segments from a previous music track, even if paths and
        # settings look the same.
        self._on_analyze()
        if self._analysis is None:
            try:
                if getattr(self, "_direct_run_active", False):
                    self._set_direct_run_active(False)
            except Exception:
                pass
            return
        # Stash paths so we can continue once the clip scan finishes.
        self._pending_audio = audio
        self._pending_video = video
        self._pending_out_dir = out_dir

        # Optional minimum clip length configured in the UI.
        min_len: float | None = None
        if self.check_min_clip.isChecked():
            try:
                min_len = float(self.spin_min_clip.value())
            except Exception:
                min_len = None

        # Start background scan of folder / clip list so the UI stays responsive.
        self.progress.setValue(0)
        self.progress.setFormat("Scanning video clips...")

        self._scan_worker = SourceScanWorker(
            video_input=video,
            ffprobe=self._ffprobe,
            min_clip_length=min_len,
            parent=self,
        )
        # Reuse the same progress bar for scanning as for rendering.
        self._scan_worker.progress.connect(self._on_worker_progress)
        self._scan_worker.finished_ok.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    
    def _on_queue_generate(self) -> None:
        """Queue the current Music Clip Creator job instead of rendering immediately."""
        try:
            self._queue_requested = True
        except Exception:
            pass
        # Reuse the same pipeline as Generate (analyze + scan + build timeline),
        # but the final step will enqueue a headless render job.
        self._on_generate()

    def _continue_generate_with_sources(
        self,
        audio: str,
        video: str,
        out_dir: str,
        sources: List[ClipSource],
    ) -> None:
        # Measure actual audio duration so 'fill full music length' can match the real track
        audio_duration = _ffprobe_duration(self._ffprobe, audio)

        # Master No-FX state
        nofx = self.check_nofx.isChecked()

        idx = self.combo_fx.currentIndex()
        if nofx:
            fx_level = "none"
        elif idx == 0:
            fx_level = "minimal"
        elif idx == 1:
            fx_level = "moderate"
        else:
            fx_level = "high"

        if self.check_micro_chorus.isChecked():
            micro_mode = 1
        elif self.check_micro_all.isChecked():
            micro_mode = 2
        elif getattr(self, "check_micro_verses", None) is not None and self.check_micro_verses.isChecked():
            micro_mode = 3
        else:
            micro_mode = 0

        beats_per = max(1, self.spin_beats_per_seg.value())

        trans_idx = self.combo_transitions.currentIndex()
        transition_mode = trans_idx  # 0,1,2
        if nofx:
            # Force hard cuts only when No-FX is enabled.
            transition_mode = 1

        clip_order_mode = self.combo_clip_order.currentIndex()  # 0,1,2

        force_full_length = self.check_full_length.isChecked()
        # Determine the random seed used for this run.
        # If 'Use fixed seed' is enabled, use the value from the spin box.
        # Otherwise, generate a new random seed and show it in the UI.
        if self.check_use_seed.isChecked():
            seed_enabled = True
            seed_value = int(self.spin_seed.value())
        else:
            seed_value = random.randint(0, 999999999)
            seed_enabled = True
            try:
                # Reflect the seed that was actually used back into the UI.
                self.spin_seed.setValue(seed_value)
            except Exception:
                pass

        # Slow motion configuration from UI
        slow_enabled = self.check_slow_enable.isChecked() and not nofx
        slow_sections: List[str] = []
        if slow_enabled:
            if self.check_slow_intro.isChecked():
                slow_sections.append("intro")
            if self.check_slow_break.isChecked():
                slow_sections.append("break")
            if self.check_slow_chorus.isChecked():
                slow_sections.append("chorus")
            if self.check_slow_drop.isChecked():
                slow_sections.append("drop")
            if self.check_slow_outro.isChecked():
                slow_sections.append("outro")
        slow_factor = max(0.10, min(1.0, self.slider_slow_factor.value() / 100.0)) if slow_enabled else 1.0
        slow_random = bool(self.check_slow_random.isChecked()) if slow_enabled else False

        # Safety: when still images are used, disable slow‑motion segments and
        # force a single, stable transition mode instead of random transitions.
        has_images = bool(getattr(self, "image_sources", []))
        if has_images and slow_enabled:
            slow_enabled = False
            slow_sections = []
            slow_factor = 1.0
            slow_random = False

        # Compute whether random transitions are allowed for this run.
        transition_random_enabled = (self.check_trans_random.isChecked() and not nofx)
        if has_images:
            # With still images, random transitions can cause ffmpeg issues,
            # so we silently disable them and stick to Motion blur whip‑cuts.
            transition_random_enabled = False
            transition_mode = 5

        # Break impact FX configuration (first beat after each break)
        impact_enabled = self.check_impact_enable.isChecked() and not nofx
        impact_flash = self.check_impact_flash.isChecked()
        impact_shock = self.check_impact_shock.isChecked()
        impact_echo_trail = self.check_impact_echo.isChecked()
        impact_confetti = self.check_impact_confetti.isChecked()
        impact_color_cycle = self.check_impact_colorcycle.isChecked()
        impact_zoom = self.check_impact_zoom.isChecked()
        impact_shake = self.check_impact_shake.isChecked()
        impact_fog = self.check_impact_fog.isChecked()
        impact_fire_gold = self.check_impact_fire_gold.isChecked()
        impact_fire_multi = self.check_impact_fire_multi.isChecked()
        impact_random = self.check_impact_random.isChecked()

        impact_flash_strength = self.slider_impact_flash.value() / 100.0
        impact_flash_speed_ms = int(self.slider_impact_flash_speed.value()) if hasattr(self, "slider_impact_flash_speed") else 250
        impact_shock_strength = self.slider_impact_shock.value() / 100.0
        impact_echo_trail_strength = self.slider_impact_echo.value() / 100.0
        impact_confetti_density = self.slider_impact_confetti.value() / 100.0
        impact_color_cycle_speed = self.slider_impact_colorcycle.value() / 100.0
        impact_zoom_amount = self.slider_impact_zoom.value() / 100.0
        impact_shake_strength = self.slider_impact_shake.value() / 100.0
        impact_fog_density = self.slider_impact_fog.value() / 100.0
        impact_fire_gold_intensity = self.slider_impact_fire_gold.value() / 100.0
        impact_fire_multi_intensity = self.slider_impact_fire_multi.value() / 100.0

        # Hidden break-impact FX (fog, fireworks) are always disabled.
        impact_fog = False
        impact_fire_gold = False
        impact_fire_multi = False
        impact_fog_density = 0.0
        impact_fire_gold_intensity = 0.0
        impact_fire_multi_intensity = 0.0

        segments = build_timeline(
            self._analysis,
            sources,
            fx_level=fx_level,
            microclip_mode=micro_mode,
            beats_per_segment=beats_per,
            transition_mode=transition_mode,
            clip_order_mode=clip_order_mode,
            force_full_length=force_full_length,
            seed_enabled=seed_enabled,
            seed_value=seed_value,
            transition_random=transition_random_enabled,
            transition_modes_enabled=sorted(self._enabled_transition_modes),
            intro_transitions_only=bool(getattr(self, 'check_intro_transitions_only', None) and self.check_intro_transitions_only.isChecked()),
            slow_motion_enabled=slow_enabled,
            slow_motion_factor=slow_factor,
            slow_motion_sections=slow_sections,
            slow_motion_random=slow_random,
            cine_enable=(bool(self.check_cine_enable.isChecked()) and not nofx),
            cine_freeze=bool(self.check_cine_freeze.isChecked()),
            cine_stutter=bool(self.check_cine_stutter.isChecked()),
            cine_reverse=bool(self.check_cine_reverse.isChecked()),
            cine_speedup_forward=bool(self.check_cine_speedup_forward.isChecked()),
            cine_speedup_forward_factor=float(self.spin_cine_speedup_forward.value()),
            cine_speedup_backward=bool(self.check_cine_speedup_backward.isChecked()),
            cine_speedup_backward_factor=float(self.spin_cine_speedup_backward.value()),
            cine_speed_ramp=bool(self.check_cine_speed_ramp.isChecked()),
            cine_freeze_len=self.slider_cine_freeze_len.value() / 100.0,
            cine_freeze_zoom=self.slider_cine_freeze_zoom.value() / 100.0,
            cine_tear_v=bool(self.check_cine_tear_v.isChecked()),
            cine_tear_v_strength=self.slider_cine_tear_v_strength.value() / 100.0,
            cine_tear_h=bool(self.check_cine_tear_h.isChecked()),
            cine_tear_h_strength=self.slider_cine_tear_h_strength.value() / 100.0,
            cine_color_cycle=bool(self.check_cine_color_cycle.isChecked()),
            cine_color_cycle_speed_ms=int(self.slider_cine_color_cycle_speed.value()),
            cine_stutter_repeats=int(self.spin_cine_stutter_repeats.value()),
            cine_reverse_len=self.slider_cine_reverse_len.value() / 100.0,
            cine_ramp_in=self.slider_cine_ramp_in.value() / 100.0,
            cine_ramp_out=self.slider_cine_ramp_out.value() / 100.0,
            cine_boomerang=bool(self.check_cine_boomerang.isChecked()),
            cine_boomerang_bounces=int(self.slider_cine_boomerang_bounces.value()),
            cine_dimension=bool(self.check_cine_dimension.isChecked()),
            cine_pan916=bool(self.check_cine_pan916.isChecked()),
            cine_pan916_speed_ms=int(self.slider_cine_pan916_speed.value()),
            cine_pan916_parts=int(self.slider_cine_pan916_parts.value()),
            cine_pan916_transparent=bool(self.check_cine_pan916_transparent.isChecked()),
            cine_pan916_random=bool(getattr(self, "check_cine_pan916_random", None) and self.check_cine_pan916_random.isChecked()),
            cine_mosaic=bool(self.check_cine_mosaic.isChecked()),
            cine_mosaic_screens=int(self.slider_cine_mosaic_screens.value()),
            cine_mosaic_random=bool(self.check_cine_mosaic_random.isChecked()),
            cine_flip=bool(self.check_cine_flip.isChecked()),
            cine_rotate=bool(self.check_cine_rotate.isChecked()),
            cine_rotate_max_degrees=float(self.slider_cine_rotate_degrees.value()),
            cine_multiply=bool(self.check_cine_multiply.isChecked()),
            cine_multiply_screens=int(self.slider_cine_multiply_screens.value()),
            cine_multiply_random=bool(self.check_cine_multiply_random.isChecked()),
            cine_dolly=bool(self.check_cine_dolly.isChecked()),
            cine_dolly_strength=self.slider_cine_dolly_strength.value() / 100.0,
            cine_kenburns=bool(self.check_cine_kenburns.isChecked()),
            cine_kenburns_strength=self.slider_cine_kenburns_strength.value() / 100.0,
            cine_motion_dir=int(self.combo_cine_motion_dir.currentIndex()),
            audio_duration=audio_duration,
            impact_enable=impact_enabled,
            impact_flash=impact_flash,
            impact_shock=impact_shock,
            impact_echo_trail=impact_echo_trail,
            impact_confetti=impact_confetti,
            impact_color_cycle=impact_color_cycle,
            impact_zoom=impact_zoom,
            impact_shake=impact_shake,
            impact_fog=impact_fog,
            impact_fire_gold=impact_fire_gold,
            impact_fire_multi=impact_fire_multi,
            impact_random=impact_random,
            impact_flash_strength=impact_flash_strength,
            impact_flash_speed_ms=impact_flash_speed_ms,
            strobe_on_time=bool(getattr(self, 'check_strobe_on_time', None) and self.check_strobe_on_time.isChecked() and not nofx),
            strobe_on_time_times=self._get_strobe_times() if (getattr(self, 'check_strobe_on_time', None) and self.check_strobe_on_time.isChecked() and not nofx) else None,
            impact_shock_strength=impact_shock_strength,
            impact_echo_trail_strength=impact_echo_trail_strength,
            impact_confetti_density=impact_confetti_density,
            impact_color_cycle_speed=impact_color_cycle_speed,
            impact_zoom_amount=impact_zoom_amount,
            impact_shake_strength=impact_shake_strength,
            impact_fog_density=impact_fog_density,
            impact_fire_gold_intensity=impact_fire_gold_intensity,
            impact_fire_multi_intensity=impact_fire_multi_intensity,
            image_sources=self.image_sources,
            section_overrides=self._section_media,
            image_segment_interval=int(getattr(self, "image_segment_interval", 0) or 0),
        )
        if not segments:
            self._error("Timeline empty", "Failed to build a video timeline.")
            try:
                if getattr(self, "_direct_run_active", False):
                    self._set_direct_run_active(False)
            except Exception:
                pass
            return
        fixed = self._get_fixed_resolution()


        if fixed:


            target_res = fixed


        else:


            target_res = self._target_resolution(sources, video)

        if self._worker is not None and self._worker.isRunning():
            self._error("Busy", "A render is already running.")
            return

        # Persist current options before kicking off the render
        self._save_settings()

        self.progress.setValue(0)
        self.progress.setFormat("Starting render...")

        # Derive visual strategy enum from UI:
        # 0 = single visual for whole track
        # 1 = new random visual every segment
        # 2 = one visual per section type (intro/verse/chorus/break/outro)
        visual_strategy = 0
        if getattr(self, "check_visual_strategy_segment", None) and self.check_visual_strategy_segment.isChecked():
            visual_strategy = 1
        elif getattr(self, "check_visual_strategy_section", None) and self.check_visual_strategy_section.isChecked():
            visual_strategy = 2

        
        # If requested, enqueue instead of render-now.
        if getattr(self, "_queue_requested", False):
            try:
                self._queue_requested = False
            except Exception:
                pass
            try:
                self._enqueue_render_job(
                    audio=audio,
                    out_dir=out_dir,
                    analysis=self._analysis,
                    segments=segments,
                    target_res=target_res,
                    transition_mode=transition_mode,
                    use_visual_overlay=bool(getattr(self, "check_visual_overlay", None) and self.check_visual_overlay.isChecked()),
                    visual_strategy=visual_strategy,
                    visual_section_overrides=getattr(self, "_visual_section_overrides", None) if visual_strategy == 2 else None,
                    visual_overlay_opacity=float(getattr(self, "visual_overlay_opacity", 0.25)),
                )
            except Exception as e:
                try:
                    self._error("Queue error", str(e))
                except Exception:
                    pass
            return

        self._worker = RenderWorker(
            audio_path=audio,
            output_dir=out_dir,
            analysis=self._analysis,
            segments=segments,
            ffmpeg=self._ffmpeg,
            ffprobe=self._ffprobe,
            target_resolution=target_res,
            fit_mode=self.combo_fit.currentIndex(),
            transition_mode=transition_mode,
            intro_fade=self.check_intro_fade.isChecked(),
            outro_fade=self.check_outro_fade.isChecked(),
            keep_source_bitrate=bool(getattr(self, "check_keep_source_bitrate", None) and self.check_keep_source_bitrate.isChecked()),
            use_visual_overlay=bool(getattr(self, "check_visual_overlay", None) and self.check_visual_overlay.isChecked()),
            visual_strategy=visual_strategy,
            visual_section_overrides=getattr(self, "_visual_section_overrides", None) if visual_strategy == 2 else None,
            visual_overlay_opacity=float(getattr(self, "visual_overlay_opacity", 0.25)),
            strobe_on_time_times=self._get_strobe_times() if (getattr(self, "check_strobe_on_time", None) and self.check_strobe_on_time.isChecked() and not nofx) else None,
            strobe_flash_strength=float(getattr(self, "slider_impact_flash", None).value() / 100.0) if hasattr(self, "slider_impact_flash") else 0.0,
            strobe_flash_speed_ms=int(getattr(self, "slider_impact_flash_speed", None).value()) if hasattr(self, "slider_impact_flash_speed") else 250,
        )
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.finished_ok.connect(self._on_worker_finished)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.start()


    
    def _enqueue_render_job(
        self,
        audio: str,
        out_dir: str,
        analysis: "MusicAnalysisResult",
        segments: List["TimelineSegment"],
        target_res: Optional[Tuple[int, int]],
        transition_mode: int,
        use_visual_overlay: bool,
        visual_strategy: int,
        visual_section_overrides: Optional[Dict[str, Optional[str]]],
        visual_overlay_opacity: float,
    ) -> None:
        """Serialize the current timeline and enqueue a headless render job in Queue."""
        try:
            from helpers.queue_adapter import enqueue_tool_job
        except Exception:
            # Fallback when helpers is not a package (dev runs)
            from queue_adapter import enqueue_tool_job  # type: ignore

        # Deterministic output name for the queued job (so the Queue can point to the produced file).
        base = os.path.splitext(os.path.basename(audio))[0]
        safe_base = _sanitize_stem(base)
        ts = datetime.now().strftime("%d%m%H%M%S")
        out_name = f"{safe_base}_clip_{ts}.mp4"
        out_final = os.path.join(out_dir, out_name)

        payload_dir = os.path.join(out_dir, "_payloads")
        _ensure_dir(payload_dir)
        payload_path = os.path.join(payload_dir, f"mclip_job_{ts}.json")

        payload = {
            "audio_path": audio,
            "output_dir": out_dir,
            "ffmpeg": self._ffmpeg,
            "ffprobe": self._ffprobe,
            "target_resolution": list(target_res) if target_res else None,
            "fit_mode": int(self.combo_fit.currentIndex()),
            "transition_mode": int(transition_mode),
            "intro_fade": bool(self.check_intro_fade.isChecked()),
            "outro_fade": bool(self.check_outro_fade.isChecked()),
            "keep_source_bitrate": bool(getattr(self, "check_keep_source_bitrate", None) and self.check_keep_source_bitrate.isChecked()),
            "use_visual_overlay": bool(use_visual_overlay),
            "visual_strategy": int(visual_strategy),
            "visual_section_overrides": visual_section_overrides,
            "visual_overlay_opacity": float(visual_overlay_opacity),
            "out_name_override": out_name,
            "strobe_on_time_times": self._get_strobe_times() if (getattr(self, "check_strobe_on_time", None) and self.check_strobe_on_time.isChecked() and not (getattr(self, "check_nofx", None) and self.check_nofx.isChecked())) else None,
            "strobe_flash_strength": float(getattr(self, "slider_impact_flash", None).value() / 100.0) if hasattr(self, "slider_impact_flash") else 0.0,
            "strobe_flash_speed_ms": int(getattr(self, "slider_impact_flash_speed", None).value()) if hasattr(self, "slider_impact_flash_speed") else 250,
            "analysis": _analysis_to_dict(analysis),
            "segments": _segments_to_list(segments),
        }
        with open(payload_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        # Build a portable python command to run the headless job.
        py = sys.executable or "python"
        payload_path_esc = payload_path.replace("\\", "\\\\").replace("'", "\\'")
        cmd_code = (
            "import sys\n"
            f"p=r'{payload_path_esc}'\n"
            "try:\n"
            "    import helpers.auto_music_sync as m\n"
            "except Exception:\n"
            "    import auto_music_sync as m\n"
            "sys.exit(m.run_queue_payload(p))\n"
        )
        cmd = [py, "-u", "-c", cmd_code]

        args = {
            "cmd": cmd,
            "outfile": out_final,
            "label": f"Music Clip Creator: {safe_base}",
        }
        try:
            # Ensure the command runs from app root so imports resolve.
            args["cwd"] = str(Path(".").resolve())
        except Exception:
            pass

        # Enqueue as a tools_ffmpeg job so the existing worker logic can run it.
        ok = enqueue_tool_job("tools_ffmpeg", audio, out_dir, args, priority=560)
        if not ok:
            raise RuntimeError("Failed to enqueue job (job file could not be written).")

        try:
            self.progress.setValue(0)
            self.progress.setFormat("Queued.")
        except Exception:
            pass

        # Switch to Queue tab automatically (same behavior as Tools tab).
        try:
            tabs = self.parent()
            while tabs is not None and not isinstance(tabs, QTabWidget):
                tabs = tabs.parent()
            if isinstance(tabs, QTabWidget):
                for i in range(tabs.count()):
                    if str(tabs.tabText(i)).lower().startswith("queue"):
                        tabs.setCurrentIndex(i)
                        break
        except Exception:
            pass

        try:
            self._show_toast("Added to Queue, follow progress in the worker or the Queue Tab")
        except Exception:
            pass

    def _update_sources_list(self) -> None:
        if not hasattr(self, "list_sources"):
            return
        self.list_sources.clear()
        for s in getattr(self, "clip_sources", []):
            item = QListWidgetItem(s.path)
            try:
                item.setToolTip(f"Duration: {s.duration:.1f} s (Video)")
            except Exception:
                item.setToolTip("Video clip")
            self.list_sources.addItem(item)
        for s in getattr(self, "image_sources", []):
            item = QListWidgetItem(s.path)
            try:
                item.setToolTip(f"Duration: {s.duration:.1f} s (Image)")
            except Exception:
                item.setToolTip("Image")
            self.list_sources.addItem(item)

    def _on_scan_finished(self, sources: list) -> None:
        # Scan worker finished successfully; clear handle first.
        self._scan_worker = None

        if not sources:
            self._error("No video clips", "Could not find any usable video sources.")
            self.progress.setValue(0)
            self.progress.setFormat("Ready.")
            self._pending_audio = None
            self._pending_video = None
            self._pending_out_dir = None
            try:
                if getattr(self, "_direct_run_active", False):
                    self._set_direct_run_active(False)
            except Exception:
                pass
            return

        if not (self._pending_audio and self._pending_video and self._pending_out_dir):
            # Missing context; abort gracefully.
            self._error("Internal error", "Missing pending paths after clip scan.")
            self.progress.setValue(0)
            self.progress.setFormat("Ready.")
            self._pending_audio = None
            self._pending_video = None
            self._pending_out_dir = None
            try:
                if getattr(self, "_direct_run_active", False):
                    self._set_direct_run_active(False)
            except Exception:
                pass
            return

        # Cache discovered sources for the UI list
        try:
            self.clip_sources = list(sources)
            self._update_sources_list()
        except Exception:
            pass

        audio = self._pending_audio
        video = self._pending_video
        out_dir = self._pending_out_dir

        # Clear pending state before moving on.
        self._pending_audio = None
        self._pending_video = None
        self._pending_out_dir = None

        # Continue with the same logic that used to live in _on_generate.
        self._continue_generate_with_sources(audio, video, out_dir, sources)

    def _on_scan_failed(self, msg: str) -> None:
        # Background scan failed; reset state and show a friendly error.
        self._scan_worker = None
        self.progress.setValue(0)
        self.progress.setFormat("Ready.")
        self._pending_audio = None
        self._pending_video = None
        self._pending_out_dir = None


        # Reset UI state (DIRECT RUN label / queued flag) after scan errors.
        try:
            self._queue_requested = False
        except Exception:
            pass
        try:
            if getattr(self, "_direct_run_active", False):
                self._set_direct_run_active(False)
        except Exception:
            pass

        if msg.startswith("__no_clips__"):
            self._error("No video clips", "Could not find any usable video sources.")
        elif msg.startswith("__all_short__"):
            try:
                val = float(msg.split(":", 1)[1])
            except Exception:
                val = 0.0
            if val > 0:
                self._error(
                    "All clips too short",
                    f"All discovered clips were shorter than {val:.1f}s and were ignored.",
                )
            else:
                self._error(
                    "All clips too short",
                    "All discovered clips were shorter than the configured minimum length.",
                )
        else:
            self._error("Clip scan failed", msg)

    # worker callbacks

    def _on_worker_progress(self, pct: int, msg: str) -> None:
        self.progress.setValue(max(0, min(100, pct)))
        if msg:
            self.progress.setFormat(msg)

    def _on_worker_finished(self, out_path: str) -> None:
        try:
            if getattr(self, "_direct_run_active", False):
                self._set_direct_run_active(False)
        except Exception:
            pass

        self.progress.setValue(100)
        self.progress.setFormat("Done.")
        QMessageBox.information(
            self,
            "Music Clip created",
            f"Finished generating music clip:\n{out_path}",
            QMessageBox.Ok,
        )
        self._worker = None

    def _on_worker_failed(self, msg: str) -> None:
        try:
            if getattr(self, "_direct_run_active", False):
                self._set_direct_run_active(False)
        except Exception:
            pass

        self.progress.setValue(0)
        self.progress.setFormat("Failed.")
        self._error("Music Clip Creator failed", msg)
        self._worker = None




class OneClickVideoClipTab(QWidget):
    """Dedicated tab wrapper for the Music Clip Creator with a vertical scrollbar."""
    def __init__(self, main=None, parent=None):
        super().__init__(parent)
        self.main = main

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # --- Fancy banner at the top ---
        banner_wrap = QWidget(self)
        bw = QVBoxLayout(banner_wrap)
        bw.setContentsMargins(6, 4, 6, 0)
        bw.setSpacing(0)

        banner = QLabel('Create a music videoclip', banner_wrap)
        banner.setObjectName('mvBanner')
        banner.setAlignment(Qt.AlignCenter)
        banner.setFixedHeight(48)
        banner.setStyleSheet(
            "#mvBanner {"
            " font-size: 15px;"
            " font-weight: 600;"
            " padding: 8px 17px;"
            " border-radius: 12px;"
            " margin: 0 0 6px 0;"
            " color: white;"
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #cd28ff,"
            "   stop:0.5 #9f4df2,"
            "   stop:1 #28ffbb"
            " );"
            " letter-spacing: 0.5px;"
            "}"
        )
        bw.addWidget(banner)

        # Apply persisted banner settings (hide / greyscale) immediately.
        try:
            s = QSettings("FrameVision", "FrameVision")
            en = bool(s.value("banner_enabled", True, type=bool))
            col = bool(s.value("banner_colored", True, type=bool))
            try:
                banner_wrap.setVisible(en)
            except Exception:
                pass
            try:
                banner.setVisible(en)
            except Exception:
                pass
            if not col:
                try:
                    orig = banner.styleSheet() or ""
                    banner.setProperty("fv_banner_orig_style", orig)
                    # Force grey gradient (same logic as Settings tab helper)
                    key = "background:"
                    idx2 = orig.find(key)
                    grey_block = (
                        " background: qlineargradient("
                        "   x1:0, y1:0, x2:1, y2:0,"
                        "   stop:0 #e0e0e0,"
                        "   stop:0.5 #b0b0b0,"
                        "   stop:1 #707070"
                        " );"
                    )
                    if idx2 == -1:
                        banner.setStyleSheet(orig + grey_block)
                    else:
                        semi = orig.find(";", idx2)
                        if semi == -1:
                            semi = len(orig)
                        banner.setStyleSheet(orig[:idx2] + grey_block + orig[semi+1:])
                except Exception:
                    pass
        except Exception:
            pass

        outer.addWidget(banner_wrap)
        outer.addSpacing(4)

        sc = QScrollArea(self)
        sc.setWidgetResizable(True)
        sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sc.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        sc.setFrameShape(QFrame.NoFrame)

        cont = QWidget(sc)
        sc.setWidget(cont)

        lay = QVBoxLayout(cont)
        lay.setContentsMargins(6, 4, 6, 6)
        lay.setSpacing(6)

        self.inner = AutoMusicSyncWidget(parent=cont, sticky_footer=True)
        lay.addWidget(self.inner)

        outer.addWidget(sc, 1)

        # Sticky footer: keep the main action buttons visible while scrolling.
        try:
            div = QFrame(self)
            div.setFrameShape(QFrame.HLine)
            div.setFrameShadow(QFrame.Sunken)
            outer.addWidget(div)
        except Exception:
            pass
        try:
            outer.addWidget(self.inner.footer_bar, 0)
        except Exception:
            pass

# ------------------------- integration entry point -------------------------


def install_auto_music_sync_tool(parent, section) -> AutoMusicSyncWidget:
    """Install the Music Clip Creator UI into a CollapsibleSection."""
    content = getattr(section, "content", None)
    parent_widget = content if content is not None else section

    # Wrap in an internal scroll area + sticky footer so the action buttons
    # stay visible while scrolling.
    tab = OneClickVideoClipTab(main=parent, parent=parent_widget)
    tab.setObjectName("MusicClipCreatorTab")
    try:
        tab.inner.setObjectName("MusicClipCreatorWidget")
    except Exception:
        pass

    if content is not None:
        layout = content.layout()
        if layout is None:
            layout = QVBoxLayout(content)
    else:
        layout = section.layout()
        if layout is None:
            layout = QVBoxLayout(section)

    layout.addWidget(tab)
    return tab.inner


def cleanup_temp_dir(output_path: str) -> None:
    """Best-effort removal of temp files for the music clip creator.

    This looks for a sibling 'temp' directory next to the chosen output file
    and removes any intermediate segment files if possible.
    """
    import shutil
    from pathlib import Path

    if not output_path:
        return
    p = Path(output_path)
    # If user picked a folder, we still look for a 'temp' subdir.
    parent = p if p.is_dir() else p.parent
    temp_dir = parent / "temp"
    if temp_dir.is_dir():
        try:
            for child in temp_dir.iterdir():
                if child.is_file() and child.suffix in {".mp4", ".mkv", ".mov", ".ts"}:
                    child.unlink(missing_ok=True)
        except Exception:
            # Don't crash the UI if cleanup fails.
            pass


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = AutoMusicSyncWidget()
    w.setWindowTitle("Music Clip Creator (Standalone Test)")
    w.resize(900, 620)
    w.show()
    sys.exit(app.exec())