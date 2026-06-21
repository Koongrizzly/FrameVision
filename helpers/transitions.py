"""Transition utilities for Music Clip Creator.

This module centralizes:
- Transition labels shown in UI
- Random transition selection mapping
- Per-segment visual transition filter snippets (fade-safe)
- The Manage Transitions dialog helper
"""

from __future__ import annotations

import os
import traceback
from datetime import datetime
from typing import Optional, Set, List, Any
import random

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QCheckBox,
    QDialogButtonBox,
    QMessageBox,
)

# Keep labels in one place (used by dropdown + manage dialog).
TRANSITION_NAMES: List[str] = [
    "Soft film dissolves (REAL)",
    "Hard cuts",
    "Scale punch (zoom)",
    "Chromatic glitch",
    "VHS analog warp",
    "Motion blur whip-cuts",
    "Flash/dip mix",
    "Creative festival mix (strobes + hits)",
    "Directional slide (REAL)",
    "Wipe test",
    "Smooth zoom crossfade (REAL)",
]


def manage_transitions_dialog(parent: Any, enabled_modes: Set[int]) -> Optional[Set[int]]:
    """Show the Manage Transitions dialog. Returns the new set, or None if cancelled."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Manage transitions")
    layout = QVBoxLayout(dlg)

    info = QLabel(
        "Select which transition styles can be used when\n"
        "'Random transitions' is enabled."
    )
    info.setWordWrap(True)
    layout.addWidget(info)

    checkboxes: List[QCheckBox] = []
    for i, name in enumerate(TRANSITION_NAMES):
        cb = QCheckBox(name, dlg)
        cb.setChecked(i in enabled_modes)
        layout.addWidget(cb)
        checkboxes.append(cb)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dlg)
    layout.addWidget(buttons)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)

    if dlg.exec() != QDialog.Accepted:
        return None

    selected = {i for i, cb in enumerate(checkboxes) if cb.isChecked()}
    if not selected:
        QMessageBox.warning(
            parent,
            "No transitions selected",
            "At least one transition style must be enabled for random mode.",
            QMessageBox.Ok,
        )
        return None
    return selected


def update_transition_visibility(check_random: Any, label_widget: Any, combo_widget: Any) -> None:
    """Hide/show the dropdown row depending on random toggle."""
    try:
        is_random = bool(check_random.isChecked())
    except Exception:
        return
    visible = not is_random
    try:
        label_widget.setVisible(visible)
    except Exception:
        pass
    try:
        combo_widget.setVisible(visible)
    except Exception:
        pass


def select_transition_code(mode_for_segment: int, energy: str) -> str:
    """Map UI transition mode to an internal transition code.

    REAL between-clip transitions must be stitched with ffmpeg `xfade`.
    """
    # UI indices (TRANSITION_LABELS):
    # 0 Soft film dissolves (REAL)      -> stitched dissolve
    # 1 Hard cuts                       -> none
    # 2 Scale punch (zoom)              -> per-clip
    # 3 Chromatic glitch                -> per-clip
    # 4 VHS analog warp                 -> per-clip
    # 5 Motion blur whip-cuts           -> per-clip
    # 6 Flash/dip mix                   -> mixed
    # 7 Creative festival mix           -> mixed
    # 8 Directional slide (REAL)        -> stitched slide
    # 9 Wipe test                       -> stitched wipe
    # 10 Smooth zoom crossfade (REAL)   -> stitched zoomin

    if mode_for_segment == 1:
        return "none"

    # REAL stitched transitions
    if mode_for_segment == 0:
        return "t_exposure_dissolve"
    if mode_for_segment == 8:
        return "t_slide"
    if mode_for_segment == 9:
        return "t_wipe"
    if mode_for_segment == 10:
        return "t_zoom_xfade"

    # Mixed presets (keep behavior)
    if mode_for_segment == 6:
        if energy == "high" and random.random() < 0.55:
            return "flashcut"
        if energy in ("mid", "high") and random.random() < 0.25:
            return "flashcolor"
        return "none"

    if mode_for_segment == 7:
        if energy == "high":
            return random.choice(["flashcolor", "rgb_split", "whip", "none"])
        if energy == "mid":
            return random.choice(["flashcolor", "whip", "none"])
        return "none"

    # Per-clip FX
    if mode_for_segment == 2:
        return "t_zoom_pulse"
    if mode_for_segment == 3:
        return "t_rgb_split"
    if mode_for_segment == 4:
        return "t_vhs"
    if mode_for_segment == 5:
        return "t_motion_blur"

    return "none"

    if mode_for_segment == 2:  # Mixed (white flash + cuts)
        if energy == "high" and random.random() < 0.5:
            return "flashcut"  # white flash
        return "none"

    if mode_for_segment == 3:  # Creative mix (color flashes + subtle whip)
        if energy == "high":
            if random.random() < 0.6:
                return "flashcolor"
            return "rgb_split"
        if energy == "mid":
            r = random.random()
            if r < 0.3:
                return "flashcolor"
            if r < 0.5:
                return "whip"
            return "none"
        return "none"

    if mode_for_segment == 4:
        return "t_zoom_pulse"
    if mode_for_segment == 5:
        return "t_rgb_split"
    if mode_for_segment == 6:
        return "t_vhs"
    if mode_for_segment == 7:
        return "t_motion_blur"
    if mode_for_segment == 8:
        return "t_push"
    if mode_for_segment == 9:
        return "t_wipe"
    if mode_for_segment == 10:
        return "t_smooth_zoom"

    return "none"


def transition_vf_parts(seg: Any, i: int) -> List[str]:
    """Return a list of video filter parts to apply for seg.transition."""
    vf_parts: List[str] = []
    tr = getattr(seg, "transition", None) or "none"
    try:
        dur = float(getattr(seg, "duration", 0.0) or 0.0)
    except Exception:
        dur = 0.0

    if tr == "flashcut":
        # White flash: short brightness pop at the start (~30–60ms).
        flash_d = max(0.06, min(0.06, dur / 6.0 if dur else 0.06))
        vf_parts.append(f"eq=brightness=0.45:enable='between(t,0,{flash_d:.3f})'")

    elif tr == "flashcolor":
        # Colored flash: short hue-shifted flash at the start.
        flash_d = max(0.4, min(1.2, dur * 0.8 if dur else 0.4))
        hue_shift = (i * 37) % 360
        vf_parts.append(
            f"eq=brightness=0.35:enable='between(t,0,{flash_d:.3f})',"
            f"hue=h={hue_shift}:enable='between(t,0,{flash_d:.3f})'"
        )

    elif tr == "whip":
        # Zoom pulse / scale pulse: beat-style zoom using scale+crop.
        base_period = 0.3
        zoom_amount = 0.02
        zoom_expr = f"1+{zoom_amount}*abs(sin(2*3.14159*t/{base_period:.3f}))"
        vf_parts.append(f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,crop=iw:ih")

    elif tr == "t_zoom_pulse":
        base_period = 0.5
        zoom_amount = 0.03
        zoom_expr = f"1+{zoom_amount}*abs(sin(2*3.14159*t/{base_period:.3f}))"
        vf_parts.append(f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,crop=iw:ih")

    elif tr == "t_rgb_split":
        vf_parts.append("chromashift=u=2:v=-2")

    elif tr == "t_vhs":
        vf_parts.append(
            "noise=alls=15:allf=t+u,"
            "scale=iw:ih/2:flags=neighbor,"
            "scale=iw:ih:flags=neighbor"
        )

    elif tr == "t_motion_blur":
        vf_parts.append("tblend=all_mode=average,framestep=1")

    elif tr == "t_push":
        push_len = min(0.75, max(0.20, dur * 0.50 if dur else 0.20))
        vf_parts.append(
            "scale=iw*1.10:ih*1.10:eval=frame,"
            f"crop=iw:ih:x='(iw*0.10)*min(t/{push_len:.3f},1)':y='0'"
        )

    elif tr == "t_wipe":
        # Real between-clip wipe is applied during stitching; no per-segment filter here.
        pass

    elif tr == "t_smooth_zoom":
        total = max(0.20, float(dur or 1.0))
        zoom_expr = "1+0.15*t/{:.3f}".format(total)
        vf_parts.append(f"scale=iw*({zoom_expr}):ih*({zoom_expr}):eval=frame,crop=iw:ih")

    # else: none
    return vf_parts


# ------------------------------ logging -----------------------------------

def _project_root() -> str:
    # helpers/transitions.py -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_logs_dir() -> str:
    logs_dir = os.path.join(_project_root(), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def write_log(prefix: str, text: str, ext: str = ".log") -> str:
    """Write a log file under <project_root>/logs and return its path."""
    try:
        logs_dir = _ensure_logs_dir()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prefix = "".join(c if (c.isalnum() or c in ("-", "_")) else "_" for c in (prefix or "log"))
        filename = f"{safe_prefix}_{ts}{ext if ext.startswith('.') else '.' + ext}"
        path = os.path.join(logs_dir, filename)
        with open(path, "w", encoding="utf-8", errors="replace") as f:
            f.write(text or "")
        return path
    except Exception:
        return ""


def log_exception(prefix: str, extra: str = "") -> str:
    """Log the current exception traceback (+ optional extra) and return its path."""
    try:
        msg = traceback.format_exc()
        if extra:
            msg = (extra + "\n\n" + msg) if msg else extra
        return write_log(prefix, msg, ext=".txt")
    except Exception:
        return ""
