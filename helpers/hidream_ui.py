#!/usr/bin/env python3
from __future__ import annotations


# --- FrameVision media-explorer results opener ------------------------------
def _fv_open_results_in_media_explorer(widget, folder, preset="images") -> bool:
    """Open/scan a results folder in FrameVision Media Explorer when embedded.

    Falls back to the operating-system file explorer when the main FrameVision
    helper is not available (for standalone tool runs).
    """
    try:
        from pathlib import Path as _Path
        _folder = _Path(folder).expanduser()
        try:
            _folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        _folder_s = str(_folder)
    except Exception:
        return False

    def _try_main(_mw) -> bool:
        try:
            if _mw is not None and hasattr(_mw, "open_media_explorer_folder"):
                try:
                    _mw.open_media_explorer_folder(_folder_s, preset=preset, include_subfolders=False)
                    return True
                except TypeError:
                    kwargs = {"include_subfolders": False}
                    if preset == "images":
                        kwargs.update({"want_images": True, "want_videos": False, "want_audio": False})
                    elif preset == "videos":
                        kwargs.update({"want_images": False, "want_videos": True, "want_audio": False})
                    elif preset == "audio":
                        kwargs.update({"want_images": False, "want_videos": False, "want_audio": True})
                    _mw.open_media_explorer_folder(_folder_s, **kwargs)
                    return True
        except Exception:
            pass
        return False

    try:
        _w = widget
        while _w is not None:
            if _try_main(_w):
                return True
            try:
                _w = _w.parent()
            except Exception:
                break
    except Exception:
        pass

    try:
        from PySide6.QtWidgets import QApplication as _QApplication
        _app = _QApplication.instance()
        if _app is not None:
            for _w in _app.topLevelWidgets():
                if _try_main(_w):
                    return True
    except Exception:
        pass

    try:
        from PySide6.QtGui import QDesktopServices as _QDesktopServices
        from PySide6.QtCore import QUrl as _QUrl
        _QDesktopServices.openUrl(_QUrl.fromLocalFile(_folder_s))
        return True
    except Exception:
        pass

    try:
        import os as _os, sys as _sys, subprocess as _subprocess
        if _os.name == "nt":
            _os.startfile(_folder_s)  # type: ignore[attr-defined]
        elif _sys.platform == "darwin":
            _subprocess.Popen(["open", _folder_s])
        else:
            _subprocess.Popen(["xdg-open", _folder_s])
        return True
    except Exception:
        return False
# ---------------------------------------------------------------------------

import os
import re
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QSize, QProcess, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QPlainTextEdit, QProgressBar, QSpinBox,
    QDoubleSpinBox, QScrollArea, QSplitter, QTabWidget, QTextEdit, QToolButton,
    QVBoxLayout, QWidget, QGroupBox,
)

def _load_theme_helpers():
    try:
        from .themes import apply_theme as _apply, list_themes as _list
        return _apply, _list
    except Exception:
        try:
            from themes import apply_theme as _apply, list_themes as _list
            return _apply, _list
        except Exception:
            return None, None


def _load_hud_colorizer():
    try:
        from .hud_colorizer import auto_install_hud_colorizer as _install
        return _install
    except Exception:
        try:
            from hud_colorizer import auto_install_hud_colorizer as _install
            return _install
        except Exception:
            return None

_THEME_APPLY, _THEME_LIST = _load_theme_helpers()
_HUD_INSTALL = _load_hud_colorizer()
FALLBACK_THEME_NAMES = ["Evening", "Day", "Night"]

APP_TITLE = "HiDream Studio"

MODEL_DEFAULTS = {
    "base": {
        "label": "Base / Full BF16",
        "folder": "HiDream-O1-Image-BF16",
        "variant": "full",
        "steps": 50,
        "guidance_scale": 5.0,
        "shift": 3.0,
        "scheduler": "flash",
        "timesteps": "none",
        "note": "Base/full model. CFG works. Preferred default: Euler/Flash, 50 steps.",
    },
    "base_fp8": {
        "label": "Base / Full FP8",
        "folder": "HiDream-O1-Image-FP8",
        "variant": "full",
        "steps": 50,
        "guidance_scale": 5.0,
        "shift": 3.0,
        "scheduler": "flash",
        "timesteps": "none",
        "note": "Base/full FP8 model. Same behavior as Base BF16 with lower VRAM use.",
    },
    "dev": {
        "label": "Dev BF16",
        "folder": "HiDream-O1-Image-Dev-BF16",
        "variant": "dev",
        "steps": 28,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler": "flash",
        "timesteps": "dev",
        "note": "Dev model. CFG/negative prompts are effectively disabled; use 28-step Dev timesteps.",
    },
    "dev_2604_bf16": {
        "label": "Dev 2604 BF16",
        "folder": "HiDream-O1-Image-Dev-2604-BF16",
        "variant": "dev",
        "steps": 28,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler": "flash",
        "timesteps": "dev",
        "note": "Updated Dev 2604 BF16 model. CFG/negative prompts are disabled; use 28-step Dev timesteps. Installs as a separate folder so the old Dev model stays available.",
    },
    "dev_fp8": {
        "label": "Dev FP8",
        "folder": "HiDream-O1-Image-Dev-FP8",
        "variant": "dev",
        "steps": 28,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler": "flash",
        "timesteps": "dev",
        "note": "Dev FP8 model. Lowest VRAM option; CFG/negative prompts are effectively disabled.",
    },
}

RESOLUTION_PRESETS = [
    ("Landscape 4:3 — 640×480", 640, 480),
    ("Landscape 4:3 — 1024×768", 1024, 768),
    ("Landscape wide — 832×480", 832, 480),
    ("Landscape 16:9-ish — 1024×576", 1024, 576),
    ("Landscape 16:9-ish — 1280×704", 1280, 704),
    ("Landscape 16:9-ish — 1600×896", 1600, 896),
    ("Landscape 16:9-ish — 1920×1088", 1920, 1088),
    ("Landscape 16:9 — 2560×1440", 2560, 1440),
    ("Portrait 3:4 — 480×640", 480, 640),
    ("Portrait 3:4 — 768×1024", 768, 1024),
    ("Portrait tall — 480×832", 480, 832),
    ("Portrait 9:16-ish — 576×1024", 576, 1024),
    ("Portrait 9:16-ish — 704×1280", 704, 1280),
    ("Portrait 9:16-ish — 896×1600", 896, 1600),
    ("Portrait 9:16-ish — 1088×1920", 1088, 1920),
    ("Portrait 9:16 — 1440×2560", 1440, 2560),
    ("Square — 512×512", 512, 512),
    ("Square — 768×768", 768, 768),
    ("Square — 1024×1024", 1024, 1024),
    ("Square — 1536×1536", 1536, 1536),
    ("Square — 2048×2048", 2048, 2048),
]


REFERENCE_SAFE_RESOLUTION_PRESETS = [
    ("1600 × 896", 1600, 896),
    ("1920 × 1088", 1920, 1088),
    ("896 × 1600", 896, 1600),
    ("1088 × 1920", 1088, 1920),
    ("1024 × 1024", 1024, 1024),
    ("1536 × 1536", 1536, 1536),
]

REFERENCE_SAFE_DEFAULTS = {
    "edit": (1600, 896),
    "multi": (1600, 896),
}

LEGACY_RESOLUTION_PRESET_MAP = {
    (1280, 720): (1280, 704),
    (1600, 900): (1600, 896),
    (1920, 1080): (1920, 1088),
    (720, 1280): (704, 1280),
    (900, 1600): (896, 1600),
    (1080, 1920): (1088, 1920),
}

MULTI_REFERENCE_ROLES = [
    "Main subject",
    "Additional subject",
    "Style",
    "Clothing / look",
    "Object / prop",
    "Background / environment",
    "General reference",
]


def candidate_roots() -> list[Path]:
    starts = [Path(__file__).resolve().parent, Path.cwd().resolve()]
    roots: list[Path] = []
    for start in starts:
        p = start
        for _ in range(6):
            if p not in roots:
                roots.append(p)
            if p.parent == p:
                break
            p = p.parent
    return roots


def find_first(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def discover_paths() -> dict[str, Path | None]:
    roots = candidate_roots()
    env_candidates, runner_candidates, hidream_root_candidates = [], [], []
    for root in roots:
        env_candidates.extend([
            # HiDream must use its own portable FrameVision env.
            # Do not fall back to .images_models, .hidream_bf16, or loose root envs.
            root / "environments" / ".hidream_dev" / "python.exe",
        ])
        runner_candidates.extend([
            root / "models" / "hidream_bf16" / "run_hidream.py",
            root / "models" / "hidream_bf16" / "run_hidream_bf16.py",
            root / "models" / "hidream" / "run_hidream.py",
        ])
        hidream_root_candidates.extend([
            root / "models" / "hidream_bf16",
            root / "models" / "hidream",
        ])
    hidream_root = find_first(hidream_root_candidates)
    return {
        "env_python": find_first(env_candidates),
        "runner": find_first(runner_candidates),
        "hidream_root": hidream_root,
        "repo_dir": hidream_root / "HiDream-O1-Image" if hidream_root else None,
        "output_dir": hidream_root / "results" if hidream_root else None,
    }


PATHS = discover_paths()
ENV_PY = PATHS["env_python"]
RUNNER = PATHS["runner"]
CLI_PATH = Path(__file__).resolve().with_name("hidream_cli.py")
HIDREAM_ROOT = PATHS["hidream_root"]
DEFAULT_OUTPUT = PATHS["output_dir"] or (Path(__file__).resolve().parents[1] / "models" / "hidream_bf16" / "results")
DEFAULT_OFFLOAD_FOLDER = Path(__file__).resolve().parents[1] / "temp" / "hidream_offload"
SETTINGS_PATH = Path(__file__).resolve().parents[1] / "presets" / "setsave" / "hidream_ui.json"
QUEUE_PATH = Path(__file__).resolve().parents[1] / "presets" / "setsave" / "hidream_queue.json"


class ImagePreview(QLabel):
    def __init__(self, text: str = "No image loaded") -> None:
        super().__init__(text)
        self._path: Path | None = None
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 280)
        self.setStyleSheet("QLabel { background: #111827; color: #d1d5db; border: 1px solid #374151; border-radius: 10px; padding: 8px; }")
        self.setScaledContents(False)

    def set_image(self, path: str | Path | None) -> None:
        self._path = Path(path) if path else None
        if not self._path or not self._path.exists():
            self._pixmap = None
            self.setPixmap(QPixmap())
            self.setText("No image loaded")
            return
        pm = QPixmap(str(self._path))
        if pm.isNull():
            self._pixmap = None
            self.setText(f"Could not load image:\n{self._path}")
            return
        self._pixmap = pm
        self._refresh()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if not self._pixmap:
            return
        self.setPixmap(self._pixmap.scaled(self.size() - QSize(18, 18), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ReferenceList(QListWidget):
    def __init__(self, preview: ImagePreview) -> None:
        super().__init__()
        self.preview = preview
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(120, 90))
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSpacing(8)
        self.setMinimumHeight(140)
        self.itemClicked.connect(self._clicked)

    def add_images(self, paths: list[str]) -> None:
        for p in paths:
            path = Path(p)
            if not path.exists():
                continue
            item = QListWidgetItem(path.name)
            item.setData(Qt.UserRole, str(path))
            pm = QPixmap(str(path))
            if not pm.isNull():
                item.setIcon(pm.scaled(120, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.addItem(item)
        if self.count() and not self.currentItem():
            self.setCurrentRow(0)
            self.preview.set_image(self.item(0).data(Qt.UserRole))

    def paths(self) -> list[str]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def clear_images(self) -> None:
        self.clear()
        self.preview.set_image(None)

    def remove_selected(self) -> None:
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        if self.count():
            self.setCurrentRow(0)
            self.preview.set_image(self.item(0).data(Qt.UserRole))
        else:
            self.preview.set_image(None)

    def _clicked(self, item: QListWidgetItem) -> None:
        self.preview.set_image(item.data(Qt.UserRole))



class MultiReferenceList(QListWidget):
    def __init__(self, preview: ImagePreview) -> None:
        super().__init__()
        self.preview = preview
        self.setSelectionMode(QListWidget.SingleSelection)
        self.setSpacing(6)
        self.setMinimumHeight(260)
        self.currentItemChanged.connect(self._current_item_changed)

    def add_images(self, paths: list[str]) -> None:
        first_new_row = self.count()
        for p in paths:
            path = Path(p)
            if not path.exists():
                continue
            item = QListWidgetItem()
            item.setData(Qt.UserRole, str(path))
            item.setSizeHint(QSize(260, 82))

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(6, 4, 6, 4)
            row_layout.setSpacing(8)

            thumb = QLabel()
            thumb.setFixedSize(76, 58)
            thumb.setAlignment(Qt.AlignCenter)
            thumb.setStyleSheet("QLabel { background: #111827; border: 1px solid #374151; border-radius: 6px; }")
            pm = QPixmap(str(path))
            if not pm.isNull():
                thumb.setPixmap(pm.scaled(72, 54, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                thumb.setText("Image")
            row_layout.addWidget(thumb)

            info_col = QVBoxLayout()
            name = QLabel(path.name)
            name.setToolTip(str(path))
            name.setWordWrap(False)
            role = QComboBox()
            for role_name in MULTI_REFERENCE_ROLES:
                role.addItem(role_name, role_name)
            role.currentIndexChanged.connect(self._role_changed)
            info_col.addWidget(name)
            info_col.addWidget(role)
            row_layout.addLayout(info_col, 1)

            item.setData(Qt.UserRole + 1, role)
            self.addItem(item)
            self.setItemWidget(item, row)

        if self.count() and self.currentRow() < 0:
            self.setCurrentRow(first_new_row if first_new_row < self.count() else 0)

    def _role_changed(self, *_args) -> None:
        window = self.window()
        if hasattr(window, "save_settings") and not getattr(window, "_loading_settings", False):
            window.save_settings()

    def paths(self) -> list[str]:
        return [self.item(i).data(Qt.UserRole) for i in range(self.count())]

    def references(self) -> list[dict]:
        refs: list[dict] = []
        for i in range(self.count()):
            item = self.item(i)
            combo = item.data(Qt.UserRole + 1)
            role = combo.currentData() if isinstance(combo, QComboBox) else "General reference"
            refs.append({"path": item.data(Qt.UserRole), "role": str(role or "General reference")})
        return refs

    def clear_images(self) -> None:
        self.clear()
        self.preview.set_image(None)

    def remove_selected(self) -> None:
        row = self.currentRow()
        if row < 0:
            return
        self.takeItem(row)
        if self.count():
            self.setCurrentRow(min(row, self.count() - 1))
        else:
            self.preview.set_image(None)

    def move_selected(self, direction: int) -> None:
        row = self.currentRow()
        new_row = row + direction
        if row < 0 or new_row < 0 or new_row >= self.count():
            return
        item = self.takeItem(row)
        self.insertItem(new_row, item)
        # Rebuild the item widget because takeItem detaches it.
        path = Path(str(item.data(Qt.UserRole)))
        combo = item.data(Qt.UserRole + 1)
        current_role = combo.currentData() if isinstance(combo, QComboBox) else "General reference"
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(6, 4, 6, 4)
        row_layout.setSpacing(8)
        thumb = QLabel()
        thumb.setFixedSize(76, 58)
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setStyleSheet("QLabel { background: #111827; border: 1px solid #374151; border-radius: 6px; }")
        pm = QPixmap(str(path))
        if not pm.isNull():
            thumb.setPixmap(pm.scaled(72, 54, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            thumb.setText("Image")
        row_layout.addWidget(thumb)
        info_col = QVBoxLayout()
        name = QLabel(path.name)
        name.setToolTip(str(path))
        role = QComboBox()
        for role_name in MULTI_REFERENCE_ROLES:
            role.addItem(role_name, role_name)
        index = role.findData(current_role)
        if index >= 0:
            role.setCurrentIndex(index)
        role.currentIndexChanged.connect(self._role_changed)
        info_col.addWidget(name)
        info_col.addWidget(role)
        row_layout.addLayout(info_col, 1)
        item.setData(Qt.UserRole + 1, role)
        self.setItemWidget(item, row_widget)
        self.setCurrentRow(new_row)
        self.preview.set_image(path)

    def _current_item_changed(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None) -> None:
        if current:
            self.preview.set_image(current.data(Qt.UserRole))
        else:
            self.preview.set_image(None)

class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, expanded: bool = False) -> None:
        super().__init__()
        self.toggle_btn = QToolButton()
        self.toggle_btn.setText(title)
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(expanded)
        self.toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_btn.toggled.connect(self._set_expanded_internal)
        self.content = content
        self.content.setVisible(expanded)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.toggle_btn)
        layout.addWidget(self.content)

    def _set_expanded_internal(self, expanded: bool) -> None:
        self.toggle_btn.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)

    def is_expanded(self) -> bool:
        return self.toggle_btn.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.toggle_btn.setChecked(expanded)


class HiDreamUI(QMainWindow):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        try:
            self.setProperty("_fv_skip_restore", True)
            self.setProperty("_fv_skip_snapshot", True)
        except Exception:
            pass
        self.setWindowTitle(APP_TITLE)
        self.resize(1320, 840)
        self.process: QProcess | None = None
        self.last_output: Path | None = None
        self.queue_jobs: list[dict] = []
        self.current_queue_job_id: str | None = None
        self._stopping_current_job = False
        self._loading_settings = False
        self._syncing_generation_controls = False
        self._syncing_negative_prompt = False
        self._switch_defaults_on_model_change = True
        self._last_reference_folder = ""
        self._last_dialog_folder = ""
        self._gen_widget_sets: dict[str, dict[str, object]] = {}
        self._model_settings_cache = {key: self.default_generation_settings_for_model(key) for key in MODEL_DEFAULTS}
        self._active_model_key = "base"
        self._hud_timer: QTimer | None = None
        self._hud_last_net: tuple[float, int, int] | None = None
        self._hud_colorizer_installed = False
        self._use_framevision_queue = True
        self._own_queue_tab_visible = True
        self._saved_splitter_sizes = [620, 700]
        self._build_ui()
        # Theme and HUD controls were useful in standalone mode, but inside FrameVision
        # the host app owns theme and system-monitor UI. Do not apply a local theme
        # or start a local HUD timer from this embedded tab.
        self._verify_paths()
        self.load_settings()
        self.apply_queue_backend_ui(initial=True)
        self._connect_settings_autosave()
        self.update_model_status()
        self.load_queue()
        if not self.using_framevision_queue():
            self.start_next_queue_job()

    def _build_ui(self) -> None:
        root = QWidget()
        main = QVBoxLayout(root)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(APP_TITLE)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        header.addWidget(title)
        self.model_combo = QComboBox()
        for key, info in MODEL_DEFAULTS.items():
            self.model_combo.addItem(info["label"], key)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        header.addWidget(QLabel("Model"))
        header.addWidget(self.model_combo)
        self.model_status = QLabel("")
        header.addWidget(self.model_status, 1)
        self.open_output_btn = QPushButton("View results")
        self.open_output_btn.setToolTip("Open the HiDream output folder in FrameVision Media Explorer.")
        self.open_output_btn.clicked.connect(self.view_results)
        header.addWidget(self.open_output_btn)
        main.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)
        self.main_splitter = splitter
        main.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 8, 0)
        self.tabs = QTabWidget()
        left_layout.addWidget(self.tabs, 1)
        self.tabs.addTab(self._make_generate_tab(), "Create image")
        self.tabs.addTab(self._make_edit_tab(), "Edit / reference images")
        self.tabs.addTab(self._make_multi_reference_tab(), "Multi-reference")
        self.tabs.addTab(self._make_settings_tab(), "Settings")
        self.queue_tab = self._make_queue_tab()
        self._queue_tab_index = self.tabs.addTab(self.queue_tab, "Queue")

        right = QWidget()
        self.right_panel = right
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        self.preview = ImagePreview("Result preview")
        right_layout.addWidget(self.preview, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        right_layout.addWidget(self.progress)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(1600)
        self.log_box.setMinimumHeight(190)
        right_layout.addWidget(self.log_box)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([620, 700])
        self._saved_splitter_sizes = [620, 700]
        main.addWidget(self._make_footer())
        self.setCentralWidget(root)
        self._active_model_key = self.current_model_key()

    def _make_footer(self) -> QWidget:
        footer = QFrame()
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(0, 4, 0, 0)
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        layout.addStretch()
        self.view_results_btn = QPushButton("View results")
        self.view_results_btn.setToolTip("Open the HiDream output folder in FrameVision Media Explorer.")
        self.view_results_btn.clicked.connect(self.view_results)
        layout.addWidget(self.view_results_btn)
        self.stop_btn = QPushButton("Stop current")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        layout.addWidget(self.stop_btn)
        return footer

    def _wrap_scroll_area(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(widget)
        return area

    def _make_generation_settings_box(self, ui_key: str, title: str = "Generation settings") -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)

        preset_combo = QComboBox()
        if ui_key in ("edit", "multi"):
            for label, width, height in REFERENCE_SAFE_RESOLUTION_PRESETS:
                preset_combo.addItem(label, f"{width}x{height}")
        else:
            for label, width, height in RESOLUTION_PRESETS:
                preset_combo.addItem(label, f"{width}x{height}")
        preset_combo.addItem("Custom size", "custom")

        width_spin = QSpinBox()
        width_spin.setRange(256, 4096)
        width_spin.setSingleStep(64)
        default_w, default_h = REFERENCE_SAFE_DEFAULTS.get(ui_key, (1280, 704))
        width_spin.setValue(default_w)

        height_spin = QSpinBox()
        height_spin.setRange(256, 4096)
        height_spin.setSingleStep(64)
        height_spin.setValue(default_h)

        size_row = QHBoxLayout()
        size_row.addWidget(width_spin)
        size_row.addWidget(QLabel("×"))
        size_row.addWidget(height_spin)
        form.addRow("Resolution preset", preset_combo)
        form.addRow("Width / height", size_row)

        steps_spin = QSpinBox()
        steps_spin.setRange(1, 120)
        form.addRow("Steps", steps_spin)

        cfg_spin = QDoubleSpinBox()
        cfg_spin.setRange(0.0, 20.0)
        cfg_spin.setDecimals(2)
        cfg_spin.setSingleStep(0.25)
        form.addRow("CFG / guidance", cfg_spin)

        shift_spin = QDoubleSpinBox()
        shift_spin.setRange(0.1, 10.0)
        shift_spin.setDecimals(2)
        shift_spin.setSingleStep(0.25)
        form.addRow("Shift", shift_spin)

        seed_spin = QSpinBox()
        seed_spin.setRange(-1, 2147483647)
        seed_spin.setValue(-1)
        form.addRow("Seed (-1 random)", seed_spin)

        scheduler_combo = QComboBox()
        scheduler_combo.addItem("Flash / Euler", "flash")
        scheduler_combo.addItem("Default / UniPC", "default")
        form.addRow("Scheduler", scheduler_combo)

        timesteps_combo = QComboBox()
        timesteps_combo.addItem("None", "none")
        timesteps_combo.addItem("Dev fixed 28-step list", "dev")
        form.addRow("Timesteps", timesteps_combo)

        widgets = {
            "preset_combo": preset_combo,
            "width_spin": width_spin,
            "height_spin": height_spin,
            "steps_spin": steps_spin,
            "cfg_spin": cfg_spin,
            "shift_spin": shift_spin,
            "seed_spin": seed_spin,
            "scheduler_combo": scheduler_combo,
            "timesteps_combo": timesteps_combo,
        }
        self._gen_widget_sets[ui_key] = widgets
        if ui_key == "main":
            self.resolution_preset_combo = preset_combo
            self.width_spin = width_spin
            self.height_spin = height_spin
            self.steps_spin = steps_spin
            self.cfg_spin = cfg_spin
            self.shift_spin = shift_spin
            self.seed_spin = seed_spin
            self.scheduler_combo = scheduler_combo
            self.timesteps_combo = timesteps_combo

        self._connect_generation_widget_set(ui_key)
        return box

    def _connect_generation_widget_set(self, ui_key: str) -> None:
        widgets = self._gen_widget_sets[ui_key]
        widgets["preset_combo"].currentIndexChanged.connect(lambda _=None, key=ui_key: self.on_resolution_preset_changed(key))
        for name in ["width_spin", "height_spin", "steps_spin", "cfg_spin", "shift_spin", "seed_spin"]:
            widgets[name].valueChanged.connect(lambda _=None, key=ui_key: self.on_generation_controls_value_changed(key))
        widgets["scheduler_combo"].currentIndexChanged.connect(lambda _=None, key=ui_key: self.on_generation_controls_value_changed(key))
        widgets["timesteps_combo"].currentIndexChanged.connect(lambda _=None, key=ui_key: self.on_generation_controls_value_changed(key))

    def _make_advanced_settings_box(self) -> QGroupBox:
        box = QGroupBox("Advanced generation settings")
        form = QFormLayout(box)

        self.noise_start_spin = QDoubleSpinBox()
        self.noise_start_spin.setRange(0.0, 20.0)
        self.noise_start_spin.setDecimals(2)
        self.noise_start_spin.setSingleStep(0.25)
        self.noise_start_spin.setValue(7.5)
        form.addRow("Noise start", self.noise_start_spin)

        self.noise_end_spin = QDoubleSpinBox()
        self.noise_end_spin.setRange(0.0, 20.0)
        self.noise_end_spin.setDecimals(2)
        self.noise_end_spin.setSingleStep(0.25)
        self.noise_end_spin.setValue(7.5)
        form.addRow("Noise end", self.noise_end_spin)

        self.noise_clip_spin = QDoubleSpinBox()
        self.noise_clip_spin.setRange(0.0, 10.0)
        self.noise_clip_spin.setDecimals(2)
        self.noise_clip_spin.setSingleStep(0.25)
        self.noise_clip_spin.setValue(2.5)
        form.addRow("Noise clip std", self.noise_clip_spin)

        return box

    def _make_offload_box(self) -> QGroupBox:
        box = QGroupBox("Offloading / memory options")
        layout = QVBoxLayout(box)
        note = QLabel(
            "Experimental. Leave this off for normal CUDA-only loading. "
            "When enabled, the UI will pass device_map=auto only if the runner supports it."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #9ca3af;")
        layout.addWidget(note)

        self.auto_offload_check = QCheckBox("Try automatic CPU offload")
        self.auto_offload_check.setChecked(False)
        layout.addWidget(self.auto_offload_check)

        folder_row = QHBoxLayout()
        self.offload_folder_edit = QLineEdit(str(DEFAULT_OFFLOAD_FOLDER))
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_offload_folder)
        folder_row.addWidget(self.offload_folder_edit, 1)
        folder_row.addWidget(browse)
        layout.addWidget(QLabel("Offload folder"))
        layout.addLayout(folder_row)

        return box

    def _make_generate_tab(self) -> QWidget:
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)

        content = QWidget()
        layout = QVBoxLayout(content)
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Describe the image you want to create...")
        self.prompt_edit.setPlainText("A highly realistic photo of an elegant woman sitting alone in a warm restaurant, natural skin texture, correct hands, realistic lighting, shallow depth of field.")
        self.prompt_edit.setMinimumHeight(170)
        layout.addWidget(QLabel("Prompt"))
        layout.addWidget(self.prompt_edit)

        self.negative_prompt_label = QLabel("Negative prompt")
        self.negative_prompt_edit = QTextEdit()
        self.negative_prompt_edit.setPlaceholderText("Optional negatives for Full models only. Hidden for Dev variants.")
        self.negative_prompt_edit.setMaximumHeight(78)
        self.negative_prompt_edit.setMinimumHeight(56)
        layout.addWidget(self.negative_prompt_label)
        layout.addWidget(self.negative_prompt_edit)

        self.main_generation_section = CollapsibleSection("Generation settings", self._make_generation_settings_box("main"), expanded=True)
        layout.addWidget(self.main_generation_section)
        layout.addStretch()

        outer_layout.addWidget(self._wrap_scroll_area(content), 1)
        self.generate_btn = QPushButton("Create image")
        self.generate_btn.setMinimumHeight(42)
        self.generate_btn.clicked.connect(self.enqueue_create_image)
        outer_layout.addWidget(self.generate_btn)
        return outer

    def _make_edit_tab(self) -> QWidget:
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)

        content = QWidget()
        layout = QVBoxLayout(content)
        self.edit_prompt = QTextEdit()
        self.edit_prompt.setPlaceholderText("Describe the edit. Example: change the pose, keep face identity and details unchanged.")
        self.edit_prompt.setMinimumHeight(130)
        layout.addWidget(QLabel("Edit instruction"))
        layout.addWidget(self.edit_prompt)

        self.edit_negative_prompt_label = QLabel("Negative prompt")
        self.edit_negative_prompt_edit = QTextEdit()
        self.edit_negative_prompt_edit.setPlaceholderText("Optional negatives for Full models only. Hidden for Dev variants.")
        self.edit_negative_prompt_edit.setMaximumHeight(78)
        self.edit_negative_prompt_edit.setMinimumHeight(56)
        layout.addWidget(self.edit_negative_prompt_label)
        layout.addWidget(self.edit_negative_prompt_edit)

        controls = QHBoxLayout()
        add_btn = QPushButton("Add reference image(s)")
        add_btn.clicked.connect(self.add_reference_images)
        rem_btn = QPushButton("Remove selected")
        rem_btn.clicked.connect(lambda: self.ref_list.remove_selected())
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(lambda: self.ref_list.clear_images())
        controls.addWidget(add_btn)
        controls.addWidget(rem_btn)
        controls.addWidget(clear_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self.ref_preview = ImagePreview("Click a reference thumbnail for a larger preview")
        self.ref_preview.setMinimumSize(360, 220)
        self.ref_list = ReferenceList(self.ref_preview)
        layout.addWidget(QLabel("Reference images"))
        layout.addWidget(self.ref_list)
        layout.addWidget(QLabel("Reference preview"))
        layout.addWidget(self.ref_preview, 1)

        self.keep_aspect = QCheckBox("Keep original aspect ratio when exactly one reference image is used")
        self.keep_aspect.setChecked(True)
        layout.addWidget(self.keep_aspect)

        self.edit_generation_section = CollapsibleSection("Generation settings", self._make_generation_settings_box("edit"), expanded=False)
        layout.addWidget(self.edit_generation_section)
        layout.addStretch()

        outer_layout.addWidget(self._wrap_scroll_area(content), 1)
        self.edit_btn = QPushButton("Run edit / reference generation")
        self.edit_btn.setMinimumHeight(42)
        self.edit_btn.clicked.connect(self.enqueue_edit_image)
        outer_layout.addWidget(self.edit_btn)
        return outer


    def _make_multi_reference_tab(self) -> QWidget:
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(8)

        content = QWidget()
        layout = QVBoxLayout(content)

        intro = QLabel(
            "Use this tab for advanced reference workflows: identity + second subject + clothing/look + style + object/background references. "
            "The simple Edit / reference images tab stays unchanged for clean one-image edits."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #9ca3af;")
        layout.addWidget(intro)

        self.multi_prompt = QTextEdit()
        self.multi_prompt.setPlaceholderText(
            "Describe what to generate, what to keep from each reference, and how the references should be combined."
        )
        self.multi_prompt.setMinimumHeight(140)
        layout.addWidget(QLabel("Multi-reference instruction"))
        layout.addWidget(self.multi_prompt)

        self.multi_negative_prompt_label = QLabel("Negative prompt")
        self.multi_negative_prompt_edit = QTextEdit()
        self.multi_negative_prompt_edit.setPlaceholderText("Optional negatives for Full models only. Hidden for Dev variants.")
        self.multi_negative_prompt_edit.setMaximumHeight(78)
        self.multi_negative_prompt_edit.setMinimumHeight(56)
        layout.addWidget(self.multi_negative_prompt_label)
        layout.addWidget(self.multi_negative_prompt_edit)

        controls = QHBoxLayout()
        add_btn = QPushButton("Add reference image(s)")
        add_btn.clicked.connect(self.add_multi_reference_images)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(lambda: self.multi_ref_list.remove_selected())
        clear_btn = QPushButton("Clear all")
        clear_btn.clicked.connect(lambda: self.multi_ref_list.clear_images())
        move_up_btn = QPushButton("Move up")
        move_up_btn.clicked.connect(lambda: self.multi_ref_list.move_selected(-1))
        move_down_btn = QPushButton("Move down")
        move_down_btn.clicked.connect(lambda: self.multi_ref_list.move_selected(1))
        controls.addWidget(add_btn)
        controls.addWidget(remove_btn)
        controls.addWidget(clear_btn)
        controls.addWidget(move_up_btn)
        controls.addWidget(move_down_btn)
        controls.addStretch()
        layout.addLayout(controls)

        self.multi_ref_preview = ImagePreview("Click a reference item for a larger preview")
        self.multi_ref_preview.setMinimumSize(360, 230)
        self.multi_ref_list = MultiReferenceList(self.multi_ref_preview)
        layout.addWidget(QLabel("Ordered references / roles"))
        layout.addWidget(self.multi_ref_list)
        layout.addWidget(QLabel("Selected reference preview"))
        layout.addWidget(self.multi_ref_preview, 1)

        self.multi_generation_section = CollapsibleSection("Generation settings", self._make_generation_settings_box("multi"), expanded=False)
        layout.addWidget(self.multi_generation_section)
        layout.addStretch()

        outer_layout.addWidget(self._wrap_scroll_area(content), 1)
        self.multi_run_btn = QPushButton("Run multi-reference generation")
        self.multi_run_btn.setMinimumHeight(42)
        self.multi_run_btn.clicked.connect(self.enqueue_multi_reference_image)
        outer_layout.addWidget(self.multi_run_btn)
        return outer

    def _make_settings_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)

        output_box = QGroupBox("Default output folder")
        output_form = QFormLayout(output_box)
        self.output_dir_edit = QLineEdit(str(DEFAULT_OUTPUT))
        browse = QPushButton("Browse")
        browse.clicked.connect(self.browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self.output_dir_edit, 1)
        output_row.addWidget(browse)
        output_form.addRow("Output folder", output_row)
        layout.addWidget(output_box)

        behavior_box = QGroupBox("Behavior / environment")
        behavior_form = QFormLayout(behavior_box)

        self.switch_defaults_on_model_change_check = QCheckBox("Switch to defaults on model change")
        self.switch_defaults_on_model_change_check.setChecked(True)
        self.switch_defaults_on_model_change_check.setToolTip(
            "When enabled, switching between Base and Dev resets only model-specific generation controls "
            "to that model's defaults: steps, CFG/guidance availability and value, shift, scheduler, "
            "timesteps, noise start, noise end and noise clip. Resolution, prompts, seed choices and "
            "output folder stay as the user's current selection."
        )
        behavior_form.addRow("Model switching", self.switch_defaults_on_model_change_check)

        self.framevision_queue_check = QCheckBox("Use FrameVision queue")
        self.framevision_queue_check.setChecked(True)
        self.framevision_queue_check.setToolTip(
            "When enabled, HiDream uses the main FrameVision queue/worker instead of the built-in HiDream queue. "
            "The local preview/log side panel and the built-in Queue tab are hidden in this mode."
        )
        behavior_form.addRow("Queue backend", self.framevision_queue_check)

        env_widget = QWidget()
        env_layout = QVBoxLayout(env_widget)
        env_layout.setContentsMargins(0, 0, 0, 0)
        env_layout.setSpacing(6)

        self.environment_info_label = QLabel("Environment not checked yet. Click Check Environment to run the diagnostic.")
        self.environment_info_label.setWordWrap(True)
        self.environment_info_label.setTextFormat(Qt.PlainText)
        self.environment_info_label.setStyleSheet("color: #cbd5e1; padding: 8px; border: 1px solid #334155; border-radius: 8px; background: rgba(15, 23, 42, 0.45);")
        env_layout.addWidget(self.environment_info_label)

        env_btn_row = QHBoxLayout()
        env_btn_row.setContentsMargins(0, 0, 0, 0)
        self.environment_check_btn = QPushButton("Check Environment")
        self.environment_check_btn.setToolTip("Runs Python/Torch/CUDA/Triton/Flash/SageAttention checks for the HiDream environment.")
        self.environment_check_btn.clicked.connect(self.check_environment_info)
        env_btn_row.addWidget(self.environment_check_btn)
        env_btn_row.addStretch()
        env_layout.addLayout(env_btn_row)

        behavior_form.addRow("Environment", env_widget)
        layout.addWidget(behavior_box)

        layout.addWidget(self._make_advanced_settings_box())
        layout.addWidget(self._make_offload_box())

        model_box = QGroupBox("Model info")
        model_layout = QVBoxLayout(model_box)
        self.model_note = QLabel("")
        self.model_note.setWordWrap(True)
        self.model_note.setStyleSheet("color: #9ca3af;")
        model_layout.addWidget(self.model_note)
        layout.addWidget(model_box)

        layout.addStretch()
        return self._wrap_scroll_area(content)

    def _make_queue_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        info = QLabel("Built-in HiDream queue. When FrameVision queue mode is enabled in Settings, this tab is hidden and HiDream jobs are sent to the main FrameVision queue instead.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #9ca3af;")
        layout.addWidget(info)
        self.queue_list = QListWidget()
        self.queue_list.setMinimumHeight(320)
        self.queue_list.itemClicked.connect(self._queue_item_clicked)
        layout.addWidget(self.queue_list, 1)
        btns = QHBoxLayout()
        self.move_queue_up_btn = QPushButton("Move up")
        self.move_queue_up_btn.clicked.connect(lambda: self.move_selected_queue_job(-1))
        self.move_queue_down_btn = QPushButton("Move down")
        self.move_queue_down_btn.clicked.connect(lambda: self.move_selected_queue_job(1))
        self.remove_queue_btn = QPushButton("Remove selected")
        self.remove_queue_btn.clicked.connect(self.remove_selected_queue_job)
        self.retry_queue_btn = QPushButton("Retry selected")
        self.retry_queue_btn.clicked.connect(self.retry_selected_queue_job)
        self.clear_finished_queue_btn = QPushButton("Clear finished")
        self.clear_finished_queue_btn.clicked.connect(self.clear_finished_queue_jobs)
        btns.addWidget(self.move_queue_up_btn)
        btns.addWidget(self.move_queue_down_btn)
        btns.addWidget(self.remove_queue_btn)
        btns.addWidget(self.retry_queue_btn)
        btns.addWidget(self.clear_finished_queue_btn)
        btns.addStretch()
        layout.addLayout(btns)
        self.queue_status_label = QLabel("Queue: empty")
        self.queue_status_label.setStyleSheet("color: #bfdbfe;")
        layout.addWidget(self.queue_status_label)
        return w

    def using_framevision_queue(self) -> bool:
        if hasattr(self, "framevision_queue_check"):
            return bool(self.framevision_queue_check.isChecked())
        return bool(getattr(self, "_use_framevision_queue", True))

    def on_framevision_queue_toggled(self) -> None:
        self._use_framevision_queue = self.using_framevision_queue()
        self.apply_queue_backend_ui()
        self.save_settings()

    def apply_queue_backend_ui(self, initial: bool = False) -> None:
        use_fv = self.using_framevision_queue()
        self._use_framevision_queue = use_fv

        if hasattr(self, "tabs") and hasattr(self, "_queue_tab_index"):
            try:
                self.tabs.tabBar().setTabVisible(self._queue_tab_index, not use_fv)
                self._own_queue_tab_visible = not use_fv
            except Exception:
                try:
                    self.tabs.setTabEnabled(self._queue_tab_index, not use_fv)
                    self._own_queue_tab_visible = not use_fv
                except Exception:
                    pass
            if use_fv and self.tabs.currentIndex() == self._queue_tab_index:
                try:
                    self.tabs.setCurrentIndex(0)
                except Exception:
                    pass

        if hasattr(self, "right_panel") and hasattr(self, "main_splitter"):
            if use_fv:
                try:
                    sizes = self.main_splitter.sizes()
                    if sizes and len(sizes) >= 2 and sizes[1] > 0:
                        self._saved_splitter_sizes = sizes
                except Exception:
                    pass
                self.right_panel.setVisible(False)
                try:
                    self.main_splitter.setSizes([1, 0])
                except Exception:
                    pass
            else:
                self.right_panel.setVisible(True)
                try:
                    sizes = self._saved_splitter_sizes if getattr(self, "_saved_splitter_sizes", None) else [620, 700]
                    self.main_splitter.setSizes(sizes)
                except Exception:
                    pass

        if hasattr(self, "stop_btn"):
            try:
                self.stop_btn.setVisible(not use_fv)
                if use_fv:
                    self.stop_btn.setEnabled(False)
            except Exception:
                pass

        if not initial:
            try:
                if use_fv:
                    self.status_label.setText("Using FrameVision queue")
                else:
                    self.status_label.setText("Using built-in HiDream queue")
                    self.start_next_queue_job()
            except Exception:
                pass

    def switch_to_framevision_queue_tab(self) -> None:
        try:
            win = self.window()
            for tabs in win.findChildren(QTabWidget):
                if tabs is self.tabs:
                    continue
                for i in range(tabs.count()):
                    if tabs.tabText(i).strip().lower() == "queue":
                        tabs.setCurrentIndex(i)
                        return
        except Exception:
            pass

    def _apply_theme(self, theme_name: str | None = None) -> None:
        name = theme_name
        try:
            if name is None and hasattr(self, "theme_combo"):
                name = self.theme_combo.currentData() or self.theme_combo.currentText()
        except Exception:
            name = None
        name = str(name or "Evening")

        if callable(_THEME_APPLY):
            try:
                app = QApplication.instance()
                if app:
                    _THEME_APPLY(app, name)
                    return
            except Exception as exc:
                try:
                    self.log(f"Theme helper failed for {name}: {exc}; using built-in fallback.")
                except Exception:
                    pass

        self.setStyleSheet("""
            QMainWindow, QWidget { background: #0b1220; color: #e5e7eb; font-size: 13px; }
            QGroupBox { border: 1px solid #334155; border-radius: 10px; margin-top: 12px; padding: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #bfdbfe; }
            QTextEdit, QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListWidget {
                background: #111827; color: #e5e7eb; border: 1px solid #374151; border-radius: 7px; padding: 5px;
            }
            QPushButton { background: #1f6feb; color: white; border: none; border-radius: 8px; padding: 8px 12px; font-weight: 600; }
            QPushButton:hover { background: #388bfd; }
            QPushButton:disabled { background: #374151; color: #9ca3af; }
            QTabWidget::pane { border: 1px solid #334155; border-radius: 10px; }
            QTabBar::tab { background: #111827; color: #d1d5db; padding: 8px 12px; border-top-left-radius: 8px; border-top-right-radius: 8px; }
            QTabBar::tab:selected { background: #1e293b; color: #ffffff; }
            QProgressBar { background: #111827; border: 1px solid #374151; border-radius: 6px; text-align: center; }
            QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
            QToolButton { background: #111827; color: #e5e7eb; border: 1px solid #374151; border-radius: 8px; padding: 8px 10px; font-weight: 600; text-align: left; }
            QScrollArea { border: none; background: transparent; }
        """)

    def _install_hud_colorizer_if_enabled(self) -> None:
        try:
            if not getattr(self, "hud_colorizer_check", None) or not self.hud_colorizer_check.isChecked():
                return
            if self._hud_colorizer_installed:
                return
            if callable(_HUD_INSTALL):
                _HUD_INSTALL()
                self._hud_colorizer_installed = True
            else:
                self.log("HUD colorizer helper not found; put hud_colorizer.py next to hidream_ui.py.")
        except Exception as exc:
            self.log(f"HUD colorizer failed: {exc}")

    def _set_hud_running(self, enabled: bool) -> None:
        if not hasattr(self, "hud_label"):
            return
        if enabled:
            self.hud_label.setVisible(True)
            if self._hud_timer is None:
                self._hud_timer = QTimer(self)
                self._hud_timer.setInterval(3000)
                self._hud_timer.timeout.connect(self.refresh_hud_monitor)
            if not self._hud_timer.isActive():
                self._hud_timer.start()
            self.refresh_hud_monitor()
            self._install_hud_colorizer_if_enabled()
        else:
            if self._hud_timer is not None:
                self._hud_timer.stop()
            self.hud_label.setTextFormat(Qt.PlainText)
            self.hud_label.setText("HUD: disabled")

    def _format_net_speed(self, bytes_per_sec: float) -> str:
        try:
            value = float(bytes_per_sec)
        except Exception:
            value = 0.0
        if value >= 1024 * 1024:
            return f"{value / (1024 * 1024):.1f} MB/s"
        return f"{value / 1024:.0f} KB/s"

    def _read_system_metrics(self) -> dict:
        metrics = {
            "cpu_pct": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
            "ram_pct": None,
            "gpu_pct": None,
            "vram_used_gb": None,
            "vram_total_gb": None,
            "vram_pct": None,
            "gpu_temp": None,
            "dl_bps": 0.0,
            "ul_bps": 0.0,
        }
        try:
            import psutil  # type: ignore
            metrics["cpu_pct"] = int(round(psutil.cpu_percent(interval=None)))
            mem = psutil.virtual_memory()
            metrics["ram_used_gb"] = float(mem.used) / (1024 ** 3)
            metrics["ram_total_gb"] = float(mem.total) / (1024 ** 3)
            metrics["ram_pct"] = int(round(float(mem.percent)))
            try:
                net = psutil.net_io_counters()
                now = time.time()
                last = self._hud_last_net
                if last:
                    last_time, last_recv, last_sent = last
                    elapsed = max(0.001, now - float(last_time))
                    metrics["dl_bps"] = max(0.0, (int(net.bytes_recv) - int(last_recv)) / elapsed)
                    metrics["ul_bps"] = max(0.0, (int(net.bytes_sent) - int(last_sent)) / elapsed)
                self._hud_last_net = (now, int(net.bytes_recv), int(net.bytes_sent))
            except Exception:
                pass
        except Exception:
            pass

        try:
            query = "memory.total,memory.used,temperature.gpu,utilization.gpu"
            out = subprocess.check_output(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=2,
            ).strip()
            if out:
                line = out.splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    total_mb = float(parts[0])
                    used_mb = float(parts[1])
                    temp = int(float(parts[2]))
                    gpu_pct = int(float(parts[3]))
                    metrics["vram_total_gb"] = total_mb / 1024.0
                    metrics["vram_used_gb"] = used_mb / 1024.0
                    metrics["vram_pct"] = int(round((used_mb / total_mb) * 100.0)) if total_mb > 0 else None
                    metrics["gpu_temp"] = temp
                    metrics["gpu_pct"] = gpu_pct
        except Exception:
            pass
        return metrics

    def refresh_hud_monitor(self) -> None:
        if not hasattr(self, "hud_label"):
            return
        if hasattr(self, "hud_monitor_check") and not self.hud_monitor_check.isChecked():
            self.hud_label.setTextFormat(Qt.PlainText)
            self.hud_label.setText("HUD: disabled")
            return
        m = self._read_system_metrics()
        if self.hud_colorizer_check.isChecked():
            self.hud_label.setTextFormat(Qt.RichText)
        else:
            self.hud_label.setTextFormat(Qt.PlainText)

        gpu = "GPU : n/a"
        if m.get("gpu_pct") is not None:
            gpu = f"GPU : {m['gpu_pct']}%"
            if m.get("gpu_temp") is not None:
                gpu += f" {m['gpu_temp']}C"
        vram = "VRAM : n/a"
        if m.get("vram_used_gb") is not None and m.get("vram_total_gb") is not None:
            vram = f"VRAM : {m['vram_used_gb']:.1f}/{m['vram_total_gb']:.1f} GB"
            if m.get("vram_pct") is not None:
                vram += f" {m['vram_pct']}%"
        ddr = "DDR n/a"
        if m.get("ram_used_gb") is not None and m.get("ram_total_gb") is not None:
            ddr = f"DDR {m['ram_used_gb']:.1f}/{m['ram_total_gb']:.1f} GB"
            if m.get("ram_pct") is not None:
                ddr += f" {m['ram_pct']}%"
        cpu = "CPU n/a"
        if m.get("cpu_pct") is not None:
            cpu = f"CPU {m['cpu_pct']}%"
        net = f"DL {self._format_net_speed(float(m.get('dl_bps') or 0.0))}  UL {self._format_net_speed(float(m.get('ul_bps') or 0.0))}"
        self.hud_label.setText(f"{gpu}  |  {vram}  |  {ddr}  |  {cpu}  |  {net}")

    def on_theme_changed(self) -> None:
        self._apply_theme()
        self.save_settings()

    def on_hud_colorizer_changed(self) -> None:
        self._install_hud_colorizer_if_enabled()
        self.refresh_hud_monitor()
        self.save_settings()

    def on_hud_monitor_changed(self) -> None:
        self._set_hud_running(bool(self.hud_monitor_check.isChecked()))
        self.save_settings()

    def current_model_key(self) -> str:
        return self.model_combo.currentData() or "base"

    def model_variant(self, key: str | None = None) -> str:
        key = key or self.current_model_key()
        info = MODEL_DEFAULTS.get(key, {})
        if isinstance(info, dict):
            return str(info.get("variant", "full"))
        return "full"

    def is_full_variant(self, key: str | None = None) -> bool:
        return self.model_variant(key) == "full"

    def is_dev_variant(self, key: str | None = None) -> bool:
        return self.model_variant(key) == "dev"

    def model_dir(self, key: str | None = None) -> Path | None:
        key = key or self.current_model_key()
        if not HIDREAM_ROOT:
            return None
        return HIDREAM_ROOT / MODEL_DEFAULTS[key]["folder"]

    def model_installed(self, key: str | None = None) -> bool:
        d = self.model_dir(key)
        return bool(d and d.exists() and (d / "config.json").exists())

    def set_combo_by_data(self, combo: QComboBox, value) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def default_generation_settings_for_model(self, key: str) -> dict:
        info = MODEL_DEFAULTS[key]
        return {
            "resolution_preset": "1280x704",
            "width": 1280,
            "height": 704,
            "steps": info["steps"],
            "guidance_scale": info["guidance_scale"],
            "shift": info["shift"],
            "seed": -1,
            "scheduler_name": info["scheduler"],
            "timesteps": info["timesteps"],
            "noise_scale_start": 7.5,
            "noise_scale_end": 7.5,
            "noise_clip_std": 2.5,
            "negative_prompt": "",
        }

    def model_switch_default_values_for_model(self, key: str) -> dict:
        info = MODEL_DEFAULTS[key]
        return {
            "steps": info["steps"],
            "guidance_scale": info["guidance_scale"],
            "shift": info["shift"],
            "scheduler_name": info["scheduler"],
            "timesteps": info["timesteps"],
            "noise_scale_start": 7.5,
            "noise_scale_end": 7.5,
            "noise_clip_std": 2.5,
        }

    def apply_model_switch_defaults_to_settings(self, key: str, settings: dict | None = None) -> dict:
        merged = dict(settings or self.default_generation_settings_for_model(key))
        merged.update(self.model_switch_default_values_for_model(key))
        if self.is_dev_variant(key):
            merged["guidance_scale"] = 0.0
        return merged

    def current_user_carried_settings(self) -> dict:
        if "main" not in self._gen_widget_sets:
            return {}
        widgets = self._gen_widget_sets["main"]
        width = widgets["width_spin"].value()
        height = widgets["height_spin"].value()
        return {
            "resolution_preset": self.match_resolution_preset(width, height),
            "width": width,
            "height": height,
            "seed": widgets["seed_spin"].value(),
        }

    def switch_defaults_on_model_change_enabled(self) -> bool:
        if hasattr(self, "switch_defaults_on_model_change_check"):
            return self.switch_defaults_on_model_change_check.isChecked()
        return bool(getattr(self, "_switch_defaults_on_model_change", True))

    def on_switch_defaults_on_model_change_toggled(self) -> None:
        self._switch_defaults_on_model_change = self.switch_defaults_on_model_change_enabled()
        self.save_settings()

    def generation_settings_from_widgets(self, ui_key: str) -> dict:
        widgets = self._gen_widget_sets[ui_key]
        width = widgets["width_spin"].value()
        height = widgets["height_spin"].value()
        return {
            "resolution_preset": self.match_resolution_preset(width, height),
            "width": width,
            "height": height,
            "steps": widgets["steps_spin"].value(),
            "guidance_scale": 0.0 if self.is_dev_variant() else widgets["cfg_spin"].value(),
            "shift": widgets["shift_spin"].value(),
            "seed": widgets["seed_spin"].value(),
            "scheduler_name": widgets["scheduler_combo"].currentData(),
            "timesteps": widgets["timesteps_combo"].currentData(),
        }

    def current_advanced_settings(self) -> dict:
        return {
            "noise_scale_start": self.noise_start_spin.value(),
            "noise_scale_end": self.noise_end_spin.value(),
            "noise_clip_std": self.noise_clip_spin.value(),
        }

    def current_offload_settings(self) -> dict:
        return {
            "try_auto_cpu_offload": self.auto_offload_check.isChecked(),
            "offload_folder": self.offload_folder_edit.text().strip() or str(DEFAULT_OFFLOAD_FOLDER),
        }

    def parse_resolution_preset(self, value: str | None) -> tuple[int, int] | None:
        if not value or value == "custom":
            return None
        m = re.match(r"^(\d+)x(\d+)$", str(value))
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def match_resolution_preset(self, width: int, height: int) -> str:
        for _label, pw, ph in RESOLUTION_PRESETS:
            if pw == width and ph == height:
                return f"{pw}x{ph}"
        return "custom"

    def apply_generation_settings_to_widgets(self, ui_key: str, settings: dict) -> None:
        widgets = self._gen_widget_sets[ui_key]
        width = int(settings.get("width", 1280))
        height = int(settings.get("height", 704))
        preset = settings.get("resolution_preset") or self.match_resolution_preset(width, height)
        if (width, height) in LEGACY_RESOLUTION_PRESET_MAP and str(preset) in {"custom", f"{width}x{height}"}:
            width, height = LEGACY_RESOLUTION_PRESET_MAP[(width, height)]
            preset = self.match_resolution_preset(width, height)
        self._syncing_generation_controls = True
        try:
            widgets["width_spin"].setValue(width)
            widgets["height_spin"].setValue(height)
            widgets["steps_spin"].setValue(int(settings.get("steps", MODEL_DEFAULTS[self.current_model_key()]["steps"])))
            cfg_value = 0.0 if self.is_dev_variant() else float(settings.get("guidance_scale", MODEL_DEFAULTS[self.current_model_key()]["guidance_scale"]))
            widgets["cfg_spin"].setValue(cfg_value)
            widgets["cfg_spin"].setEnabled(self.is_full_variant())
            widgets["shift_spin"].setValue(float(settings.get("shift", MODEL_DEFAULTS[self.current_model_key()]["shift"])))
            widgets["seed_spin"].setValue(int(settings.get("seed", -1)))
            self.set_combo_by_data(widgets["scheduler_combo"], settings.get("scheduler_name", MODEL_DEFAULTS[self.current_model_key()]["scheduler"]))
            self.set_combo_by_data(widgets["timesteps_combo"], settings.get("timesteps", MODEL_DEFAULTS[self.current_model_key()]["timesteps"]))
            self.set_combo_by_data(widgets["preset_combo"], preset if preset else "custom")
            if widgets["preset_combo"].currentData() is None:
                self.set_combo_by_data(widgets["preset_combo"], "custom")
        finally:
            self._syncing_generation_controls = False

    def apply_advanced_settings_to_widgets(self, settings: dict) -> None:
        self.noise_start_spin.setValue(float(settings.get("noise_scale_start", 7.5)))
        self.noise_end_spin.setValue(float(settings.get("noise_scale_end", 7.5)))
        self.noise_clip_spin.setValue(float(settings.get("noise_clip_std", 2.5)))

    def store_current_model_settings(self, key: str | None = None) -> None:
        key = key or self.current_model_key()
        settings = self.default_generation_settings_for_model(key)
        if "main" in self._gen_widget_sets:
            settings.update(self.generation_settings_from_widgets("main"))
        settings.update(self.current_advanced_settings())
        self._model_settings_cache[key] = settings

    def apply_model_defaults(self, key: str) -> None:
        self._model_settings_cache[key] = self.default_generation_settings_for_model(key)
        self.apply_model_settings_to_ui(key)

    def reference_safe_settings(self, ui_key: str, base: dict) -> dict:
        settings = dict(base)
        if ui_key in REFERENCE_SAFE_DEFAULTS:
            w, h = REFERENCE_SAFE_DEFAULTS[ui_key]
            cur_w = int(settings.get("width", 0) or 0)
            cur_h = int(settings.get("height", 0) or 0)
            # If the current/shared defaults are small landscape buckets, upgrade reference modes
            # to a moderate safe bucket. Users can still manually choose higher/lower custom sizes.
            if cur_w > cur_h and (cur_w < 1600 or cur_h < 896):
                settings["width"] = w
                settings["height"] = h
                settings["resolution_preset"] = f"{w}x{h}"
        return settings

    def apply_model_settings_to_ui(self, key: str) -> None:
        settings = dict(self.default_generation_settings_for_model(key))
        settings.update(self._model_settings_cache.get(key, {}))
        self.apply_generation_settings_to_widgets("main", settings)
        if "edit" in self._gen_widget_sets:
            self.apply_generation_settings_to_widgets("edit", self.reference_safe_settings("edit", settings))
        if "multi" in self._gen_widget_sets:
            self.apply_generation_settings_to_widgets("multi", self.reference_safe_settings("multi", settings))
        self.apply_advanced_settings_to_widgets(settings)
        self.refresh_model_specific_ui(key)

    def on_resolution_preset_changed(self, source: str) -> None:
        if self._syncing_generation_controls:
            return
        widgets = self._gen_widget_sets[source]
        dims = self.parse_resolution_preset(widgets["preset_combo"].currentData())
        if dims:
            self._syncing_generation_controls = True
            try:
                widgets["width_spin"].setValue(dims[0])
                widgets["height_spin"].setValue(dims[1])
            finally:
                self._syncing_generation_controls = False
        self.on_generation_controls_value_changed(source)

    def on_generation_controls_value_changed(self, source: str) -> None:
        if self._syncing_generation_controls:
            return
        settings = self.generation_settings_from_widgets(source)
        if self.is_dev_variant():
            settings["guidance_scale"] = 0.0
        self._syncing_generation_controls = True
        try:
            self.set_combo_by_data(self._gen_widget_sets[source]["preset_combo"], settings.get("resolution_preset", "custom"))
            merged = dict(self.default_generation_settings_for_model(self.current_model_key()))
            merged.update(settings)
            for other in self._gen_widget_sets:
                if other != source:
                    target_settings = self.reference_safe_settings(other, merged) if source == "main" else merged
                    self.apply_generation_settings_to_widgets(other, target_settings)
        finally:
            self._syncing_generation_controls = False
        current = dict(self._model_settings_cache.get(self.current_model_key(), self.default_generation_settings_for_model(self.current_model_key())))
        current.update(settings)
        self._model_settings_cache[self.current_model_key()] = current
        if not self._loading_settings:
            self.save_settings()

    def refresh_model_specific_ui(self, key: str | None = None) -> None:
        key = key or self.current_model_key()
        is_full = self.is_full_variant(key)
        if hasattr(self, "negative_prompt_label"):
            self.negative_prompt_label.setVisible(is_full)
            self.negative_prompt_edit.setVisible(is_full)
        if hasattr(self, "edit_negative_prompt_label"):
            self.edit_negative_prompt_label.setVisible(is_full)
            self.edit_negative_prompt_edit.setVisible(is_full)
        if hasattr(self, "multi_negative_prompt_label"):
            self.multi_negative_prompt_label.setVisible(is_full)
            self.multi_negative_prompt_edit.setVisible(is_full)
        for widgets in self._gen_widget_sets.values():
            cfg_spin = widgets.get("cfg_spin")
            if cfg_spin is not None:
                cfg_spin.setEnabled(is_full)
                if self.is_dev_variant(key):
                    cfg_spin.setValue(0.0)

    def on_model_changed(self) -> None:
        key = self.current_model_key()
        if self._loading_settings:
            self._active_model_key = key
            self.refresh_model_specific_ui(key)
            self.update_model_status()
            return

        previous = getattr(self, "_active_model_key", None)
        carried_user_settings = self.current_user_carried_settings()
        if previous in MODEL_DEFAULTS:
            self.store_current_model_settings(previous)

        self._active_model_key = key
        if self.switch_defaults_on_model_change_enabled():
            base_settings = dict(self.default_generation_settings_for_model(key))
            base_settings.update(self._model_settings_cache.get(key, {}))
            base_settings.update(carried_user_settings)
            self._model_settings_cache[key] = self.apply_model_switch_defaults_to_settings(key, base_settings)

        self.apply_model_settings_to_ui(key)
        self.update_model_status()
        self.save_settings()

    def sync_negative_prompt(self, source: str) -> None:
        if self._syncing_negative_prompt:
            return
        self._syncing_negative_prompt = True
        try:
            editors = []
            if hasattr(self, "negative_prompt_edit"):
                editors.append(("main", self.negative_prompt_edit))
            if hasattr(self, "edit_negative_prompt_edit"):
                editors.append(("edit", self.edit_negative_prompt_edit))
            if hasattr(self, "multi_negative_prompt_edit"):
                editors.append(("multi", self.multi_negative_prompt_edit))
            source_editor = next((editor for name, editor in editors if name == source), editors[0][1] if editors else None)
            value = source_editor.toPlainText() if source_editor is not None else ""
            for name, editor in editors:
                if name != source and editor.toPlainText() != value:
                    editor.setPlainText(value)
        finally:
            self._syncing_negative_prompt = False
        if not self._loading_settings:
            self.save_settings()

    def check_environment_info(self) -> None:
        """Run the optional HiDream environment diagnostic on demand.

        The check can spawn the HiDream Python environment and import Torch, so it
        must not run during UI construction. Keeping it behind this button avoids
        slowing normal FrameVision startup and normal HiDream tab opening.
        """
        try:
            if hasattr(self, "environment_check_btn"):
                self.environment_check_btn.setEnabled(False)
            if hasattr(self, "environment_info_label"):
                self.environment_info_label.setText("Checking environment...")
        except Exception:
            pass
        QTimer.singleShot(0, self._run_environment_info_check)

    def _run_environment_info_check(self) -> None:
        try:
            text = self.environment_info_text()
        except Exception as exc:
            text = f"Environment check failed: {exc.__class__.__name__}: {exc}"
        try:
            if hasattr(self, "environment_info_label"):
                self.environment_info_label.setText(text)
        finally:
            try:
                if hasattr(self, "environment_check_btn"):
                    self.environment_check_btn.setEnabled(True)
            except Exception:
                pass

    def environment_info_text(self) -> str:
        def short_python(path: Path | None) -> str:
            if not path or not path.exists():
                return "not found"
            try:
                return subprocess.check_output([str(path), "-c", "import sys; print(sys.version.split()[0])"], text=True, stderr=subprocess.STDOUT, timeout=6).strip() or "unknown"
            except Exception as exc:
                return f"error: {exc.__class__.__name__}"

        def env_probe(path: Path | None) -> dict:
            result = {"torch": "n/a", "cuda": "n/a", "cuda_available": "n/a", "triton": "no", "flash": "no", "sage": "no"}
            if not path or not path.exists():
                return result
            code = '\nimport importlib.util\nmods = {\n    "triton": "triton",\n    "flash": "flash_attn",\n    "sage": "sageattention",\n}\ntry:\n    import torch\n    print("torch=" + str(getattr(torch, "__version__", "unknown")))\n    print("cuda=" + str(getattr(torch.version, "cuda", None)))\n    print("cuda_available=" + str(torch.cuda.is_available()))\nexcept Exception as e:\n    print("torch=error:" + e.__class__.__name__)\n    print("cuda=n/a")\n    print("cuda_available=n/a")\nfor key, mod in mods.items():\n    print(key + "=" + ("yes" if importlib.util.find_spec(mod) else "no"))\n'
            try:
                out = subprocess.check_output([str(path), "-c", code], text=True, stderr=subprocess.STDOUT, timeout=10)
                for line in out.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        if k in result:
                            result[k] = v.strip()
            except Exception as exc:
                result["torch"] = f"error: {exc.__class__.__name__}"
            return result

        probe = env_probe(ENV_PY)
        ui_py = sys.version.split()[0]
        env_py = short_python(ENV_PY)
        cuda = probe.get("cuda") or "n/a"
        cuda_available = probe.get("cuda_available") or "n/a"
        return (
            f"UI Python: {ui_py} | Env Python: {env_py}\n"
            f"Torch: {probe.get('torch', 'n/a')} | CUDA build: {cuda} | CUDA available: {cuda_available}\n"
            f"Triton: {probe.get('triton', 'no')} | Flash Attention: {probe.get('flash', 'no')} | SageAttention: {probe.get('sage', 'no')}"
        )

    def update_model_status(self) -> None:
        key = self.current_model_key()
        info = MODEL_DEFAULTS[key]
        installed = self.model_installed(key)
        if installed:
            self.model_status.setText("Installed")
            self.model_status.setStyleSheet("color: #86efac;")
        else:
            self.model_status.setText("Missing — Open 'Opetional downloads' and download this model")
            self.model_status.setStyleSheet("color: #fca5a5;")
        self.model_note.setText(info["note"] + (" Installed." if installed else " Missing: run install.bat again and choose this model."))

    def _verify_paths(self) -> None:
        self.log("Path discovery:")
        self.log(f"  UI file: {Path(__file__).resolve()}")
        self.log(f"  env python: {ENV_PY}")
        self.log(f"  runner: {RUNNER}")
        self.log(f"  hidream root: {HIDREAM_ROOT}")
        self.log(f"  repo dir: {PATHS['repo_dir']}")
        self.log(f"  output dir: {DEFAULT_OUTPUT}")
        for key, info in MODEL_DEFAULTS.items():
            self.log(f"  {info['label']}: {self.model_dir(key)} | installed={self.model_installed(key)}")
        missing = []
        if not ENV_PY or not ENV_PY.exists():
            missing.append("environment python")
        if not RUNNER or not RUNNER.exists():
            missing.append("shared runner")
        if not PATHS["repo_dir"] or not PATHS["repo_dir"].exists():
            missing.append("HiDream repo folder")
        if missing:
            self.status_label.setText("Install paths missing")
            self.log("Missing: " + ", ".join(missing))
        else:
            self.status_label.setText("Ready")

    def settings_data(self) -> dict:
        self.store_current_model_settings(self.current_model_key())
        current_settings = dict(self._model_settings_cache.get(self.current_model_key(), self.default_generation_settings_for_model(self.current_model_key())))
        return {
            "model_key": self.current_model_key(),
            "prompt": self.prompt_edit.toPlainText(),
            "edit_prompt": self.edit_prompt.toPlainText(),
            "multi_prompt": self.multi_prompt.toPlainText() if hasattr(self, "multi_prompt") else "",
            "negative_prompt": self.negative_prompt_edit.toPlainText(),
            "multi_negative_prompt": self.multi_negative_prompt_edit.toPlainText() if hasattr(self, "multi_negative_prompt_edit") else self.negative_prompt_edit.toPlainText(),
            "model_settings": self._model_settings_cache,
            "width": current_settings.get("width", 1280),
            "height": current_settings.get("height", 704),
            "resolution_preset": current_settings.get("resolution_preset", "1280x704"),
            "steps": current_settings.get("steps", MODEL_DEFAULTS[self.current_model_key()]["steps"]),
            "guidance_scale": current_settings.get("guidance_scale", MODEL_DEFAULTS[self.current_model_key()]["guidance_scale"]),
            "shift": current_settings.get("shift", MODEL_DEFAULTS[self.current_model_key()]["shift"]),
            "seed": current_settings.get("seed", -1),
            "scheduler_name": current_settings.get("scheduler_name", MODEL_DEFAULTS[self.current_model_key()]["scheduler"]),
            "timesteps": current_settings.get("timesteps", MODEL_DEFAULTS[self.current_model_key()]["timesteps"]),
            "noise_scale_start": current_settings.get("noise_scale_start", 7.5),
            "noise_scale_end": current_settings.get("noise_scale_end", 7.5),
            "noise_clip_std": current_settings.get("noise_clip_std", 2.5),
            "output_dir": self.output_dir_edit.text(),
            "switch_defaults_on_model_change": self.switch_defaults_on_model_change_enabled(),
            "use_framevision_queue": self.using_framevision_queue(),
            "offload_settings": self.current_offload_settings(),
            "last_reference_folder": self._last_reference_folder,
            "last_dialog_folder": self._last_dialog_folder,
            "keep_original_aspect": self.keep_aspect.isChecked(),
            "active_tab": self.tabs.currentIndex(),
            "main_generation_expanded": self.main_generation_section.is_expanded() if hasattr(self, "main_generation_section") else True,
            "edit_generation_expanded": self.edit_generation_section.is_expanded(),
            "multi_generation_expanded": self.multi_generation_section.is_expanded() if hasattr(self, "multi_generation_section") else False,
        }

    def load_settings(self) -> None:
        self._loading_settings = True
        try:
            if not SETTINGS_PATH.exists():
                self.log(f"Settings: no saved settings yet ({SETTINGS_PATH})")
                self.apply_model_defaults(self.current_model_key())
                return
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                self.apply_model_defaults(self.current_model_key())
                return

            self._switch_defaults_on_model_change = bool(data.get("switch_defaults_on_model_change", True))
            if hasattr(self, "switch_defaults_on_model_change_check"):
                self.switch_defaults_on_model_change_check.setChecked(self._switch_defaults_on_model_change)
            self._use_framevision_queue = bool(data.get("use_framevision_queue", True))
            if hasattr(self, "framevision_queue_check"):
                self.framevision_queue_check.setChecked(self._use_framevision_queue)

            self._model_settings_cache = {key: self.default_generation_settings_for_model(key) for key in MODEL_DEFAULTS}
            raw_model_settings = data.get("model_settings")
            if isinstance(raw_model_settings, dict):
                for key, saved in raw_model_settings.items():
                    if key in MODEL_DEFAULTS and isinstance(saved, dict):
                        merged = self.default_generation_settings_for_model(key)
                        merged.update(saved)
                        self._model_settings_cache[key] = merged
            else:
                legacy_key = str(data.get("model_key", self.current_model_key()))
                if legacy_key not in MODEL_DEFAULTS:
                    legacy_key = "base"
                legacy = self.default_generation_settings_for_model(legacy_key)
                legacy.update({
                    "width": int(data.get("width", legacy["width"])),
                    "height": int(data.get("height", legacy["height"])),
                    "resolution_preset": data.get("resolution_preset", self.match_resolution_preset(int(data.get("width", legacy["width"])), int(data.get("height", legacy["height"])))),
                    "steps": int(data.get("steps", legacy["steps"])),
                    "guidance_scale": float(data.get("guidance_scale", legacy["guidance_scale"])),
                    "shift": float(data.get("shift", legacy["shift"])),
                    "seed": int(data.get("seed", legacy["seed"])),
                    "scheduler_name": data.get("scheduler_name", legacy["scheduler_name"]),
                    "timesteps": data.get("timesteps", legacy["timesteps"]),
                    "noise_scale_start": float(data.get("noise_scale_start", legacy["noise_scale_start"])),
                    "noise_scale_end": float(data.get("noise_scale_end", legacy["noise_scale_end"])),
                    "noise_clip_std": float(data.get("noise_clip_std", legacy["noise_clip_std"])),
                })
                self._model_settings_cache[legacy_key] = legacy

            if self._switch_defaults_on_model_change:
                for model_key in list(self._model_settings_cache):
                    if model_key in MODEL_DEFAULTS:
                        self._model_settings_cache[model_key] = self.apply_model_switch_defaults_to_settings(model_key, self._model_settings_cache[model_key])

            key = str(data.get("model_key", "base"))
            if key in MODEL_DEFAULTS:
                self.set_combo_by_data(self.model_combo, key)
            self.prompt_edit.setPlainText(str(data.get("prompt", self.prompt_edit.toPlainText())))
            self.edit_prompt.setPlainText(str(data.get("edit_prompt", self.edit_prompt.toPlainText())))
            if hasattr(self, "multi_prompt"):
                self.multi_prompt.setPlainText(str(data.get("multi_prompt", self.multi_prompt.toPlainText())))
            saved_negative = str(data.get("negative_prompt", data.get("negative", "")))
            self.negative_prompt_edit.setPlainText(saved_negative)
            if hasattr(self, "edit_negative_prompt_edit"):
                self.edit_negative_prompt_edit.setPlainText(saved_negative)
            if hasattr(self, "multi_negative_prompt_edit"):
                self.multi_negative_prompt_edit.setPlainText(str(data.get("multi_negative_prompt", saved_negative)))
            if data.get("output_dir"):
                self.output_dir_edit.setText(str(data.get("output_dir")))
            # Theme/HUD settings from older standalone saves are intentionally ignored
            # when this UI is embedded in FrameVision. The host app owns those controls.
            self._last_reference_folder = str(data.get("last_reference_folder", "") or "")
            self._last_dialog_folder = str(data.get("last_dialog_folder", "") or "")
            self.keep_aspect.setChecked(bool(data.get("keep_original_aspect", self.keep_aspect.isChecked())))

            offload = data.get("offload_settings", {}) if isinstance(data.get("offload_settings"), dict) else {}
            # Backward-compatible import from the previous placeholder offload UI.
            try_auto = bool(offload.get("try_auto_cpu_offload", offload.get("offload_model_to_cpu", False)))
            self.auto_offload_check.setChecked(try_auto)
            self.offload_folder_edit.setText(str(offload.get("offload_folder", DEFAULT_OFFLOAD_FOLDER)))

            self._active_model_key = self.current_model_key()
            self.apply_model_settings_to_ui(self._active_model_key)
            if hasattr(self, "main_generation_section"):
                self.main_generation_section.set_expanded(bool(data.get("main_generation_expanded", True)))
            self.edit_generation_section.set_expanded(bool(data.get("edit_generation_expanded", False)))
            if hasattr(self, "multi_generation_section"):
                self.multi_generation_section.set_expanded(bool(data.get("multi_generation_expanded", False)))

            active_tab = int(data.get("active_tab", self.tabs.currentIndex()))
            if 0 <= active_tab < self.tabs.count():
                self.tabs.setCurrentIndex(active_tab)
            self.log(f"Settings loaded: {SETTINGS_PATH}")
        except Exception as exc:
            self.log(f"Settings: failed to load {SETTINGS_PATH}: {exc}")
            self.apply_model_defaults(self.current_model_key())
        finally:
            self._loading_settings = False

    def save_settings(self) -> None:
        if self._loading_settings:
            return
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(json.dumps(self.settings_data(), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            self.log(f"Settings: failed to save {SETTINGS_PATH}: {exc}")

    def _connect_settings_autosave(self) -> None:
        for widget in [self.noise_start_spin, self.noise_end_spin, self.noise_clip_spin]:
            widget.valueChanged.connect(self.save_settings)
        self.negative_prompt_edit.textChanged.connect(lambda: self.sync_negative_prompt("main"))
        if hasattr(self, "edit_negative_prompt_edit"):
            self.edit_negative_prompt_edit.textChanged.connect(lambda: self.sync_negative_prompt("edit"))
        if hasattr(self, "multi_negative_prompt_edit"):
            self.multi_negative_prompt_edit.textChanged.connect(lambda: self.sync_negative_prompt("multi"))
        if hasattr(self, "multi_prompt"):
            self.multi_prompt.textChanged.connect(self.save_settings)
        self.output_dir_edit.editingFinished.connect(self.save_settings)
        self.keep_aspect.toggled.connect(self.save_settings)
        self.tabs.currentChanged.connect(self.save_settings)
        self.switch_defaults_on_model_change_check.toggled.connect(lambda _=None: self.on_switch_defaults_on_model_change_toggled())
        self.framevision_queue_check.toggled.connect(lambda _=None: self.on_framevision_queue_toggled())
        self.auto_offload_check.toggled.connect(self.save_settings)
        self.offload_folder_edit.editingFinished.connect(self.save_settings)
        if hasattr(self, "main_generation_section"):
            self.main_generation_section.toggle_btn.toggled.connect(self.save_settings)
        self.edit_generation_section.toggle_btn.toggled.connect(self.save_settings)
        if hasattr(self, "multi_generation_section"):
            self.multi_generation_section.toggle_btn.toggled.connect(self.save_settings)

    def browse_output_dir(self) -> None:
        start = self.output_dir_edit.text() or self._last_dialog_folder or str(DEFAULT_OUTPUT)
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", start)
        if folder:
            self.output_dir_edit.setText(folder)
            self._last_dialog_folder = folder
            self.save_settings()

    def browse_offload_folder(self) -> None:
        start = self.offload_folder_edit.text() or str(DEFAULT_OFFLOAD_FOLDER)
        folder = QFileDialog.getExistingDirectory(self, "Choose offload folder", start)
        if folder:
            self.offload_folder_edit.setText(folder)
            self.save_settings()

    def add_reference_images(self) -> None:
        start = self._last_reference_folder or self._last_dialog_folder or str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(self, "Add reference images", start, "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)")
        if files:
            self.ref_list.add_images(files)
            self._last_reference_folder = str(Path(files[0]).parent)
            self._last_dialog_folder = self._last_reference_folder
            self.save_settings()

    def add_multi_reference_images(self) -> None:
        start = self._last_reference_folder or self._last_dialog_folder or str(Path.home())
        files, _ = QFileDialog.getOpenFileNames(self, "Add multi-reference images", start, "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)")
        if files:
            self.multi_ref_list.add_images(files)
            self._last_reference_folder = str(Path(files[0]).parent)
            self._last_dialog_folder = self._last_reference_folder
            self.save_settings()

    def multi_reference_prompt_with_roles(self, prompt: str, refs: list[dict]) -> str:
        if not refs:
            return prompt
        lines = [prompt.strip(), "", "Reference image roles:"]
        for index, ref in enumerate(refs, start=1):
            path = Path(str(ref.get("path", "")))
            role = str(ref.get("role") or "General reference")
            lines.append(f"{index}. {role}: {path.name}")
        return "\n".join(lines).strip()

    def selected_generation_settings(self, ui_key: str = "main", negative_prompt: str | None = None) -> dict:
        self.store_current_model_settings(self.current_model_key())
        settings = dict(self._model_settings_cache.get(self.current_model_key(), self.default_generation_settings_for_model(self.current_model_key())))
        if ui_key in self._gen_widget_sets:
            settings.update(self.generation_settings_from_widgets(ui_key))
            if ui_key in ("edit", "multi"):
                settings = self.reference_safe_settings(ui_key, settings)
            self._model_settings_cache[self.current_model_key()] = dict(settings)
        settings.update(self.current_advanced_settings())
        if self.is_dev_variant():
            settings["guidance_scale"] = 0.0
            settings["negative_prompt"] = ""
        else:
            if negative_prompt is None:
                negative_prompt = self.negative_prompt_edit.toPlainText().strip()
            settings["negative_prompt"] = str(negative_prompt or "").strip()
        settings["offload_settings"] = self.current_offload_settings()
        return settings

    def model_label(self, key: str) -> str:
        info = MODEL_DEFAULTS.get(key)
        return str(info.get("label", key)) if isinstance(info, dict) else key

    def cli_supports_option(self, option: str) -> bool:
        try:
            return bool(CLI_PATH and CLI_PATH.exists() and option in CLI_PATH.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return False

    def build_args_for_job(self, job: dict) -> list[str]:
        if not ENV_PY or not CLI_PATH.exists():
            raise RuntimeError("Environment or HiDream CLI path is missing.")
        settings = job.get("settings") or {}
        out = Path(str(job.get("output_path", ""))).expanduser()
        args = [
            str(ENV_PY), str(CLI_PATH),
            "--model_key", str(job.get("model_key") or "base"),
            "--width", str(int(settings.get("width", 1280))),
            "--height", str(int(settings.get("height", 704))),
            "--steps", str(int(settings.get("steps", 28))),
            "--guidance_scale", str(float(settings.get("guidance_scale", 0.0))),
            "--shift", str(float(settings.get("shift", 1.0))),
            "--seed", str(int(settings.get("seed", -1))),
            "--scheduler_name", str(settings.get("scheduler_name", "flash")),
            "--timesteps", str(settings.get("timesteps", "none")),
            "--noise_scale_start", str(float(settings.get("noise_scale_start", 7.5))),
            "--noise_scale_end", str(float(settings.get("noise_scale_end", 7.5))),
            "--noise_clip_std", str(float(settings.get("noise_clip_std", 2.5))),
            "--output_image", str(out),
            "--prompt", str(job.get("prompt", "")),
            "--resolution_mode", "framevision",
        ]
        offload = settings.get("offload_settings", {}) if isinstance(settings.get("offload_settings"), dict) else {}
        try_auto_offload = bool(offload.get("try_auto_cpu_offload", False))
        if self.cli_supports_option("--device_map"):
            if try_auto_offload:
                offload_folder = Path(str(offload.get("offload_folder") or DEFAULT_OFFLOAD_FOLDER)).expanduser()
                offload_folder.mkdir(parents=True, exist_ok=True)
                args.extend(["--device_map", "auto", "--offload_folder", str(offload_folder)])
            else:
                args.extend(["--device_map", "cuda"])
        elif try_auto_offload:
            self.log("Auto offload selected, but this runner does not support --device_map yet; running CUDA-only.")

        negative_prompt = str(settings.get("negative_prompt", "")).strip()
        if self.is_full_variant(str(job.get("model_key") or "base")) and negative_prompt:
            args.extend(["--negative_prompt", negative_prompt])
        refs = [str(x) for x in (job.get("refs") or []) if str(x).strip()]
        if refs:
            args.extend(["--ref_images", *refs])
        if bool(job.get("keep_original_aspect")):
            args.append("--keep_original_aspect")
        return args

    def enqueue_framevision_job(self, mode: str) -> bool:
        try:
            try:
                from helpers.queue_adapter import enqueue_hidream_from_widget as _enq
            except Exception:
                from queue_adapter import enqueue_hidream_from_widget as _enq

            jid = _enq(self, mode=mode)
            self.status_label.setText("Added to FrameVision queue")
            try:
                self.switch_to_framevision_queue_tab()
            except Exception:
                pass
            try:
                self.log(f"Added to FrameVision queue: {jid}")
            except Exception:
                pass
            return True
        except Exception as exc:
            QMessageBox.warning(self, "Queue error", f"Could not add HiDream job to the FrameVision queue.\n\n{exc}")
            return False

    def output_path(self, prefix: str) -> Path:
        out_dir = Path(self.output_dir_edit.text()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        reserved = {str(Path(str(j.get("output_path", ""))).expanduser()) for j in self.queue_jobs}
        index = 1
        while True:
            candidate = out_dir / f"{prefix}_{index:04d}.png"
            if not candidate.exists() and str(candidate.expanduser()) not in reserved:
                return candidate
            index += 1

    def require_selected_model(self) -> bool:
        key = self.current_model_key()
        if self.model_installed(key):
            return True
        QMessageBox.warning(self, "Model not installed", f"{MODEL_DEFAULTS[key]['label']} is not installed yet.\n\nRun install.bat again and choose this model, or choose Both.")
        return False

    def enqueue_create_image(self) -> None:
        if not self.require_selected_model():
            return
        prompt = self.prompt_edit.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Enter a prompt first.")
            return
        if self.using_framevision_queue():
            self.enqueue_framevision_job("create")
            return
        key = self.current_model_key()
        out = self.output_path(f"hidream_{key}_create")
        self.add_queue_job({
            "type": "create",
            "model_key": key,
            "prompt": prompt,
            "refs": [],
            "keep_original_aspect": False,
            "settings": self.selected_generation_settings("main", self.negative_prompt_edit.toPlainText().strip()),
            "output_path": str(out),
        })

    def enqueue_edit_image(self) -> None:
        if not self.require_selected_model():
            return
        prompt = self.edit_prompt.toPlainText().strip()
        refs = self.ref_list.paths()
        if not prompt:
            QMessageBox.warning(self, "Missing edit instruction", "Enter an edit instruction first.")
            return
        if not refs:
            QMessageBox.warning(self, "Missing reference image", "Add at least one reference image.")
            return
        if self.using_framevision_queue():
            self.enqueue_framevision_job("edit")
            return
        key = self.current_model_key()
        out = self.output_path(f"hidream_{key}_edit")
        self.add_queue_job({
            "type": "edit",
            "model_key": key,
            "prompt": prompt,
            "refs": refs,
            "keep_original_aspect": self.keep_aspect.isChecked(),
            "settings": self.selected_generation_settings("edit", self.edit_negative_prompt_edit.toPlainText().strip() if hasattr(self, "edit_negative_prompt_edit") else None),
            "output_path": str(out),
        })

    def enqueue_multi_reference_image(self) -> None:
        if not self.require_selected_model():
            return
        prompt = self.multi_prompt.toPlainText().strip()
        refs = self.multi_ref_list.references()
        if not prompt:
            QMessageBox.warning(self, "Missing instruction", "Enter a multi-reference instruction first.")
            return
        if not refs:
            QMessageBox.warning(self, "Missing reference images", "Add at least one reference image.")
            return
        if self.using_framevision_queue():
            self.enqueue_framevision_job("multi_reference")
            return
        key = self.current_model_key()
        out = self.output_path(f"hidream_{key}_multi_ref")
        full_prompt = self.multi_reference_prompt_with_roles(prompt, refs)
        self.add_queue_job({
            "type": "multi_reference",
            "model_key": key,
            "prompt": full_prompt,
            "raw_prompt": prompt,
            "refs": [ref["path"] for ref in refs],
            "ref_roles": refs,
            "keep_original_aspect": False,
            "settings": self.selected_generation_settings("multi", self.multi_negative_prompt_edit.toPlainText().strip() if hasattr(self, "multi_negative_prompt_edit") else None),
            "output_path": str(out),
        })

    # Backward-compatible method names in case an external script calls them.
    def generate_image(self) -> None:
        self.enqueue_create_image()

    def edit_image(self) -> None:
        self.enqueue_edit_image()

    def add_queue_job(self, job: dict) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        job = dict(job)
        job.setdefault("id", uuid4().hex)
        job.setdefault("status", "pending")
        job.setdefault("created_at", now)
        job.setdefault("started_at", "")
        job.setdefault("finished_at", "")
        job.setdefault("error", "")
        self.queue_jobs.append(job)
        self.log(f"Queued {job.get('type', 'job')} job for {self.model_label(str(job.get('model_key', '')))}: {Path(str(job.get('output_path'))).name}")
        self.refresh_queue_list()
        self.save_queue()
        self.start_next_queue_job()

    def queue_job_title(self, job: dict) -> str:
        status = str(job.get("status", "pending")).upper()
        job_type = job.get("type")
        kind = "Multi-reference" if job_type == "multi_reference" else ("Edit" if job_type == "edit" else "Create")
        model = self.model_label(str(job.get("model_key", "")))
        output = Path(str(job.get("output_path", ""))).name
        prompt = " ".join(str(job.get("prompt", "")).split())
        if len(prompt) > 90:
            prompt = prompt[:87] + "..."
        return f"[{status}] {kind} | {model} | {output}\n{prompt}"

    def refresh_queue_list(self) -> None:
        if not hasattr(self, "queue_list"):
            return
        selected_id = None
        cur = self.queue_list.currentItem()
        if cur:
            selected_id = cur.data(Qt.UserRole)
        self.queue_list.blockSignals(True)
        self.queue_list.clear()
        for job in self.queue_jobs:
            item = QListWidgetItem(self.queue_job_title(job))
            item.setData(Qt.UserRole, job.get("id"))
            if job.get("status") == "running":
                item.setText("▶ " + item.text())
            self.queue_list.addItem(item)
            if selected_id and selected_id == job.get("id"):
                self.queue_list.setCurrentItem(item)
        self.queue_list.blockSignals(False)
        pending = sum(1 for j in self.queue_jobs if j.get("status") == "pending")
        running = sum(1 for j in self.queue_jobs if j.get("status") == "running")
        done = sum(1 for j in self.queue_jobs if j.get("status") == "done")
        failed = sum(1 for j in self.queue_jobs if j.get("status") in {"failed", "cancelled"})
        if hasattr(self, "queue_status_label"):
            self.queue_status_label.setText(f"Queue: {pending} pending, {running} running, {done} finished, {failed} failed/cancelled")

    def queue_job_by_id(self, job_id: str | None) -> dict | None:
        if not job_id:
            return None
        for job in self.queue_jobs:
            if job.get("id") == job_id:
                return job
        return None

    def selected_queue_job(self) -> dict | None:
        item = self.queue_list.currentItem() if hasattr(self, "queue_list") else None
        return self.queue_job_by_id(item.data(Qt.UserRole)) if item else None

    def _queue_item_clicked(self, item: QListWidgetItem) -> None:
        job = self.queue_job_by_id(item.data(Qt.UserRole))
        if not job:
            return
        out = Path(str(job.get("output_path", ""))).expanduser()
        if out.exists():
            self.preview.set_image(out)

    def move_selected_queue_job(self, direction: int) -> None:
        job = self.selected_queue_job()
        if not job or job.get("status") == "running":
            return
        idx = self.queue_jobs.index(job)
        new_idx = max(0, min(len(self.queue_jobs) - 1, idx + direction))
        if new_idx == idx:
            return
        self.queue_jobs[idx], self.queue_jobs[new_idx] = self.queue_jobs[new_idx], self.queue_jobs[idx]
        self.refresh_queue_list()
        for i in range(self.queue_list.count()):
            if self.queue_list.item(i).data(Qt.UserRole) == job.get("id"):
                self.queue_list.setCurrentRow(i)
                break
        self.save_queue()

    def remove_selected_queue_job(self) -> None:
        job = self.selected_queue_job()
        if not job or job.get("status") == "running":
            QMessageBox.information(self, "Queue", "The running job cannot be removed. Use Stop current first.")
            return
        self.queue_jobs = [j for j in self.queue_jobs if j.get("id") != job.get("id")]
        self.refresh_queue_list()
        self.save_queue()

    def retry_selected_queue_job(self) -> None:
        job = self.selected_queue_job()
        if not job or job.get("status") == "running":
            return
        job["status"] = "pending"
        job["started_at"] = ""
        job["finished_at"] = ""
        job["error"] = ""
        old_out = Path(str(job.get("output_path", "")))
        if old_out.name:
            prefix = old_out.stem.rsplit("_", 1)[0]
            job["output_path"] = str(self.output_path(prefix))
        self.refresh_queue_list()
        self.save_queue()
        self.start_next_queue_job()

    def clear_finished_queue_jobs(self) -> None:
        self.queue_jobs = [j for j in self.queue_jobs if j.get("status") not in {"done", "failed", "cancelled"}]
        self.refresh_queue_list()
        self.save_queue()

    def queue_payload_for_disk(self) -> dict:
        unfinished = []
        for job in self.queue_jobs:
            if job.get("status") in {"pending", "running"}:
                copy = dict(job)
                unfinished.append(copy)
        return {"version": 1, "jobs": unfinished}

    def load_queue(self) -> None:
        try:
            if not QUEUE_PATH.exists():
                self.refresh_queue_list()
                return
            data = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
            jobs = data.get("jobs", []) if isinstance(data, dict) else []
            self.queue_jobs = []
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                if job.get("status") == "running":
                    job["status"] = "pending"
                    job["error"] = "Restarted after app close; queued again."
                elif job.get("status") != "pending":
                    continue
                job.setdefault("id", uuid4().hex)
                job.setdefault("type", "create")
                job.setdefault("settings", {})
                job.setdefault("refs", [])
                job.setdefault("keep_original_aspect", False)
                self.queue_jobs.append(job)
            self.refresh_queue_list()
            if self.queue_jobs:
                self.log(f"Queue restored: {len(self.queue_jobs)} unfinished job(s) from {QUEUE_PATH}")
        except Exception as exc:
            self.log(f"Queue: failed to load {QUEUE_PATH}: {exc}")
            self.refresh_queue_list()

    def save_queue(self) -> None:
        try:
            QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
            QUEUE_PATH.write_text(json.dumps(self.queue_payload_for_disk(), indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            self.log(f"Queue: failed to save {QUEUE_PATH}: {exc}")

    def start_next_queue_job(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            return
        if not ENV_PY or not ENV_PY.exists() or not RUNNER or not RUNNER.exists():
            return
        job = next((j for j in self.queue_jobs if j.get("status") == "pending"), None)
        if not job:
            self.refresh_queue_list()
            return
        key = str(job.get("model_key") or "base")
        if key not in MODEL_DEFAULTS or not self.model_installed(key):
            job["status"] = "failed"
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            job["error"] = f"Model not installed or unknown: {key}"
            self.log(f"Queue job failed before start: {job['error']}")
            self.refresh_queue_list()
            self.save_queue()
            self.start_next_queue_job()
            return
        try:
            args = self.build_args_for_job(job)
        except Exception as exc:
            job["status"] = "failed"
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            job["error"] = str(exc)
            self.refresh_queue_list()
            self.save_queue()
            self.start_next_queue_job()
            return
        out = Path(str(job.get("output_path"))).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.current_queue_job_id = str(job.get("id"))
        job["status"] = "running"
        job["started_at"] = datetime.now().isoformat(timespec="seconds")
        job["error"] = ""
        self.refresh_queue_list()
        self.save_queue()
        self.start_process(args, out, job)

    def start_process(self, args: list[str], expected_output: Path, job: dict | None = None) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Busy", "A generation is already running.")
            return
        if not ENV_PY or not ENV_PY.exists() or not RUNNER or not RUNNER.exists():
            QMessageBox.critical(self, "Missing install", "The HiDream environment or shared runner was not found. Expected env: /environments/.hidream_dev/python.exe. Run install.bat first.")
            return
        self.save_settings()
        self.last_output = expected_output
        self.preview.setText("Generating...")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        model_text = self.model_label(str(job.get("model_key", self.current_model_key()))) if job else self.model_label(self.current_model_key())
        self.status_label.setText(f"Generating queued job... {model_text}")
        self.stop_btn.setEnabled(True)
        self.log("\n--- Starting HiDream queued job ---")
        self.log(" ".join(f'"{a}"' if " " in a else a for a in args))
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        env = self.process.processEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUTF8", "1")
        env.insert("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.finished.connect(self._process_finished)
        self.process.start(args[0], args[1:])

    def _read_process_output(self) -> None:
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.log(data.rstrip())
            self.update_progress_from_text(data)

    def update_progress_from_text(self, text: str) -> None:
        matches = re.findall(r"(\d+)\s*/\s*(\d+)", text)
        if matches:
            cur, total = map(int, matches[-1])
            if total > 0:
                self.progress.setValue(max(0, min(100, int(cur * 100 / total))))

    def _process_finished(self, code: int, status) -> None:
        self.stop_btn.setEnabled(False)
        job = self.queue_job_by_id(self.current_queue_job_id)
        stopped = self._stopping_current_job
        self._stopping_current_job = False
        if job:
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
        if stopped:
            self.status_label.setText("Cancelled")
            self.log("Process cancelled by user.")
            if job:
                job["status"] = "cancelled"
                job["error"] = "Cancelled by user."
        elif code == 0:
            self.progress.setValue(100)
            self.status_label.setText("Finished")
            if job:
                job["status"] = "done"
            if self.last_output and self.last_output.exists():
                self.preview.set_image(self.last_output)
                self.log(f"Result: {self.last_output}")
            else:
                self.log("Process finished, but expected output was not found.")
                if job:
                    job["status"] = "failed"
                    job["error"] = "Expected output was not found."
        else:
            self.status_label.setText(f"Failed ({code})")
            self.log(f"Process failed with exit code {code}.")
            if job:
                job["status"] = "failed"
                job["error"] = f"Exit code {code}"
        self.current_queue_job_id = None
        self.process = None
        self.refresh_queue_list()
        self.save_queue()
        self.start_next_queue_job()

    def stop_generation(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self._stopping_current_job = True
            self.log("Stopping current queue job...")
            self.process.kill()

    def view_results(self) -> None:
        folder = Path(self.output_dir_edit.text()).expanduser()
        _fv_open_results_in_media_explorer(self, folder, preset="images")

    def open_output_folder(self) -> None:
        folder = Path(self.output_dir_edit.text()).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(str(folder))
        else:
            subprocess.Popen(["xdg-open", str(folder)])

    def closeEvent(self, event) -> None:
        self.save_settings()
        self.save_queue()
        if self.process and self.process.state() != QProcess.NotRunning:
            self.log("Closing app: saving current queue state and stopping the running process.")
            try:
                self.process.finished.disconnect(self._process_finished)
            except Exception:
                pass
            self.process.kill()
            self.process.waitForFinished(1500)
        super().closeEvent(event)

    def log(self, text: str) -> None:
        self.log_box.appendPlainText(str(text))


def main() -> None:
    app = QApplication(sys.argv)
    win = HiDreamUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
