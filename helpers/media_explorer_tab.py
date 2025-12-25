
# media_explorer_tab.py
# A lightweight, extensible Media Explorer tab (PySide6) for Windows 10/11.
# - Scans a folder (optionally subfolders) for Images / Video / Audio
# - Shows a sortable/searchable table with per-file metadata
# - Uses ffprobe/ffplay/ffmpeg from <root>/presets/bin/ (auto-discovered)
# - No thumbnails (uses default per-type icons)
#
# Drop this file into your project and import MediaExplorerTab as a QWidget tab.

from __future__ import annotations

import os
import sys
import json
import math
import shutil
import subprocess
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


# -----------------------------
# Small dialogs (Tree view)
# -----------------------------


class FolderPickDialog(QtWidgets.QDialog):
    """Ask the user for a folder with OK/Cancel (plus Browse…)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, title: str = "Select folder") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._folder: str = ""

        l = QtWidgets.QVBoxLayout(self)
        l.setContentsMargins(10, 10, 10, 10)
        l.setSpacing(8)

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.ed = QtWidgets.QLineEdit(self)
        self.ed.setPlaceholderText("Select a folder…")

        self.btn_browse = QtWidgets.QToolButton(self)
        self.btn_browse.setText("Browse…")
        self.btn_browse.clicked.connect(self._browse)

        row.addWidget(self.ed, 1)
        row.addWidget(self.btn_browse, 0)

        self.buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.accepted.connect(self._accept)
        self.buttons.rejected.connect(self.reject)

        l.addLayout(row)
        l.addWidget(self.buttons)

        # Disable OK until a valid folder exists.
        self._btn_ok = self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        if self._btn_ok:
            self._btn_ok.setEnabled(False)
        self.ed.textChanged.connect(self._on_text_changed)

    def folder(self) -> str:
        return self._folder

    def _browse(self) -> None:
        start = self.ed.text().strip()
        if not start:
            try:
                start = str(Path.home())
            except Exception:
                start = ""
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            self.ed.setText(folder)

    def _on_text_changed(self, text: str) -> None:
        text = (text or "").strip()
        ok = bool(text) and Path(text).expanduser().is_dir()
        if self._btn_ok:
            self._btn_ok.setEnabled(ok)

    def _accept(self) -> None:
        p = (self.ed.text() or "").strip()
        if not p:
            return
        if not Path(p).expanduser().is_dir():
            QtWidgets.QMessageBox.warning(self, "Tree view", "Folder does not exist.")
            return
        self._folder = str(Path(p).expanduser())
        self.accept()


class FolderTreeDialog(QtWidgets.QDialog):
    """Show a simple directory-only tree view for a chosen root folder."""

    def __init__(self, root_folder: str, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tree view")
        self.setModal(False)
        self.resize(720, 520)

        root_folder = (root_folder or "").strip()

        l = QtWidgets.QVBoxLayout(self)
        l.setContentsMargins(10, 10, 10, 10)
        l.setSpacing(8)

        self.lbl = QtWidgets.QLabel(root_folder, self)
        self.lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl.setWordWrap(True)
        l.addWidget(self.lbl, 0)

        self.tree = QtWidgets.QTreeView(self)
        self.tree.setHeaderHidden(True)
        self.tree.setAnimated(True)
        self.tree.setIndentation(18)
        self.tree.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        # Right-click: allow quickly opening & scanning a subfolder in Media Explorer.
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._on_tree_context_menu)

        self.fs_model = QtWidgets.QFileSystemModel(self)
        self.fs_model.setFilter(
            QtCore.QDir.Filter.AllDirs | QtCore.QDir.Filter.NoDotAndDotDot
        )
        self.fs_model.setRootPath(root_folder)

        self.tree.setModel(self.fs_model)
        # Show only the name column.
        for c in range(1, self.fs_model.columnCount()):
            self.tree.setColumnHidden(c, True)

        try:
            idx = self.fs_model.index(root_folder)
            self.tree.setRootIndex(idx)
        except Exception:
            pass

        l.addWidget(self.tree, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        l.addWidget(buttons, 0)

    def _on_tree_context_menu(self, pos: QtCore.QPoint) -> None:
        try:
            idx = self.tree.indexAt(pos)
            if not idx.isValid():
                return
            folder = self.fs_model.filePath(idx)
            if not folder:
                return
            # Only allow folders.
            if not Path(folder).is_dir():
                return

            menu = QtWidgets.QMenu(self)
            act_open_scan = menu.addAction("Open && Scan in Media Explorer")
            chosen = menu.exec(self.tree.viewport().mapToGlobal(pos))
            if chosen == act_open_scan:
                cb = None
                try:
                    cb = getattr(self.parent(), "open_and_scan", None)
                except Exception:
                    cb = None
                if callable(cb):
                    cb(folder)
                self.close()
        except Exception:
            pass


# -----------------------------
# Utilities
# -----------------------------

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"
}
VIDEO_EXTS = {
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".wmv"
}
AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".aiff", ".aif"
}



# Windows: move files to Recycle Bin (Trash) instead of permanent delete.
# Uses SHFileOperationW with FOF_ALLOWUNDO.
def move_to_recycle_bin(path_str: str) -> Tuple[bool, str]:
    try:
        if os.name != "nt":
            # Best-effort fallback for non-Windows.
            os.remove(path_str)
            return True, ""
        p = os.path.abspath(path_str)

        class SHFILEOPSTRUCTW(ctypes.Structure):
            _fields_ = [
                ("hwnd", wintypes.HWND),
                ("wFunc", wintypes.UINT),
                ("pFrom", wintypes.LPCWSTR),
                ("pTo", wintypes.LPCWSTR),
                ("fFlags", wintypes.WORD),
                ("fAnyOperationsAborted", wintypes.BOOL),
                ("hNameMappings", wintypes.LPVOID),
                ("lpszProgressTitle", wintypes.LPCWSTR),
            ]

        FO_DELETE = 0x0003
        FOF_SILENT = 0x0004
        FOF_NOCONFIRMATION = 0x0010
        FOF_ALLOWUNDO = 0x0040
        FOF_NOERRORUI = 0x0400

        flags = FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_NOERRORUI | FOF_SILENT

        # SHFileOperation expects a double-NUL terminated string.
        buf = ctypes.create_unicode_buffer(p + "\0\0")

        op = SHFILEOPSTRUCTW()
        op.hwnd = None
        op.wFunc = FO_DELETE
        op.pFrom = ctypes.cast(buf, wintypes.LPCWSTR)
        op.pTo = None
        op.fFlags = flags
        op.fAnyOperationsAborted = False
        op.hNameMappings = None
        op.lpszProgressTitle = None

        res = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
        if res != 0 or bool(op.fAnyOperationsAborted):
            return False, f"Recycle Bin move failed (code {int(res)})"
        return True, ""
    except Exception as e:
        # As a last resort, attempt permanent delete.
        try:
            os.remove(path_str)
            return True, ""
        except Exception as e2:
            return False, f"{e} / fallback delete failed: {e2}"


def human_size(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return ""
    n = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(n)} {unit}"
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TB"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None or seconds <= 0:
        return ""
    s = int(round(seconds))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m}:{sec:02d}"


def safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def parse_fps(rate: Optional[str]) -> Optional[float]:
    if not rate or rate == "0/0":
        return None
    if "/" in rate:
        a, b = rate.split("/", 1)
        try:
            num = float(a)
            den = float(b)
            if den == 0:
                return None
            return num / den
        except Exception:
            return None
    return safe_float(rate)


def detect_media_type(ext: str) -> str:
    ext = ext.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return "other"


def norm_path(p: str) -> str:
    """Normalized path key for favorites (case-insensitive on Windows)."""
    try:
        return str(Path(p).expanduser().resolve()).lower()
    except Exception:
        return (p or "").strip().lower()


def canon_path(p: str) -> str:
    """Canonical absolute path string for persistence."""
    try:
        return str(Path(p).expanduser().resolve())
    except Exception:
        return (p or "").strip()


def resolve_presets_bin(start: Optional[Path] = None) -> Optional[Path]:
    """
    Find <root>/presets/bin where ffprobe.exe, ffplay.exe, ffmpeg.exe live.
    We search upward from:
      - provided start (if any)
      - current working directory
      - this file's directory
    """
    candidates: List[Path] = []
    if start:
        candidates.append(start)
    candidates.append(Path.cwd())
    try:
        candidates.append(Path(__file__).resolve().parent)
    except Exception:
        pass

    # Also support env var override
    env_root = os.environ.get("APP_ROOT") or os.environ.get("FRAMEVISION_ROOT") or os.environ.get("FRAMELAB_ROOT")
    if env_root:
        candidates.insert(0, Path(env_root))

    checked: set[Path] = set()

    for base in candidates:
        base = base.resolve()
        for parent in [base] + list(base.parents)[:8]:
            if parent in checked:
                continue
            checked.add(parent)
            bin_dir = parent / "presets" / "bin"
            ffprobe = bin_dir / "ffprobe.exe"
            ffplay = bin_dir / "ffplay.exe"
            ffmpeg = bin_dir / "ffmpeg.exe"
            if ffprobe.exists() and ffplay.exists() and ffmpeg.exists():
                return bin_dir

    return None


# -----------------------------
# Data model
# -----------------------------

@dataclass
class MediaItem:
    path: str
    name: str
    ext: str
    media_type: str  # image|video|audio|other
    found_in: str

    size_bytes: Optional[int] = None
    mtime: Optional[float] = None

    # video/audio
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    bitrate_bps: Optional[int] = None
    codec: Optional[str] = None

    # image/video
    width: Optional[int] = None
    height: Optional[int] = None

    # audio
    sample_rate: Optional[int] = None
    channels: Optional[int] = None

    # free-form for future
    extra: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}

    @property
    def mtime_dt(self) -> Optional[datetime]:
        if self.mtime is None:
            return None
        try:
            return datetime.fromtimestamp(self.mtime)
        except Exception:
            return None

    def resolution_str(self) -> str:
        if self.width and self.height:
            return f"{self.width}×{self.height}"
        return ""

    def fps_str(self) -> str:
        if self.fps is None or self.fps <= 0:
            return ""
        # show up to 3 decimals but trim trailing zeros
        s = f"{self.fps:.3f}".rstrip("0").rstrip(".")
        return s

    def quality_str(self) -> str:
        # Basic "quality" label: codec + bitrate/sample rate, tailored by type
        parts: List[str] = []
        if self.codec:
            parts.append(self.codec)
        if self.media_type in ("video", "audio"):
            if self.bitrate_bps and self.bitrate_bps > 0:
                parts.append(f"{int(self.bitrate_bps/1000)} kb/s")
        if self.media_type == "audio":
            if self.sample_rate and self.sample_rate > 0:
                parts.append(f"{int(self.sample_rate/1000)} kHz")
            if self.channels:
                if self.channels == 1:
                    parts.append("mono")
                elif self.channels == 2:
                    parts.append("stereo")
                else:
                    parts.append(f"{self.channels}ch")
        if self.media_type == "image":
            # codec isn't typically meaningful; use extension as format
            if not parts:
                parts.append(self.ext.upper().lstrip("."))
        return " • ".join(parts)


class Column:
    ICON = 0
    NAME = 1
    TYPE = 2
    EXT = 3
    SIZE = 4
    MODIFIED = 5
    DURATION = 6
    RESOLUTION = 7
    FPS = 8
    QUALITY = 9
    FOUND_IN = 10

    labels = [
        "", "Name", "Type", "Ext", "Size", "Modified", "Duration", "Resolution", "FPS", "Quality", "Found in"
    ]


# -----------------------------
# ffprobe helpers
# -----------------------------

class FFTools(QtCore.QObject):
    toolsChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.bin_dir: Optional[Path] = resolve_presets_bin()
        self.ffprobe: Optional[Path] = (self.bin_dir / "ffprobe.exe") if self.bin_dir else None
        self.ffplay: Optional[Path] = (self.bin_dir / "ffplay.exe") if self.bin_dir else None
        self.ffmpeg: Optional[Path] = (self.bin_dir / "ffmpeg.exe") if self.bin_dir else None

    def is_ready(self) -> bool:
        return bool(self.ffprobe and self.ffprobe.exists())

    def refresh(self) -> None:
        self.bin_dir = resolve_presets_bin()
        self.ffprobe = (self.bin_dir / "ffprobe.exe") if self.bin_dir else None
        self.ffplay = (self.bin_dir / "ffplay.exe") if self.bin_dir else None
        self.ffmpeg = (self.bin_dir / "ffmpeg.exe") if self.bin_dir else None
        self.toolsChanged.emit()

    def probe(self, file_path: str, timeout_s: int = 15) -> Optional[Dict[str, Any]]:
        if not self.ffprobe or not self.ffprobe.exists():
            return None
        cmd = [
            str(self.ffprobe),
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            file_path,
        ]
        try:
            p = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
            if p.returncode != 0 or not p.stdout.strip():
                return None
            return json.loads(p.stdout)
        except Exception:
            return None

    def play_with_ffplay(self, file_path: str, is_audio: bool) -> bool:
        if not self.ffplay or not self.ffplay.exists():
            return False
        args = ["-autoexit"]
        if is_audio:
            args += ["-nodisp"]
        args.append(file_path)
        # Use QProcess so we don't block.
        return QtCore.QProcess.startDetached(str(self.ffplay), args)


# -----------------------------
# Worker thread: scan + metadata extraction
# -----------------------------

class ScanParams(QtCore.QObject):
    def __init__(self) -> None:
        super().__init__()
        self.root_folder: str = ""
        self.include_subfolders: bool = True
        self.want_images: bool = True
        self.want_videos: bool = True
        self.want_audio: bool = True


class ScanWorker(QtCore.QThread):
    progressChanged = QtCore.Signal(int, int)  # current, total
    # NOTE: pushing one item per signal can still make the UI sluggish with huge folders.
    # We keep itemReady for compatibility, but the tab uses itemsReady (batched).
    itemReady = QtCore.Signal(object)          # MediaItem
    itemsReady = QtCore.Signal(object)         # List[MediaItem]
    statusChanged = QtCore.Signal(str)
    finishedOk = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, params: ScanParams, fftools: FFTools, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.params = params
        self.fftools = fftools
        self._abort = False

    def request_abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            folder = Path(self.params.root_folder).expanduser().resolve()
            if not folder.exists() or not folder.is_dir():
                self.failed.emit("Folder not found.")
                return

            self.statusChanged.emit("Scanning files…")

            exts: set[str] = set()
            if self.params.want_images:
                exts |= IMAGE_EXTS
            if self.params.want_videos:
                exts |= VIDEO_EXTS
            if self.params.want_audio:
                exts |= AUDIO_EXTS

            # Gather files first (fast), so we know total.
            paths: List[Path] = []
            if self.params.include_subfolders:
                for p in folder.rglob("*"):
                    if self._abort:
                        self.statusChanged.emit("Scan aborted.")
                        return
                    if p.is_file() and p.suffix.lower() in exts:
                        paths.append(p)
            else:
                for p in folder.iterdir():
                    if self._abort:
                        self.statusChanged.emit("Scan aborted.")
                        return
                    if p.is_file() and p.suffix.lower() in exts:
                        paths.append(p)

            total = len(paths)
            self.statusChanged.emit(f"Found {total} files. Reading metadata…")
            if total == 0:
                self.progressChanged.emit(0, 0)
                self.finishedOk.emit()
                return

            # Emit results in batches so the UI stays responsive.
            batch: List[MediaItem] = []
            batch_size = 80  # tweak as needed; higher = fewer UI updates

            for i, p in enumerate(paths, start=1):
                if self._abort:
                    self.statusChanged.emit("Scan aborted.")
                    return

                item = self._build_item(p)
                batch.append(item)

                # Push a batch to the UI periodically.
                if len(batch) >= batch_size:
                    self.itemsReady.emit(batch)
                    batch = []

                # Throttle progress signals (they run on the UI thread).
                if i == 1 or i == total or (i % 10 == 0):
                    self.progressChanged.emit(i, total)

            # Flush remainder
            if batch:
                self.itemsReady.emit(batch)

            self.statusChanged.emit("Done.")
            self.finishedOk.emit()

        except Exception as e:
            self.failed.emit(str(e))

    def _build_item(self, p: Path) -> MediaItem:
        st = p.stat()
        ext = p.suffix.lower()
        media_type = detect_media_type(ext)

        item = MediaItem(
            path=str(p),
            name=p.name,
            ext=ext,
            media_type=media_type,
            found_in=str(p.parent),
            size_bytes=st.st_size,
            mtime=st.st_mtime,
        )

        # Precompute normalized path (used for Favorites).
        try:
            item.extra["norm_path"] = norm_path(item.path)
        except Exception:
            pass

        # Images: use QImageReader to get size quickly without decoding full image.
        if media_type == "image":
            try:
                reader = QtGui.QImageReader(str(p))
                sz = reader.size()
                if sz.isValid():
                    item.width = sz.width()
                    item.height = sz.height()
                # codec for image = format if available
                fmt = bytes(reader.format()).decode("ascii", errors="ignore").strip().upper()
                if fmt:
                    item.codec = fmt
            except Exception:
                pass
            return item

        # Video/Audio: use ffprobe if available
        probe = self.fftools.probe(str(p)) if self.fftools.is_ready() else None
        if not probe:
            return item

        fmt = probe.get("format") or {}
        streams = probe.get("streams") or []

        item.duration_sec = safe_float(fmt.get("duration"))
        item.bitrate_bps = safe_int(fmt.get("bit_rate"))

        # Prefer first matching stream
        if media_type == "video":
            v = next((s for s in streams if s.get("codec_type") == "video"), None)
            if v:
                item.codec = v.get("codec_name") or item.codec
                item.width = safe_int(v.get("width"))
                item.height = safe_int(v.get("height"))
                item.fps = parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate"))
                # Sometimes stream has more accurate bitrate
                sb = safe_int(v.get("bit_rate"))
                if sb and sb > 0:
                    item.bitrate_bps = sb
        elif media_type == "audio":
            a = next((s for s in streams if s.get("codec_type") == "audio"), None)
            if a:
                item.codec = a.get("codec_name") or item.codec
                item.sample_rate = safe_int(a.get("sample_rate"))
                item.channels = safe_int(a.get("channels"))
                sb = safe_int(a.get("bit_rate"))
                if sb and sb > 0:
                    item.bitrate_bps = sb

        return item


# -----------------------------
# Qt Model + Proxy for sorting/filtering
# -----------------------------

ROLE_SORT_NUM = QtCore.Qt.ItemDataRole.UserRole + 1
ROLE_SORT_STR = QtCore.Qt.ItemDataRole.UserRole + 2
ROLE_ITEM = QtCore.Qt.ItemDataRole.UserRole + 3


class MediaTableModel(QtCore.QAbstractTableModel):
    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._items: List[MediaItem] = []
        # Favorites are stored as a set of normalized absolute paths.
        self._favorites_norm: set[str] = set()

    def set_favorites(self, favorites_norm: set[str]) -> None:
        self._favorites_norm = set(favorites_norm or set())
        # Refresh visible text (star prefix) without resetting the model.
        try:
            if self._items:
                a = self.index(0, 0)
                b = self.index(len(self._items) - 1, self.columnCount() - 1)
                self.dataChanged.emit(a, b)
        except Exception:
            pass

    def _is_favorite(self, item: MediaItem) -> bool:
        try:
            k = item.extra.get("norm_path") if item.extra else None
            if not k:
                k = norm_path(item.path)
            return k in self._favorites_norm
        except Exception:
            return False


    def clear(self) -> None:
        self.beginResetModel()
        self._items = []
        self.endResetModel()

    def add_item(self, item: MediaItem) -> None:
        row = len(self._items)
        self.beginInsertRows(QtCore.QModelIndex(), row, row)
        self._items.append(item)
        self.endInsertRows()

    def add_items(self, items: List[MediaItem]) -> None:
        """Batch insert for performance (avoids one insert signal per item)."""
        if not items:
            return
        row0 = len(self._items)
        row1 = row0 + len(items) - 1
        self.beginInsertRows(QtCore.QModelIndex(), row0, row1)
        self._items.extend(items)
        self.endInsertRows()

    def item_at(self, row: int) -> Optional[MediaItem]:
        if 0 <= row < len(self._items):
            return self._items[row]
        return None

    def remove_rows(self, rows: List[int]) -> None:
        """Remove multiple rows (source model rows). Rows are removed safely in reverse order."""
        if not rows:
            return
        for r in sorted(set(rows), reverse=True):
            if r < 0 or r >= len(self._items):
                continue
            self.beginRemoveRows(QtCore.QModelIndex(), r, r)
            try:
                del self._items[r]
            finally:
                self.endRemoveRows()

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._items)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(Column.labels)

    def headerData(self, section: int, orientation: QtCore.Qt.Orientation, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            if 0 <= section < len(Column.labels):
                return Column.labels[section]
        return None

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        item = self._items[index.row()]
        col = index.column()

        if role == ROLE_ITEM:
            return item

        if role == QtCore.Qt.ItemDataRole.ToolTipRole:
            # Rich tooltip: quick glance
            lines = [
                item.name,
                item.path,
                f"Type: {item.media_type}",
            ]
            if self._is_favorite(item):
                lines.append("★ Favorite")
            lines.append(f"Size: {human_size(item.size_bytes)}")
            if item.mtime_dt:
                lines.append(f"Modified: {item.mtime_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            if item.duration_sec:
                lines.append(f"Duration: {format_duration(item.duration_sec)}")
            if item.width and item.height:
                lines.append(f"Resolution: {item.width}×{item.height}")
            if item.fps:
                lines.append(f"FPS: {item.fps_str()}")
            q = item.quality_str()
            if q:
                lines.append(f"Quality: {q}")
            return "\n".join(lines)

        if role == QtCore.Qt.ItemDataRole.DecorationRole and col == Column.ICON:
            return self._icon_for(item)

        if role in (QtCore.Qt.ItemDataRole.DisplayRole, ROLE_SORT_STR, ROLE_SORT_NUM):
            return self._data_for(item, col, role)

        return None

    def _icon_for(self, item: MediaItem) -> QtGui.QIcon:
        # Use standard Qt icons; caller style controls look.
        style = QtWidgets.QApplication.instance().style() if QtWidgets.QApplication.instance() else None
        if not style:
            return QtGui.QIcon()
        if item.media_type == "image":
            return style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon)
        if item.media_type == "video":
            return style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaPlay)
        if item.media_type == "audio":
            return style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_MediaVolume)
        return style.standardIcon(QtWidgets.QStyle.StandardPixmap.SP_FileIcon)

    def _data_for(self, item: MediaItem, col: int, role: int):
        # Display strings
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == Column.ICON:
                return ""
            if col == Column.NAME:
                return ("★ " + item.name) if self._is_favorite(item) else item.name
            if col == Column.TYPE:
                return item.media_type
            if col == Column.EXT:
                return item.ext.lstrip(".").upper()
            if col == Column.SIZE:
                return human_size(item.size_bytes)
            if col == Column.MODIFIED:
                return item.mtime_dt.strftime("%Y-%m-%d %H:%M:%S") if item.mtime_dt else ""
            if col == Column.DURATION:
                return format_duration(item.duration_sec)
            if col == Column.RESOLUTION:
                return item.resolution_str()
            if col == Column.FPS:
                return item.fps_str()
            if col == Column.QUALITY:
                return item.quality_str()
            if col == Column.FOUND_IN:
                return item.found_in
            return ""

        # Sorting helpers
        if role == ROLE_SORT_STR:
            if col == Column.NAME:
                # Keep sorting stable regardless of the displayed star prefix.
                return item.name or ""
            if col in (Column.TYPE, Column.EXT, Column.MODIFIED, Column.QUALITY, Column.FOUND_IN):
                return self._data_for(item, col, QtCore.Qt.ItemDataRole.DisplayRole) or ""
            return 

        if role == ROLE_SORT_NUM:
            if col == Column.SIZE:
                return item.size_bytes or 0
            if col == Column.MODIFIED:
                return item.mtime or 0
            if col == Column.DURATION:
                return item.duration_sec or 0.0
            if col == Column.FPS:
                return item.fps or 0.0
            if col == Column.RESOLUTION:
                # sort by total pixels
                if item.width and item.height:
                    return int(item.width * item.height)
                return 0
            return 0

        return None


class MediaSortFilterProxy(QtCore.QSortFilterProxyModel):
    """Proxy model that supports:
      - Search query filtering
      - Favorites-only filtering
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.setDynamicSortFilter(True)
        self._query = ""
        self._favorites_norm: set[str] = set()
        self._favorites_only: bool = False

        # Sorting is handled normally; favorites are not pinned.

    def set_query(self, q: str) -> None:
        self._query = (q or "").strip().lower()
        self.invalidateFilter()

    def set_favorites_set(self, favorites_norm: set[str]) -> None:
        self._favorites_norm = set(favorites_norm or set())
        self.invalidateFilter()
        try:
            self.invalidate()
        except Exception:
            pass

    def set_favorites_only(self, on: bool) -> None:
        self._favorites_only = bool(on)
        self.invalidateFilter()

    def _is_favorite_row(self, source_row: int, source_parent: QtCore.QModelIndex) -> bool:
        try:
            m = self.sourceModel()
            idx = m.index(source_row, Column.NAME, source_parent)
            item = m.data(idx, ROLE_ITEM)
            if item is None:
                return False
            k = item.extra.get("norm_path") if getattr(item, "extra", None) else None
            if not k:
                k = norm_path(item.path)
            return k in self._favorites_norm
        except Exception:
            return False

    def filterAcceptsRow(self, source_row: int, source_parent: QtCore.QModelIndex) -> bool:
        if self._favorites_only and not self._is_favorite_row(source_row, source_parent):
            return False

        if not self._query:
            return True

        m = self.sourceModel()
        idx_name = m.index(source_row, Column.NAME, source_parent)
        idx_path = m.index(source_row, Column.FOUND_IN, source_parent)

        # Note: NAME display may include a star prefix; use sort string which excludes it.
        name = (m.data(idx_name, ROLE_SORT_STR) or "").lower()
        found_in = (m.data(idx_path, QtCore.Qt.ItemDataRole.DisplayRole) or "").lower()
        return (self._query in name) or (self._query in found_in)

    def _cmp_values(self, lv, rv, numeric: bool) -> int:
        try:
            if numeric:
                lf = float(lv)
                rf = float(rv)
                return -1 if lf < rf else (1 if lf > rf else 0)
            ls = str(lv).lower()
            rs = str(rv).lower()
            return -1 if ls < rs else (1 if ls > rs else 0)
        except Exception:
            ls = str(lv)
            rs = str(rv)
            return -1 if ls < rs else (1 if ls > rs else 0)

    def lessThan(self, left: QtCore.QModelIndex, right: QtCore.QModelIndex) -> bool:
        m = self.sourceModel()
        col = left.column()

        numeric_cols = (Column.SIZE, Column.MODIFIED, Column.DURATION, Column.RESOLUTION, Column.FPS)

        if col in numeric_cols:
            lv = m.data(left, ROLE_SORT_NUM) or 0
            rv = m.data(right, ROLE_SORT_NUM) or 0
            return self._cmp_values(lv, rv, numeric=True) < 0

        lv = m.data(left, ROLE_SORT_STR)
        rv = m.data(right, ROLE_SORT_STR)

        # Fallback to display role if sort-role isn't provided.
        if lv is None:
            lv = m.data(left, QtCore.Qt.ItemDataRole.DisplayRole)
        if rv is None:
            rv = m.data(right, QtCore.Qt.ItemDataRole.DisplayRole)

        return self._cmp_values(lv or "", rv or "", numeric=False) < 0


# -----------------------------
# Details pane widgets
# -----------------------------

class CollapsibleSection(QtWidgets.QWidget):
    """Simple collapsible section with a header and a content area."""

    def __init__(self, title: str, collapsed: bool = True, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._collapsed = bool(collapsed)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        header = QtWidgets.QWidget(self)
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)

        self.btn = QtWidgets.QToolButton(header)
        self.btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.btn.setArrowType(QtCore.Qt.ArrowType.RightArrow if self._collapsed else QtCore.Qt.ArrowType.DownArrow)
        self.btn.setCheckable(True)
        self.btn.setChecked(not self._collapsed)
        self.btn.setAutoRaise(True)

        self.lbl = QtWidgets.QLabel(title, header)
        f = self.lbl.font()
        f.setBold(True)
        self.lbl.setFont(f)

        hl.addWidget(self.btn, 0)
        hl.addWidget(self.lbl, 1)

        self.content = QtWidgets.QFrame(self)
        self.content.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.content_l = QtWidgets.QVBoxLayout(self.content)
        self.content_l.setContentsMargins(14, 0, 0, 0)
        self.content_l.setSpacing(6)

        outer.addWidget(header, 0)
        outer.addWidget(self.content, 0)

        self.content.setVisible(not self._collapsed)
        self.btn.toggled.connect(self.set_expanded)

    def set_widget(self, w: QtWidgets.QWidget) -> None:
        # Clear existing
        while self.content_l.count():
            it = self.content_l.takeAt(0)
            if it and it.widget():
                it.widget().setParent(None)
        self.content_l.addWidget(w)

    def set_expanded(self, on: bool) -> None:
        self._collapsed = not bool(on)
        self.content.setVisible(on)
        self.btn.setArrowType(QtCore.Qt.ArrowType.DownArrow if on else QtCore.Qt.ArrowType.RightArrow)


class SmartTextValue(QtWidgets.QWidget):
    """Preview long text with Expand + Copy buttons."""

    def __init__(self, text: str, limit: int = 200, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._full = text or ""
        self._limit = max(40, int(limit))
        self._expanded = False

        l = QtWidgets.QHBoxLayout(self)
        l.setContentsMargins(0, 0, 0, 0)
        l.setSpacing(6)

        self.lbl = QtWidgets.QLabel(self)
        self.lbl.setWordWrap(True)
        self.lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl.setText(self._preview_text())

        self.btn_expand = QtWidgets.QToolButton(self)
        self.btn_expand.setText("Expand")
        self.btn_expand.setToolTip("Show full text.")
        self.btn_expand.setAutoRaise(True)

        self.btn_copy = QtWidgets.QToolButton(self)
        self.btn_copy.setText("Copy")
        self.btn_copy.setToolTip("Copy full text to clipboard.")
        self.btn_copy.setAutoRaise(True)

        l.addWidget(self.lbl, 1)
        l.addWidget(self.btn_expand, 0)
        l.addWidget(self.btn_copy, 0)

        self.btn_expand.clicked.connect(self._toggle)
        self.btn_copy.clicked.connect(self._copy)

    def _preview_text(self) -> str:
        t = self._full
        if self._expanded or len(t) <= self._limit:
            return t
        return t[: self._limit].rstrip() + "…"

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self.lbl.setText(self._preview_text())
        self.btn_expand.setText("Collapse" if self._expanded else "Expand")

    def _copy(self) -> None:
        try:
            QtWidgets.QApplication.clipboard().setText(self._full)
        except Exception:
            pass

# -----------------------------
# The Tab UI
# -----------------------------

class MediaExplorerTab(QtWidgets.QWidget):
    """A QWidget suitable for inserting into a QTabWidget.

    Public helpers:
      - set_root_folder(path)
      - rescan()
      - clear()
    """

    # Emitted when the user asks to open a file *inside* FrameVision's built-in player.
    # FrameVision's MainWindow may connect to this if desired, but this tab also
    # attempts a best-effort direct open via self.window().video.open(...).
    openInPlayerRequested = QtCore.Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        self.fftools = FFTools(self)

        self.model = MediaTableModel(self)
        self.proxy = MediaSortFilterProxy(self)
        self.proxy.setSourceModel(self.model)

        self._scanner: Optional[ScanWorker] = None

        self._build_ui()

        # Persist table column order/sizes and sort between runs.
        self._settings = QtCore.QSettings()
        self._settings_group = "MediaExplorerTab"
        self._restore_table_state()
        # Favorites (persisted via QSettings)
        self._favorites_map: Dict[str, str] = self._load_favorites_map()
        self._favorites_norm: set[str] = set(self._favorites_map.keys())
        self.model.set_favorites(self._favorites_norm)
        self.proxy.set_favorites_set(self._favorites_norm)


        # Fix header readability on dark themes (some styles leave headers white).
        self._apply_header_theme()

        self._connect()
        self._ensure_default_sort_newest_first()
        # Tools are auto-discovered; UI does not expose a manual refresh/status row.

        # Nice-to-have: remember last folder in-session (you can swap to QSettings later)
        self._last_folder: Optional[str] = None

        # In-app clipboard for Copy/Cut/Paste between folders.
        self._clip_mode: Optional[str] = None  # 'copy' or 'cut'
        self._clip_paths: List[str] = []
        self._pending_paste: Optional[Dict[str, Any]] = None

    # ---------- UI ----------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Top controls
        top = QtWidgets.QFrame(self)
        top.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        top_l = QtWidgets.QGridLayout(top)
        top_l.setContentsMargins(8, 8, 8, 8)
        top_l.setHorizontalSpacing(8)
        top_l.setVerticalSpacing(6)

        self.ed_folder = QtWidgets.QLineEdit(top)
        self.ed_folder.setPlaceholderText("Select a folder to scan…")
        self.ed_folder.setToolTip("Folder to scan for media files.")

        self.btn_browse = QtWidgets.QToolButton(top)
        self.btn_browse.setText("Browse…")
        self.btn_browse.setToolTip("Choose a folder.")

        self.btn_rescan = QtWidgets.QToolButton(top)
        self.btn_rescan.setText("Scan")
        self.btn_rescan.setToolTip("Scan the selected folder and load matching media files.")

        self.btn_stop = QtWidgets.QToolButton(top)
        self.btn_stop.setText("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop the current scan.")

        self.btn_tree_view = QtWidgets.QToolButton(top)
        self.btn_tree_view.setText("Tree view")
        self.btn_tree_view.setToolTip("Pick a folder and show its subfolder tree.")

        self.cb_subfolders = QtWidgets.QCheckBox("Include subfolders", top)
        self.cb_subfolders.setChecked(True)
        self.cb_subfolders.setToolTip("If enabled, scans all subfolders recursively.")

        self.cb_images = QtWidgets.QCheckBox("Images", top)
        self.cb_images.setChecked(True)
        self.cb_images.setToolTip("Include image files.")

        self.cb_videos = QtWidgets.QCheckBox("Video", top)
        self.cb_videos.setChecked(True)
        self.cb_videos.setToolTip("Include video files.")

        self.cb_audio = QtWidgets.QCheckBox("Sound", top)
        self.cb_audio.setChecked(True)
        self.cb_audio.setToolTip("Include audio files.")

        self.ed_search = QtWidgets.QLineEdit(top)
        self.ed_search.setPlaceholderText("Search (name or folder)…")
        self.ed_search.setClearButtonEnabled(True)
        self.ed_search.setToolTip("Filter results by file name or the 'Found in' folder path.")

        
        self.btn_favs_only = QtWidgets.QToolButton(top)
        self.btn_favs_only.setText("★ Only")
        self.btn_favs_only.setCheckable(True)
        self.btn_favs_only.setChecked(False)
        self.btn_favs_only.setToolTip("Show only favorite files (toggle).")

        top_l.addWidget(QtWidgets.QLabel("Folder:", top), 0, 0)
        top_l.addWidget(self.ed_folder, 0, 1, 1, 4)
        top_l.addWidget(self.btn_browse, 0, 5)
        top_l.addWidget(self.btn_rescan, 0, 6)
        top_l.addWidget(self.btn_stop, 0, 7)
        # Filters row: keep the checkboxes packed tightly and let them use the full row width.
        # This avoids truncating labels too aggressively when the window gets narrow.
        filters_wrap = QtWidgets.QWidget(top)
        filters_l = QtWidgets.QHBoxLayout(filters_wrap)
        filters_l.setContentsMargins(0, 0, 0, 0)
        filters_l.setSpacing(8)
        filters_l.addWidget(self.btn_tree_view)
        filters_l.addWidget(self.cb_subfolders)
        filters_l.addWidget(self.cb_images)
        filters_l.addWidget(self.cb_videos)
        filters_l.addWidget(self.cb_audio)
        filters_l.addStretch(1)

        top_l.addWidget(filters_wrap, 1, 1, 1, 7)

        top_l.addWidget(QtWidgets.QLabel("Search:", top), 2, 0)
        top_l.addWidget(self.ed_search, 2, 1, 1, 6)
        top_l.addWidget(self.btn_favs_only, 2, 7)

        layout.addWidget(top)

        # Splitter: table + details
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical, self)

        # Table
        self.table = QtWidgets.QTableView(self)
        self.table.setModel(self.proxy)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        # Allow users to reorder columns by dragging the header.
        hh = self.table.horizontalHeader()
        hh.setSectionsMovable(True)
        hh.setSectionsClickable(True)
        hh.setHighlightSections(False)
        self.table.horizontalHeader().setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.setToolTip("Right click for actions. Double click to open in player.")

        # Column sizing
        self.table.setColumnWidth(Column.ICON, 26)
        self.table.setColumnWidth(Column.NAME, 260)
        self.table.setColumnWidth(Column.MODIFIED, 160)
        self.table.setColumnWidth(Column.RESOLUTION, 110)
        self.table.setColumnWidth(Column.SIZE, 90)
        self.table.setColumnWidth(Column.TYPE, 80)
        self.table.setColumnWidth(Column.EXT, 60)
        self.table.setColumnWidth(Column.DURATION, 90)
        self.table.setColumnWidth(Column.FPS, 70)
        self.table.setColumnWidth(Column.QUALITY, 190)

        # Bottom status row
        status_bar = QtWidgets.QFrame(self)
        status_l = QtWidgets.QHBoxLayout(status_bar)
        status_l.setContentsMargins(0, 0, 0, 0)
        status_l.setSpacing(8)

        self.progress = QtWidgets.QProgressBar(status_bar)
        self.progress.setMinimum(0)
        self.progress.setMaximum(0)
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)

        self.lbl_status = QtWidgets.QLabel("Ready.", status_bar)
        self.lbl_status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        self.lbl_count = QtWidgets.QLabel("0 items", status_bar)
        self.lbl_count.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        status_l.addWidget(self.progress, 0)
        status_l.addWidget(self.lbl_status, 1)
        status_l.addWidget(self.lbl_count, 0)

        table_wrap = QtWidgets.QWidget(self)
        table_wrap_l = QtWidgets.QVBoxLayout(table_wrap)
        table_wrap_l.setContentsMargins(0, 0, 0, 0)
        table_wrap_l.setSpacing(6)
        table_wrap_l.addWidget(self.table, 1)
        table_wrap_l.addWidget(status_bar, 0)

        splitter.addWidget(table_wrap)

        # Details pane
        details = QtWidgets.QFrame(self)
        details.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        details_l = QtWidgets.QVBoxLayout(details)
        details_l.setContentsMargins(8, 8, 8, 8)
        details_l.setSpacing(6)

        # Scrollable details content
        self.details_scroll = QtWidgets.QScrollArea(details)
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.details_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.details_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.details_root = QtWidgets.QWidget(details)
        self.details_root_l = QtWidgets.QVBoxLayout(self.details_root)
        self.details_root_l.setContentsMargins(0, 0, 0, 0)
        self.details_root_l.setSpacing(8)

        # Header: filename + type badge
        header = QtWidgets.QWidget(self.details_root)
        header_l = QtWidgets.QHBoxLayout(header)
        header_l.setContentsMargins(0, 0, 0, 0)
        header_l.setSpacing(8)

        self.lbl_file_name = QtWidgets.QLabel(header)
        self.lbl_file_name.setText("Select an item to see details.")
        self.lbl_file_name.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.lbl_file_name.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_file_name.setWordWrap(True)
        self.lbl_file_name.setStyleSheet("font-size: 16px; font-weight: 700;")

        self.lbl_type_badge = QtWidgets.QLabel(header)
        self.lbl_type_badge.setText("")
        self.lbl_type_badge.setVisible(False)
        self.lbl_type_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.lbl_type_badge.setStyleSheet(
            "QLabel{"
            "padding: 2px 10px;"
            "border-radius: 10px;"
            "background-color: palette(mid);"
            "color: palette(text);"
            "}"
        )

        header_l.addWidget(self.lbl_file_name, 1)
        header_l.addWidget(self.lbl_type_badge, 0, QtCore.Qt.AlignmentFlag.AlignTop)

        # Core info rows (kept from original pane)
        info = QtWidgets.QWidget(self.details_root)
        info_l = QtWidgets.QGridLayout(info)
        info_l.setContentsMargins(0, 0, 0, 0)
        info_l.setHorizontalSpacing(10)
        info_l.setVerticalSpacing(6)

        def _mk_key(s: str) -> QtWidgets.QLabel:
            lab = QtWidgets.QLabel(s, info)
            f = lab.font()
            f.setBold(True)
            lab.setFont(f)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            return lab

        def _mk_val() -> QtWidgets.QLabel:
            lab = QtWidgets.QLabel("", info)
            lab.setWordWrap(True)
            lab.setTextFormat(QtCore.Qt.TextFormat.PlainText)
            lab.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            lab.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
            return lab

        self._val_full_path = _mk_val()
        self._val_size_modified = _mk_val()
        self._val_res_quality = _mk_val()

        info_l.addWidget(_mk_key("Full path:"), 0, 0)
        info_l.addWidget(self._val_full_path, 0, 1)
        info_l.addWidget(_mk_key("Size + Modified:"), 1, 0)
        info_l.addWidget(self._val_size_modified, 1, 1)
        info_l.addWidget(_mk_key("Resolution + Quality:"), 2, 0)
        info_l.addWidget(self._val_res_quality, 2, 1)
        info_l.setColumnStretch(1, 1)

        # Metadata (JSON)
        meta = QtWidgets.QWidget(self.details_root)
        meta_l = QtWidgets.QVBoxLayout(meta)
        meta_l.setContentsMargins(0, 6, 0, 0)
        meta_l.setSpacing(6)

        self.lbl_meta_title = QtWidgets.QLabel("Metadata (JSON)", meta)
        f = self.lbl_meta_title.font()
        f.setBold(True)
        self.lbl_meta_title.setFont(f)

        self.lbl_meta_status = QtWidgets.QLabel("Select a file to load metadata JSON.", meta)
        self.lbl_meta_status.setWordWrap(True)
        self.lbl_meta_status.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self.lbl_meta_status.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        self.smart_table = QtWidgets.QTableWidget(meta)
        self.smart_table.setColumnCount(2)
        self.smart_table.setRowCount(0)
        self.smart_table.setHorizontalHeaderLabels(["Key", "Value"])
        self.smart_table.horizontalHeader().setVisible(False)
        self.smart_table.verticalHeader().setVisible(False)
        self.smart_table.setShowGrid(False)
        self.smart_table.setWordWrap(True)
        self.smart_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.smart_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.smart_table.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.smart_table.setAlternatingRowColors(False)
        self.smart_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.smart_table.horizontalHeader().setStretchLastSection(True)
        self.smart_table.setVisible(False)

        self.json_tree = QtWidgets.QTreeWidget(meta)
        self.json_tree.setColumnCount(2)
        self.json_tree.setHeaderLabels(["Key", "Value"])
        self.json_tree.setHeaderHidden(True)
        self.json_tree.setRootIsDecorated(True)
        self.json_tree.setAlternatingRowColors(False)
        self.json_tree.setUniformRowHeights(False)
        self.json_tree.setExpandsOnDoubleClick(True)
        self.json_tree.setWordWrap(True)

        self.section_everything = CollapsibleSection("Everything else", collapsed=True, parent=meta)
        self.section_everything.set_widget(self.json_tree)
        self.section_everything.setVisible(False)

        meta_l.addWidget(self.lbl_meta_title, 0)
        meta_l.addWidget(self.lbl_meta_status, 0)
        meta_l.addWidget(self.smart_table, 0)
        meta_l.addWidget(self.section_everything, 0)

        # Compose scroll content
        self.details_root_l.addWidget(header, 0)
        self.details_root_l.addWidget(info, 0)
        self.details_root_l.addWidget(meta, 0)
        self.details_root_l.addStretch(1)

        self.details_scroll.setWidget(self.details_root)

        self.btn_open_folder = QtWidgets.QToolButton(details)
        self.btn_open_folder.setText("Open folder")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.setToolTip("Open the containing folder of the selected file in Explorer.")

        self.btn_copy_path = QtWidgets.QToolButton(details)
        self.btn_copy_path.setText("Copy path")
        self.btn_copy_path.setEnabled(False)
        self.btn_copy_path.setToolTip("Copy full file path of the selected item.")

        details_buttons = QtWidgets.QHBoxLayout()
        details_buttons.setContentsMargins(0, 0, 0, 0)
        details_buttons.setSpacing(8)
        details_buttons.addWidget(self.btn_open_folder)
        details_buttons.addWidget(self.btn_copy_path)
        details_buttons.addStretch(1)

        details_l.addWidget(self.details_scroll, 1)
        details_l.addLayout(details_buttons, 0)

        splitter.addWidget(details)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)

    def _connect(self) -> None:
        self.btn_browse.clicked.connect(self._browse_folder)
        self.btn_rescan.clicked.connect(self.rescan)
        self.btn_stop.clicked.connect(self._stop_scan)
        self.btn_tree_view.clicked.connect(self._open_tree_view)

        self.ed_search.textChanged.connect(self.proxy.set_query)

        self.btn_favs_only.toggled.connect(self._set_favorites_only)

        # Keyboard: 'F' toggles favorite on selected files (table focused).
        try:
            self._sc_toggle_fav = QtGui.QShortcut(QtGui.QKeySequence("F"), self.table)
            self._sc_toggle_fav.activated.connect(self._toggle_favorite_selected)
        except Exception:
            pass


        # Keyboard: Delete key moves selected files to the Windows Recycle Bin (table focused).
        try:
            self._sc_delete_from_drive = QtGui.QShortcut(QtGui.QKeySequence("Delete"), self.table)
            self._sc_delete_from_drive.setContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
            self._sc_delete_from_drive.activated.connect(self._delete_selected_from_drive)
        except Exception:
            pass

        self.table.doubleClicked.connect(self._on_double_click)
        self.table.customContextMenuRequested.connect(self._open_context_menu)

        sel = self.table.selectionModel()
        sel.selectionChanged.connect(self._update_details)

        
        # Persist column order/size/sort when user changes headers.
        try:
            hh = self.table.horizontalHeader()
            hh.sectionMoved.connect(lambda *_: self._schedule_save_table_state())
            hh.sectionResized.connect(lambda *_: self._schedule_save_table_state())
            hh.sortIndicatorChanged.connect(lambda *_: self._schedule_save_table_state())
        except Exception:
            pass

        # Persist row height when user resizes rows.
        try:
            vh = self.table.verticalHeader()
            vh.sectionResized.connect(lambda *_: self._schedule_save_row_height())
        except Exception:
            pass

        # Ensure we flush state on app exit.
        try:
            app = QtWidgets.QApplication.instance()
            if app:
                try:
                    app.aboutToQuit.connect(self._save_table_state_now, QtCore.Qt.ConnectionType.UniqueConnection)
                    app.aboutToQuit.connect(self._save_row_height_now, QtCore.Qt.ConnectionType.UniqueConnection)
                except Exception:
                    # Fallback if UniqueConnection isn't supported in this binding.
                    app.aboutToQuit.connect(self._save_table_state_now)
                    app.aboutToQuit.connect(self._save_row_height_now)
        except Exception:
            pass

        self.btn_open_folder.clicked.connect(self._open_selected_folder)
        self.btn_copy_path.clicked.connect(self._copy_selected_path)

        # No tools UI: keep auto-discovery silent.

    # ---------- Public API ----------

    def set_root_folder(self, folder: str) -> None:
        self.ed_folder.setText(folder)


    def open_and_scan(self, folder: str) -> None:
        """Convenience helper: set folder and immediately rescan."""
        try:
            self.set_root_folder(folder)
        except Exception:
            pass
        try:
            self.rescan()
        except Exception:
            pass

    def clear(self) -> None:
        self._stop_scan()
        self.model.clear()
        self._update_counts()
        self.lbl_file_name.setText("Select an item to see details.")
        self.lbl_type_badge.setVisible(False)
        self._val_full_path.setText("")
        self._val_size_modified.setText("")
        self._val_res_quality.setText("")
        self._set_metadata_message("Select a file to load metadata JSON.")
        self.btn_open_folder.setEnabled(False)
        self.btn_copy_path.setEnabled(False)

    def rescan(self) -> None:
        folder = self.ed_folder.text().strip()
        if not folder:
            QtWidgets.QMessageBox.information(self, "Media Explorer", "Please select a folder first.")
            return
        if self._scanner and self._scanner.isRunning():
            QtWidgets.QMessageBox.information(self, "Media Explorer", "A scan is already running.")
            return

        # Basic validation
        p = Path(folder).expanduser()
        if not p.exists() or not p.is_dir():
            QtWidgets.QMessageBox.warning(self, "Media Explorer", "Folder does not exist.")
            return

        # Reset
        self.model.clear()
        self._update_counts()

        # Scans can add thousands of rows. If sorting is enabled during insertion,
        # Qt will keep resorting the proxy which can make the whole app feel frozen.
        # Temporarily disable sorting and re-enable (and sort once) at the end.
        try:
            self._scan_prev_sorting_enabled = bool(self.table.isSortingEnabled())
        except Exception:
            self._scan_prev_sorting_enabled = True
        try:
            self._scan_prev_dynamic_sort = bool(self.proxy.dynamicSortFilter())
        except Exception:
            self._scan_prev_dynamic_sort = True
        try:
            self.table.setSortingEnabled(False)
        except Exception:
            pass
        try:
            self.proxy.setDynamicSortFilter(False)
        except Exception:
            pass

        params = ScanParams()
        params.root_folder = str(p)
        params.include_subfolders = self.cb_subfolders.isChecked()
        params.want_images = self.cb_images.isChecked()
        params.want_videos = self.cb_videos.isChecked()
        params.want_audio = self.cb_audio.isChecked()

        if not (params.want_images or params.want_videos or params.want_audio):
            QtWidgets.QMessageBox.information(self, "Media Explorer", "Please enable at least one media type.")
            return

        self._last_folder = params.root_folder

        # Start worker
        self._scanner = ScanWorker(params, self.fftools, self)
        self._scanner.itemsReady.connect(self._on_items_ready)
        self._scanner.progressChanged.connect(self._on_progress)
        self._scanner.statusChanged.connect(self.lbl_status.setText)
        self._scanner.failed.connect(self._on_failed)
        self._scanner.finishedOk.connect(self._on_finished)

        self.btn_stop.setEnabled(True)
        self.btn_rescan.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate until we know totals

        self._scanner.start()

    # ---------- Scan callbacks ----------

    @QtCore.Slot(object)
    def _on_items_ready(self, items: object) -> None:
        """Receive a batch of MediaItem objects from the scanner thread."""
        try:
            # Preferred: one model insert for the whole batch.
            self.model.add_items(list(items or []))
        except Exception:
            # Fallback: insert one-by-one (still works, just slower).
            try:
                for it in (items or []):
                    self.model.add_item(it)
            except Exception:
                pass
        self._update_counts(defer=True)

    # Backward compatibility (unused by default)
    @QtCore.Slot(object)
    def _on_item_ready(self, item: MediaItem) -> None:
        self.model.add_item(item)
        self._update_counts(defer=True)

    @QtCore.Slot(int, int)
    def _on_progress(self, cur: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
            return
        if self.progress.maximum() != total:
            self.progress.setRange(0, total)
        self.progress.setValue(cur)

    @QtCore.Slot()
    def _on_finished(self) -> None:
        self.btn_stop.setEnabled(False)
        self.btn_rescan.setEnabled(True)
        self.progress.setVisible(False)
        self._update_counts()
        self._restore_sorting_after_scan()
        # Default sort: newest first (Modified desc) unless the user has a saved sort preference.
        # Only apply if sorting is enabled.
        try:
            if self.table.isSortingEnabled():
                col = self._settings.value(f"{self._settings_group}/sort_col", None)
                order = self._settings.value(f"{self._settings_group}/sort_order", None)
                if col is None or order is None:
                    self.table.sortByColumn(Column.MODIFIED, QtCore.Qt.SortOrder.DescendingOrder)
                else:
                    self.table.sortByColumn(int(col), QtCore.Qt.SortOrder(int(order)))
        except Exception:
            try:
                if self.table.isSortingEnabled():
                    self.table.sortByColumn(Column.MODIFIED, QtCore.Qt.SortOrder.DescendingOrder)
            except Exception:
                pass

        # If the user requested Paste and we auto-scanned the destination folder first,
        # run the paste now (on the GUI thread).
        try:
            pending = getattr(self, "_pending_paste", None)
            if pending and isinstance(pending, dict) and pending.get("stage") == "after_scan":
                self._pending_paste = None
                QtCore.QTimer.singleShot(0, lambda p=pending: self._do_paste_pending(p))
        except Exception:
            pass


    @QtCore.Slot(str)
    def _on_failed(self, msg: str) -> None:
        try:
            self._pending_paste = None
        except Exception:
            pass

        self.btn_stop.setEnabled(False)
        self.btn_rescan.setEnabled(True)
        self.progress.setVisible(False)
        self._restore_sorting_after_scan()
        self.lbl_status.setText(f"Error: {msg}")
        QtWidgets.QMessageBox.warning(self, "Media Explorer", msg)

    def _stop_scan(self) -> None:
        if self._scanner and self._scanner.isRunning():
            self._scanner.request_abort()
            self._scanner.wait(1500)
        self.btn_stop.setEnabled(False)
        self.btn_rescan.setEnabled(True)
        self.progress.setVisible(False)
        self._restore_sorting_after_scan()

    def _restore_sorting_after_scan(self) -> None:
        """Re-enable sorting that we temporarily disabled for scan performance."""
        # Restore dynamic sort/filter on the proxy first.
        try:
            prev_dyn = bool(getattr(self, "_scan_prev_dynamic_sort", True))
            self.proxy.setDynamicSortFilter(prev_dyn)
        except Exception:
            pass
        # Restore view sorting.
        try:
            prev_sort = bool(getattr(self, "_scan_prev_sorting_enabled", True))
            self.table.setSortingEnabled(prev_sort)
        except Exception:
            pass


    # ---------- Details & actions ----------

    def _selected_source_rows(self) -> List[int]:
        rows: List[int] = []
        sel = self.table.selectionModel()
        if not sel:
            return rows
        for idx in sel.selectedRows():
            src = self.proxy.mapToSource(idx)
            rows.append(src.row())
        return sorted(set(rows))

    def _selected_items(self) -> List[MediaItem]:
        items: List[MediaItem] = []
        for r in self._selected_source_rows():
            it = self.model.item_at(r)
            if it:
                items.append(it)
        return items

    # ---------- Favorites ----------

    def _favorites_settings_key(self) -> str:
        return f"{self._settings_group}/favorites_json"

    def _load_favorites_map(self) -> Dict[str, str]:
        """Return mapping: norm_path -> canonical_path."""
        try:
            raw = self._settings.value(self._favorites_settings_key(), "")
            if isinstance(raw, (QtCore.QByteArray, bytes)):
                try:
                    raw = bytes(raw).decode("utf-8", errors="ignore")
                except Exception:
                    raw = str(raw)
            raw = (raw or "").strip()
            if not raw:
                return {}
            arr = json.loads(raw)
            if not isinstance(arr, list):
                return {}
            out: Dict[str, str] = {}
            for p in arr:
                if not p:
                    continue
                cp = canon_path(str(p))
                np = norm_path(cp)
                out[np] = cp
            return out
        except Exception:
            return {}

    def _save_favorites_map(self) -> None:
        try:
            # Persist canonical paths (human-friendly) but keyed by normalized paths in memory.
            paths = sorted(set(self._favorites_map.values()), key=lambda s: s.lower())
            self._settings.setValue(self._favorites_settings_key(), json.dumps(paths))
        except Exception:
            pass

    def _is_favorite_path(self, file_path: str) -> bool:
        try:
            return norm_path(file_path) in self._favorites_norm
        except Exception:
            return False

    def _refresh_favorites_views(self, rows: Optional[List[int]] = None) -> None:
        """Update model/proxy and repaint affected rows."""
        try:
            self.model.set_favorites(self._favorites_norm)
            self.proxy.set_favorites_set(self._favorites_norm)
        except Exception:
            pass

        if rows:
            try:
                for r in rows:
                    a = self.model.index(r, 0)
                    b = self.model.index(r, self.model.columnCount() - 1)
                    self.model.dataChanged.emit(a, b)
            except Exception:
                pass

        try:
            self._update_counts()
            self._update_details()
        except Exception:
            pass

    def _set_favorites_only(self, on: bool) -> None:
        try:
            self.proxy.set_favorites_only(bool(on))
            self._update_counts()
        except Exception:
            pass

    def _toggle_favorite_selected(self) -> None:
        items = self._selected_items()
        rows = self._selected_source_rows()
        if not items or not rows:
            return

        # If any selected item is not a favorite -> add all. Otherwise remove all.
        want_add = any(not self._is_favorite_path(it.path) for it in items)

        changed_rows: List[int] = []
        for r, it in zip(rows, items):
            try:
                np = norm_path(it.path)
                if want_add:
                    self._favorites_map[np] = canon_path(it.path)
                    self._favorites_norm.add(np)
                else:
                    self._favorites_map.pop(np, None)
                    self._favorites_norm.discard(np)
                changed_rows.append(r)
            except Exception:
                continue

        self._save_favorites_map()
        self._refresh_favorites_views(rows=changed_rows)




    
    def _norm_path(self, p: str) -> str:
        try:
            return str(Path(p).expanduser().resolve()).lower()
        except Exception:
            return (p or "").strip().lower()

    def _is_path_loaded_elsewhere(self, file_path: str) -> Tuple[bool, str]:
        """Best-effort check: is this file currently loaded somewhere in the host app?"""
        target = self._norm_path(file_path)
        try:
            w = self.window()
        except Exception:
            w = None
        if not w:
            return (False, "")

        # Prefer explicit host helpers if present.
        for fn_name in ("is_path_loaded", "is_file_loaded", "is_media_loaded", "is_loaded"):
            try:
                fn = getattr(w, fn_name, None)
                if callable(fn):
                    r = fn(file_path)
                    if isinstance(r, tuple) and len(r) >= 1:
                        loaded = bool(r[0])
                        where = str(r[1]) if loaded and len(r) > 1 else ""
                        if loaded:
                            return (True, where)
                    if r is True:
                        return (True, "the app")
            except Exception:
                pass

        # Common MainWindow attributes.
        for attr in ("current_media_path", "current_path", "loaded_path", "active_path"):
            try:
                v = getattr(w, attr, None)
                if v and self._norm_path(str(v)) == target:
                    return (True, "the app")
            except Exception:
                pass

        # Built-in player (FrameVision convention: self.window().video).
        try:
            video = getattr(w, "video", None)
            if video is not None:
                for attr in ("path", "current_path", "file_path", "source_path", "current_file"):
                    try:
                        v = getattr(video, attr, None)
                        if v and self._norm_path(str(v)) == target:
                            return (True, "the built-in player")
                    except Exception:
                        pass
                player = getattr(video, "player", None)
                if player is not None:
                    for attr in ("path", "current_path", "file_path", "source_path", "current_file"):
                        try:
                            v = getattr(player, attr, None)
                            if v and self._norm_path(str(v)) == target:
                                return (True, "the built-in player")
                        except Exception:
                            pass
        except Exception:
            pass

        return (False, "")

    def _warn_in_use(self, file_path: str, where: str = "") -> None:
        where = where or "the app"
        QtWidgets.QMessageBox.information(
            self,
            "Media Explorer",
            "That file appears to be loaded in {where}.\n\n"
            "Please unload/close it first, then try again.\n\n"
            "File:\n{fp}".format(where=where, fp=file_path),
        )

    def _refresh_item_from_disk(self, item: MediaItem) -> None:
        """Re-read quick metadata for a single item after rename."""
        try:
            p = Path(item.path).expanduser()
            if not p.exists() or not p.is_file():
                return
            st = p.stat()
            item.name = p.name
            item.ext = p.suffix.lower()
            item.media_type = detect_media_type(item.ext)
            item.found_in = str(p.parent)
            item.size_bytes = st.st_size
            item.mtime = st.st_mtime

            # Reset type-specific fields
            item.duration_sec = None
            item.fps = None
            item.bitrate_bps = None
            item.codec = None
            item.width = None
            item.height = None
            item.sample_rate = None
            item.channels = None

            if item.media_type == "image":
                try:
                    reader = QtGui.QImageReader(str(p))
                    sz = reader.size()
                    if sz.isValid():
                        item.width = sz.width()
                        item.height = sz.height()
                    fmt = bytes(reader.format()).decode("ascii", errors="ignore").strip().upper()
                    if fmt:
                        item.codec = fmt
                except Exception:
                    pass
                return

            probe = self.fftools.probe(str(p)) if self.fftools.is_ready() else None
            if not probe:
                return

            fmt = probe.get("format") or {}
            streams = probe.get("streams") or []

            item.duration_sec = safe_float(fmt.get("duration"))
            item.bitrate_bps = safe_int(fmt.get("bit_rate"))

            if item.media_type == "video":
                v = next((s for s in streams if s.get("codec_type") == "video"), None)
                if v:
                    item.codec = v.get("codec_name") or item.codec
                    item.width = safe_int(v.get("width"))
                    item.height = safe_int(v.get("height"))
                    item.fps = parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate"))
                    sb = safe_int(v.get("bit_rate"))
                    if sb and sb > 0:
                        item.bitrate_bps = sb
            elif item.media_type == "audio":
                a = next((s for s in streams if s.get("codec_type") == "audio"), None)
                if a:
                    item.codec = a.get("codec_name") or item.codec
                    item.sample_rate = safe_int(a.get("sample_rate"))
                    item.channels = safe_int(a.get("channels"))
                    sb = safe_int(a.get("bit_rate"))
                    if sb and sb > 0:
                        item.bitrate_bps = sb
        except Exception:
            return

    def _rename_selected(self) -> None:
        rows = self._selected_source_rows()
        items = self._selected_items()
        if len(items) != 1 or len(rows) != 1:
            QtWidgets.QMessageBox.information(self, "Media Explorer", "Please select exactly one file to rename.")
            return

        it = items[0]
        loaded, where = self._is_path_loaded_elsewhere(it.path)
        if loaded:
            self._warn_in_use(it.path, where)
            return

        old_path = Path(it.path)
        if not old_path.exists():
            QtWidgets.QMessageBox.warning(self, "Media Explorer", "That file no longer exists on disk.")
            return

        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Rename file",
            "New filename:",
            QtWidgets.QLineEdit.EchoMode.Normal,
            it.name,
        )
        if not ok:
            return
        new_name = (new_name or "").strip()
        if not new_name:
            return

        bad = set('\\/:*?"<>|')
        if any(ch in bad for ch in new_name):
            QtWidgets.QMessageBox.warning(self, "Media Explorer", "That filename contains invalid characters.")
            return

        # If user omitted extension, keep original.
        if "." not in new_name:
            new_name = new_name + it.ext

        new_path = old_path.with_name(new_name)
        if self._norm_path(str(new_path)) == self._norm_path(str(old_path)):
            return
        if new_path.exists():
            QtWidgets.QMessageBox.warning(self, "Media Explorer", "A file with that name already exists.")
            return

        try:
            old_path.rename(new_path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Media Explorer", f"Rename failed: {e}")
            return

        # Update model item in place
        it.path = str(new_path)
        it.name = new_path.name
        it.ext = new_path.suffix.lower()
        it.media_type = detect_media_type(it.ext)
        it.found_in = str(new_path.parent)
        self._refresh_item_from_disk(it)

        r = rows[0]
        try:
            a = self.model.index(r, 0)
            b = self.model.index(r, self.model.columnCount() - 1)
            self.model.dataChanged.emit(a, b)
        except Exception:
            pass

        self._update_counts()
        self._update_details()

    def _delete_selected_from_drive(self) -> None:
        rows = self._selected_source_rows()
        items = self._selected_items()
        if not items or not rows:
            return

        # Block if any selected file is loaded somewhere.
        for it in items:
            loaded, where = self._is_path_loaded_elsewhere(it.path)
            if loaded:
                self._warn_in_use(it.path, where)
                return

        count = len(items)
        preview = ", ".join([it.name for it in items[:3]])
        if count > 3:
            preview += f", … (+{count-3} more)"

        msg = (
            f"Move {count} file(s) to the Recycle Bin?\n\n"
            "This will move them to the Windows Recycle Bin (Trash).\n\n"
            f"Selected: {preview}"
        )
        r = QtWidgets.QMessageBox.question(
            self,
            "Move to Recycle Bin",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if r != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        deleted_rows: List[int] = []
        errors: List[str] = []

        for row, it in zip(rows, items):
            try:
                p = Path(it.path)
                if not p.exists():
                    deleted_rows.append(row)
                else:
                    ok, err = move_to_recycle_bin(str(p))
                    if ok:
                        deleted_rows.append(row)
                    else:
                        raise RuntimeError(err)
            except Exception as e:
                errors.append(f"{it.name}: {e}")

        if deleted_rows:
            try:
                self.model.remove_rows(deleted_rows)
            except Exception:
                pass

        # Clear selection & update UI
        try:
            self.table.clearSelection()
        except Exception:
            pass
        self.lbl_file_name.setText("Select an item to see details.")
        self.lbl_type_badge.setVisible(False)
        self._val_full_path.setText("")
        self._val_size_modified.setText("")
        self._val_res_quality.setText("")
        self._set_metadata_message("Select a file to load metadata JSON.")
        self._update_counts()

        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Media Explorer",
                "Some files could not be moved to the Recycle Bin:\n\n" + "\n".join(errors[:10]),
            )


    def _set_metadata_message(self, msg: str) -> None:
        try:
            self.lbl_meta_status.setText(msg or "")
            self.lbl_meta_status.setVisible(True)
            self.smart_table.setVisible(False)
            self.section_everything.setVisible(False)
            try:
                self.smart_table.setRowCount(0)
            except Exception:
                pass
            try:
                self.json_tree.clear()
            except Exception:
                pass
        except Exception:
            pass


    def _load_sidecar_json(self, media_path: str) -> Optional[Any]:
        """Find a metadata JSON sidecar near the media file and parse it.

        Matching rules (in priority order):
          1) <basename>.json
          2) <basename>*.json   (e.g. <basename>_input_params.json)
        If multiple matches exist, we pick the best candidate by heuristic scoring.
        """
        try:
            p = Path(media_path)
            if not p.exists():
                return None

            folder = p.parent
            base = p.stem

            candidates = []

            # 1) Exact match: <basename>.json
            exact = folder / f"{base}.json"
            if exact.exists() and exact.is_file():
                candidates.append(exact)

            # 2) Prefix match: <basename>*.json (covers _input_params.json etc.)
            try:
                for f in folder.glob(f"{base}*.json"):
                    if f.is_file():
                        candidates.append(f)
            except Exception:
                pass

            # De-dup while preserving order
            seen = set()
            uniq = []
            for c in candidates:
                cp = str(c.resolve()) if hasattr(c, "resolve") else str(c)
                if cp in seen:
                    continue
                seen.add(cp)
                uniq.append(c)

            if not uniq:
                return None

            def score(f: Path) -> tuple:
                name = f.name.lower()
                s = 0
                if name == f"{base.lower()}.json":
                    s += 1000
                if name.startswith(base.lower() + "_"):
                    s += 200
                if "input_params" in name:
                    s += 120
                if "params" in name:
                    s += 60
                if "metadata" in name:
                    s += 40
                if "settings" in name:
                    s += 20
                # Prefer shorter filenames (less noisy suffixes)
                s -= min(len(name), 300) / 10.0
                # Prefer newest if still tied
                try:
                    mtime = f.stat().st_mtime
                except Exception:
                    mtime = 0
                return (s, mtime)

            best = sorted(uniq, key=score, reverse=True)[0]

            raw = best.read_text(encoding="utf-8", errors="ignore")
            if not raw.strip():
                return None
            return json.loads(raw)
        except Exception:
            return None

    def _populate_metadata_json(self, data: Any) -> None:
        """Populate smart fields + a collapsible tree for the rest."""
        # Smart fields
        rows = self._extract_smart_fields(data)

        if rows:
            self.smart_table.setVisible(True)
            self.smart_table.setRowCount(len(rows))
            self.smart_table.setColumnCount(2)
            self.smart_table.setColumnWidth(0, 160)

            for r, (k, v) in enumerate(rows):
                key_item = QtWidgets.QTableWidgetItem(str(k))
                f = key_item.font()
                f.setBold(True)
                key_item.setFont(f)
                self.smart_table.setItem(r, 0, key_item)

                # value cell
                if isinstance(v, str) and len(v) > 200:
                    w = SmartTextValue(v, limit=200, parent=self.smart_table)
                    self.smart_table.setCellWidget(r, 1, w)
                    try:
                        w.btn_expand.clicked.connect(lambda *_: self.smart_table.resizeRowsToContents())
                    except Exception:
                        pass
                else:
                    val = self._format_smart_value(v)
                    lab = QtWidgets.QLabel(val, self.smart_table)
                    lab.setWordWrap(True)
                    lab.setTextFormat(QtCore.Qt.TextFormat.PlainText)
                    lab.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
                    self.smart_table.setCellWidget(r, 1, lab)

            try:
                self.smart_table.resizeRowsToContents()
            except Exception:
                pass
        else:
            self.smart_table.setVisible(False)
            try:
                self.smart_table.setRowCount(0)
            except Exception:
                pass

        # Everything else (collapsed)
        try:
            self.json_tree.clear()
            self._build_json_tree(self.json_tree, data)
            try:
                self.json_tree.collapseAll()
            except Exception:
                pass
        except Exception:
            pass

        self.lbl_meta_status.setVisible(False)
        self.section_everything.setVisible(True)
        try:
            self.section_everything.btn.setChecked(False)
        except Exception:
            self.section_everything.set_expanded(False)  # collapse

    def _format_smart_value(self, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            if isinstance(v, float):
                s = f"{v:.6f}".rstrip("0").rstrip(".")
                return s
            return str(v)
        if isinstance(v, str):
            return v
        if isinstance(v, list):
            if not v:
                return "0 items"
            if all(isinstance(x, (str, int, float, bool)) for x in v) and len(v) <= 12:
                return ", ".join([str(x) for x in v])
            return f"{len(v)} items"
        if isinstance(v, dict):
            for k in ("name", "model", "id", "title", "checkpoint"):
                if k in v and isinstance(v.get(k), (str, int, float)):
                    return str(v.get(k))
            return f"{len(v)} fields"
        try:
            return str(v)
        except Exception:
            return ""

    def _extract_smart_fields(self, data: Any) -> List[Tuple[str, Any]]:
        """Find common AI metadata fields anywhere in the JSON."""
        def find_any(keys: List[str]) -> Optional[Any]:
            return self._find_first_by_keys(data, set([k.lower() for k in keys]))

        out: List[Tuple[str, Any]] = []

        prompt = find_any(["prompt", "positive_prompt", "positive", "text", "caption", "description"])
        neg = find_any(["negative_prompt", "neg_prompt", "negative", "uc", "uncond_prompt"])

        model = find_any(["model", "checkpoint", "ckpt", "base_model", "model_name", "sd_model_checkpoint"])
        lora = find_any(["lora", "loras", "lora_name", "lora_names", "lora_models", "lora_list"])

        seed = find_any(["seed", "random_seed"])
        steps = find_any(["steps", "num_steps", "sampling_steps"])
        cfg = find_any(["cfg", "cfg_scale", "guidance_scale"])
        sampler = find_any(["sampler", "sampler_name", "scheduler", "sampling_method"])

        # Resolution: prefer explicit width/height if available
        width = self._find_first_by_keys(data, {"width", "image_width"})
        height = self._find_first_by_keys(data, {"height", "image_height"})
        resolution = None
        try:
            wi = safe_int(width) if width is not None else None
            hi = safe_int(height) if height is not None else None
            if wi and hi:
                resolution = f"{wi}×{hi}"
        except Exception:
            resolution = None
        if resolution is None:
            resolution = find_any(["resolution", "size", "image_size", "video_size"])

        fps = find_any(["fps", "frame_rate"])
        duration = find_any(["duration", "duration_sec", "seconds", "length"])

        # Add in requested order
        if isinstance(prompt, str) and prompt.strip():
            out.append(("Prompt", prompt))
        if isinstance(neg, str) and neg.strip():
            out.append(("Negative prompt", neg))

        if model not in (None, "", {}, []):
            out.append(("Model / checkpoint", model))
        if lora not in (None, "", {}, []):
            out.append(("LoRA", lora))

        if seed not in (None, "", {}, []):
            out.append(("Seed", seed))
        if steps not in (None, "", {}, []):
            out.append(("Steps", steps))
        if cfg not in (None, "", {}, []):
            out.append(("CFG", cfg))
        if sampler not in (None, "", {}, []):
            out.append(("Sampler", sampler))

        if resolution not in (None, "", {}, []):
            out.append(("Resolution", resolution))
        if fps not in (None, "", {}, []):
            out.append(("FPS", fps))
        if duration not in (None, "", {}, []):
            out.append(("Duration", duration))

        return out

    def _find_first_by_keys(self, obj: Any, keys_lc: set[str]) -> Optional[Any]:
        try:
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if str(k).lower() in keys_lc and v not in (None, ""):
                        return v
                for v in obj.values():
                    r = self._find_first_by_keys(v, keys_lc)
                    if r not in (None, ""):
                        return r
            elif isinstance(obj, list):
                for v in obj:
                    r = self._find_first_by_keys(v, keys_lc)
                    if r not in (None, ""):
                        return r
        except Exception:
            return None
        return None

    def _build_json_tree(self, tree: QtWidgets.QTreeWidget, data: Any) -> None:
        """Build a structured tree view: objects are expandable groups; arrays show 'N items'."""
        tree.setColumnCount(2)
        tree.setHeaderHidden(True)

        # Hard safety caps to keep UI responsive
        max_depth = 12
        max_children = 400

        def add_item(parent: Optional[QtWidgets.QTreeWidgetItem], key: str, value: Any, depth: int) -> None:
            if depth > max_depth:
                it = QtWidgets.QTreeWidgetItem([str(key), "…"])
                if parent is None:
                    tree.addTopLevelItem(it)
                else:
                    parent.addChild(it)
                return

            if isinstance(value, dict):
                label = f"{len(value)} fields"
                node = QtWidgets.QTreeWidgetItem([str(key), label])
                if parent is None:
                    tree.addTopLevelItem(node)
                else:
                    parent.addChild(node)
                n = 0
                for k2, v2 in value.items():
                    n += 1
                    if n > max_children:
                        node.addChild(QtWidgets.QTreeWidgetItem(["…", f"truncated ({len(value)} fields)"]))
                        break
                    add_item(node, str(k2), v2, depth + 1)
                return

            if isinstance(value, list):
                label = f"{len(value)} items"
                node = QtWidgets.QTreeWidgetItem([str(key), label])
                if parent is None:
                    tree.addTopLevelItem(node)
                else:
                    parent.addChild(node)
                n = 0
                for i, v2 in enumerate(value):
                    n += 1
                    if n > max_children:
                        node.addChild(QtWidgets.QTreeWidgetItem(["…", f"truncated ({len(value)} items)"]))
                        break
                    add_item(node, f"[{i}]", v2, depth + 1)
                return

            node = QtWidgets.QTreeWidgetItem([str(key), ""])
            if parent is None:
                tree.addTopLevelItem(node)
            else:
                parent.addChild(node)

            if isinstance(value, str) and len(value) > 200:
                w = SmartTextValue(value, limit=200, parent=tree)
                tree.setItemWidget(node, 1, w)
                node.setText(1, "")
            else:
                node.setText(1, self._format_smart_value(value))

        if isinstance(data, dict):
            for k, v in data.items():
                add_item(None, str(k), v, 0)
        elif isinstance(data, list):
            add_item(None, "root", data, 0)
        else:
            add_item(None, "value", data, 0)

    def _update_details(self) -> None:
        items = self._selected_items()
        self.btn_open_folder.setEnabled(bool(items))
        self.btn_copy_path.setEnabled(bool(items))

        if not items:
            # Empty state
            self.lbl_file_name.setText("Select an item to see details.")
            self.lbl_type_badge.setVisible(False)
            self._val_full_path.setText("")
            self._val_size_modified.setText("")
            self._val_res_quality.setText("")
            self._set_metadata_message("Select a file to load metadata JSON.")
            return

        it = items[0]

        # Header
        self.lbl_file_name.setText(it.name or "")
        badge = (it.media_type or "other").lower()
        if badge in ("image", "video", "audio"):
            self.lbl_type_badge.setText(badge)
            self.lbl_type_badge.setVisible(True)
        else:
            self.lbl_type_badge.setVisible(False)

        # Kept info rows
        self._val_full_path.setText(it.path or "")

        size_s = human_size(it.size_bytes) if it.size_bytes is not None else ""
        mod_s = it.mtime_dt.strftime("%y-%m%d %H:%M:%S") if it.mtime_dt else ""
        if size_s and mod_s:
            self._val_size_modified.setText(f"{size_s} - {mod_s}")
        else:
            self._val_size_modified.setText(size_s or mod_s or "")

        res_s = it.resolution_str()
        q_s = it.quality_str()
        if res_s and q_s:
            self._val_res_quality.setText(f"{res_s} • {q_s}")
        else:
            self._val_res_quality.setText(res_s or q_s or "")

        # Metadata JSON sidecar (same folder, same base name)
        data = self._load_sidecar_json(it.path)
        if data is None:
            self._set_metadata_message("Couldn’t find/read metadata JSON (invalid format)")
            return

        self._populate_metadata_json(data)


    def _open_selected_folder(self) -> None:
        items = self._selected_items()
        if not items:
            return
        folder = Path(items[0].path).parent
        if folder.exists():
            os.startfile(str(folder))  # Windows

    def _copy_selected_path(self) -> None:
        items = self._selected_items()
        if not items:
            return
        QtWidgets.QApplication.clipboard().setText(items[0].path)

    def _on_double_click(self, idx: QtCore.QModelIndex) -> None:
        src = self.proxy.mapToSource(idx)
        item = self.model.item_at(src.row())
        if not item:
            return
        # Double-click opens inside FrameVision's built-in player (not external apps).
        if not self._open_in_player(item):
            # Fallback: external open if we can't locate the in-app player.
            self._open_external(item)

    def _open_in_player(self, item: MediaItem) -> bool:
        """Try to open the item inside FrameVision's player."""
        p = str(item.path or "").strip()
        if not p:
            return False

        # 1) Emit a signal for apps that want to handle it explicitly.
        try:
            self.openInPlayerRequested.emit(p)
        except Exception:
            pass

        # 2) Best-effort direct open against the host MainWindow (FrameVision).
        try:
            w = self.window()
            if w is None:
                return False
            video = getattr(w, "video", None)
            if video is None:
                return False
            if not hasattr(video, "open"):
                return False
            video.open(Path(p))
            return True
        except Exception:
            return False

    def _open_external(self, item: MediaItem) -> None:
        """Open using Windows default associated application."""
        p = str(item.path or "").strip()
        if not p:
            return
        try:
            os.startfile(p)  # Windows default
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Media Explorer", "Could not open the selected file.")


    # ---------- Copy / Cut / Paste (in-app clipboard) ----------

    def _clip_has_items(self) -> bool:
        try:
            return bool(self._clip_paths)
        except Exception:
            return False

    def _current_folder_exists(self) -> bool:
        try:
            folder = self.ed_folder.text().strip()
            if not folder:
                return False
            p = Path(folder).expanduser()
            return p.exists() and p.is_dir()
        except Exception:
            return False

    def _current_folder_canon(self) -> str:
        folder = self.ed_folder.text().strip()
        try:
            return str(Path(folder).expanduser().resolve())
        except Exception:
            return str(Path(folder).expanduser())

    def _clipboard_copy_selected(self) -> None:
        items = self._selected_items()
        if not items:
            return
        paths = [str(x.path) for x in items if getattr(x, "path", None)]
        paths = [p for p in paths if p]
        if not paths:
            return
        self._clip_mode = "copy"
        self._clip_paths = paths
        try:
            QtWidgets.QApplication.clipboard().setText("\n".join(paths))
        except Exception:
            pass
        try:
            self.lbl_status.setText(f"Copied {len(paths)} item(s).")
        except Exception:
            pass

    def _clipboard_cut_selected(self) -> None:
        items = self._selected_items()
        if not items:
            return
        paths = [str(x.path) for x in items if getattr(x, "path", None)]
        paths = [p for p in paths if p]
        if not paths:
            return

        # Block if any selected file is loaded somewhere.
        for p in paths:
            loaded, where = self._is_path_loaded_elsewhere(p)
            if loaded:
                self._warn_in_use(p, where)
                return

        self._clip_mode = "cut"
        self._clip_paths = paths
        try:
            QtWidgets.QApplication.clipboard().setText("\n".join(paths))
        except Exception:
            pass
        try:
            self.lbl_status.setText(f"Cut {len(paths)} item(s).")
        except Exception:
            pass

    def _clipboard_paste(self) -> None:
        if not self._clip_has_items():
            return

        folder = self.ed_folder.text().strip()
        if not folder:
            QtWidgets.QMessageBox.information(self, "Media Explorer", "Please select a destination folder first.")
            return

        dest_dir = Path(folder).expanduser()
        if not dest_dir.exists() or not dest_dir.is_dir():
            QtWidgets.QMessageBox.warning(self, "Media Explorer", "Destination folder does not exist.")
            return

        dest_canon = self._current_folder_canon()

        # If a scan is running, stop it so we can scan the destination folder cleanly.
        try:
            if self._scanner and self._scanner.isRunning():
                self._stop_scan()
        except Exception:
            pass

        # Auto-scan first if user changed folders since last scan.
        need_scan = True
        try:
            if self._last_folder and self._norm_path(dest_canon) == self._norm_path(self._last_folder):
                need_scan = False
        except Exception:
            need_scan = True

        pending = {
            "stage": "after_scan" if need_scan else "direct",
            "dest_folder": dest_canon,
            "mode": self._clip_mode or "copy",
            "paths": list(self._clip_paths or []),
        }

        if need_scan:
            self._pending_paste = pending
            try:
                self.lbl_status.setText("Scanning destination before paste…")
            except Exception:
                pass
            self.rescan()
            return

        # Direct paste (already in the right folder)
        self._do_paste_pending(pending)

    def _do_paste_pending(self, pending: Dict[str, Any]) -> None:
        """Perform a pending paste request (optionally after a destination scan)."""
        if not pending:
            return

        dest_folder = (pending.get("dest_folder") or "").strip()
        if not dest_folder:
            return

        # If the user switched folders again while scanning, abort (safer than pasting into the wrong place).
        try:
            cur = self._current_folder_canon()
            if self._norm_path(cur) != self._norm_path(dest_folder):
                QtWidgets.QMessageBox.information(
                    self,
                    "Media Explorer",
                    "Paste cancelled because the destination folder changed while scanning.",
                )
                return
        except Exception:
            pass

        mode = (pending.get("mode") or "copy").lower().strip()
        if mode not in ("copy", "cut"):
            mode = "copy"

        src_paths = [str(x) for x in (pending.get("paths") or []) if str(x).strip()]
        if not src_paths:
            return

        created, moved_all_ok = self._paste_files_to_folder(src_paths, dest_folder, mode)

        # If it was a cut/move and everything succeeded, clear clipboard (Windows-like behavior).
        if mode == "cut" and moved_all_ok:
            try:
                self._clip_mode = None
                self._clip_paths = []
            except Exception:
                pass

        # Add newly created files into the current view (cheap, avoids a full rescan).
        try:
            if created:
                new_items: List[MediaItem] = []
                for fp in created:
                    try:
                        mi = self._build_item_for_path(Path(fp))
                        if mi:
                            # Respect current media-type toggles
                            if mi.media_type == "image" and not self.cb_images.isChecked():
                                continue
                            if mi.media_type == "video" and not self.cb_videos.isChecked():
                                continue
                            if mi.media_type == "audio" and not self.cb_audio.isChecked():
                                continue
                            new_items.append(mi)
                    except Exception:
                        continue
                if new_items:
                    self.model.add_items(new_items)
                    self._update_counts()
        except Exception:
            pass

    def _unique_dest_path(self, dest_dir: Path, name: str) -> Path:
        """Return a destination path that does not exist yet (Windows-like '(1)' suffix)."""
        base = Path(name).stem
        ext = Path(name).suffix
        cand = dest_dir / (base + ext)
        if not cand.exists():
            return cand
        i = 1
        while i < 10000:
            cand = dest_dir / f"{base} ({i}){ext}"
            if not cand.exists():
                return cand
            i += 1
        # fallback
        return dest_dir / f"{base} ({int(datetime.now().timestamp())}){ext}"

    def _paste_files_to_folder(self, src_paths: List[str], dest_folder: str, mode: str) -> Tuple[List[str], bool]:
        """Copy/move files to dest_folder. Returns (created_paths, all_ok)."""
        dest_dir = Path(dest_folder)
        created: List[str] = []
        errors: List[str] = []

        # Normalize destination
        try:
            dest_dir = dest_dir.expanduser().resolve()
        except Exception:
            dest_dir = dest_dir.expanduser()

        for sp in src_paths:
            try:
                src = Path(sp).expanduser()
                if not src.exists() or not src.is_file():
                    errors.append(f"Missing: {sp}")
                    continue

                # Prevent no-op moves (cut within same folder)
                try:
                    if mode == "cut":
                        if src.parent.resolve() == dest_dir:
                            continue
                except Exception:
                    pass

                dst = self._unique_dest_path(dest_dir, src.name)

                if mode == "copy":
                    shutil.copy2(str(src), str(dst))
                else:
                    # cut/move
                    shutil.move(str(src), str(dst))
                    # Move favorites along with the file (if it was favorited).
                    try:
                        old_n = norm_path(str(src))
                        if old_n in self._favorites_norm:
                            try:
                                del self._favorites_norm[old_n]
                            except Exception:
                                pass
                            try:
                                del self._favorites_map[old_n]
                            except Exception:
                                pass
                            new_c = str(dst)
                            new_n = norm_path(new_c)
                            self._favorites_norm[new_n] = True
                            self._favorites_map[new_n] = new_c
                            self._save_favorites_map()
                    except Exception:
                        pass

                created.append(str(dst))
            except Exception as e:
                errors.append(f"{sp}: {e}")

        # Feedback
        try:
            if created and not errors:
                verb = "Pasted" if mode == "copy" else "Moved"
                self.lbl_status.setText(f"{verb} {len(created)} item(s).")
            elif created and errors:
                verb = "Pasted" if mode == "copy" else "Moved"
                self.lbl_status.setText(f"{verb} {len(created)} item(s) with {len(errors)} error(s).")
                QtWidgets.QMessageBox.warning(self, "Media Explorer", "Some files could not be pasted:\n\n" + "\n".join(errors[:10]))
            elif errors and not created:
                QtWidgets.QMessageBox.warning(self, "Media Explorer", "Paste failed:\n\n" + "\n".join(errors[:10]))
        except Exception:
            pass

        all_ok = (len(errors) == 0)
        return created, all_ok

    def _build_item_for_path(self, p: Path) -> Optional[MediaItem]:
        """Build a MediaItem for a file path (best-effort metadata)."""
        try:
            if not p.exists() or not p.is_file():
                return None
            item = MediaItem()
            item.path = str(p)
            item.name = p.name
            item.ext = p.suffix.lower()
            item.media_type = detect_media_type(item.ext)
            item.found_in = str(p.parent)

            try:
                st = p.stat()
                item.size_bytes = int(st.st_size)
                item.mtime = float(st.st_mtime)
            except Exception:
                item.size_bytes = None
                item.mtime = None

            # Reset type-specific fields
            item.duration_sec = None
            item.fps = None
            item.bitrate_bps = None
            item.codec = None
            item.width = None
            item.height = None
            item.sample_rate = None
            item.channels = None

            if item.media_type == "image":
                try:
                    reader = QtGui.QImageReader(str(p))
                    sz = reader.size()
                    if sz.isValid():
                        item.width = sz.width()
                        item.height = sz.height()
                    fmt = bytes(reader.format()).decode("ascii", errors="ignore").strip().upper()
                    if fmt:
                        item.codec = fmt
                except Exception:
                    pass
                return item

            probe = self.fftools.probe(str(p)) if self.fftools.is_ready() else None
            if not probe:
                return item

            fmt = probe.get("format") or {}
            streams = probe.get("streams") or []

            item.duration_sec = safe_float(fmt.get("duration"))
            item.bitrate_bps = safe_int(fmt.get("bit_rate"))

            if item.media_type == "video":
                v = next((s for s in streams if s.get("codec_type") == "video"), None)
                if v:
                    item.codec = v.get("codec_name") or item.codec
                    item.width = safe_int(v.get("width"))
                    item.height = safe_int(v.get("height"))
                    item.fps = parse_fps(v.get("avg_frame_rate") or v.get("r_frame_rate"))
                    sb = safe_int(v.get("bit_rate"))
                    if sb and sb > 0:
                        item.bitrate_bps = sb
            elif item.media_type == "audio":
                a = next((s for s in streams if s.get("codec_type") == "audio"), None)
                if a:
                    item.codec = a.get("codec_name") or item.codec
                    item.sample_rate = safe_int(a.get("sample_rate"))
                    item.channels = safe_int(a.get("channels"))
                    sb = safe_int(a.get("bit_rate"))
                    if sb and sb > 0:
                        item.bitrate_bps = sb

            return item
        except Exception:
            return None
    def _open_context_menu(self, pos: QtCore.QPoint) -> None:
        sel_items = self._selected_items()
        it = sel_items[0] if sel_items else None

        menu = QtWidgets.QMenu(self)

        # Favorites (only when a selection exists)
        if sel_items:
            want_add = any(not self._is_favorite_path(x.path) for x in sel_items)
            label = "★ Add to favorites" if want_add else "★ Remove from favorites"
            if len(sel_items) > 1:
                label += f" ({len(sel_items)})"
            act_toggle_fav = menu.addAction(label)
            act_toggle_fav.setToolTip("Toggle favorite for the selected file(s). (Shortcut: F)")
            try:
                act_toggle_fav.setShortcut(QtGui.QKeySequence("F"))
            except Exception:
                pass

            act_favs_only = menu.addAction("Show only favorites")
            act_favs_only.setCheckable(True)
            act_favs_only.setChecked(self.btn_favs_only.isChecked())
            act_favs_only.setToolTip("Filter the table to only show favorites.")

            menu.addSeparator()

        # Copy/Cut/Paste (in-app clipboard)
        act_copy_files = menu.addAction("Copy")
        act_copy_files.setToolTip("Copy selected file(s) into the Media Explorer clipboard.")
        act_copy_files.setEnabled(bool(sel_items))
        try:
            act_copy_files.setShortcut(QtGui.QKeySequence.Copy)
        except Exception:
            pass

        act_cut_files = menu.addAction("Cut")
        act_cut_files.setToolTip("Cut (move) selected file(s) into the Media Explorer clipboard.")
        act_cut_files.setEnabled(bool(sel_items))
        try:
            act_cut_files.setShortcut(QtGui.QKeySequence.Cut)
        except Exception:
            pass

        act_paste_files = menu.addAction("Paste")
        act_paste_files.setToolTip(
            "Paste file(s) from the Media Explorer clipboard into the folder in the Folder box.\n"
            "If you changed folders, Media Explorer will auto-scan first."
        )
        act_paste_files.setEnabled(self._clip_has_items() and self._current_folder_exists())
        try:
            act_paste_files.setShortcut(QtGui.QKeySequence.Paste)
        except Exception:
            pass

        menu.addSeparator()

        # Open / play actions
        act_open_player = menu.addAction("Open in player")
        act_open_player.setToolTip("Open the file inside FrameVision's built-in player.")
        act_open_player.setEnabled(bool(it))

        act_open_external = menu.addAction("Open in external Windows app")
        act_open_external.setToolTip("Open the file with the default app associated in Windows.")
        act_open_external.setEnabled(bool(it))

        act_play = None
        if it and it.media_type in ("video", "audio"):
            act_play = menu.addAction("Play with ffplay")
            act_play.setToolTip("Play using ffplay from presets/bin (autoexit).")

        menu.addSeparator()

        act_open_folder = menu.addAction("Open containing folder")
        act_open_folder.setToolTip("Open the folder where this file lives.")
        act_open_folder.setEnabled(bool(it))

        act_copy_path = menu.addAction("Copy full path")
        act_copy_path.setToolTip("Copy the full path to clipboard.")
        act_copy_path.setEnabled(bool(it))

        act_copy_name = menu.addAction("Copy filename")
        act_copy_name.setToolTip("Copy just the filename.")
        act_copy_name.setEnabled(bool(it))

        menu.addSeparator()

        act_rename = menu.addAction("Rename…")
        act_rename.setToolTip("Rename the selected file on disk.")
        act_rename.setEnabled(bool(it) and len(sel_items) == 1)

        act_delete = menu.addAction("Move to Recycle Bin…")
        act_delete.setToolTip("Move the selected file(s) to the Windows Recycle Bin (Trash).")
        act_delete.setEnabled(bool(sel_items))

        menu.addSeparator()

        act_rescan = menu.addAction("Rescan")
        act_rescan.setToolTip("Rescan the current folder with current filters.")

        chosen = menu.exec(self.table.viewport().mapToGlobal(pos))
        if not chosen:
            return

        # Favorites actions (if present)
        try:
            if "act_toggle_fav" in locals() and chosen == act_toggle_fav:
                self._toggle_favorite_selected()
                return
            if "act_favs_only" in locals() and chosen == act_favs_only:
                self.btn_favs_only.setChecked(act_favs_only.isChecked())
                return
        except Exception:
            pass

        # Copy/Cut/Paste actions
        if chosen == act_copy_files:
            self._clipboard_copy_selected()
            return
        if chosen == act_cut_files:
            self._clipboard_cut_selected()
            return
        if chosen == act_paste_files:
            self._clipboard_paste()
            return

        # Open/play actions
        if chosen == act_open_player and it:
            if not self._open_in_player(it):
                self._open_external(it)
        elif chosen == act_open_external and it:
            self._open_external(it)
        elif act_play and chosen == act_play and it:
            if not self.fftools.play_with_ffplay(it.path, is_audio=(it.media_type == "audio")):
                QtWidgets.QMessageBox.information(self, "Media Explorer", "ffplay not found. Expected in presets/bin.")
        elif chosen == act_open_folder and it:
            self._open_selected_folder()
        elif chosen == act_copy_path and it:
            self._copy_selected_path()
        elif chosen == act_copy_name and it:
            QtWidgets.QApplication.clipboard().setText(it.name)
        elif chosen == act_rename:
            self._rename_selected()
        elif chosen == act_delete:
            self._delete_selected_from_drive()
        elif chosen == act_rescan:
            self.rescan()

    # ---------- Misc ----------

    def _open_tree_view(self) -> None:
        """Ask for a folder (OK/Cancel) then show its subfolder tree."""
        try:
            start = self.ed_folder.text().strip() or self._last_folder or ""
        except Exception:
            start = ""

        pick = FolderPickDialog(self, title="Tree view")
        try:
            if start:
                pick.ed.setText(start)
        except Exception:
            pass

        if pick.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        folder = pick.folder()
        if not folder:
            return

        dlg = FolderTreeDialog(folder, self)
        dlg.exec()

    def _browse_folder(self) -> None:
        start = self._last_folder or self.ed_folder.text().strip() or str(Path.home())
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", start)
        if folder:
            self.ed_folder.setText(folder)

    def _update_counts(self, defer: bool = False) -> None:
        # Defer updating often during scan to reduce UI overhead.
        if defer:
            # simple timer coalesce
            if not hasattr(self, "_count_timer"):
                self._count_timer = QtCore.QTimer(self)
                self._count_timer.setSingleShot(True)
                self._count_timer.timeout.connect(self._update_counts)
            self._count_timer.start(120)
            return
        total = self.proxy.rowCount()
        self.lbl_count.setText(f"{total} items")


    def _ensure_default_sort_newest_first(self) -> None:
        """If no sort preference was saved yet, default to Modified desc (newest first)."""
        try:
            col = self._settings.value(f"{self._settings_group}/sort_col", None)
            order = self._settings.value(f"{self._settings_group}/sort_order", None)
            if col is None or order is None:
                self._settings.setValue(f"{self._settings_group}/sort_col", int(Column.MODIFIED))
                self._settings.setValue(f"{self._settings_group}/sort_order", int(QtCore.Qt.SortOrder.DescendingOrder))
                self.table.sortByColumn(Column.MODIFIED, QtCore.Qt.SortOrder.DescendingOrder)
        except Exception:
            pass


    # ---------- Column state + theming ----------

    def _apply_header_theme(self) -> None:
        'Ensure header stays readable on dark themes (some Windows styles keep it bright).'
        try:
            hh = self.table.horizontalHeader()
            # Use palette-driven colors so it adapts to any theme.
            hh.setStyleSheet(
                "QHeaderView::section{"
                "background-color: palette(button);"
                "color: palette(text);"
                "padding: 4px 6px;"
                "border: 1px solid palette(mid);"
                "}"
                "QHeaderView::section:pressed{"
                "background-color: palette(midlight);"
                "}"
                "QTableCornerButton::section{"
                "background-color: palette(button);"
                "border: 1px solid palette(mid);"
                "}"
            )
        except Exception:
            pass

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        # Re-apply header styling after theme/palette changes.
        try:
            if event.type() in (QtCore.QEvent.Type.PaletteChange, QtCore.QEvent.Type.StyleChange):
                self._apply_header_theme()
        except Exception:
            pass


    def _schedule_save_table_state(self) -> None:
        # Coalesce frequent header events.
        if not hasattr(self, "_state_timer"):
            self._state_timer = QtCore.QTimer(self)
            self._state_timer.setSingleShot(True)
            self._state_timer.timeout.connect(self._save_table_state_now)
        self._state_timer.start(250)

    def _schedule_save_row_height(self) -> None:
        # Coalesce frequent row resize events.
        if not hasattr(self, "_rowh_timer"):
            self._rowh_timer = QtCore.QTimer(self)
            self._rowh_timer.setSingleShot(True)
            self._rowh_timer.timeout.connect(self._save_row_height_now)
        self._rowh_timer.start(250)

    def _save_row_height_now(self) -> None:
        try:
            vh = self.table.verticalHeader()
            # If the user resized any row, treat the current default as the preference.
            # Qt doesn't always update defaultSectionSize automatically, so infer from the
            # current row if possible.
            row_h = int(vh.defaultSectionSize())
            # Ensure row 0 size wins if it was explicitly resized.
            try:
                if self.proxy.rowCount() > 0:
                    row_h = int(vh.sectionSize(0))
            except Exception:
                pass
            if row_h > 8:
                self._settings.setValue(f"{self._settings_group}/row_height", row_h)
                self._settings.sync()
        except Exception:
            pass

    def _save_table_state_now(self) -> None:
        try:
            hh = self.table.horizontalHeader()

            # 1) Save native header state as a QByteArray (most reliable).
            try:
                state = hh.saveState()
                if isinstance(state, QtCore.QByteArray) and not state.isEmpty():
                    self._settings.setValue(f"{self._settings_group}/header_state", state)
            except Exception:
                pass

            # 2) Save explicit order + sizes to beat quirky restore behavior on some styles.
            try:
                cols = self.model.columnCount()
                order = [int(hh.logicalIndex(v)) for v in range(cols)]
                sizes = {str(i): int(hh.sectionSize(i)) for i in range(cols)}
                self._settings.setValue(f"{self._settings_group}/header_order_json", json.dumps(order))
                self._settings.setValue(f"{self._settings_group}/header_sizes_json", json.dumps(sizes))
            except Exception:
                pass

            # 3) Save sort indicator.
            try:
                self._settings.setValue(f"{self._settings_group}/sort_col", int(hh.sortIndicatorSection()))
                self._settings.setValue(f"{self._settings_group}/sort_order", int(hh.sortIndicatorOrder()))
            except Exception:
                pass

            self._settings.sync()
        except Exception:
            pass

    def _restore_table_state(self) -> None:
        # Restore column order/sizes and sort indicator if previously saved.
        try:
            hh = self.table.horizontalHeader()
            blocker = QtCore.QSignalBlocker(hh)

            # If stretchLastSection is enabled, it can fight restored widths.
            # We'll disable during restore, then re-enable afterwards.
            try:
                had_stretch = bool(hh.stretchLastSection())
            except Exception:
                had_stretch = False
            try:
                hh.setStretchLastSection(False)
            except Exception:
                pass

            # 1) Native QByteArray state.
            restored_any = False
            try:
                st = self._settings.value(f"{self._settings_group}/header_state", None)
                if isinstance(st, QtCore.QByteArray) and not st.isEmpty():
                    hh.restoreState(st)
                    restored_any = True
            except Exception:
                pass

            # 2) Back-compat: legacy base64 key if present.
            if not restored_any:
                try:
                    b64 = self._settings.value(f"{self._settings_group}/header_state_b64", "")
                    if isinstance(b64, QtCore.QByteArray):
                        b64 = bytes(b64).decode("ascii", errors="ignore")
                    b64 = (b64 or "").strip()
                    if b64:
                        state = QtCore.QByteArray.fromBase64(b64.encode("ascii", errors="ignore"))
                        if not state.isEmpty():
                            hh.restoreState(state)
                            restored_any = True
                except Exception:
                    pass

            # 3) Explicit order + widths (wins last).
            try:
                cols = self.model.columnCount()
                order_s = self._settings.value(f"{self._settings_group}/header_order_json", "")
                sizes_s = self._settings.value(f"{self._settings_group}/header_sizes_json", "")
                if isinstance(order_s, QtCore.QByteArray):
                    order_s = bytes(order_s).decode("utf-8", errors="ignore")
                if isinstance(sizes_s, QtCore.QByteArray):
                    sizes_s = bytes(sizes_s).decode("utf-8", errors="ignore")

                order = []
                if (order_s or "").strip():
                    order = json.loads(order_s)

                sizes = {}
                if (sizes_s or "").strip():
                    sizes = json.loads(sizes_s)

                # Sanitize order: must include all columns exactly once.
                if isinstance(order, list) and order:
                    order = [int(x) for x in order if isinstance(x, (int, float, str))]
                    order = [int(x) for x in order if 0 <= int(x) < cols]
                    seen = set()
                    cleaned = []
                    for x in order:
                        if x not in seen:
                            cleaned.append(x)
                            seen.add(x)
                    for i in range(cols):
                        if i not in seen:
                            cleaned.append(i)
                    order = cleaned[:cols]

                    # Apply visual order using moveSection.
                    for target_visual, logical in enumerate(order):
                        cur_visual = hh.visualIndex(int(logical))
                        if cur_visual != target_visual and cur_visual >= 0:
                            hh.moveSection(cur_visual, target_visual)

                # Apply widths.
                if isinstance(sizes, dict) and sizes:
                    for i in range(cols):
                        w = sizes.get(str(i), None)
                        try:
                            w = int(w)
                        except Exception:
                            w = None
                        if w is not None and w >= 24:
                            hh.resizeSection(i, w)
            except Exception:
                pass

            # Row height
            try:
                row_h = self._settings.value(f"{self._settings_group}/row_height", None)
                if row_h is not None:
                    row_h = int(row_h)
                    if row_h >= 8:
                        vh = self.table.verticalHeader()
                        vh.setDefaultSectionSize(row_h)
            except Exception:
                pass

            del blocker  # explicit

            # Re-enable stretch last section if it was enabled originally.
            try:
                hh.setStretchLastSection(had_stretch)
            except Exception:
                pass

            # Sort indicator (proxy sorting is enabled; apply after restore)
            try:
                col = self._settings.value(f"{self._settings_group}/sort_col", None)
                order = self._settings.value(f"{self._settings_group}/sort_order", None)
                if col is not None and order is not None:
                    col = int(col)
                    order = QtCore.Qt.SortOrder(int(order))
                    if 0 <= col < len(Column.labels):
                        self.table.sortByColumn(col, order)
            except Exception:
                pass

        except Exception:
            pass

            # sort indicator (proxy sorting is enabled; apply after restore)
            col = self._settings.value(f"{self._settings_group}/sort_col", None)
            order = self._settings.value(f"{self._settings_group}/sort_order", None)
            if col is not None and order is not None:
                try:
                    col = int(col)
                    order = QtCore.Qt.SortOrder(int(order))
                    if 0 <= col < len(Column.labels):
                        self.table.sortByColumn(col, order)
                except Exception:
                    pass
        except Exception:
            pass


# -----------------------------
# Standalone test harness
# -----------------------------

def _standalone() -> None:
    app = QtWidgets.QApplication(sys.argv)
    w = QtWidgets.QMainWindow()
    w.setWindowTitle("Media Explorer (Tab) – standalone test")
    tab = MediaExplorerTab()
    w.setCentralWidget(tab)
    w.resize(1200, 720)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _standalone()
