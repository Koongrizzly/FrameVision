
"""
Reusable Batch Selection Dialog (PySide6)
----------------------------------------
This module extracts and generalizes the "Batch…" file picker dialog from the
trim tool so it can be reused by any tool that feeds a list of files into the
shared queue system.

Key features:
- Add individual files or whole folders (recursive).
- Filter by allowed extensions.
- De-duplicate entries.
- Remove selected items with Delete/Backspace or via button.
- "If output file already exists" options: Skip / Overwrite / Auto rename.
- Returns a list of selected files and the chosen conflict policy.

Usage (example):
    from helpers.batch import BatchSelectDialog
    dlg = BatchSelectDialog(parent, title="Batch trim")
    if dlg.exec() == dlg.Accepted:
        files = dlg.selected_files()
        conflict = dlg.conflict_mode()  # 'skip' | 'overwrite' | 'version'
        # enqueue your jobs here...

Convenience one-liner:
    files, conflict = BatchSelectDialog.pick(parent, title="Batch", exts=BatchSelectDialog.VIDEO_EXTS)
    if files is not None:
        # enqueue...
"""

from __future__ import annotations
import os
from typing import Iterable, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QGroupBox, QRadioButton, QDialogButtonBox, QPushButton, QHBoxLayout,
    QFileDialog
)


class _DeletableListWidget(QListWidget):
    """QListWidget that deletes selected items on Delete/Backspace."""
    def keyPressEvent(self, ev):  # type: ignore[override]
        try:
            if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                for it in list(self.selectedItems()):
                    row = self.row(it)
                    self.takeItem(row)
                return
        except Exception:
            pass
        super().keyPressEvent(ev)


class BatchSelectDialog(QDialog):
    IMAGE_EXTS = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.gif'}
    # Common extension presets
    VIDEO_EXTS = {".mp4",".mov",".mkv",".avi",".m4v",".webm",".ts",".m2ts",".wmv",".flv",".mpg",".mpeg",".3gp",".3g2",".ogv"}

    def __init__(
        self,
        parent=None,
        *,
        title: str = "Batch",
        exts: Optional[Sequence[str]] = None,
        start_dir: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._start_dir = start_dir or ""

        # Normalize extension set
        self._exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (exts or self.VIDEO_EXTS)}

        v = QVBoxLayout(self)

        # File list
        v.addWidget(QLabel("Files to process:"))
        self.files_list = _DeletableListWidget(self)
        self.files_list.setSelectionMode(QListWidget.ExtendedSelection)
        v.addWidget(self.files_list)

        # Conflict group
        grp = QGroupBox("If output file already exists:")
        gvl = QVBoxLayout(grp)
        self.rb_skip = QRadioButton("Skip existing")
        self.rb_over = QRadioButton("Overwrite")
        self.rb_ver  = QRadioButton("Auto rename (Versioned filename)")
        self.rb_ver.setChecked(True)
        gvl.addWidget(self.rb_skip)
        gvl.addWidget(self.rb_over)
        gvl.addWidget(self.rb_ver)
        v.addWidget(grp)

        # Add controls
        row = QHBoxLayout()
        self.btn_add_files  = QPushButton("Add files…")
        self.btn_add_folder = QPushButton("Add folder…")
        row.addWidget(self.btn_add_files)
        row.addWidget(self.btn_add_folder)
        v.addLayout(row)

        # Removal controls
        row2 = QHBoxLayout()
        self.btn_remove_sel = QPushButton("Remove selected")
        self.btn_clear      = QPushButton("Clear list")
        row2.addWidget(self.btn_remove_sel)
        row2.addWidget(self.btn_clear)
        v.addLayout(row2)

        # OK/Cancel
        self.box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        v.addWidget(self.box)

        # Wire up
        self.btn_add_files.clicked.connect(self._on_add_files)
        self.btn_add_folder.clicked.connect(self._on_add_folder)
        self.btn_remove_sel.clicked.connect(self._delete_selected)
        self.btn_clear.clicked.connect(self.files_list.clear)
        self.box.accepted.connect(self.accept)
        self.box.rejected.connect(self.reject)

    # ---------------------------- helpers ----------------------------
    def _delete_selected(self) -> None:
        try:
            for it in list(self.files_list.selectedItems()):
                row = self.files_list.row(it)
                self.files_list.takeItem(row)
        except Exception:
            pass

    def _file_filter(self) -> str:
        # e.g., "Video files (*.mp4 *.mov ...);;All files (*)"
        if not self._exts:
            return "All files (*)"
        star_exts = " ".join(f"*{e}" for e in sorted(self._exts))
        return f"Supported files ({star_exts});;All files (*)"

    def _ext_ok(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in self._exts

    def _add_files(self, paths: Iterable[str]) -> None:
        existing = {self.files_list.item(i).data(Qt.UserRole) for i in range(self.files_list.count())}
        for p in paths:
            if p and os.path.isfile(p) and self._ext_ok(p) and p not in existing:
                it = QListWidgetItem(os.path.basename(p))
                it.setToolTip(p)
                it.setData(Qt.UserRole, p)
                self.files_list.addItem(it)

    def _on_add_files(self) -> None:
        filt = self._file_filter()
        paths, _ = QFileDialog.getOpenFileNames(self, "Select files", self._start_dir, filt)
        self._add_files(paths)

    def _on_add_folder(self) -> None:
        dirp = QFileDialog.getExistingDirectory(self, "Select folder", self._start_dir)
        if not dirp:
            return
        to_add: List[str] = []
        for root, _dirs, files in os.walk(dirp):
            for f in files:
                fp = os.path.join(root, f)
                if self._ext_ok(fp):
                    to_add.append(fp)
        self._add_files(sorted(to_add))

    # --------------------------- results API -------------------------
    def selected_files(self) -> List[str]:
        return [self.files_list.item(i).data(Qt.UserRole) for i in range(self.files_list.count())]

    def conflict_mode(self) -> str:
        # 'skip' | 'overwrite' | 'version'
        if self.rb_over.isChecked():
            return "overwrite"
        if self.rb_skip.isChecked():
            return "skip"
        return "version"

    # ------------------------- convenience API ----------------------
    @classmethod
    def pick(
        cls,
        parent=None,
        *,
        title: str = "Batch",
        exts: Optional[Sequence[str]] = None,
        start_dir: str = "",
    ) -> Tuple[Optional[List[str]], Optional[str]]:
        """
        Show a modal dialog and return (files, conflict) if accepted,
        or (None, None) if cancelled.
        """
        dlg = cls(parent, title=title, exts=exts, start_dir=start_dir)
        if dlg.exec() != QDialog.Accepted:
            return None, None
        files = dlg.selected_files()
        if not files:
            return [], dlg.conflict_mode()
        return files, dlg.conflict_mode()
