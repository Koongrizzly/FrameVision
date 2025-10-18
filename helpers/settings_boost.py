
from __future__ import annotations
import os, shutil
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QGroupBox, QCheckBox, QLabel,
    QDialogButtonBox, QMessageBox
)

def _empty_dir_keep_folder(dir_path: Path) -> int:
    """
    Remove all contents of *dir_path* but keep the folder itself.
    Returns number of entries successfully removed (best-effort).
    """
    removed = 0
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        for name in os.listdir(dir_path):
            p = dir_path / name
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    except Exception:
        pass
    return removed

def _make_cache_dialog(parent) -> QDialog:
    dlg = QDialog(parent)
    dlg.setWindowTitle("Clear program cache")
    lay = QVBoxLayout(dlg); lay.setContentsMargins(12,12,12,12); lay.setSpacing(10)

    group = QGroupBox(dlg)
    v = QVBoxLayout(group); v.setContentsMargins(10,10,10,10); v.setSpacing(8)

    # Defaults: Python bytecode + Temp ON (per user preference)
    chk_pyc = QCheckBox("Python bytecode (__pycache__, *.pyc/*.pyo)"); chk_pyc.setChecked(True)
    chk_temp = QCheckBox("Delete temp files (output/_temp, work)"); chk_temp.setChecked(True)
    chk_logs = QCheckBox("Logs folder"); chk_logs.setChecked(False)
    chk_thumbs = QCheckBox("Thumbnails (Frames/Previews)"); chk_thumbs.setChecked(False)
    chk_qt = QCheckBox("Qt cache (QSettings, etc.)")
    chk_hf = QCheckBox("HuggingFace cache (models)")

    for cb in (chk_pyc, chk_temp, chk_logs, chk_thumbs, chk_qt, chk_hf):
        v.addWidget(cb)

    tip = QLabel("Tip: This does not remove your rendered outputs or presets.")
    tip.setWordWrap(True)
    tip.setStyleSheet("color:#888;")
    v.addWidget(tip)

    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)

    def update_ok():
        bb.button(QDialogButtonBox.Ok).setEnabled(any(cb.isChecked() for cb in (chk_pyc, chk_temp, chk_logs, chk_thumbs, chk_qt, chk_hf)))
    for cb in (chk_pyc, chk_temp, chk_logs, chk_thumbs, chk_qt, chk_hf):
        cb.toggled.connect(update_ok)
    update_ok()

    def on_accept():
        # lazy import so helpers path doesn't matter
        try:
            from helpers.cleanup_cache import run_cleanup
        except Exception:
            from cleanup_cache import run_cleanup  # fallback if pathing differs

        # Try with clean_temp parameter (new API); fall back gracefully if older version.
        try:
            res = run_cleanup(
                project_root=str(Path.cwd()),
                clean_pyc=chk_pyc.isChecked(),
                clean_logs=chk_logs.isChecked(),
                clean_thumbs=chk_thumbs.isChecked(),
                clean_qt_cache=chk_qt.isChecked(),
                clean_hf_cache=chk_hf.isChecked(),
                clean_temp=chk_temp.isChecked()
            )
        except TypeError:
            # Older cleanup implementation: call without clean_temp, then do a local temp wipe if requested.
            res = run_cleanup(
                project_root=str(Path.cwd()),
                clean_pyc=chk_pyc.isChecked(),
                clean_logs=chk_logs.isChecked(),
                clean_thumbs=chk_thumbs.isChecked(),
                clean_qt_cache=chk_qt.isChecked(),
                clean_hf_cache=chk_hf.isChecked()
            )
            if chk_temp.isChecked():
                base = Path.cwd()
                res["temp"] = 0
                for d in (base / "output" / "_temp", base / "work"):
                    res["temp"] += _empty_dir_keep_folder(d)

        msg_lines = [f"  {k}: {v}" for k, v in res.items()]
        msg = "Cache cleared:\n" + "\n".join(msg_lines)
        QMessageBox.information(parent or dlg, "Clear cache", msg)
        dlg.accept()

    bb.accepted.connect(on_accept)
    bb.rejected.connect(dlg.reject)

    lay.addWidget(group)
    lay.addWidget(bb)
    return dlg
