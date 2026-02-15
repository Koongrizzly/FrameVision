import os
import sys
import json
import traceback
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone

from PySide6.QtCore import QDateTime, QObject, Signal, QThread
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QHeaderView, QGroupBox, QGridLayout, QComboBox,
    QDateTimeEdit, QSpinBox, QCheckBox, QTextEdit, QMessageBox,
    QProgressBar
)

APP_NAME = "Media Date Changer"
STATE_DIR = Path.home() / ".media_date_changer"
STATE_DIR.mkdir(parents=True, exist_ok=True)
LAST_RUN_JSON = STATE_DIR / "last_run_backup.json"

MEDIA_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".heic", ".heif",
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg"
}

def is_windows() -> bool:
    return os.name == "nt"

def show_fatal_dialog(title: str, details: str):
    try:
        QMessageBox.critical(None, title, details)
    except Exception:
        print(details, file=sys.stderr)

def install_excepthook():
    def _hook(exc_type, exc, tb):
        msg = "".join(traceback.format_exception(exc_type, exc, tb))
        show_fatal_dialog(f"{APP_NAME} - Error", msg)
    sys.excepthook = _hook

# --- Windows creation time support (no external deps) ---
if is_windows():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    GENERIC_WRITE = 0x40000000
    FILE_SHARE_READ = 0x00000001
    FILE_SHARE_WRITE = 0x00000002
    FILE_SHARE_DELETE = 0x00000004
    OPEN_EXISTING = 3
    FILE_ATTRIBUTE_NORMAL = 0x00000080

    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", wintypes.DWORD),
                    ("dwHighDateTime", wintypes.DWORD)]

    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE

    kernel32.SetFileTime.argtypes = (wintypes.HANDLE, ctypes.POINTER(FILETIME),
                                     ctypes.POINTER(FILETIME), ctypes.POINTER(FILETIME))
    kernel32.SetFileTime.restype = wintypes.BOOL

    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL

    def _dt_to_filetime(dt_local_naive: datetime) -> FILETIME:
        try:
            local_tz = datetime.now().astimezone().tzinfo
        except Exception:
            local_tz = timezone.utc
        aware_local = dt_local_naive.replace(tzinfo=local_tz)
        dt_utc = aware_local.astimezone(timezone.utc)

        EPOCH_AS_FILETIME = 116444736000000000  # 1970-01-01 as FILETIME
        HUNDREDS_OF_NANOSECONDS = 10_000_000
        ft = int(dt_utc.timestamp() * HUNDREDS_OF_NANOSECONDS) + EPOCH_AS_FILETIME
        return FILETIME(ft & 0xFFFFFFFF, (ft >> 32) & 0xFFFFFFFF)

    def set_windows_creation_time(path: Path, dt_local_naive: datetime) -> None:
        h = kernel32.CreateFileW(
            str(path),
            GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            None,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            None
        )
        # IMPORTANT: invalid handles can otherwise crash SetFileTime on some machines.
        if not h or h == INVALID_HANDLE_VALUE:
            raise OSError(ctypes.get_last_error(), f"CreateFileW failed for {path}")

        try:
            ctime = _dt_to_filetime(dt_local_naive)
            ok = kernel32.SetFileTime(h, ctypes.byref(ctime), None, None)
            if not ok:
                raise OSError(ctypes.get_last_error(), f"SetFileTime failed for {path}")
        finally:
            kernel32.CloseHandle(h)

@dataclass
class FileTimes:
    path: str
    atime: float
    mtime: float
    ctime: float | None

def read_times(p: Path) -> FileTimes:
    st = p.stat()
    return FileTimes(str(p), st.st_atime, st.st_mtime, st.st_ctime if is_windows() else None)

def apply_times(p: Path, new_dt_local_naive: datetime, set_access: bool, set_modified: bool, set_creation: bool) -> None:
    st = p.stat()
    atime = st.st_atime
    mtime = st.st_mtime

    try:
        local_tz = datetime.now().astimezone().tzinfo
    except Exception:
        local_tz = timezone.utc
    ts = new_dt_local_naive.replace(tzinfo=local_tz).timestamp()

    if set_access:
        atime = ts
    if set_modified:
        mtime = ts
    if set_access or set_modified:
        os.utime(p, (atime, mtime))

    if set_creation and is_windows():
        set_windows_creation_time(p, new_dt_local_naive)

class BatchWorker(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    row_status = Signal(int, str)
    finished = Signal(bool, str)

    def __init__(self, files: list[Path], mode: str, target_dt: datetime,
                 shift: timedelta, set_access: bool, set_modified: bool, set_creation: bool,
                 dry_run: bool, parent=None):
        super().__init__(parent)
        self.files = files
        self.mode = mode
        self.target_dt = target_dt
        self.shift = shift
        self.set_access = set_access
        self.set_modified = set_modified
        self.set_creation = set_creation
        self.dry_run = dry_run
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        backups: dict[str, dict] = {}
        total = len(self.files)
        ok_count = 0

        try:
            for i, p in enumerate(self.files):
                if self._abort:
                    self.finished.emit(False, "Aborted.")
                    return

                try:
                    before = read_times(p)
                    backups[str(p)] = {"atime": before.atime, "mtime": before.mtime, "ctime": before.ctime}

                    if self.mode == "Set":
                        new_dt = self.target_dt
                    else:
                        new_dt = datetime.fromtimestamp(before.mtime).replace(microsecond=0) - self.shift

                    if self.dry_run:
                        self.row_status.emit(i, "Preview (no changes)")
                        self.log.emit(f"[DRY] {p.name} -> {new_dt.isoformat(' ')}")
                    else:
                        apply_times(p, new_dt, self.set_access, self.set_modified, self.set_creation)
                        self.row_status.emit(i, "Updated")
                        self.log.emit(f"[OK] {p.name} -> {new_dt.isoformat(' ')}")
                        ok_count += 1

                except Exception as e:
                    self.row_status.emit(i, f"Error: {e}")
                    self.log.emit(f"[ERR] {p} :: {e}")
                finally:
                    self.progress.emit(i + 1, total)

            if not self.dry_run and ok_count > 0:
                LAST_RUN_JSON.write_text(json.dumps({"created_at": datetime.now().isoformat(), "files": backups}, indent=2), encoding="utf-8")

            self.finished.emit(True, f"Done. Updated {ok_count}/{total} file(s).")
        except Exception:
            self.finished.emit(False, "Unexpected error:\n" + traceback.format_exc())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(980, 680)

        self.files: list[Path] = []
        self.worker_thread: QThread | None = None
        self.worker: BatchWorker | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        self.btn_add_files = QPushButton("Add files")
        self.btn_add_folder = QPushButton("Add folder")
        self.btn_remove = QPushButton("Remove selected")
        self.btn_clear = QPushButton("Clear")
        top.addWidget(self.btn_add_files)
        top.addWidget(self.btn_add_folder)
        top.addStretch(1)
        top.addWidget(self.btn_remove)
        top.addWidget(self.btn_clear)
        root.addLayout(top)

        hint = QLabel("Tip: drag & drop files/folders into the window. Only common image/video types are kept.")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["File", "Type", "Size", "Modified", "New (preview)", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        root.addWidget(self.table, 1)

        opts = QGroupBox("Batch options")
        og = QGridLayout(opts)

        self.mode = QComboBox()
        self.mode.addItems(["Set", "Shift (make older)"])

        self.dt_edit = QDateTimeEdit(QDateTime.currentDateTime())
        self.dt_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dt_edit.setCalendarPopup(True)

        self.shift_days = QSpinBox()
        self.shift_days.setRange(0, 36500)
        self.shift_days.setValue(365)
        self.shift_days.setSuffix(" days")

        self.shift_hours = QSpinBox()
        self.shift_hours.setRange(0, 23)
        self.shift_hours.setValue(0)
        self.shift_hours.setSuffix(" hours")

        self.shift_minutes = QSpinBox()
        self.shift_minutes.setRange(0, 59)
        self.shift_minutes.setValue(0)
        self.shift_minutes.setSuffix(" min")

        self.chk_modified = QCheckBox("Modified time")
        self.chk_modified.setChecked(True)
        self.chk_accessed = QCheckBox("Accessed time")
        self.chk_accessed.setChecked(False)
        self.chk_creation = QCheckBox("Creation time (Windows)")
        self.chk_creation.setChecked(is_windows())
        self.chk_creation.setEnabled(is_windows())

        self.chk_dry = QCheckBox("Dry run (preview only)")
        self.chk_dry.setChecked(True)

        self.btn_preview = QPushButton("Preview")
        self.btn_apply = QPushButton("Apply")
        self.btn_abort = QPushButton("Abort")
        self.btn_abort.setEnabled(False)
        self.btn_restore = QPushButton("Restore last run")

        row = 0
        og.addWidget(QLabel("Mode"), row, 0)
        og.addWidget(self.mode, row, 1, 1, 3)
        row += 1
        og.addWidget(QLabel("Set all to"), row, 0)
        og.addWidget(self.dt_edit, row, 1, 1, 3)
        row += 1
        og.addWidget(QLabel("Make older by"), row, 0)
        og.addWidget(self.shift_days, row, 1)
        og.addWidget(self.shift_hours, row, 2)
        og.addWidget(self.shift_minutes, row, 3)
        row += 1
        og.addWidget(QLabel("Change"), row, 0)
        og.addWidget(self.chk_modified, row, 1)
        og.addWidget(self.chk_accessed, row, 2)
        og.addWidget(self.chk_creation, row, 3)
        row += 1
        og.addWidget(self.chk_dry, row, 1, 1, 3)
        row += 1

        btns = QHBoxLayout()
        btns.addWidget(self.btn_preview)
        btns.addWidget(self.btn_apply)
        btns.addWidget(self.btn_abort)
        btns.addStretch(1)
        btns.addWidget(self.btn_restore)
        og.addLayout(btns, row, 0, 1, 4)

        root.addWidget(opts)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Log...")
        root.addWidget(self.log, 1)

        self.btn_add_files.clicked.connect(self.add_files)
        self.btn_add_folder.clicked.connect(self.add_folder)
        self.btn_remove.clicked.connect(self.remove_selected)
        self.btn_clear.clicked.connect(self.clear_all)
        self.mode.currentIndexChanged.connect(self.refresh_preview_column)
        self.dt_edit.dateTimeChanged.connect(self.refresh_preview_column)
        self.shift_days.valueChanged.connect(self.refresh_preview_column)
        self.shift_hours.valueChanged.connect(self.refresh_preview_column)
        self.shift_minutes.valueChanged.connect(self.refresh_preview_column)
        self.chk_dry.stateChanged.connect(self.on_dry_changed)
        self.btn_preview.clicked.connect(self.refresh_preview_column)
        self.btn_apply.clicked.connect(self.apply_changes)
        self.btn_abort.clicked.connect(self.abort_worker)
        self.btn_restore.clicked.connect(self.restore_last_run)

        self.setAcceptDrops(True)

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        self.menuBar().addAction(act_quit)

        self.on_dry_changed()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        self.add_paths(paths)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose media files")
        if paths:
            self.add_paths([Path(p) for p in paths])

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose folder")
        if folder:
            self.add_paths([Path(folder)])

    def add_paths(self, paths: list[Path]):
        added = 0
        for p in paths:
            if p.is_dir():
                for fp in p.rglob("*"):
                    if fp.is_file() and fp.suffix.lower() in MEDIA_EXTS:
                        added += self._add_one(fp)
            elif p.is_file() and p.suffix.lower() in MEDIA_EXTS:
                added += self._add_one(p)

        if added:
            self.log_line(f"Added {added} file(s). Total: {len(self.files)}")
            self.refresh_table()
        else:
            self.log_line("No supported media found.")

    def _add_one(self, p: Path) -> int:
        try:
            p = p.resolve()
        except Exception:
            pass
        if p in self.files:
            return 0
        self.files.append(p)
        return 1

    def remove_selected(self):
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()}, reverse=True)
        for r in rows:
            if 0 <= r < len(self.files):
                self.files.pop(r)
        self.refresh_table()

    def clear_all(self):
        self.files.clear()
        self.table.setRowCount(0)
        self.progress.setValue(0)
        self.log_line("Cleared.")

    def refresh_table(self):
        self.table.setRowCount(len(self.files))
        for i, p in enumerate(self.files):
            try:
                st = p.stat()
                size = st.st_size
                mtime = datetime.fromtimestamp(st.st_mtime).replace(microsecond=0)
                ext = p.suffix.lower().lstrip(".")
                typ = "video" if ext in {"mp4","mov","mkv","avi","webm","m4v","wmv","mpg","mpeg"} else "image"
            except Exception:
                size = 0
                mtime = datetime.fromtimestamp(0)
                typ = "?"
            self.table.setItem(i, 0, QTableWidgetItem(str(p)))
            self.table.setItem(i, 1, QTableWidgetItem(typ))
            self.table.setItem(i, 2, QTableWidgetItem(self._fmt_size(size)))
            self.table.setItem(i, 3, QTableWidgetItem(mtime.isoformat(" ")))
            self.table.setItem(i, 4, QTableWidgetItem(""))
            self.table.setItem(i, 5, QTableWidgetItem(""))
        self.refresh_preview_column()

    def refresh_preview_column(self):
        if not self.files:
            return

        mode = self.mode.currentText()
        target_dt = self.dt_edit.dateTime().toPython().replace(microsecond=0)
        shift = timedelta(days=self.shift_days.value(), hours=self.shift_hours.value(), minutes=self.shift_minutes.value())

        for i, p in enumerate(self.files):
            try:
                st = p.stat()
                mtime = datetime.fromtimestamp(st.st_mtime).replace(microsecond=0)
                new_dt = target_dt if mode == "Set" else (mtime - shift)
                self.table.item(i, 4).setText(new_dt.isoformat(" "))
                self.table.item(i, 5).setText("")
            except Exception as e:
                self.table.item(i, 4).setText("")
                self.table.item(i, 5).setText(f"Error: {e}")

    def on_dry_changed(self):
        if self.chk_dry.isChecked():
            self.btn_apply.setText("Apply (disabled in dry run)")
            self.btn_apply.setEnabled(False)
        else:
            self.btn_apply.setText("Apply")
            self.btn_apply.setEnabled(True)

    def _fmt_size(self, n: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        s = float(n)
        for u in units:
            if s < 1024.0 or u == units[-1]:
                return f"{s:.1f} {u}" if u != "B" else f"{int(s)} {u}"
            s /= 1024.0
        return f"{n} B"

    def log_line(self, msg: str):
        self.log.append(msg)

    def apply_changes(self):
        if not self.files:
            QMessageBox.information(self, APP_NAME, "Add some files first.")
            return

        if not (self.chk_modified.isChecked() or self.chk_accessed.isChecked() or (self.chk_creation.isChecked() and is_windows())):
            QMessageBox.warning(self, APP_NAME, "Select at least one timestamp to change.")
            return

        mode = "Set" if self.mode.currentText().startswith("Set") else "Shift"
        target_dt = self.dt_edit.dateTime().toPython().replace(microsecond=0)
        shift = timedelta(days=self.shift_days.value(), hours=self.shift_hours.value(), minutes=self.shift_minutes.value())
        dry_run = self.chk_dry.isChecked()

        self.progress.setValue(0)
        self.btn_abort.setEnabled(True)
        self.btn_apply.setEnabled(False)
        self.btn_preview.setEnabled(False)
        self.btn_add_files.setEnabled(False)
        self.btn_add_folder.setEnabled(False)
        self.btn_remove.setEnabled(False)
        self.btn_clear.setEnabled(False)

        self.worker_thread = QThread()
        self.worker = BatchWorker(
            files=self.files.copy(),
            mode=mode,
            target_dt=target_dt,
            shift=shift,
            set_access=self.chk_accessed.isChecked(),
            set_modified=self.chk_modified.isChecked(),
            set_creation=self.chk_creation.isChecked() and is_windows(),
            dry_run=dry_run
        )
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log_line)
        self.worker.row_status.connect(self.on_row_status)

        # Cleanup order: stop thread, then update UI
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.on_finished)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    def abort_worker(self):
        if self.worker:
            self.worker.abort()
            self.log_line("Abort requested...")

    def on_progress(self, done: int, total: int):
        self.progress.setValue(int((done / max(1, total)) * 100))

    def on_row_status(self, row: int, status: str):
        if 0 <= row < self.table.rowCount():
            self.table.item(row, 5).setText(status)

    def on_finished(self, ok: bool, msg: str):
        try:
            self.btn_abort.setEnabled(False)
            self.btn_preview.setEnabled(True)
            self.btn_add_files.setEnabled(True)
            self.btn_add_folder.setEnabled(True)
            self.btn_remove.setEnabled(True)
            self.btn_clear.setEnabled(True)
            self.on_dry_changed()
            self.log_line(msg)
            if not ok:
                QMessageBox.warning(self, APP_NAME, msg)

            self.worker = None
            self.worker_thread = None
            self.refresh_table()
        except Exception:
            show_fatal_dialog(f"{APP_NAME} - Error", traceback.format_exc())

    def restore_last_run(self):
        if not LAST_RUN_JSON.exists():
            QMessageBox.information(self, APP_NAME, "No previous backup found.")
            return
        try:
            data = json.loads(LAST_RUN_JSON.read_text(encoding="utf-8"))
            files = data.get("files", {})
            if not files:
                QMessageBox.information(self, APP_NAME, "Backup file is empty.")
                return
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Could not read backup: {e}")
            return

        resp = QMessageBox.question(self, APP_NAME, f"This will restore timestamps for {len(files)} file(s) from the last run.\nContinue?")
        if resp != QMessageBox.Yes:
            return

        restored = 0
        for path_str, times in files.items():
            p = Path(path_str)
            try:
                if not p.exists():
                    self.log_line(f"[MISS] {p} (not found)")
                    continue
                atime = float(times.get("atime", p.stat().st_atime))
                mtime = float(times.get("mtime", p.stat().st_mtime))
                os.utime(p, (atime, mtime))

                if is_windows() and times.get("ctime") is not None:
                    dt = datetime.fromtimestamp(float(times["ctime"])).replace(microsecond=0)
                    try:
                        set_windows_creation_time(p, dt)
                    except Exception as e:
                        self.log_line(f"[WARN] Could not restore creation time for {p.name}: {e}")

                restored += 1
                self.log_line(f"[RESTORE] {p.name}")
            except Exception as e:
                self.log_line(f"[ERR] Restore failed for {p}: {e}")

        self.log_line(f"Restore complete: {restored}/{len(files)}")
        self.refresh_table()

def main():
    app = QApplication(sys.argv)
    install_excepthook()
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
