
# helpers/bug_reporter.py  — FrameVision bug reporter (stable export)
from __future__ import annotations

# ---- FrameVision: Import probe (runs when Bug Reporter opens) ----
def _fv_probe_imports():
    try:
        import sys, time, tempfile
        from pathlib import Path as _P
        names = [
            "helpers.sysmon",
            "helpers.bug_injector",
            "helpers.bug_reporter",
            "helpers.tools_tab",
            "helpers.social_buttons",
            "helpers.settings_enhancer",
            "helpers.temp_units_injector",
            "helpers.settings_tab",
            "helpers.settings_layout",
            "helpers.theme_autowire",
            "helpers.ui_fixups",
        ]
        status_lines = []
        for name in names:
            mod = sys.modules.get(name)
            path = getattr(mod, "__file__", None) if mod else None
            status_lines.append(f"{name}: {'LOADED' if mod else 'not loaded'}" + (f' — {path}' if path else ''))
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        body = f"[{ts}] import probe (from bug_reporter)\\n" + "\\n".join(status_lines) + "\\n\\n"
        # 1) Project root (next to helpers/)
        try:
            proj = _P(__file__).resolve().parent.parent
            d1 = proj / "diagnostics"
            d1.mkdir(parents=True, exist_ok=True)
            (d1 / "imports.log").write_text(body, encoding="utf-8", errors="ignore")
        except Exception:
            pass
        # 2) User Documents
        try:
            home = _P.home()
            d2 = home / "Documents" / "FrameVision" / "diagnostics"
            d2.mkdir(parents=True, exist_ok=True)
            (d2 / "imports.log").write_text(body, encoding="utf-8", errors="ignore")
        except Exception:
            pass
        # 3) Temp folder
        try:
            tmp = _P(tempfile.gettempdir())
            d3 = tmp / "FrameVision_diagnostics"
            d3.mkdir(parents=True, exist_ok=True)
            (d3 / "imports.log").write_text(body, encoding="utf-8", errors="ignore")
        except Exception:
            pass
    except Exception:
        pass

import os, json, shutil, zipfile, pathlib, urllib.parse, importlib.util, smtplib
from datetime import datetime
from email.message import EmailMessage

from PySide6.QtCore import Qt, QUrl, QMimeData, QTimer
from PySide6.QtGui import QDesktopServices, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QCheckBox,
    QPushButton, QFileDialog, QListWidget, QListWidgetItem, QAbstractItemView,
    QMessageBox, QSplitter, QSizePolicy, QComboBox, QToolButton, QFrame, QWidget, QFormLayout, QSpinBox
)

APP_EMAIL = "frame.vision@mail.com"

def _safe_import_psutil():
    try:
        if importlib.util.find_spec("psutil") is not None:
            import psutil
            return psutil
    except Exception:
        return None
    return None

def build_snapshot_text() -> str:
    lines = []
    try:
        import platform as _p
        lines.append("=== FrameVision Snapshot ===")
        lines.append(f"Time: {datetime.now().isoformat(timespec='seconds')}")
        lines.append(f"Python: {_p.python_version()}")
        lines.append(f"OS: {_p.platform()}")
        psutil = _safe_import_psutil()
        if psutil:
            lines.append(f"CPU logical cores: {psutil.cpu_count()} (physical: {psutil.cpu_count(logical=False)})")
            vm = psutil.virtual_memory()
            lines.append(f"RAM: {round(vm.total/(1024**3),1)} GiB total, {round(vm.available/(1024**3),1)} GiB free")
    except Exception:
        pass
    return "\\n".join(lines) + "\\n"

def ensure_logs_dir() -> pathlib.Path:
    p = pathlib.Path("logs")
    p.mkdir(parents=True, exist_ok=True)
    return p

class SmtpSettings(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from PySide6.QtCore import QSettings
        self._settings = QSettings()
        form = QFormLayout(self)
        self.ed_host = QLineEdit(self._settings.value("bugreport/smtp_host", ""))
        self.sp_port = QSpinBox(); self.sp_port.setRange(1,65535); self.sp_port.setValue(int(self._settings.value("bugreport/smtp_port", 587)))
        self.ed_user = QLineEdit(self._settings.value("bugreport/smtp_user", ""))
        self.ed_pass = QLineEdit(); self.ed_pass.setEchoMode(QLineEdit.Password)
        self.chk_tls = QCheckBox("Use STARTTLS (recommended)"); self.chk_tls.setChecked(False)  # default removed; will restore via UI
        self.chk_ssl = QCheckBox("Use SSL")
        self.chk_remember = QCheckBox("Remember settings (password not stored)"); self.chk_remember.setChecked(False)  # default removed; will restore via UI
        form.addRow("SMTP server:", self.ed_host)
        form.addRow("Port:", self.sp_port)
        form.addRow("Username:", self.ed_user)
        form.addRow("Password:", self.ed_pass)
        form.addRow(self.chk_tls); form.addRow(self.chk_ssl); form.addRow(self.chk_remember)
    def save(self):
        from PySide6.QtCore import QSettings
        if self.chk_remember.isChecked():
            s = QSettings()
            s.setValue("bugreport/smtp_host", self.ed_host.text())
            s.setValue("bugreport/smtp_port", self.sp_port.value())
            s.setValue("bugreport/smtp_user", self.ed_user.text())
    def values(self):
        return dict(
            host=self.ed_host.text().strip(),
            port=int(self.sp_port.value()),
            user=self.ed_user.text().strip(),
            password=self.ed_pass.text(),
            use_tls=self.chk_tls.isChecked(),
            use_ssl=self.chk_ssl.isChecked(),
        )

class BugReporter(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Bug report")
        self.resize(920, 720)
        try: self.setWindowIcon(QIcon.fromTheme("dialog-warning"))
        except Exception: pass

        outer = QVBoxLayout(self)
        intro = QLabel("Describe the problem. Optional: include logs and a system snapshot. Click “Create ZIP”, then “Email ZIP…”")
        intro.setWordWrap(True); outer.addWidget(intro)

        splitter = QSplitter(Qt.Vertical); outer.addWidget(splitter, 1)

        top = QWidget(); top_l = QVBoxLayout(top)
        row0 = QHBoxLayout()
        row0.addWidget(QLabel("Subject:"))
        self.ed_subject = QLineEdit(); self.ed_subject.setPlaceholderText("Short title, e.g., 'Crash after pressing Long description'")
        row0.addWidget(self.ed_subject, 2)
        row0.addWidget(QLabel("From (optional):"))
        self.ed_from = QLineEdit(); self.ed_from.setPlaceholderText("Optional name / handle")
        row0.addWidget(self.ed_from, 1)
        row0.addWidget(QLabel("Say hi:"))
        self.cb_hi = QComboBox(); self.cb_hi.addItems(["—","😀","🙂","😉","🤖","🙏","🙌","😂","❤️","✨","🔥","👍"])
        row0.addWidget(self.cb_hi, 0)
        top_l.addLayout(row0)

        self.txt = QTextEdit(); self.txt.setAcceptRichText(False)
        self.txt.setPlainText(build_snapshot_text())
        self.txt.setMinimumHeight(260); self.txt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        top_l.addWidget(self.txt, 1)

        opts = QHBoxLayout()
        self.chk_snapshot = QCheckBox("Include system snapshot"); self.chk_snapshot.setChecked(False)  # default removed; will restore via UI
        self.chk_logs = QCheckBox("Include logs folder"); self.chk_logs.setChecked(False)  # default removed; will restore via UI
        opts.addWidget(self.chk_snapshot); opts.addWidget(self.chk_logs); opts.addStretch(1)
        top_l.addLayout(opts)
        splitter.addWidget(top)

        files_box = QWidget(); files_l = QVBoxLayout(files_box)
        self.file_list = QListWidget(); self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        files_l.addWidget(self.file_list, 1)
        row_files = QHBoxLayout()
        self.btn_add = QPushButton("Add files…"); self.btn_remove = QPushButton("Remove selected")
        row_files.addWidget(self.btn_add); row_files.addWidget(self.btn_remove); row_files.addStretch(1)
        files_l.addLayout(row_files)
        splitter.addWidget(files_box)

        # Actions row
        actions = QHBoxLayout()
        self.btn_zip = QPushButton("Create ZIP")
        self.btn_email = QPushButton("Email ZIP…")
        self.btn_close = QPushButton("Close")
        actions.addWidget(self.btn_zip); actions.addWidget(self.btn_email)
        actions.addStretch(1); actions.addWidget(self.btn_close)
        outer.addLayout(actions)

        self.lbl_status = QLabel(""); outer.addWidget(self.lbl_status)

        # Signals
        self.btn_add.clicked.connect(self._add_files)
        self.btn_remove.clicked.connect(self._remove_files)
        self.btn_zip.clicked.connect(self._create_zip)
        self.btn_email.clicked.connect(self._email_zip)
        self.btn_close.clicked.connect(self.close)

        self._zip_path: pathlib.Path | None = None

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add files", "", "All files (*.*)")
        for f in files:
            if f: self.file_list.addItem(QListWidgetItem(f))

    def _remove_files(self):
        for it in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(it))

    def _stage_zip_files(self, temp_stage: pathlib.Path):
        # snapshot
        if self.chk_snapshot.isChecked():
            snap = self.txt.toPlainText().strip() or build_snapshot_text()
            (temp_stage / "snapshot.txt").write_text(snap, encoding="utf-8")
        # logs
        if self.chk_logs.isChecked():
            logs_dir = pathlib.Path("logs")
            if logs_dir.exists():
                for p in logs_dir.rglob("*"):
                    if p.is_file():
                        try: shutil.copy2(p, temp_stage / p.name)
                        except Exception: pass
        # attachments
        for i in range(self.file_list.count()):
            p = pathlib.Path(self.file_list.item(i).text())
            if p.exists() and p.is_file():
                try: shutil.copy2(p, temp_stage / p.name)
                except Exception: pass

    def _create_zip(self):
        logs_root = ensure_logs_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = logs_root / f"bugreport_{stamp}.zip"
        temp_stage = logs_root / f"_tmp_{stamp}"; temp_stage.mkdir(parents=True, exist_ok=True)
        try:
            self._stage_zip_files(temp_stage)
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
                for p in temp_stage.rglob("*"):
                    if p.is_file(): z.write(p, arcname=p.name)
            self._zip_path = zip_path
            self.lbl_status.setText(f"Created: {zip_path}")
            try: QDesktopServices.openUrl(QUrl.fromLocalFile(str(zip_path.parent.resolve())))
            except Exception: pass
        except Exception as e:
            QMessageBox.critical(self, "Bug report", f"Failed to create ZIP:\\n{e}")
        finally:
            try: shutil.rmtree(temp_stage, ignore_errors=True)
            except Exception: pass

    def _email_zip(self):
        subject = self.ed_subject.text().strip() or "FrameVision bug report"
        notes = self.txt.toPlainText().strip()
        from_name = self.ed_from.text().strip()
        hi = self.cb_hi.currentText()
        body_lines = ["Hi,","", "Please find my bug report attached."]
        if self._zip_path: body_lines += ["", f"ZIP path on disk: {self._zip_path}"]
        if from_name or (hi and hi != "—"): body_lines += ["", "— " + (from_name or "user") + (" " + hi if hi and hi != "—" else "")]
        if notes: body_lines += ["", "Notes:", notes]
        body = "\\n".join(body_lines)
        url = f"mailto:{APP_EMAIL}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"
        QDesktopServices.openUrl(QUrl(url))

def open_bug_reporter(parent=None):
    try:
        _fv_probe_imports()
    except Exception:
        pass
    """Stable export used by Settings screen. Keep name EXACTLY as-is."""
    dlg = BugReporter(parent); dlg.show(); return dlg
