
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

APP_EMAIL = "framevision.mail@mail.com"

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
        self.chk_logs = QCheckBox("Include log file"); self.chk_logs.setChecked(False)  # default removed; will restore via UI
        self.cb_log = QComboBox(); self.cb_log.setMinimumWidth(320); self.cb_log.setEnabled(False)
        self.cb_log.setToolTip("Select a log file to paste into the email body (and optionally include in the ZIP).")
        self.btn_refresh_logs = QToolButton(); self.btn_refresh_logs.setText("↻"); self.btn_refresh_logs.setToolTip("Refresh log list")
        self.btn_refresh_logs.setEnabled(False)
        opts.addWidget(self.chk_snapshot); opts.addWidget(self.chk_logs); opts.addWidget(self.cb_log); opts.addWidget(self.btn_refresh_logs); opts.addStretch(1)
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
        # Log selection
        self._refresh_log_choices()
        self.chk_logs.toggled.connect(self._on_log_toggle)
        self.btn_refresh_logs.clicked.connect(self._refresh_log_choices)
        self.btn_close.clicked.connect(self.close)

        self._zip_path: pathlib.Path | None = None

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Add files", "", "All files (*.*)")
        for f in files:
            if f: self.file_list.addItem(QListWidgetItem(f))

    def _remove_files(self):
        for it in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(it))


    def _refresh_log_choices(self):
        """Populate the log selector with recent *log* files (not ZIPs)."""
        try:
            self.cb_log.blockSignals(True)
            self.cb_log.clear()
            logs_dir = ensure_logs_dir()
            files = []
            try:
                for p in logs_dir.iterdir():
                    if not p.is_file():
                        continue
                    # Exclude bugreport ZIPs and temp staging folders/files
                    name = p.name.lower()
                    if name.endswith(".zip"):
                        continue
                    if name.startswith("bugreport_") or name.startswith("_tmp_"):
                        continue
                    files.append(p)
            except Exception:
                files = []

            def _key(p: pathlib.Path):
                try:
                    return p.stat().st_mtime
                except Exception:
                    return 0

            files.sort(key=_key, reverse=True)

            if not files:
                self.cb_log.addItem("(no logs found)", None)
                self.cb_log.setEnabled(False)
                self.btn_refresh_logs.setEnabled(False)
                return

            for p in files[:200]:
                try:
                    size_kb = int(p.stat().st_size / 1024)
                except Exception:
                    size_kb = 0
                self.cb_log.addItem(f"{p.name}  ({size_kb} KB)", str(p))

            en = bool(self.chk_logs.isChecked())
            self.cb_log.setEnabled(en)
            self.btn_refresh_logs.setEnabled(en)
        finally:
            try:
                self.cb_log.blockSignals(False)
            except Exception:
                pass

    def _on_log_toggle(self, checked: bool):
        self.cb_log.setEnabled(bool(checked))
        self.btn_refresh_logs.setEnabled(bool(checked))
        if checked and self.cb_log.count() == 0:
            self._refresh_log_choices()

    def _get_selected_log_path(self) -> pathlib.Path | None:
        try:
            data = self.cb_log.currentData()
            if not data:
                return None
            return pathlib.Path(str(data))
        except Exception:
            return None

    def _unique_dest(self, folder: pathlib.Path, filename: str) -> pathlib.Path:
        base = pathlib.Path(filename).name
        cand = folder / base
        if not cand.exists():
            return cand
        stem = cand.stem
        suf = cand.suffix
        for i in range(2, 9999):
            cand2 = folder / f"{stem}_{i}{suf}"
            if not cand2.exists():
                return cand2
        return folder / f"{stem}_{datetime.now().strftime('%H%M%S')}{suf}"

    def _included_summary_lines(self) -> list[str]:
        lines = []
        lines.append(f"Include system snapshot: {'Yes' if self.chk_snapshot.isChecked() else 'No'}")
        if self.chk_logs.isChecked():
            lp = self._get_selected_log_path()
            lines.append(f"Include log file: {lp.name if lp else '(none selected)'}")
        else:
            lines.append("Include log file: No")
        extra = []
        for i in range(self.file_list.count()):
            try:
                extra.append(pathlib.Path(self.file_list.item(i).text()).name)
            except Exception:
                pass
        if extra:
            lines.append("Extra attachments: " + ", ".join(extra))
        return lines

    def _stage_zip_files(self, temp_stage: pathlib.Path):
        # snapshot
        if self.chk_snapshot.isChecked():
            snap = self.txt.toPlainText().strip() or build_snapshot_text()
            (temp_stage / "snapshot.txt").write_text(snap, encoding="utf-8")
        # logs (single file)
        if self.chk_logs.isChecked():
            lp = self._get_selected_log_path()
            if lp and lp.exists() and lp.is_file():
                try:
                    shutil.copy2(lp, self._unique_dest(temp_stage, lp.name))
                except Exception:
                    pass

        # attachments
        for i in range(self.file_list.count()):
            p = pathlib.Path(self.file_list.item(i).text())
            if p.exists() and p.is_file():
                try: shutil.copy2(p, temp_stage / p.name)
                except Exception: pass


    def _read_text_tail(self, path: pathlib.Path, *, max_bytes: int = 180_000, max_lines: int = 220) -> str:
        """Read the tail of a text file, safely."""
        try:
            if not path or not path.exists() or not path.is_file():
                return ""
            size = path.stat().st_size
            with open(path, "rb") as f:
                if size > max_bytes:
                    try:
                        f.seek(max(0, size - max_bytes))
                    except Exception:
                        pass
                data = f.read()
            txt = data.decode("utf-8", errors="ignore")
            lines = txt.splitlines()
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            return "\n".join(lines).strip() + "\n"
        except Exception:
            return ""

    def _extract_traceback_blocks(self, text: str, *, max_blocks: int = 2, max_lines: int = 90, max_chars: int = 2600) -> list[str]:
        """Extract recent Python traceback blocks from text."""
        if not text:
            return []
        marker = "Traceback (most recent call last):"
        idxs = []
        start = 0
        while True:
            i = text.find(marker, start)
            if i < 0:
                break
            idxs.append(i)
            start = i + len(marker)
        if not idxs:
            return []

        blocks = []
        for a, b in zip(idxs, idxs[1:] + [len(text)]):
            chunk = text[a:b].strip()
            if not chunk:
                continue
            lines = chunk.splitlines()
            if len(lines) > max_lines:
                lines = lines[:max_lines]
                lines.append("...(traceback truncated)")
            chunk2 = "\n".join(lines).strip()
            if len(chunk2) > max_chars:
                chunk2 = chunk2[: max_chars - 80].rstrip() + "\n...(traceback truncated)"
            blocks.append(chunk2)

        # Most recent first
        blocks = list(reversed(blocks))[:max_blocks]
        return blocks

    def _iter_recent_log_files(self, *, limit: int = 5) -> list[pathlib.Path]:
        logs_dir = ensure_logs_dir()
        files = []
        try:
            for p in logs_dir.iterdir():
                if not p.is_file():
                    continue
                name = p.name.lower()
                if name.endswith(".zip"):
                    continue
                if name.startswith("bugreport_") or name.startswith("_tmp_"):
                    continue
                files.append(p)
        except Exception:
            return []
        files.sort(key=lambda p: getattr(p.stat(), "st_mtime", 0), reverse=True)
        return files[:limit]

    def _compose_email_body(self, zip_abs: str) -> tuple[str, str, bool]:
        """Return (full_body, mailto_body, was_truncated_for_mailto)."""
        subject = self.ed_subject.text().strip() or "FrameVision bug report"
        notes = (self.txt.toPlainText() or "").strip()
        from_name = self.ed_from.text().strip()
        hi = self.cb_hi.currentText()

        lines = []
        lines += ["Hi,", ""]
        lines += ["Bug report pasted below (most email apps cannot auto-attach files via a mailto link).", ""]
        lines += [f"Subject: {subject}"]
        if from_name:
            lines += [f"From: {from_name}"]
        lines += ["", "Included:"]
        lines += ["- " + s for s in self._included_summary_lines()]
        lines += ["", f"ZIP path on disk: {zip_abs}"]

        # Snapshot section (fresh snapshot text). Avoid duplicating if Notes already starts with the snapshot.
        snap_already = False
        try:
            snap_already = notes.lstrip().startswith("=== FrameVision Snapshot ===")
        except Exception:
            snap_already = False
        if self.chk_snapshot.isChecked() and not snap_already:
            snap = build_snapshot_text().strip()
            if snap:
                lines += ["", "=== System snapshot ===", snap]

        # Log section (paste tail)
        log_tail = ""
        selected_log = None
        if self.chk_logs.isChecked():
            selected_log = self._get_selected_log_path()
            if selected_log and selected_log.exists() and selected_log.is_file():
                log_tail = self._read_text_tail(selected_log)

        if log_tail:
            lines += ["", f"=== Log excerpt (tail): {selected_log.name} ===", log_tail.rstrip()]

        # Tracebacks section
        tb_blocks = []
        tb_source = None
        if log_tail:
            tb_blocks = self._extract_traceback_blocks(log_tail)
            tb_source = selected_log
        else:
            # If no log selected, try recent logs for tracebacks (useful when snapshot is enabled)
            for p in self._iter_recent_log_files(limit=5):
                tail = self._read_text_tail(p)
                blocks = self._extract_traceback_blocks(tail)
                if blocks:
                    tb_blocks = blocks
                    tb_source = p
                    break

        if tb_blocks:
            header = "=== Recent tracebacks (most recent first) ==="
            if tb_source:
                header += f"  [{tb_source.name}]"
            lines += ["", header]
            for i, b in enumerate(tb_blocks, 1):
                lines += [f"\n--- Traceback #{i} ---", b.strip()]

        if notes:
            lines += ["", "=== Notes ===", notes]

        # Signature
        if from_name or (hi and hi != "—"):
            sig = (from_name or "user") + ((" " + hi) if hi and hi != "—" else "")
            lines += ["", "— " + sig]

        full_body = "\n".join(lines).strip() + "\n"

        # Build mailto-safe body (keep shorter) but always copy the full body to clipboard.
        mailto_body = full_body
        was_trunc = False

        # Hard cap plain length
        max_plain = 8500
        if len(mailto_body) > max_plain:
            mailto_body = mailto_body[: max_plain - 200].rstrip() + "\n\n...(truncated for email body; full report copied to clipboard)\n"
            was_trunc = True

        # Cap encoded length (mailto URL)
        try:
            enc = urllib.parse.quote(mailto_body)
            max_enc = 12000
            if len(enc) > max_enc:
                cut = 6000
                mailto_body = mailto_body[: cut - 200].rstrip() + "\n\n...(truncated for email body; full report copied to clipboard)\n"
                was_trunc = True
        except Exception:
            pass

        return full_body, mailto_body, was_trunc


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
        # Ensure ZIP exists (useful for manual attach and for SMTP sending)
        if not self._zip_path or not self._zip_path.exists():
            self._create_zip()
            if not self._zip_path or not self._zip_path.exists():
                QMessageBox.warning(self, "Bug report", "Please create a ZIP first.")
                return

        subject = self.ed_subject.text().strip() or "FrameVision bug report"
        zip_abs = str(self._zip_path.resolve())

        # Compose body (and a mailto-safe version)
        full_body, mailto_body, was_trunc = self._compose_email_body(zip_abs)

        # Offer SMTP (attach ZIP) vs mail app (body only)
        mb = QMessageBox(self)
        mb.setWindowTitle("Email report")
        mb.setIcon(QMessageBox.Information)
        mb.setText(
            "Choose how you want to send your report:\n\n"
            "• Open email app: pastes details into the email body (ZIP cannot be auto-attached).\n"
            "• Send via SMTP: sends the email directly and attaches the ZIP."
        )
        btn_smtp = mb.addButton("Send via SMTP (attach ZIP)", QMessageBox.AcceptRole)
        btn_mail = mb.addButton("Open email app (body only)", QMessageBox.ActionRole)
        btn_cancel = mb.addButton(QMessageBox.Cancel)
        mb.setDefaultButton(btn_mail)
        mb.exec()

        clicked = mb.clickedButton()
        if clicked == btn_cancel:
            return

        if clicked == btn_smtp:
            dlg = QDialog(self)
            dlg.setWindowTitle("SMTP settings")
            dlg.resize(520, 260)
            v = QVBoxLayout(dlg)

            info = QLabel(
                "Enter your SMTP details to send the report with the ZIP attached.\n"
                "Tip: Gmail/Outlook may require an app password."
            )
            info.setWordWrap(True)
            v.addWidget(info)

            smtpw = SmtpSettings(dlg)
            v.addWidget(smtpw, 1)

            row = QHBoxLayout()
            btn_send = QPushButton("Send")
            btn_close = QPushButton("Cancel")
            row.addStretch(1)
            row.addWidget(btn_send)
            row.addWidget(btn_close)
            v.addLayout(row)

            def do_send():
                cfg = smtpw.values()
                if not cfg.get("host") or not cfg.get("user"):
                    QMessageBox.warning(dlg, "SMTP", "Please enter at least SMTP server and user/email.")
                    return
                try:
                    msg = EmailMessage()
                    msg["From"] = cfg["user"]
                    msg["To"] = APP_EMAIL
                    msg["Subject"] = subject
                    msg.set_content(full_body)

                    with open(zip_abs, "rb") as f:
                        msg.add_attachment(
                            f.read(),
                            maintype="application",
                            subtype="zip",
                            filename=self._zip_path.name,
                        )

                    if cfg.get("use_ssl"):
                        server = smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=25)
                    else:
                        server = smtplib.SMTP(cfg["host"], cfg["port"], timeout=25)

                    try:
                        server.ehlo()
                        if cfg.get("use_tls") and not cfg.get("use_ssl"):
                            server.starttls()
                            server.ehlo()
                        server.login(cfg["user"], cfg.get("password", ""))
                        server.send_message(msg)
                    finally:
                        try:
                            server.quit()
                        except Exception:
                            pass

                    self.lbl_status.setText(f"Sent via SMTP to: {APP_EMAIL} (attached: {self._zip_path.name})")
                    QMessageBox.information(dlg, "Bug report", "Sent! The ZIP was attached.")
                    dlg.accept()

                except Exception as e:
                    QMessageBox.critical(dlg, "Bug report", f"SMTP send failed:\n{e}")

            btn_send.clicked.connect(do_send)
            btn_close.clicked.connect(dlg.reject)
            dlg.exec()
            return

        # Open email client and paste details into body.
        # Copy full body to clipboard in case the mailto body is truncated by the email client.
        try:
            QGuiApplication.clipboard().setText(full_body)
        except Exception:
            pass

        # Also open ZIP folder for quick manual attach if desired
        try:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._zip_path.parent.resolve())))
        except Exception:
            pass

        url = f"mailto:{APP_EMAIL}?subject={urllib.parse.quote(subject)}&body={urllib.parse.quote(mailto_body)}"
        QDesktopServices.openUrl(QUrl(url))

        msg = (
            "Your email client was opened.\n\n"
            "Most mail apps can't auto-attach files from a mailto link.\n"
            "The full report text was copied to your clipboard (paste if needed).\n\n"
            f"ZIP on disk (attach manually if you want):\n{zip_abs}"
        )
        if was_trunc:
            msg += "\n\nNote: The email body may be truncated by your email client. Paste from clipboard to restore the full text."
        QMessageBox.information(self, "Bug report", msg)

def open_bug_reporter(parent=None):
    try:
        _fv_probe_imports()
    except Exception:
        pass
    """Stable export used by Settings screen. Keep name EXACTLY as-is."""
    dlg = BugReporter(parent); dlg.show(); return dlg
