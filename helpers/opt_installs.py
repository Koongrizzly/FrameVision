"""
FrameVision Optional Installs UI (PySide6)

This module provides a small dialog that lets users select and run optional
install add-ons. It mirrors the "optional installs" strategies from
install_menu.bat by invoking the same extra installer scripts (when present).

Intended usage:
- Import in FrameVision and open the dialog from a menu item.
- Can also be run standalone for testing: `python -m helpers.opt_installs`
"""

from __future__ import annotations

import os
import sys
import shutil
import stat
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict

from PySide6 import QtCore, QtGui, QtWidgets


# -----------------------------
# Models / Tasks
# -----------------------------

@dataclass(frozen=True)
class OptionalInstall:
    key: str
    title: str
    description: str
    # Returns (program, args, working_dir) or None when missing prerequisites.
    runner: Callable[[Path], Optional[Tuple[str, List[str], Path]]]


def _root_from_this_file() -> Path:
    """
    Determine the FrameVision root folder.

    - If this file lives in <root>/helpers/, return <root>
    - If this file lives directly in <root>/, return <root>
    """
    try:
        p = Path(__file__).resolve()
        if p.parent.name.lower() == "helpers":
            return p.parent.parent
        return p.parent
    except Exception:
        return Path.cwd().resolve()


def _venv_python(root: Path) -> Optional[Path]:
    """
    Prefer <root>/.venv python. Fall back to current interpreter.
    """
    candidates = [
        root / ".venv" / "Scripts" / "python.exe",   # Windows
        root / ".venv" / "bin" / "python",           # *nix
        root / ".venv" / "bin" / "python3",
    ]
    for c in candidates:
        if c.exists():
            return c
    # If we're running inside FrameVision already, sys.executable is fine.
    if Path(sys.executable).exists():
        return Path(sys.executable)
    return None


def _cmd_call_bat(script_path: Path, cwd: Path) -> Tuple[str, List[str], Path]:
    """
    On Windows, run .bat through cmd.exe /c call
    """
    return ("cmd.exe", ["/c", "call", str(script_path)], cwd)


def _run_wan22(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "wan22_setup.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_zimage(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "zimage_install.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)



def _run_zimage_fp16(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    """Install Z-Image Turbo safetensors (fp16 diffusion + text encoder + VAE)."""
    script = root / "presets" / "extra_env" / "zimage_gguf_install.py"
    if not script.exists():
        return None
    py = _venv_python(root)
    if py is None or (not py.exists()):
        return None
    target = (root / "models" / "Z-Image-Turbo").resolve()
    args = [
        "-u",
        str(script),
        str(root),
        str(target),
        "--mode",
        "safetensors",
        "--precision",
        "fp16",
    ]
    return (str(py), args, root)


def _run_zimage_gguf(root: Path, quant: str) -> Optional[Tuple[str, List[str], Path]]:
    """
    Z-Image Turbo GGUF installer (no .bat wrapper).
    Supported quants: Q4_0, Q5_0, Q6_K, Q8_0
    """
    script = root / "presets" / "extra_env" / "zimage_gguf_install.py"
    if not script.exists():
        return None

    py = _venv_python(root)
    if not py:
        return None

    target = (root / "models" / "Z-Image-Turbo GGUF").resolve()
    args = [
        "-u",
        str(script),
        str(root),
        str(target),
        "--quant",
        quant,
        "--match-text-quant",
        "1",
    ]
    return (str(py), args, root)


def _run_zimage_gguf5(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    # Back-compat key: was "GGUF5 model" -> now installs diffusion Q5_0.
    return _run_zimage_gguf(root, "Q5_0")


def _run_zimage_gguf4(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_gguf(root, "Q4_0")


def _run_zimage_gguf6(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_gguf(root, "Q6_K")


def _run_zimage_gguf8(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    return _run_zimage_gguf(root, "Q8_0")


def _run_ace(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "ace_setup.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_gfpgan(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "presets" / "extra_env" / "gfpgan_install.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_hunyuan15(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    # Note: matches install_menu.bat spelling: hunuyan15_install.bat
    script = root / "presets" / "extra_env" / "hunuyan15_install.bat"
    if not script.exists():
        return None
    return _cmd_call_bat(script, root)


def _run_sdxl_juggernaut(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "scripts" / "download_sd_models.py"
    py = _venv_python(root)
    if (not script.exists()) or (py is None):
        return None
    return (str(py), ["-u", str(script)], root / "scripts")


def _run_background_remover_inpainter(root: Path) -> Optional[Tuple[str, List[str], Path]]:
    script = root / "scripts" / "background_download.py"
    py = _venv_python(root)
    if (not script.exists()) or (py is None):
        return None
    # Run from root so the script can resolve relative paths consistently.
    return (str(py), ["-u", str(script)], root)


def _default_installs() -> List[OptionalInstall]:
    # Titles/descriptions copied from install_menu.bat "extra options" page.
    return [
        OptionalInstall(
            key="wan22",
            title="WAN 2.2 5B. Text/image/video to Video with extender",
            description="VRAM: 24GB recommended (offloading works with less, but very slow). Disk: +30GB.",
            runner=_run_wan22,
        ),
        OptionalInstall(
            key="hunyuan15",
            title="HunyuanVideo 1.5, text/image/video to Video with extender",
            description="VRAM: 16GB recommended for 480p. Disk: distilled I2V 480p included; up to ~35GiB more for extra models.",
            runner=_run_hunyuan15,
        ),
        OptionalInstall(
            key="zimage",
            title="Z-image Turbo Text to Image",
            description="VRAM: 16GB+ for best quality (FP16). If you have <16GB, use the GGUF options (Q4–Q8).",
            runner=_run_zimage,
        ),
        OptionalInstall(
            key="zimage_fp16",
            title="Full 16 FP model",
            description="VRAM: 16GB recommended. Disk: ~21GB download (diffusion fp16 + text encoder + VAE). Tip: try bf16 if fp16 shows black images.",
            runner=_run_zimage_fp16,
        ),
        OptionalInstall(
            key="zimage_gguf4",
            title="Z-Image Turbo GGUF (Q4_0)",
            description="VRAM: ~6GB+. Smallest/fastest. Installs diffusion Q4_0 + matching Qwen3 text encoder.",
            runner=_run_zimage_gguf4,
        ),
        OptionalInstall(
            key="zimage_gguf5",
            title="Z-Image Turbo GGUF (Q5_0)",
            description="VRAM: ~8GB+. Balanced size/quality. Installs diffusion Q5_0 + matching Qwen3 text encoder.",
            runner=_run_zimage_gguf5,
        ),
        OptionalInstall(
            key="zimage_gguf6",
            title="Z-Image Turbo GGUF (Q6_K)",
            description="VRAM: ~10–12GB+. Higher quality. Installs diffusion Q6_K + matching Qwen3 text encoder.",
            runner=_run_zimage_gguf6,
        ),
        OptionalInstall(
            key="zimage_gguf8",
            title="Z-Image Turbo GGUF (Q8_0)",
            description="VRAM: ~12–16GB+. Best quality / largest. Installs diffusion Q8_0 + matching Qwen3 text encoder.",
            runner=_run_zimage_gguf8,
        ),


        OptionalInstall(
            key="ace",
            title="Ace step Music Creation",
            description="VRAM: 6GB+ recommended for speed (runs on any machine). Disk: ~6GiB. Hit-or-miss results—batch a bunch and keep the best.",
            runner=_run_ace,
        ),
        OptionalInstall(
            key="gfpgan",
            title="GFPGAN Face restorer/enhancer",
            description="VRAM: optional (CPU works). Download <400MB; env uses ~5GiB disk.",
            runner=_run_gfpgan,
        ),
                OptionalInstall(
            key="bgrem_inpaint",
            title="Background remover and inpainter",
            description="VRAM: 4–6GB recommended. Downloads 2 background removers + an SD1.5 inpaint base model (~5GB disk).",
            runner=_run_background_remover_inpainter,
        ),
OptionalInstall(
            key="sdxljugg",
            title="Juggernaut XL V9. Model for SDXL Text to image",
            description="VRAM: 6–12GB. Disk: ~6.5GB. SDXL model for txt2img. find more (CyberRealisticXL, DreamshaperXL,EpicRealismXL,...) at https://civitai.com/",
            runner=_run_sdxl_juggernaut,
        ),
    ]


# -----------------------------
# Environment folders (optional installs)
# -----------------------------
# These folders are safe to delete when re-installing an optional component:
# they contain only the Python environment, not the downloaded model weights.
_ENV_DIR_BY_KEY = {
    "ace": Path("presets") / "extra_env" / ".ace_env",
    "hunyuan15": Path(".hunyuan15_env"),
    "gfpgan": Path("models") / "gfpgan" / ".GFPGAN",
    "wan22": Path(".wan_venv"),
    "zimage": Path(".zimage_env"),
    # Not on the UI list yet, but reserved for future use.
    "comfui": Path(".comfui_env"),
}

# -----------------------------
# UI
# -----------------------------

class _OptionRow(QtWidgets.QWidget):
    toggled = QtCore.Signal(bool)

    def __init__(self, opt: OptionalInstall, parent: Optional[QtWidgets.QWidget] = None, indent: int = 0) -> None:
        super().__init__(parent)
        self.opt = opt

        self.checkbox = QtWidgets.QCheckBox()
        self.checkbox.stateChanged.connect(lambda s: self.toggled.emit(s == QtCore.Qt.Checked))

        title = QtWidgets.QLabel(opt.title)
        f = title.font()
        f.setBold(True)
        title.setFont(f)

        desc = QtWidgets.QLabel(opt.description)
        desc.setWordWrap(True)
        desc.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        left = QtWidgets.QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(2)
        left.addWidget(title)
        left.addWidget(desc)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10 + max(0, int(indent)), 8, 10, 8)
        lay.setSpacing(10)
        lay.addWidget(self.checkbox, 0, QtCore.Qt.AlignTop)
        lay.addLayout(left, 1)

        self.setObjectName("OptionalInstallRow")
        self.setStyleSheet("""
        QWidget#OptionalInstallRow {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
        }
        """)

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)


class _ToastWidget(QtWidgets.QFrame):
    """Lightweight, non-modal toast popup shown inside the dialog."""

    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("ToastWidget")
        self.setVisible(False)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self._lbl = QtWidgets.QLabel("")
        self._lbl.setWordWrap(True)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.addWidget(self._lbl)

        self.setStyleSheet("""
        QFrame#ToastWidget {
            background: rgba(0, 0, 0, 0.82);
            color: white;
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 12px;
        }
        """)

    def show_message(self, msg: str, msec: int = 2800) -> None:
        self._lbl.setText(msg)
        self.adjustSize()

        # Bottom-right corner with padding.
        parent = self.parentWidget()
        if parent is not None:
            pad = 18
            x = max(pad, parent.width() - self.width() - pad)
            y = max(pad, parent.height() - self.height() - pad)
            self.move(x, y)

        self.setVisible(True)
        self.raise_()
        QtCore.QTimer.singleShot(max(800, int(msec)), self.hide)



class OptionalInstallsDialog(QtWidgets.QDialog):
    """
    Standalone dialog that runs selected optional installs sequentially and shows live logs.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, root_dir: Optional[Path] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optional installs")
        self.setMinimumSize(820, 560)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        self.root_dir: Path = (root_dir or _root_from_this_file()).resolve()
        self.installs: List[OptionalInstall] = _default_installs()

        self._process: Optional[QtCore.QProcess] = None
        self._queue: List[OptionalInstall] = []
        self._running: Optional[OptionalInstall] = None

        # Log handling
        self._log_fp = None  # type: ignore
        self._log_file_path: Optional[Path] = None

        # Auto-continue / pause handling
        self._pause_key_sent: bool = False


        # Header
        title = QtWidgets.QLabel("Optional installs")
        tf = title.font()
        tf.setPointSize(tf.pointSize() + 6)
        tf.setBold(True)
        title.setFont(tf)

        subtitle = QtWidgets.QLabel(
            "Select extra installs for FrameVision. When you press Start, installs run one-by-one and the log appears below."
        )
        subtitle.setWordWrap(True)

        # Header window controls (minimize / maximize / close)
        header_row = QtWidgets.QWidget()
        header_lay = QtWidgets.QHBoxLayout(header_row)
        header_lay.setContentsMargins(0, 0, 0, 0)
        header_lay.setSpacing(8)

        header_lay.addWidget(title)
        header_lay.addStretch(1)

        def _mk_hdr_btn(std_icon: QtWidgets.QStyle.StandardPixmap, tooltip: str) -> QtWidgets.QPushButton:
            btn = QtWidgets.QPushButton()
            btn.setFlat(True)
            btn.setToolTip(tooltip)
            btn.setIcon(self.style().standardIcon(std_icon))
            btn.setFixedSize(34, 28)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFocusPolicy(QtCore.Qt.NoFocus)
            return btn

        self._btn_min = _mk_hdr_btn(QtWidgets.QStyle.SP_TitleBarMinButton, "Minimize")
        self._btn_max = _mk_hdr_btn(QtWidgets.QStyle.SP_TitleBarMaxButton, "Maximize / Restore")
        self._btn_x = _mk_hdr_btn(QtWidgets.QStyle.SP_TitleBarCloseButton, "Close")

        self._btn_min.clicked.connect(self.showMinimized)
        self._btn_max.clicked.connect(lambda: self.showNormal() if self.isMaximized() else self.showMaximized())
        self._btn_x.clicked.connect(self.close)

        header_lay.addWidget(self._btn_min, 0, QtCore.Qt.AlignTop)
        header_lay.addWidget(self._btn_max, 0, QtCore.Qt.AlignTop)
        header_lay.addWidget(self._btn_x, 0, QtCore.Qt.AlignTop)

        # Light hover styling; close button gets a stronger hover
        header_row.setStyleSheet("""
            QPushButton { border: none; background: transparent; border-radius: 6px; }
            QPushButton:hover { background: rgba(255,255,255,0.08); }
            QPushButton:pressed { background: rgba(255,255,255,0.14); }
        """)
        self._btn_x.setStyleSheet("""
            QPushButton { border: none; background: transparent; border-radius: 6px; }
            QPushButton:hover { background: rgba(255, 60, 60, 0.55); }
            QPushButton:pressed { background: rgba(255, 60, 60, 0.75); }
        """)

        # Options list
        self.rows: List[_OptionRow] = []
        opts_box = QtWidgets.QWidget()
        opts_lay = QtWidgets.QVBoxLayout(opts_box)
        opts_lay.setContentsMargins(0, 0, 0, 0)
        opts_lay.setSpacing(10)

        by_key = {i.key: i for i in self.installs}
        row_by_key: Dict[str, _OptionRow] = {}

        for opt in self.installs:
            # Indent *all* Z-image model download options consistently (avoid alternating rows).
            is_zimage_extra = opt.key.startswith("zimage_") and opt.key != "zimage"
            row = _OptionRow(opt, indent=(26 if is_zimage_extra else 0))
            if opt.key.startswith("zimage_") and opt.key != "zimage":
                row.toggled.connect(lambda checked, k=opt.key: self._on_zimage_model_toggled(k, checked))
            self.rows.append(row)
            row_by_key[opt.key] = row
            opts_lay.addWidget(row)

        self._row_by_key = row_by_key

        opts_lay.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(opts_box)

        # Log + progress
        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setWordWrap(True)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Install log will appear here...")

        # Log tools (below progress bar)
        self.auto_continue_chk = QtWidgets.QCheckBox('Auto-continue when installer says "Press any key to continue"')
        self.auto_continue_chk.setChecked(True)
        self.auto_continue_chk.setToolTip(
            "Some .bat installers call PAUSE, which can hang in this embedded log.\n"
            "When enabled, FrameVision will try to auto-send Enter to continue."
        )

        self.save_log_chk = QtWidgets.QCheckBox("Save log to file")
        self.save_log_chk.setChecked(False)
        self.save_log_chk.setToolTip("Write the full optional-installs log to a file in <root>/logs/.")

        self.open_logs_btn = QtWidgets.QPushButton("Open logs folder")
        self.open_logs_btn.setToolTip("Open the FrameVision logs folder.")
        self.open_logs_btn.clicked.connect(self._open_logs_folder)

        log_tools = QtWidgets.QHBoxLayout()
        log_tools.setContentsMargins(0, 0, 0, 0)
        log_tools.setSpacing(10)
        log_tools.addWidget(self.auto_continue_chk)
        log_tools.addWidget(self.save_log_chk)
        log_tools.addStretch(1)
        log_tools.addWidget(self.open_logs_btn)

        # Buttons
        self.start_btn = QtWidgets.QPushButton("Start optional installs")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.continue_btn = QtWidgets.QPushButton("Send Enter")
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.setEnabled(False)

        self.continue_btn.setEnabled(False)
        self.continue_btn.setToolTip("If an installer shows 'Press any key to continue', click this to continue.")
        self.continue_btn.clicked.connect(self._send_enter_to_process)

        self.start_btn.clicked.connect(self._on_start)
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.close_btn.clicked.connect(self.accept)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.start_btn)
        btns.addStretch(1)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.continue_btn)
        btns.addWidget(self.close_btn)

        # Layout
        top = QtWidgets.QVBoxLayout(self)
        top.setContentsMargins(14, 14, 14, 14)
        top.setSpacing(10)
        top.addWidget(header_row)
        top.addWidget(subtitle)
        top.addWidget(scroll, 2)
        top.addWidget(self.status_lbl)
        top.addWidget(self.progress)
        top.addLayout(log_tools)
        top.addWidget(self.log, 1)
        top.addLayout(btns)

        # Nice default font size for log
        lf = self.log.font()
        lf.setPointSize(max(9, lf.pointSize()))
        self.log.setFont(lf)

        # Toast popup (non-modal)
        self._toast_widget = _ToastWidget(self)

        self._append_line(f"Root: {self.root_dir}")

    # ---- logging helpers

    def _append_line(self, s: str) -> None:
        self.log.appendPlainText(s)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

        # Optional: mirror log output to a file.
        try:
            fp = getattr(self, "_log_fp", None)
            if fp is not None:
                fp.write(s + "\n")
                fp.flush()
        except Exception:
            pass


    def _toast(self, msg: str, msec: int = 2800) -> None:
        """Show a small, non-blocking toast message."""
        try:
            tw = getattr(self, "_toast_widget", None)
            if tw is not None:
                tw.show_message(msg, msec=msec)
        except Exception:
            # Toasts should never crash the installer UI.
            pass


    def _logs_dir(self) -> Path:
        return (self.root_dir / "logs").resolve()

    def _open_logs_folder(self) -> None:
        try:
            p = self._logs_dir()
            p.mkdir(parents=True, exist_ok=True)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(p)))
        except Exception:
            # Best-effort only.
            pass

    def _start_log_file_if_enabled(self) -> None:
        # Called at the start of a run.
        self._close_log_file()
        try:
            if not getattr(self, "save_log_chk", None) or (not self.save_log_chk.isChecked()):
                return
            logs_dir = self._logs_dir()
            logs_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._log_file_path = logs_dir / f"optional_installs_{ts}.log"
            self._log_fp = open(self._log_file_path, "w", encoding="utf-8", errors="replace")
            self._append_line(f"[INFO] Saving log to: {self._log_file_path}")
        except Exception:
            # If log file fails, continue without it.
            self._log_fp = None
            self._log_file_path = None

    def _close_log_file(self) -> None:
        try:
            fp = getattr(self, "_log_fp", None)
            if fp is not None:
                fp.flush()
                fp.close()
        except Exception:
            pass
        self._log_fp = None
        self._log_file_path = None

    def _send_enter_to_process(self) -> None:
        """Manually send Enter to the running process (useful for PAUSE)."""
        if self._process is None:
            return
        try:
            self._process.write(b"\r\n")
            self._process.waitForBytesWritten(250)
            self._append_line("[INFO] Sent Enter to installer input.")
            self._toast("Sent Enter to continue…")
        except Exception:
            pass

    def _maybe_auto_continue(self, text_line: str) -> None:
        """Detect common PAUSE prompts and auto-continue if enabled."""
        try:
            if self._pause_key_sent:
                return
            if not self.auto_continue_chk.isChecked():
                return
            s = (text_line or "").lower()
            if "press any key to continue" in s:
                self._pause_key_sent = True
                self._append_line("[AUTO] Detected PAUSE prompt — sending Enter…")
                try:
                    if self._process is not None:
                        self._process.write(b"\r\n")
                        self._process.waitForBytesWritten(250)
                except Exception:
                    pass
        except Exception:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """If an installer is running, confirm before closing so we don't leave background processes."""
        try:
            running = (self._process is not None and self._process.state() != QtCore.QProcess.NotRunning) or bool(self._queue)
        except Exception:
            running = False

        if running:
            mb = QtWidgets.QMessageBox(self)
            mb.setIcon(QtWidgets.QMessageBox.Warning)
            mb.setWindowTitle("Installer running")
            mb.setText("An optional install is still running.")
            mb.setInformativeText("Do you want to cancel the install and close this window?")
            btn_keep = mb.addButton("Keep running", QtWidgets.QMessageBox.RejectRole)
            btn_close = mb.addButton("Cancel & close", QtWidgets.QMessageBox.AcceptRole)
            mb.setDefaultButton(btn_keep)
            mb.exec()

            if mb.clickedButton() == btn_keep:
                event.ignore()
                return

            # Cancel the run, then close.
            self._on_cancel()

        super().closeEvent(event)


    # ---- env folder safety (optional installs)

    def _env_dir_for(self, opt: OptionalInstall) -> Optional[Path]:
        rel = _ENV_DIR_BY_KEY.get(opt.key)
        if rel is None:
            return None
        return (self.root_dir / rel).resolve()

    def _rmtree_safe(self, p: Path) -> Tuple[bool, str]:
        """
        Robust recursive delete for Windows (handles read-only files).
        """
        try:
            if not p.exists():
                return True, ""
            if p.is_file():
                p.unlink()
                return True, ""
            def _onerror(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    raise
            shutil.rmtree(p, onerror=_onerror)
            return True, ""
        except Exception as e:
            return False, str(e)

    def _maybe_reset_env(self, opt: OptionalInstall) -> str:
        """Handle existing optional-install environments.

        Returns:
            "proceed"   -> continue and run the installer as normal
            "keep"      -> user chose to keep the existing env folder
            "reinstall" -> env folder was deleted; proceed to reinstall
            "cancel"    -> user cancelled (queue cleared / UI updated)
        """
        env_dir = self._env_dir_for(opt)
        if env_dir is None:
            return "proceed"
        if not env_dir.exists():
            return "proceed"

        # Pause and warn user.
        mb = QtWidgets.QMessageBox(self)
        mb.setIcon(QtWidgets.QMessageBox.Warning)
        mb.setWindowTitle("Existing environment found")
        mb.setText(f"An existing environment folder was found for:\n\n{opt.title}")
        mb.setInformativeText(
            "FrameVision found an existing environment folder for this component.\n\n"
            "Choose what to do:\n"
            "• Continue without reinstalling environment: keep the env and proceed (fastest).\n"
            "• Delete and reinstall environment: deletes ONLY the env folder below, then reinstalls it.\n"
            "Downloaded models/weights will not be removed.\n\n"
            f"Environment folder:\n{env_dir}"
        )
        btn_cancel = mb.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        btn_keep = mb.addButton("Continue without reinstalling environment", QtWidgets.QMessageBox.AcceptRole)
        btn_reinstall = mb.addButton("Delete and reinstall environment", QtWidgets.QMessageBox.DestructiveRole)
        mb.setDefaultButton(btn_keep)
        mb.exec()

        if mb.clickedButton() == btn_cancel:
            self._append_line("")
            self._append_line("[CANCEL] User cancelled (existing env found).")
            self._queue.clear()
            self._running = None
            self.status_lbl.setText("Cancelled.")
            self._set_running_ui(False)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            return "cancel"

        if mb.clickedButton() == btn_keep:
            self._append_line("")
            self._append_line("[INFO] Keeping existing env folder (no reinstall).")
            self._toast(f"Keeping existing environment: {opt.title}")
            return "keep"

        # User chose delete + reinstall: delete environment folder and proceed.
        self._append_line("")
        self._toast(f"Deleting existing environment to reinstall: {opt.title}")

        self._append_line(f"[INFO] Deleting existing env folder: {env_dir}")
        ok, err = self._rmtree_safe(env_dir)
        if not ok:
            self._append_line(f"[ERROR] Failed to delete env folder: {err}")
            QtWidgets.QMessageBox.critical(
                self,
                "Could not delete environment folder",
                f"Failed to delete:\n{env_dir}\n\nError:\n{err}",
            )
            # Safer to cancel the whole run than continue in a half-broken state.
            self._queue.clear()
            self._running = None
            self.status_lbl.setText("Cancelled (delete failed).")
            self._set_running_ui(False)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            return "cancel"

        self._append_line("[OK] Existing env folder deleted.")
        self._toast(f"Environment deleted. Reinstalling: {opt.title}")
        return "reinstall"

    # ---- selection helpers



    def _on_zimage_model_toggled(self, key: str, checked: bool) -> None:
        """Keep Z-image model selections sane and warn when prerequisites are missing."""
        if not checked:
            return

        # Make GGUF quants mutually exclusive to avoid accidentally downloading 2–4 large models.
        if key.startswith("zimage_gguf"):
            try:
                for k, row in getattr(self, "_row_by_key", {}).items():
                    if k.startswith("zimage_gguf") and k != key and row.is_checked():
                        row.set_checked(False)
            except Exception:
                pass

        # Prereqs for the model downloader.
        script = self.root_dir / "presets" / "extra_env" / "zimage_gguf_install.py"
        if not script.exists():
            self._toast("Missing installer: presets/extra_env/zimage_gguf_install.py")
            return

        py = _venv_python(self.root_dir)
        if py is None or (not py.exists()):
            self._toast("Python .venv not found. Run the main installer first so FrameVision creates .venv.")
            return

        # Important: GGUF + safetensors downloads run inside FrameVision's .venv and do NOT require rebuilding the Z-image env.
        if key.startswith("zimage_gguf"):
            self._toast("GGUF model will download when you press Start (no env reinstall needed).")
        elif key == "zimage_fp16":
            self._toast("Full FP16 model will download when you press Start (no env reinstall needed).")


    def selected_installs(self) -> List[OptionalInstall]:
        # Preserve install order from self.installs.
        checked_keys = [row.opt.key for row in self.rows if row.is_checked()]
        checked_set = set(checked_keys)

        ordered: List[OptionalInstall] = []
        for opt in self.installs:
            if opt.key in checked_set:
                ordered.append(opt)

        # Z-image is a bit fragile for first-time users: if someone selects a model download
        # (GGUF / FP16) but does NOT have the Z-image runtime env yet, install the env first.
        # This is silent: we only auto-add the env step when it is missing, so no "existing env"
        # prompt will appear.
        self._auto_added_zimage_env = False
        try:
            wants_zimage_models = any(k.startswith("zimage_") and k != "zimage" for k in checked_set)
            if wants_zimage_models and ("zimage" not in checked_set):
                env_rel = _ENV_DIR_BY_KEY.get("zimage")
                env_dir = (self.root_dir / env_rel).resolve() if env_rel is not None else None
                if env_dir is not None and (not env_dir.exists()):
                    for opt in self.installs:
                        if opt.key == "zimage":
                            ordered.insert(0, opt)
                            self._auto_added_zimage_env = True
                            break
        except Exception:
            pass

        return ordered

    # ---- run control

    def _set_running_ui(self, running: bool) -> None:
        for r in self.rows:
            r.setEnabled(not running)
        self.start_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(True)
        # Enable manual "Send Enter" only while a process is expected to be running.
        try:
            self.continue_btn.setEnabled(running)
        except Exception:
            pass
        self.close_btn.setEnabled(not running and (self._running is None) and (not self._queue))

    def _on_start(self) -> None:
        selected = self.selected_installs()
        if not selected:
            QtWidgets.QMessageBox.information(self, "Optional installs", "No optional installs selected.")
            return

        self._queue = list(selected)
        self._running = None
        self._pause_key_sent = False
        self._start_log_file_if_enabled()
        self._append_line("")
        self._append_line("=== Starting optional installs ===")
        if getattr(self, "_auto_added_zimage_env", False):
            self._append_line("[INFO] Z-image env not found — installing it first so Z-image can run after model download.")

        self.progress.setRange(0, len(selected))
        self.progress.setValue(0)

        self._set_running_ui(True)
        self._run_next()

    def _on_cancel(self) -> None:
        # If currently running, attempt to terminate.
        if self._process is not None and self._process.state() != QtCore.QProcess.NotRunning:
            self._append_line("")
            self._append_line("[CANCEL] Terminating current installer...")
            self._process.kill()
            self._process = None

        # If we were mid-queue, clear it.
        self._queue.clear()
        self._running = None
        self.status_lbl.setText("Cancelled.")
        self._set_running_ui(False)
        self.close_btn.setEnabled(True)
        self._close_log_file()

    def _run_next(self) -> None:
        if not self._queue:
            self._append_line("")
            self._append_line("=== Optional installs completed ===")
            self._close_log_file()
            self.status_lbl.setText("Optional installs completed.")
            self._set_running_ui(False)
            self.cancel_btn.setEnabled(False)
            self.close_btn.setEnabled(True)
            return

        self._running = self._queue.pop(0)
        idx_done = self.progress.value()
        self.status_lbl.setText(f"Running: {self._running.title}")
        self._append_line("")
        self._append_line(f"[RUN] {self._running.title}")

        cmd = self._running.runner(self.root_dir)
        if cmd is None:
            self._append_line(f"[WARN] Missing installer/script for: {self._running.title}")
            self.progress.setValue(idx_done + 1)
            self._run_next()
            return

        decision = self._maybe_reset_env(self._running)
        if decision == "cancel":
            return

        # Special case:
        # If the user chooses to keep an existing Z-image env, do NOT re-run zimage_install.bat.
        # That .bat tends to reinstall anyway; keeping means we should skip straight to the next step.
        if decision == "keep" and self._running.key == "zimage":
            self._append_line("[INFO] Skipping Z-image environment install (keeping existing env).")
            self._toast("Z-image env kept. Continuing to downloads…")
            self.progress.setValue(idx_done + 1)
            self._running = None
            self._run_next()
            return

        program, args, cwd = cmd

        # QProcess setup
        self._pause_key_sent = False
        proc = QtCore.QProcess(self)
        self._process = proc
        proc.setProgram(program)
        proc.setArguments(args)
        proc.setWorkingDirectory(str(cwd))

        # Set ROOT env var like install_menu.bat does.
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("ROOT", str(self.root_dir) + (os.sep if not str(self.root_dir).endswith(os.sep) else ""))
        proc.setProcessEnvironment(env)

        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        proc.readyReadStandardOutput.connect(self._on_proc_output)
        proc.readyReadStandardError.connect(self._on_proc_output)
        proc.finished.connect(self._on_proc_finished)
        proc.errorOccurred.connect(self._on_proc_error)

        # Start
        proc.start()

        if not proc.waitForStarted(3000):
            self._append_line(f"[ERROR] Failed to start: {program}")
            self.progress.setValue(idx_done + 1)
            self._run_next()

    def _on_proc_output(self) -> None:
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        if data:
            # Keep original line breaks
            for line in data.splitlines():
                self._append_line(line)
                self._maybe_auto_continue(line)

    def _on_proc_error(self, err: QtCore.QProcess.ProcessError) -> None:
        self._append_line(f"[ERROR] Process error: {err}")

    def _on_proc_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        done_count = self.progress.value() + 1
        self.progress.setValue(done_count)

        if exit_status != QtCore.QProcess.NormalExit:
            self._append_line(f"[ERROR] Installer crashed (exit_code={exit_code}).")
        elif exit_code != 0:
            self._append_line(f"[WARN] Installer finished with exit_code={exit_code}.")
        else:
            self._append_line("[OK] Finished.")

        self._process = None
        self._running = None
        self._run_next()


# -----------------------------
# Public API (import friendly)
# -----------------------------

def show_optional_installs(parent: Optional[QtWidgets.QWidget] = None, root_dir: Optional[str] = None) -> int:
    """
    Import-friendly entry point.

    Returns QDialog.exec() result.
    """
    root_path = Path(root_dir).resolve() if root_dir else None

    # If FrameVision already has a QApplication, don't create another.
    app = QtWidgets.QApplication.instance()
    created_app = False
    if app is None:
        created_app = True
        app = QtWidgets.QApplication(sys.argv)

    dlg = OptionalInstallsDialog(parent=parent, root_dir=root_path)

    # If we're running inside FrameVision already, show modeless so the main app stays usable.
    if not created_app:
        dlg.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        return 0

    # Standalone testing: run as a modal dialog.
    res = dlg.exec()

    # If we created the app, close it cleanly.
    app.quit()
    return res


def main() -> None:
    show_optional_installs(parent=None, root_dir=None)


if __name__ == "__main__":
    main()
