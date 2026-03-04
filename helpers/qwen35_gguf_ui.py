# -*- coding: utf-8 -*-
"""
FrameVision - Qwen3.5 GGUF UI (PySide6)

What this is:
- A self-contained PySide6 widget (and runnable window) to select a Qwen3.5 GGUF model + quant file
  from FrameVision's models folder and run it via a local CLI binary (default: presets/bin/sd-cli.exe).

Auto-discovery:
- Models are discovered from: <FrameVisionRoot>/models/qwen35_gguf/<ModelFolder>/
  Example: models/qwen35_gguf/4B/*.gguf, models/qwen35_gguf/9B/*.gguf, models/qwen35_gguf/2B/*.gguf ...
- "mmproj-*.gguf" files are excluded from the main model list (kept available separately if needed later).

Notes:
- This UI does not assume a specific CLI argument schema beyond requiring a "model" file.
  Different gguf runners use different flags. To stay robust, you can:
    1) choose a "Runner Preset" (simple templates), or
    2) use "Extra args" to override/extend the command.
- Default runner points to presets/bin/sd-cli.exe because that's what you requested.
  If your runner is actually llama.cpp (llama-cli.exe), you can browse to it or change the path.

Integration:
- Drop this file in /helpers/ and import Qwen35Pane where you want.
- You can also run it directly: python helpers/qwen35_gguf_ui.py

"""

from __future__ import annotations

import os
import shlex
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


# -----------------------------
# Paths / discovery
# -----------------------------

def _framevision_root_from_helpers() -> str:
    """
    Expected file location:
        <root>/helpers/qwen35_gguf_ui.py
    """
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def _default_runner_path(fv_root: str) -> str:
    """
    Prefer an LLM GGUF runner (llama.cpp) if present, otherwise fall back.

    Search order:
      1) <root>/presets/bin/llama/**/llama-cli.exe (newest by folder mtime)
      2) <root>/presets/bin/llama-cli.exe
      3) <root>/presets/bin/sd-cli.exe  (image models; NOT suitable for LLM GGUFs)
    """
    llama_root = os.path.join(fv_root, "presets", "bin", "llama")
    candidates = []
    if os.path.isdir(llama_root):
        for dirpath, dirnames, filenames in os.walk(llama_root):
            for fn in filenames:
                if fn.lower() == "llama-cli.exe":
                    candidates.append(os.path.join(dirpath, fn))
    # newest installed first (by parent folder mtime)
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(os.path.dirname(p)), reverse=True)
        return candidates[0]

    direct = os.path.join(fv_root, "presets", "bin", "llama-cli.exe")
    if os.path.exists(direct):
        return direct

    sdcli = os.path.join(fv_root, "presets", "bin", "sd-cli.exe")
    return sdcli


def _models_root(fv_root: str) -> str:
    return os.path.join(fv_root, "models", "qwen35_gguf")


def _is_gguf(fn: str) -> bool:
    return fn.lower().endswith(".gguf")


def _is_mmproj(fn: str) -> bool:
    return os.path.basename(fn).lower().startswith("mmproj-") and fn.lower().endswith(".gguf")


def discover_model_folders(models_root: str) -> List[str]:
    if not os.path.isdir(models_root):
        return []
    out = []
    for name in os.listdir(models_root):
        p = os.path.join(models_root, name)
        if os.path.isdir(p):
            out.append(name)
    # keep "4B/9B" near top if present, then others
    def sort_key(x: str) -> Tuple[int, str]:
        xl = x.lower()
        if xl == "4b":
            return (0, xl)
        if xl == "9b":
            return (1, xl)
        return (2, xl)
    out.sort(key=sort_key)
    return out


def discover_ggufs_in_folder(folder_path: str) -> Tuple[List[str], List[str]]:
    """
    Returns (model_ggufs, mmproj_ggufs) as basenames sorted.
    """
    if not os.path.isdir(folder_path):
        return ([], [])
    ggufs = []
    mmps = []
    for fn in os.listdir(folder_path):
        if not _is_gguf(fn):
            continue
        if _is_mmproj(fn):
            mmps.append(fn)
        else:
            ggufs.append(fn)
    ggufs.sort(key=lambda s: s.lower())
    mmps.sort(key=lambda s: s.lower())
    return ggufs, mmps


def infer_quant_tag(filename: str) -> str:
    """
    Best-effort quant tag inference for dropdown display.
    Examples:
      Qwen3.5-9B-Q4_K_M.gguf -> Q4_K_M
      Qwen3.5-4B-Q6_K.gguf -> Q6_K
      something-q8_0.gguf -> Q8_0
    """
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]

    # Common patterns in GGUF naming
    candidates = [
        "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
        "Q6_K", "Q8_0",
        "F16", "BF16",
        "IQ1_S", "IQ2_S", "IQ2_XS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS",
    ]
    upper = name.upper()
    for c in candidates:
        if c in upper:
            return c

    # fallback: last dash segment
    parts = name.split("-")
    if parts:
        tail = parts[-1].upper()
        if "Q" in tail or "F16" in tail or "BF16" in tail:
            return tail
    return "UNKNOWN"


# -----------------------------
# Runner presets (templates)
# -----------------------------

@dataclass
class RunnerPreset:
    name: str
    # command format (list of tokens) where {runner}, {model_path}, {prompt}, {extra} are placeholders
    # extra is already tokenized from the UI field.
    tokens: List[str]
    help: str


def runner_presets() -> List[RunnerPreset]:
    """
    A couple of common patterns.
    Since you requested sd-cli.exe, we include a generic preset that only passes the model path
    + prompt. You can adapt quickly using Extra args.
    """
    return [
        RunnerPreset(
            name="Generic (runner model prompt)",
            tokens=["{runner}", "-m", "{model_path}", "-p", "{prompt}", "{extra}"],
            help="Most llama.cpp-like CLIs use -m for model and -p for prompt. If your runner differs, use Extra args or custom.",
        ),
        RunnerPreset(
            name="Generic (runner model --prompt prompt)",
            tokens=["{runner}", "-m", "{model_path}", "--prompt", "{prompt}", "{extra}"],
            help="Alternate prompt flag style.",
        ),
        RunnerPreset(
            name="Custom (use Extra args only)",
            tokens=["{runner}", "{extra}"],
            help="You fully define flags in Extra args. Use {MODEL} and {PROMPT} in Extra args to inject values.",
        ),
    ]


def build_command(preset: RunnerPreset, runner: str, model_path: str, prompt: str, extra_args: str) -> List[str]:
    extra_tokens: List[str] = []
    if extra_args.strip():
        # allow placeholders inside extra args for Custom preset convenience
        extra_expanded = extra_args.replace("{MODEL}", model_path).replace("{PROMPT}", prompt)
        extra_tokens = shlex.split(extra_expanded, posix=False)

    cmd: List[str] = []
    for t in preset.tokens:
        if t == "{runner}":
            cmd.append(runner)
        elif t == "{model_path}":
            cmd.append(model_path)
        elif t == "{prompt}":
            cmd.append(prompt)
        elif t == "{extra}":
            cmd.extend(extra_tokens)
        else:
            cmd.append(t)
    # remove empty tokens
    cmd = [c for c in cmd if c is not None and str(c).strip() != ""]
    return cmd


# -----------------------------
# Collapsible panel helper
# -----------------------------

class CollapsibleSection(QtWidgets.QWidget):
    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None, collapsed: bool = True):
        super().__init__(parent)

        self.toggle = QtWidgets.QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(not collapsed)
        self.toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
        self.toggle.clicked.connect(self._on_toggle)

        self.content = QtWidgets.QFrame()
        self.content.setFrameShape(QtWidgets.QFrame.NoFrame)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

        self.content.setVisible(not collapsed)

    def _on_toggle(self):
        expanded = self.toggle.isChecked()
        self.toggle.setArrowType(QtCore.Qt.DownArrow if expanded else QtCore.Qt.RightArrow)
        self.content.setVisible(expanded)

    def setContentLayout(self, layout: QtWidgets.QLayout):
        # Clear previous
        old = self.content.layout()
        if old is not None:
            while old.count():
                item = old.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
            old.deleteLater()
        self.content.setLayout(layout)


# -----------------------------
# Prompt editor (Enter to send)
# -----------------------------

class PromptEdit(QtWidgets.QPlainTextEdit):
    sendRequested = QtCore.Signal(str)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Enter = send, Shift+Enter = newline
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                return super().keyPressEvent(event)
            text = self.toPlainText().strip()
            if text:
                self.sendRequested.emit(text)
                self.clear()
            event.accept()
            return
        return super().keyPressEvent(event)

# -----------------------------
# Main Pane
# -----------------------------

class Qwen35Pane(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.fv_root = _framevision_root_from_helpers()
        self.models_root = _models_root(self.fv_root)

        # --- top collapsible settings
        self.section = CollapsibleSection("Model & runner settings", collapsed=True)

        self.cmb_model_folder = QtWidgets.QComboBox()
        self.cmb_quant = QtWidgets.QComboBox()
        self.btn_refresh = QtWidgets.QPushButton("Refresh")

        self.ed_runner = QtWidgets.QLineEdit()
        self.ed_runner.setText(_default_runner_path(self.fv_root))
        self.btn_browse_runner = QtWidgets.QPushButton("Browse…")

        self.cmb_preset = QtWidgets.QComboBox()
        self.lbl_preset_help = QtWidgets.QLabel()
        self.lbl_preset_help.setWordWrap(True)
        self.lbl_preset_help.setStyleSheet("color: #999;")

        self.ed_extra_args = QtWidgets.QLineEdit()
        self.ed_extra_args.setPlaceholderText('Extra args (optional). For Custom preset, use {MODEL} and {PROMPT} placeholders.')

        # basic generation knobs (optional; passed via Extra args depending on runner)
        self.sp_temp = QtWidgets.QDoubleSpinBox()
        self.sp_temp.setRange(0.0, 2.0)
        self.sp_temp.setSingleStep(0.05)
        self.sp_temp.setValue(0.7)

        self.sp_top_p = QtWidgets.QDoubleSpinBox()
        self.sp_top_p.setRange(0.0, 1.0)
        self.sp_top_p.setSingleStep(0.05)
        self.sp_top_p.setValue(0.9)

        self.sp_ctx = QtWidgets.QSpinBox()
        self.sp_ctx.setRange(128, 262144)
        self.sp_ctx.setSingleStep(128)
        self.sp_ctx.setValue(8192)

        self.sp_threads = QtWidgets.QSpinBox()
        self.sp_threads.setRange(1, 256)
        self.sp_threads.setValue(max(1, (os.cpu_count() or 8) // 2))

        self.chk_use_knobs = QtWidgets.QCheckBox("Append basic knobs to Extra args")
        self.chk_use_knobs.setChecked(False)
        self.chk_use_knobs.setToolTip("If enabled, adds common flags to Extra args (works only if your runner supports them).")

        self._build_section_layout()

        # --- main prompt + run area
        self.ed_prompt = PromptEdit()
        self.ed_prompt.setPlaceholderText("Type your prompt here…")

        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        self.chk_interactive = QtWidgets.QCheckBox("Interactive chat (Enter to send)")
        self.chk_interactive.setChecked(True)
        self.chk_interactive.setToolTip("Keeps llama-cli running (conversation mode) and sends prompts via stdin. Default: -n 800, repeat penalty, reverse prompt stop. Disable for one-shot runs.")

        self.ed_cmd_preview = QtWidgets.QLineEdit()
        self.ed_cmd_preview.setReadOnly(True)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)

        self.proc: Optional[QtCore.QProcess] = None

        # Layout
        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(self.section)

        # Output first (chat-like), bigger
        main.addWidget(QtWidgets.QLabel("Answer / Output"))
        main.addWidget(self.log, 3)

        # Prompt at the bottom, smaller (about ~5 lines)
        main.addWidget(QtWidgets.QLabel("Prompt"))
        # Set a compact height; user can still scroll.
        fm = QtGui.QFontMetrics(self.ed_prompt.font())
        line_h = fm.lineSpacing()
        target_h = int(line_h * 5.5) + 14
        self.ed_prompt.setMinimumHeight(target_h)
        self.ed_prompt.setMaximumHeight(int(line_h * 8) + 20)

        main.addWidget(self.ed_prompt, 0)

        run_row = QtWidgets.QHBoxLayout()
        run_row.addWidget(self.btn_run)
        run_row.addWidget(self.btn_stop)
        run_row.addWidget(self.chk_interactive)
        run_row.addStretch(1)
        main.addLayout(run_row)

        # Command preview is still useful but should not dominate the chat layout
        main.addWidget(QtWidgets.QLabel("Command preview"))
        main.addWidget(self.ed_cmd_preview)

        # Wire up
        self.btn_refresh.clicked.connect(self.refresh)
        self.cmb_model_folder.currentIndexChanged.connect(self._on_model_folder_changed)
        self.cmb_quant.currentIndexChanged.connect(self._update_cmd_preview)
        self.ed_runner.textChanged.connect(self._update_cmd_preview)
        self.ed_extra_args.textChanged.connect(self._update_cmd_preview)
        self.ed_prompt.textChanged.connect(self._update_cmd_preview)
        self.cmb_preset.currentIndexChanged.connect(self._on_preset_changed)
        self.btn_browse_runner.clicked.connect(self._browse_runner)
        self.btn_run.clicked.connect(self._run)
        self.btn_stop.clicked.connect(self._stop)
        self.ed_prompt.sendRequested.connect(self._send_prompt)
        self.chk_interactive.toggled.connect(self._on_interactive_toggled)
        self.chk_use_knobs.toggled.connect(self._update_cmd_preview)
        self.sp_temp.valueChanged.connect(self._update_cmd_preview)
        self.sp_top_p.valueChanged.connect(self._update_cmd_preview)
        self.sp_ctx.valueChanged.connect(self._update_cmd_preview)
        self.sp_threads.valueChanged.connect(self._update_cmd_preview)

        # init
        self._populate_presets()
        self.refresh()

    def _build_section_layout(self):
        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(10, 8, 10, 8)
        r = 0

        grid.addWidget(QtWidgets.QLabel("Model folder"), r, 0)
        grid.addWidget(self.cmb_model_folder, r, 1)
        grid.addWidget(self.btn_refresh, r, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("GGUF quant"), r, 0)
        grid.addWidget(self.cmb_quant, r, 1, 1, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Runner exe"), r, 0)
        grid.addWidget(self.ed_runner, r, 1)
        grid.addWidget(self.btn_browse_runner, r, 2)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Runner preset"), r, 0)
        grid.addWidget(self.cmb_preset, r, 1, 1, 2)
        r += 1

        grid.addWidget(self.lbl_preset_help, r, 0, 1, 3)
        r += 1

        grid.addWidget(QtWidgets.QLabel("Extra args"), r, 0)
        grid.addWidget(self.ed_extra_args, r, 1, 1, 2)
        r += 1

        knobs_row = QtWidgets.QHBoxLayout()
        knobs_row.addWidget(QtWidgets.QLabel("temp"))
        knobs_row.addWidget(self.sp_temp)
        knobs_row.addSpacing(10)
        knobs_row.addWidget(QtWidgets.QLabel("top_p"))
        knobs_row.addWidget(self.sp_top_p)
        knobs_row.addSpacing(10)
        knobs_row.addWidget(QtWidgets.QLabel("ctx"))
        knobs_row.addWidget(self.sp_ctx)
        knobs_row.addSpacing(10)
        knobs_row.addWidget(QtWidgets.QLabel("threads"))
        knobs_row.addWidget(self.sp_threads)
        knobs_row.addStretch(1)

        grid.addWidget(self.chk_use_knobs, r, 0, 1, 3)
        r += 1
        grid.addLayout(knobs_row, r, 0, 1, 3)
        r += 1

        wrap = QtWidgets.QVBoxLayout()
        wrap.addLayout(grid)
        wrap.addStretch(1)
        self.section.setContentLayout(wrap)

    def _populate_presets(self):
        self._presets = runner_presets()
        self.cmb_preset.clear()
        for p in self._presets:
            self.cmb_preset.addItem(p.name)
        self.cmb_preset.setCurrentIndex(0)
        self._on_preset_changed()

    def refresh(self):
        self.cmb_model_folder.blockSignals(True)
        self.cmb_quant.blockSignals(True)
        try:
            self.cmb_model_folder.clear()
            folders = discover_model_folders(self.models_root)
            if not folders:
                self.cmb_model_folder.addItem("(no models found)")
                self.cmb_model_folder.setEnabled(False)
                self.cmb_quant.setEnabled(False)
            else:
                self.cmb_model_folder.setEnabled(True)
                self.cmb_quant.setEnabled(True)
                for f in folders:
                    self.cmb_model_folder.addItem(f)
            # trigger quant fill
            self._on_model_folder_changed()
        finally:
            self.cmb_model_folder.blockSignals(False)
            self.cmb_quant.blockSignals(False)
        self._update_cmd_preview()

    def _on_model_folder_changed(self):
        folder = self.cmb_model_folder.currentText()
        self.cmb_quant.clear()

        if not folder or folder.startswith("("):
            self.cmb_quant.setEnabled(False)
            self._update_cmd_preview()
            return

        self.cmb_quant.setEnabled(True)
        path = os.path.join(self.models_root, folder)
        ggufs, mmps = discover_ggufs_in_folder(path)

        # build quant mapping to show friendly labels while keeping filename
        self._quant_items: List[Tuple[str, str]] = []  # (label, filename)
        for fn in ggufs:
            tag = infer_quant_tag(fn)
            label = f"{tag}  —  {fn}"
            self._quant_items.append((label, fn))

        if not self._quant_items:
            self.cmb_quant.addItem("(no gguf files)")
            self.cmb_quant.setEnabled(False)
        else:
            for label, fn in self._quant_items:
                self.cmb_quant.addItem(label, userData=fn)
            self.cmb_quant.setEnabled(True)

        self._update_cmd_preview()

    def _on_preset_changed(self):
        idx = self.cmb_preset.currentIndex()
        if idx < 0 or idx >= len(self._presets):
            self.lbl_preset_help.setText("")
        else:
            self.lbl_preset_help.setText(self._presets[idx].help)
        self._update_cmd_preview()

    def _browse_runner(self):
        start = self.ed_runner.text().strip()
        start_dir = os.path.dirname(start) if start and os.path.exists(os.path.dirname(start)) else self.fv_root
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select runner exe", start_dir, "Executables (*.exe);;All files (*.*)")
        if fn:
            self.ed_runner.setText(fn)

    def _selected_model_path(self) -> Optional[str]:
        folder = self.cmb_model_folder.currentText()
        if not folder or folder.startswith("("):
            return None
        fn = self.cmb_quant.currentData()
        if not fn:
            # fallback to parsing label if no userData
            label = self.cmb_quant.currentText()
            if "—" in label:
                fn = label.split("—", 1)[1].strip()
        if not fn or fn.startswith("("):
            return None
        return os.path.join(self.models_root, folder, fn)

    def _effective_extra_args(self) -> str:
        extra = self.ed_extra_args.text().strip()
        if not self.chk_use_knobs.isChecked():
            return extra

        # Common flags (llama.cpp-style); harmless if user is using a compatible runner.
        # If not compatible, user can disable or edit.
        knobs = []
        knobs += ["--temp", str(self.sp_temp.value())]
        knobs += ["--top-p", str(self.sp_top_p.value())]
        knobs += ["--ctx-size", str(self.sp_ctx.value())]
        knobs += ["--threads", str(self.sp_threads.value())]

        knob_str = " ".join(shlex.quote(k) for k in knobs)
        if extra:
            return f"{extra} {knob_str}"
        return knob_str

    def _update_cmd_preview(self):
        runner = self.ed_runner.text().strip()
        model_path = self._selected_model_path()
        prompt = (self.ed_prompt.toPlainText() or "").strip()
        extra = self._effective_extra_args()

        if not runner or not model_path:
            self.ed_cmd_preview.setText("")
            return

        preset = self._presets[self.cmb_preset.currentIndex()] if 0 <= self.cmb_preset.currentIndex() < len(self._presets) else self._presets[0]
        cmd = build_command(preset, runner, model_path, prompt, extra)

        # display as a single string (Windows quoting)
        preview = " ".join(shlex.quote(c) for c in cmd)

        # Hint: sd-cli.exe is for Stable Diffusion GGUFs, not LLM GGUFs like Qwen3.5 text models.
        if os.path.basename(runner).lower() == "sd-cli.exe":
            self.ed_cmd_preview.setText(preview + "   [NOTE: sd-cli.exe cannot run LLM GGUFs; install/use llama-cli.exe]")
        else:
            self.ed_cmd_preview.setText(preview)
        return
    def _append_log(self, s: str):
        """
        Stream-safe log append.

        QProcess often emits tiny stdout chunks (sometimes a single token) without newlines.
        Using appendPlainText() would add a newline for every chunk, causing "one word per line".
        We instead insert raw text at the end and keep the view pinned to bottom.
        """
        if not s:
            return
        # Normalize Windows newlines
        s = s.replace("\r\n", "\n").replace("\r", "\n")

        cursor = self.log.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(s)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _on_interactive_toggled(self, on: bool):
        # If mode changes while running, stop so behavior is predictable
        if self.proc:
            self._append_log("\n[Mode changed — restarting runner]\n")
            self._stop(force_kill=True)
    def _ensure_process(self) -> bool:
        """Start runner if not running."""
        if self.proc and self.proc.state() != QtCore.QProcess.NotRunning:
            return True

        runner = self.ed_runner.text().strip()
        if not runner or not os.path.exists(runner):
            QtWidgets.QMessageBox.warning(self, "Runner not found", "Runner exe not found. Please browse to the correct exe.")
            return False

        model_path = self._selected_model_path()
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, "Model not found", "Selected GGUF file not found. Please refresh and select a valid model.")
            return False

        preset = self._presets[self.cmb_preset.currentIndex()] if 0 <= self.cmb_preset.currentIndex() < len(self._presets) else self._presets[0]
        extra = self._effective_extra_args()

        # Force interactive flags unless user already provided them
        add_flags = []
        lower_extra = " " + extra.lower() + " "
        if " -cnv " not in lower_extra and " --conversation " not in lower_extra:
            add_flags.append("-cnv")
        if " -p " not in lower_extra and " --prompt " not in lower_extra:
            add_flags += ["-p", ""]

        if add_flags:
            extra = (extra + " " + " ".join(shlex.quote(x) for x in add_flags)).strip()

        # Build command with empty prompt; we will send prompts via stdin
        cmd = build_command(preset, runner, model_path, "", extra)

        self._append_log(f"\n$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")

        self.proc = QtCore.QProcess(self)
        self.proc.setProgram(cmd[0])
        self.proc.setArguments(cmd[1:])
        self.proc.setWorkingDirectory(self.fv_root)
        self.proc.readyReadStandardOutput.connect(self._read_stdout)
        self.proc.readyReadStandardError.connect(self._read_stderr)
        self.proc.finished.connect(self._finished)

        self.proc.start()
        if not self.proc.waitForStarted(5000):
            self._append_log("ERROR: Failed to start process.\n")
            self.proc = None
            return False

        self.btn_stop.setEnabled(True)
        return True
    def _send_prompt(self, prompt_text: str):
        prompt_text = (prompt_text or "").strip()
        if not prompt_text:
            return

        if self.chk_interactive.isChecked():
            # Start runner if needed
            if not (self.proc and self.proc.state() != QtCore.QProcess.NotRunning):
                self._run()
                if not (self.proc and self.proc.state() != QtCore.QProcess.NotRunning):
                    return

            self._append_log(f"\nYou: {prompt_text}\n")
            try:
                self.proc.write((prompt_text + "\n").encode("utf-8"))
                self.proc.waitForBytesWritten(1000)
            except Exception:
                self._append_log("\n[ERROR: failed to write to runner stdin]\n")
            return

        # One-shot fallback: run it
        self.ed_prompt.setPlainText(prompt_text)
        self._run()

    def _build_llama_interactive_cmd(self) -> list[str]:
        runner = self.ed_runner.text().strip()
        model_path = self._selected_model_path()
        extra = (self._effective_extra_args() or "").strip()

        # Strip flags we control in interactive mode to avoid duplicates / confusing behavior
        tokens = shlex.split(extra, posix=False) if extra else []
        filtered: list[str] = []
        skip_next = False
        for t in tokens:
            if skip_next:
                skip_next = False
                continue
            tl = t.lower()

            # remove interactive/conversation/prompt flags (we set these)
            if tl in ("--interactive", "-cnv", "--conversation", "-p", "--prompt"):
                if tl in ("-p", "--prompt"):
                    skip_next = True
                continue

            filtered.append(t)

        # Base interactive command (conversation mode)
        cmd = [runner, "-m", model_path, "-cnv", "-p", ""] + filtered

        # Add safer defaults if the user didn't provide them in Extra args.
        # These prevent runaway generation and reduce repetition.
        lower = " " + " ".join(tokens).lower() + " "

        def has_any(*flags: str) -> bool:
            return any(f" {f} " in lower for f in flags)

        # Max tokens (n-predict). If omitted, some builds may run very long.
        if not has_any("-n", "--n-predict", "--max-tokens"):
            cmd += ["-n", "800"]

        # Repetition controls
        if not has_any("--repeat-penalty", "--presence-penalty", "--frequency-penalty"):
            cmd += ["--repeat-penalty", "1.10"]
        if not has_any("--repeat-last-n"):
            cmd += ["--repeat-last-n", "256"]

        # Sampling defaults (keep reasonable)
        if not has_any("--temp", "-t"):
            cmd += ["--temp", "0.7"]
        if not has_any("--top-p"):
            cmd += ["--top-p", "0.9"]

        # Stop when the model starts a new user turn (helps avoid looping)
        # Some builds use -r / --reverse-prompt.
        if not has_any("-r", "--reverse-prompt"):
            cmd += ["-r", "User:"]

        return cmd


    def _run(self):
        # Interactive mode: start runner once and keep it alive.
        if self.chk_interactive.isChecked():
            if self.proc and self.proc.state() != QtCore.QProcess.NotRunning:
                self._append_log("\n[Runner already running]\n")
                return

            runner = self.ed_runner.text().strip()
            if not runner or not os.path.exists(runner):
                QtWidgets.QMessageBox.warning(self, "Runner not found", "Runner exe not found. Please browse to the correct exe.")
                return

            model_path = self._selected_model_path()
            if not model_path or not os.path.exists(model_path):
                QtWidgets.QMessageBox.warning(self, "Model not found", "Selected GGUF file not found. Please refresh and select a valid model.")
                return

            cmd = self._build_llama_interactive_cmd()

            self.proc = QtCore.QProcess(self)
            self.proc.setProgram(cmd[0])
            self.proc.setArguments(cmd[1:])
            self.proc.setWorkingDirectory(self.fv_root)
            self.proc.readyReadStandardOutput.connect(self._read_stdout)
            self.proc.readyReadStandardError.connect(self._read_stderr)
            self.proc.finished.connect(self._finished)

            # show command once (clean)
            self._append_log("\n$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
            self._append_log("[Runner starting…]\n")

            self.btn_stop.setEnabled(True)
            self.proc.start()
            if not self.proc.waitForStarted(5000):
                self._append_log("ERROR: Failed to start process.\n")
                self.proc = None
                self.btn_stop.setEnabled(False)
                return

            self._append_log("[Runner ready — type a prompt and press Enter]\n")
            return

        # One-shot mode: existing behavior (use current prompt)
        runner = self.ed_runner.text().strip()
        if not runner or not os.path.exists(runner):
            QtWidgets.QMessageBox.warning(self, "Runner not found", "Runner exe not found. Please browse to the correct exe.")
            return

        model_path = self._selected_model_path()
        if not model_path or not os.path.exists(model_path):
            QtWidgets.QMessageBox.warning(self, "Model not found", "Selected GGUF file not found. Please refresh and select a valid model.")
            return

        prompt = (self.ed_prompt.toPlainText() or "").strip()
        if not prompt:
            QtWidgets.QMessageBox.information(self, "Prompt required", "Type a prompt first.")
            return

        preset = self._presets[self.cmb_preset.currentIndex()] if 0 <= self.cmb_preset.currentIndex() < len(self._presets) else self._presets[0]
        extra = self._effective_extra_args()
        cmd = build_command(preset, runner, model_path, prompt, extra)

        self._append_log(f"\nYou: {prompt}\n")
        self._append_log("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")

        if self.proc:
            self._stop(force_kill=True)

        self.proc = QtCore.QProcess(self)
        self.proc.setProgram(cmd[0])
        self.proc.setArguments(cmd[1:])
        self.proc.setWorkingDirectory(self.fv_root)
        self.proc.readyReadStandardOutput.connect(self._read_stdout)
        self.proc.readyReadStandardError.connect(self._read_stderr)
        self.proc.finished.connect(self._finished)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        self.proc.start()
        if not self.proc.waitForStarted(5000):
            self._append_log("ERROR: Failed to start process.\n")
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.proc = None

    def _stop(self, force_kill: bool = False):
        if not self.proc:
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return
        self._append_log("\n[Stopping…]\n")
        try:
            self.proc.kill()
        except Exception:
            pass
        self.proc = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _finished(self, code: int, status: QtCore.QProcess.ExitStatus):
        self._append_log(f"\n[Process finished] code={code} status={int(status)}\n")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.proc = None




    def _read_stdout(self):
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self._append_log(data)

    def _read_stderr(self):
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardError()).decode("utf-8", errors="replace")
        if data:
            self._append_log(data)

class _MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FrameVision - Qwen3.5 GGUF (UI)")
        self.resize(980, 720)
        self.setCentralWidget(Qwen35Pane(self))


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = _MainWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
