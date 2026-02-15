import os
import sys
import json
import time
import traceback
import subprocess
import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------
# FrameVision tool layout
# This file should live in: <root>/helpers/qwentts_ui.py
# Expected folders:
#   <root>/environments/.qwen3tts/
#   <root>/presets/setsave/        (settings/presets)
#   <root>/models/
#   <root>/output/audio/qwen3tts/
#
# FrameVision integration goals:
#  - UI runs in the host (FrameVision) Python process (no self-relaunch).
#  - NO model preloading in the UI process (no torch / qwen_tts imports here).
#  - All heavy work (downloads, torch, model load, generation) happens in a
#    one-shot subprocess using:
#      <root>/environments/.qwen3tts/Scripts/python.exe
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
ENV_PY = ROOT_DIR / "environments" / ".qwen3tts" / "Scripts" / "python.exe"

HELPERS_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output" / "audio" / "qwen3tts"
PRESETS_DIR = ROOT_DIR / "presets" / "setsave"

# Hugging Face repos we support in this UI
HF_MODELS = {
    "Tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Base (for Voice Clone)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

SUPPORTED_SPEAKERS_FALLBACK = ["aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian"]
SUPPORTED_LANGS_FALLBACK = ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German", "Spanish", "Portuguese", "Russian", "Italian", "Arabic"]


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def repo_to_local_dir(repo_id: str) -> str:
    return str(MODELS_DIR / repo_id.split("/")[-1])


DEFAULT_TOKENIZER_DIR = repo_to_local_dir(HF_MODELS["Tokenizer"])
DEFAULT_CUSTOMVOICE_DIR = repo_to_local_dir(HF_MODELS["CustomVoice"])
DEFAULT_BASE_DIR = repo_to_local_dir(HF_MODELS["Base (for Voice Clone)"])
DEFAULT_VOICEDESIGN_DIR = repo_to_local_dir(HF_MODELS["VoiceDesign"])


def safe_lower_speaker(s: str) -> str:
    return (s or "").strip().lower()


# -----------------------------
# Qt imports (lightweight)
# -----------------------------
from PySide6 import QtCore, QtWidgets  # noqa: E402


class CollapsibleSection(QtWidgets.QWidget):
    """A simple arrow-toggle section that can remember its collapsed state."""

    toggled = QtCore.Signal(bool)  # collapsed

    def __init__(self, title: str, content: QtWidgets.QWidget, collapsed: bool = True, parent=None):
        super().__init__(parent)
        self._content = content

        self._btn = QtWidgets.QToolButton()
        self._btn.setStyleSheet("QToolButton { border: none; font-weight: 600; }")
        self._btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._btn.setText(title)
        self._btn.setCheckable(True)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._btn, 1)
        header.addStretch(0)

        wrap = QtWidgets.QWidget()
        wrap.setLayout(header)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(wrap)
        lay.addWidget(self._content)

        self._btn.toggled.connect(self._on_toggled)
        self.setCollapsed(collapsed)

    def isCollapsed(self) -> bool:
        return not self._btn.isChecked()

    def setCollapsed(self, collapsed: bool):
        self._btn.blockSignals(True)
        self._btn.setChecked(not collapsed)
        self._btn.setArrowType(QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
        self._content.setVisible(not collapsed)
        self._btn.blockSignals(False)

    def _on_toggled(self, checked: bool):
        collapsed = not checked
        self._btn.setArrowType(QtCore.Qt.RightArrow if collapsed else QtCore.Qt.DownArrow)
        self._content.setVisible(checked)
        self.toggled.emit(collapsed)


# -----------------------------
# Preset dataclasses (UI only)
# -----------------------------
@dataclass
class CommonSettings:
    device: str = "auto"  # auto / cpu / cuda:0 ...
    dtype: str = "bfloat16"  # bfloat16 / float16 / float32
    attn_impl: str = "sdpa"  # sdpa / none / flash_attention_2
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 2048
    output_name: str = "qwen_tts"
    add_timestamp: bool = True


@dataclass
class CustomVoicePreset:
    text: str = "Hello from Qwen3 TTS."
    language: str = "English"
    speaker: str = "ryan"
    instruct: str = ""
    model_path: str = DEFAULT_CUSTOMVOICE_DIR
    tokenizer_path: str = DEFAULT_TOKENIZER_DIR
    common: CommonSettings = field(default_factory=CommonSettings)


@dataclass
class VoiceClonePreset:
    text: str = "Hello. This is a voice clone test."
    language: str = "English"
    ref_audio_path: str = ""
    ref_text: str = ""
    x_vector_only_mode: bool = False
    instruct: str = ""
    model_path: str = DEFAULT_BASE_DIR
    tokenizer_path: str = DEFAULT_TOKENIZER_DIR
    common: CommonSettings = field(default_factory=CommonSettings)


@dataclass
class VoiceDesignPreset:
    text: str = "Hello. This is a voice design test."
    language: str = "English"
    voice_description: str = "A calm, friendly voice, slightly smiling, clear pronunciation."
    model_path: str = DEFAULT_VOICEDESIGN_DIR
    tokenizer_path: str = DEFAULT_TOKENIZER_DIR
    common: CommonSettings = field(default_factory=CommonSettings)


# -----------------------------
# Subprocess bridge
# -----------------------------
def _env_ok() -> bool:
    return ENV_PY.exists()


def _subproc_cmd(task: str) -> list[str]:
    # Run this same file in "worker mode" under the qwen3tts environment python.
    return [str(ENV_PY), "-u", str(Path(__file__).resolve()), "--worker", "--task", task]


class EnvTaskWorker(QtCore.QObject):
    """
    Runs a task in the qwen3tts environment python.
    - Sends one JSON payload via stdin
    - Receives streaming log lines and a final JSON result
    """

    log = QtCore.Signal(str)
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, task: str, payload: dict):
        super().__init__()
        self.task = task
        self.payload = payload

    @QtCore.Slot()
    def run(self):
        if not _env_ok():
            self.failed.emit(f"Missing environment python: {ENV_PY}")
            return

        try:
            env = os.environ.copy()
            env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

            p = subprocess.Popen(
                _subproc_cmd(self.task),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT_DIR),
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            try:
                payload_str = json.dumps(self.payload, ensure_ascii=False)
            except Exception:
                payload_str = "{}"

            if p.stdin:
                p.stdin.write(payload_str)
                p.stdin.close()

            result = None
            # Stream output. Worker prints lines, and ends with: __RESULT__{json}
            if p.stdout:
                for line in p.stdout:
                    line = (line or "").rstrip("\n")
                    if not line:
                        continue
                    if line.startswith("__LOG__"):
                        self.log.emit(line[len("__LOG__"):].lstrip())
                    elif line.startswith("__RESULT__"):
                        js = line[len("__RESULT__"):].strip()
                        try:
                            result = json.loads(js)
                        except Exception:
                            result = {"ok": False, "error": f"Failed to parse result JSON: {js}"}
                    else:
                        # Pass through unknown lines
                        self.log.emit(line)

            rc = p.wait()
            if rc != 0 and (result is None):
                self.failed.emit(f"Worker exited with code {rc}.")
                return

            if result is None:
                result = {"ok": False, "error": "No result returned from worker."}

            if result.get("ok", False):
                self.done.emit(result)
            else:
                self.failed.emit(result.get("error", "Unknown error (worker returned ok=false)."))
        except Exception:
            self.failed.emit(traceback.format_exc())


# -----------------------------
# Main UI
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("Qwen3-TTS (FrameVision tool)")
        self.resize(1040, 800)

        # Autosave (debounced) to presets/setsave/qwentts.json
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(250)
        self._autosave_timer.timeout.connect(self._save_last_state)

        self._build_ui()
        self._wire_autosave_signals()
        self._load_last_state()
        self._update_model_status()
        self._update_env_status()
        self._refresh_gpu_info_async()

    # ---------- UI ----------
    def _build_ui(self):
        outer = QtWidgets.QWidget()
        self.setCentralWidget(outer)
        outer_layout = QtWidgets.QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        outer_layout.addWidget(scroll, 1)

        cw = QtWidgets.QWidget()
        scroll.setWidget(cw)
        layout = QtWidgets.QVBoxLayout(cw)

        # Environment status
        env_content = QtWidgets.QWidget()
        env_lay = QtWidgets.QHBoxLayout(env_content)
        env_lay.setContentsMargins(0, 0, 0, 0)
        self.lbl_env = QtWidgets.QLabel("")
        self.btn_env_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_env_refresh.clicked.connect(self._update_env_status)
        env_lay.addWidget(self.lbl_env, 1)
        env_lay.addWidget(self.btn_env_refresh, 0)

        self.section_env = CollapsibleSection("Environment", env_content, collapsed=False)
        layout.addWidget(self.section_env)

        # Model manager (download only; no preloading)
        mm_content = QtWidgets.QWidget()
        mm_layout = QtWidgets.QGridLayout(mm_content)

        self.chk_tokenizer = QtWidgets.QCheckBox("Tokenizer (required)")
        self.chk_custom = QtWidgets.QCheckBox("CustomVoice (9 speakers)")
        self.chk_base = QtWidgets.QCheckBox("Base (needed for Voice Clone)")
        self.chk_design = QtWidgets.QCheckBox("VoiceDesign (prompt-made voices)")

        self.chk_tokenizer.setChecked(True)
        self.chk_custom.setChecked(True)

        self.btn_download = QtWidgets.QPushButton("Download selected")
        self.btn_open_models = QtWidgets.QPushButton("Open models folder")
        self.lbl_model_status = QtWidgets.QLabel("")
        self.lbl_model_status.setWordWrap(True)

        mm_layout.addWidget(self.chk_tokenizer, 0, 0)
        mm_layout.addWidget(self.chk_custom, 0, 1)
        mm_layout.addWidget(self.chk_base, 1, 0)
        mm_layout.addWidget(self.chk_design, 1, 1)
        mm_layout.addWidget(self.btn_download, 0, 2, 1, 1)
        mm_layout.addWidget(self.btn_open_models, 1, 2, 1, 1)
        mm_layout.addWidget(self.lbl_model_status, 2, 0, 1, 3)

        # --- Models section (paths) ---
        # Move per-tab model/tokenizer paths into this section (keeps tabs cleaner).
        self.tokenizer_path = QtWidgets.QLineEdit(DEFAULT_TOKENIZER_DIR)

        self.cv_model_path = QtWidgets.QLineEdit(DEFAULT_CUSTOMVOICE_DIR)
        self.vc_model_path = QtWidgets.QLineEdit(DEFAULT_BASE_DIR)
        self.vd_model_path = QtWidgets.QLineEdit(DEFAULT_VOICEDESIGN_DIR)

        self.models_tabs = QtWidgets.QTabWidget()
        self.models_tabs.setDocumentMode(True)
        self.models_tabs.setTabPosition(QtWidgets.QTabWidget.North)

        def _mk_model_tab(label: str, le: QtWidgets.QLineEdit) -> QtWidgets.QWidget:
            w = QtWidgets.QWidget()
            f = QtWidgets.QFormLayout(w)
            f.setContentsMargins(0, 0, 0, 0)
            f.addRow("Model path", le)
            return w

        self.models_tabs.addTab(_mk_model_tab("CustomVoice", self.cv_model_path), "CustomVoice")
        self.models_tabs.addTab(_mk_model_tab("Voice Clone (Base)", self.vc_model_path), "Voice Clone (Base)")
        self.models_tabs.addTab(_mk_model_tab("Voice Design", self.vd_model_path), "Voice Design")

        models_form = QtWidgets.QFormLayout()
        models_form.setContentsMargins(0, 6, 0, 0)
        models_form.addRow("Tokenizer path", self.tokenizer_path)
        models_form.addRow("", self.models_tabs)
        models_wrap = QtWidgets.QWidget()
        models_wrap.setLayout(models_form)

        mm_layout.addWidget(models_wrap, 3, 0, 1, 3)

        self.section_mm = CollapsibleSection("Model Manager", mm_content, collapsed=True)
        layout.addWidget(self.section_mm)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        # Common settings (collapsible)
        common_content = QtWidgets.QWidget()
        common_layout = QtWidgets.QFormLayout(common_content)

        self.device = QtWidgets.QComboBox()
        self.device.addItems(["auto", "cpu", "cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        self.device.setToolTip("Where to run the model (resolved inside the qwen3tts environment).")

        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItems(["bfloat16", "float16", "float32"])

        self.attn = QtWidgets.QComboBox()
        self.attn.addItems(["sdpa", "flash_attention_2", "none"])
        self.attn.setCurrentText("sdpa")

        self.temperature = QtWidgets.QDoubleSpinBox()
        self.temperature.setRange(0.05, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(0.8)

        self.top_p = QtWidgets.QDoubleSpinBox()
        self.top_p.setRange(0.05, 1.0)
        self.top_p.setSingleStep(0.01)
        self.top_p.setValue(0.95)

        self.top_k = QtWidgets.QSpinBox()
        self.top_k.setRange(0, 200)
        self.top_k.setValue(50)

        self.max_new_tokens = QtWidgets.QSpinBox()
        self.max_new_tokens.setRange(64, 8192)
        self.max_new_tokens.setValue(2048)

        out_row = QtWidgets.QHBoxLayout()
        self.output_name = QtWidgets.QLineEdit("qwen_tts")
        self.add_timestamp = QtWidgets.QCheckBox("Add timestamp")
        self.add_timestamp.setChecked(True)
        out_row.addWidget(self.output_name, 2)
        out_row.addWidget(self.add_timestamp, 1)

        common_layout.addRow("Device", self.device)
        common_layout.addRow("DType", self.dtype)
        common_layout.addRow("Attention", self.attn)
        common_layout.addRow("Temperature", self.temperature)
        common_layout.addRow("Top-p", self.top_p)
        common_layout.addRow("Top-k", self.top_k)
        common_layout.addRow("Max new tokens", self.max_new_tokens)
        common_layout.addRow("Output name", out_row)

        self.section_common = CollapsibleSection("Common generation settings", common_content, collapsed=True)
        layout.addWidget(self.section_common)

        # Tabs
        self.tab_custom = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_custom, "CustomVoice")
        self._build_custom_tab()

        self.tab_clone = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_clone, "Voice Clone (Base)")
        self._build_clone_tab()

        self.tab_design = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_design, "Voice Design")
        self._build_design_tab()

        # Bottom buttons + log
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Generate WAV (current tab)")
        self.btn_refresh_gpu = QtWidgets.QPushButton("Refresh GPU info")
        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_refresh_gpu)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.lbl_gpu = QtWidgets.QLabel("GPU: (checking...)")

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)

        logs_content = QtWidgets.QWidget()
        logs_lay = QtWidgets.QVBoxLayout(logs_content)
        logs_lay.setContentsMargins(0, 0, 0, 0)
        logs_lay.setSpacing(6)
        logs_lay.addWidget(self.lbl_gpu)
        logs_lay.addWidget(self.log, 1)

        self.section_logs = CollapsibleSection("Logs", logs_content, collapsed=True)
        layout.addWidget(self.section_logs, 1)

        # Tooltips (help text)
        self._apply_tooltips()

        # Signals
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_open_models.clicked.connect(self._open_models)
        self.btn_download.clicked.connect(self._download_selected)
        self.btn_refresh_gpu.clicked.connect(self._refresh_gpu_info_async)

    def _apply_tooltips(self):
        """Set help tooltips for all controls (kept in one place for easy editing)."""

        # Environment
        self.lbl_env.setToolTip(
            "Shows whether the dedicated Qwen3-TTS Python environment is installed.\n"
            "If missing, installs/optional-installs need to create: environments/.qwen3tts/."
        )
        self.btn_env_refresh.setToolTip("Re-check if the qwen3tts environment Python exists.")

        # Model Manager
        self.section_mm.setToolTip("Download the model folders into /models/. No model is loaded in the UI process.")
        self.chk_tokenizer.setToolTip(
            "Tokenizer (required).\n"
            "Without this, generation will fail even if the model weights are present."
        )
        self.chk_custom.setToolTip(
            "CustomVoice model (recommended default).\n"
            "Includes a small set of built-in speakers (e.g., ryan, serena)."
        )
        self.chk_base.setToolTip(
            "Base model used for Voice Clone.\n"
            "Download this if you want to clone a voice from a reference audio file."
        )
        self.chk_design.setToolTip(
            "VoiceDesign model.\n"
            "Lets you create a new voice from a text description (no reference audio)."
        )
        self.btn_download.setToolTip(
            "Download the selected repos from Hugging Face into /models/.\n"
            "Tip: set HF_TOKEN / HUGGINGFACEHUB_API_TOKEN if you hit rate limits."
        )
        self.btn_open_models.setToolTip("Open the local /models/ folder in Explorer.")
        self.lbl_model_status.setToolTip("Quick check: whether each model folder exists and is non-empty.")

        # Model / tokenizer paths (inside Model Manager)
        self.tokenizer_path.setToolTip("Folder for the Tokenizer model (downloaded into /models/).")
        self.cv_model_path.setToolTip(
            "Folder for the CustomVoice model (downloaded into /models/).\n"
            "Default is the expected local folder name."
        )
        self.vc_model_path.setToolTip("Folder for the Base model (needed for voice cloning).")
        self.vd_model_path.setToolTip("Folder for the VoiceDesign model (downloaded into /models/).")

        # Generate + GPU
        self.btn_generate.setToolTip(
            "Generate a WAV using the currently selected tab.\n"
            "Output goes to: output/audio/qwen3tts/."
        )
        self.btn_refresh_gpu.setToolTip("Queries CUDA/GPU availability inside the qwen3tts environment.")
        self.lbl_gpu.setToolTip("Detected GPU(s) reported by the qwen3tts environment.")

        # Common generation settings
        self.section_common.setToolTip("Sampling + runtime settings shared by all tabs.")
        self.device.setToolTip(
            "Where to run the model.\n"
            "• auto (default): uses CUDA if available, otherwise CPU\n"
            "• cpu: safest but slow\n"
            "• cuda:0 / cuda:1 ...: pick a specific GPU"
        )
        self.dtype.setToolTip(
            "Compute precision.\n"
            "• bfloat16 (default): good quality, stable on modern GPUs\n"
            "• float16: lower VRAM, sometimes less stable\n"
            "• float32: best compatibility (especially CPU), slowest"
        )
        self.attn.setToolTip(
            "Attention backend.\n"
            "• sdpa (default): good speed/compat\n"
            "• flash_attention_2: fastest + lower VRAM, requires flash_attn installed\n"
            "• none: most compatible fallback, usually slower"
        )
        self.temperature.setToolTip(
            "Randomness / expressiveness. Default: 0.80\n"
            "Lower (0.50–0.75) = more consistent, less surprise\n"
            "Higher (0.90–1.20) = more variation/energy, more risk of odd phrasing"
        )
        self.top_p.setToolTip(
            "Nucleus sampling cutoff. Default: 0.95\n"
            "Lower (0.85–0.92) = safer/more deterministic\n"
            "Higher (0.96–0.99) = more creative, can get messy if too high"
        )
        self.top_k.setToolTip(
            "Top-k sampling. Default: 50\n"
            "0 = disable top-k (use only top-p)\n"
            "20–80 is a practical range; lower = safer, higher = more variety."
        )
        self.max_new_tokens.setToolTip(
            "Upper limit for generated tokens (affects max audio length). Default: 2048\n"
            "If output cuts off early, increase this. If generation is slow, reduce it."
        )
        self.output_name.setToolTip(
            "Base filename for the output WAV.\n"
            "If 'Add timestamp' is enabled, the timestamp is appended automatically."
        )
        self.add_timestamp.setToolTip(
            "When enabled (default), outputs won't overwrite each other because a timestamp is added."
        )

        # CustomVoice tab
        self.cv_text.setToolTip(
            "Text to speak.\n"
            "Tip: for natural speech, use punctuation and short sentences."
        )
        self.cv_language.setToolTip(
            "Language hint (Auto may work, but setting the correct language usually improves pronunciation)."
        )
        self.cv_speaker.setToolTip(
            "Speaker ID (built-in voices).\n"
            "Tip: click 'Refresh speakers/languages' after downloading the model to get the exact list."
        )
        self.btn_cv_refresh.setToolTip(
            "Reads the supported speaker/language list from the model (runs in the environment)."
        )
        self.cv_instruct.setToolTip(
            "Optional style instruction (works like a short direction). Examples:\n"
            "• 'calm and warm, medium pace'\n"
            "• 'excited, smiling, faster delivery'\n"
            "Keep it short; too much can reduce clarity."
        )

        # Voice Clone tab
        self.vc_text.setToolTip("Text you want spoken by the cloned voice.")
        self.vc_language.setToolTip("Language hint for the target speech.")
        self.vc_ref_audio.setToolTip(
            "Reference audio file (WAV/MP3/FLAC/M4A).\n"
            "Best results: clean, single speaker, minimal background noise, 5–20 seconds."
        )
        self.btn_pick_audio.setToolTip("Pick a reference audio file for cloning.")
        self.vc_ref_text.setToolTip(
            "Optional transcript of the reference audio.\n"
            "Providing this can improve cloning accuracy when the reference is long or unclear."
        )
        self.vc_xvector.setToolTip(
            "If enabled, uses only speaker embedding (x-vector) and ignores ref_text.\n"
            "Useful when you don't know the exact transcript."
        )
        self.vc_instruct.setToolTip(
            "Optional style direction for the generated speech (pace, emotion, emphasis)."
        )

        # Voice Design tab
        self.vd_text.setToolTip("Text to speak using the designed voice.")
        self.vd_language.setToolTip("Language hint for the generated speech.")
        self.vd_desc.setToolTip(
            "Describe the voice you want. Keep it concrete. Examples:\n"
            "• 'deep, gentle, slow-paced, radio host'\n"
            "• 'young, bright, playful, slight laugh'\n"
            "Add pronunciation hints like 'clear enunciation' if needed."
        )

        # Logs
        self.section_logs.setToolTip("Shows streaming logs from the worker subprocess.")
        self.log.setToolTip("Worker output. If something fails, scroll here for the traceback.")

    def _build_custom_tab(self):
        lay = QtWidgets.QFormLayout(self.tab_custom)

        self.cv_text = QtWidgets.QPlainTextEdit()
        self.cv_text.setPlainText("Okay, quick test: the rain taps the window like tiny drums. If you can hear this clearly, it works.")

        self.cv_language = QtWidgets.QComboBox()
        self.cv_language.addItems(SUPPORTED_LANGS_FALLBACK)
        self.cv_language.setCurrentText("English")

        self.cv_speaker = QtWidgets.QComboBox()
        self.cv_speaker.setEditable(True)
        self.cv_speaker.addItems(SUPPORTED_SPEAKERS_FALLBACK)
        self.cv_speaker.setCurrentText("ryan")

        self.cv_instruct = QtWidgets.QLineEdit("")

        self.btn_cv_refresh = QtWidgets.QPushButton("Refresh speakers/languages from model")
        self.btn_cv_refresh.clicked.connect(self._refresh_supported_from_model_async)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.cv_speaker, 2)
        row.addWidget(self.btn_cv_refresh, 1)

        lay.addRow("Text", self.cv_text)
        lay.addRow("Language", self.cv_language)
        lay.addRow("Speaker", row)
        lay.addRow("Style instruction", self.cv_instruct)

    def _build_clone_tab(self):
        lay = QtWidgets.QFormLayout(self.tab_clone)

        self.vc_text = QtWidgets.QPlainTextEdit()
        self.vc_text.setPlainText("Hey—this is a quick voice clone test. If this sounds like the reference speaker, it worked.")

        self.vc_language = QtWidgets.QComboBox()
        self.vc_language.addItems(SUPPORTED_LANGS_FALLBACK)
        self.vc_language.setCurrentText("English")

        ref_row = QtWidgets.QHBoxLayout()
        self.vc_ref_audio = QtWidgets.QLineEdit("")
        self.vc_ref_audio.setPlaceholderText("Select a reference WAV/MP3…")
        self.btn_pick_audio = QtWidgets.QPushButton("Browse…")
        self.btn_pick_audio.clicked.connect(self._pick_ref_audio)
        ref_row.addWidget(self.vc_ref_audio, 3)
        ref_row.addWidget(self.btn_pick_audio, 1)

        self.vc_ref_text = QtWidgets.QLineEdit("")
        self.vc_xvector = QtWidgets.QCheckBox("x_vector_only_mode (no ref_text)")

        self.vc_instruct = QtWidgets.QLineEdit("")

        lay.addRow("Text", self.vc_text)
        lay.addRow("Language", self.vc_language)
        lay.addRow("Reference audio", ref_row)
        lay.addRow("Reference text", self.vc_ref_text)
        lay.addRow("", self.vc_xvector)
        lay.addRow("Style instruction", self.vc_instruct)

    def _build_design_tab(self):
        lay = QtWidgets.QFormLayout(self.tab_design)

        self.vd_text = QtWidgets.QPlainTextEdit()
        self.vd_text.setPlainText("Hello. This is voice design. If the voice matches the description, it worked.")

        self.vd_language = QtWidgets.QComboBox()
        self.vd_language.addItems(SUPPORTED_LANGS_FALLBACK)
        self.vd_language.setCurrentText("English")

        self.vd_desc = QtWidgets.QPlainTextEdit()
        self.vd_desc.setPlainText("A calm, friendly voice, slightly smiling, clear pronunciation.")

        lay.addRow("Text", self.vd_text)
        lay.addRow("Language", self.vd_language)
        lay.addRow("Voice description", self.vd_desc)

    # ---------- Autosave ----------
    def _schedule_autosave(self):
        try:
            self._autosave_timer.start()
        except Exception:
            pass

    def _wire_autosave_signals(self):
        def c(sig):
            try:
                sig.connect(lambda *args, **kwargs: self._schedule_autosave())
            except Exception:
                pass

        # Common
        c(self.tabs.currentChanged)
        c(self.device.currentTextChanged)
        c(self.dtype.currentTextChanged)
        c(self.attn.currentTextChanged)
        c(self.temperature.valueChanged)
        c(self.top_p.valueChanged)
        c(self.top_k.valueChanged)
        c(self.max_new_tokens.valueChanged)
        c(self.output_name.textChanged)
        c(self.add_timestamp.toggled)

        # Collapse states
        c(self.section_mm.toggled)
        c(self.section_common.toggled)
        c(self.section_logs.toggled)
        c(self.section_env.toggled)

        # CustomVoice
        c(self.cv_model_path.textChanged)
        c(self.tokenizer_path.textChanged)
        c(self.cv_text.textChanged)
        c(self.cv_language.currentTextChanged)
        c(self.cv_speaker.currentTextChanged)
        c(self.cv_speaker.editTextChanged)
        c(self.cv_instruct.textChanged)

        # Voice Clone
        c(self.vc_model_path.textChanged)
        c(self.vc_text.textChanged)
        c(self.vc_language.currentTextChanged)
        c(self.vc_ref_audio.textChanged)
        c(self.vc_ref_text.textChanged)
        c(self.vc_xvector.toggled)
        c(self.vc_instruct.textChanged)

        # Voice Design
        c(self.vd_model_path.textChanged)
        c(self.vd_text.textChanged)
        c(self.vd_language.currentTextChanged)
        c(self.vd_desc.textChanged)

    # ---------- Helpers ----------
    def _log(self, s: str):
        self.log.appendPlainText(s)

    def _update_env_status(self):
        if _env_ok():
            self.lbl_env.setText(f"✓ Env python found: {ENV_PY}")
            self.btn_download.setEnabled(True)
            self.btn_generate.setEnabled(True)
        else:
            self.lbl_env.setText(f"✗ Missing env python: {ENV_PY}")
            self.btn_download.setEnabled(False)
            self.btn_generate.setEnabled(False)

    def _open_models(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(MODELS_DIR))
        except Exception:
            pass

    def _update_model_status(self):
        def ok(p: str) -> bool:
            d = Path(p)
            return d.exists() and any(d.iterdir())

        rows = [
            ("Tokenizer", ok(DEFAULT_TOKENIZER_DIR), DEFAULT_TOKENIZER_DIR),
            ("CustomVoice", ok(DEFAULT_CUSTOMVOICE_DIR), DEFAULT_CUSTOMVOICE_DIR),
            ("Base", ok(DEFAULT_BASE_DIR), DEFAULT_BASE_DIR),
            ("VoiceDesign", ok(DEFAULT_VOICEDESIGN_DIR), DEFAULT_VOICEDESIGN_DIR),
        ]
        msg = []
        for name, present, path in rows:
            msg.append(f"{'✓' if present else '✗'} {name}: {Path(path).name}")
        self.lbl_model_status.setText(" | ".join(msg))

    def _state_file(self) -> Path:
        return PRESETS_DIR / "qwentts.json"

    def _collect_common(self) -> dict:
        return {
            "device": self.device.currentText(),
            "dtype": self.dtype.currentText(),
            "attn_impl": self.attn.currentText(),
            "temperature": float(self.temperature.value()),
            "top_p": float(self.top_p.value()),
            "top_k": int(self.top_k.value()),
            "max_new_tokens": int(self.max_new_tokens.value()),
            "output_name": self.output_name.text().strip() or "qwen_tts",
            "add_timestamp": bool(self.add_timestamp.isChecked()),
        }

    def _collect_payload(self) -> tuple[str, dict]:
        common = self._collect_common()
        tab = self.tabs.currentWidget()

        tok = self.tokenizer_path.text().strip()

        if tab is self.tab_custom:
            return "custom", {
                "text": self.cv_text.toPlainText(),
                "language": self.cv_language.currentText(),
                "speaker": self.cv_speaker.currentText(),
                "instruct": self.cv_instruct.text(),
                "model_path": self.cv_model_path.text().strip(),
                "tokenizer_path": tok,
                "common": common,
            }

        if tab is self.tab_clone:
            return "clone", {
                "text": self.vc_text.toPlainText(),
                "language": self.vc_language.currentText(),
                "ref_audio_path": self.vc_ref_audio.text(),
                "ref_text": self.vc_ref_text.text(),
                "x_vector_only_mode": bool(self.vc_xvector.isChecked()),
                "instruct": self.vc_instruct.text(),
                "model_path": self.vc_model_path.text().strip(),
                "tokenizer_path": tok,
                "common": common,
            }

        return "design", {
            "text": self.vd_text.toPlainText(),
            "language": self.vd_language.currentText(),
            "voice_description": self.vd_desc.toPlainText(),
            "model_path": self.vd_model_path.text().strip(),
            "tokenizer_path": tok,
            "common": common,
        }

    def _save_last_state(self):
        try:
            state = {
                "tab": self.tabs.currentIndex(),
                "ui": {
                    "env_collapsed": bool(self.section_env.isCollapsed()),
                    "mm_collapsed": bool(self.section_mm.isCollapsed()),
                    "common_collapsed": bool(self.section_common.isCollapsed()),
                    "logs_collapsed": bool(self.section_logs.isCollapsed()),
                },
                "common": self._collect_common(),
                "models": {
                    "tokenizer_path": self.tokenizer_path.text(),
                    "custom_model_path": self.cv_model_path.text(),
                    "base_model_path": self.vc_model_path.text(),
                    "voicedesign_model_path": self.vd_model_path.text(),
                },
                "custom": {
                    "text": self.cv_text.toPlainText(),
                    "language": self.cv_language.currentText(),
                    "speaker": self.cv_speaker.currentText(),
                    "instruct": self.cv_instruct.text(),
                },
                "clone": {
                    "text": self.vc_text.toPlainText(),
                    "language": self.vc_language.currentText(),
                    "ref_audio_path": self.vc_ref_audio.text(),
                    "ref_text": self.vc_ref_text.text(),
                    "x_vector_only_mode": bool(self.vc_xvector.isChecked()),
                    "instruct": self.vc_instruct.text(),
                },
                "design": {
                    "text": self.vd_text.toPlainText(),
                    "language": self.vd_language.currentText(),
                    "voice_description": self.vd_desc.toPlainText(),
                }
            }
            self._state_file().write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_last_state(self):
        try:
            f = self._state_file()
            if not f.exists():
                return
            state = json.loads(f.read_text(encoding="utf-8"))

            ui = state.get("ui", {}) or {}
            try:
                self.section_env.setCollapsed(bool(ui.get("env_collapsed", False)))
                self.section_mm.setCollapsed(bool(ui.get("mm_collapsed", True)))
                self.section_common.setCollapsed(bool(ui.get("common_collapsed", True)))
                self.section_logs.setCollapsed(bool(ui.get("logs_collapsed", True)))
            except Exception:
                pass

            c = state.get("common", {})
            if c:
                self.device.setCurrentText(c.get("device", "auto"))
                self.dtype.setCurrentText(c.get("dtype", "bfloat16"))
                self.attn.setCurrentText(c.get("attn_impl", "sdpa"))
                self.temperature.setValue(float(c.get("temperature", 0.8)))
                self.top_p.setValue(float(c.get("top_p", 0.95)))
                self.top_k.setValue(int(c.get("top_k", 50)))
                self.max_new_tokens.setValue(int(c.get("max_new_tokens", 2048)))
                self.output_name.setText(c.get("output_name", "qwen_tts"))
                self.add_timestamp.setChecked(bool(c.get("add_timestamp", True)))

            # New format (preferred)
            m = state.get("models", {}) or {}
            if m:
                self.tokenizer_path.setText(m.get("tokenizer_path", DEFAULT_TOKENIZER_DIR))
                self.cv_model_path.setText(m.get("custom_model_path", DEFAULT_CUSTOMVOICE_DIR))
                self.vc_model_path.setText(m.get("base_model_path", DEFAULT_BASE_DIR))
                self.vd_model_path.setText(m.get("voicedesign_model_path", DEFAULT_VOICEDESIGN_DIR))

            cv = state.get("custom", {})
            if cv:
                # Back-compat: older state files stored paths here
                if not m:
                    self.cv_model_path.setText(cv.get("model_path", DEFAULT_CUSTOMVOICE_DIR))
                    self.tokenizer_path.setText(cv.get("tokenizer_path", DEFAULT_TOKENIZER_DIR))
                self.cv_text.setPlainText(cv.get("text", ""))
                self.cv_language.setCurrentText(cv.get("language", "English"))
                self.cv_speaker.setCurrentText(safe_lower_speaker(cv.get("speaker", "ryan")))
                self.cv_instruct.setText(cv.get("instruct", ""))

            vc = state.get("clone", {})
            if vc:
                if not m:
                    self.vc_model_path.setText(vc.get("model_path", DEFAULT_BASE_DIR))
                    # tokenizer already taken from custom in older format
                self.vc_text.setPlainText(vc.get("text", ""))
                self.vc_language.setCurrentText(vc.get("language", "English"))
                self.vc_ref_audio.setText(vc.get("ref_audio_path", ""))
                self.vc_ref_text.setText(vc.get("ref_text", ""))
                self.vc_xvector.setChecked(bool(vc.get("x_vector_only_mode", False)))
                self.vc_instruct.setText(vc.get("instruct", ""))

            vd = state.get("design", {})
            if vd:
                if not m:
                    self.vd_model_path.setText(vd.get("model_path", DEFAULT_VOICEDESIGN_DIR))
                self.vd_text.setPlainText(vd.get("text", ""))
                self.vd_language.setCurrentText(vd.get("language", "English"))
                self.vd_desc.setPlainText(vd.get("voice_description", ""))

            self.tabs.setCurrentIndex(int(state.get("tab", 0)))
        except Exception:
            pass

    # ---------- Downloads ----------
    def _download_selected(self):
        items = []
        if self.chk_tokenizer.isChecked():
            items.append(("Tokenizer", HF_MODELS["Tokenizer"]))
        if self.chk_custom.isChecked():
            items.append(("CustomVoice", HF_MODELS["CustomVoice"]))
        if self.chk_base.isChecked():
            items.append(("Base", HF_MODELS["Base (for Voice Clone)"]))
        if self.chk_design.isChecked():
            items.append(("VoiceDesign", HF_MODELS["VoiceDesign"]))

        if not items:
            self._log("Nothing selected for download.")
            return

        self.btn_download.setEnabled(False)
        self._log("Starting downloads...")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or ""

        self._run_env_task(
            task="download",
            payload={"items": items, "token": token},
            on_ok=self._on_download_done,
        )

    def _on_download_done(self, result: dict):
        self.btn_download.setEnabled(True)
        self._update_model_status()
        self._log("Downloads completed.")
        if result.get("warnings"):
            for w in result["warnings"]:
                self._log(w)

    # ---------- GPU info ----------
    def _refresh_gpu_info_async(self):
        self._run_env_task(task="gpuinfo", payload={}, on_ok=self._on_gpuinfo)

    def _on_gpuinfo(self, result: dict):
        if result.get("cuda_available"):
            gpus = result.get("gpus", [])
            if gpus:
                self.lbl_gpu.setText("GPU: " + " | ".join(gpus))
            else:
                self.lbl_gpu.setText("GPU: cuda available (no names returned)")
        else:
            self.lbl_gpu.setText("GPU: none (CPU mode)")

    # ---------- Supported lists ----------
    def _refresh_supported_from_model_async(self):
        model_dir = self.cv_model_path.text().strip()
        if not (Path(model_dir).exists() and any(Path(model_dir).iterdir())):
            self._log("CustomVoice model folder not found. Download it in Model Manager first.")
            return

        common = self._collect_common()
        payload = {
            "model_path": model_dir,
            "tokenizer_path": self.tokenizer_path.text().strip(),
            "common": common,
        }
        self._run_env_task(task="supported", payload=payload, on_ok=self._on_supported_lists)

    def _on_supported_lists(self, result: dict):
        speakers = result.get("speakers") or SUPPORTED_SPEAKERS_FALLBACK
        langs = result.get("languages") or SUPPORTED_LANGS_FALLBACK

        try:
            cur = self.cv_speaker.currentText()
            self.cv_speaker.blockSignals(True)
            self.cv_speaker.clear()
            self.cv_speaker.addItems([safe_lower_speaker(x) for x in speakers])
            self.cv_speaker.setEditable(True)
            cur2 = safe_lower_speaker(cur)
            normalized = [safe_lower_speaker(x) for x in speakers]
            self.cv_speaker.setCurrentText(cur2 if cur2 in normalized else safe_lower_speaker(speakers[0]))
            self.cv_speaker.blockSignals(False)
        except Exception:
            pass

        try:
            cur = self.cv_language.currentText()
            self.cv_language.blockSignals(True)
            self.cv_language.clear()
            self.cv_language.addItems([str(x) for x in langs])
            self.cv_language.setCurrentText(cur if cur in langs else str(langs[0]))
            self.cv_language.blockSignals(False)
        except Exception:
            pass

        self._log("Refreshed supported speakers/languages from model.")

    # ---------- Generate ----------
    def _on_generate(self):
        self._save_last_state()

        mode, payload = self._collect_payload()

        mpath = payload.get("model_path", "")
        if not (mpath and Path(mpath).exists() and any(Path(mpath).iterdir())):
            self._log("Model folder missing. Use Model Manager to download it first.")
            return

        self._log(f"Starting generation ({mode})...")
        self.btn_generate.setEnabled(False)

        self._run_env_task(
            task="generate",
            payload={"mode": mode, "payload": payload},
            on_ok=self._on_generate_done,
            on_fail=self._on_generate_fail,
            reenable_generate=True,
        )

    def _on_generate_done(self, result: dict):
        out_path = result.get("out_path", "")
        self._log(f"Done: {out_path}")
        self.btn_generate.setEnabled(True)
        self._update_model_status()

    def _on_generate_fail(self, err: str):
        self._log("ERROR:\n" + err)
        self.btn_generate.setEnabled(True)

    # ---------- Utility ----------
    def _pick_ref_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select reference audio", "", "Audio (*.wav *.mp3 *.flac *.m4a);;All files (*.*)")
        if path:
            self.vc_ref_audio.setText(path)

    def _update_env_status_logonly(self, result: dict):
        # Used by tasks that don't otherwise touch UI
        pass

    def _run_env_task(self, task: str, payload: dict, on_ok=None, on_fail=None, reenable_generate: bool = False):
        """
        Run an environment task via QThread so the UI stays responsive.
        """

        def _fail(msg: str):
            if on_fail is not None:
                on_fail(msg)
            else:
                self._log("ERROR:\n" + msg)
            if reenable_generate:
                self.btn_generate.setEnabled(True)
            self.btn_download.setEnabled(True)

        def _ok(res: dict):
            if on_ok is not None:
                on_ok(res)
            if reenable_generate:
                self.btn_generate.setEnabled(True)
            self.btn_download.setEnabled(True)

        self._thread = QtCore.QThread()
        self._worker = EnvTaskWorker(task=task, payload=payload)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log)
        self._worker.done.connect(_ok)
        self._worker.failed.connect(_fail)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def closeEvent(self, event):
        try:
            self._save_last_state()
        except Exception:
            pass
        return super().closeEvent(event)


# -----------------------------
# Worker-mode implementation
# -----------------------------
def _try_inject_repo_path(root: Path) -> None:
    """
    Fallback if qwen_tts isn't installed in the environment:
    add a nearby repo folder to sys.path.
    """
    candidates = [
        root / "repo",
        root / "repos" / "Qwen3-TTS",
        root / "Qwen3-TTS",
        root / "presets" / "extra_env" / "repo",
        root / "presets" / "extra_env" / "Qwen3-TTS",
    ]
    for c in candidates:
        try:
            if (c / "qwen_tts").exists():
                sys.path.insert(0, str(c))
                return
        except Exception:
            pass


def _worker_log(s: str):
    try:
        print("__LOG__ " + str(s), flush=True)
    except Exception:
        pass


def _worker_result(ok: bool, **kw):
    obj = {"ok": bool(ok), **kw}
    try:
        print("__RESULT__" + json.dumps(obj, ensure_ascii=False), flush=True)
    except Exception:
        # last resort
        try:
            print("__RESULT__" + json.dumps({"ok": False, "error": "Failed to encode result JSON"}, ensure_ascii=False), flush=True)
        except Exception:
            pass


def _resolve_device(torch_mod, device_choice: str):
    if device_choice == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    return device_choice


def _resolve_dtype(torch_mod, dtype_choice: str):
    if dtype_choice == "bfloat16":
        return torch_mod.bfloat16
    if dtype_choice == "float16":
        return torch_mod.float16
    return torch_mod.float32


def _resolve_attn(attn_choice: str):
    if attn_choice == "none":
        return None
    return attn_choice


def _has_flash_attn() -> bool:
    return importlib.util.find_spec("flash_attn") is not None


def _load_model(Qwen3TTSModel, model_path_or_id: str, tokenizer_path: str, common: dict):
    import torch  # local import (worker only)

    device = _resolve_device(torch, common.get("device", "auto"))
    dtype_choice = common.get("dtype", "bfloat16")
    dtype = _resolve_dtype(torch, dtype_choice)
    attn_choice = common.get("attn_impl", "sdpa")
    attn_impl = _resolve_attn(attn_choice)

    # Guard: FlashAttention2 can't be used unless installed
    if attn_choice == "flash_attention_2" and not _has_flash_attn():
        _worker_log("FlashAttention2 selected but flash_attn is not installed; falling back to SDPA.")
        attn_choice = "sdpa"
        attn_impl = "sdpa"

    kwargs = {"device_map": device, "dtype": dtype}
    if attn_impl is not None:
        kwargs["attn_implementation"] = attn_impl

    # Some package versions may accept a tokenizer path. Try it, then fall back.
    if tokenizer_path:
        for key in ("tokenizer_path", "tokenizer_dir", "tokenizer"):
            try:
                _worker_log(f"Loading model with {key}=... (if supported)")
                return Qwen3TTSModel.from_pretrained(model_path_or_id, **{key: tokenizer_path}, **kwargs)
            except TypeError:
                continue
            except Exception as e:
                msg = str(e).lower()
                if "unexpected keyword" in msg:
                    continue
                # If it failed for another reason, try without tokenizer key later.
                break

    _worker_log("Loading model...")
    try:
        return Qwen3TTSModel.from_pretrained(model_path_or_id, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if ("flash_attn" in msg) or ("flashattention2" in msg):
            _worker_log("FlashAttention2 not usable here; retrying with SDPA...")
            kwargs["attn_implementation"] = "sdpa"
            try:
                return Qwen3TTSModel.from_pretrained(model_path_or_id, **kwargs)
            except Exception:
                kwargs.pop("attn_implementation", None)
                return Qwen3TTSModel.from_pretrained(model_path_or_id, **kwargs)
        if dtype_choice == "bfloat16":
            _worker_log("bfloat16 load failed; retrying with float16...")
            kwargs["dtype"] = torch.float16
            return Qwen3TTSModel.from_pretrained(model_path_or_id, **kwargs)
        raise


def _worker_task_gpuinfo():
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            gpus = []
            for i in range(torch.cuda.device_count()):
                try:
                    gpus.append(f"cuda:{i} = {torch.cuda.get_device_name(i)}")
                except Exception:
                    gpus.append(f"cuda:{i}")
            _worker_result(True, cuda_available=True, gpus=gpus)
        else:
            _worker_result(True, cuda_available=False, gpus=[])
    except Exception:
        _worker_result(False, error=traceback.format_exc())


def _worker_task_download(payload: dict):
    try:
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        from huggingface_hub import snapshot_download  # type: ignore

        items = payload.get("items") or []
        token = payload.get("token") or None

        warnings = []
        for label, repo_id in items:
            local_dir = Path(repo_to_local_dir(repo_id))
            local_dir.mkdir(parents=True, exist_ok=True)
            if local_dir.exists() and any(local_dir.iterdir()):
                _worker_log(f"✓ Already present: {label}")
                continue

            _worker_log(f"Downloading: {label} ({repo_id}) ...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                token=token,
                resume_download=True,
            )
            _worker_log(f"✓ Downloaded: {label}")

        _worker_result(True, warnings=warnings)
    except Exception:
        _worker_result(False, error=traceback.format_exc())


def _out_path(base_name: str, add_ts: bool) -> Path:
    name = (base_name or "qwen_tts").strip()
    if add_ts:
        name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    return OUTPUT_DIR / f"{name}.wav"


def _worker_task_supported(payload: dict):
    try:
        # Optional: make SoX discoverable for qwen-tts (Windows friendly)
        try:
            import static_sox  # type: ignore
            static_sox.add_paths(weak=True)
        except Exception:
            pass

        _try_inject_repo_path(ROOT_DIR)

        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except Exception:
            from qwen_tts.inference import Qwen3TTSModel  # type: ignore

        model_path = payload.get("model_path", "")
        tokenizer_path = payload.get("tokenizer_path", "") or ""
        common = payload.get("common") or {}

        model = _load_model(Qwen3TTSModel, model_path, tokenizer_path, common)

        speakers = []
        languages = []
        try:
            speakers = list(model.get_supported_speakers())
        except Exception:
            speakers = SUPPORTED_SPEAKERS_FALLBACK
        try:
            languages = list(model.get_supported_languages())
        except Exception:
            languages = SUPPORTED_LANGS_FALLBACK

        _worker_result(True, speakers=speakers, languages=languages)
    except Exception:
        _worker_result(False, error=traceback.format_exc())


def _worker_task_generate(payload: dict):
    try:
        # Optional: make SoX discoverable for qwen-tts (Windows friendly)
        try:
            import static_sox  # type: ignore
            static_sox.add_paths(weak=True)
        except Exception:
            pass

        _try_inject_repo_path(ROOT_DIR)

        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except Exception:
            from qwen_tts.inference import Qwen3TTSModel  # type: ignore

        mode = payload.get("mode", "")
        p = payload.get("payload") or {}
        common = p.get("common") or {}

        model_path = p.get("model_path", "")
        tokenizer_path = p.get("tokenizer_path", "") or ""

        model = _load_model(Qwen3TTSModel, model_path, tokenizer_path, common)

        gen_kwargs = {
            "max_new_tokens": int(common.get("max_new_tokens", 2048)),
            "temperature": float(common.get("temperature", 0.8)),
            "top_p": float(common.get("top_p", 0.95)),
            "top_k": int(common.get("top_k", 50)),
        }

        out_wav = _out_path(common.get("output_name", "qwen_tts"), bool(common.get("add_timestamp", True)))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        import soundfile as sf  # type: ignore

        if mode == "custom":
            wavs, sr = model.generate_custom_voice(
                text=p.get("text", ""),
                language=p.get("language", "English"),
                speaker=safe_lower_speaker(p.get("speaker", "")),
                instruct=(p.get("instruct", "").strip() or None),
                non_streaming_mode=True,
                **gen_kwargs,
            )
            sf.write(str(out_wav), wavs[0], sr)

        elif mode == "clone":
            ref_audio_path = (p.get("ref_audio_path") or "").strip()
            if not ref_audio_path:
                raise ValueError("Please choose a reference audio file.")
            wavs, sr = model.generate_voice_clone(
                text=p.get("text", ""),
                language=p.get("language", "English"),
                ref_audio=ref_audio_path,
                ref_text=(p.get("ref_text", "").strip() or None),
                x_vector_only_mode=bool(p.get("x_vector_only_mode", False)),
                instruct=(p.get("instruct", "").strip() or None),
                non_streaming_mode=True,
                **gen_kwargs,
            )
            sf.write(str(out_wav), wavs[0], sr)

        elif mode == "design":
            wavs, sr = model.generate_voice_design(
                text=p.get("text", ""),
                language=p.get("language", "English"),
                instruct=(p.get("voice_description", "").strip() or ""),
                non_streaming_mode=True,
                **gen_kwargs,
            )
            sf.write(str(out_wav), wavs[0], sr)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        _worker_result(True, out_path=str(out_wav))
    except Exception:
        _worker_result(False, error=traceback.format_exc())


def worker_main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--task", default="")
    args = ap.parse_args(argv if argv is not None else sys.argv[1:])

    # Read JSON payload from stdin (single blob)
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    # Ensure basic dirs exist
    ensure_dirs()

    if args.task == "gpuinfo":
        _worker_task_gpuinfo()
    elif args.task == "download":
        _worker_task_download(payload)
    elif args.task == "supported":
        _worker_task_supported(payload)
    elif args.task == "generate":
        _worker_task_generate(payload)
    else:
        _worker_result(False, error=f"Unknown task: {args.task}")


# -----------------------------
# Entrypoint
# -----------------------------
def main(argv=None):
    ensure_dirs()

    import argparse
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--embedded", action="store_true")
    ap.add_argument("--embed-token", default="")
    ap.add_argument("--print-hwnd", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--task", default="")
    args, qt_argv = ap.parse_known_args(argv if argv is not None else sys.argv[1:])

    if args.worker:
        # Worker mode should run in the qwen3tts environment python.
        worker_main(["--worker", "--task", args.task])
        return

    # UI mode
    app = QtWidgets.QApplication([sys.argv[0], *qt_argv])
    w = MainWindow()

    if args.embed_token:
        try:
            w.setWindowTitle(f"Qwen3-TTS::{args.embed_token}")
        except Exception:
            pass

    if args.embedded:
        # Helps avoid a second taskbar entry when the host reparents the window.
        try:
            w.setWindowFlag(QtCore.Qt.Tool, True)
        except Exception:
            pass
        try:
            w.setWindowFlag(QtCore.Qt.FramelessWindowHint, True)
        except Exception:
            pass

    w.show()

    if args.print_hwnd or args.embedded:
        def _emit_hwnd():
            try:
                hwnd = int(w.winId())
            except Exception:
                hwnd = 0
            try:
                print(json.dumps({"hwnd": hwnd, "token": args.embed_token}), flush=True)
            except Exception:
                pass

        try:
            QtCore.QTimer.singleShot(0, _emit_hwnd)
        except Exception:
            _emit_hwnd()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
