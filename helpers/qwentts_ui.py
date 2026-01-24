import os
import sys
import json
import time
import traceback
import importlib.util
from dataclasses import dataclass, asdict, field
from pathlib import Path
# ---------- FrameVision tool layout ----------
# This file lives in: <root>/helpers/qwentts_ui.py
# Expected folders:
#   <root>/environments/.qwen3tts/
#   <root>/presets/extra_env/
#   <root>/presets/setsave/        (settings/presets)
#   <root>/models/
#   <root>/output/
ROOT_DIR = Path(__file__).resolve().parents[1]

ENV_PY = ROOT_DIR / "environments" / ".qwen3tts" / "Scripts" / "python.exe"

def _relaunch_in_env() -> None:
    """If launched with the wrong Python, relaunch using the tool venv."""
    try:
        if ENV_PY.exists():
            cur = Path(sys.executable).resolve()
            tgt = ENV_PY.resolve()
            if cur != tgt:
                os.execv(str(tgt), [str(tgt), str(Path(__file__).resolve()), *sys.argv[1:]])
    except Exception:
        # If anything goes wrong, continue; the UI will show import errors later.
        pass

_relaunch_in_env()

def _try_inject_repo_path() -> None:
    """Fallback if qwen_tts isn't installed: add a nearby repo folder to sys.path."""
    candidates = [
        ROOT_DIR / "repo",
        ROOT_DIR / "repos" / "Qwen3-TTS",
        ROOT_DIR / "Qwen3-TTS",
        ROOT_DIR / "presets" / "extra_env" / "repo",
        ROOT_DIR / "presets" / "extra_env" / "Qwen3-TTS",
    ]
    for c in candidates:
        try:
            if (c / "qwen_tts").exists():
                sys.path.insert(0, str(c))
                return
        except Exception:
            pass

# Make SoX discoverable for qwen-tts (Windows friendly, no system install needed)
try:
    import static_sox  # type: ignore
    static_sox.add_paths(weak=True)
except Exception:
    pass

import torch
from PySide6 import QtCore, QtWidgets


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


# qwen-tts entrypoint (robust across package layouts)
_try_inject_repo_path()
try:
    from qwen_tts import Qwen3TTSModel  # type: ignore
except Exception:
    from qwen_tts.inference import Qwen3TTSModel  # type: ignore

# Optional deps used only for some features
HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None

HELPERS_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "output"
PRESETS_DIR = ROOT_DIR / "presets" / "setsave"

# Hugging Face repos we support in this UI
HF_MODELS = {
    "Tokenizer": "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Base (for Voice Clone)": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

def repo_to_local_dir(repo_id: str) -> str:
    return str(MODELS_DIR / repo_id.split("/")[-1])

DEFAULT_TOKENIZER_DIR = repo_to_local_dir(HF_MODELS["Tokenizer"])
DEFAULT_CUSTOMVOICE_DIR = repo_to_local_dir(HF_MODELS["CustomVoice"])
DEFAULT_BASE_DIR = repo_to_local_dir(HF_MODELS["Base (for Voice Clone)"])
DEFAULT_VOICEDESIGN_DIR = repo_to_local_dir(HF_MODELS["VoiceDesign"])

SUPPORTED_SPEAKERS_FALLBACK = ["aiden","dylan","eric","ono_anna","ryan","serena","sohee","uncle_fu","vivian"]
SUPPORTED_LANGS_FALLBACK = ["Auto","Chinese","English","Japanese","Korean","French","German","Spanish","Portuguese","Russian","Italian","Arabic"]

def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)

def safe_lower_speaker(s: str) -> str:
    return (s or "").strip().lower()

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


class HFDownloadWorker(QtCore.QObject):
    log = QtCore.Signal(str)
    done = QtCore.Signal(bool)

    def __init__(self, items: list[tuple[str, str]], token: str | None = None):
        super().__init__()
        self.items = items
        self.token = token

    @QtCore.Slot()
    def run(self):
        try:
            # Disable hf_transfer if user has it enabled globally (prevents the "hf_transfer not installed" crash).
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
            from huggingface_hub import snapshot_download  # type: ignore

            for label, repo_id in self.items:
                local_dir = Path(repo_to_local_dir(repo_id))
                if local_dir.exists() and any(local_dir.iterdir()):
                    self.log.emit(f"✓ Already present: {label}")
                    continue

                self.log.emit(f"Downloading: {label} ({repo_id}) ...")
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    token=self.token,
                    resume_download=True,
                )
                self.log.emit(f"✓ Downloaded: {label}")

            self.done.emit(True)
        except Exception:
            self.log.emit("ERROR during download:\n" + traceback.format_exc())
            self.done.emit(False)


class ModelCache(QtCore.QObject):
    status = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self._model = None
        self._loaded_key = None

    def _resolve_device(self, device_choice: str):
        if device_choice == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_choice

    def _resolve_dtype(self, dtype_choice: str):
        if dtype_choice == "bfloat16":
            return torch.bfloat16
        if dtype_choice == "float16":
            return torch.float16
        return torch.float32

    def _resolve_attn(self, attn_choice: str):
        if attn_choice == "none":
            return None
        return attn_choice

    def load_if_needed(self, model_path_or_id: str, device_choice: str, dtype_choice: str, attn_choice: str):
        device = self._resolve_device(device_choice)

        # Guard: FlashAttention2 cannot be used unless flash_attn is installed.
        if attn_choice == "flash_attention_2" and not HAS_FLASH_ATTN:
            self.status.emit("FlashAttention2 selected but flash_attn is not installed; falling back to SDPA.")
            attn_choice = "sdpa"

        dtype = self._resolve_dtype(dtype_choice)
        attn_impl = self._resolve_attn(attn_choice)

        key = (model_path_or_id, device, str(dtype), str(attn_impl))
        if self._model is not None and self._loaded_key == key:
            return self._model

        self.status.emit("Loading model (first time can take a while)...")

        kwargs = {
            "device_map": device,
            "dtype": dtype,
        }
        if attn_impl is not None:
            kwargs["attn_implementation"] = attn_impl

        try:
            model = Qwen3TTSModel.from_pretrained(model_path_or_id, **kwargs)
        except Exception as e:
            msg = str(e)
            if ("flash_attn" in msg.lower()) or ("flashattention2" in msg.lower()):
                self.status.emit("FlashAttention2 is not usable here; retrying with SDPA...")
                kwargs.pop("attn_implementation", None)
                kwargs["attn_implementation"] = "sdpa"
                try:
                    model = Qwen3TTSModel.from_pretrained(model_path_or_id, **kwargs)
                except Exception:
                    kwargs.pop("attn_implementation", None)
                    model = Qwen3TTSModel.from_pretrained(model_path_or_id, **kwargs)
            elif dtype_choice == "bfloat16":
                self.status.emit("bfloat16 load failed; retrying with float16...")
                kwargs["dtype"] = torch.float16
                model = Qwen3TTSModel.from_pretrained(model_path_or_id, **kwargs)
            else:
                raise

        self._model = model
        self._loaded_key = key
        self.status.emit("Model loaded.")
        return self._model


class TTSWorker(QtCore.QObject):
    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    log = QtCore.Signal(str)

    def __init__(self, mode: str, payload: dict, cache: ModelCache):
        super().__init__()
        self.mode = mode
        self.payload = payload
        self.cache = cache

    def _out_path(self, base_name: str, add_ts: bool) -> Path:
        name = (base_name or "qwen_tts").strip()
        if add_ts:
            name = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        return OUTPUT_DIR / f"{name}.wav"

    @QtCore.Slot()
    def run(self):
        try:
            common = self.payload["common"]
            model_path = self.payload["model_path"]
            model = self.cache.load_if_needed(model_path, common["device"], common["dtype"], common["attn_impl"])

            gen_kwargs = {
                "max_new_tokens": int(common["max_new_tokens"]),
                "temperature": float(common["temperature"]),
                "top_p": float(common["top_p"]),
                "top_k": int(common["top_k"]),
            }

            out_wav = self._out_path(common["output_name"], bool(common["add_timestamp"]))

            import soundfile as sf

            if self.mode == "custom":
                wavs, sr = model.generate_custom_voice(
                    text=self.payload["text"],
                    language=self.payload["language"],
                    speaker=safe_lower_speaker(self.payload["speaker"]),
                    instruct=self.payload["instruct"].strip() if self.payload["instruct"] else None,
                    non_streaming_mode=True,
                    **gen_kwargs,
                )
                sf.write(str(out_wav), wavs[0], sr)

            elif self.mode == "clone":
                ref_audio_path = self.payload["ref_audio_path"].strip()
                if not ref_audio_path:
                    raise ValueError("Please choose a reference audio file.")
                # qwen-tts supports passing a path directly
                wavs, sr = model.generate_voice_clone(
                    text=self.payload["text"],
                    language=self.payload["language"],
                    ref_audio=ref_audio_path,
                    ref_text=self.payload["ref_text"].strip() if self.payload["ref_text"] else None,
                    x_vector_only_mode=bool(self.payload["x_vector_only_mode"]),
                    instruct=self.payload["instruct"].strip() if self.payload["instruct"] else None,
                    non_streaming_mode=True,
                    **gen_kwargs,
                )
                sf.write(str(out_wav), wavs[0], sr)

            elif self.mode == "design":
                wavs, sr = model.generate_voice_design(
                    text=self.payload["text"],
                    language=self.payload["language"],
                    instruct=self.payload["voice_description"].strip(),
                    non_streaming_mode=True,
                    **gen_kwargs,
                )
                sf.write(str(out_wav), wavs[0], sr)

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            self.finished.emit(str(out_wav))
        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("Qwen3-TTS Desktop UI (CustomVoice / Voice Clone / Voice Design)")
        self.resize(1040, 800)

        self.cache = ModelCache()
        self.cache.status.connect(self._log)

        self._build_ui()
        self._load_last_state()
        self._update_gpu_info()
        self._update_model_status()

    # ---------- UI ----------
    def _build_ui(self):
        # Wrap the whole UI in a scroll area so smaller windows don't cause widgets to collide.
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
        # Top: model manager (collapsible)
        mm_content = QtWidgets.QWidget()
        mm_layout = QtWidgets.QGridLayout(mm_content)

        self.chk_tokenizer = QtWidgets.QCheckBox("Tokenizer (required)")
        self.chk_custom = QtWidgets.QCheckBox("CustomVoice (9 speakers)")
        self.chk_base = QtWidgets.QCheckBox("Base (needed for Voice Clone)")
        self.chk_design = QtWidgets.QCheckBox("VoiceDesign (prompt-made voices)")

        self.chk_tokenizer.setChecked(True)
        self.chk_custom.setChecked(True)

        self.btn_download = QtWidgets.QPushButton("Download selected")
        self.btn_download.setToolTip("Downloads via Hugging Face into models/.\n"
                                     "If you see auth errors, set HF_TOKEN in your environment.")
        self.btn_open_models = QtWidgets.QPushButton("Open models folder")
        self.btn_open_models.setToolTip("Open the local models folder.")

        self.lbl_model_status = QtWidgets.QLabel("")
        self.lbl_model_status.setWordWrap(True)

        mm_layout.addWidget(self.chk_tokenizer, 0, 0)
        mm_layout.addWidget(self.chk_custom, 0, 1)
        mm_layout.addWidget(self.chk_base, 1, 0)
        mm_layout.addWidget(self.chk_design, 1, 1)
        mm_layout.addWidget(self.btn_download, 0, 2, 1, 1)
        mm_layout.addWidget(self.btn_open_models, 1, 2, 1, 1)
        mm_layout.addWidget(self.lbl_model_status, 2, 0, 1, 3)

        self.section_mm = CollapsibleSection("Model Manager", mm_content, collapsed=True)
        self.section_mm.setToolTip("Download the models this UI supports into the local 'models' folder.\n"
                                   "This does NOT require admin rights and does not touch system Python.\n"
                                   "If a model already exists, it will be skipped.")
        layout.addWidget(self.section_mm)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)
        # Common settings (collapsible; used by all tabs)
        common_content = QtWidgets.QWidget()
        common_layout = QtWidgets.QFormLayout(common_content)

        self.device = QtWidgets.QComboBox()
        devs = ["auto", "cpu"]
        if torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        self.device.addItems(devs)
        self.device.setToolTip("Where to run the model.\n"
                               "Auto uses CUDA if available, otherwise CPU.")

        self.dtype = QtWidgets.QComboBox()
        self.dtype.addItems(["bfloat16", "float16", "float32"])
        self.dtype.setToolTip("Precision. bfloat16 is fastest on some GPUs, but not all.\n"
                              "If you see load errors, pick float16.")

        self.attn = QtWidgets.QComboBox()
        opts = ["flash_attention_2", "sdpa", "none"]
        if not HAS_FLASH_ATTN:
            opts.remove("flash_attention_2")
        self.attn.addItems(opts)
        self.attn.setCurrentText("sdpa")
        self.attn.setToolTip("Attention implementation.\n"
                             "SDPA is the safe default. FlashAttention2 needs flash_attn installed.")

        self.temperature = QtWidgets.QDoubleSpinBox()
        self.temperature.setRange(0.05, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setValue(0.8)
        self.temperature.setToolTip("Higher = more variation, lower = more stable.")

        self.top_p = QtWidgets.QDoubleSpinBox()
        self.top_p.setRange(0.05, 1.0)
        self.top_p.setSingleStep(0.01)
        self.top_p.setValue(0.95)
        self.top_p.setToolTip("Nucleus sampling cutoff (probability mass).")

        self.top_k = QtWidgets.QSpinBox()
        self.top_k.setRange(0, 200)
        self.top_k.setValue(50)
        self.top_k.setToolTip("Top-k sampling. 0 disables top-k.")

        self.max_new_tokens = QtWidgets.QSpinBox()
        self.max_new_tokens.setRange(64, 8192)
        self.max_new_tokens.setValue(2048)
        self.max_new_tokens.setToolTip("Upper bound for generation length. Too high may slow things down.")

        out_row = QtWidgets.QHBoxLayout()
        self.output_name = QtWidgets.QLineEdit("qwen_tts")
        self.output_name.setToolTip("Base filename (without extension).")
        self.add_timestamp = QtWidgets.QCheckBox("Add timestamp")
        self.add_timestamp.setChecked(True)
        self.add_timestamp.setToolTip("Appends YYYYMMDD_HHMMSS to avoid overwriting.")
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
        self.section_common.setToolTip("These settings apply to ALL modes.\n"
                                       "If something fails to load, try dtype=float16 and attention=sdpa.")
        layout.addWidget(self.section_common)

        # Tab: CustomVoice
        self.tab_custom = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_custom, "CustomVoice")
        self._build_custom_tab()

        # Tab: Voice Clone
        self.tab_clone = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_clone, "Voice Clone (Base)")
        self._build_clone_tab()

        # Tab: Voice Design
        self.tab_design = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_design, "Voice Design")
        self._build_design_tab()

        # Bottom: buttons + log
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Generate WAV (current tab)")
        self.btn_generate.setToolTip("Generates audio using the currently selected mode/tab.")
        self.btn_open_output = QtWidgets.QPushButton("Open output folder")
        self.btn_open_output.setToolTip("Open output/ where WAVs are saved.")
        self.btn_save_preset = QtWidgets.QPushButton("Save preset")
        self.btn_load_preset = QtWidgets.QPushButton("Load preset")

        btn_row.addWidget(self.btn_generate)
        btn_row.addWidget(self.btn_open_output)
        btn_row.addWidget(self.btn_save_preset)
        btn_row.addWidget(self.btn_load_preset)

        layout.addLayout(btn_row)
        self.lbl_gpu = QtWidgets.QLabel("")

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

        # Signals
        self.btn_generate.clicked.connect(self._on_generate)
        self.btn_open_output.clicked.connect(self._open_output)
        self.btn_save_preset.clicked.connect(self._save_preset)
        self.btn_load_preset.clicked.connect(self._load_preset)
        self.btn_open_models.clicked.connect(self._open_models)
        self.btn_download.clicked.connect(self._download_selected)

        # Remember collapse state immediately
        self.section_mm.toggled.connect(lambda _c: self._save_last_state())
        self.section_common.toggled.connect(lambda _c: self._save_last_state())
        self.section_logs.toggled.connect(lambda _c: self._save_last_state())

    def _build_custom_tab(self):
        lay = QtWidgets.QFormLayout(self.tab_custom)

        self.cv_model_path = QtWidgets.QLineEdit(DEFAULT_CUSTOMVOICE_DIR)
        self.cv_model_path.setToolTip("Folder for the CustomVoice model. Default is models/Qwen3-TTS-12Hz-1.7B-CustomVoice")
        self.cv_tokenizer_path = QtWidgets.QLineEdit(DEFAULT_TOKENIZER_DIR)
        self.cv_tokenizer_path.setToolTip("Folder for the tokenizer. Default is models/Qwen3-TTS-Tokenizer")

        self.cv_text = QtWidgets.QPlainTextEdit()
        self.cv_text.setPlainText("Okay, quick test: the rain taps the window like tiny drums. If you can hear this clearly, it works.")
        self.cv_text.setToolTip("The text to speak.")

        self.cv_language = QtWidgets.QComboBox()
        self.cv_language.addItems(SUPPORTED_LANGS_FALLBACK)
        self.cv_language.setCurrentText("English")
        self.cv_language.setToolTip("Language label. Auto is fine if unsure.")

        self.cv_speaker = QtWidgets.QComboBox()
        self.cv_speaker.setEditable(True)
        self.cv_speaker.addItems(SUPPORTED_SPEAKERS_FALLBACK)
        self.cv_speaker.setCurrentText("ryan")
        self.cv_speaker.setToolTip("Built-in speakers supported by CustomVoice.\n"
                                   "If you type an unsupported name, generation will fail.")

        self.cv_instruct = QtWidgets.QLineEdit("")
        self.cv_instruct.setToolTip("Style instruction (delivery). Example:\n"
                                    "\"casual, friendly, short pauses, slightly amused\"")

        self.btn_cv_refresh = QtWidgets.QPushButton("Refresh speakers from model")
        self.btn_cv_refresh.setToolTip("Loads the model quickly (once) to ask it for supported speakers/languages,\n"
                                       "then updates the dropdowns.")
        self.btn_cv_refresh.clicked.connect(self._refresh_supported_from_model)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.cv_speaker, 2)
        row.addWidget(self.btn_cv_refresh, 1)

        lay.addRow("Model path", self.cv_model_path)
        lay.addRow("Tokenizer path", self.cv_tokenizer_path)
        lay.addRow("Text", self.cv_text)
        lay.addRow("Language", self.cv_language)
        lay.addRow("Speaker", row)
        lay.addRow("Style instruction", self.cv_instruct)

    def _build_clone_tab(self):
        lay = QtWidgets.QFormLayout(self.tab_clone)

        self.vc_model_path = QtWidgets.QLineEdit(DEFAULT_BASE_DIR)
        self.vc_model_path.setToolTip("Folder for the Base model (needed for voice cloning).")
        self.vc_tokenizer_path = QtWidgets.QLineEdit(DEFAULT_TOKENIZER_DIR)

        self.vc_text = QtWidgets.QPlainTextEdit()
        self.vc_text.setPlainText("Hey—this is a quick voice clone test. If this sounds like the reference speaker, it worked.")
        self.vc_language = QtWidgets.QComboBox()
        self.vc_language.addItems(SUPPORTED_LANGS_FALLBACK)
        self.vc_language.setCurrentText("English")

        ref_row = QtWidgets.QHBoxLayout()
        self.vc_ref_audio = QtWidgets.QLineEdit("")
        self.vc_ref_audio.setPlaceholderText("Select a reference WAV/MP3…")
        self.vc_ref_audio.setToolTip("Reference audio file. Short, clean speech works best (5–20 seconds).")
        self.btn_pick_audio = QtWidgets.QPushButton("Browse…")
        self.btn_pick_audio.setToolTip("Choose a reference audio file for cloning.")
        self.btn_pick_audio.clicked.connect(self._pick_ref_audio)
        ref_row.addWidget(self.vc_ref_audio, 3)
        ref_row.addWidget(self.btn_pick_audio, 1)

        self.vc_ref_text = QtWidgets.QLineEdit("")
        self.vc_ref_text.setToolTip("Optional but recommended: exact transcript of the reference audio.\n"
                                    "Helps the clone match identity more reliably.")
        self.vc_xvector = QtWidgets.QCheckBox("x_vector_only_mode (no ref_text)")
        self.vc_xvector.setToolTip("If enabled, the model won't require ref_text.\n"
                                   "Quality may be worse than using a transcript.")

        self.vc_instruct = QtWidgets.QLineEdit("")
        self.vc_instruct.setToolTip("Delivery instruction (not identity). Example: \"slow, warm, conversational\"")

        lay.addRow("Model path", self.vc_model_path)
        lay.addRow("Tokenizer path", self.vc_tokenizer_path)
        lay.addRow("Text", self.vc_text)
        lay.addRow("Language", self.vc_language)
        lay.addRow("Reference audio", ref_row)
        lay.addRow("Reference text", self.vc_ref_text)
        lay.addRow("", self.vc_xvector)
        lay.addRow("Style instruction", self.vc_instruct)

    def _build_design_tab(self):
        lay = QtWidgets.QFormLayout(self.tab_design)

        self.vd_model_path = QtWidgets.QLineEdit(DEFAULT_VOICEDESIGN_DIR)
        self.vd_model_path.setToolTip("Folder for the VoiceDesign model (prompt-created voices).")
        self.vd_tokenizer_path = QtWidgets.QLineEdit(DEFAULT_TOKENIZER_DIR)

        self.vd_text = QtWidgets.QPlainTextEdit()
        self.vd_text.setPlainText("Hello. This is voice design. If the voice matches the description, it worked.")
        self.vd_language = QtWidgets.QComboBox()
        self.vd_language.addItems(SUPPORTED_LANGS_FALLBACK)
        self.vd_language.setCurrentText("English")

        self.vd_desc = QtWidgets.QPlainTextEdit()
        self.vd_desc.setPlainText("A calm, friendly voice, slightly smiling, clear pronunciation.")
        self.vd_desc.setToolTip("Describe the voice identity here (age, tone, accent, energy).\n"
                                "Example: \"young adult, gentle British accent, playful and curious\"")

        lay.addRow("Model path", self.vd_model_path)
        lay.addRow("Tokenizer path", self.vd_tokenizer_path)
        lay.addRow("Text", self.vd_text)
        lay.addRow("Language", self.vd_language)
        lay.addRow("Voice description", self.vd_desc)

    # ---------- Helpers ----------
    def _log(self, s: str):
        self.log.appendPlainText(s)

    def _update_gpu_info(self):
        if torch.cuda.is_available():
            parts = []
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                parts.append(f"cuda:{i} = {name}")
            self.lbl_gpu.setText("GPU: " + " | ".join(parts))
        else:
            self.lbl_gpu.setText("GPU: none (CPU mode)")

    def _open_output(self):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(OUTPUT_DIR))

    def _open_models(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(str(MODELS_DIR))

    def _update_model_status(self):
        def ok(p: str) -> bool:
            d = Path(p)
            return d.exists() and any(d.iterdir())

        rows = []
        rows.append(("Tokenizer", ok(DEFAULT_TOKENIZER_DIR), DEFAULT_TOKENIZER_DIR))
        rows.append(("CustomVoice", ok(DEFAULT_CUSTOMVOICE_DIR), DEFAULT_CUSTOMVOICE_DIR))
        rows.append(("Base", ok(DEFAULT_BASE_DIR), DEFAULT_BASE_DIR))
        rows.append(("VoiceDesign", ok(DEFAULT_VOICEDESIGN_DIR), DEFAULT_VOICEDESIGN_DIR))

        msg = []
        for name, present, path in rows:
            msg.append(f"{'✓' if present else '✗'} {name}: {Path(path).name}")
        self.lbl_model_status.setText(" | ".join(msg))

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
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

        self.dl_thread = QtCore.QThread()
        self.dl_worker = HFDownloadWorker(items, token=token)
        self.dl_worker.moveToThread(self.dl_thread)
        self.dl_thread.started.connect(self.dl_worker.run)
        self.dl_worker.log.connect(self._log)
        self.dl_worker.done.connect(self._on_download_done)
        self.dl_worker.done.connect(self.dl_thread.quit)
        self.dl_thread.finished.connect(self.dl_thread.deleteLater)
        self.dl_thread.start()

    def _on_download_done(self, ok: bool):
        self.btn_download.setEnabled(True)
        self._update_model_status()
        if ok:
            self._log("Downloads completed.")
        else:
            self._log("Downloads finished with errors (see above).")

    def _pick_ref_audio(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select reference audio", "", "Audio (*.wav *.mp3 *.flac *.m4a);;All files (*.*)")
        if path:
            self.vc_ref_audio.setText(path)

    def _refresh_supported_from_model(self):
        # This loads the CustomVoice model if available, then queries supported speakers/languages.
        try:
            model_dir = self.cv_model_path.text().strip()
            if not (Path(model_dir).exists() and any(Path(model_dir).iterdir())):
                self._log("CustomVoice model folder not found. Download it in Model Manager first.")
                return

            common = self._collect_common()
            model = self.cache.load_if_needed(model_dir, common["device"], common["dtype"], common["attn_impl"])

            speakers = []
            langs = []
            try:
                speakers = list(model.get_supported_speakers())
            except Exception:
                speakers = SUPPORTED_SPEAKERS_FALLBACK
            try:
                langs = list(model.get_supported_languages())
            except Exception:
                langs = SUPPORTED_LANGS_FALLBACK

            if speakers:
                cur = self.cv_speaker.currentText()
                self.cv_speaker.blockSignals(True)
                self.cv_speaker.clear()
                self.cv_speaker.addItems([safe_lower_speaker(x) for x in speakers])
                self.cv_speaker.setEditable(True)
                self.cv_speaker.setCurrentText(safe_lower_speaker(cur) if safe_lower_speaker(cur) in [safe_lower_speaker(x) for x in speakers] else safe_lower_speaker(speakers[0]))
                self.cv_speaker.blockSignals(False)

            if langs:
                cur = self.cv_language.currentText()
                self.cv_language.blockSignals(True)
                self.cv_language.clear()
                self.cv_language.addItems([str(x) for x in langs])
                self.cv_language.setCurrentText(cur if cur in langs else str(langs[0]))
                self.cv_language.blockSignals(False)

            self._log("Refreshed supported speakers/languages from model.")
        except Exception:
            self._log("ERROR refreshing supported lists:\n" + traceback.format_exc())

    # ---------- Presets ----------
    def _preset_path(self, name: str) -> Path:
        safe = "".join(c for c in name if c.isalnum() or c in ("-", "_", " ")).strip().replace(" ", "_")
        if not safe:
            safe = "preset"
        return PRESETS_DIR / f"{safe}.json"

    def _state_file(self) -> Path:
        return PRESETS_DIR / "_last_state.json"

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
        if tab is self.tab_custom:
            return "custom", {
                "text": self.cv_text.toPlainText(),
                "language": self.cv_language.currentText(),
                "speaker": self.cv_speaker.currentText(),
                "instruct": self.cv_instruct.text(),
                "model_path": self.cv_model_path.text().strip(),
                "tokenizer_path": self.cv_tokenizer_path.text().strip(),
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
                "tokenizer_path": self.vc_tokenizer_path.text().strip(),
                "common": common,
            }
        return "design", {
            "text": self.vd_text.toPlainText(),
            "language": self.vd_language.currentText(),
            "voice_description": self.vd_desc.toPlainText(),
            "model_path": self.vd_model_path.text().strip(),
            "tokenizer_path": self.vd_tokenizer_path.text().strip(),
            "common": common,
        }

    def _save_last_state(self):
        try:
            state = {
                "tab": self.tabs.currentIndex(),
                "ui": {
                    "mm_collapsed": bool(self.section_mm.isCollapsed()),
                    "common_collapsed": bool(self.section_common.isCollapsed()),
                    "logs_collapsed": bool(self.section_logs.isCollapsed()),
                },
                "common": self._collect_common(),
                "custom": {
                    "model_path": self.cv_model_path.text(),
                    "tokenizer_path": self.cv_tokenizer_path.text(),
                    "text": self.cv_text.toPlainText(),
                    "language": self.cv_language.currentText(),
                    "speaker": self.cv_speaker.currentText(),
                    "instruct": self.cv_instruct.text(),
                },
                "clone": {
                    "model_path": self.vc_model_path.text(),
                    "tokenizer_path": self.vc_tokenizer_path.text(),
                    "text": self.vc_text.toPlainText(),
                    "language": self.vc_language.currentText(),
                    "ref_audio_path": self.vc_ref_audio.text(),
                    "ref_text": self.vc_ref_text.text(),
                    "x_vector_only_mode": bool(self.vc_xvector.isChecked()),
                    "instruct": self.vc_instruct.text(),
                },
                "design": {
                    "model_path": self.vd_model_path.text(),
                    "tokenizer_path": self.vd_tokenizer_path.text(),
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

            # UI (collapse state)
            ui = state.get("ui", {}) or {}
            # Default is collapsed; only expand if state says so.
            try:
                self.section_mm.setCollapsed(bool(ui.get("mm_collapsed", True)))
                self.section_common.setCollapsed(bool(ui.get("common_collapsed", True)))
                self.section_logs.setCollapsed(bool(ui.get("logs_collapsed", True)))
            except Exception:
                pass

            # common
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

            cv = state.get("custom", {})
            if cv:
                self.cv_model_path.setText(cv.get("model_path", DEFAULT_CUSTOMVOICE_DIR))
                self.cv_tokenizer_path.setText(cv.get("tokenizer_path", DEFAULT_TOKENIZER_DIR))
                self.cv_text.setPlainText(cv.get("text", ""))
                self.cv_language.setCurrentText(cv.get("language", "English"))
                self.cv_speaker.setCurrentText(safe_lower_speaker(cv.get("speaker", "ryan")))
                self.cv_instruct.setText(cv.get("instruct", ""))

            vc = state.get("clone", {})
            if vc:
                self.vc_model_path.setText(vc.get("model_path", DEFAULT_BASE_DIR))
                self.vc_tokenizer_path.setText(vc.get("tokenizer_path", DEFAULT_TOKENIZER_DIR))
                self.vc_text.setPlainText(vc.get("text", ""))
                self.vc_language.setCurrentText(vc.get("language", "English"))
                self.vc_ref_audio.setText(vc.get("ref_audio_path", ""))
                self.vc_ref_text.setText(vc.get("ref_text", ""))
                self.vc_xvector.setChecked(bool(vc.get("x_vector_only_mode", False)))
                self.vc_instruct.setText(vc.get("instruct", ""))

            vd = state.get("design", {})
            if vd:
                self.vd_model_path.setText(vd.get("model_path", DEFAULT_VOICEDESIGN_DIR))
                self.vd_tokenizer_path.setText(vd.get("tokenizer_path", DEFAULT_TOKENIZER_DIR))
                self.vd_text.setPlainText(vd.get("text", ""))
                self.vd_language.setCurrentText(vd.get("language", "English"))
                self.vd_desc.setPlainText(vd.get("voice_description", ""))

            self.tabs.setCurrentIndex(int(state.get("tab", 0)))
        except Exception:
            pass

    def _save_preset(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok:
            return
        path = self._preset_path(name)
        state = json.loads(self._state_file().read_text(encoding="utf-8")) if self._state_file().exists() else {}
        self._save_last_state()
        state = json.loads(self._state_file().read_text(encoding="utf-8"))
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        self._log(f"Saved preset: {path.name}")

    def _load_preset(self):
        files = sorted(PRESETS_DIR.glob("*.json"))
        files = [f for f in files if f.name != "_last_state.json"]
        if not files:
            self._log("No presets found in presets/setsave/.")
            return
        items = [f.name for f in files]
        choice, ok = QtWidgets.QInputDialog.getItem(self, "Load preset", "Choose preset:", items, 0, False)
        if not ok:
            return
        data = json.loads((PRESETS_DIR / choice).read_text(encoding="utf-8"))
        # Write to last state then load
        self._state_file().write_text(json.dumps(data, indent=2), encoding="utf-8")
        self._load_last_state()
        self._log(f"Loaded preset: {choice}")

    # ---------- Generate ----------
    def _on_generate(self):
        self._save_last_state()

        mode, payload = self._collect_payload()

        # quick checks
        mpath = payload.get("model_path", "")
        if not (mpath and Path(mpath).exists() and any(Path(mpath).iterdir())):
            self._log("Model folder missing. Use Model Manager to download it first.")
            return

        self._log(f"Starting generation ({mode})...")
        self.btn_generate.setEnabled(False)

        self.thread = QtCore.QThread()
        self.worker = TTSWorker(mode, payload, self.cache)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _on_done(self, out_path: str):
        self._log(f"Done: {out_path}")
        self.btn_generate.setEnabled(True)
        self._update_model_status()

    def _on_failed(self, tb: str):
        self._log("ERROR:\n" + tb)
        self.btn_generate.setEnabled(True)


    def closeEvent(self, event):
        try:
            self._save_last_state()
        except Exception:
            pass
        return super().closeEvent(event)


def main():
    ensure_dirs()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()