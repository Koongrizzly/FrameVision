import json
import os
import sys
import re
import subprocess
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer, QProcessEnvironment
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QToolButton, QFileDialog,
    QMessageBox, QGroupBox, QFormLayout, QInputDialog, QTabWidget, QPlainTextEdit, QScrollArea, QLayout
)
from PySide6.QtGui import QImage, QPixmap


# ---- PromptTool preset catalog (reused for the Qwen prompt enhancer) ----
# We import the same preset definitions used by Tools → Prompt enhancement (prompt.py),
# but keep Hunyuan's selection saved in hunyuan15_settings.json (so it does NOT override the Prompt tool).
try:
    from helpers import prompt as _prompt_mod  # type: ignore
except Exception:
    try:
        import prompt as _prompt_mod  # type: ignore
    except Exception:
        _prompt_mod = None

try:
    _PROMPT_PRESET_DEFS = getattr(_prompt_mod, "PRESET_DEFS", {}) or {}
except Exception:
    _PROMPT_PRESET_DEFS = {}

try:
    _PROMPT_LENGTH_PRESETS = getattr(_prompt_mod, "LENGTH_PRESETS", {}) or {}
except Exception:
    _PROMPT_LENGTH_PRESETS = {}

if "Default" not in _PROMPT_PRESET_DEFS:
    _PROMPT_PRESET_DEFS = dict({"Default": {}}, **(_PROMPT_PRESET_DEFS or {}))

def _merge_neg_csv(base: str, extra: str) -> str:
    """Combine and de-duplicate comma-separated negatives while preserving order (case-insensitive)."""
    parts: list[str] = []
    for chunk in (base or "", extra or ""):
        if not chunk:
            continue
        for t in [x.strip() for x in str(chunk).split(",")]:
            if not t:
                continue
            if t.lower() not in [p.lower() for p in parts]:
                parts.append(t)
    return ", ".join(parts)

APP_TITLE = "HunyuanVideo-1.5 — One Click (CUDA RTX)"

MAX_LOG_LINES = 1500


# ---- Sidecar JSON for outputs (Direct run + Queue) ----
_QUEUE_SIDECAR_META_FLAG = "--__sidecar_meta"

# Queue jobs run in the worker with an argv list (no shell). To write a sidecar JSON
# without modifying the CLI script, we wrap the CLI call via python -c + runpy, then
# write <output>.json on success.
_QUEUE_SIDECAR_WRAPPER_CODE = r"""
import sys, json, runpy
from pathlib import Path

def _pop_meta(argv):
    meta = None
    if '--__sidecar_meta' in argv:
        i = argv.index('--__sidecar_meta')
        if i + 1 < len(argv):
            try:
                meta = json.loads(argv[i+1])
            except Exception:
                meta = None
        # Remove flag + value so the CLI never sees it.
        try:
            del argv[i:i+2]
        except Exception:
            pass
    return meta, argv

def _find_output_path(argv):
    # argv is like: [cli.py, generate, ..., --output, X, ...]
    out = None
    for j, a in enumerate(argv):
        if a == '--output' and j + 1 < len(argv):
            out = argv[j+1]
            break
    return out

def main():
    argv = list(sys.argv[1:])
    if not argv:
        raise SystemExit(2)

    meta, argv = _pop_meta(argv)

    cli_path = Path(argv[0]).resolve()
    cli_argv = [str(cli_path)] + argv[1:]
    sys.argv = cli_argv

    exit_code = 0
    try:
        runpy.run_path(str(cli_path), run_name='__main__')
    except SystemExit as e:
        try:
            exit_code = int(e.code) if e.code is not None else 0
        except Exception:
            exit_code = 1
    except Exception:
        exit_code = 1

    if exit_code == 0:
        out = None
        try:
            if isinstance(meta, dict) and meta.get('output'):
                out = str(meta.get('output'))
        except Exception:
            out = None
        if not out:
            out = _find_output_path(argv)

        if out:
            try:
                outp = Path(out)
                if outp.is_file():
                    jpath = outp.with_suffix('.json')
                    payload = meta if isinstance(meta, dict) else {}
                    payload = dict(payload)
                    payload.setdefault('output', str(outp))
                    tmp = jpath.with_suffix(jpath.suffix + '.tmp')
                    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
                    tmp.replace(jpath)
            except Exception as ex:
                try:
                    print(f'[sidecar] failed: {ex}')
                except Exception:
                    pass

    raise SystemExit(exit_code)

if __name__ == '__main__':
    main()
"""
# Extend-chain merge seam fix (Hunyuan 1.5)
# Hunyuan segments can "pause" for a split second at the start of each new chunk.
# These settings help hide the join by trimming the start of appended chunks before merging.
# - DROP_FRAMES: frames removed from the START of every appended segment (not the first).
# - BLEND_FRAMES: optional micro crossfade length in frames (0 disables; concat-only).
EXTEND_JOIN_DROP_FRAMES = 1
EXTEND_JOIN_BLEND_FRAMES = 1

MODEL_PRESETS = [
    ("480p Text-to-Video", "480p_t2v"),
    ("480p Text-to-Video (distilled)", "480p_t2v_distilled"),
    ("720p Text-to-Video", "720p_t2v"),
    ("480p Image-to-Video (step-distilled, 8–12 steps)", "480p_i2v_step_distilled"),
    ("480p Image-to-Video (distilled", "480p_i2v_distilled"),
    ("480p Image-to-Video", "480p_i2v"),
    ("720p Image-to-Video (distilled)", "720p_i2v_distilled"),
    ("720p Image-to-Video", "720p_i2v"),
    
]

ATTN_PRESETS = [
    ("Auto", "auto"),
    ("flash_hub (4090/A100 class)", "flash_hub"),
    ("flash_varlen_hub", "flash_varlen_hub"),
    ("sage_hub (fallback)", "sage_hub"),
    ("Default (no override)", "default"),
]

RESOLUTION_PRESETS = [
    ("192p (for quick testing)", 240),
    ("240p", 320),
    ("288p", 384),
    ("304p", 420),
    ("368p", 480),
    ("384p", 512),
    ("432p", 576),
    ("444p", 600),
    ("480p", 650),
    ("496p (for 720p models)", 672),
    ("528p (for 720p models)", 704),
    ("576p (for 720p models)", 768),
    ("624p (for 720p models)", 832),
    ("672p (for 720p models)", 896),
    ("720p (for 720p models)", 960),
    
]



# Auto Aspect Ratio mode uses a single "bucket" number (target_size) instead of width/height.
# Lower = less VRAM, Higher = more detail but more VRAM.
AUTO_BUCKET_PRESETS = [
    ("Auto bucket: 192p-ish (small)", 240),
    ("Auto bucket: 240p-ish", 320),
    ("Auto bucket: 288p-ish (recommended)", 384),
    ("Auto bucket: 304p-ish", 420),
    ("Auto bucket: 368p-ish", 480),
    ("Auto bucket: 384p-ish", 512),
]



def root_dir() -> Path:
    """Best-effort app root resolution."""
    try:
        from helpers.framevision_app import ROOT as _ROOT  # type: ignore
        return Path(_ROOT).resolve()
    except Exception:
        return Path(__file__).resolve().parents[1]


def settings_path() -> Path:
    return root_dir() / "presets" / "presets" / "hunyuan15_settings.json"


def _ffmpeg_exe() -> str | None:
    """Best-effort lookup for ffmpeg.exe (used for extend-chain helpers)."""
    try:
        base = root_dir()
    except Exception:
        base = Path(__file__).resolve().parents[1]

    candidates = [
        base / "bin" / "ffmpeg.exe",
        base / "ffmpeg.exe",
        base / "presets" / "bin" / "ffmpeg.exe",
        base / "presets" / "ffmpeg.exe",
        base / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
        base / "presets" / "tools" / "ffmpeg" / "bin" / "ffmpeg.exe",
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def _ffprobe_exe() -> str | None:
    """Best-effort lookup for ffprobe.exe (used for seam-blended merges)."""
    try:
        base = root_dir()
    except Exception:
        base = Path(__file__).resolve().parents[1]

    candidates = [
        base / "bin" / "ffprobe.exe",
        base / "ffprobe.exe",
        base / "presets" / "bin" / "ffprobe.exe",
        base / "presets" / "ffprobe.exe",
        base / "tools" / "ffmpeg" / "bin" / "ffprobe.exe",
        base / "presets" / "tools" / "ffmpeg" / "bin" / "ffprobe.exe",
    ]
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return str(p)
        except Exception:
            continue
    return None


def _temp_dir() -> Path:
    d = root_dir() / "temp"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _extend_frames_dir() -> Path:
    d = root_dir() / "output" / "video" / "hunyuan15" / "_extend_frames"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def load_settings() -> dict:
    p = settings_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_settings(data: dict) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


class Hunyuan15ToolWidget(QWidget):
    """Embedded Tools-tab widget for HunyuanVideo 1.5."""

    def __init__(self, parent=None, standalone: bool = False):
        super().__init__(parent)

        if standalone:
            self.setWindowTitle(APP_TITLE)
            self.resize(980, 680)

        self.proc: QProcess | None = None

        # Extend-chain state (initial clip → chained segments using last frame as next start image)
        self._extend_active: bool = False
        self._extend_remaining: int = 0
        self._extend_segments: list[Path] = []
        self._extend_frame_index: int = 0

        # Video-to-video state (use last frame of a chosen source video as the start image)
        self._video2video_path: Path | None = None
        self._extend_auto_merge: bool = False
        self._extend_pending_output: Path | None = None
        self._extend_base_out: Path | None = None
        self._extend_base_model: str | None = None
        self._extend_base_prompt: str = ""
        self._extend_base_seed: int = -1

        # Remember exact output path used for the last direct run (helps extend bookkeeping)
        self._last_run_out_path: Path | None = None

        # Track what the current QProcess was launched for (install/download/generate)
        self._current_task: str | None = None

        # Pending sidecar JSON write for Direct run (and extend segments)
        self._pending_sidecar_out: Path | None = None
        self._pending_sidecar_meta: dict | None = None

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        # Log storage: newest lines at the top (no scrolling needed).
        self._log_lines: list[str] = []
        self._log_pending: list[str] = []
        self._log_direct_active: bool = False
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(3000)
        self._log_flush_timer.timeout.connect(self._flush_pending_logs)

        self.btn_install = QPushButton("Install/Update Cuda")
        self.btn_download = QPushButton("Download Models")
        self.btn_generate = QPushButton("Direct run")
        try:
            self.btn_generate.setToolTip(
                "Direct run starts immediately (no Queue).\n"
                "Tip: avoid running other heavy VRAM apps at the same time.\n"
                "If \"Use Queue\" is ON, this button will queue the job instead."
            )
        except Exception:
            pass
        self.btn_queue = QPushButton("Use Queue")
        self.btn_queue.setCheckable(True)
        self.btn_queue.setChecked(False)
        self.btn_batch = QPushButton("Batch")
        self.btn_use_current = QPushButton("Use Current")
        self.btn_use_current.setToolTip("Use the current Media Player frame/image as the Start image (I2V).")
        self.btn_stop = QPushButton("Cancel")
        self.btn_open_out = QPushButton("View results")
        self.btn_open_out.setToolTip("Open Media Explorer and scan the current output folder.")
        self.btn_clear_log = QPushButton("Clear Log")
        self.btn_clear_log.setToolTip("Clear the log output in this panel.")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setToolTip("Stop the currently running process (if any).")
        self.btn_queue.setToolTip(
            "Toggle Queue mode.\n"
            "ON: clicking \"Generate Video\" will queue the job to the Queue tab.\n\n"
            "Tip: Extend (extra segments) requires Direct run, so Queue will be disabled while Extend > 0."
        )
        self.btn_batch.setToolTip(
            "Queue multiple jobs with the same prompt/settings (useful for seed variations)."
        )

        self.prompt = QPlainTextEdit()
        self.prompt.setPlaceholderText("Describe the video you want (English works best).")
        # Prompt: multi-line (at least 3 lines)
        try:
            fm = self.prompt.fontMetrics()
            h = int(fm.lineSpacing() * 3 + 18)
            self.prompt.setFixedHeight(max(h, 72))
        except Exception:
            try:
                self.prompt.setFixedHeight(72)
            except Exception:
                pass

        # Negative prompt (optional)
        self.negative_prompt = QPlainTextEdit()
        try:
            self.negative_prompt.setPlaceholderText("Things you do NOT want to see (optional).")
        except Exception:
            pass
        # Keep it compact
        try:
            self.negative_prompt.setFixedHeight(64)
        except Exception:
            pass


        # Optional: Start image for Image-to-Video models
        self.start_image = QLineEdit()
        self.start_image.setPlaceholderText("Optional: pick a start image for Image-to-Video models (use *_i2v models).")
        self.btn_pick_image = QToolButton()
        self.btn_pick_image.setText("...")
        self.btn_pick_image.setToolTip("Choose a start image (for *_i2v models).")
        self.btn_clear_image = QToolButton()
        self.btn_clear_image.setText("Clear")
        self.btn_clear_image.setToolTip("Clear the start image.")



        # Video-to-video helper: pick a source video, use its last frame as Start image
        self.chk_video2video = QCheckBox("")
        self.chk_video2video.setToolTip("Pick a source video; its last frame becomes the Start image for Image-to-Video (I2V).")
        self.btn_pick_v2v = QToolButton()
        self.btn_pick_v2v.setText("load video")
        self.btn_pick_v2v.setToolTip("Choose a source video; we'll extract its last frame as the Start image.")
        self.btn_use_last_v2v = QToolButton()
        self.btn_use_last_v2v.setText("Use last")
        self.btn_use_last_v2v.setToolTip("Use the most recently created output video as the source.")
        self.lbl_video2_info = QLabel("No source selected")
        self.lbl_video2_info.setToolTip("Current Video→Video source video.")
        try:
            self.btn_pick_v2v.setEnabled(False)
            self.btn_use_last_v2v.setEnabled(False)
        except Exception:
            pass

        self.model = QComboBox()
        for label, key in MODEL_PRESETS:
            self.model.addItem(label, key)

        self.attn = QComboBox()
        for label, key in ATTN_PRESETS:
            self.attn.addItem(label, key)
        try:
            self.attn.setToolTip(
                "<b>Auto is recommended.</b><br>"
                "Auto will try fast attention backends (if available) and fall back safely if not.<br>"
                "<b>Windows:</b> Kernel Hub backends may print errors until Windows kernels are released. "
                "You can ignore those messages; generation will fall back to a working backend (default or flash)."
            )
        except Exception:
            pass

        self.resolution = QComboBox()
        for label, h in RESOLUTION_PRESETS:
            self.resolution.addItem(label, h)
        # default to 480p
        try:
            idx = self.resolution.findData(480)
            if idx >= 0:
                self.resolution.setCurrentIndex(idx)
        except Exception:
            pass

        # Aspect ratio for manual Resolution presets
        # - Landscape: 16:9 (default)
        # - Portrait: 9:16
        # - Square: 1:1
        self.aspect = QComboBox()
        self.aspect.addItem("Landscape (16:9)", "landscape")
        self.aspect.addItem("Portrait (9:16)", "portrait")
        self.aspect.addItem("Square (1:1)", "square")
        try:
            self.aspect.setCurrentIndex(0)
        except Exception:
            pass
        try:
            self.aspect.setToolTip(
                "Pick the output aspect ratio when Auto Aspect Ratio is OFF.\n"
                "Landscape is the default. Portrait lets you render 9:16 using the same low-res presets."
            )
        except Exception:
            pass

        self.frames = QSpinBox()
        self.frames.setRange(16, 241)
        self.frames.setValue(61)

        self.steps = QSpinBox()
        self.steps.setRange(4, 80)
        self.steps.setValue(30)

        self.fps = QSpinBox()
        self.fps.setRange(1, 60)
        self.fps.setValue(15)

        self.bitrate_kbps = QSpinBox()
        self.bitrate_kbps.setRange(0, 50000)
        self.bitrate_kbps.setValue(2000)
        self.bitrate_kbps.setToolTip(
            "Target bitrate for the final MP4 encode in kbps (uses ffmpeg). "
            "Set to 0 to disable re-encode and keep the raw export."
        )

        self.seed = QSpinBox()
        self.seed.setRange(-1, 2_000_000_000)
        self.seed.setValue(-1)

        # Extend-chain controls (Direct run only)
        self.extend = QSpinBox()
        self.extend.setRange(0, 25)
        self.extend.setValue(0)
        self.extend.setToolTip(
            "Extend the video by generating extra chained segments. "
            "Each new segment uses the last frame of the previous segment as the next start image. "
            "Direct run only (Queue/Batch will ignore this)."
        )

        self.extend_merge = QCheckBox("Auto-merge segments")
        self.extend_merge.setChecked(True)
        self.extend_merge.setToolTip(
            "After all segments are generated, merge them into a single MP4 using ffmpeg concat (fast copy when possible)."
        )

        self.offload = QCheckBox("Enable model CPU offload")
        try:
            # Qt tooltips support rich text; we use <b> to highlight the key guidance for new users.
            self.offload.setToolTip(
                "<b>Recommended if you have less than 24 GB VRAM.</b><br>"
                "Moves parts of the model to system RAM to avoid OOM, but can slow generation.<br>"
                "Good default for 720p / longer runs on 8–16 GB GPUs."
            )
        except Exception:
            pass

        # --- Extra advanced toggles (Diffusers hooks) ---
        # Group Offload (experimental)
        self.group_offload = QCheckBox("Enable Group Offload")
        try:
            self.group_offload.setToolTip(
                "<b>Experimental:</b> Offload transformer blocks to CPU between steps for lower VRAM.<br>"
                "Can help on 8–16 GB GPUs, but may slow down generation.<br>"
                "If you already use <b>Model CPU offload</b>, keep this OFF (they can conflict)."
            )
        except Exception:
            pass

        # First Block Cache (dynamic caching)
        self.first_block_cache = QCheckBox("Enable FirstBlockCache")
        try:
            self.first_block_cache.setToolTip(
                "<b>Speed booster:</b> skips parts of the transformer when changes are small.<br>"
                "Higher threshold = faster but can reduce quality. Lower threshold = safer but less speedup."
            )
        except Exception:
            pass

        # Threshold controls (only visible when FirstBlockCache is enabled)
        self.fbc_thresh_slider = QSlider(Qt.Horizontal)
        self.fbc_thresh_slider.setRange(0, 500)  # 0.000 → 0.500
        self.fbc_thresh_slider.setSingleStep(1)
        self.fbc_thresh_spin = QDoubleSpinBox()
        self.fbc_thresh_spin.setDecimals(3)
        self.fbc_thresh_spin.setRange(0.0, 0.5)
        self.fbc_thresh_spin.setSingleStep(0.005)
        self.fbc_thresh_spin.setValue(0.05)
        try:
            tip = (
                "FirstBlockCache threshold (absmean residual diff).\n"
                "Try 0.05 as a safe start.\n"
                "Higher = more skipping (faster), lower = safer quality."
            )
            self.fbc_thresh_slider.setToolTip(tip)
            self.fbc_thresh_spin.setToolTip(tip)
        except Exception:
            pass

        self.fbc_thresh_row = QWidget()
        _fbc_lay = QHBoxLayout(self.fbc_thresh_row)
        _fbc_lay.setContentsMargins(0, 0, 0, 0)
        _fbc_lay.addWidget(self.fbc_thresh_slider, 1)
        _fbc_lay.addWidget(self.fbc_thresh_spin, 0)
        self.fbc_thresh_row.setVisible(False)

        # Wire FirstBlockCache threshold controls
        try:
            self._set_fbc_threshold(float(self.fbc_thresh_spin.value()))
        except Exception:
            pass
        try:
            self.first_block_cache.toggled.connect(self._on_fbc_toggled)
        except Exception:
            pass
        try:
            self.fbc_thresh_slider.valueChanged.connect(self._on_fbc_slider_changed)
        except Exception:
            pass
        try:
            self.fbc_thresh_spin.valueChanged.connect(self._on_fbc_spin_changed)
        except Exception:
            pass

        # Pyramid Attention Broadcast (experimental)
        self.pab = QCheckBox("Enable Pyramid attn. broadcast")
        try:
            self.pab.setToolTip(
                "<b>Experimental:</b> reuse attention states across timesteps for faster inference.<br>"
                "May slightly change motion/texture. If you see artifacts, turn it OFF."
            )
        except Exception:
            pass

        self.tiling = QCheckBox("Enable VAE tiling")
        try:
            self.tiling.setToolTip(
                "Reduce VAE decode VRAM by processing tiles instead of full frames.<br>"
                "<b>Usually safe to keep ON.</b>"
            )
        except Exception:
            pass

        self.attn_slicing = QCheckBox("Attention slicing")
        self.attn_slicing.setToolTip(
            "Run attention in smaller chunks to reduce peak VRAM (slower).<br>"
            "<b>Turn OFF for maximum speed</b> if you have enough VRAM."
        )
        self.vae_slicing = QCheckBox("VAE slicing")
        self.vae_slicing.setToolTip(
            "Decode with VAE slicing to reduce peak VRAM (slower).<br>"
            "If your VRAM is stable, <b>turn OFF for faster decoding</b>."
        )
        self.output_name = QLineEdit("hunyuan15_output.mp4")
        self.output_name.setPlaceholderText("Output filename (saved to ./output/video/huyuan15/)")

        # Advanced section (collapsed by default; always starts closed after restart)
        self.adv_toggle = QToolButton()
        self.adv_toggle.setText("Advanced")
        self.adv_toggle.setCheckable(True)
        self.adv_toggle.setChecked(False)
        self.adv_toggle.setArrowType(Qt.RightArrow)
        self.adv_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.adv_toggle.toggled.connect(self._toggle_advanced)

        self.adv_body = QWidget()
        adv_body_lay = QVBoxLayout(self.adv_body)
        adv_body_lay.setContentsMargins(8, 6, 0, 0)

        adv_form = QFormLayout()
        adv_form.addRow(QLabel("Attention backend:"), self.attn)

        # Offload / performance toggles (top)
        adv_form.addRow(self.offload)
        adv_form.addRow(self.group_offload)

        # Diffusers hook toggles
        adv_form.addRow(self.first_block_cache)
        adv_form.addRow(QLabel("FBC threshold:"), self.fbc_thresh_row)
        adv_form.addRow(self.pab)

        # Memory savers
        adv_form.addRow(self.attn_slicing)
        adv_form.addRow(self.vae_slicing)
        adv_form.addRow(self.tiling)
        adv_body_lay.addLayout(adv_form)

        adv_btns = QHBoxLayout()
        adv_btns.addWidget(self.btn_install)
        adv_btns.addWidget(self.btn_download)
        adv_btns.addStretch(1)
        adv_body_lay.addLayout(adv_btns)

        self.adv_body.setVisible(False)

        self.adv_box = QGroupBox()
        self.adv_box.setTitle("")
        adv_outer = QVBoxLayout(self.adv_box)
        adv_outer.setContentsMargins(8, 6, 8, 6)
        hdr = QHBoxLayout()
        hdr.addWidget(self.adv_toggle)
        hdr.addStretch(1)
        adv_outer.addLayout(hdr)
        adv_outer.addWidget(self.adv_body)

        # Layout (scrollable – helps when embedded in Tools tab)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Pinned model selector (stays visible; does not scroll with Settings)
        self._pinned_model_row = QWidget()
        pinned_lay = QHBoxLayout(self._pinned_model_row)
        pinned_lay.setContentsMargins(8, 6, 8, 0)
        pinned_lay.setSpacing(8)
        pinned_lay.addWidget(QLabel("Model:"))
        pinned_lay.addWidget(self.model, 1)
        pinned_lay.addStretch(1)
        outer.addWidget(self._pinned_model_row, 0)


        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        try:
            self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        except Exception:
            pass
        outer.addWidget(self._scroll, 1)

        content = QWidget()
        self._scroll.setWidget(content)
        top = QVBoxLayout(content)
        top.setContentsMargins(0, 0, 0, 0)
        try:
            top.setSizeConstraint(QLayout.SetMinimumSize)
        except Exception:
            pass

        grp = QGroupBox("Settings")
        form = QFormLayout(grp)
        form.addRow(QLabel("Prompt:"), self.prompt)
        # Prompt helper row (Enhance + Clear)
        prompt_btn_row = QHBoxLayout()
        self.btn_prompt_enhance = QPushButton("Enhance prompt (Qwen)")
        try:
            self.btn_prompt_enhance.setToolTip(
                "Expand this prompt with the Qwen3-VL prompt helper (running in its own .venv). "
                "Great for adding detail and variety to Hunyuan 1.5 prompts."
            )
        except Exception:
            pass
        try:
            self.btn_prompt_enhance.clicked.connect(self._on_enhance_prompt_clicked)
        except Exception:
            pass


        # Presets dropdown (same preset names as Tools → Prompt enhancement)
        self.combo_prompt_preset = QComboBox()
        try:
            self.combo_prompt_preset.setToolTip("Prompt enhancer preset (same list as Tools → Prompt enhancement).")
        except Exception:
            pass
        try:
            self.combo_prompt_preset.currentIndexChanged.connect(self._on_prompt_preset_changed)
        except Exception:
            pass
        try:
            self._rebuild_prompt_preset_combo()
        except Exception:
            pass

        self.btn_prompt_clear = QPushButton("Clear")
        try:
            self.btn_prompt_clear.setToolTip("Clear the main prompt box so you can start over.")
        except Exception:
            pass
        try:
            self.btn_prompt_clear.clicked.connect(self._on_clear_prompt_clicked)
        except Exception:
            pass

        try:
            prompt_btn_row.addWidget(self.btn_prompt_enhance)
            prompt_btn_row.addWidget(self.combo_prompt_preset)
            prompt_btn_row.addWidget(self.btn_prompt_clear)
            prompt_btn_row.addStretch(1)
            prompt_btn_wrap = QWidget(self)
            prompt_btn_wrap.setLayout(prompt_btn_row)
            form.addRow("", prompt_btn_wrap)
        except Exception:
            pass

        # Negative prompt
        try:
            form.addRow(QLabel("Negative prompt:"), self.negative_prompt)
        except Exception:
            pass

        img_row = QWidget()
        img_lay = QHBoxLayout(img_row)
        img_lay.setContentsMargins(0, 0, 0, 0)
        img_lay.addWidget(self.start_image, 1)
        img_lay.addWidget(self.btn_pick_image)
        img_lay.addWidget(self.btn_clear_image)
        form.addRow(QLabel("Start image (I2V):"), img_row)

        v2v_row = QWidget()
        v2v_lay = QHBoxLayout(v2v_row)
        v2v_lay.setContentsMargins(0, 0, 0, 0)
        v2v_lay.addWidget(self.chk_video2video)
        v2v_lay.addWidget(self.btn_pick_v2v)
        v2v_lay.addWidget(self.btn_use_last_v2v)
        v2v_lay.addWidget(self.lbl_video2_info, 1)
        form.addRow(QLabel("Video→Video:"), v2v_row)
        self._lbl_aspect = QLabel("Aspect:")
        form.addRow(self._lbl_aspect, self.aspect)
        self._lbl_resolution = QLabel("Resolution:")
        form.addRow(self._lbl_resolution, self.resolution)

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Frames"))
        h1.addWidget(self.frames)
        h1.addSpacing(12)
        h1.addWidget(QLabel("Steps"))
        h1.addWidget(self.steps)
        h1.addSpacing(12)
        h1.addWidget(QLabel("FPS"))
        h1.addWidget(self.fps)
        h1.addSpacing(12)
        h1.addWidget(QLabel("Seed (-1=random)"))
        h1.addWidget(self.seed)
        form.addRow(h1)

        h_br = QHBoxLayout()
        h_br.addWidget(QLabel("Bitrate (kbps)"))
        h_br.addWidget(self.bitrate_kbps)
        h_br.addStretch(1)
        form.addRow(h_br)

        h_ext = QHBoxLayout()
        h_ext.addWidget(QLabel("Extend"))
        h_ext.addWidget(self.extend)
        h_ext.addWidget(QLabel("extra segments"))
        h_ext.addSpacing(12)
        h_ext.addWidget(self.extend_merge)
        h_ext.addStretch(1)
        form.addRow(h_ext)

        form.addRow(QLabel("Output filename:"), self.output_name)
        form.addRow(self.adv_box)

        btns1 = QHBoxLayout()
        btns1.addWidget(self.btn_generate)
        btns1.addWidget(self.btn_queue)
        btns1.addWidget(self.btn_batch)
        btns1.addWidget(self.btn_use_current)
        btns1.addStretch(1)

        btns2 = QHBoxLayout()
        btns2.addWidget(self.btn_stop)
        btns2.addWidget(self.btn_open_out)
        btns2.addWidget(self.btn_clear_log)
        btns2.addStretch(1)

        # Scrollable content: settings + logs
        top.addWidget(grp)
        top.addWidget(QLabel("Log:"))
        top.addWidget(self.log, 1)

        # Bottom action buttons (pinned to the bottom; do not scroll)
        bottom = QWidget()
        bottom_lay = QVBoxLayout(bottom)
        bottom_lay.setContentsMargins(0, 0, 0, 0)
        bottom_lay.setSpacing(6)
        bottom_lay.addLayout(btns1)
        bottom_lay.addLayout(btns2)
        outer.addWidget(bottom, 0)

        # Signals
        self.btn_install.clicked.connect(self.on_install)
        self.btn_download.clicked.connect(self.on_download)
        self.btn_generate.clicked.connect(self.on_generate)
        self.btn_queue.toggled.connect(self.on_queue_toggled)
        self.btn_batch.clicked.connect(self.on_batch_queue)
        self.btn_use_current.clicked.connect(self.on_use_current)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_open_out.clicked.connect(self.on_view_results)
        self.btn_clear_log.clicked.connect(self.on_clear_log)
        self.btn_pick_image.clicked.connect(self.on_pick_image)
        self.btn_clear_image.clicked.connect(self.on_clear_image)

        self.chk_video2video.toggled.connect(self.on_toggle_video2video)
        self.btn_pick_v2v.clicked.connect(self.on_pick_v2v_video)
        self.btn_use_last_v2v.clicked.connect(self.on_use_last_output_as_v2v)

        try:
            self.extend.valueChanged.connect(self._on_extend_value_changed)
        except Exception:
            pass

        # Autosave (debounced) — saves all settings (including Advanced) even if you don't run a job.
        # Prevent startup UI signals from overwriting settings on launch.
        self._settings_save_enabled = False

        self._autosave_timer = QTimer(self)
        try:
            self._autosave_timer.setSingleShot(True)
            self._autosave_timer.setInterval(450)
            self._autosave_timer.timeout.connect(self._persist)
        except Exception:
            pass
        try:
            self._wire_autosave()
        except Exception:
            pass

        # Load saved settings
        QTimer.singleShot(0, self._restore)

    # ---------- settings ----------
    def _restore(self):
        s = load_settings()
        if s.get("prompt"):
            self.prompt.setPlainText(s["prompt"])

        # Restore prompt enhancer preset (local to this tool)
        try:
            want_p = str(s.get("prompt_preset", "") or "Default")
            self._rebuild_prompt_preset_combo(want_p)
        except Exception:
            pass
        try:
            if s.get("negative_prompt") is not None:
                self.negative_prompt.setPlainText(str(s.get("negative_prompt") or ""))
        except Exception:
            pass
        self.frames.setValue(int(s.get("frames", self.frames.value())))
        self.steps.setValue(int(s.get("steps", self.steps.value())))
        self.fps.setValue(int(s.get("fps", self.fps.value())))
        try:
            self.bitrate_kbps.setValue(int(s.get("bitrate_kbps", self.bitrate_kbps.value())))
        except Exception:
            pass
        self.seed.setValue(int(s.get("seed", self.seed.value())))
        try:
            self.extend.setValue(int(s.get("extend", self.extend.value())))
        except Exception:
            self.extend.setValue(0)
        try:
            self.extend_merge.setChecked(bool(s.get("extend_merge", self.extend_merge.isChecked())))
        except Exception:
            pass
        try:
            self._on_extend_value_changed(int(self.extend.value()))
        except Exception:
            pass
        self.offload.setChecked(bool(s.get("offload", True)))
        self.attn_slicing.setChecked(bool(s.get("attn_slicing", False)))
        self.vae_slicing.setChecked(bool(s.get("vae_slicing", False)))
        self.tiling.setChecked(bool(s.get("tiling", True)))

        try:
            self.group_offload.setChecked(bool(s.get("group_offload", False)))
        except Exception:
            pass
        try:
            self.first_block_cache.setChecked(bool(s.get("first_block_cache", False)))
        except Exception:
            pass
        try:
            self._set_fbc_threshold(float(s.get("fbc_threshold", 0.05)))
        except Exception:
            pass
        try:
            self.fbc_thresh_row.setVisible(bool(self.first_block_cache.isChecked()))
        except Exception:
            pass
        try:
            self.pab.setChecked(bool(s.get("pab", False)))
        except Exception:
            pass


        model_key = s.get("model", None)
        if model_key:
            idx = self.model.findData(model_key)
            if idx >= 0:
                self.model.setCurrentIndex(idx)

        attn_key = s.get("attn", None)
        if attn_key:
            idx = self.attn.findData(attn_key)
            if idx >= 0:
                self.attn.setCurrentIndex(idx)

        res_h = s.get("res_h", None)
        if res_h:
            try:
                idx = self.resolution.findData(int(res_h))
                if idx >= 0:
                    self.resolution.setCurrentIndex(idx)
            except Exception:
                pass

        # Aspect (manual mode)
        try:
            asp = s.get("aspect", None)
            if asp is not None and getattr(self, 'aspect', None) is not None:
                idx = self.aspect.findData(str(asp))
                if idx >= 0:
                    self.aspect.setCurrentIndex(idx)
        except Exception:
            pass

        # Restore optional start image
        try:
            self.start_image.setText(str(s.get("start_image", "") or ""))
        except Exception:
            pass
        try:
            self.output_name.setText(str(s.get("output_name", self.output_name.text()) or self.output_name.text()))
        except Exception:
            pass

        # Restore Video→Video
        try:
            self.chk_video2video.setChecked(bool(s.get("v2v_enabled", False)))
        except Exception:
            pass
        try:
            src = (s.get("v2v_source", "") or "").strip()
            if src:
                p = Path(src)
                if p.exists():
                    self._video2video_path = p
                    try:
                        size_mb = p.stat().st_size / (1024 * 1024)
                        self.lbl_video2_info.setText(f"{p.name}  ({size_mb:.1f} MB)")
                    except Exception:
                        self.lbl_video2_info.setText(p.name)
                else:
                    self._video2video_path = None
        except Exception:
            pass
        try:
            self.btn_pick_v2v.setEnabled(self.chk_video2video.isChecked())
            self.btn_use_last_v2v.setEnabled(self.chk_video2video.isChecked())
        except Exception:
            pass


        # Restore Queue toggle (Use Queue) — forced OFF when Extend > 0
        try:
            self._set_queue_enabled(bool(s.get("use_queue", False)), persist=False, quiet=True)
        except Exception:
            pass

        # Sensible defaults
        if not self.prompt.toPlainText().strip():
            self.prompt.setPlainText(
                "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
            )

        if "offload" not in s:
            self.offload.setChecked(True)
        if "bitrate_kbps" not in s:
            try:
                self.bitrate_kbps.setValue(2000)
            except Exception:
                pass
        if "tiling" not in s:
            self.tiling.setChecked(True)

        # Advanced always starts collapsed (closed) after restart.
        try:
            self.adv_toggle.setChecked(False)
        except Exception:
            pass

        # Enable/disable merge checkbox based on Extend value
        try:
            self._on_extend_value_changed(int(self.extend.value()))
        except Exception:
            pass


        # Delay autosave so other startup init can't rewrite your JSON.
        try:
            QTimer.singleShot(1500, self._enable_settings_save)
        except Exception:
            try:
                self._settings_save_enabled = True
            except Exception:
                pass


    # ---------- autosave ----------
    def _enable_settings_save(self) -> None:
        """Enable autosave after startup restore has settled."""
        try:
            self._settings_save_enabled = True
        except Exception:
            pass

    def _schedule_persist(self, *args) -> None:
        """Debounced autosave for all UI settings."""
        if not getattr(self, "_settings_save_enabled", True):
            return
        try:
            t = getattr(self, "_autosave_timer", None)
            if t is not None:
                t.start()
                return
        except Exception:
            pass
        try:
            self._persist()
        except Exception:
            pass

    def _wire_autosave(self) -> None:
        """Connect common widgets to autosave."""
        # text widgets
        try:
            self.prompt.textChanged.connect(self._schedule_persist)
        except Exception:
            pass
        try:
            self.negative_prompt.textChanged.connect(self._schedule_persist)
        except Exception:
            pass
        try:
            self.start_image.textChanged.connect(self._schedule_persist)
        except Exception:
            pass
        try:
            self.output_name.textChanged.connect(self._schedule_persist)
        except Exception:
            pass

        # combos
        for cb_name in ("model", "attn", "resolution", "aspect"):
            try:
                cb = getattr(self, cb_name, None)
                if cb is not None:
                    cb.currentIndexChanged.connect(self._schedule_persist)
            except Exception:
                pass

        # spinboxes
        for sb_name in ("frames", "steps", "fps", "bitrate_kbps", "seed", "extend"):
            try:
                sb = getattr(self, sb_name, None)
                if sb is not None:
                    sb.valueChanged.connect(self._schedule_persist)
            except Exception:
                pass

        # checkboxes
        for chk_name in ("extend_merge", "offload", "group_offload", "first_block_cache", "pab", "attn_slicing", "vae_slicing", "tiling", "chk_video2video"):
            try:
                chk = getattr(self, chk_name, None)
                if chk is not None:
                    chk.toggled.connect(self._schedule_persist)
            except Exception:
                pass

        # queue toggle button (it's a QPushButton, but checkable)
        try:
            self.btn_queue.toggled.connect(self._schedule_persist)
        except Exception:
            pass

    def _persist(self):
        if not getattr(self, "_settings_save_enabled", True):
            return
        s = dict(
            prompt=self.prompt.toPlainText().strip(),
            negative_prompt=(self.negative_prompt.toPlainText().strip() if getattr(self, 'negative_prompt', None) is not None else ''),
            model=self.model.currentData(),
            attn=self.attn.currentData(),
            frames=int(self.frames.value()),
            steps=int(self.steps.value()),
            fps=int(self.fps.value()),
            bitrate_kbps=int(self.bitrate_kbps.value()),
            seed=int(self.seed.value()),
            extend=int(self.extend.value()),
            extend_merge=bool(self.extend_merge.isChecked()),
            offload=bool(self.offload.isChecked()),
            group_offload=bool(getattr(self, 'group_offload', None) and self.group_offload.isChecked()),
            first_block_cache=bool(getattr(self, 'first_block_cache', None) and self.first_block_cache.isChecked()),
            fbc_threshold=float(getattr(self, 'fbc_thresh_spin', None).value() if getattr(self, 'fbc_thresh_spin', None) is not None else 0.05),
            pab=bool(getattr(self, 'pab', None) and self.pab.isChecked()),
            attn_slicing=bool(self.attn_slicing.isChecked()),
            vae_slicing=bool(self.vae_slicing.isChecked()),
            tiling=bool(self.tiling.isChecked()),
            res_h=int(self.resolution.currentData()),
            aspect=str(getattr(self, 'aspect', None).currentData() if getattr(self,'aspect',None) else 'landscape'),
            start_image=self.start_image.text().strip(),
            output_name=self.output_name.text().strip(),
            v2v_enabled=bool(getattr(self, 'chk_video2video', None) and self.chk_video2video.isChecked()),
            v2v_source=str(getattr(self, '_video2video_path', '') or ''),
            use_queue=bool(getattr(self, 'btn_queue', None) and self.btn_queue.isChecked()),
            prompt_preset=str(getattr(self, 'combo_prompt_preset', None).currentText() if getattr(self, 'combo_prompt_preset', None) is not None else "Default"),
        )
        save_settings(s)

    def _toggle_advanced(self, checked: bool) -> None:
        try:
            self.adv_body.setVisible(bool(checked))
            self.adv_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        except Exception:
            pass

    # ---------- FirstBlockCache helpers ----------
    def _set_fbc_threshold(self, value: float) -> None:
        """Sync slider + spinbox for the FirstBlockCache threshold."""
        try:
            v = float(value)
        except Exception:
            v = 0.05
        if v < 0.0:
            v = 0.0
        if v > 0.5:
            v = 0.5
        # slider is 0..500, representing 0.000..0.500
        sv = int(round(v * 1000.0))
        if sv < 0:
            sv = 0
        if sv > 500:
            sv = 500
        try:
            self.fbc_thresh_slider.blockSignals(True)
            self.fbc_thresh_slider.setValue(sv)
        except Exception:
            pass
        try:
            self.fbc_thresh_slider.blockSignals(False)
        except Exception:
            pass
        try:
            self.fbc_thresh_spin.blockSignals(True)
            self.fbc_thresh_spin.setValue(v)
        except Exception:
            pass
        try:
            self.fbc_thresh_spin.blockSignals(False)
        except Exception:
            pass

    def _on_fbc_toggled(self, checked: bool) -> None:
        try:
            self.fbc_thresh_row.setVisible(bool(checked))
        except Exception:
            pass
        try:
            self._schedule_persist()
        except Exception:
            pass

    def _on_fbc_slider_changed(self, sv: int) -> None:
        try:
            sv = int(sv)
        except Exception:
            sv = 50
        if sv < 0:
            sv = 0
        if sv > 500:
            sv = 500
        v = float(sv) / 1000.0
        try:
            self.fbc_thresh_spin.blockSignals(True)
            self.fbc_thresh_spin.setValue(v)
        except Exception:
            pass
        try:
            self.fbc_thresh_spin.blockSignals(False)
        except Exception:
            pass
        try:
            self._schedule_persist()
        except Exception:
            pass

    def _on_fbc_spin_changed(self, v: float) -> None:
        try:
            vv = float(v)
        except Exception:
            vv = 0.05
        sv = int(round(vv * 1000.0))
        if sv < 0:
            sv = 0
        if sv > 500:
            sv = 500
        try:
            self.fbc_thresh_slider.blockSignals(True)
            self.fbc_thresh_slider.setValue(sv)
        except Exception:
            pass
        try:
            self.fbc_thresh_slider.blockSignals(False)
        except Exception:
            pass
        try:
            self._schedule_persist()
        except Exception:
            pass

    def _on_extend_value_changed(self, v: int) -> None:
        """UI helper: enable auto-merge only when Extend is > 0."""
        try:
            v = int(v)
        except Exception:
            v = 0
        try:
            self.extend_merge.setEnabled(v > 0)
        except Exception:
            pass

        # Extend is Direct-run only; disable Queue while Extend > 0.
        try:
            ext_now = int(v)
        except Exception:
            ext_now = 0
        try:
            if ext_now > 0:
                self._set_queue_enabled(False, persist=True, quiet=True)
                try:
                    self.btn_queue.setEnabled(False)
                except Exception:
                    pass
            else:
                try:
                    self.btn_queue.setEnabled(True)
                except Exception:
                    pass
                # Refresh text/tooltips (state may be unchanged)
                try:
                    self._set_queue_enabled(bool(self.btn_queue.isChecked()), persist=False, quiet=True)
                except Exception:
                    pass
        except Exception:
            pass

        # Debounced autosave
        try:
            self._schedule_persist()
        except Exception:
            pass

    # ---------- Queue toggle helpers ----------

    def _queue_enabled(self) -> bool:
        try:
            return bool(self.btn_queue.isChecked())
        except Exception:
            return False

    def _set_queue_enabled(self, enabled: bool, persist: bool = True, quiet: bool = False) -> None:
        """Apply Queue toggle state, update UI, and optionally persist.

        Rule:
        - If Extend > 0, Queue is forced OFF and the toggle is disabled.
        """
        # Enforce Extend rule
        try:
            ext = int(self.extend.value())
        except Exception:
            ext = 0
        if ext > 0:
            enabled = False

        # Apply checked state without recursion
        try:
            self.btn_queue.blockSignals(True)
        except Exception:
            pass
        try:
            self.btn_queue.setChecked(bool(enabled))
        except Exception:
            pass
        try:
            self.btn_queue.blockSignals(False)
        except Exception:
            pass

        # Update text (simple visual hint)
        try:
            self.btn_queue.setText("Use Queue ✓" if enabled else "Use Queue")
        except Exception:
            pass


        # Update main action label based on Queue
        try:
            self.btn_generate.setText("Generate Video" if enabled else "Direct run")
        except Exception:
            pass
        # Enable/disable toggle based on Extend
        try:
            self.btn_queue.setEnabled(ext <= 0)
        except Exception:
            pass

        # Tooltips (include your requested warnings)
        try:
            q_tip = (
                "Toggle Queue mode.\n"
                "ON: clicking \"Generate Video\" will queue the job to the Queue tab.\n\n"
                "Tip: Extend (extra segments) requires Direct run, so Queue is disabled while Extend > 0."
            )
            self.btn_queue.setToolTip(q_tip)
        except Exception:
            pass

        try:
            run_tip = (
                "Generate Video will queue the job to the Queue tab.\n"
                "Tip: avoid running other heavy VRAM apps at the same time.\n"
                "Extend requires Direct run (Queue OFF)."
            ) if enabled else (
                "Direct run starts immediately (no Queue).\n"
                "Tip: avoid running other heavy VRAM apps at the same time."
            )
            self.btn_generate.setToolTip(run_tip)
        except Exception:
            pass

        if persist:
            try:
                self._persist()
            except Exception:
                pass
        if (not quiet) and ext > 0:
            try:
                self._append("[ui] Extend > 0: Queue has been disabled (Extend is Direct-run only).")
            except Exception:
                pass

    def on_queue_toggled(self, checked: bool) -> None:
        """Slot: Use Queue toggle changed."""
        self._set_queue_enabled(bool(checked), persist=True, quiet=False)
    # ---------- output sidecar (.json) ----------
    def _build_sidecar_meta(self, prompt: str, out_path: Path, seed: int, mode: str = "direct", extra: dict | None = None) -> dict:
        meta: dict = {}
        try:
            meta["engine"] = "hunyuan15"
            meta["mode"] = str(mode)
            meta["created_at"] = datetime.now().isoformat(timespec="seconds")
            meta["output"] = str(out_path)
            meta["prompt"] = str(prompt or "")
        except Exception:
            pass

        try:
            meta["negative_prompt"] = (self.negative_prompt.toPlainText() or "").strip()
        except Exception:
            meta["negative_prompt"] = ""

        try:
            meta["model"] = str(self.model.currentData() or "")
            meta["model_label"] = str(self.model.currentText() or "")
        except Exception:
            pass

        try:
            meta["attn"] = str(self.attn.currentData() or "")
        except Exception:
            pass

        try:
            meta["frames"] = int(self.frames.value())
            meta["steps"] = int(self.steps.value())
            meta["fps"] = int(self.fps.value())
            meta["bitrate_kbps"] = int(self.bitrate_kbps.value())
        except Exception:
            pass

        try:
            meta["seed"] = int(seed)
        except Exception:
            meta["seed"] = -1

        try:
            meta["start_image"] = (self._start_image_path() or "")
        except Exception:
            meta["start_image"] = ""

        try:
            meta["v2v_source"] = str(getattr(self, "_video2video_path", "") or "")
        except Exception:
            meta["v2v_source"] = ""

        try:
            meta["resolution_bucket"] = int(self.resolution.currentData() or 0)
            meta["aspect"] = str(getattr(self, 'aspect', None).currentData() if getattr(self, 'aspect', None) else 'landscape')
        except Exception:
            pass

        try:
            meta["flags"] = {
                "offload": bool(self.offload.isChecked()),
                "group_offload": bool(getattr(self, 'group_offload', None) and self.group_offload.isChecked()),
                "first_block_cache": bool(getattr(self, 'first_block_cache', None) and self.first_block_cache.isChecked()),
                "pab": bool(getattr(self, 'pab', None) and self.pab.isChecked()),
                "attn_slicing": bool(self.attn_slicing.isChecked()),
                "vae_slicing": bool(self.vae_slicing.isChecked()),
                "tiling": bool(self.tiling.isChecked()),
            }
        except Exception:
            pass

        if isinstance(extra, dict) and extra:
            try:
                meta.update(extra)
            except Exception:
                pass
        return meta

    def _write_sidecar_json(self, out_path: Path, meta: dict) -> bool:
        try:
            p = Path(str(out_path))
        except Exception:
            return False
        if not p.is_file():
            return False

        try:
            jpath = p.with_suffix(".json")
            payload = dict(meta or {})
            payload.setdefault("output", str(p))
            try:
                payload.setdefault("file_size_bytes", int(p.stat().st_size))
            except Exception:
                pass
            tmp = jpath.with_suffix(jpath.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(jpath)
            try:
                self._append(f"[sidecar] wrote {jpath.name}")
            except Exception:
                pass
            return True
        except Exception as e:
            try:
                self._append(f"[sidecar] ERROR: {e}")
            except Exception:
                pass
            return False




    # ---------- process helpers ----------

    # ---------------- Prompt enhancer (Qwen) ----------------

    def _on_clear_prompt_clicked(self):
        try:
            self.prompt.setPlainText("")
            self.prompt.setFocus()
        except Exception:
            pass

    def _on_enhance_prompt_clicked(self):
        """Enhance the prompt using the shared Qwen helper without freezing the UI."""
        # Guard against re-entry
        try:
            if getattr(self, "_qwen_proc", None) is not None and self._qwen_proc.state() != QProcess.NotRunning:
                try:
                    QMessageBox.information(self, "Prompt enhancer", "Qwen is already enhancing a prompt.")
                except Exception:
                    pass
                return
        except Exception:
            pass

        # Collect current prompt
        try:
            base_prompt = (self.prompt.toPlainText() or "").strip()
        except Exception:
            base_prompt = ""
        if not base_prompt:
            try:
                QMessageBox.warning(self, "Prompt enhancer", "Please enter a base prompt first.")
            except Exception:
                pass
            return

        app_root = root_dir()

        # Locate dedicated .venv Python (Qwen environment)
        py_path = None
        try:
            venv = app_root / ".venv"
            win_py = venv / "Scripts" / "python.exe"
            nix_py = venv / "bin" / "python"
            if os.name == "nt" and win_py.exists():
                py_path = win_py
            elif nix_py.exists():
                py_path = nix_py
        except Exception:
            py_path = None

        if py_path is None:
            try:
                QMessageBox.critical(
                    self,
                    "Prompt enhancer",
                    "Could not find a dedicated .venv Python.\nExpected .venv/Scripts/python.exe or .venv/bin/python next to the app folder."
                )
            except Exception:
                pass
            return

        # Use shared CLI helper
        cli_path = app_root / "helpers" / "prompt_enhancer_cli.py"
        if not cli_path.exists():
            try:
                QMessageBox.critical(
                    self,
                    "Prompt enhancer",
                    "helpers/prompt_enhancer_cli.py is missing.\nThis button reuses the Txt2Img Qwen helper."
                )
            except Exception:
                pass
            return

        # Temporarily apply the selected preset + force VIDEO target for this enhancement run
        try:
            st = self._prompttool_read_settings()
            overrides = self._prompttool_overrides_for_preset(st)
            self._prompttool_force_settings(overrides)
            try:
                self._append(f"[prompt] Enhancer preset: {overrides.get('preset','Default')}")
            except Exception:
                pass
        except Exception:
            pass

        # Build command
        cmd = [str(py_path), str(cli_path), "--seed", base_prompt]

        # Ensure the QProcess exists
        self._ensure_qwen_process()

        # Reset buffers + start
        try:
            self._qwen_stdout_buf = bytearray()
            self._qwen_stderr_buf = bytearray()
        except Exception:
            self._qwen_stdout_buf = bytearray()
            self._qwen_stderr_buf = bytearray()

        # UI busy state
        self._set_qwen_busy(True)

        # Environment
        try:
            env = QProcessEnvironment.systemEnvironment()
            env.insert("PYTHONUTF8", "1")
            env.insert("PYTHONIOENCODING", "utf-8")
            env.insert("PYTHONUNBUFFERED", "1")
            self._qwen_proc.setProcessEnvironment(env)
        except Exception:
            pass

        try:
            self._qwen_proc.setWorkingDirectory(str(app_root))
        except Exception:
            pass

        try:
            self._append("Enhancing prompt with Qwen3-VL…, You can change detailed settings in Tools/prompt enhancement for more variation")
        except Exception:
            pass

        try:
            self._qwen_proc.start(cmd[0], cmd[1:])
        except Exception as e:
            self._set_qwen_busy(False)
            try:
                QMessageBox.critical(self, "Prompt enhancer", f"Failed to start Qwen helper: {e}")
            except Exception:
                pass

    def _ensure_qwen_process(self):
        """Lazy-init the Qwen QProcess and its signal wiring."""
        if getattr(self, "_qwen_proc", None) is not None:
            return
        self._qwen_proc = QProcess(self)
        # Keep stdout/stderr separate so JSON parsing of stdout stays clean.
        try:
            self._qwen_proc.setProcessChannelMode(QProcess.SeparateChannels)
        except Exception:
            pass
        self._qwen_proc.readyReadStandardOutput.connect(self._read_qwen_stdout)
        self._qwen_proc.readyReadStandardError.connect(self._read_qwen_stderr)
        self._qwen_proc.finished.connect(self._on_qwen_finished)
        self._qwen_stdout_buf = bytearray()
        self._qwen_stderr_buf = bytearray()

    def _set_qwen_busy(self, busy: bool):
        """Disable the enhance button while Qwen runs (prevents double-click issues)."""
        btn = getattr(self, "btn_prompt_enhance", None)
        if btn is None:
            return
        try:
            if not hasattr(self, "_qwen_btn_text"):
                self._qwen_btn_text = btn.text()
        except Exception:
            self._qwen_btn_text = "Enhance prompt (Qwen)"

        try:
            btn.setEnabled(not busy)
        except Exception:
            pass
        try:
            btn.setText("Enhancing…" if busy else str(getattr(self, "_qwen_btn_text", "Enhance prompt (Qwen)")))
        except Exception:
            pass

    def _read_qwen_stdout(self):
        try:
            self._qwen_stdout_buf += bytes(self._qwen_proc.readAllStandardOutput())
        except Exception:
            pass

    def _read_qwen_stderr(self):
        try:
            self._qwen_stderr_buf += bytes(self._qwen_proc.readAllStandardError())
        except Exception:
            pass


    # ---------------- Prompt enhancer (Qwen) shared settings (PromptTool) ----------------

    def _prompttool_settings_path(self) -> Path:
        return root_dir() / "presets" / "setsave" / "prompt.json"

    def _prompttool_read_settings(self) -> dict:
        p = self._prompttool_settings_path()
        if not p.exists():
            return {}
        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
            obj = json.loads(raw) if raw.strip() else {}
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _prompttool_force_settings(self, overrides: dict) -> None:
        """Temporarily apply overrides to presets/setsave/prompt.json for this Qwen run.

        We restore the full previous file contents when Qwen finishes, so this does NOT
        permanently change the Tools → Prompt enhancement settings.
        """
        p = self._prompttool_settings_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        had_file = bool(p.exists())
        prev_data: dict = {}
        if had_file:
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
                obj = json.loads(raw) if raw.strip() else {}
                if isinstance(obj, dict):
                    prev_data = obj
            except Exception:
                prev_data = {}

        data = dict(prev_data)
        try:
            if isinstance(overrides, dict):
                data.update(overrides)
        except Exception:
            pass

        try:
            p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            self._prompttool_restore_info = None
            return

        self._prompttool_restore_info = {
            "path": str(p),
            "had_file": had_file,
            "prev_data": prev_data,
        }

    def _prompttool_restore_settings(self) -> None:
        info = getattr(self, "_prompttool_restore_info", None)
        self._prompttool_restore_info = None
        if not info:
            return
        try:
            p = Path(info.get("path", ""))
        except Exception:
            return
        had_file = bool(info.get("had_file", False))
        prev_data = info.get("prev_data", {})
        if not had_file:
            # File didn't exist before; delete it (best-effort).
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass
            return
        # Restore previous contents
        try:
            if not isinstance(prev_data, dict):
                prev_data = {}
            p.write_text(json.dumps(prev_data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # ---------------- Prompt enhancer preset dropdown (Hunyuan-local) ----------------

    def _current_prompt_preset(self) -> str:
        cb = getattr(self, "combo_prompt_preset", None)
        if cb is None:
            return "Default"
        try:
            v = (cb.currentText() or "").strip()
        except Exception:
            v = ""
        return v or "Default"

    def _rebuild_prompt_preset_combo(self, want: str | None = None) -> None:
        cb = getattr(self, "combo_prompt_preset", None)
        if cb is None:
            return
        names = list((_PROMPT_PRESET_DEFS or {"Default": {}}).keys())
        if "Default" not in names:
            names.insert(0, "Default")
        names = [n for n in names if n]
        rest = sorted([n for n in names if n != "Default"], key=lambda s: str(s).lower())
        ordered = ["Default"] + rest

        try:
            cb.blockSignals(True)
        except Exception:
            pass
        try:
            cb.clear()
            for n in ordered:
                cb.addItem(n)
        except Exception:
            pass

        if want is None:
            want = self._current_prompt_preset()
        try:
            idx = cb.findText(str(want))
            if idx < 0:
                idx = cb.findText("Default")
            if idx >= 0:
                cb.setCurrentIndex(idx)
        except Exception:
            pass

        try:
            cb.blockSignals(False)
        except Exception:
            pass

    def _on_prompt_preset_changed(self, *args) -> None:
        """Local preset selection changed (autosave only)."""
        try:
            self._schedule_persist()
        except Exception:
            pass

    def _prompttool_overrides_for_preset(self, st: dict) -> dict:
        """Compute PromptTool overrides so the CLI enhancer actually uses the selected preset."""
        preset = self._current_prompt_preset()
        pdef = (_PROMPT_PRESET_DEFS or {}).get(preset) or (_PROMPT_PRESET_DEFS or {}).get("Default") or {}
        defaults = {}
        try:
            defaults = pdef.get("defaults", {}) or {}
        except Exception:
            defaults = {}

        # Start from current PromptTool settings (so we can merge cleanly)
        cur_style = ""
        cur_negs = ""
        try:
            cur_style = str((st or {}).get("style", "") or "").strip()
        except Exception:
            cur_style = ""
        try:
            cur_negs = str((st or {}).get("negatives", "") or "").strip()
        except Exception:
            cur_negs = ""

        # Preset style / negatives (prompt.py applies these at generation time; we inject to be robust)
        p_style = ""
        p_negs = ""
        try:
            p_style = str(pdef.get("style", "") or "").strip()
        except Exception:
            p_style = ""
        try:
            # Prefer explicit defaults negatives if present; otherwise use preset's additive negatives
            if isinstance(defaults, dict) and isinstance(defaults.get("negatives"), str):
                p_negs = str(defaults.get("negatives") or "").strip()
            else:
                p_negs = str(pdef.get("negatives", "") or "").strip()
        except Exception:
            p_negs = ""

        new_style = cur_style
        if p_style:
            new_style = (f"{cur_style}, {p_style}" if cur_style else p_style)

        new_negs = cur_negs
        if p_negs:
            new_negs = _merge_neg_csv(cur_negs, p_negs) if cur_negs else p_negs

        overrides: dict = {
            "target": "video",  # Hunyuan 1.5 wants video phrasing
            "preset": preset,
            "last_used_preset": preset,
            "style": new_style,
            "negatives": new_negs,
        }

        # Apply preset defaults (length/temperature). We keep target forced to video for this run.
        try:
            if isinstance(defaults, dict):
                length = defaults.get("length")
                if isinstance(length, str) and length in (_PROMPT_LENGTH_PRESETS or {}):
                    overrides["length_choice"] = length
                    try:
                        _, tokens = (_PROMPT_LENGTH_PRESETS or {}).get(length)
                        overrides["max_new_tokens"] = int(tokens)
                    except Exception:
                        pass
                temp = defaults.get("temperature")
                if isinstance(temp, (int, float)):
                    overrides["temperature"] = float(temp)
        except Exception:
            pass

        return overrides

    def _on_qwen_finished(self, exit_code: int, exit_status):
        self._set_qwen_busy(False)

        # Restore PromptTool settings (we temporarily forced target/preset/style/negatives for this run)
        try:
            self._prompttool_restore_settings()
        except Exception:
            pass

        try:
            out_txt = bytes(getattr(self, "_qwen_stdout_buf", b"")).decode("utf-8", "ignore").strip()
        except Exception:
            out_txt = ""
        try:
            err_txt = bytes(getattr(self, "_qwen_stderr_buf", b"")).decode("utf-8", "ignore").strip()
        except Exception:
            err_txt = ""

        if exit_code != 0:
            msg = err_txt or out_txt or f"Exit code {exit_code}"
            if len(msg) > 2000:
                msg = msg[:2000] + "..."
            try:
                QMessageBox.critical(self, "Prompt enhancer", "Qwen prompt helper failed:\n\n" + msg)
            except Exception:
                pass
            return

        # Parse JSON payload from stdout
        data = None
        try:
            data = json.loads(out_txt)
        except Exception:
            data = None

        if not isinstance(data, dict) or not data.get("ok"):
            msg = out_txt or err_txt or "Unexpected response from helper."
            if len(msg) > 2000:
                msg = msg[:2000] + "..."
            try:
                QMessageBox.critical(
                    self,
                    "Prompt enhancer",
                    "Qwen prompt helper returned an unexpected payload:\n\n" + msg
                )
            except Exception:
                pass
            return

        new_prompt = data.get("prompt") or ""
        if new_prompt:
            try:
                self.prompt.setPlainText(new_prompt)
            except Exception:
                pass
            try:
                self._append("Prompt enhanced with Qwen3-VL")
            except Exception:
                pass

    def _render_log(self) -> None:
        try:
            self.log.setPlainText("\n".join(self._log_lines))
            try:
                cur = self.log.textCursor()
                cur.movePosition(cur.Start)
                self.log.setTextCursor(cur)
                self.log.ensureCursorVisible()
            except Exception:
                pass
        except Exception:
            pass

    def _push_log_lines(self, lines: list[str], render: bool = True) -> None:
        if not lines:
            return
        for ln in lines:
            # Newest first
            self._log_lines.insert(0, (ln or "").rstrip("\n"))
        if len(self._log_lines) > MAX_LOG_LINES:
            self._log_lines = self._log_lines[:MAX_LOG_LINES]
        if render:
            self._render_log()

    def _flush_pending_logs(self) -> None:
        if not getattr(self, "_log_pending", None):
            return
        try:
            pending = list(self._log_pending)
            self._log_pending.clear()
        except Exception:
            return
        if pending:
            self._push_log_lines(pending, render=True)

    def _append(self, text: str):
        # Split to keep ordering consistent even when we get multi-line chunks.
        lines = [ln.rstrip("\n") for ln in (text.splitlines() or [text])]
        if getattr(self, "_log_direct_active", False):
            # Direct runs can spam output; batch UI updates every 3s while running.
            try:
                self._log_pending.extend(lines)
            except Exception:
                pass
            return
        self._push_log_lines(lines, render=True)

    def _set_busy(self, busy: bool):
        for b in (self.btn_install, self.btn_download, self.btn_generate, self.btn_queue, self.btn_batch):
            try:
                b.setEnabled(not busy)
            except Exception:
                pass
        try:
            self.btn_stop.setEnabled(bool(busy))
        except Exception:
            pass

    def _run(self, program: str, args: list[str], cwd: Path):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Busy", "A process is running, please wait")
            return

        self._set_busy(True)
        self._append(f"\n>>> {program} {' '.join(args)}")
        self.proc = QProcess(self)
        self.proc.setProgram(program)
        self.proc.setArguments(args)
        self.proc.setWorkingDirectory(str(cwd))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)

        self.proc.readyReadStandardOutput.connect(self._on_read)
        self.proc.finished.connect(self._on_finished)
        self.proc.start()

    # ---------- extend-chain helpers ----------
    def _map_model_to_i2v(self, model_key: str | None) -> str | None:
        """Map a model key to an i2v key for chained segments.

        - If already i2v, return unchanged.
        - For *_t2v variants, swap to *_i2v when possible.
        """
        if not model_key:
            return None
        k = str(model_key)
        if "_i2v" in k:
            return k
        # Common patterns in this tool
        if k.endswith("_t2v_distilled"):
            return k.replace("_t2v_distilled", "_i2v_distilled")
        if k.endswith("_t2v"):
            return k.replace("_t2v", "_i2v")
        # Fallback: swap first t2v token
        if "t2v" in k and "i2v" not in k:
            return k.replace("t2v", "i2v", 1)
        return None

    def _next_extend_frame_path(self) -> Path:
        """Return a unique file path for the next last-frame snapshot."""
        try:
            self._extend_frame_index += 1
        except Exception:
            self._extend_frame_index = 1
        idx = int(getattr(self, "_extend_frame_index", 1))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Keep name short and safe
        prefix = "h15"
        try:
            raw = (self._extend_base_prompt or "").strip()
            if raw:
                w = re.split(r"\s+", raw)[:3]
                safe = []
                for token in w:
                    token = re.sub(r"[^A-Za-z0-9]+", "", token)
                    if token:
                        safe.append(token)
                if safe:
                    prefix = "_".join(safe)[:24]
        except Exception:
            pass
        return _extend_frames_dir() / f"{prefix}_{ts}_{idx:02d}.png"

    def _extract_last_frame(self, video: Path, dest: Path) -> bool:
        """Extract last frame of a video to dest (used for extend-chain)."""
        ff = _ffmpeg_exe()
        if not ff:
            return False
        try:
            cmd = [
                ff,
                "-y",
                "-hide_banner",
                "-sseof",
                "-0.05",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(dest),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return dest.exists()
        except Exception:
            return False

    def _auto_merge_extend_segments(self) -> Path | None:
        """Auto-merge all generated extend segments into a single MP4.

        Hunyuan 1.5 segments can briefly "pause" at the start of each appended chunk.
        To hide the seam we:
        - Trim the first EXTEND_JOIN_DROP_FRAMES frames of every appended segment (not the first).
        - Optionally apply a very short crossfade (EXTEND_JOIN_BLEND_FRAMES) if ffprobe is available.
        - Fall back to the original concat behavior if anything goes wrong.
        """
        segments: list[Path] = []
        try:
            for p in getattr(self, "_extend_segments", []) or []:
                if isinstance(p, Path) and p.is_file():
                    segments.append(p)
        except Exception:
            segments = []

        if len(segments) < 2:
            return None

        ff = _ffmpeg_exe()
        if not ff:
            self._append("[extend] ffmpeg not found; cannot auto-merge segments.")
            return None

        # Output path
        base_out = getattr(self, "_extend_base_out", None)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            if isinstance(base_out, Path):
                merged = base_out.with_name(f"{base_out.stem}_merged_{ts}{base_out.suffix or '.mp4'}")
            else:
                out_dir = root_dir() / "output" / "video" / "hunyuan15"
                out_dir.mkdir(parents=True, exist_ok=True)
                merged = out_dir / f"h15_extend_merged_{ts}.mp4"
        except Exception:
            merged = root_dir() / "output" / "video" / "hunyuan15" / f"h15_extend_merged_{ts}.mp4"

        # Read settings
        try:
            drop_frames = int(globals().get("EXTEND_JOIN_DROP_FRAMES", 0) or 0)
        except Exception:
            drop_frames = 0
        try:
            blend_frames = int(globals().get("EXTEND_JOIN_BLEND_FRAMES", 0) or 0)
        except Exception:
            blend_frames = 0
        try:
            fps = int(getattr(self, "fps", None).value()) if getattr(self, "fps", None) is not None else 15
        except Exception:
            fps = 15

        # Normalize / trim segments into temp files (safe, keeps merge stable)
        work_dir = _temp_dir() / f"hunyuan15_extend_merge_{ts}"
        try:
            work_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        norm_segments: list[Path] = []
        norm_ok = True

        for i, seg in enumerate(segments):
            out_norm = work_dir / f"seg_{i:03d}.mp4"
            df = drop_frames if (i > 0) else 0
            # Only trim if we actually have something to trim.
            vf = "setpts=PTS-STARTPTS"
            if df > 0:
                vf = f"trim=start_frame={df},setpts=PTS-STARTPTS"

            try:
                cmd = [
                    ff,
                    "-y",
                    "-hide_banner",
                    "-i",
                    str(seg),
                    "-an",
                    "-vf",
                    vf,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    str(out_norm),
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                if not out_norm.is_file():
                    norm_ok = False
                    break
                norm_segments.append(out_norm)
            except Exception:
                norm_ok = False
                break

        if not norm_ok or len(norm_segments) != len(segments):
            # Fallback to original concat without trimming.
            self._append("[extend] seam-fix preprocess failed; falling back to standard merge.")
            norm_segments = list(segments)

        # Optional: tiny crossfade blend across joins (if enabled and ffprobe exists)
        if blend_frames > 0:
            ffprobe = _ffprobe_exe()
            if not ffprobe:
                self._append("[extend] blend requested but ffprobe not found; using concat merge instead.")
            else:
                try:
                    blend_dur = max(0.0, float(blend_frames) / float(max(1, fps)))
                except Exception:
                    blend_dur = 0.0

                if blend_dur > 0.0 and len(norm_segments) >= 2:
                    # Get per-segment durations
                    durs: list[float] = []
                    for p in norm_segments:
                        try:
                            cmdp = [
                                ffprobe,
                                "-v",
                                "error",
                                "-show_entries",
                                "format=duration",
                                "-of",
                                "default=nk=1:nw=1",
                                str(p),
                            ]
                            r = subprocess.run(cmdp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                            txt = (r.stdout or b"").decode("utf-8", "ignore").strip()
                            d = float(txt) if txt else 0.0
                        except Exception:
                            d = 0.0
                        durs.append(max(0.0, d))

                    # Build xfade chain
                    inputs: list[str] = []
                    for p in norm_segments:
                        inputs += ["-i", str(p)]

                    fc_parts: list[str] = []
                    # Start with first two
                    cum = durs[0]
                    off = max(0.0, cum - blend_dur)
                    fc_parts.append(f"[0:v][1:v]xfade=transition=fade:duration={blend_dur:.6f}:offset={off:.6f}[v1]")
                    # Chain additional
                    for idx in range(2, len(norm_segments)):
                        cum += durs[idx - 1]
                        off = max(0.0, cum - blend_dur * idx)
                        fc_parts.append(f"[v{idx-1}][{idx}:v]xfade=transition=fade:duration={blend_dur:.6f}:offset={off:.6f}[v{idx}]")

                    last_tag = f"v{len(norm_segments)-1}"
                    filter_complex = ";".join(fc_parts)

                    self._append(f"[extend] merging {len(norm_segments)} segments with seam blend → {merged.name}")
                    cmd = [ff, "-y", "-hide_banner"] + inputs + [
                        "-filter_complex",
                        filter_complex,
                        "-map",
                        f"[{last_tag}]",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        str(merged),
                    ]
                    try:
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                        if merged.is_file():
                            self._append("[extend] auto-merge completed (blended).")
                            return merged
                    except Exception:
                        self._append("[extend] blended merge failed; falling back to concat merge.")

        # Concat merge (fast copy when possible; fallback re-encode)
        concat_file = work_dir / "concat.txt"
        try:
            lines = [f"file '{p.as_posix()}'\n" for p in norm_segments]
            with open(concat_file, "w", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as e:
            self._append(f"[extend] failed to prepare concat list: {e}")
            return None

        self._append(f"[extend] merging {len(norm_segments)} segments → {merged.name}")
        try:
            cmd = [
                ff,
                "-y",
                "-hide_banner",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(merged),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if merged.is_file():
                self._append("[extend] auto-merge completed.")
                return merged
        except Exception:
            pass

        try:
            cmd = [
                ff,
                "-y",
                "-hide_banner",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(merged),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if merged.is_file():
                self._append("[extend] auto-merge completed (re-encoded).")
                return merged
        except Exception as e:
            self._append(f"[extend] merge failed: {e}")
            return None

        return None


    def _start_extend_segment(self, last_segment: Path) -> bool:
        """Start the next chained segment using the last frame of last_segment."""
        if not getattr(self, "_extend_active", False):
            return False
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self._append("[extend] process still running; cannot continue chain.")
            self._extend_active = False
            self._extend_remaining = 0
            self._extend_pending_output = None
            return False

        frame_path = self._next_extend_frame_path()
        if not self._extract_last_frame(last_segment, frame_path):
            self._append(f"[extend] failed to extract last frame from {last_segment.name}; stopping chain.")
            self._extend_active = False
            self._extend_remaining = 0
            self._extend_pending_output = None
            return False

        base_out = getattr(self, "_extend_base_out", None)
        if not isinstance(base_out, Path):
            base_out = last_segment
        idx = 0
        try:
            idx = len(getattr(self, "_extend_segments", []) or [])
        except Exception:
            idx = 0
        suffix = base_out.suffix or ".mp4"
        next_out = base_out.with_name(f"{base_out.stem}_ext{idx:02d}{suffix}")

        # For extend segments we always need an i2v model.
        base_model = getattr(self, "_extend_base_model", None)
        next_model = self._map_model_to_i2v(base_model or str(self.model.currentData() or ""))
        if not next_model:
            # Last-resort fallback based on resolution.
            try:
                h = int(self.resolution.currentData())
            except Exception:
                h = 480
            next_model = "720p_i2v" if h >= 720 else "480p_i2v"

        self._append(
            f"[extend] starting chained segment {idx} using {frame_path.name} → {next_out.name}"
        )

        # Build args without touching the user's selected model UI.
        args = self._build_generate_args(
            prompt=str(getattr(self, "_extend_base_prompt", "") or ""),
            out_path=next_out,
            seed_override=int(getattr(self, "_extend_base_seed", int(self.seed.value()))),
            model_key_override=next_model,
            image_override=str(frame_path),
        )

        self._extend_pending_output = next_out
        self._last_run_out_path = next_out
        self._current_task = "generate"

        # Prepare sidecar metadata for this segment
        try:
            seg_idx = int(getattr(self, "_extend_frame_index", 0) or 0)
        except Exception:
            seg_idx = 0
        try:
            self._pending_sidecar_out = next_out
            self._pending_sidecar_meta = self._build_sidecar_meta(
                prompt=str(getattr(self, "_extend_base_prompt", "") or ""),
                out_path=next_out,
                seed=int(getattr(self, "_extend_base_seed", int(self.seed.value()))),
                mode="direct",
                extra={"extend_segment": int(seg_idx)},
            )
        except Exception:
            self._pending_sidecar_out = next_out
            self._pending_sidecar_meta = None

        # Decrement remaining AFTER launching (matches WAN behavior).
        if self._extend_remaining > 0:
            self._extend_remaining -= 1

        venv_py = self._env_python()
        self._run(str(venv_py), args, root_dir())
        return True

    def _handle_extend_after_finished(self, code: int, segment_path: Path | None) -> bool:
        """Bookkeeping after each segment; returns True if chain continues."""
        if not getattr(self, "_extend_active", False):
            return False

        if code != 0 or segment_path is None or not segment_path.is_file():
            self._append("[extend] last run failed or output missing; stopping extend chain.")
            self._extend_active = False
            self._extend_remaining = 0
            self._extend_segments = []
            self._extend_pending_output = None
            return False

        try:
            self._extend_segments.append(segment_path)
        except Exception:
            pass

        if self._extend_remaining > 0:
            return self._start_extend_segment(segment_path)

        # Done; optionally merge
        merged: Path | None = None
        if getattr(self, "_extend_auto_merge", False):
            try:
                merged = self._auto_merge_extend_segments()
            except Exception:
                merged = None

            if isinstance(merged, Path) and merged.is_file():
                try:
                    segs = []
                    try:
                        segs = [str(p) for p in (getattr(self, "_extend_segments", []) or []) if isinstance(p, Path)]
                    except Exception:
                        segs = []
                    meta = self._build_sidecar_meta(
                        prompt=str(getattr(self, "_extend_base_prompt", "") or ""),
                        out_path=merged,
                        seed=int(getattr(self, "_extend_base_seed", int(self.seed.value()))),
                        mode="direct",
                        extra={"extend_merged": True, "segments": segs},
                    )
                    self._write_sidecar_json(merged, meta)
                except Exception:
                    pass

        self._extend_active = False
        self._extend_segments = []
        self._extend_pending_output = None
        return False

    def _on_read(self):
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode(errors="ignore")
        if not data:
            return
        # Some tools (HF hub / tqdm) update a single line using carriage returns.
        # Convert them to visible line breaks so the log panel shows activity.
        data = data.replace("\r", "\n")
        for line in data.split("\n"):
            line = (line or "").rstrip()
            if line:
                self._append(line)

    def _on_finished(self, code: int, status):
        self._append(f"\n<<< finished (code={code})")
        # Flush any buffered output.
        try:
            self._flush_pending_logs()
        except Exception:
            pass

        # Extend-chain handling (Direct run only)
        continued = False
        if getattr(self, "_extend_active", False) and (getattr(self, "_current_task", None) == "generate"):
            seg: Path | None = None
            try:
                p = getattr(self, "_extend_pending_output", None)
                if isinstance(p, Path) and p.is_file():
                    seg = p
            except Exception:
                seg = None
            if seg is None:
                try:
                    p = getattr(self, "_last_run_out_path", None)
                    if isinstance(p, Path) and p.is_file():
                        seg = p
                except Exception:
                    seg = None
            try:
                self._extend_pending_output = None
            except Exception:
                pass

            # Sidecar JSON for this segment (even if the chain continues)
            try:
                if code == 0 and seg is not None and isinstance(seg, Path) and seg.is_file():
                    meta = getattr(self, "_pending_sidecar_meta", None)
                    if not isinstance(meta, dict) or not meta:
                        meta = self._build_sidecar_meta(
                            prompt=str(getattr(self, "_extend_base_prompt", "") or ""),
                            out_path=seg,
                            seed=int(getattr(self, "_extend_base_seed", int(self.seed.value()))),
                            mode="direct",
                            extra={"extend": int(getattr(self, "_extend_total", 0) or 0)},
                        )
                    else:
                        try:
                            meta["output"] = str(seg)
                        except Exception:
                            pass
                    self._write_sidecar_json(seg, meta)
            except Exception:
                pass

            try:
                continued = bool(self._handle_extend_after_finished(code, seg))
            except Exception:
                continued = False

        if continued:
            # Another segment has been launched; keep the log timer and Busy state.
            return

        # Sidecar JSON for a successful Direct run
        try:
            if code == 0 and (getattr(self, "_current_task", None) == "generate"):
                outp = None
                try:
                    outp = getattr(self, "_pending_sidecar_out", None)
                except Exception:
                    outp = None
                if not isinstance(outp, Path) or not outp.is_file():
                    try:
                        p2 = getattr(self, "_last_run_out_path", None)
                        if isinstance(p2, Path) and p2.is_file():
                            outp = p2
                    except Exception:
                        outp = None
                if isinstance(outp, Path) and outp.is_file():
                    meta = getattr(self, "_pending_sidecar_meta", None)
                    if not isinstance(meta, dict) or not meta:
                        meta = self._build_sidecar_meta(
                            prompt=str(getattr(self, "_last_prompt", "") or ""),
                            out_path=outp,
                            seed=int(getattr(self, "_last_seed", int(self.seed.value()))),
                            mode="direct",
                            extra={"extend": int(getattr(self, "_extend_total", 0) or 0)},
                        )
                    else:
                        try:
                            meta["output"] = str(outp)
                        except Exception:
                            pass
                    self._write_sidecar_json(outp, meta)
        except Exception:
            pass

        # Stop the direct-run log timer and reset UI state.
        try:
            self._log_flush_timer.stop()
        except Exception:
            pass
        self._log_direct_active = False
        self._current_task = None
        self._set_busy(False)

    def on_stop(self):
        p = self.proc
        if not p or p.state() == QProcess.NotRunning:
            return
        self._append("\n>>> stopping…")
        # If we were in an extend chain, stop the chain as well.
        try:
            self._extend_active = False
            self._extend_remaining = 0
            self._extend_pending_output = None
        except Exception:
            pass
        try:
            p.terminate()
        except Exception:
            pass

        # Escalate to kill if it doesn't exit quickly.
        def _kill_if_needed():
            try:
                if p.state() != QProcess.NotRunning:
                    self._append(">>> force kill")
                    p.kill()
            except Exception:
                pass

        QTimer.singleShot(2500, _kill_if_needed)

    # ---------- env paths ----------
    def _env_python(self) -> Path:
        # MUST use python.exe from the tool's own /.hunyuan15_env/
        return root_dir() / ".hunyuan15_env" / "Scripts" / "python.exe"

    def _need_env(self) -> bool:
        return self._env_python().exists()

    # ---------- actions ----------
    def on_install(self):
        self._persist()
        self._current_task = "install"
        try:
            self._extend_active = False
            self._extend_remaining = 0
            self._extend_pending_output = None
        except Exception:
            pass
        bat = root_dir() / "presets" / "extra_env" / "hunuyan15_install.bat"
        if not bat.exists():
            QMessageBox.critical(self, "Missing file", f"Not found: {bat}")
            return
        # Use cmd.exe /c so QProcess runs .bat reliably
        self._run("cmd.exe", ["/c", str(bat)], root_dir())

    def _model_is_i2v(self) -> bool:
        try:
            return "_i2v" in str(self.model.currentData() or "")
        except Exception:
            return False

    def _start_image_path(self) -> str:
        try:
            return (self.start_image.text() or '').strip()
        except Exception:
            return ''

    
    def _validate_i2v(self) -> bool:
        is_i2v = self._model_is_i2v()
        img = self._start_image_path()

        # If Video→Video is enabled, auto-create a start image from the chosen source video.
        try:
            if getattr(self, "chk_video2video", None) and self.chk_video2video.isChecked():
                # Ensure we are on an I2V model, otherwise the Start image will be ignored.
                if not is_i2v:
                    try:
                        self._ensure_i2v_model_selected()
                    except Exception:
                        pass
                    is_i2v = self._model_is_i2v()

                if is_i2v and not img:
                    src: Path | None = None
                    try:
                        p0 = getattr(self, "_video2video_path", None)
                        if p0:
                            p0 = Path(p0)
                            if p0.exists():
                                src = p0
                    except Exception:
                        src = None

                    if src is None:
                        try:
                            src = self._find_latest_output_video()
                        except Exception:
                            src = None

                    if src is not None:
                        try:
                            if self._apply_v2v_source(src, silent=True):
                                img = self._start_image_path()
                        except Exception:
                            pass
        except Exception:
            pass

        if is_i2v and not img:
            QMessageBox.warning(
                self,
                "Missing start image",
                """You're using an Image-to-Video model.

Pick a Start image, or enable Video→Video and pick a source video.""",
            )
            return False

        if (not is_i2v) and img:
            self._push_log_lines(
                [
                    "[ui] Start image is set, but the current model is Text-to-Video; it will be ignored (choose an *_i2v model to use it)."
                ]
            )
        return True

    # ---------- Use Current (Media Player frame -> Start image) ----------

    def _find_main_with_video(self):
        """Best-effort search for the main window that owns the .video player."""
        try:
            p = self.parent()
            while p is not None:
                if hasattr(p, "video"):
                    return p
                try:
                    p = p.parent()
                except Exception:
                    break
        except Exception:
            pass
        try:
            for w in QApplication.topLevelWidgets():
                if hasattr(w, "video"):
                    return w
        except Exception:
            pass
        return None

    def _current_media_path(self) -> Path | None:
        """Best-effort: return the currently loaded media file path from the main player."""
        main = None
        try:
            main = self._find_main_with_video()
        except Exception:
            main = None
        if main is None:
            return None

        # Prefer the app's tracked current_path (kept in sync by player.open() usage across tabs)
        try:
            cur = getattr(main, "current_path", None)
            if cur:
                p = Path(str(cur)).expanduser()
                if p.exists():
                    return p
        except Exception:
            pass

        # Fallback: ask the video widget for a path / url
        try:
            video = getattr(main, "video", None)
        except Exception:
            video = None
        if video is None:
            return None

        for attr in ("current_path", "path", "file_path", "filepath", "source", "filename", "file"):
            try:
                val = getattr(video, attr, None)
            except Exception:
                val = None
            if not val:
                continue
            try:
                # QUrl -> local file
                if hasattr(val, "toLocalFile"):
                    val = val.toLocalFile()
            except Exception:
                pass
            try:
                p = Path(str(val)).expanduser()
                if p.exists():
                    return p
            except Exception:
                continue

        return None

    def _player_position_seconds(self, video_obj) -> float | None:
        """Best-effort: extract current playback position as seconds (float)."""
        try:
            objs = [video_obj]
            for a in ("player", "mediaPlayer", "mp", "qplayer"):
                try:
                    o = getattr(video_obj, a, None)
                except Exception:
                    o = None
                if o is not None:
                    objs.append(o)

            for o in objs:
                if o is None:
                    continue
                for name in ("position", "currentPosition", "pos", "position_ms", "current_ms", "currentTime"):
                    if not hasattr(o, name):
                        continue
                    try:
                        v = getattr(o, name)
                    except Exception:
                        continue
                    try:
                        v = v() if callable(v) else v
                    except Exception:
                        continue
                    if not isinstance(v, (int, float)):
                        continue
                    v = float(v)
                    # Heuristics: QMediaPlayer returns ms; some custom players may return seconds.
                    if "ms" in name or v > 10000.0:
                        return max(0.0, v / 1000.0)
                    return max(0.0, v)
        except Exception:
            pass
        return None

    def _grab_current_qimage(self) -> QImage | None:
        """Grab a QImage for the currently visible frame/image."""
        try:
            main = self._find_main_with_video()
            if main is None:
                return None
            video = getattr(main, "video", None)
            if video is None:
                return None

            # 1) Prefer a direct currentFrame QImage
            img = getattr(video, "currentFrame", None)
            if isinstance(img, QImage) and not img.isNull():
                return img

            # 2) Try video.label.pixmap()
            try:
                label = getattr(video, "label", None)
                if label is not None and hasattr(label, "pixmap"):
                    pm = label.pixmap()
                    if pm is not None and not pm.isNull():
                        return pm.toImage()
            except Exception:
                pass

            # 3) As a last resort, grab the video widget itself
            try:
                if hasattr(video, "grab"):
                    pm = video.grab()
                    if isinstance(pm, QPixmap) and not pm.isNull():
                        return pm.toImage()
            except Exception:
                pass
        except Exception:
            pass
        return None

    def _save_qimage_jpg95(self, qimg: QImage, out_path: Path) -> bool:
        """Save QImage as JPEG (quality 95) while preserving pixel dimensions."""
        try:
            img = qimg
            try:
                if img.hasAlphaChannel():
                    img = img.convertToFormat(QImage.Format_RGB32)
            except Exception:
                pass
            return bool(img.save(str(out_path), "JPG", 95))
        except Exception:
            return False

    def _export_current_media_to_temp_jpg(self) -> Path | None:
        """Export the current player frame/image to a temp JPG at original resolution."""
        try:
            import time as _time
            out_jpg = _temp_dir() / f"hunyuan15_current_{int(_time.time())}.jpg"
        except Exception:
            return None

        main = None
        try:
            main = self._find_main_with_video()
        except Exception:
            main = None

        video = None
        try:
            video = getattr(main, "video", None) if main is not None else None
        except Exception:
            video = None

        src = self._current_media_path()

        # 1) If the player has a real source path, prefer exporting from that (full-res).
        if src is not None:
            ext = (src.suffix or "").lower()
            img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            vid_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg"}
            try:
                if ext in img_exts:
                    q = QImage(str(src))
                    if not q.isNull() and self._save_qimage_jpg95(q, out_jpg):
                        return out_jpg
                elif ext in vid_exts:
                    ff = _ffmpeg_exe()
                    if ff:
                        sec = self._player_position_seconds(video) if video is not None else None
                        tmp_png = out_jpg.with_suffix(".png")
                        cmd = [ff, "-y", "-hide_banner"]
                        if sec is not None:
                            cmd += ["-ss", f"{sec:.3f}"]
                        cmd += ["-i", str(src), "-frames:v", "1", str(tmp_png)]
                        try:
                            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                            q = QImage(str(tmp_png))
                            if not q.isNull() and self._save_qimage_jpg95(q, out_jpg):
                                try:
                                    tmp_png.unlink(missing_ok=True)
                                except Exception:
                                    try:
                                        if tmp_png.exists():
                                            tmp_png.unlink()
                                    except Exception:
                                        pass
                                return out_jpg
                        except Exception:
                            pass
            except Exception:
                pass

        # 2) Fallback: whatever is currently visible (may be scaled).
        qimg = self._grab_current_qimage()
        if qimg is None or qimg.isNull():
            return None
        if self._save_qimage_jpg95(qimg, out_jpg):
            return out_jpg
        return None

    def on_use_current(self):
        """Use the current Media Player frame/image as the Start image (I2V)."""
        # Safety: requires an Image-to-Video model.
        try:
            cur_model = str(self.model.currentData() or "")
        except Exception:
            cur_model = ""
        if "_i2v" not in cur_model:
            try:
                QMessageBox.warning(
                    self,
                    "Use Current",
                    "You're currently on a text-to-video model.\n\n"
                    "Change the model to an Image-to-Video preset ( *_i2v ) first, then click 'Use Current' again."
                )
            except Exception:
                pass
            return

        tmp_inp = self._export_current_media_to_temp_jpg()
        if tmp_inp is None or not Path(str(tmp_inp)).exists():
            try:
                QMessageBox.warning(
                    self,
                    "No current frame",
                    "No current frame or image was found.\n\n"
                    "Load an image or pause a video in the Media Player first."
                )
            except Exception:
                pass
            return

        try:
            self.start_image.setText(str(tmp_inp))
        except Exception:
            pass

        try:
            q = QImage(str(tmp_inp))
            if not q.isNull():
                self._append(f"Using Media Player current frame as start image (JPG 95%, {q.width()}x{q.height()}):\n{tmp_inp}")
            else:
                self._append(f"Using Media Player current frame as start image (JPG 95%):\n{tmp_inp}")
        except Exception:
            try:
                self._append(f"Using Media Player current frame as start image (JPG 95%):\n{tmp_inp}")
            except Exception:
                pass


    def on_pick_image(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Select start image",
            str(root_dir()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff);;All files (*)",
        )
        if fn:
            try:
                self.start_image.setText(fn)
            except Exception:
                pass

    def on_clear_image(self):
        try:
            self.start_image.setText("")
        except Exception:
            pass

    def on_toggle_video2video(self, checked: bool):
        try:
            self.btn_pick_v2v.setEnabled(bool(checked))
            self.btn_use_last_v2v.setEnabled(bool(checked))
        except Exception:
            pass
        if checked:
            try:
                if not getattr(self, "_video2video_path", None):
                    self._append("[ui] Video→Video enabled: pick a source video (or use last output).")
            except Exception:
                pass
        try:
            self._schedule_persist()
        except Exception:
            pass

    def on_pick_v2v_video(self):
        try:
            fn, _ = QFileDialog.getOpenFileName(
                self,
                "Select source video",
                "",
                "Video files (*.mp4 *.mov *.mkv *.webm *.avi);;All files (*.*)",
            )
        except Exception:
            fn = ""
        if not fn:
            return
        try:
            self._apply_v2v_source(Path(fn).resolve(), silent=False)
        except Exception as e:
            try:
                self._append(f"[v2v] failed to set source: {e}")
            except Exception:
                pass

    def on_use_last_output_as_v2v(self):
        try:
            p = self._find_latest_output_video()
        except Exception:
            p = None
        if not p:
            try:
                QMessageBox.information(self, "Video→Video", "Couldn't find any previous Hunyuan15 output videos yet.")
            except Exception:
                pass
            return
        try:
            self._apply_v2v_source(Path(p).resolve(), silent=False)
        except Exception as e:
            try:
                self._append(f"[v2v] failed to use last output: {e}")
            except Exception:
                pass

    def _find_latest_output_video(self) -> Path | None:
        """Find newest output MP4 in output/video/hunyuan15/."""
        out_dir = root_dir() / "output" / "video" / "hunyuan15"
        try:
            if not out_dir.exists():
                return None
        except Exception:
            return None

        newest: Path | None = None
        newest_mtime = -1.0
        try:
            for p in out_dir.glob("*.mp4"):
                try:
                    if not p.is_file():
                        continue
                    mt = p.stat().st_mtime
                    if mt > newest_mtime:
                        newest_mtime = mt
                        newest = p
                except Exception:
                    continue
        except Exception:
            return None
        return newest

    def _ensure_i2v_model_selected(self) -> bool:
        """Switch model selection to an I2V variant, matching current model/resolution when possible."""
        try:
            cur = str(self.model.currentData() or "")
        except Exception:
            cur = ""
        mapped = ""
        if cur:
            if "_i2v" in cur:
                mapped = cur
            elif "_t2v" in cur:
                mapped = cur.replace("_t2v", "_i2v")

        if mapped:
            try:
                idx = self.model.findData(mapped)
                if idx >= 0:
                    self.model.setCurrentIndex(idx)
                    return True
            except Exception:
                pass

        try:
            h = int(self.resolution.currentData() or 480)
        except Exception:
            h = 480
        target = "720p_i2v" if h >= 720 else "480p_i2v"
        try:
            idx = self.model.findData(target)
            if idx >= 0:
                self.model.setCurrentIndex(idx)
                return True
        except Exception:
            pass
        return False

    def _apply_v2v_source(self, video: Path, silent: bool = False) -> bool:
        """Set Video→Video source, extract last frame, and fill Start image."""
        try:
            p = Path(video)
        except Exception:
            return False
        if not p.exists():
            if not silent:
                try:
                    QMessageBox.warning(self, "Video→Video", "Selected source video no longer exists on disk.")
                except Exception:
                    pass
            return False

        try:
            if not self.chk_video2video.isChecked():
                self.chk_video2video.setChecked(True)
        except Exception:
            pass

        self._video2video_path = p

        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            info = f"{p.name}  ({size_mb:.1f} MB)"
        except Exception:
            info = p.name
        try:
            self.lbl_video2_info.setText(info)
        except Exception:
            pass

        try:
            self._ensure_i2v_model_selected()
        except Exception:
            pass

        try:
            frame_path = self._next_extend_frame_path()
            ok = self._extract_last_frame(p, frame_path)
            if not ok:
                if not silent:
                    QMessageBox.warning(self, "Video→Video", "Failed to extract last frame from the source video.")
                return False
            try:
                self.start_image.setText(str(frame_path))
            except Exception:
                pass
            try:
                self._append(f"[v2v] Using last frame from {p.name} as Start image.")
            except Exception:
                pass
            try:
                self._schedule_persist()
            except Exception:
                pass
            return True
        except Exception as e:
            if not silent:
                try:
                    QMessageBox.warning(self, "Video→Video", f"Error while preparing start image: {e}")
                except Exception:
                    pass
            return False


    def on_download(self):
        self._persist()
        self._current_task = "download"
        try:
            self._extend_active = False
            self._extend_remaining = 0
            self._extend_pending_output = None
        except Exception:
            pass
        if not self._need_env():
            QMessageBox.warning(self, "Install first", "Please run Install / Update first.")
            return

        venv_py = self._env_python()
        model_key = self.model.currentData()

        self._run(
            str(venv_py),
            [str(root_dir() / "helpers" / "hunyuan15_cli.py"), "download", "--model", model_key],
            root_dir(),
        )

    def on_generate(self):
        self._persist()
        if not self._need_env():
            QMessageBox.warning(self, "Install first", "Please run Install / Update first.")
            return

        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Please enter a prompt.")
            return

        if not self._validate_i2v():
            return

        # If Queue toggle is ON, route Direct run to Queue (unless Extend is enabled).
        try:
            ext_now = int(self.extend.value())
        except Exception:
            ext_now = 0
        if self._queue_enabled() and ext_now <= 0:
            try:
                self.on_queue_generate()
            except Exception:
                pass
            return

        venv_py = self._env_python()
        model_key = self.model.currentData()
        out_path = self._apply_autoname(prompt, int(self.seed.value()), extra=("i2v" if (self._model_is_i2v() and self._start_image_path()) else None))

        args = self._build_generate_args(prompt=prompt, out_path=out_path, seed_override=self.seed.value())

        # Prepare extend-chain state (Direct run only)
        try:
            ext = int(self.extend.value())
        except Exception:
            ext = 0
        if ext > 0 and not _ffmpeg_exe():
            self._push_log_lines([
                "[ui] Extend is enabled but ffmpeg.exe was not found; extend will be disabled for this run."
            ])
            ext = 0

        self._extend_active = ext > 0
        self._extend_remaining = int(ext)
        self._extend_segments = []
        self._extend_frame_index = 0
        self._extend_auto_merge = bool(self.extend_merge.isChecked())
        self._extend_pending_output = (out_path if self._extend_active else None)
        self._extend_base_out = out_path
        self._extend_base_model = str(model_key) if model_key is not None else None
        self._extend_base_prompt = str(prompt)
        self._extend_base_seed = int(self.seed.value())
        self._last_run_out_path = out_path
        self._current_task = "generate"

        # Prepare sidecar metadata for this run (written when the process finishes successfully)
        try:
            self._pending_sidecar_out = out_path
            self._pending_sidecar_meta = self._build_sidecar_meta(prompt=prompt, out_path=out_path, seed=int(self.seed.value()), mode="direct", extra={"extend": int(ext)})
        except Exception:
            self._pending_sidecar_out = out_path
            self._pending_sidecar_meta = None

        # Direct run: keep newest log lines at top, and batch UI refresh every 3s while running.
        self._log_direct_active = True
        try:
            self._log_flush_timer.start()
        except Exception:
            pass

        self._run(str(venv_py), args, root_dir())

    def on_queue_generate(self):
        """Queue a single generation job to the app's Queue tab."""
        self._persist()
        try:
            if int(self.extend.value()) > 0:
                self._push_log_lines(["[ui] Extend is Direct-run only; queue will create a single clip (no chaining)."], render=True)
        except Exception:
            pass
        if not self._need_env():
            QMessageBox.warning(self, "Install first", "Please run Install / Update first.")
            return
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Please enter a prompt.")
            return

        if not self._validate_i2v():
            return

        out_path = self._apply_autoname(prompt, int(self.seed.value()), extra=("i2v" if (self._model_is_i2v() and self._start_image_path()) else None))
        args = self._build_generate_args(prompt=prompt, out_path=out_path, seed_override=self.seed.value())
        meta = self._build_sidecar_meta(prompt=prompt, out_path=out_path, seed=int(self.seed.value()), mode="queue", extra={"label": f"hunyuan15: {prompt[:64]}", "queued": True})
        self._queue_job(args=args, outfile=out_path, label=f"hunyuan15: {prompt[:64]}", sidecar_meta=meta)

    def on_batch_queue(self):
        """Queue multiple jobs with the current settings.

        - For Text→Video models: queue seed variations (existing behavior).
        - For Image→Video models: show the Wan2.2-style batch strategy popup (files/folder/repeat).
        """
        self._persist()
        try:
            if int(self.extend.value()) > 0:
                self._push_log_lines(
                    ["[ui] Extend is Direct-run only; batch queue will create single clips (no chaining)."],
                    render=True,
                )
        except Exception:
            pass

        if not self._need_env():
            QMessageBox.warning(self, "Install first", "Please run Install / Update first.")
            return

        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Please enter a prompt.")
            return

        if not self._validate_i2v():
            return

        # Image→Video: choose batch strategy (files / folder / repeat current image)
        if self._model_is_i2v():
            try:
                self._show_i2v_batch_dialog()
            except Exception as e:
                self._append(f"[ui] batch dialog error: {e}")
            return

        # Text→Video: seed variations
        n, ok = QInputDialog.getInt(
            self,
            "Batch queue",
            "How many variations to queue?",
            4,
            1,
            30,
            1,
        )
        if not ok:
            return

        base_seed = int(self.seed.value())

        queued = 0
        for i in range(int(n)):
            # Seed: increment if user set an explicit seed; otherwise keep random (-1)
            seed_override = -1
            if base_seed >= 0:
                seed_override = base_seed + i

            extra = f"b{i+1:02d}" if seed_override < 0 else None
            out_path = self._apply_autoname(prompt, seed_override, extra=extra)

            args = self._build_generate_args(prompt=prompt, out_path=out_path, seed_override=seed_override)
            meta = self._build_sidecar_meta(
                prompt=prompt,
                out_path=out_path,
                seed=int(seed_override),
                mode="queue",
                extra={
                    "label": f"hunyuan15(b{i+1:02d}): {prompt[:48]}",
                    "queued": True,
                    "batch_index": int(i) + 1,
                    "batch_total": int(n),
                },
            )
            jid = self._queue_job(
                args=args,
                outfile=out_path,
                label=f"hunyuan15(b{i+1:02d}): {prompt[:48]}",
                quiet=True,
                sidecar_meta=meta,
            )
            if jid:
                queued += 1

        self._append(f"[queue] queued {queued} job(s) to jobs/pending.")
        self._goto_queue_tab()

    # ---------- I2V batch strategy (Wan2.2-style) ----------

    def _show_i2v_batch_dialog(self):
        box = QMessageBox(self)
        box.setWindowTitle("Image batch")
        box.setText("You're using an Image→Video model. Choose a batch strategy:")
        btn_files = box.addButton("Batch images (Files…)", QMessageBox.AcceptRole)
        btn_folder = box.addButton("Batch images (Folder…)", QMessageBox.AcceptRole)
        btn_repeat = box.addButton("Repeat current image…", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()

        clicked = box.clickedButton()
        if clicked == btn_files:
            self._run_i2v_batch_multi(mode="files")
        elif clicked == btn_folder:
            self._run_i2v_batch_multi(mode="folder")
        elif clicked == btn_repeat:
            self._run_i2v_batch_repeat()
        else:
            return

    def _run_i2v_batch_multi(self, mode: str = "files"):
        # Pick images
        image_paths: list[Path] = []
        try:
            start_dir = getattr(self, "_last_i2v_batch_dir", "")
        except Exception:
            start_dir = ""

        if mode == "folder":
            folder = QFileDialog.getExistingDirectory(self, "Select folder with images", start_dir or "")
            if not folder:
                return
            try:
                self._last_i2v_batch_dir = folder
            except Exception:
                pass
            image_paths = self._collect_images_from_folder(folder)
        else:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select images",
                start_dir or "",
                "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff)",
            )
            if not files:
                return
            try:
                self._last_i2v_batch_dir = str(Path(files[0]).parent)
            except Exception:
                pass
            image_paths = [Path(p) for p in files if p]

        image_paths = [p for p in image_paths if p.exists()]
        if not image_paths:
            QMessageBox.information(self, "No images", "No supported image files were found.")
            return

        # Confirm
        if QMessageBox.question(
            self,
            "Confirm batch",
            f"{len(image_paths)} images loaded. Continue and add them to the queue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return

        self._enqueue_i2v_batch_jobs(image_paths)

    def _run_i2v_batch_repeat(self):
        img = self._start_image_path()
        if not img or not Path(img).exists():
            QMessageBox.warning(self, "Missing start image", "Pick a Start image first.")
            return

        n, ok = QInputDialog.getInt(
            self,
            "Repeat current image",
            "How many copies to queue?",
            4,
            1,
            200,
            1,
        )
        if not ok:
            return

        if QMessageBox.question(
            self,
            "Confirm batch",
            f"{int(n)} copies will be queued using the current Start image. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return

        self._enqueue_i2v_batch_jobs([Path(img)] * int(n), repeat_mode=True)

    def _collect_images_from_folder(self, folder: str) -> list[Path]:
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
        paths: list[Path] = []
        try:
            d = Path(folder)
            if not d.exists():
                return []
            for p in d.iterdir():
                try:
                    if p.is_file() and p.suffix.lower() in exts:
                        paths.append(p)
                except Exception:
                    pass
        except Exception:
            return []
        paths.sort(key=lambda p: p.name.lower())
        return paths

    def _enqueue_i2v_batch_jobs(self, image_paths: list[Path], repeat_mode: bool = False):
        """Enqueue one Queue job per image using the current UI settings."""
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Missing prompt", "Please enter a prompt.")
            return

        base_seed = int(self.seed.value())
        slug = ""
        try:
            slug = self._first3_words_slug(prompt)
        except Exception:
            slug = "prompt"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        total = len(image_paths)
        queued = 0

        for i, imgp in enumerate(image_paths):
            try:
                seed_override = -1
                if base_seed >= 0:
                    seed_override = base_seed + i

                seed_part = str(seed_override) if seed_override >= 0 else "rand"
                img_stem = imgp.stem[:48]
                fname = f"H15_{slug}_{seed_part}_{ts}_i2v_{img_stem}_b{i+1:02d}.mp4"
                out_path = self._output_path_for_name(fname)

                args = self._build_generate_args(
                    prompt=prompt,
                    out_path=out_path,
                    seed_override=seed_override,
                    image_override=str(imgp),
                )

                meta = self._build_sidecar_meta(
                    prompt=prompt,
                    out_path=out_path,
                    seed=int(seed_override),
                    mode="queue",
                    extra={
                        "queued": True,
                        "batch_mode": "repeat" if repeat_mode else "multi",
                        "batch_total": int(total),
                        "batch_index": int(i) + 1,
                        "source_image": str(imgp),
                    },
                )

                label = f"hunyuan15(i2v {i+1:02d}/{total}): {imgp.stem[:40]}"
                jid = self._queue_job(
                    args=args,
                    outfile=out_path,
                    label=label,
                    quiet=True,
                    sidecar_meta=meta,
                )
                if jid:
                    queued += 1
            except Exception as e:
                self._append(f"[queue] ERROR (i2v batch): {e}")

        self._append(f"[queue] queued {queued} job(s) to jobs/pending.")
        self._goto_queue_tab()

    def _goto_queue_tab(self):
        try:
            win = self.window()
            tabs = win.findChild(QTabWidget)
            if tabs:
                for i in range(tabs.count()):
                    if tabs.tabText(i).strip().lower() == "queue":
                        tabs.setCurrentIndex(i)
                        break
        except Exception:
            pass

    def _first3_words_slug(self, prompt: str) -> str:
        words = [w for w in re.split(r"\s+", (prompt or "").strip()) if w]
        words = words[:3] if words else ["prompt"]
        cleaned: list[str] = []
        for w in words:
            w2 = re.sub(r"[^A-Za-z0-9]+", "_", w).strip("_")
            if w2:
                cleaned.append(w2)
        if not cleaned:
            cleaned = ["prompt"]
        slug = "_".join(cleaned)
        return slug[:40]

    def _current_output_dir(self) -> Path:
        """Return the folder where the current output_name will be written."""
        try:
            name = (self.output_name.text() or "").strip()
        except Exception:
            name = ""
        try:
            out_path = self._output_path_for_name(name)
        except Exception:
            out_path = root_dir() / "output" / "video" / "hunyuan15" / "hunyuan15_output.mp4"
        try:
            out_dir = Path(str(out_path)).parent
        except Exception:
            out_dir = root_dir() / "output" / "video" / "hunyuan15"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return out_dir

    def _goto_media_explorer_tab(self):
        """Switch main tabs to Media Explorer and return the MediaExplorerTab widget (best-effort)."""
        win = None
        try:
            win = self.window()
        except Exception:
            win = None

        tabs = None
        try:
            if win is not None:
                tabs = getattr(win, "tabs", None)
        except Exception:
            tabs = None

        if tabs is None:
            try:
                tabs = win.findChild(QTabWidget) if win is not None else None
            except Exception:
                tabs = None

        if tabs is None:
            return None

        try:
            for i in range(tabs.count()):
                if (tabs.tabText(i) or "").strip().lower() == "media explorer":
                    try:
                        tabs.setCurrentIndex(i)
                    except Exception:
                        pass
                    w = None
                    try:
                        w = tabs.widget(i)
                    except Exception:
                        w = None
                    try:
                        if isinstance(w, QScrollArea):
                            w = w.widget()
                    except Exception:
                        pass
                    return w
        except Exception:
            pass
        return None

    def on_view_results(self):
        """Open Media Explorer tab and scan the current output folder."""
        try:
            out_dir = self._current_output_dir()
        except Exception:
            out_dir = root_dir() / "output" / "video" / "hunyuan15"
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        tab = self._goto_media_explorer_tab()
        if tab is None:
            try:
                QMessageBox.information(self, "View results", "Media Explorer tab not found.")
            except Exception:
                pass
            return

        try:
            if hasattr(tab, "set_root_folder"):
                tab.set_root_folder(str(out_dir))
            if hasattr(tab, "rescan"):
                tab.rescan()
        except Exception as e:
            try:
                QMessageBox.warning(self, "View results", f"Could not scan folder:\n{out_dir}\n\n{e}")
            except Exception:
                pass

    def _make_autoname(self, prompt: str, seed: int, extra: str | None = None) -> str:
        slug = self._first3_words_slug(prompt)
        seed_part = str(seed) if seed >= 0 else "rand"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"H15_{slug}_{seed_part}_{ts}"
        if extra:
            base = f"{base}_{extra}"
        return f"{base}.mp4"

    def _apply_autoname(self, prompt: str, seed: int, extra: str | None = None) -> Path:
        name = self._make_autoname(prompt, seed, extra=extra)
        try:
            self.output_name.setText(name)
        except Exception:
            pass
        return self._output_path_for_name(name)

    def _output_path_for_name(self, name: str) -> Path:
        out_dir = root_dir() / "output" / "video" / "hunyuan15"
        out_dir.mkdir(parents=True, exist_ok=True)
        p = Path(name.strip()) if name else Path("hunyuan15_output.mp4")
        if p.is_absolute():
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return p
        return out_dir / p

    def _res_to_hw(self, p: int, aspect_mode: str | None = None) -> tuple[int, int]:
        """Convert a single 'p' number into (height, width) aligned to multiples of 16.

        Historically the Resolution presets store a single number, and we derive the other side from an aspect ratio.

        aspect_mode:
            - 'landscape' (16:9): p is treated as height.
            - 'portrait'  (9:16): p is treated as width.
            - 'square'    (1:1):  p is treated as both sides.
        """
        try:
            p = int(p)
        except Exception:
            p = 480

        # Some presets use 0 to mean "just use a small square" for quick testing.
        if p <= 0:
            p = 256

        # Align base 'p' to 16 and clamp minimum.
        p = int(round(p / 16.0) * 16)
        if p < 256:
            p = 256

        mode = (aspect_mode or "landscape").strip().lower()
        if mode not in ("landscape", "portrait", "square"):
            mode = "landscape"

        if mode == "square":
            h = p
            w = p
        elif mode == "portrait":
            # p is width, derive height for 9:16
            w = p
            h = int(round((w * 16.0 / 9.0) / 16.0) * 16)
        else:
            # landscape: p is height, derive width for 16:9
            h = p
            w = int(round((h * 16.0 / 9.0) / 16.0) * 16)

        # Final safety clamps
        if w < 256:
            w = 256
        if h < 256:
            h = 256
        return (h, w)

    def _build_generate_args(
        self,
        prompt: str,
        out_path: Path,
        seed_override: int,
        model_key_override: str | None = None,
        image_override: str | None = None,
    ) -> list[str]:
        model_key = model_key_override if model_key_override is not None else self.model.currentData()
        attn_key = self.attn.currentData()
        args = [
            str(root_dir() / "helpers" / "hunyuan15_cli.py"),
            "generate",
            "--model",
            str(model_key),
            "--prompt",
            str(prompt),
            "--output",
            str(out_path),
            "--frames",
            str(self.frames.value()),
            "--steps",
            str(self.steps.value()),
            "--fps",
            str(self.fps.value()),
            "--bitrate-kbps",
            str(int(self.bitrate_kbps.value())),
        ]
        # Negative prompt (optional)
        try:
            neg = (self.negative_prompt.toPlainText() or "").strip()
        except Exception:
            neg = ""
        if neg:
            args += ["--negative", neg]

        # Start image is only used for *_i2v models
        img = (image_override if image_override is not None else self._start_image_path())
        if img and ("_i2v" in str(model_key)):
            args += ["--image", str(img)]
        # Resolution / aspect handling
        try:
            h, w = self._res_to_hw(
                int(self.resolution.currentData()),
                str(getattr(self, 'aspect', None).currentData() if getattr(self, 'aspect', None) else 'landscape'),
            )
            args += ["--height", str(h), "--width", str(w)]
        except Exception:
            pass

        args += [
            "--attn",
            str(attn_key),
        ]
        try:
            if int(seed_override) >= 0:
                args += ["--seed", str(int(seed_override))]
        except Exception:
            pass
        if self.offload.isChecked():
            args += ["--offload"]
        if getattr(self, 'group_offload', None) is not None and self.group_offload.isChecked():
            args += ["--group-offload"]
        if getattr(self, 'first_block_cache', None) is not None and self.first_block_cache.isChecked():
            args += ["--first-block-cache"]
            try:
                thr = float(self.fbc_thresh_spin.value())
            except Exception:
                thr = 0.05
            args += ["--first-block-cache-threshold", str(thr)]
        if getattr(self, 'pab', None) is not None and self.pab.isChecked():
            args += ["--pyramid-attn-broadcast"]
        if self.attn_slicing.isChecked():
            args += ["--attn-slicing"]
        if self.vae_slicing.isChecked():
            args += ["--vae-slicing"]
        if self.tiling.isChecked():
            args += ["--tiling"]
        return args

    def _queue_job(self, args: list[str], outfile: Path, label: str = "hunyuan15", quiet: bool = False, sidecar_meta: dict | None = None) -> str | None:
        """Enqueue a command to the app's queue system.

        We reuse the existing tools queue job schema (tools_ffmpeg) because it's already
        supported by the worker and accepts an arbitrary command list in ffmpeg_cmd.
        """
        try:
            try:
                from helpers.queue_adapter import enqueue_tool_job as enq  # type: ignore
            except Exception:
                from queue_adapter import enqueue_tool_job as enq  # type: ignore

            venv_py = self._env_python()
            cmd_list = [str(venv_py)] + list(args)

            # If sidecar_meta is provided, wrap the CLI call so the worker writes <output>.json on success.
            if isinstance(sidecar_meta, dict) and sidecar_meta:
                try:
                    meta_json = json.dumps(sidecar_meta, ensure_ascii=False)
                except Exception:
                    meta_json = "{}"
                cmd_list = [str(venv_py), "-u", "-c", _QUEUE_SIDECAR_WRAPPER_CODE] + list(args) + [_QUEUE_SIDECAR_META_FLAG, meta_json]

            # Make the Queue row human-friendly:
            # - Queue UI prefers args['label'] when present.
            # - Also store a friendly model name so the DONE row can show it.
            try:
                friendly_model = str(self.model.currentText()).strip()
            except Exception:
                friendly_model = ""
            try:
                model_key = str(self.model.currentData()).strip()
            except Exception:
                model_key = ""

            # IMPORTANT: give the queue a real, existing "input" path.
            # The Queue UI uses job['input'] for thumbnails/subtitles and can behave badly with an empty string.
            in_path = ""
            try:
                img = self._start_image_path()
                if img and Path(img).exists():
                    in_path = str(img)
            except Exception:
                in_path = ""
            if not in_path:
                try:
                    in_path = str(root_dir() / "helpers" / "hunyuan15_cli.py")
                except Exception:
                    in_path = "hunyuan15_cli.py"

            jid = enq(
                "tools_ffmpeg",
                in_path,
                str(Path(str(outfile)).parent),
                {
                    "ffmpeg_cmd": cmd_list,
                    "outfile": str(outfile),
                    "cwd": str(root_dir()),
                    "label": str(label),
                    "engine": "hunyuan15",
                    "ai_model": friendly_model or model_key or "HunyuanVideo 1.5",
                    "model": model_key,
                },
                priority=650,
            )
            if not quiet:
                self._append(f"[queue] queued to jobs/pending: {jid}")
                self._goto_queue_tab()
            return str(jid)
        except Exception as e:
            if not quiet:
                self._append(f"[queue] ERROR: {e}")
            return None

    def on_open_output(self):
        out_dir = root_dir() / "output" / "video" / "hunyuan15"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Windows: explorer
        self._run("explorer.exe", [str(out_dir)], root_dir())

    def on_clear_log(self):
        try:
            self._log_lines.clear()
        except Exception:
            pass
        try:
            self._log_pending.clear()
        except Exception:
            pass
        try:
            self.log.clear()
        except Exception:
            pass


# Backward-compatible alias (older code imported Runner)
Runner = Hunyuan15ToolWidget


def install_hunyuan15_tool(parent_pane, section):
    """Installer-style entrypoint used by helpers/tools_tab.py."""
    wrap = QWidget()
    lay = QVBoxLayout(wrap)
    lay.setContentsMargins(0, 0, 0, 0)
    widget = Hunyuan15ToolWidget(parent=wrap, standalone=False)
    lay.addWidget(widget)
    section.setContentLayout(lay)
    return widget


def main():
    app = QApplication(sys.argv)
    w = Hunyuan15ToolWidget(standalone=True)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()