# FrameVision helper: Bernini-R 1.3B small UI
# Apache-2.0 compatible helper wrapper around ByteDance Bernini inference scripts.
# This file intentionally does not vendor/copy Bernini model code.

from __future__ import annotations


# --- FrameVision media-explorer results opener ------------------------------
def _fv_open_results_in_media_explorer(widget, folder, preset="images") -> bool:
    """Open/scan a results folder in FrameVision Media Explorer when embedded.

    Falls back to the operating-system file explorer when the main FrameVision
    helper is not available (for standalone tool runs).
    """
    try:
        from pathlib import Path as _Path
        _folder = _Path(folder).expanduser()
        try:
            _folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        _folder_s = str(_folder)
    except Exception:
        return False

    def _try_main(_mw) -> bool:
        try:
            if _mw is not None and hasattr(_mw, "open_media_explorer_folder"):
                try:
                    _mw.open_media_explorer_folder(_folder_s, preset=preset, include_subfolders=False)
                    return True
                except TypeError:
                    kwargs = {"include_subfolders": False}
                    if preset == "images":
                        kwargs.update({"want_images": True, "want_videos": False, "want_audio": False})
                    elif preset == "videos":
                        kwargs.update({"want_images": False, "want_videos": True, "want_audio": False})
                    elif preset == "audio":
                        kwargs.update({"want_images": False, "want_videos": False, "want_audio": True})
                    _mw.open_media_explorer_folder(_folder_s, **kwargs)
                    return True
        except Exception:
            pass
        return False

    try:
        _w = widget
        while _w is not None:
            if _try_main(_w):
                return True
            try:
                _w = _w.parent()
            except Exception:
                break
    except Exception:
        pass

    try:
        from PySide6.QtWidgets import QApplication as _QApplication
        _app = _QApplication.instance()
        if _app is not None:
            for _w in _app.topLevelWidgets():
                if _try_main(_w):
                    return True
    except Exception:
        pass

    try:
        from PySide6.QtGui import QDesktopServices as _QDesktopServices
        from PySide6.QtCore import QUrl as _QUrl
        _QDesktopServices.openUrl(_QUrl.fromLocalFile(_folder_s))
        return True
    except Exception:
        pass

    try:
        import os as _os, sys as _sys, subprocess as _subprocess
        if _os.name == "nt":
            _os.startfile(_folder_s)  # type: ignore[attr-defined]
        elif _sys.platform == "darwin":
            _subprocess.Popen(["open", _folder_s])
        else:
            _subprocess.Popen(["xdg-open", _folder_s])
        return True
    except Exception:
        return False
# ---------------------------------------------------------------------------

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QProcess, QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


# From bernini/cli.py in the uploaded Bernini repo.
GUIDANCE_MODES = [
    "rv2v",
    "v2v",
    "v2v_chain",
    "t2v",
    "r2v_apg",
    "v2v_apg",
    "t2v_apg",
    "vae_txt_vit_wapg",
    "rv2v_wapg",
]

# From gradio_demo.py / Bernini prompt_enhancer task mappings.
# Keep the UI task list close to the fuller Bernini/ComfyUI enhancer list so the
# right task-specific prompt template is selected when --use_pe is enabled.
TASK_TYPE_CHOICES = [
    "t2i",
    "t2v",
    "i2i",
    "r2i",
    "i2v",
    "r2v",
    "v2v",
    "vi2v",
    "rv2v",
    "ads2v",
    "vrc2v",
    "mv2v",
]
TEXT_ONLY_TASKS = {"t2i", "t2v"}
EDIT_OR_REFERENCE_TASKS = set(TASK_TYPE_CHOICES) - TEXT_ONLY_TASKS
GUIDANCE_MODE_BY_TASK = {
    "t2i": "t2v_apg",
    "t2v": "t2v_apg",
    "i2i": "v2v",
    "r2i": "r2v_apg",
    "i2v": "t2v_apg",
    "r2v": "r2v_apg",
    "v2v": "v2v_apg",
    "vi2v": "v2v_apg",
    "rv2v": "rv2v",
    "ads2v": "v2v_apg",
    "vrc2v": "v2v_apg",
    "mv2v": "v2v_apg",
}
TASK_INPUTS = {
    "t2i": {"video": False, "image_role": "none", "images": False},
    "t2v": {"video": False, "image_role": "none", "images": False},
    "i2i": {"video": False, "image_role": "source", "images": False},
    "r2i": {"video": False, "image_role": "reference", "images": True},
    "i2v": {"video": False, "image_role": "source", "images": False},
    "r2v": {"video": False, "image_role": "reference", "images": True},
    "v2v": {"video": True, "image_role": "none", "images": False},
    "vi2v": {"video": True, "image_role": "none", "images": False},
    "rv2v": {"video": True, "image_role": "reference", "images": True},
    "ads2v": {"video": True, "image_role": "reference", "images": True},
    "vrc2v": {"video": True, "image_role": "none", "images": False},
    "mv2v": {"video": True, "image_role": "none", "images": False},
}
IMAGE_TASKS = {"t2i", "i2i", "r2i"}

TASK_LABELS = {
    "t2i": "Text → image",
    "t2v": "Text → video",
    "i2i": "Image editing",
    "r2i": "Subject/reference → image",
    "i2v": "Image → video",
    "r2v": "Subject/reference → video",
    "v2v": "Video editing",
    "vi2v": "Video edit — content propagation",
    "rv2v": "Video edit with reference",
    "ads2v": "Ads / content insertion",
    "vrc2v": "Video edit — action / position",
    "mv2v": "Video edit — style / motion",
}

DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

ASPECT_RATIOS = {
    "1:1": (1, 1),
    "16:9": (16, 9),
    "9:16": (9, 16),
}

DEFAULT_ASPECT_SIZES = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
}


TOOLTIPS = {
    "task_type": (
        "What Bernini should do. t2v is normal text-to-video. t2i is a still image. "
        "v2v edits an existing video. i2i edits one image. r2v/rv2v use reference images."
    ),
    "guidance_mode": (
        "Bernini guidance recipe. Leave 'Use repo task default guidance' enabled unless you are testing. "
        "The default changes automatically per task, e.g. t2v_apg for text-to-video and v2v_apg for video edits."
    ),
    "auto_guidance": "Keeps guidance_mode matched to the selected task so normal users do not need to touch it.",
    "prompt": "Main instruction for the image/video/edit. For edit tasks, describe what should change, not just the scene.",
    "system_prompt": (
        "Advanced override for Bernini's internal system prompt. Leave empty for normal use; the repo picks a task-specific system prompt."
    ),
    "source_video": "Used by video edit tasks such as v2v, mv2v, rv2v and ads2v. You can select one or more videos when the task supports it.",
    "source_image": "Used by image edit tasks such as i2i. Not needed for plain text-to-video.",
    "reference_images": "Reference identity/object/style images for r2v, rv2v and ads2v tasks. Put each file on its own line or use the browse button.",
    "output_filename": "Optional custom output name. If empty, the helper creates a timestamped name in the Bernini output folder.",
    "resolution": "Choose a simple aspect ratio helper: 1:1, 16:9 or 9:16. Width and height stay editable beside it.",
    "keep_source_size": "When on, Bernini receives width=0 and height=0 so source-based tasks keep the source size.",
    "use_flash_attention": "Attention backend request. On asks Bernini for FlashAttention/FA2. Off asks for SDPA.",
    "width": "Output width in pixels. Use multiples of 16 or repo-tested presets for safer results. 0 means use source size for source-based tasks.",
    "height": "Output height in pixels. Use multiples of 16 or repo-tested presets for safer results. 0 means use source size for source-based tasks.",
    "frames": "Number of output frames. 1 creates an image-like result; 33/49 is a fast video test; 81+ is a longer run.",
    "fps": "Frames per second written to the video file. This changes playback speed/duration, not the number of generated frames.",
    "max_image_size": "For source/edit tasks this acts like the main max width / source resize control. Example: wide 480p is usually about 832 or 848, not 480.",
    "source_size_note": "For image/video edit tasks, Bernini mainly follows Keep source and Max source width. Width/height are used mostly for text-only tasks.",
    "steps": "Denoising steps. More steps can improve quality but take longer. 20 is a fast probe; 40 is a normal quality starting point.",
    "seed": "Same seed plus same settings should give similar output. When Random seed is on, a fresh seed is generated automatically each time you click Generate and the new number is shown here before the job starts.",
    "negative_prompt_override": "Enable only when you want to replace Bernini's default negative prompt. Leave off for normal repo defaults.",
    "negative_prompt": "Things Bernini should avoid. Only sent to the command when the override checkbox is enabled.",
    "omega_vid": (
        "Video/source guidance weight. Higher values make the result follow the source video/motion more strongly in video/edit tasks. "
        "Too high can make edits weaker or stiff."
    ),
    "omega_img": (
        "Image/reference guidance weight. Higher values make Bernini follow source or reference images more strongly. "
        "Useful for identity/object consistency, but too high can over-constrain the result."
    ),
    "omega_txt": (
        "Text prompt guidance weight. Higher values push the generation harder toward the written prompt. "
        "Too high can create artifacts or overcooked motion."
    ),
    "omega_tgt": (
        "Target/edited-result guidance weight used by some Bernini guidance modes. Leave at default unless testing specific edit behavior."
    ),
    "omega_scale": (
        "Overall APG guidance scaling. Think of it as a strength multiplier for the advanced guidance recipe. "
        "Small changes are safer than big jumps."
    ),
    "eta": "APG update/mixing strength. Usually keep default; changing it can affect stability and how aggressively guidance updates.",
    "momentum": "APG momentum term. Default 0 is safest. Positive/negative values can change how guidance accumulates over steps.",
    "planning_step": (
        "Step where Bernini starts or applies its planning-style guidance. Lower = earlier influence; higher = later influence. "
        "Leave default unless a specific task looks under/over-guided."
    ),
    "vit_txt_cfg": "Vision-language text CFG strength used in Bernini's VIT/planning controls. Usually leave at default.",
    "vit_img_cfg": "Vision-language image/reference CFG strength. Higher can make image references matter more, but can reduce freedom.",
    "vit_denoising_step": "How many early denoising/planning steps use the VIT guidance path. Leave default for normal use.",
    "flow_shift": "Scheduler/noise timing shift. Affects motion/detail timing. Default 5 is a safe starting point from the repo setup.",
    "norm_threshold": "Guidance norm clamp values. These cap extreme guidance updates to reduce instability/artifacts. Leave at default first.",
    "env_python": "Python executable inside the Bernini conda environment. Normally environments/.bernini_r/python.exe.",
    "repo_dir": "Folder containing the Bernini repo files, including infer_single_gpu.py.",
    "model_dir": "Local Hugging Face Diffusers model/config folder for Bernini-R 1.3B.",
    "output_dir": "Folder where generated Bernini videos/images are saved. Default is output/video/bernini.",
    "case_dir": "Temporary folder where this UI writes Bernini JSON case files before starting a job.",
    "use_unipc": "Use Bernini's UniPC scheduler flag. Default on matches the repo path used in tests.",
    "use_src_tgt_id": "Enable Bernini source/target identity rotary setting. Leave on unless a repo test says otherwise.",
    "interpolate_src_id": "Interpolate source identity IDs across frames. Leave on for smoother source/reference behavior.",
    "max_trained_src_id": "Maximum trained source ID used by the repo identity logic. Default 5 comes from the Bernini arguments.",
    "high_noise_ckpt": "Optional separate high-noise checkpoint. Leave empty when using the self-contained Diffusers folder.",
    "low_noise_ckpt": "Optional separate low-noise checkpoint. Leave empty when using the self-contained Diffusers folder.",
    "expandable_segments": "Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True. On Windows it may warn that it is unsupported, but it is harmless.",
    "cudnn_v8_fallback": "Use PyTorch legacy cuDNN convolution path. This can avoid Conv3D engine errors during high-resolution VAE decode.",
    "cuda_visible_devices": "GPU index to expose to Bernini. For a normal single-GPU setup this can stay 0.",
    "extra_env": "Optional environment variables for experiments, one per line as NAME=value.",
    "use_pe": "Use Bernini's task-aware prompt enhancer through local Qwen3-VL.",
    "pe_model": "Optional prompt enhancer model name when --use_pe is enabled.",
    "command_preview": "Exact command that will be run. Useful for debugging or copying to a console.",
    "logs": "Live process output and errors from the current helper session.",
    "random_seed_toggle": "When on, clicking Generate first creates a new random seed and shows it in the seed box. That new number is the seed used for the run.",
    "use_framevision_queue": "When enabled, Bernini jobs are added to FrameVision's queue instead of running directly in this panel. The Queue tab is not opened automatically.",
    "show_adv_guidance": "Show or hide the advanced guidance tab. Leave off for a simpler UI unless you are actively testing Bernini guidance internals.",
}

GUIDANCE_HELP_TEXT = (
    "Advanced controls. For normal generation, keep these defaults and only change prompt, task, size, frames, steps and seed. "
    "These values are Bernini guidance/scheduler knobs from the repo; small changes are safer than big jumps."
)

FRAMEVISION_ATTENTION_PY = '# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate\n#\n# Licensed under the Apache License, Version 2.0 (the "License");\n# This FrameVision-local patch only changes backend selection/logging.\n# It keeps the original Bernini varlen attention contract intact.\n#\n# FRAMEVISION_ATTENTION_BACKEND_PATCH_V1\n\n"""Variable-length attention with explicit backend selection/logging.\n\nEnvironment:\n  BERNINI_ATTENTION_BACKEND=auto|fa2|sdpa|sage|fa3\n\nBackend order in auto:\n  1. FlashAttention-3 if available\n  2. FlashAttention-2 if available\n  3. PyTorch SDPA\n\nSage is only used when explicitly requested with BERNINI_ATTENTION_BACKEND=sage.\n"""\n\nimport os\nimport torch\nimport torch.nn.functional as F\n\n_BACKEND = None\n_flash_varlen = None\n_sage_varlen = None\n_LOGGED = False\n\n\ndef _print_once(message: str) -> None:\n    global _LOGGED\n    if not _LOGGED:\n        print(message, flush=True)\n        _LOGGED = True\n\n\ndef _request_backend() -> str:\n    return (os.environ.get("BERNINI_ATTENTION_BACKEND") or "auto").strip().lower()\n\n\ndef _try_fa3():\n    try:\n        from flash_attn_interface import flash_attn_varlen_func\n        return flash_attn_varlen_func\n    except Exception:\n        return None\n\n\ndef _try_fa2():\n    try:\n        from flash_attn import flash_attn_varlen_func\n        return flash_attn_varlen_func\n    except Exception:\n        return None\n\n\ndef _try_sage():\n    try:\n        from sageattention import sageattn_varlen\n        return sageattn_varlen\n    except Exception:\n        return None\n\n\ndef _select_backend():\n    global _BACKEND, _flash_varlen, _sage_varlen\n    if _BACKEND is not None:\n        return\n\n    requested = _request_backend()\n\n    if requested == "sdpa":\n        _BACKEND = "sdpa"\n        _print_once("[Bernini attention] requested=sdpa selected=sdpa")\n        return\n\n    if requested == "sage":\n        _sage_varlen = _try_sage()\n        if _sage_varlen is not None:\n            _BACKEND = "sage"\n            _print_once("[Bernini attention] requested=sage selected=sage")\n            return\n        _print_once("[Bernini attention] requested=sage selected=fallback_pending reason=sage_import_failed")\n\n    if requested == "fa3":\n        _flash_varlen = _try_fa3()\n        if _flash_varlen is not None:\n            _BACKEND = "fa3"\n            _print_once("[Bernini attention] requested=fa3 selected=fa3")\n            return\n        _print_once("[Bernini attention] requested=fa3 selected=fallback_pending reason=fa3_import_failed")\n\n    if requested == "fa2":\n        _flash_varlen = _try_fa2()\n        if _flash_varlen is not None:\n            _BACKEND = "fa2"\n            _print_once("[Bernini attention] requested=fa2 selected=fa2")\n            return\n        _print_once("[Bernini attention] requested=fa2 selected=fallback_pending reason=fa2_import_failed")\n\n    _flash_varlen = _try_fa3()\n    if _flash_varlen is not None:\n        _BACKEND = "fa3"\n        _print_once(f"[Bernini attention] requested={requested} selected=fa3")\n        return\n\n    _flash_varlen = _try_fa2()\n    if _flash_varlen is not None:\n        _BACKEND = "fa2"\n        _print_once(f"[Bernini attention] requested={requested} selected=fa2")\n        return\n\n    _BACKEND = "sdpa"\n    _print_once(f"[Bernini attention] requested={requested} selected=sdpa")\n\n\ndef get_attention_backend() -> str:\n    _select_backend()\n    return _BACKEND\n\n\ndef _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal):\n    """Varlen attention via SDPA: run each sample segment, then concatenate."""\n    cq = cu_seqlens_q.tolist()\n    ck = cu_seqlens_k.tolist()\n    outs = []\n    for i in range(len(cq) - 1):\n        qi = q[cq[i] : cq[i + 1]].transpose(0, 1).unsqueeze(0)\n        ki = k[ck[i] : ck[i + 1]].transpose(0, 1).unsqueeze(0)\n        vi = v[ck[i] : ck[i + 1]].transpose(0, 1).unsqueeze(0)\n        oi = F.scaled_dot_product_attention(qi, ki, vi, is_causal=causal)\n        outs.append(oi.squeeze(0).transpose(0, 1))\n    return torch.cat(outs, dim=0)\n\n\ndef varlen_attention(\n    q,\n    k,\n    v,\n    cu_seqlens_q,\n    cu_seqlens_k,\n    max_seqlen_q,\n    max_seqlen_k,\n    causal: bool = False,\n):\n    """Variable-length attention. Returns [total_q_tokens, num_heads, head_dim]."""\n    _select_backend()\n\n    if _BACKEND == "sage":\n        try:\n            return _sage_varlen(\n                q,\n                k,\n                v,\n                cu_seqlens_q=cu_seqlens_q,\n                cu_seqlens_k=cu_seqlens_k,\n                max_seqlen_q=int(max_seqlen_q),\n                max_seqlen_k=int(max_seqlen_k),\n                is_causal=causal,\n            )\n        except TypeError:\n            return _sage_varlen(\n                q,\n                k,\n                v,\n                cu_seqlens_q,\n                cu_seqlens_k,\n                int(max_seqlen_q),\n                int(max_seqlen_k),\n                is_causal=causal,\n            )\n\n    if _BACKEND == "fa3":\n        out = _flash_varlen(\n            q,\n            k,\n            v,\n            cu_seqlens_q=cu_seqlens_q,\n            cu_seqlens_k=cu_seqlens_k,\n            max_seqlen_q=int(max_seqlen_q),\n            max_seqlen_k=int(max_seqlen_k),\n            causal=causal,\n        )\n        return out[0] if isinstance(out, tuple) else out\n\n    if _BACKEND == "fa2":\n        return _flash_varlen(\n            q,\n            k,\n            v,\n            cu_seqlens_q,\n            cu_seqlens_k,\n            int(max_seqlen_q),\n            int(max_seqlen_k),\n            causal=causal,\n        )\n\n    return _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)\n'


def _guess_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent, Path.cwd(), here.parent]:
        if (candidate / "presets").exists() or (candidate / "models").exists() or (candidate / "helpers").exists():
            return candidate.resolve()
    return Path.cwd().resolve()


def _split_paths(text: str) -> List[str]:
    raw = text.replace("\r", "\n").replace(";", "\n").split("\n")
    return [x.strip().strip('"') for x in raw if x.strip().strip('"')]


def _quote_command(parts: List[str]) -> str:
    return subprocess.list2cmdline([str(p) for p in parts])


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):  # noqa: N802 - Qt override
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):  # noqa: N802 - Qt override
        event.ignore()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):  # noqa: N802 - Qt override
        event.ignore()


class PathRow(QWidget):
    def __init__(self, mode: str = "file", caption: str = "Select", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.mode = mode
        self.caption = caption
        self.edit = QLineEdit()
        self.edit.setClearButtonEnabled(True)
        self.button = QToolButton()
        self.button.setText("…")
        self.button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:  # noqa: N802 - Qt style compatibility
        self.edit.setText(value or "")

    def _browse(self) -> None:
        start = self.text() or str(_guess_root())
        if self.mode == "dir":
            value = QFileDialog.getExistingDirectory(self, self.caption, start)
        elif self.mode == "save":
            value, _ = QFileDialog.getSaveFileName(self, self.caption, start)
        elif self.mode == "multi":
            values, _ = QFileDialog.getOpenFileNames(self, self.caption, start)
            value = ";".join(values)
        else:
            value, _ = QFileDialog.getOpenFileName(self, self.caption, start)
        if value:
            self.setText(value)


class BerniniSmallWidget(QWidget):
    """Standalone FrameVision UI helper for Bernini-R 1.3B.

    Import options used by FrameVision can call either:
        widget = BerniniSmallWidget(parent)
    or:
        widget = create_widget(parent)
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.root = _guess_root()
        self.settings_file = self.root / "presets" / "setsave" / "bernini_small_settings.json"
        self.legacy_settings_file = self.root / "presets" / "bernini_small_settings.json"
        self.temp_dir = self.root / "temp" / "bernini_small"
        self.log_dir = self.root / "logs"
        self.latest_output: Optional[Path] = None
        self.process: Optional[QProcess] = None
        self._last_case_file: Optional[Path] = None
        self._busy = False
        self._loading_settings = True
        self._saw_saved_output = False
        self._auto_closing_process = False
        self._done_watchdog = QTimer(self)
        self._done_watchdog.setInterval(2500)
        self._done_watchdog.timeout.connect(self._check_output_done)

        self.setWindowTitle("Bernini-R 1.3B")
        self.resize(1100, 820)

        self._build_ui()
        self._connect_ui()
        self._load_settings()
        self._update_random_seed_toggle_text(self.random_seed_btn.isChecked())
        self._update_guidance_tab_visibility(self.show_adv_guidance_cb.isChecked())
        self._apply_task_defaults(force=True)
        self._sync_queue_button_text()
        self._refresh_command_preview()
        self._loading_settings = False

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        title = QLabel("Bernini-R 1.3B")
        title.setStyleSheet("font-weight: 700; font-size: 16px;")
        root_layout.addWidget(title)

        # Tabs are outside the scroll areas, so they stay fixed while a tab's content scrolls.
        self.tabs = QTabWidget()
        self.generate_tab = self._make_generation_tab()
        self.settings_tab = self._make_settings_tab()
        self.guidance_tab = self._make_guidance_tab()
        self.generate_tab_index = self.tabs.addTab(self.generate_tab, "Generate")
        self.settings_tab_index = self.tabs.addTab(self.settings_tab, "Settings")
        self.guidance_tab_index = self.tabs.addTab(self.guidance_tab, "Adv. guidance settings")
        root_layout.addWidget(self.tabs, 1)

        # Fixed action bar: never scrolls away.
        action_bar = QFrame()
        action_bar.setFrameShape(QFrame.StyledPanel)
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(8, 8, 8, 8)
        self.status_label = QLabel("Ready")
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setToolTip("Start the Bernini job with the current settings. This button stays fixed so it is always reachable.")
        self.generate_btn.setMinimumHeight(34)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop the currently running Bernini process.")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setEnabled(False)
        self.open_output_btn = QPushButton("Open Output")
        self.open_output_btn.setToolTip("Open the latest generated file if it exists.")
        self.open_folder_btn = QPushButton("View results")
        self.open_folder_btn.setToolTip("Open the configured Bernini output folder in FrameVision Media Explorer.")
        action_layout.addWidget(self.generate_btn)
        action_layout.addWidget(self.stop_btn)
        action_layout.addWidget(self.open_output_btn)
        action_layout.addWidget(self.open_folder_btn)
        action_layout.addWidget(self.status_label, 1)
        root_layout.addWidget(action_bar)

    def _scroll_page(self, inner: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(inner)
        return scroll

    def _tip_label(self, text: str, tip_key: str) -> QLabel:
        label = QLabel(text)
        label.setToolTip(TOOLTIPS.get(tip_key, ""))
        return label

    def _set_tip(self, widget: QWidget, tip_key: str) -> QWidget:
        widget.setToolTip(TOOLTIPS.get(tip_key, ""))
        return widget

    def _make_generation_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        task_group = QGroupBox("Task")
        task_form = QFormLayout(task_group)
        self.task_combo = self._set_tip(NoWheelComboBox(), "task_type")
        for key in TASK_TYPE_CHOICES:
            self.task_combo.addItem(f"{key} — {TASK_LABELS.get(key, key)}", key)
        self.guidance_combo = self._set_tip(NoWheelComboBox(), "guidance_mode")
        self.guidance_combo.addItems(GUIDANCE_MODES)
        self.guidance_combo.hide()
        self.auto_guidance_cb = self._set_tip(QCheckBox("Use repo task default guidance"), "auto_guidance")
        self.auto_guidance_cb.setChecked(True)
        self.auto_guidance_cb.hide()
        task_form.addRow(self._tip_label("Task type", "task_type"), self.task_combo)
        layout.addWidget(task_group)

        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)
        self.prompt_edit = self._set_tip(QPlainTextEdit(), "prompt")
        self.prompt_edit.setPlaceholderText("Describe the video/image or the edit instruction.")
        self.prompt_edit.setMinimumHeight(120)
        self.system_prompt_edit = self._set_tip(QPlainTextEdit(), "system_prompt")
        self.system_prompt_edit.setPlaceholderText("Optional. Leave empty to let Bernini choose the system prompt for the task type.")
        self.system_prompt_edit.setMinimumHeight(62)

        prompt_header = QWidget()
        prompt_header_layout = QHBoxLayout(prompt_header)
        prompt_header_layout.setContentsMargins(0, 0, 0, 0)
        prompt_header_layout.setSpacing(10)
        prompt_header_layout.addWidget(self._tip_label("Main prompt / edit instruction", "prompt"))
        prompt_header_layout.addStretch(1)
        self.use_prompt_enhancer_cb = self._set_tip(QCheckBox("Use Bernini prompt enhancer (local Qwen3-VL)"), "use_pe")
        self.use_prompt_enhancer_cb.setChecked(False)
        prompt_header_layout.addWidget(self.use_prompt_enhancer_cb, 0)

        prompt_layout.addWidget(prompt_header)
        prompt_layout.addWidget(self.prompt_edit)
        layout.addWidget(prompt_group)

        media_group = QGroupBox("Inputs")
        media_form = QFormLayout(media_group)
        self.video_row = self._set_tip(PathRow("multi", "Select source video(s)"), "source_video")
        self.video_row.edit.setToolTip(TOOLTIPS["source_video"])
        self.video_row.button.setToolTip(TOOLTIPS["source_video"])
        self.image_row = self._set_tip(PathRow("file", "Select source image"), "source_image")
        self.image_row.edit.setToolTip(TOOLTIPS["source_image"])
        self.image_row.button.setToolTip(TOOLTIPS["source_image"])
        self.images_row = self._set_tip(PathRow("multi", "Select reference image(s)"), "reference_images")
        self.images_row.edit.setToolTip(TOOLTIPS["reference_images"])
        self.images_row.button.setToolTip(TOOLTIPS["reference_images"])
        self.output_name = self._set_tip(QLineEdit(), "output_filename")
        self.output_name.setPlaceholderText("Optional filename. Extension is added if missing.")
        media_form.addRow(self._tip_label("Source video(s)", "source_video"), self.video_row)
        media_form.addRow(self._tip_label("Source image", "source_image"), self.image_row)
        media_form.addRow(self._tip_label("Reference image(s)", "reference_images"), self.images_row)
        layout.addWidget(media_group)

        size_group = QGroupBox("Size / duration")
        size_grid = QGridLayout(size_group)
        self.resolution_combo = self._set_tip(NoWheelComboBox(), "resolution")
        for name in ASPECT_RATIOS:
            self.resolution_combo.addItem(name)
        self.width_spin = self._spin(0, 4096, 1280, step=16)
        self.width_spin.setToolTip(TOOLTIPS["width"])
        self.height_spin = self._spin(0, 4096, 720, step=16)
        self.height_spin.setToolTip(TOOLTIPS["height"])
        self.keep_source_size_cb = self._set_tip(QCheckBox("Keep source (0×0)"), "keep_source_size")
        self.keep_source_size_cb.setChecked(False)
        self.num_frames_spin = self._spin(1, 481, 81, step=8)
        self.num_frames_spin.setToolTip(TOOLTIPS["frames"])
        self.fps_spin = self._spin(1, 60, 16)
        self.fps_spin.setToolTip(TOOLTIPS["fps"])
        self.max_image_size_spin = self._spin(64, 4096, 848, step=16)
        self.max_image_size_spin.setToolTip(TOOLTIPS["max_image_size"])
        self.steps_spin = self._spin(1, 200, 40)
        self.steps_spin.setToolTip(TOOLTIPS["steps"])
        self.seed_spin = self._spin(0, 2_147_483_647, 42)
        self.seed_spin.setToolTip(TOOLTIPS["seed"])
        self.random_seed_btn = self._set_tip(QCheckBox("Random seed"), "random_seed_toggle")
        self.random_seed_btn.setChecked(True)

        self.source_size_note = QLabel(TOOLTIPS["source_size_note"])
        self.source_size_note.setWordWrap(True)
        self.source_size_note.setStyleSheet("color: #666;")
        self.source_size_note.setToolTip(TOOLTIPS["source_size_note"])

        size_grid.addWidget(self._tip_label("Aspect / size", "resolution"), 0, 0)
        aspect_row = QWidget()
        aspect_layout = QHBoxLayout(aspect_row)
        aspect_layout.setContentsMargins(0, 0, 0, 0)
        aspect_layout.addWidget(self.resolution_combo, 0)
        aspect_layout.addWidget(self._tip_label("W", "width"), 0)
        aspect_layout.addWidget(self.width_spin, 1)
        aspect_layout.addWidget(self._tip_label("H", "height"), 0)
        aspect_layout.addWidget(self.height_spin, 1)
        aspect_layout.addWidget(self.keep_source_size_cb, 0)
        size_grid.addWidget(aspect_row, 0, 1, 1, 3)

        size_grid.addWidget(self._tip_label("Frames", "frames"), 1, 0)
        size_grid.addWidget(self.num_frames_spin, 1, 1)
        size_grid.addWidget(self._tip_label("FPS", "fps"), 1, 2)
        size_grid.addWidget(self.fps_spin, 1, 3)
        size_grid.addWidget(self._tip_label("Maximum width", "max_image_size"), 2, 0)
        size_grid.addWidget(self.max_image_size_spin, 2, 1)
        size_grid.addWidget(self._tip_label("Steps", "steps"), 2, 2)
        size_grid.addWidget(self.steps_spin, 2, 3)
        size_grid.addWidget(self._tip_label("Seed", "seed"), 3, 0)
        seed_row = QWidget()
        seed_layout = QHBoxLayout(seed_row)
        seed_layout.setContentsMargins(0, 0, 0, 0)
        seed_layout.addWidget(self.seed_spin, 1)
        seed_layout.addWidget(self.random_seed_btn)
        size_grid.addWidget(seed_row, 3, 1, 1, 3)
        size_grid.addWidget(self.source_size_note, 4, 0, 1, 4)
        layout.addWidget(size_group)

        neg_group = QGroupBox("Negative prompt")
        neg_layout = QVBoxLayout(neg_group)
        self.override_neg_cb = self._set_tip(QCheckBox("Override Bernini default negative prompt"), "negative_prompt_override")
        self.neg_prompt_edit = self._set_tip(QPlainTextEdit(), "negative_prompt")
        self.neg_prompt_edit.setPlainText(DEFAULT_NEG_PROMPT)
        self.neg_prompt_edit.setMinimumHeight(82)
        self.neg_prompt_edit.setEnabled(False)
        neg_layout.addWidget(self.override_neg_cb)
        neg_layout.addWidget(self.neg_prompt_edit)
        layout.addWidget(neg_group)

        layout.addStretch(1)
        outer.addWidget(self._scroll_page(inner))
        return page

    def _make_guidance_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        guide_note = QLabel(GUIDANCE_HELP_TEXT)
        guide_note.setWordWrap(True)
        guide_note.setStyleSheet("color: #666; padding: 4px 2px;")
        guide_note.setToolTip(GUIDANCE_HELP_TEXT)
        layout.addWidget(guide_note)

        guide_group = QGroupBox("Guidance weights")
        guide_group.setToolTip("Weights that decide how strongly Bernini follows video/source, image/reference, text prompt and target guidance.")
        form = QFormLayout(guide_group)
        self.omega_vid_spin = self._dspin(-20.0, 20.0, 1.25, decimals=3, step=0.05)
        self.omega_vid_spin.setToolTip(TOOLTIPS["omega_vid"])
        self.omega_img_spin = self._dspin(-20.0, 20.0, 4.5, decimals=3, step=0.05)
        self.omega_img_spin.setToolTip(TOOLTIPS["omega_img"])
        self.omega_txt_spin = self._dspin(-20.0, 20.0, 4.0, decimals=3, step=0.05)
        self.omega_txt_spin.setToolTip(TOOLTIPS["omega_txt"])
        self.omega_tgt_spin = self._dspin(-20.0, 20.0, 0.5, decimals=3, step=0.05)
        self.omega_tgt_spin.setToolTip(TOOLTIPS["omega_tgt"])
        self.omega_scale_spin = self._dspin(-20.0, 20.0, 0.8, decimals=3, step=0.05)
        self.omega_scale_spin.setToolTip(TOOLTIPS["omega_scale"])
        self.eta_spin = self._dspin(0.0, 10.0, 0.5, decimals=3, step=0.05)
        self.eta_spin.setToolTip(TOOLTIPS["eta"])
        self.momentum_spin = self._dspin(-10.0, 10.0, 0.0, decimals=3, step=0.05)
        self.momentum_spin.setToolTip(TOOLTIPS["momentum"])
        form.addRow(self._tip_label("omega_vid", "omega_vid"), self.omega_vid_spin)
        form.addRow(self._tip_label("omega_img", "omega_img"), self.omega_img_spin)
        form.addRow(self._tip_label("omega_txt", "omega_txt"), self.omega_txt_spin)
        form.addRow(self._tip_label("omega_tgt", "omega_tgt"), self.omega_tgt_spin)
        form.addRow(self._tip_label("omega_scale", "omega_scale"), self.omega_scale_spin)
        form.addRow(self._tip_label("eta", "eta"), self.eta_spin)
        form.addRow(self._tip_label("momentum", "momentum"), self.momentum_spin)
        layout.addWidget(guide_group)

        planning_group = QGroupBox("Planning / VIT controls")
        pform = QFormLayout(planning_group)
        self.planning_step_spin = self._spin(0, 200, 25)
        self.planning_step_spin.setToolTip(TOOLTIPS["planning_step"])
        self.vit_txt_cfg_spin = self._dspin(0.0, 20.0, 1.2, decimals=3, step=0.05)
        self.vit_txt_cfg_spin.setToolTip(TOOLTIPS["vit_txt_cfg"])
        self.vit_img_cfg_spin = self._dspin(0.0, 20.0, 1.0, decimals=3, step=0.05)
        self.vit_img_cfg_spin.setToolTip(TOOLTIPS["vit_img_cfg"])
        self.vit_denoising_step_spin = self._spin(0, 200, 5)
        self.vit_denoising_step_spin.setToolTip(TOOLTIPS["vit_denoising_step"])
        self.flow_shift_spin = self._dspin(-20.0, 20.0, 5.0, decimals=3, step=0.1)
        self.flow_shift_spin.setToolTip(TOOLTIPS["flow_shift"])
        pform.addRow(self._tip_label("planning_step", "planning_step"), self.planning_step_spin)
        pform.addRow(self._tip_label("vit_txt_cfg", "vit_txt_cfg"), self.vit_txt_cfg_spin)
        pform.addRow(self._tip_label("vit_img_cfg", "vit_img_cfg"), self.vit_img_cfg_spin)
        pform.addRow(self._tip_label("vit_denoising_step", "vit_denoising_step"), self.vit_denoising_step_spin)
        pform.addRow(self._tip_label("flow_shift", "flow_shift"), self.flow_shift_spin)
        layout.addWidget(planning_group)

        norm_group = QGroupBox("Norm thresholds")
        nform = QFormLayout(norm_group)
        self.norm1_spin = self._dspin(0.0, 500.0, 50.0, decimals=2, step=1.0)
        self.norm1_spin.setToolTip(TOOLTIPS["norm_threshold"])
        self.norm2_spin = self._dspin(0.0, 500.0, 50.0, decimals=2, step=1.0)
        self.norm2_spin.setToolTip(TOOLTIPS["norm_threshold"])
        self.norm3_spin = self._dspin(0.0, 500.0, 50.0, decimals=2, step=1.0)
        self.norm3_spin.setToolTip(TOOLTIPS["norm_threshold"])
        nform.addRow(self._tip_label("norm_threshold 1", "norm_threshold"), self.norm1_spin)
        nform.addRow(self._tip_label("norm_threshold 2", "norm_threshold"), self.norm2_spin)
        nform.addRow(self._tip_label("norm_threshold 3", "norm_threshold"), self.norm3_spin)
        layout.addWidget(norm_group)

        layout.addStretch(1)
        outer.addWidget(self._scroll_page(inner))
        return page

    def _make_settings_tab(self) -> QWidget:
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        top_options = QWidget()
        top_options.setObjectName("bernini_top_options")
        top_form = QFormLayout(top_options)
        top_form.setContentsMargins(0, 0, 0, 0)
        self.show_adv_guidance_cb = self._set_tip(QCheckBox(""), "show_adv_guidance")
        self.show_adv_guidance_cb.setChecked(False)
        self.use_flash_attention_cb = self._set_tip(QCheckBox(""), "use_flash_attention")
        self.use_flash_attention_cb.setChecked(True)
        self.use_framevision_queue_cb = self._set_tip(QCheckBox(""), "use_framevision_queue")
        self.use_framevision_queue_cb.setChecked(False)
        top_form.addRow(self._tip_label("Advanced guidance tab", "show_adv_guidance"), self.show_adv_guidance_cb)
        top_form.addRow(self._tip_label("FlashAttention", "use_flash_attention"), self.use_flash_attention_cb)
        top_form.addRow(self._tip_label("Queue", "use_framevision_queue"), self.use_framevision_queue_cb)
        layout.addWidget(top_options)

        prompt_settings_group = QGroupBox("Prompt / hidden defaults")
        psform = QFormLayout(prompt_settings_group)
        psform.addRow(self._tip_label("System prompt override", "system_prompt"), self.system_prompt_edit)
        layout.addWidget(prompt_settings_group)

        paths_group = QGroupBox("Locations")
        pform = QFormLayout(paths_group)
        self.env_python_row = self._set_tip(PathRow("file", "Select Bernini Python"), "env_python")
        self.env_python_row.edit.setToolTip(TOOLTIPS["env_python"])
        self.env_python_row.button.setToolTip(TOOLTIPS["env_python"])
        self.repo_dir_row = self._set_tip(PathRow("dir", "Select Bernini repo folder"), "repo_dir")
        self.repo_dir_row.edit.setToolTip(TOOLTIPS["repo_dir"])
        self.repo_dir_row.button.setToolTip(TOOLTIPS["repo_dir"])
        self.model_dir_row = self._set_tip(PathRow("dir", "Select Bernini-R 1.3B model folder"), "model_dir")
        self.model_dir_row.edit.setToolTip(TOOLTIPS["model_dir"])
        self.model_dir_row.button.setToolTip(TOOLTIPS["model_dir"])
        self.output_dir_row = self._set_tip(PathRow("dir", "Select output folder"), "output_dir")
        self.output_dir_row.edit.setToolTip(TOOLTIPS["output_dir"])
        self.output_dir_row.button.setToolTip(TOOLTIPS["output_dir"])
        self.case_dir_row = self._set_tip(PathRow("dir", "Select temporary case folder"), "case_dir")
        self.case_dir_row.edit.setToolTip(TOOLTIPS["case_dir"])
        self.case_dir_row.button.setToolTip(TOOLTIPS["case_dir"])
        pform.addRow(self._tip_label("Python", "env_python"), self.env_python_row)
        pform.addRow(self._tip_label("Repo folder", "repo_dir"), self.repo_dir_row)
        pform.addRow(self._tip_label("Model/config folder", "model_dir"), self.model_dir_row)
        pform.addRow(self._tip_label("Output folder", "output_dir"), self.output_dir_row)
        pform.addRow(self._tip_label("Output filename", "output_filename"), self.output_name)
        pform.addRow(self._tip_label("Case/temp folder", "case_dir"), self.case_dir_row)
        layout.addWidget(paths_group)

        model_group = QGroupBox("Model flags from Bernini repo")
        mform = QFormLayout(model_group)
        self.use_unipc_cb = self._set_tip(QCheckBox("--use_unipc"), "use_unipc")
        self.use_unipc_cb.setChecked(True)
        self.use_src_tgt_id_cb = self._set_tip(QCheckBox("--use_src_tgt_id"), "use_src_tgt_id")
        self.use_src_tgt_id_cb.setChecked(True)
        self.interpolate_src_id_cb = self._set_tip(QCheckBox("--interpolate_src_id"), "interpolate_src_id")
        self.interpolate_src_id_cb.setChecked(True)
        self.max_trained_src_id_spin = self._spin(0, 64, 5)
        self.max_trained_src_id_spin.setToolTip(TOOLTIPS["max_trained_src_id"])
        self.high_noise_row = self._set_tip(PathRow("file", "Select high-noise checkpoint"), "high_noise_ckpt")
        self.high_noise_row.edit.setToolTip(TOOLTIPS["high_noise_ckpt"])
        self.high_noise_row.button.setToolTip(TOOLTIPS["high_noise_ckpt"])
        self.low_noise_row = self._set_tip(PathRow("file", "Select low-noise checkpoint"), "low_noise_ckpt")
        self.low_noise_row.edit.setToolTip(TOOLTIPS["low_noise_ckpt"])
        self.low_noise_row.button.setToolTip(TOOLTIPS["low_noise_ckpt"])
        mform.addRow(self._tip_label("UniPC scheduler", "use_unipc"), self.use_unipc_cb)
        mform.addRow(self._tip_label("Source/target ID rotary", "use_src_tgt_id"), self.use_src_tgt_id_cb)
        mform.addRow(self._tip_label("Interpolate source IDs", "interpolate_src_id"), self.interpolate_src_id_cb)
        mform.addRow(self._tip_label("max_trained_src_id", "max_trained_src_id"), self.max_trained_src_id_spin)
        mform.addRow(self._tip_label("high_noise_ckpt", "high_noise_ckpt"), self.high_noise_row)
        mform.addRow(self._tip_label("low_noise_ckpt", "low_noise_ckpt"), self.low_noise_row)
        layout.addWidget(model_group)

        runtime_group = QGroupBox("Runtime / one-time settings")
        rform = QFormLayout(runtime_group)
        self.expandable_segments_cb = self._set_tip(QCheckBox("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"), "expandable_segments")
        self.expandable_segments_cb.setChecked(True)
        self.cudnn_v8_fallback_cb = self._set_tip(QCheckBox("cuDNN v8 fallback"), "cudnn_v8_fallback")
        self.cudnn_v8_fallback_cb.setChecked(True)
        self.cuda_visible_devices_edit = self._set_tip(QLineEdit("0"), "cuda_visible_devices")
        self.extra_env_edit = self._set_tip(QPlainTextEdit(), "extra_env")
        self.extra_env_edit.setPlaceholderText("Optional extra env vars, one per line: NAME=value")
        self.extra_env_edit.setMinimumHeight(70)
        self.pe_model_edit = self._set_tip(QLineEdit(), "pe_model")
        self.pe_model_edit.setPlaceholderText("Legacy/API model name, optional. Local Qwen backend does not need this.")
        offload_label = QLabel("")
        offload_label.setWordWrap(True)
        rform.addRow(self._tip_label("CUDA allocator", "expandable_segments"), self.expandable_segments_cb)
        rform.addRow(self._tip_label("cuDNN", "cudnn_v8_fallback"), self.cudnn_v8_fallback_cb)
        rform.addRow(self._tip_label("CUDA_VISIBLE_DEVICES", "cuda_visible_devices"), self.cuda_visible_devices_edit)
        rform.addRow(self._tip_label("Extra env", "extra_env"), self.extra_env_edit)
        rform.addRow(self._tip_label("pe_model", "pe_model"), self.pe_model_edit)
        rform.addRow("Offload notes", offload_label)
        layout.addWidget(runtime_group)

        command_group = QGroupBox("Command preview")
        c_layout = QVBoxLayout(command_group)
        self.command_preview = self._set_tip(QPlainTextEdit(), "command_preview")
        self.command_preview.setReadOnly(True)
        self.command_preview.setMinimumHeight(96)
        self.copy_command_btn = QPushButton("Copy Command")
        self.copy_command_btn.setToolTip("Copy the exact command preview to the clipboard for console testing.")
        c_layout.addWidget(self.command_preview)
        c_layout.addWidget(self.copy_command_btn, 0, Qt.AlignRight)
        layout.addWidget(command_group)

        logs_group = QGroupBox("Logs")
        l_layout = QVBoxLayout(logs_group)
        self.log_view = self._set_tip(QPlainTextEdit(), "logs")
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(260)
        self.clear_log_btn = QPushButton("Clear logs")
        l_layout.addWidget(self.log_view)
        l_layout.addWidget(self.clear_log_btn, 0, Qt.AlignRight)
        layout.addWidget(logs_group)

        layout.addStretch(1)
        outer.addWidget(self._scroll_page(inner))
        return page

    def _spin(self, minimum: int, maximum: int, value: int, step: int = 1) -> NoWheelSpinBox:
        spin = NoWheelSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setKeyboardTracking(False)
        return spin

    def _dspin(self, minimum: float, maximum: float, value: float, decimals: int = 2, step: float = 0.1) -> NoWheelDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setKeyboardTracking(False)
        return spin

    def _connect_ui(self) -> None:
        self.task_combo.currentIndexChanged.connect(lambda *_: self._apply_task_defaults())
        self.resolution_combo.currentTextChanged.connect(self._apply_resolution_preset)
        self.keep_source_size_cb.toggled.connect(self._toggle_keep_source_size)
        self.width_spin.valueChanged.connect(self._width_changed)
        self.height_spin.valueChanged.connect(self._height_changed)
        self.override_neg_cb.toggled.connect(self.neg_prompt_edit.setEnabled)
        self.random_seed_btn.toggled.connect(self._update_random_seed_toggle_text)
        self.show_adv_guidance_cb.toggled.connect(self._update_guidance_tab_visibility)
        self.generate_btn.clicked.connect(self.generate)
        self.use_framevision_queue_cb.toggled.connect(self._sync_queue_button_text)
        self.use_framevision_queue_cb.toggled.connect(self._save_settings_safe)
        self.stop_btn.clicked.connect(self.stop)
        self.open_folder_btn.clicked.connect(self.view_results)
        self.open_output_btn.clicked.connect(self.open_latest_output)
        self.copy_command_btn.clicked.connect(self.copy_command)
        self.clear_log_btn.clicked.connect(self.log_view.clear)

        widgets = [
            self.task_combo,
            self.prompt_edit,
            self.system_prompt_edit,
            self.video_row.edit,
            self.image_row.edit,
            self.images_row.edit,
            self.output_name,
            self.resolution_combo,
            self.keep_source_size_cb,
            self.width_spin,
            self.height_spin,
            self.num_frames_spin,
            self.fps_spin,
            self.max_image_size_spin,
            self.steps_spin,
            self.seed_spin,
            self.override_neg_cb,
            self.neg_prompt_edit,
            self.omega_vid_spin,
            self.omega_img_spin,
            self.omega_txt_spin,
            self.omega_tgt_spin,
            self.omega_scale_spin,
            self.eta_spin,
            self.momentum_spin,
            self.planning_step_spin,
            self.vit_txt_cfg_spin,
            self.vit_img_cfg_spin,
            self.vit_denoising_step_spin,
            self.flow_shift_spin,
            self.norm1_spin,
            self.norm2_spin,
            self.norm3_spin,
            self.show_adv_guidance_cb,
            self.env_python_row.edit,
            self.repo_dir_row.edit,
            self.model_dir_row.edit,
            self.output_dir_row.edit,
            self.case_dir_row.edit,
            self.use_unipc_cb,
            self.use_src_tgt_id_cb,
            self.interpolate_src_id_cb,
            self.max_trained_src_id_spin,
            self.high_noise_row.edit,
            self.low_noise_row.edit,
            self.expandable_segments_cb,
            self.cudnn_v8_fallback_cb,
            self.cuda_visible_devices_edit,
            self.use_flash_attention_cb,
            self.use_framevision_queue_cb,
            self.extra_env_edit,
            self.use_prompt_enhancer_cb,
            self.pe_model_edit,
        ]
        for widget in widgets:
            if isinstance(widget, (QLineEdit, QPlainTextEdit)):
                widget.textChanged.connect(self._refresh_command_preview)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._refresh_command_preview)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self._refresh_command_preview)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self._refresh_command_preview)

    # ------------------------------------------------------------------ settings
    def _defaults(self) -> Dict[str, object]:
        return {
            "env_python": str(self.root / "environments" / ".bernini_r" / "python.exe"),
            "repo_dir": str(self.root / "models" / "bernini_r_1p3b" / "Bernini"),
            "model_dir": str(self.root / "models" / "bernini_r_1p3b" / "Bernini-R-1.3B-Diffusers"),
            "output_dir": str(self.root / "output" / "video" / "bernini"),
            "case_dir": str(self.root / "temp" / "bernini_small"),
            "prompt": "",
            "system_prompt": "",
            "task": "t2v",
            "guidance": "t2v_apg",
            "auto_guidance": True,
            "aspect_ratio": "16:9",
            "keep_source_size": False,
            "width": 1280,
            "height": 720,
            "num_frames": 81,
            "fps": 16,
            "max_image_size": 848,
            "steps": 40,
            "seed": 42,
            "random_seed": True,
            "show_adv_guidance": False,
            "use_flash_attention": True,
            "cudnn_v8_fallback": True,
            "use_framevision_queue": False,
        }

    def _load_settings(self) -> None:
        values = self._defaults()
        try:
            # FrameVision user settings belong under /presets/setsave/.
            # Keep a one-time fallback for older helper builds that saved beside /presets/.
            source_file = self.settings_file if self.settings_file.exists() else self.legacy_settings_file
            if source_file.exists():
                values.update(json.loads(source_file.read_text(encoding="utf-8")))
        except Exception as exc:
            self._log(f"[WARN] Could not read settings: {exc}")

        self.env_python_row.setText(str(values.get("env_python", "")))
        self.repo_dir_row.setText(str(values.get("repo_dir", "")))
        self.model_dir_row.setText(str(values.get("model_dir", "")))
        self.output_dir_row.setText(str(values.get("output_dir", "")))
        self.case_dir_row.setText(str(values.get("case_dir", "")))
        self.prompt_edit.setPlainText(str(values.get("prompt", "")))
        self.system_prompt_edit.setPlainText(str(values.get("system_prompt", "")))
        self._set_combo_data(self.task_combo, str(values.get("task", "t2v")))
        self._set_combo_text(self.guidance_combo, str(values.get("guidance", "t2v_apg")))
        self.auto_guidance_cb.setChecked(True)
        self.video_row.setText(str(values.get("video", "")))
        self.image_row.setText(str(values.get("image", "")))
        self.images_row.setText(str(values.get("images", "")))
        self.output_name.setText(str(values.get("output_name", "")))
        self._set_combo_text(self.resolution_combo, str(values.get("aspect_ratio", "16:9")))
        self.width_spin.setValue(int(values.get("width", 1280)))
        self.height_spin.setValue(int(values.get("height", 720)))
        self.keep_source_size_cb.setChecked(bool(values.get("keep_source_size", False)))
        self.num_frames_spin.setValue(int(values.get("num_frames", 81)))
        self.fps_spin.setValue(int(values.get("fps", 16)))
        self.max_image_size_spin.setValue(int(values.get("max_image_size", 848)))
        self.steps_spin.setValue(int(values.get("steps", 40)))
        self.seed_spin.setValue(int(values.get("seed", 42)))
        self.random_seed_btn.setChecked(bool(values.get("random_seed", True)))
        self.show_adv_guidance_cb.setChecked(bool(values.get("show_adv_guidance", False)))
        self.override_neg_cb.setChecked(bool(values.get("override_neg", False)))
        self.neg_prompt_edit.setPlainText(str(values.get("neg_prompt", DEFAULT_NEG_PROMPT)))
        self.omega_vid_spin.setValue(float(values.get("omega_vid", 1.25)))
        self.omega_img_spin.setValue(float(values.get("omega_img", 4.5)))
        self.omega_txt_spin.setValue(float(values.get("omega_txt", 4.0)))
        self.omega_tgt_spin.setValue(float(values.get("omega_tgt", 0.5)))
        self.omega_scale_spin.setValue(float(values.get("omega_scale", 0.8)))
        self.eta_spin.setValue(float(values.get("eta", 0.5)))
        self.momentum_spin.setValue(float(values.get("momentum", 0.0)))
        self.planning_step_spin.setValue(int(values.get("planning_step", 25)))
        self.vit_txt_cfg_spin.setValue(float(values.get("vit_txt_cfg", 1.2)))
        self.vit_img_cfg_spin.setValue(float(values.get("vit_img_cfg", 1.0)))
        self.vit_denoising_step_spin.setValue(int(values.get("vit_denoising_step", 5)))
        self.flow_shift_spin.setValue(float(values.get("flow_shift", 5.0)))
        self.norm1_spin.setValue(float(values.get("norm1", 50.0)))
        self.norm2_spin.setValue(float(values.get("norm2", 50.0)))
        self.norm3_spin.setValue(float(values.get("norm3", 50.0)))
        self.use_unipc_cb.setChecked(bool(values.get("use_unipc", True)))
        self.use_src_tgt_id_cb.setChecked(bool(values.get("use_src_tgt_id", True)))
        self.interpolate_src_id_cb.setChecked(bool(values.get("interpolate_src_id", True)))
        self.max_trained_src_id_spin.setValue(int(values.get("max_trained_src_id", 5)))
        self.high_noise_row.setText(str(values.get("high_noise_ckpt", "")))
        self.low_noise_row.setText(str(values.get("low_noise_ckpt", "")))
        self.expandable_segments_cb.setChecked(bool(values.get("expandable_segments", True)))
        self.cudnn_v8_fallback_cb.setChecked(bool(values.get("cudnn_v8_fallback", True)))
        self.cuda_visible_devices_edit.setText(str(values.get("cuda_visible_devices", "0")))
        self.use_flash_attention_cb.setChecked(bool(values.get("use_flash_attention", True)))
        self.use_framevision_queue_cb.setChecked(bool(values.get("use_framevision_queue", False)))
        self.extra_env_edit.setPlainText(str(values.get("extra_env", "")))
        self.use_prompt_enhancer_cb.setChecked(bool(values.get("use_pe", False)))
        self.pe_model_edit.setText(str(values.get("pe_model", "")))

    def _save_settings(self) -> None:
        data = {
            "env_python": self.env_python_row.text(),
            "repo_dir": self.repo_dir_row.text(),
            "model_dir": self.model_dir_row.text(),
            "output_dir": self.output_dir_row.text(),
            "case_dir": self.case_dir_row.text(),
            "prompt": self.prompt_edit.toPlainText(),
            "system_prompt": self.system_prompt_edit.toPlainText(),
            "task": self.current_task(),
            "guidance": self.guidance_combo.currentText(),
            "auto_guidance": True,
            "video": self.video_row.text(),
            "image": self.image_row.text(),
            "images": self.images_row.text(),
            "output_name": self.output_name.text(),
            "aspect_ratio": self.resolution_combo.currentText(),
            "keep_source_size": self.keep_source_size_cb.isChecked(),
            "width": self.width_spin.value(),
            "height": self.height_spin.value(),
            "num_frames": self.num_frames_spin.value(),
            "fps": self.fps_spin.value(),
            "max_image_size": self.max_image_size_spin.value(),
            "steps": self.steps_spin.value(),
            "seed": self.seed_spin.value(),
            "random_seed": self.random_seed_btn.isChecked(),
            "show_adv_guidance": self.show_adv_guidance_cb.isChecked(),
            "override_neg": self.override_neg_cb.isChecked(),
            "neg_prompt": self.neg_prompt_edit.toPlainText(),
            "omega_vid": self.omega_vid_spin.value(),
            "omega_img": self.omega_img_spin.value(),
            "omega_txt": self.omega_txt_spin.value(),
            "omega_tgt": self.omega_tgt_spin.value(),
            "omega_scale": self.omega_scale_spin.value(),
            "eta": self.eta_spin.value(),
            "momentum": self.momentum_spin.value(),
            "planning_step": self.planning_step_spin.value(),
            "vit_txt_cfg": self.vit_txt_cfg_spin.value(),
            "vit_img_cfg": self.vit_img_cfg_spin.value(),
            "vit_denoising_step": self.vit_denoising_step_spin.value(),
            "flow_shift": self.flow_shift_spin.value(),
            "norm1": self.norm1_spin.value(),
            "norm2": self.norm2_spin.value(),
            "norm3": self.norm3_spin.value(),
            "use_unipc": self.use_unipc_cb.isChecked(),
            "use_src_tgt_id": self.use_src_tgt_id_cb.isChecked(),
            "interpolate_src_id": self.interpolate_src_id_cb.isChecked(),
            "max_trained_src_id": self.max_trained_src_id_spin.value(),
            "high_noise_ckpt": self.high_noise_row.text(),
            "low_noise_ckpt": self.low_noise_row.text(),
            "expandable_segments": self.expandable_segments_cb.isChecked(),
            "cudnn_v8_fallback": self.cudnn_v8_fallback_cb.isChecked(),
            "cuda_visible_devices": self.cuda_visible_devices_edit.text(),
            "use_flash_attention": self.use_flash_attention_cb.isChecked(),
            "use_framevision_queue": self.use_framevision_queue_cb.isChecked(),
            "extra_env": self.extra_env_edit.toPlainText(),
            "use_pe": self.use_prompt_enhancer_cb.isChecked(),
            "pe_model": self.pe_model_edit.text(),
        }
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _set_combo_data(self, combo: QComboBox, value: str) -> None:
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def _set_combo_text(self, combo: QComboBox, value: str) -> None:
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    # ------------------------------------------------------------------ behavior
    def current_task(self) -> str:
        return str(self.task_combo.currentData() or "t2v")

    def _apply_task_defaults(self, force: bool = False) -> None:
        task = self.current_task()
        self._set_combo_text(self.guidance_combo, GUIDANCE_MODE_BY_TASK.get(task, "t2v_apg"))
        self._sync_prompt_enhancer_for_task(task)
        if task in IMAGE_TASKS:
            self.num_frames_spin.setValue(1)
            self.num_frames_spin.setEnabled(False)
        else:
            self.num_frames_spin.setEnabled(True)
            if self.num_frames_spin.value() <= 1:
                self.num_frames_spin.setValue(81)

        inputs = TASK_INPUTS.get(task, TASK_INPUTS["t2v"])
        self.video_row.setEnabled(bool(inputs["video"]))
        image_needed = inputs["image_role"] != "none"
        self.image_row.setEnabled(image_needed)
        self.images_row.setEnabled(bool(inputs["images"]))
        self._update_size_mode_for_task()
        self._refresh_command_preview()

    def _sync_prompt_enhancer_for_task(self, task: Optional[str] = None) -> None:
        try:
            if hasattr(self, "use_prompt_enhancer_cb"):
                self.use_prompt_enhancer_cb.setToolTip("Use Bernini's task-aware prompt enhancer through local Qwen3-VL.")
        except Exception:
            pass

    def _update_size_mode_for_task(self) -> None:
        task = self.current_task()
        source_task = task not in {"t2i", "t2v"}
        if hasattr(self, "source_size_note"):
            self.source_size_note.setVisible(source_task)
        if hasattr(self, "resolution_combo"):
            self.resolution_combo.setEnabled(not source_task)
        if hasattr(self, "keep_source_size_cb"):
            self.keep_source_size_cb.setEnabled(source_task)
        if source_task:
            self.width_spin.setEnabled(False)
            self.height_spin.setEnabled(False)
        elif not self.keep_source_size_cb.isChecked():
            self.width_spin.setEnabled(True)
            self.height_spin.setEnabled(True)

    def _apply_resolution_preset(self, name: str) -> None:
        if self.keep_source_size_cb.isChecked():
            return
        self._apply_aspect_ratio(name, anchor="default")
        self._refresh_command_preview()

    def _apply_aspect_ratio(self, name: str, anchor: str = "width") -> None:
        if getattr(self, "_size_syncing", False):
            return
        ratio = ASPECT_RATIOS.get(name or self.resolution_combo.currentText(), (16, 9))
        rw, rh = ratio
        self._size_syncing = True
        try:
            if anchor == "default":
                w, h = DEFAULT_ASPECT_SIZES.get(name or self.resolution_combo.currentText(), (1280, 720))
                self.width_spin.setValue(w)
                self.height_spin.setValue(h)
            elif anchor == "height":
                h = max(16, self.height_spin.value())
                w = int(round((h * rw / rh) / 16.0) * 16)
                self.width_spin.setValue(max(16, w))
            else:
                w = max(16, self.width_spin.value())
                h = int(round((w * rh / rw) / 16.0) * 16)
                self.height_spin.setValue(max(16, h))
        finally:
            self._size_syncing = False

    def _toggle_keep_source_size(self, checked: bool) -> None:
        self._size_syncing = True
        try:
            if checked:
                self.width_spin.setValue(0)
                self.height_spin.setValue(0)
                self.width_spin.setEnabled(False)
                self.height_spin.setEnabled(False)
            else:
                source_task = self.current_task() not in {"t2i", "t2v"}
                self.width_spin.setEnabled(not source_task)
                self.height_spin.setEnabled(not source_task)
                if self.width_spin.value() == 0 or self.height_spin.value() == 0:
                    self._apply_aspect_ratio(self.resolution_combo.currentText(), anchor="default")
        finally:
            self._size_syncing = False
        self._refresh_command_preview()

    def _width_changed(self, _value: int) -> None:
        if not self.keep_source_size_cb.isChecked():
            self._apply_aspect_ratio(self.resolution_combo.currentText(), anchor="width")
            self._refresh_command_preview()

    def _height_changed(self, _value: int) -> None:
        if not self.keep_source_size_cb.isChecked():
            self._apply_aspect_ratio(self.resolution_combo.currentText(), anchor="height")
            self._refresh_command_preview()

    def _random_seed(self) -> int:
        value = random.randint(0, 2_147_483_647)
        self.seed_spin.setValue(value)
        return value

    def _update_random_seed_toggle_text(self, checked: bool) -> None:
        self._save_settings_safe()

    def _update_guidance_tab_visibility(self, checked: bool) -> None:
        try:
            self.tabs.setTabVisible(self.guidance_tab_index, checked)
        except Exception:
            self.guidance_tab.setVisible(checked)
        if not checked and self.tabs.currentIndex() == self.guidance_tab_index:
            self.tabs.setCurrentIndex(self.generate_tab_index)
        self._save_settings_safe()

    def _save_settings_safe(self) -> None:
        if getattr(self, "_loading_settings", False):
            return
        try:
            self._save_settings()
        except Exception:
            pass

    def _output_path(self) -> Path:
        out_dir = Path(self.output_dir_row.text() or self.root / "output" / "video" / "bernini")
        out_dir.mkdir(parents=True, exist_ok=True)
        task = self.current_task()
        ext = ".png" if task in IMAGE_TASKS else ".mp4"
        name = self.output_name.text().strip()
        if not name:
            name = f"bernini_small_{task}_{time.strftime('%Y%m%d_%H%M%S')}{ext}"
        path = Path(name)
        if not path.suffix:
            path = path.with_suffix(ext)
        if not path.is_absolute():
            path = out_dir / path
        return path

    def _case_file(self, output_path: Path) -> Path:
        case_dir = Path(self.case_dir_row.text() or self.temp_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        task = self.current_task()
        prompt = self.prompt_edit.toPlainText().strip()
        case = {
            "task_type": task,
            "guidance_mode": self.guidance_combo.currentText(),
            "prompt": prompt,
            "output": str(output_path),
        }
        inputs = TASK_INPUTS.get(task, TASK_INPUTS["t2v"])
        videos = _split_paths(self.video_row.text())
        image = self.image_row.text()
        refs = _split_paths(self.images_row.text())
        if inputs["video"] and videos:
            case["video"] = videos if len(videos) > 1 else videos[0]
        if inputs["image_role"] == "source" and image:
            case["image"] = image
        elif inputs["image_role"] == "reference" and image:
            refs = [image] + refs
        if inputs["images"] and refs:
            case["images"] = refs
        case_file = case_dir / f"bernini_small_{task}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        case_file.write_text(json.dumps(case, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_case_file = case_file
        return case_file

    def _build_command(self, for_preview: bool = False) -> Tuple[List[str], Optional[Path]]:
        python_exe = self.env_python_row.text() or str(self.root / "environments" / ".bernini_r" / "python.exe")
        repo_dir = Path(self.repo_dir_row.text() or self.root / "models" / "bernini_r_1p3b" / "Bernini")
        model_dir = self.model_dir_row.text() or str(self.root / "models" / "bernini_r_1p3b" / "Bernini-R-1.3B-Diffusers")
        output_path = self._output_path()
        case_file = Path(self.case_dir_row.text() or self.temp_dir) / "bernini_small_preview_case.json"
        if not for_preview:
            case_file = self._case_file(output_path)
        script = repo_dir / "infer_single_gpu.py"
        cmd = [str(python_exe), str(script), "--config", str(model_dir), "--case", str(case_file)]

        hi = self.high_noise_row.text()
        lo = self.low_noise_row.text()
        if hi or lo:
            cmd += ["--high_noise_ckpt", hi, "--low_noise_ckpt", lo]

        if not self.use_unipc_cb.isChecked():
            cmd.append("--no-use_unipc")
        if not self.use_src_tgt_id_cb.isChecked():
            cmd.append("--no-use_src_tgt_id")
        if not self.interpolate_src_id_cb.isChecked():
            cmd.append("--no-interpolate_src_id")
        cmd += ["--max_trained_src_id", str(self.max_trained_src_id_spin.value())]

        cmd += [
            "--num_frames", str(self.num_frames_spin.value()),
            "--max_image_size", str(self.max_image_size_spin.value()),
            "--height", str(self.height_spin.value()),
            "--width", str(self.width_spin.value()),
            "--num_inference_steps", str(self.steps_spin.value()),
            "--guidance_mode", self.guidance_combo.currentText(),
            "--omega_vid", str(self.omega_vid_spin.value()),
            "--omega_img", str(self.omega_img_spin.value()),
            "--omega_txt", str(self.omega_txt_spin.value()),
            "--omega_tgt", str(self.omega_tgt_spin.value()),
            "--omega_scale", str(self.omega_scale_spin.value()),
            "--planning_step", str(self.planning_step_spin.value()),
            "--vit_txt_cfg", str(self.vit_txt_cfg_spin.value()),
            "--vit_img_cfg", str(self.vit_img_cfg_spin.value()),
            "--vit_denoising_step", str(self.vit_denoising_step_spin.value()),
            "--flow_shift", str(self.flow_shift_spin.value()),
            "--seed", str(self.seed_spin.value()),
            "--fps", str(self.fps_spin.value()),
            "--eta", str(self.eta_spin.value()),
            "--norm_threshold", str(self.norm1_spin.value()), str(self.norm2_spin.value()), str(self.norm3_spin.value()),
            "--momentum", str(self.momentum_spin.value()),
        ]
        if self.override_neg_cb.isChecked():
            cmd += ["--neg_prompt", self.neg_prompt_edit.toPlainText()]
        if self.system_prompt_edit.toPlainText().strip():
            cmd += ["--system_prompt", self.system_prompt_edit.toPlainText().strip()]
        if self.use_prompt_enhancer_cb.isChecked():
            cmd.append("--use_pe")
            if self.pe_model_edit.text().strip():
                cmd += ["--pe_model", self.pe_model_edit.text().strip()]
        return cmd, output_path

    def _validate(self) -> bool:
        errors = []
        if not Path(self.env_python_row.text()).exists():
            errors.append("Bernini Python path does not exist.")
        repo_dir = Path(self.repo_dir_row.text())
        if not (repo_dir / "infer_single_gpu.py").exists():
            errors.append("Repo folder must contain infer_single_gpu.py.")
        if not Path(self.model_dir_row.text()).exists():
            errors.append("Model/config folder does not exist.")
        if not self.prompt_edit.toPlainText().strip():
            errors.append("Prompt is empty.")
        task = self.current_task()
        inputs = TASK_INPUTS.get(task, TASK_INPUTS["t2v"])
        if inputs["video"] and not _split_paths(self.video_row.text()):
            errors.append(f"{task} needs a source video.")
        if inputs["image_role"] == "source" and not self.image_row.text():
            errors.append(f"{task} needs a source image.")
        if inputs["images"]:
            refs = _split_paths(self.images_row.text())
            if inputs["image_role"] == "reference" and self.image_row.text():
                refs.append(self.image_row.text())
            if not refs:
                errors.append(f"{task} needs at least one reference image.")
        hi, lo = self.high_noise_row.text(), self.low_noise_row.text()
        if bool(hi) != bool(lo):
            errors.append("high_noise_ckpt and low_noise_ckpt must be filled together or both empty.")
        if errors:
            QMessageBox.warning(self, "Bernini-R 1.3B", "\n".join(errors))
            return False
        return True


    def _ensure_attention_backend_patch(self) -> None:
        """Install a small local Bernini attention.py patch for backend logging/control."""
        try:
            repo_dir = Path(self.repo_dir_row.text() or self.root / "models" / "bernini_r_1p3b" / "Bernini")
            attention_file = repo_dir / "bernini" / "attention.py"
            if not attention_file.exists():
                self._log(f"[ATTENTION] attention.py not found, backend logging patch skipped: {attention_file}")
                return
            current = attention_file.read_text(encoding="utf-8")
            if "FRAMEVISION_ATTENTION_BACKEND_PATCH_V1" in current:
                self._log("[ATTENTION] Bernini attention backend patch already installed")
                return
            backup = attention_file.with_suffix(".py.framevision_backup")
            if not backup.exists():
                backup.write_text(current, encoding="utf-8")
                self._log(f"[ATTENTION] Backup written: {backup}")
            attention_file.write_text(FRAMEVISION_ATTENTION_PY, encoding="utf-8")
            self._log("[ATTENTION] Installed Bernini attention backend logging/control patch")
        except Exception as exc:
            self._log(f"[ATTENTION] Could not install backend patch: {exc}")



    def _ensure_prompt_enhancer_patch(self) -> None:
        """Install FrameVision's local-Qwen Bernini prompt_enhancer into the repo.

        Bernini imports the enhancer from the package path shown in tracebacks:
            <repo>/bernini/prompt_enhancer.py
        Some repo revisions also import a top-level prompt_enhancer.py, so we update
        both locations. This keeps --use_pe on Bernini's own enhancer path while
        replacing the OpenAI/API backend with FrameVision's local Qwen3-VL backend.
        """
        if not getattr(self, "use_prompt_enhancer_cb", None) or not self.use_prompt_enhancer_cb.isChecked():
            return
        try:
            repo_dir = Path(self.repo_dir_row.text() or self.root / "models" / "bernini_r_1p3b" / "Bernini")
            source_candidates = [
                self.root / "helpers" / "prompt_enhancer.py",
                Path(__file__).resolve().parent / "prompt_enhancer.py",
            ]
            source = next((p for p in source_candidates if p.exists()), None)
            if source is None:
                self._log("[PE] Local prompt_enhancer.py patch source not found; --use_pe may still require API settings")
                return
            src_text = source.read_text(encoding="utf-8")
            if "FRAMEVISION_LOCAL_QWEN_PE_V1" not in src_text:
                self._log("[PE] prompt_enhancer.py found, but it is not the local-Qwen FrameVision version")

            targets = [
                repo_dir / "bernini" / "prompt_enhancer.py",
                repo_dir / "prompt_enhancer.py",
            ]
            installed_any = False
            already_all = True
            for target in targets:
                target.parent.mkdir(parents=True, exist_ok=True)
                current = ""
                if target.exists():
                    try:
                        current = target.read_text(encoding="utf-8")
                    except Exception:
                        current = ""
                    backup = target.with_suffix(".py.framevision_api_backup")
                    if "FRAMEVISION_LOCAL_QWEN_PE_V1" not in current and not backup.exists():
                        backup.write_text(current, encoding="utf-8")
                        self._log(f"[PE] Original Bernini prompt_enhancer backup written: {backup}")
                if current == src_text:
                    continue
                already_all = False
                target.write_text(src_text, encoding="utf-8")
                installed_any = True
                self._log(f"[PE] Installed local-Qwen Bernini prompt enhancer: {target}")
            if already_all and not installed_any:
                self._log("[PE] Local Qwen prompt enhancer already installed in Bernini repo/package")
        except Exception as exc:
            self._log(f"[PE] Could not install local prompt enhancer patch: {exc}")

    def _runtime_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        # Keep Windows logs readable when VeOmni/Bernini print unicode symbols.
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        if self.expandable_segments_cb.isChecked():
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if getattr(self, "cudnn_v8_fallback_cb", None) and self.cudnn_v8_fallback_cb.isChecked():
            env["TORCH_CUDNN_V8_API_DISABLED"] = "1"
        cuda = self.cuda_visible_devices_edit.text().strip()
        if cuda:
            env["CUDA_VISIBLE_DEVICES"] = cuda
        env["BERNINI_ATTENTION_BACKEND"] = "fa2" if self.use_flash_attention_cb.isChecked() else "sdpa"
        if getattr(self, "use_prompt_enhancer_cb", None) and self.use_prompt_enhancer_cb.isChecked():
            env.setdefault("BERNINI_PE_BACKEND", "local_qwen")
            env.setdefault("BERNINI_PE_QWEN_MODEL_DIR", str(self.root / "models" / "describe" / "default" / "qwen3vl2b"))
        for line in self.extra_env_edit.toPlainText().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key:
                env[key] = value.strip()
        return env

    def _refresh_command_preview(self) -> None:
        try:
            cmd, out = self._build_command(for_preview=True)
            case_hint = {
                "task_type": self.current_task(),
                "guidance_mode": self.guidance_combo.currentText(),
                "prompt": self.prompt_edit.toPlainText().strip() or "<prompt>",
                "output": str(out),
            }
            text = "Case JSON preview:\n" + json.dumps(case_hint, indent=2, ensure_ascii=False)
            requested_backend = "fa2" if self.use_flash_attention_cb.isChecked() else "sdpa"
            text += "\n\nRuntime env preview:\n"
            text += f"PYTHONUTF8=1\nPYTHONIOENCODING=utf-8\nBERNINI_ATTENTION_BACKEND={requested_backend}"
            if getattr(self, "cudnn_v8_fallback_cb", None) and self.cudnn_v8_fallback_cb.isChecked():
                text += "\nTORCH_CUDNN_V8_API_DISABLED=1"
            text += "\n\nCommand preview:\n" + _quote_command(cmd)
            self.command_preview.setPlainText(text)
        except Exception as exc:
            self.command_preview.setPlainText(f"Could not build command preview: {exc}")

    # ------------------------------------------------------------------ process
    def _sync_queue_button_text(self, *_args) -> None:
        try:
            queued = bool(getattr(self, "use_framevision_queue_cb", None) and self.use_framevision_queue_cb.isChecked())
            self.generate_btn.setText("Add to queue" if queued else "Generate")
            if queued:
                self.generate_btn.setToolTip("Add the Bernini job to FrameVision's queue and stay on this tab.")
            else:
                self.generate_btn.setToolTip("Start the Bernini job directly with the current settings.")
        except Exception:
            pass

    def _enqueue_to_framevision_queue(self, cmd: List[str], output_path: Path, repo_dir: Path) -> bool:
        try:
            try:
                from helpers.queue_adapter import enqueue_tool_job as enq  # type: ignore
            except Exception:
                from queue_adapter import enqueue_tool_job as enq  # type: ignore

            log_file = self.log_dir / f"bernini_r_1p3b_queue_{time.strftime('%Y%m%d_%H%M%S')}.log"
            args = {
                "ffmpeg_cmd": cmd,
                "cmd": cmd,
                "cwd": str(repo_dir),
                "outfile": str(output_path),
                "log_file": str(log_file),
                "env": self._runtime_env(),
                "engine": "bernini_r_1p3b",
                "label": f"Bernini-R 1.3B - {self.current_task()}",
                "scan_dir": str(output_path.parent),
                "scan_ext": output_path.suffix or (".png" if self.current_task() in IMAGE_TASKS else ".mp4"),
            }
            jid = enq("tools_ffmpeg", str(output_path), str(output_path.parent), args, priority=560)
            self._log(f"[QUEUE] Added Bernini-R 1.3B job: {jid}")
            self.status_label.setText("Added to queue")
            return True
        except Exception as exc:
            self._log(f"[QUEUE_ERROR] {exc}")
            QMessageBox.warning(self, "Bernini-R 1.3B", f"Could not add job to FrameVision queue:\n{exc}")
            return False

    def generate(self) -> None:
        if self._busy:
            return
        if not self._validate():
            return
        if self.random_seed_btn.isChecked():
            new_seed = self._random_seed()
            self._log(f"[SEED] Randomized seed -> {new_seed}")
        self._save_settings()
        try:
            cmd, output_path = self._build_command(for_preview=False)
        except Exception as exc:
            QMessageBox.critical(self, "Bernini-R 1.3B", f"Could not build command:\n{exc}")
            return

        self.latest_output = output_path
        self._saw_saved_output = False
        self._auto_closing_process = False
        repo_dir = Path(self.repo_dir_row.text())
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log("\n" + "=" * 88)
        self._log(f"[START] Bernini-R 1.3B {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"[CASE] {self._last_case_file}")
        self._log(f"[OUTPUT] {output_path}")
        self._ensure_attention_backend_patch()
        self._ensure_prompt_enhancer_patch()
        requested_backend = "fa2" if self.use_flash_attention_cb.isChecked() else "sdpa"
        self._log(f"[ATTENTION] requested_backend={requested_backend} (env BERNINI_ATTENTION_BACKEND)")
        if getattr(self, "cudnn_v8_fallback_cb", None) and self.cudnn_v8_fallback_cb.isChecked():
            self._log("[ENV] TORCH_CUDNN_V8_API_DISABLED=1")
        self._log("[ENV] PYTHONUTF8=1 PYTHONIOENCODING=utf-8")
        self._log("[CMD] " + _quote_command(cmd))

        if bool(getattr(self, "use_framevision_queue_cb", None) and self.use_framevision_queue_cb.isChecked()):
            self._enqueue_to_framevision_queue(cmd, output_path, repo_dir)
            return

        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(repo_dir))
        env = self.process.processEnvironment()
        for k, v in self._runtime_env().items():
            env.insert(k, v)
        self.process.setProcessEnvironment(env)
        self.process.setProgram(cmd[0])
        self.process.setArguments(cmd[1:])
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._set_busy(True)
        self.process.start()
        if not self.process.waitForStarted(5000):
            self._log("[ERROR] Process did not start.")
            self._done_watchdog.stop()
            self._set_busy(False)
        else:
            self._done_watchdog.start()

    def stop(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self._log("[STOP] Terminating Bernini process...")
            self.process.terminate()
            QTimer.singleShot(4000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process and self.process.state() != QProcess.NotRunning:
            self._log("[STOP] Killing Bernini process...")
            self.process.kill()

    def _read_process(self) -> None:
        if not self.process:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            if "saved ->" in data:
                self._saw_saved_output = True
            self._log(data.rstrip("\n"))

    def _output_file_ready(self) -> bool:
        try:
            return bool(self.latest_output and self.latest_output.exists() and self.latest_output.stat().st_size > 0)
        except Exception:
            return False

    def _check_output_done(self) -> None:
        """Bernini sometimes leaves the Python process alive after writing the file.

        When the repo has printed its saved-output line and the file is present, the job is
        effectively complete. Close the still-running helper process so the UI becomes usable
        again instead of leaving Generate disabled forever.
        """
        if not self._busy or not self.process:
            return
        if self.process.state() == QProcess.NotRunning:
            return
        if not (self._saw_saved_output and self._output_file_ready()):
            return
        if self._auto_closing_process:
            return
        self._auto_closing_process = True
        self._log("[INFO] Output file is saved. Closing lingering Bernini process...")
        self.process.terminate()
        QTimer.singleShot(3000, self._kill_if_running)

    def _process_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._done_watchdog.stop()
        self._read_process()
        file_ready = self._output_file_ready()
        ok = (exit_status == QProcess.NormalExit and exit_code == 0) or (self._saw_saved_output and file_ready)
        status_name = getattr(exit_status, "name", str(exit_status))
        status_value = getattr(exit_status, "value", None)
        if status_value is None:
            self._log(f"[DONE] exit_code={exit_code} status={status_name}")
        else:
            self._log(f"[DONE] exit_code={exit_code} status={status_name} ({status_value})")
        if ok and self.latest_output and self.latest_output.exists():
            self.status_label.setText(f"Done: {self.latest_output.name}")
        elif ok:
            self.status_label.setText("Finished, but output file was not found yet")
        else:
            self.status_label.setText("Failed — check logs")
        self._set_busy(False)
        if self.process:
            self.process.deleteLater()
            self.process = None
        self._auto_closing_process = False

    def _process_error(self, error) -> None:
        self._log(f"[PROCESS_ERROR] {error}")
        if self._saw_saved_output and self._output_file_ready():
            self.status_label.setText(f"Done: {self.latest_output.name}")
        else:
            self.status_label.setText("Process error")
        self._done_watchdog.stop()
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.generate_btn.setEnabled(not busy)
        self._sync_queue_button_text()
        self.stop_btn.setEnabled(busy)
        self.status_label.setText("Running..." if busy else self.status_label.text())

    def _log(self, text: str) -> None:
        self.log_view.moveCursor(QTextCursor.End)
        self.log_view.insertPlainText(text + "\n")
        self.log_view.moveCursor(QTextCursor.End)

    # ------------------------------------------------------------------ actions
    def view_results(self) -> None:
        path = Path(self.output_dir_row.text() or self.root / "output" / "video" / "bernini")
        try:
            preset = "images" if self.current_task() in IMAGE_TASKS else "videos"
        except Exception:
            preset = "all"
        _fv_open_results_in_media_explorer(self, path, preset=preset)

    def open_output_folder(self) -> None:
        path = Path(self.output_dir_row.text() or self.root / "output" / "video" / "bernini")
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def open_latest_output(self) -> None:
        if self.latest_output and self.latest_output.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.latest_output)))
        else:
            QMessageBox.information(self, "Bernini-R 1.3B", "No generated output is available yet.")

    def copy_command(self) -> None:
        QApplication.clipboard().setText(self.command_preview.toPlainText())
        self.status_label.setText("Command copied")


def create_widget(parent: Optional[QWidget] = None) -> BerniniSmallWidget:
    return BerniniSmallWidget(parent)


def open_bernini_small(parent: Optional[QWidget] = None) -> BerniniSmallWidget:
    widget = BerniniSmallWidget(parent)
    widget.show()
    return widget


if __name__ == "__main__":
    app = QApplication.instance() or QApplication(sys.argv)
    w = BerniniSmallWidget()
    w.show()
    sys.exit(app.exec())
