from __future__ import annotations

# --- Minimal no-overwrite helper ---
from pathlib import Path as _P
def _unique_path(p: _P) -> _P:
    try:
        p = _P(p)
        if not p.exists():
            return p
        stem, suffix = p.stem, p.suffix
        i = 1
        while True:
            cand = p.with_name(f"{stem}_{i:03d}{suffix}")
            if not cand.exists():
                return cand
            i += 1
    except Exception:
        return _P(p)


# --- sd-cli.exe / sd.exe preferred location (GGUF engines) ---
def _preferred_sd_cli_from_presets_bin(root_dir: _P):
    """Prefer <root>/presets/bin/{sd-cli.exe, sd.exe} for GGUF engines, fall back elsewhere."""
    try:
        rd = _P(root_dir)
    except Exception:
        rd = _P('.')
    cand = [
        rd / 'presets' / 'bin' / 'sd-cli.exe',
        rd / 'presets' / 'bin' / 'sd.exe',
        rd / 'presets' / 'bin' / 'sd-cli',
        rd / 'presets' / 'bin' / 'sd',
    ]
    for c in cand:
        try:
            if c.exists() and c.is_file():
                return c
        except Exception:
            continue
    return None


# === QImageIO maxalloc disabled by patch ===
import os as _qt_img_os
_qt_img_os.environ["QT_IMAGEIO_MAXALLOC"] = "0"  # Disable env-based cap (0 = no limit)
try:
    from PySide6.QtGui import QImageReader as _QIR
    _QIR.setAllocationLimit(0)  # Disable runtime cap as well
except Exception as _e:
    pass
# === end patch ===

# --- Prompt enhancer presets (shared with helpers/prompt.py) ---
# Mirrors the strategy used in hunyuan15.py: reuse PromptTool preset definitions to drive variation.
try:
    from helpers import prompt as _prompt_mod  # type: ignore
except Exception:
    try:
        import prompt as _prompt_mod  # type: ignore
    except Exception:
        _prompt_mod = None  # type: ignore

try:
    _PROMPT_PRESET_DEFS = getattr(_prompt_mod, "PRESET_DEFS", {}) or {}
except Exception:
    _PROMPT_PRESET_DEFS = {}

try:
    _PROMPT_LENGTH_PRESETS = getattr(_prompt_mod, "LENGTH_PRESETS", {}) or {}
except Exception:
    _PROMPT_LENGTH_PRESETS = {}

if "Default" not in (_PROMPT_PRESET_DEFS or {}):
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



# --- Quiet mode: suppress harmless logs/warnings for end users ---
import os, warnings
# Progress bars & banners
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
# Transformers / Diffusers logger levels
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
        pass
try:
    from diffusers.utils import logging as _df_logging
    _df_logging.set_verbosity_error()
    try:
        _df_logging.disable_progress_bar()
    except Exception:
        pass
except Exception:
        pass
# Accelerate logger (used by pipelines)
try:
    import logging as _pylogging
    from accelerate.logging import get_logger as _acc_get_logger
    _acc_get_logger("accelerate").setLevel(_pylogging.ERROR)
except Exception:
        pass
# Specific harmless warning from CLIPTextModel init (position_ids)
warnings.filterwarnings(
    "ignore",
    message=r".*Some weights of the model checkpoint were not used when initializing CLIPTextModel:.*position_ids.*",
    category=UserWarning,
)
# -----------------------------------------------------------------
# >>> FRAMEVISION_QWEN_BEGIN

# Settings persistence toggle (shared with app Settings tab)
def _keep_settings_enabled() -> bool:
    try:
        return bool(QSettings('FrameVision','FrameVision').value('keep_settings_after_restart', True, type=bool))
    except Exception:
        return True
# Text->Image tab and generator (Diffusers-first)
import json, time, random, threading
import subprocess
import shlex
from PIL import Image

# Snap any W/H to a safe bucket: multiples of 64, clamp, and prefer known aspect presets
def _safe_snap_size(w:int, h:int, presets=None):
    try:
        presets = presets or [
            # 1:1


            ("896x896 (1:1)", 896, 896),
            ("512x512 (1:1)", 512, 512),
            (320,320),(480,480),(640,640),(960,960),(1024,1024),(1440,1440),
            # 16:9 landscape
            (1280,720),(1536,864),(1920,1080),(2560,1440),
            # 9:16 portrait
            (720,1280),(864,1536),(1080,1920),(1440,2560),
            # 640p-ish 16:9 (Apple-ish) + portrait
            (1136,640),(640,1136),
            # Cinematic + their portrait
            (1280,544),(544,1280),
            # 3:2 and 2:3
            (1152,768),(1536,1024),(768,1152),(1024,1536),
            # 7:9 and 9:7
            (1152,896),(896,1152),
            # 4:3 and 3:4
            (640,480),(800,600),(1024,768),(1104,832),(480,640),(600,800),(768,1024),(832,1104)
        ]# round to multiples of 64
        def r64(x):
            return max(256, int(round(x/64))*64)
        w64, h64 = r64(w), r64(h)
        # find nearest preset by aspect then by area difference
        aspect = w64 / h64 if h64 else 1.0
        def score(p):
            pw, ph = p
            a = pw / ph if ph else 1.0
            return (abs(a - aspect), abs((pw*ph) - (w64*h64)))
        best = min(presets, key=score)
        return best
    except Exception:
        return (w, h)
from pathlib import Path

def _record_result(job, out_dir, status: dict):
    """Persist a sidecar JSON with detailed status (including errors) so the UI can show why it fell back."""
    import time, json
    sidecar = {
        "ok": bool(status.get("files")),
        "prompt": job.get("prompt",""),
        "negative": job.get("negative",""),
        "seed": job.get("seed", 0),
        "seed_policy": job.get("seed_policy","fixed"),
        "preset": job.get("preset"),
        "batch": job.get("batch",1),
        "files": status.get("files", []),
        "created_at": time.time(),
        "engine": status.get("backend","unknown"),
        "model": status.get("model"),
        "error": status.get("error"),
        "trace": status.get("trace"),
        "steps": job.get("steps", 30),
        "requested_size": [job.get("width", 1024), job.get("height", 1024)],
        "actual_size": status.get("actual_size"),
        "lora_mode": status.get("lora_mode"),
        "loras": status.get("loras"),
        "lora_scales": status.get("lora_scales"),
        "lora_names": status.get("lora_names"),
    }
    # Write next to first image or into out_dir
    target = out_dir / (status.get("files",[None])[0] and (pathlib.Path(status["files"][0]).stem + ".json") or "last_txt2img.json")
    with open(target, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    return sidecar

def _safe_exists(path_str):
    from pathlib import Path as _P
    try:
        return _P(path_str).exists()

    except Exception:
        return False


def _aggressive_free_cuda_vram():
    """Best-effort CUDA VRAM cleanup focused on Qwen/Qwen-VL style models.

    This walks:
      - sys.modules to find obvious Qwen-related modules
      - gc.get_objects() to find live torch.nn.Module instances whose
        class or module name mentions "qwen" and that still have CUDA
        parameters.

    It then tries to move those modules to CPU via .to("cpu"), followed by
    a generic gc.collect() + torch.cuda.empty_cache()/ipc_collect().

    Everything is wrapped defensively so it can never crash callers.
    """
    try:
        import sys as _sys
        import gc as _gc
        try:
            import torch as _torch  # type: ignore
            from torch import nn as _nn  # type: ignore
        except Exception:
            _torch = None  # type: ignore
            _nn = None     # type: ignore

        # First pass: scan sys.modules for obvious Qwen modules and try to
        # move any .to()-capable attributes to CPU.
        try:
            for _name, _mod in list((_sys.modules or {}).items()):
                if not _mod:
                    continue
                try:
                    _lname = str(_name).lower()
                except Exception:
                    _lname = ""
                if "qwen" not in _lname:
                    continue
                for _attr in dir(_mod):
                    if _attr.startswith("__"):
                        continue
                    try:
                        _obj = getattr(_mod, _attr)
                    except Exception:
                        continue
                    try:
                        if hasattr(_obj, "to"):
                            try:
                                _obj.to("cpu")
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            # Best-effort only
            pass

        # Second pass: scan all live objects for torch.nn.Module instances
        # that look like Qwen/Qwen-VL models and still have CUDA params.
        try:
            if _torch is not None and _nn is not None:
                for _obj in list(_gc.get_objects()):
                    try:
                        if not isinstance(_obj, _nn.Module):
                            continue
                    except Exception:
                        continue
                    try:
                        _cls = _obj.__class__
                        _cname = getattr(_cls, "__name__", "") or ""
                        _mname = getattr(_cls, "__module__", "") or ""
                        _lname_c = str(_cname).lower()
                        _lname_m = str(_mname).lower()
                    except Exception:
                        _lname_c = ""
                        _lname_m = ""
                    if "qwen" not in _lname_c and "qwen" not in _lname_m:
                        continue

                    # Check if any parameters are on CUDA
                    try:
                        has_cuda_param = False
                        for _p in _obj.parameters():
                            try:
                                if getattr(_p, "is_cuda", False):
                                    has_cuda_param = True
                                    break
                            except Exception:
                                continue
                        if not has_cuda_param:
                            continue
                    except Exception:
                        # If we can't inspect parameters, skip
                        continue

                    # Try to move the whole module to CPU
                    try:
                        _obj.to("cpu")
                    except Exception:
                        pass
        except Exception:
            pass

        # Final GC + CUDA cache clear
        try:
            _gc.collect()
        except Exception:
            pass
        try:
            if _torch is not None and hasattr(_torch, "cuda") and _torch.cuda.is_available():
                try:
                    _torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    if hasattr(_torch.cuda, "ipc_collect"):
                        _torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        # Absolutely never crash caller from here
        pass

from typing import Callable, Optional
try:
    import requests
except Exception:
    requests = None

from PySide6.QtCore import QSettings, QTimer, Qt, Signal, QProcess, QProcessEnvironment
from PySide6.QtWidgets import (
    QMessageBox,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QSpinBox,
    QCheckBox, QFileDialog, QComboBox, QProgressBar, QGroupBox, QFormLayout, QScrollArea, QToolButton, QSlider,
    QDoubleSpinBox,
    QSizePolicy,
    QApplication,
)
# Import QShortcut correctly from QtGui; fall back to no shortcut if missing
try:
    from PySide6.QtGui import QKeySequence, QImage, QPainter, QPainterPath, QShortcut
except Exception:
    from PySide6.QtGui import QKeySequence, QImage, QPainter  # type: ignore
    QShortcut = None  # type: ignore


class _Disclosure(QWidget):
    toggled = Signal(bool)
    def __init__(self, title: str, content: QWidget, start_open: bool = False, parent=None):
        super().__init__(parent)
        self._btn = QToolButton(self)
        self._btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._btn.setArrowType(Qt.DownArrow if start_open else Qt.RightArrow)
        self._btn.setText(title)
        self._btn.setCheckable(True); self._btn.setChecked(start_open)
        self._btn.toggled.connect(self._on_clicked)
        self._body = content; self._body.setVisible(start_open)
        lay = QVBoxLayout(self); lay.setContentsMargins(6,6,6,6)
        lay.addWidget(self._btn); lay.addWidget(self._body)
    def _on_clicked(self, checked: bool):
        self._body.setVisible(checked)
        self._btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggled.emit(checked)

# --- helper: rounded pixmap for nicer thumbnails ---
class Txt2ImgPane(QWidget):


    def _apply_queue_visibility(self, checked: bool | None = None):
        """Hide progress + status when queue mode is on."""
        try:
            if checked is None:
                checked = bool(self.use_queue.isChecked())
        except Exception:
            checked = False
        try:
            self.progress.setVisible(not checked)
        except Exception:
            pass
        try:
            self.status.setVisible(not checked)
        except Exception:
            pass


    def _apply_settings_from_dict(self, s: dict):
            """Apply saved settings dict to the UI (best-effort) without triggering autosave."""
            # Restore preset selection (UI only; block signals to avoid reapplying values)
            try:
                if hasattr(self, 'preset_combo') and isinstance(s, dict):
                    _pv = s.get('preset') or s.get('preset_label') or s.get('preset_name')
                    if _pv:
                        try:
                            self.preset_combo.blockSignals(True)
                        except Exception:
                            pass
                        _pi = self.preset_combo.findText(str(_pv))
                        if _pi >= 0:
                            self.preset_combo.setCurrentIndex(_pi)
                        try:
                            self.preset_combo.blockSignals(False)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                func = globals().get('_t2i_apply_from_dict')
                if callable(func):
                    # Apply rich loader but continue so engine/output can also be restored
                    func(self, s)
            except Exception:
                pass
            # Minimal built-in mapper
            try:
                if hasattr(self, 'prompt') and 'prompt' in s:
                    self.prompt.setPlainText(s.get('prompt') or '')
                if hasattr(self, 'negative') and 'negative' in s:
                    self.negative.setPlainText(s.get('negative') or '')
            except Exception:
                pass
            try:
                if hasattr(self, "zimg_init_enable") and "init_image_enabled" in s:
                    self.zimg_init_enable.setChecked(bool(s.get("init_image_enabled", False)))
                if hasattr(self, "zimg_init_path") and "init_image" in s:
                    self.zimg_init_path.setText(str(s.get("init_image") or ""))
                if hasattr(self, "zimg_strength") and "img2img_strength" in s:
                    try:
                        self.zimg_strength.setValue(float(s.get("img2img_strength") or 0.35))
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if hasattr(self, 'seed') and 'seed' in s:
                    self.seed.setValue(int(s.get('seed') or 0))
                sp = (s.get('seed_policy') or '').lower()
                if hasattr(self, 'seed_policy') and sp:
                    try: self.seed_policy.setCurrentText('Random' if 'random' in sp else 'Fixed')
                    except Exception:
                        idx = -1
                        for _i in range(self.seed_policy.count()):
                            if ('random' in sp and 'Random' in (self.seed_policy.itemText(_i) or '')) or ('fixed' in sp and 'Fixed' in (self.seed_policy.itemText(_i) or '')):
                                idx = _i; break
                        if idx >= 0:
                            self.seed_policy.setCurrentIndex(idx) if sp is not None else None
            except Exception:
                pass
            try:
                eng = (str(s.get("engine") or s.get("backend") or "")).strip().lower()
                cb = getattr(self, "engine_combo", None)
                if cb is not None and eng:
                    idx = -1
                    try:
                        for _i in range(cb.count()):
                            data = cb.itemData(_i)
                            text = (cb.itemText(_i) or "").lower()
                            if (isinstance(data, str) and isinstance(eng, str) and data is not None and str(data).lower() == eng) or (eng and eng in text):
                                idx = _i
                                break
                    except Exception:
                        idx = -1
                    if idx >= 0:
                        cb.setCurrentIndex(idx)
            except Exception:
                pass
            try:
                if hasattr(self,'batch') and 'batch' in s:
                    self.batch.setValue(int(s.get('batch') or 1))
            except Exception:
                pass
            try:
                w = int(s.get('width', 0)); h = int(s.get('height', 0))
                if w and h:
                    if hasattr(self,'size_combo') and getattr(self,'_size_presets', None):
                        idx = -1
                        for i,(label,wv,hv) in enumerate(getattr(self,'_size_presets', [])):
                            if wv==w and hv==h:
                                idx = i; break
                        if idx>=0:
                            try: self.size_combo.setCurrentIndex(idx)
                            except Exception: pass
                    if hasattr(self,'size_manual_w'): self.size_manual_w.setValue(w)
                    if hasattr(self,'size_manual_h'): self.size_manual_h.setValue(h)
            except Exception:
                pass
            try:
                    if hasattr(self,'steps_slider') and 'steps' in s: self.steps_slider.setValue(int(s.get('steps') or 30))
                    if hasattr(self,'cfg_scale') and 'cfg_scale' in s:
                        try:
                            self.cfg_scale.setValue(float(s.get('cfg_scale')))
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                if hasattr(self,'vram_profile') and 'vram_profile' in s:
                    self.vram_profile.setCurrentText(s.get('vram_profile'))
                if hasattr(self,'sampler') and 'sampler' in s:
                    self.sampler.setCurrentText(s.get('sampler'))
                if hasattr(self,'qwen_flow_shift') and 'flow_shift' in s:
                    try:
                        self.qwen_flow_shift.setValue(int(s.get('flow_shift', 3)))
                    except Exception:
                        pass
                if hasattr(self,'attn_slicing') and 'attn_slicing' in s:
                    self.attn_slicing.setChecked(bool(s.get('attn_slicing')))
                if hasattr(self,'vae_device') and 'vae_device' in s:
                    self.vae_device.setCurrentText(str(s.get('vae_device')))
            except Exception:
                pass
            try:
                if hasattr(self,'output_path') and 'output' in s:
                    self.output_path.setText(s.get('output') or self.output_path.text())
                if hasattr(self,'show_in_player') and 'show_in_player' in s:
                    self.show_in_player.setChecked(bool(s.get('show_in_player')))
                if hasattr(self,'use_queue') and 'use_queue' in s:
                    self.use_queue.setChecked(bool(s.get('use_queue')))
            except Exception:
                pass
            # ## enforce_seed_and_size_from_saved — robustly restore tri‑state seed policy and size
            try:
                sp = str((s.get('seed_policy') or '')).strip().lower()
                if hasattr(self, 'seed_policy') and sp:
                    idx = {'fixed':0,'random':1,'increment':2}.get(sp, 0)
                    try:
                        self.seed_policy.blockSignals(True)
                    except Exception:
                        pass
                    self.seed_policy.setCurrentIndex(int(idx))
                    try:
                        self.seed_policy.blockSignals(False)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                w = s.get('width'); h = s.get('height'); spi = s.get('size_preset_index')
                if hasattr(self, 'size_combo'):
                    try:
                        if spi is not None:
                            spi = int(spi)
                            if 0 <= spi < self.size_combo.count():
                                self.size_combo.blockSignals(True)
                                self.size_combo.setCurrentIndex(spi)
                                self.size_combo.blockSignals(False)
                        if w is not None and h is not None and hasattr(self,'_size_presets'):
                            m = -1
                            for i,(label,wv,hv) in enumerate(self._size_presets):
                                if int(w)==int(wv) and int(h)==int(hv):
                                    m = i; break
                            if m >= 0:
                                self.size_combo.blockSignals(True)
                                self.size_combo.setCurrentIndex(m)
                                self.size_combo.blockSignals(False)
                    except Exception:
                        pass
                try:
                    if w is not None and hasattr(self,'size_manual_w'): self.size_manual_w.setValue(int(w))
                    if h is not None and hasattr(self,'size_manual_h'): self.size_manual_h.setValue(int(h))
                except Exception:
                    pass
            except Exception:
                pass

    
    def _get_settings_path(self):
        """Return Path to presets/setsave/txt2img.json under the app root."""
        from pathlib import Path as _P
        # Try several roots: app root (helpers/..), script dir, then CWD
        roots = []
        try:
            roots.append(_P(__file__).resolve().parent.parent)  # likely app root above helpers/
        except Exception:
            pass
        try:
            roots.append(_P(__file__).resolve().parent)  # script dir
        except Exception:
            pass
        roots.append(_P.cwd())
        for r in roots:
            try:
                target = (r / "presets" / "setsave")
                target.mkdir(parents=True, exist_ok=True)
                return target / "txt2img.json"
            except Exception:
                continue
        # Fallback to CWD without mkdir (last resort)
        return _P("./presets/setsave/txt2img.json")

    fileReady = Signal(str)

    def _settings_path(self):
        """Return Path to presets/setsave/txt2img.json; ensure parent exists."""
        from pathlib import Path as _P
        p = _P("presets")/"setsave"/"txt2img.json"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p


    def __init__(self, app_window):
        super().__init__(parent=app_window)
        self.app_window = app_window
        self.setObjectName("Txt2ImgPane")
        self._build_ui()

        # Wire actions
        self.add_to_queue.clicked.connect(lambda: self._enqueue(run_now=False))
        self.add_and_run.clicked.connect(lambda: self._enqueue(run_now=True))
        self.generate_now.clicked.connect(self._on_generate_clicked)
        try:
            self.add_to_queue.hide()
            self.add_and_run.hide()
        except Exception:
            pass
        
        # Busy animation + FS watcher
        try:
            self._busy_timer = QTimer(self)
            self._busy_timer.setInterval(120)
            self._busy_timer.timeout.connect(self._on_busy_tick)
            self._busy_frames = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏']
            self._busy_idx = 0
            self._busy_active = False
            self._busy_fs_timer = QTimer(self)
            self._busy_fs_timer.setInterval(200)
            self._busy_fs_timer.timeout.connect(self._on_busy_fs_tick)
            self._busy_watch_dir = None
            self._busy_watch_t0 = 0.0
            self._busy_watch_single = False
        except Exception:
            pass




        # Defaults
        self.use_queue.setChecked(False)  # default removed; will restore via UI
        self.show_in_player.setChecked(True)
        self.size_manual_w.setValue(768)
        self.size_manual_h.setValue(768)
        try:
            self.sampler.setCurrentText("DPM++ 2M (Karras)")
        except Exception:
            pass  # default removed; will restore via UI
        self.preset_combo.addItems(["Street Photography","Fantasy Realism","Cinematic / Moody","Product / Clean","Landscape / Golden Hour","Neon Cyberpunk","Anime Clean","Isometric 3D","Macro / Shallow DOF","Illustration / Watercolor","Realistic Portrait Pro","Outdoor Documentary","Food Styling / Clean","Architectural Interior","Fashion Illustration","Portrait / Soft"])
        self.preset_combo.setCurrentIndex(0)
        try:
            self.preset_combo.currentTextChanged.connect(self._apply_preset)
        except Exception:
            pass
        # FrameVision hard defaults
        try:
            if hasattr(self, 'sampler'):
                self.sampler.setCurrentText("DPM++ 2M (Karras)")
        except Exception:
            pass
        try:
            if hasattr(self, 'size_combo') and self.size_combo.count() > 0:
                _idx = -1
                for _i in range(self.size_combo.count()):
                    _label = self.size_combo.itemText(_i) or ''
                    if '768x768' in _label:
                        _idx = _i; break
                if _idx >= 0:
                    self.size_combo.setCurrentIndex(_idx)
        except Exception:
            pass
        try:
            if hasattr(self, 'size_manual_w'): self.size_manual_w.setValue(768)
            if hasattr(self, 'size_manual_h'): self.size_manual_h.setValue(768)
        except Exception:
            pass
        try:
            if hasattr(self, 'seed_policy'):
                # Prefer setCurrentText, fall back to search
                try:
                    self.seed_policy.setCurrentText('Random')
                except Exception:
                    _rid=-1
                    for _i in range(self.seed_policy.count()):
                        if 'Random' in (self.seed_policy.itemText(_i) or ''):
                            _rid = _i; break
                    if _rid >= 0:
                        self.seed_policy.setCurrentIndex(_rid)
        except Exception:
            pass


        # Load saved settings last, then wire autosave
        try:
            self._load_settings()
        except Exception:
            pass
        # Hi-res helper is hidden by default; force it off while hidden
        try:
            if hasattr(self, 'hires_helper') and self.hires_helper is not None:
                self.hires_helper.setChecked(False)
        except Exception:
            pass
        try:
            self._connect_autosave()
        except Exception:
            pass

        # Hotkey (Ctrl+Enter) only if QShortcut available
        try:
            # Only create the shortcut once we actually have a QApplication instance.
            app = QApplication.instance()
            if QShortcut is not None and app is not None:
                QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self._on_generate_clicked)
        except Exception:
            pass

# --- Lightweight ETA timer (time-based, no per-step) ---
        try:
            self._eta_timer = QTimer(self)
            self._eta_timer.setInterval(500)
            self._eta_timer.timeout.connect(self._on_eta_tick)
            self._eta_active = False
            self._eta_done = False
            self._eta_t0 = 0.0
            self._eta_est_total = 0.0
        except Exception:
            pass



        # --- Enforce FrameVision defaults every launch (user can still change) ---
        try:
            if hasattr(self, "show_in_player"):
                self.show_in_player.setChecked(True)
        except Exception:
            pass
        try:
            if hasattr(self, "sampler"):
                self.sampler.setCurrentText("DPM++ 2M (Karras)")
        except Exception:
            pass
        try:
            # Prefer combo 768x768 if present; also set manual fields.
            if hasattr(self, "size_combo") and self.size_combo.count() > 0:
                idx = -1
                for i in range(self.size_combo.count()):
                    label = self.size_combo.itemText(i) or ""
                    if "768x768" in label:
                        idx = i; break
                if idx >= 0:
                    self.size_combo.setCurrentIndex(idx)
            if hasattr(self, "size_manual_w"):
                self.size_manual_w.setValue(768)
            if hasattr(self, "size_manual_h"):
                self.size_manual_h.setValue(768)
        except Exception:
            pass
        try:
            # Seed policy default -> Random
            if hasattr(self, "seed_policy"):
                try:
                    self.seed_policy.setCurrentText("Random")
                except Exception:
                    idx = -1
                    for i in range(self.seed_policy.count()):
                        if "Random" in (self.seed_policy.itemText(i) or ""):
                            idx = i; break
                    if idx >= 0:
                        self.seed_policy.setCurrentIndex(idx) if sp is not None else None
        except Exception:
            pass
    def _apply_preset(self, name: str):
        # Presets only apply to SD (diffusers) engine; ignore for Z-Image and Qwen
        try:
            ek = self._engine_key_selected() if hasattr(self, '_engine_key_selected') else 'diffusers'
        except Exception:
            ek = 'diffusers'
        if str(ek).lower().strip() != 'diffusers':
            return
        presets = {
            "Fashion Illustration": {"sampler":"DPM++ 2M (Karras)","steps":32,"cfg":5.5,"size":(768,768),
                                  "neg":"blurry, watermark, logo, text"},
            "Portrait / Soft": {"sampler":"DPM++ 2M (Karras)","steps":34,"cfg":5.0,"size":(1024,1536),
                                "neg":"harsh shadows, waxy skin, low-detail"},
            "Cinematic / Moody": {"sampler":"Heun","steps":30,"cfg":5.5,"size":(1536,864),
                                  "neg":"flat lighting"},
            "Product / Clean": {"sampler":"UniPC","steps":28,"cfg":5.0,"size":(1024,1024),
                                "neg":"dirt, scratches, fingerprints"},
            "Landscape / Golden Hour": {"sampler":"DPM++ 2M (Karras)","steps":32,"cfg":5.5,"size":(1536,864),
                                        "neg":"haze, oversaturated"},
            "Neon Cyberpunk": {"sampler":"Euler a","steps":28,"cfg":5.5,"size":(1280,720),
                               "neg":"banding, color fringing"},
            "Anime Clean": {"sampler":"Euler a","steps":24,"cfg":7.5,"size":(768,1152),
                            "neg":"photorealistic skin, photo texture"},
            "Isometric 3D": {"sampler":"UniPC","steps":30,"cfg":5.5,"size":(1024,1024),
                             "neg":"photo grain, camera blur"},
            "Macro / Shallow DOF": {"sampler":"DPM++ 2M (Karras)","steps":34,"cfg":5.5,"size":(1024,1024),
                                    "neg":"motion blur"},
            "Illustration / Watercolor": {"sampler":"DDIM","steps":28,"cfg":5.0,"size":(1024,1536),
                                          "neg":"harsh contrast"},
            "Realistic Portrait Pro": {"sampler":"DPM++ 2M (Karras)","steps":36,"cfg":5.0,"size":(896,1152),
                                       "neg":"overexposed skin, plastic texture"},
            "Outdoor Documentary": {"sampler":"Heun","steps":28,"cfg":5.0,"size":(1536,864),
                                    "neg":"over-processed, HDR halos"},
            "Food Styling / Clean": {"sampler":"UniPC","steps":30,"cfg":6.0,"size":(1024,1024),
                                     "neg":"grease, messy crumbs, dull color"},
            "Architectural Interior": {"sampler":"DPM++ 2M (Karras)","steps":32,"cfg":5.5,"size":(1280,720),
                                       "neg":"distorted lines, crooked walls"},
            "Fantasy Realism": {"sampler":"Euler a","steps":32,"cfg":6.5,"size":(1024,1024),
                                "neg":"cartoonish, low-detail"},
            "Street Photography": {"sampler":"Heun","steps":30,"cfg":5.0,"size":(1152,768),
                                   "neg":"motion smear, excessive noise"}
        }
        cfg = presets.get(name)
        if not cfg:
            return
        try:
            self.sampler.setCurrentText(cfg["sampler"])
        except Exception:
            pass
        try:
            self.steps_slider.setValue(int(cfg["steps"]))
        except Exception:
            pass
        try:
            self.cfg_scale.setValue(float(cfg["cfg"]))
        except Exception:
            pass
        try:
            w,h = cfg["size"]
            if hasattr(self, "size_combo"):
                try:

                    idx = -1

                    for i in range(self.size_combo.count()):

                        d = self.size_combo.itemData(i)

                        if d and int(d[0]) == int(w) and int(d[1]) == int(h):

                            idx = i

                            break

                    if idx >= 0:

                        self.size_combo.blockSignals(True)

                        self.size_combo.setCurrentIndex(idx)

                        self.size_combo.blockSignals(False)

                    else:

                        self.size_combo.setCurrentText(f"{w}x{h} (1:1)" if w==h else f"{w}x{h}")

                except Exception:
                     pass
            if hasattr(self, "size_manual_w"): self.size_manual_w.setValue(w)
            if hasattr(self, "size_manual_h"): self.size_manual_h.setValue(h)
        except Exception:
            pass
        try:
            if self.negative.toPlainText().strip() == "":
                self.negative.setPlainText(cfg["neg"])
        except Exception:
            pass
    def _current_output_dir(self):
        """Best-effort: return Path for the last chosen txt2img output folder."""
        from pathlib import Path as _Path
        try:
            p = None
            try:
                text = self.output_path.text().strip()
                if text:
                    p = _Path(text).expanduser()
            except Exception:
                p = None
            if not p:
                return None
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return p
        except Exception:
            return None

    def _open_output_folder(self):
        """Open the current output folder in the OS file explorer."""
        try:
            out_dir = self._current_output_dir()
        except Exception:
            out_dir = None
        if out_dir is None:
            return
        try:
            from PySide6.QtGui import QDesktopServices
            from PySide6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out_dir)))
        except Exception:
            try:
                import os, subprocess, sys
                path = str(out_dir)
                if os.name == 'nt':
                    os.startfile(path)  # type: ignore[attr-defined]
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', path])
                else:
                    subprocess.Popen(['xdg-open', path])
            except Exception:
                pass

    def _open_output_in_media_explorer(self):
        """Jump to Media Explorer tab and scan the current output folder."""
        try:
            out_dir = self._current_output_dir()
        except Exception:
            out_dir = None

        if out_dir is None:
            return

        # Preferred: route through the main window so any tab can reuse it.
        try:
            mw = getattr(self, "app_window", None)
        except Exception:
            mw = None

        try:
            if mw is not None and hasattr(mw, "open_media_explorer_folder"):
                # txt2img always outputs images
                mw.open_media_explorer_folder(str(out_dir), preset="images", include_subfolders=False)
                return
        except Exception:
            pass

        # Fallback: open in OS explorer if the routing helper isn't available.
        try:
            self._open_output_folder()
        except Exception:
            pass


    # --- SDXL model safety (optional downloads helper) ---
    def _app_root(self):
        """Return the app root (folder above /helpers)."""
        try:
            from pathlib import Path as _P
            return _P(__file__).resolve().parent.parent
        except Exception:
            from pathlib import Path as _P
            return _P.cwd()

    def _sdxl_models_installed(self) -> bool:
        """True if root/models/sdxl (or SDXL) contains a real SDXL model file (>=2 GiB)."""
        try:
            root = self._app_root()
            candidates = [root / "models" / "sdxl", root / "models" / "SDXL"]
            min_bytes = 2 * 1024 * 1024 * 1024
            for d in candidates:
                try:
                    if not (d.exists() and d.is_dir()):
                        continue
                    # Some installs place files in subfolders; scan recursively but bail fast.
                    for p in d.rglob("*"):
                        try:
                            if p.is_file() and p.stat().st_size >= min_bytes:
                                return True
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _engine_key_selected(self) -> str:
        """Normalize the selected backend key even if combo userData is unavailable."""
        key = ""
        try:
            if hasattr(self, "engine_combo"):
                try:
                    data = self.engine_combo.currentData()
                except Exception:
                    data = None
                try:
                    text = self.engine_combo.currentText()
                except Exception:
                    text = ""
                if data is not None and str(data).strip():
                    key = str(data).lower()
                else:
                    key = str(text or "").lower()
        except Exception:
            key = ""

        if ("zimage" in key) or ("z-image" in key) or ("z image" in key):
            if "gguf" in key:
                return "zimage_gguf"
            return "zimage"

        if ("qwen" in key) and (("2512" in key) or ("2.5" in key) or ("12b" in key)):
            return "qwen2512"

        if ("diffusers" in key) or ("sd models" in key) or ("sd15" in key) or ("sdxl" in key):
            return "diffusers"

        return key or "diffusers"

    def _zimage_turbo_models_installed(self) -> bool:
        """True if root/models/Z-Image-Turbo contains at least one file."""
        try:
            root = self._app_root()
            d = (root / "models" / "Z-Image-Turbo")
            if not (d.exists() and d.is_dir()):
                return False
            for p in d.rglob("*"):
                try:
                    if p.is_file():
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _zimage_gguf_models_installed(self) -> bool:
        """True if root/models/Z-Image-Turbo GGUF contains a .gguf (or any file) >=3 GiB."""
        try:
            root = self._app_root()
            d = (root / "models" / "Z-Image-Turbo GGUF")
            if not (d.exists() and d.is_dir()):
                return False
            min_bytes = 3 * 1024 * 1024 * 1024
            for p in d.rglob("*"):
                try:
                    if p.is_file() and p.stat().st_size >= min_bytes:
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    
    def _qwen2512_models_installed(self) -> bool:
        """True if root/models/Qwen-Image-2512 GGUF contains a .gguf (or any file) >=3 GiB."""
        try:
            root = self._app_root()
            d = (root / "models" / "Qwen-Image-2512 GGUF")
            if not (d.exists() and d.is_dir()):
                return False
            min_bytes = 3 * 1024 * 1024 * 1024
            for p in d.rglob("*"):
                try:
                    if p.is_file() and p.stat().st_size >= min_bytes:
                        return True
                except Exception:
                    continue
        except Exception:
            return False
        return False

    def _is_sdxl_selected(self) -> bool:
        """Best-effort: detect SDXL selection from model path/label."""
        try:
            mp = ""
            try:
                if hasattr(self, "model_combo"):
                    mp = str(self.model_combo.currentData() or "")
            except Exception:
                mp = ""
            txt = ""
            try:
                if hasattr(self, "model_combo"):
                    txt = str(self.model_combo.currentText() or "")
            except Exception:
                txt = ""
            s = (mp + " " + txt).lower()
            return ("sdxl" in s)
        except Exception:
            return False

    def _open_optional_installs_file(self):
        """Open helpers/opt_installs.py in the user's default editor."""
        try:
            from pathlib import Path as _P
            root = self._app_root()
            target = root / "helpers" / "opt_installs.py"
            if not target.exists():
                target = _P.cwd() / "helpers" / "opt_installs.py"
            try:
                from PySide6.QtGui import QDesktopServices
                from PySide6.QtCore import QUrl
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
                return
            except Exception:
                pass
            try:
                import os, sys, subprocess
                p = str(target)
                if os.name == "nt":
                    os.startfile(p)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", p])
                else:
                    subprocess.Popen(["xdg-open", p])
            except Exception:
                pass
        except Exception:
            pass

    def _ensure_models_installed(self) -> bool:
        """Return False if blocked (missing required models), True otherwise."""
        try:
            ek = ""
            try:
                ek = self._engine_key_selected()
            except Exception:
                ek = ""
            ek = (ek or "").lower()

            # Z-Image Turbo (full) requires at least one file in models/Z-Image-Turbo
            if ek == "zimage":
                if self._zimage_turbo_models_installed():
                    return True
                msg = (
                    "Models are not installed yet, please select 'Z-image Turbo Text to Image' from the "
                    "'optional downloads' menu to download the correct model for this tool"
                )
                title = "Z-Image Turbo models missing"

            # Z-Image GGUF requires a >=3 GiB file in models/Z-Image-Turbo GGUF
            elif ek == "zimage_gguf":
                if self._zimage_gguf_models_installed():
                    return True
                msg = (
                    "Models are not installed yet, please select 'Z-Image Turbo GGUF' from the "
                    "'optional downloads' menu to download the correct model for this tool"
                )
                title = "Z-Image GGUF models missing"

            # qwen 2.5 12B GGUF requires a >=3 GiB file in models/Qwen-Image-2512 GGUF
            elif ek == "qwen2512":
                if self._qwen2512_models_installed():
                    return True
                msg = (
                    "Models are not installed yet, please select 'qwen 2.5 12B GGUF' from the "
                    "'optional downloads' menu to download the correct model for this tool"
                )
                title = "qwen 2.5 12B GGUF models missing"

            # Diffusers SD engine: only block when SDXL is selected and the folder lacks a real model file (>=2 GiB)
            else:
                need_sdxl = False
                try:
                    need_sdxl = bool(self._is_sdxl_selected())
                except Exception:
                    need_sdxl = False
                # If the SD model list is empty, assume the user is trying to use SDXL and show the installer hint.
                if not need_sdxl:
                    try:
                        if hasattr(self, "model_combo") and self.model_combo is not None:
                            if int(self.model_combo.count()) <= 0:
                                need_sdxl = True
                    except Exception:
                        pass
                if not need_sdxl:
                    return True
                if self._sdxl_models_installed():
                    return True
                msg = (
                    "Models are not installed yet, please select 'SDXL model' from the 'optional downloads' menu "
                    "to download the correct model for this tool"
                )
                title = "SDXL models missing"

            try:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Information)
                box.setWindowTitle(title)
                box.setText(msg)
                box.setStandardButtons(QMessageBox.Ok)
                box.setDefaultButton(QMessageBox.Ok)
                box.exec()
            except Exception:
                try:
                    QMessageBox.information(self, title, msg)
                except Exception:
                    pass

            # Open optional installs helper on OK
            try:
                self._open_optional_installs_file()
            except Exception:
                pass
            return False
        except Exception:
            return True

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10,10,10,10)
        outer.setSpacing(8)

        # Fancy purple banner at the top
        self.banner = QLabel("Text to Image with SDXL Loader")
        try:
            self._banner_default_text = self.banner.text()
        except Exception:
            self._banner_default_text = "Text to Image with SDXL Loader"
        self.banner.setObjectName("txt2imgBanner")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.banner.setFixedHeight(45)
        self.banner.setStyleSheet(
            "#txt2imgBanner {"
            " font-size: 15px;"
            " font-weight: 600;"
            " padding: 8px 17px;"
            " border-radius: 12px;"
            " margin: 0 0 6px 0;"
            " color: white;"
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #7e3cff,"
            "   stop:0.5 #a64dff,"
            "   stop:1 #c27aff"
            " );"
            " letter-spacing: 0.5px;"
            "}"
        )
        outer.addWidget(self.banner)
        outer.addSpacing(4)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll, 1)
        container = QWidget(); scroll.setWidget(container)
        root = QVBoxLayout(container)
        # Compact combo boxes so long items don't force a huge minimum width
        def _compact_combo(cb, min_chars=10):
            try:
                cb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            except Exception:
                pass
            try:
                cb.setMinimumContentsLength(int(min_chars))
            except Exception:
                pass
            try:
                cb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            except Exception:
                pass


        # Engine / backend selector
        engine_row = QHBoxLayout()
        self.engine_combo = QComboBox()
        _compact_combo(self.engine_combo, 10)
        try:
            self.engine_combo.addItem("SD models (SD15/SDXL)", "diffusers")
            self.engine_combo.addItem("Z-Image Turbo", "zimage")
            self.engine_combo.addItem("Z-Image Turbo (GGUF Low VRAM)", "zimage_gguf")
            self.engine_combo.addItem("Qwen 2.5 12B GGUF (Low VRAM)", "qwen2512")
        except Exception:
            # Fallback without userData support
            self.engine_combo.addItem("SD models (SD15/SDXL)")
            self.engine_combo.addItem("Z-Image Turbo")
            self.engine_combo.addItem("Z-Image Turbo (GGUF Low VRAM)")
            self.engine_combo.addItem("qwen 2.5 12B GGUF (Low VRAM)")
        self.engine_label = QLabel("Engine:")
        engine_row.addWidget(self.engine_label)
        engine_row.addWidget(self.engine_combo, 1)
        try:
            root.insertLayout(0, engine_row)
        except Exception:
            root.addLayout(engine_row)

        # Presets and chips
        top = QHBoxLayout()
        self.preset_label = QLabel("Preset:")
        self.preset_combo = QComboBox()
        _compact_combo(self.preset_combo, 12)
        self.style_builder_btn = QPushButton("Style Builder")
        self.style_builder_btn.hide()
        top.addWidget(self.preset_label)
        top.addWidget(self.preset_combo, 1)
        top.addWidget(self.style_builder_btn, 0)
        root.addLayout(top)

        # Prompts
        form = QFormLayout()
        self.prompt = QTextEdit(); self.prompt.setPlaceholderText("Describe the image you want…"); self.prompt.setFixedHeight(124)
        self.negative = QTextEdit(); self.negative.setPlaceholderText("What to avoid (optional)…"); self.negative.setFixedHeight(48)
        form.addRow("Prompt", self.prompt)

        # Prompt helper row (Enhance + Clear) between prompt and negatives
        prompt_btn_row = QHBoxLayout()
        self.btn_prompt_enhance = QPushButton("Enhance prompt (Qwen)")
        try:
            self.btn_prompt_enhance.setToolTip("Expand this prompt with the Qwen3-VL prompt helper (running in its own .venv) to get a longer, more varied description. Handy when Z-Image keeps giving the same face.")
        except Exception:
            pass
        try:
            self.btn_prompt_enhance.clicked.connect(self._on_enhance_prompt_clicked)
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

            # Prompt enhancer presets (same preset names as Tools → Prompt enhancement)
            try:
                self.combo_prompt_preset = QComboBox()
                try:
                    _compact_combo(self.combo_prompt_preset, 12)
                except Exception:
                    pass
                try:
                    self.combo_prompt_preset.setToolTip("Prompt enhancer preset (same list as Tools → Prompt enhancement).")
                except Exception:
                    pass
                try:
                    self.combo_prompt_preset.currentIndexChanged.connect(self._on_prompt_preset_changed)
                except Exception:
                    pass
                want = "Default"
                try:
                    want = str(QSettings('FrameVision','FrameVision').value("txt2img_prompt_preset", "Default") or "Default")
                except Exception:
                    want = "Default"
                try:
                    self._rebuild_prompt_preset_combo(want)
                except Exception:
                    try:
                        self.combo_prompt_preset.addItems(["Default"])
                        self.combo_prompt_preset.setCurrentText("Default")
                    except Exception:
                        pass
                prompt_btn_row.addWidget(self.combo_prompt_preset)
            except Exception:
                self.combo_prompt_preset = None

            prompt_btn_row.addWidget(self.btn_prompt_clear)
            prompt_btn_row.addStretch(1)
            prompt_btn_wrap = QWidget(self)
            prompt_btn_wrap.setLayout(prompt_btn_row)
            form.addRow("", prompt_btn_wrap)
        except Exception:
            pass

        form.addRow("Negative", self.negative)

        # Z-Image: optional image-to-image (init image) to help keep the same person/composition
        try:
            self.zimg_init_enable = QCheckBox("Use init image (img2img)")
            try:
                self.zimg_init_enable.setToolTip("Optional (FP16 only): provide an initial image and Z-Image will generate a variation of it. Useful for keeping the same person.")
            except Exception:
                pass
            self.zimg_init_path = QLineEdit()
            try:
                self.zimg_init_path.setPlaceholderText("Choose an image (optional)…")
            except Exception:
                pass
            self.zimg_init_browse = QPushButton("Browse")
            self.zimg_init_clear = QPushButton("Clear")
            self.zimg_strength = QDoubleSpinBox()
            try:
                self.zimg_strength.setRange(0.0, 1.0)
                self.zimg_strength.setSingleStep(0.05)
                self.zimg_strength.setDecimals(2)
                self.zimg_strength.setValue(0.35)
                self.zimg_strength.setToolTip("Denoising strength: lower = closer to the input image; higher = more change.")
            except Exception:
                pass

            def _zimg_i2i_sync_enabled(*_):
                try:
                    en = bool(self.zimg_init_enable.isChecked())
                except Exception:
                    en = False
                for _w in (getattr(self, "zimg_init_path", None),
                           getattr(self, "zimg_init_browse", None),
                           getattr(self, "zimg_init_clear", None),
                           getattr(self, "zimg_strength", None)):
                    try:
                        if _w is not None:
                            _w.setEnabled(en)
                    except Exception:
                        pass

            try:
                self.zimg_init_enable.toggled.connect(_zimg_i2i_sync_enabled)
            except Exception:
                pass

            def _zimg_pick_init_image():
                try:
                    fn, _ = QFileDialog.getOpenFileName(
                        self,
                        "Select init image",
                        "",
                        "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
                    )
                except Exception:
                    fn = ""
                if fn:
                    try:
                        self.zimg_init_path.setText(fn)
                        self.zimg_init_enable.setChecked(True)
                    except Exception:
                        pass

            def _zimg_clear_init_image():
                try:
                    self.zimg_init_path.setText("")
                except Exception:
                    pass

            try:
                self.zimg_init_browse.clicked.connect(_zimg_pick_init_image)
                self.zimg_init_clear.clicked.connect(_zimg_clear_init_image)
            except Exception:
                pass

            zimg_i2i_wrap = QWidget()
            zimg_i2i_row = QHBoxLayout(zimg_i2i_wrap)
            zimg_i2i_row.setContentsMargins(0, 0, 0, 0)
            zimg_i2i_row.addWidget(self.zimg_init_enable, 0)
            zimg_i2i_row.addWidget(self.zimg_init_path, 1)
            zimg_i2i_row.addWidget(QLabel("Strength"), 0)
            zimg_i2i_row.addWidget(self.zimg_strength, 0)
            zimg_i2i_row.addWidget(self.zimg_init_browse, 0)
            zimg_i2i_row.addWidget(self.zimg_init_clear, 0)

            self.zimg_i2i_label = QLabel("Init image")
            self.zimg_i2i_wrap = zimg_i2i_wrap
            form.addRow(self.zimg_i2i_label, self.zimg_i2i_wrap)

            _zimg_i2i_sync_enabled()
        except Exception:
            # UI is best-effort; older builds may not have all widgets available
            self.zimg_i2i_label = None
            self.zimg_i2i_wrap = None

        # Seed + seed policy (Batch moved next to Generate button)
        row = QHBoxLayout()
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(0)
        try:
            self.seed.setMaximumWidth(140)
        except Exception:
            pass
        self.seed_policy = QComboBox(); self.seed_policy.addItems(["Fixed (use seed)", "Random", "Increment"]); self.seed_policy.setCurrentIndex(1)
        try:
            _compact_combo(self.seed_policy, 8)
        except Exception:
            pass
        self.batch = QSpinBox(); self.batch.setRange(1, 64); self.batch.setValue(1)
        try:
            self.batch.setMaximumWidth(90)
        except Exception:
            pass
        row.addWidget(QLabel("Seed:")); row.addWidget(self.seed)
        row.addWidget(QLabel("")); row.addWidget(self.seed_policy)
        form.addRow(row)

        # --- Quality / Output Size (next to seed) ---
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Output size:"))
        self.size_combo = QComboBox()
        _compact_combo(self.size_combo, 12)
        # Common buckets across aspects
        self._size_presets = [
            # 1:1
            ("320x320 (1:1)", 320, 320),
            ("480x480 (1:1)", 480, 480),
            ("512x512 (1:1)", 512, 512),
            ("640x640 (1:1)", 640, 640),
            ("768x768 (1:1)", 768, 768),
            ("896x896 (1:1)", 896, 896),
            ("960x960 (1:1)", 960, 960),
            ("1024x1024 (1:1)", 1024, 1024),
            ("1280x1280 (1:1, Max recommended SDXL)", 1280, 1280),
            ("1440x1440 (1:1)", 1440, 1440),
            ("1920x1920 (1:1)", 1920, 1920),
            ("2048x2048 (1:1)", 2048, 2048),
            ("3072x3072 (1:1)", 3072, 3072),
            ("4096x4096 (1:1)", 4096, 4096),
            # 16:9/9:16
            ("640x384 (16:9)", 640, 384),
            ("1024x576 (16:9)", 1024, 576),
            ("854x480 (16:9)", 854, 480),
            ("1024x576 (16:9)", 1024, 576),
            ("1136x640 (16:9)", 1136, 640),
            ("1280x720 (16:9)", 1280, 720),
            ("1280x704 (16:9, WAN specific size)", 1280, 704),
            ("1344x768 (16:9)", 1344, 768),
            ("1536x864 (16:9, max advised for SDXL)", 1536, 864),
            ("1792x1008 (16:9)", 1792, 1008),
            ("1920x1080 (16:9), Qwen2512 only", 1920, 1080),
            ("1920x1088 (16:9)", 1920, 1088),
            ("2048x1152 (16:9)", 2048, 1152),
            ("2560x1440 (16:9)", 2560, 1440),
            ("4096x2560 (16:9)", 4096, 2560),
            ("384x640 (9:16)", 384, 640),
            ("576x1024 (9:16)", 576, 1024),
            ("640x1136 (9:16)", 640, 1136),
            ("720x1280 (9:16)", 720, 1280),
            ("704x1280 (9:16, WAN specific size)", 704, 1280),
            ("864x1536 (9:16)", 864, 1536),
            ("768x1344 (9:16)", 768, 1344),
            ("864x1536 (9:16, max advised for SDXL)", 864, 1536),
            ("972x1728 (9:16)", 972, 1728),
            ("1088x1920 (9:16)", 1088, 1920),
            ("1080x1920 (9:16), qwen 2512 only", 1080, 1920),
            ("1152x2048 (9:16)", 1152, 2048),
            ("1440x2560 (9:16)", 1440, 2560),
            ("2560x4096 (9:16)", 2560, 4096),
            
            # 21:9/9:21
            ("1280x544 (21:9)", 1280, 544),
            ("1344x576 (21:9, max advised for SDXL)", 1344, 576),
            ("1600x684 (21:9)", 1600, 684),
            ("1920x800 (21:9, ultrawide)", 1920, 800),
            ("2560x1088 (21:9, ultrawide)", 2560, 1088),

            # 9:7/7:9
            ("896x1152 (7:9)", 896, 1152),
            ("1152x896 (9:7)", 1152, 896),

            # 4:3/3:4
            ("600x800 (3:4)", 600, 800),
            ("800x600 (4:3)", 800, 600),
            ("896x672 (4:3)", 896, 672),
            ("768x1024 (3:4)", 768, 1024),
            ("1024x768 (4:3)", 1024, 768),
            ("832x1104 (3:4)", 832, 1104),
            ("1104x832 (4:3)", 1104, 832),
            ("1152x864 (4:3, max advised for SDXL)", 1152, 864),

            # 3:2/2:3
            ("960x640 (3:2)", 960, 640),
            ("768x1152 (2:3)", 768, 1152),
            ("1152x768 (3:2)", 1152, 768),
            ("1024x1536 (2:3, max advised for SDXL)", 1024, 1536),
            ("1536x1024 (3:2, max advised for SDXL)", 1536, 1024),

            # 2:1/1:2
            ("1024x512 (2:1)", 1024, 512),
            ("1280x640 (2:1, recommended)", 1280, 640),
        ]

        for label, w, h in self._size_presets:
            self.size_combo.addItem(label, (w, h))
        # default selection
        idx_default = next((i for i,(lbl,_,__) in enumerate(self._size_presets) if "768x768" in lbl), 0)  # 1280x720 default
        if 0 <= idx_default < self.size_combo.count():
            self.size_combo.setCurrentIndex(idx_default)
        # Optional manual override (advanced later; present but small)
        self.size_manual_w = QSpinBox(); self.size_manual_w = QSpinBox(); self.size_manual_w.setRange(256, 2560)
        try:
            self.size_manual_w.setMaximumWidth(95)
        except Exception:
            pass
        self.size_manual_h = QSpinBox(); self.size_manual_h.setRange(256, 2560); self.size_manual_h.setSingleStep(64)
        try:
            self.size_manual_h.setMaximumWidth(95)
        except Exception:
            pass
        self.size_lock = QCheckBox("Lock aspect")
        self.size_lock.setChecked(False)  # default removed; will restore via UI
        def _on_size_combo_changed(i):
            data = self.size_combo.itemData(i)
            if data:
                w, h = data
                self.size_manual_w.blockSignals(True); self.size_manual_h.blockSignals(True)
                self.size_manual_w.setValue(int(w)); self.size_manual_h.setValue(int(h))
                self.size_manual_w.blockSignals(False); self.size_manual_h.blockSignals(False)
                try:
                    lbl = getattr(self, "size_warning_label", None)
                    if lbl is not None and hasattr(self, "size_manual_w") and hasattr(self, "size_manual_h"):
                        wv = int(self.size_manual_w.value()); hv = int(self.size_manual_h.value())
                        lbl.setVisible((wv * hv) > (1536 * 864))
                except Exception:
                    pass
        self.size_combo.currentIndexChanged.connect(_on_size_combo_changed)
        # Ensure manual boxes reflect default only if persistence is OFF
        try:
            if not _keep_settings_enabled():
                di = idx_default
                data = self.size_combo.itemData(di)
                if data:
                    self.size_manual_w.setValue(int(data[0])); self.size_manual_h.setValue(int(data[1]))
        except Exception:
            pass
        def _sync_manual_w(v):
            if not self.size_lock.isChecked(): return
            # keep aspect from current combo data
            data = self.size_combo.currentData()
            if not data: return
            w0, h0 = data
            aspect = h0 / w0 if w0 else 1.0
            new_h = max(256, int(round(v * aspect / 64) * 64))
            # Clamp by category maxima
            try:
                if w0 == h0:
                    v = min(int(v), 1920); new_h = min(int(new_h), 1920)
                elif w0 > h0:
                    v = min(int(v), 2560); new_h = min(int(new_h), 1440)
                else:
                    v = min(int(v), 1440); new_h = min(int(new_h), 2560)
            except Exception:
                pass
            self.size_manual_w.blockSignals(True); self.size_manual_w.setValue(int(v)); self.size_manual_w.blockSignals(False)
            self.size_manual_h.blockSignals(True); self.size_manual_h.setValue(new_h); self.size_manual_h.blockSignals(False)
            try:
                lbl = getattr(self, "size_warning_label", None)
                if lbl is not None and hasattr(self, "size_manual_w") and hasattr(self, "size_manual_h"):
                    wv = int(self.size_manual_w.value()); hv = int(self.size_manual_h.value())
                    lbl.setVisible((wv * hv) > (1536 * 864))
            except Exception:
                pass
        def _sync_manual_h(v):
            if not self.size_lock.isChecked(): return
            data = self.size_combo.currentData()
            if not data: return
            w0, h0 = data
            aspect = w0 / h0 if h0 else 1.0
            new_w = max(256, int(round(v * aspect / 64) * 64))
            # Clamp by category maxima
            try:
                if w0 == h0:
                    v = min(int(v), 1920); new_w = min(int(new_w), 1920)
                elif h0 > w0:
                    v = min(int(v), 2560); new_w = min(int(new_w), 1440)
                else:
                    v = min(int(v), 1440); new_w = min(int(new_w), 2560)
            except Exception:
                pass
            self.size_manual_h.blockSignals(True); self.size_manual_h.setValue(int(v)); self.size_manual_h.blockSignals(False)
            self.size_manual_w.blockSignals(True); self.size_manual_w.setValue(new_w); self.size_manual_w.blockSignals(False)
            try:
                lbl = getattr(self, "size_warning_label", None)
                if lbl is not None and hasattr(self, "size_manual_w") and hasattr(self, "size_manual_h"):
                    wv = int(self.size_manual_w.value()); hv = int(self.size_manual_h.value())
                    lbl.setVisible((wv * hv) > (1536 * 864))
            except Exception:
                pass
        self.size_manual_w.valueChanged.connect(_sync_manual_w)
        self.size_manual_h.valueChanged.connect(_sync_manual_h)

        size_row.addWidget(self.size_combo, 2)

        # Manual W/H + Lock aspect (advanced) — hidden by default per UX request
        # Keep widgets alive for internal size bookkeeping / persistence, but do not show them.
        self.size_manual_w_label = QLabel("W:")
        self.size_manual_h_label = QLabel("H:")
        try:
            self.size_lock.setChecked(False)
        except Exception:
            pass
        for _w in (self.size_manual_w_label, self.size_manual_w,
                   self.size_manual_h_label, self.size_manual_h,
                   self.size_lock):
            try:
                _w.setVisible(False)
            except Exception:
                pass
        # (Intentionally not added to the layout)

        form.addRow(size_row)
        self.size_warning_label = QLabel("Higher resolutions can create 'clones' in the image")
        self.size_warning_label.setWordWrap(True)
        try:
            self.size_warning_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        except Exception:
            pass
        self.size_warning_label.setVisible(False)
        form.addRow(self.size_warning_label)


        # Steps slider (right behind seed section)
        steps_row = QHBoxLayout()
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(10, 100)
        self.steps_slider.setValue(30)
        try:
            self.steps_slider.setTickPosition(QSlider.TicksBelow)
            self.steps_slider.setTickInterval(10)
        except Exception:
            pass
        self.steps_value = QLabel("25")
        self.steps_default = QCheckBox("Lock to default")
        self.steps_default.setChecked(False)
        def _on_steps_default_changed(checked: bool):
            if checked:
                self.steps_slider.setValue(25)
        self.steps_default.toggled.connect(_on_steps_default_changed)
        # Hide 'Lock to default' toggle and keep it off
        try:
            self.steps_default.setChecked(False)
            self.steps_default.setVisible(False)
        except Exception:
            pass
        self.steps_slider.valueChanged.connect(lambda v: self.steps_value.setText(str(int(v))))
        steps_row.addWidget(QLabel("Steps:"))
        steps_row.addWidget(self.steps_slider, 1)
        steps_row.addWidget(self.steps_value, 0)
        # 'Lock to default' is intentionally not added to the layout anymore
        form.addRow(steps_row)

        # CFG scale
        cfg_row = QHBoxLayout()
        self.cfg_scale = QDoubleSpinBox(); self.cfg_scale.setRange(1.0, 15.0); self.cfg_scale.setSingleStep(0.1); self.cfg_scale.setDecimals(1); self.cfg_scale.setValue(7.5)
        
        # Move CFG into the Steps row per user request
        steps_row.addWidget(QLabel("CFG:"))
        steps_row.addWidget(self.cfg_scale, 0)
        # (removed) form.addRow(cfg_row) — CFG now sits on Steps row
        # Seed used (always visible)
        self.seed_used_label = QLabel("Seed used: —"); self.seed_used_label.setVisible(False)
        form.addRow(self.seed_used_label)

        # Output path + show in player + queue toggle
        out_row = QHBoxLayout()
        self.output_path = QLineEdit(str(Path("./output/images").resolve()))
        self.browse_btn = QPushButton("Browse…"); self.browse_btn.clicked.connect(self._on_browse)
        self.show_in_player = QCheckBox("Show in Player")
        self.use_queue = QCheckBox("Use queue")
        
        try:
            self.use_queue.toggled.connect(lambda *_: self._autosave_now())
        except Exception:
            pass
        try:
            self.use_queue.toggled.connect(self._apply_queue_visibility)
        except Exception:
            pass

        out_row.addWidget(QLabel("Output:")); out_row.addWidget(self.output_path, 1); out_row.addWidget(self.browse_btn)
        form.addRow(out_row)

        # VRAM profile override (kept for jobs but hidden in UI)
        vram_row = QHBoxLayout()
        self.vram_profile = QComboBox();
        try:
            _compact_combo(self.vram_profile, 6)
        except Exception:
            pass
        self.vram_profile.addItems(["Auto", "6 GB", "8 GB", "12 GB", "24 GB"])
        self.restore_auto = QPushButton("Restore Auto")
        self.restore_auto.hide(); self.restore_auto.clicked.connect(lambda: self.vram_profile.setCurrentIndex(0))
        self.vram_label = QLabel("VRAM profile:")
        vram_row.addWidget(self.vram_label); vram_row.addWidget(self.vram_profile); vram_row.addWidget(self.restore_auto)
        
        vram_row.addWidget(self.show_in_player)
        vram_row.addWidget(self.use_queue)
        form.addRow(vram_row)
        # Hide VRAM controls from the UI, keep underlying value for internal use
        try:
            self.vram_label.hide()
            self.vram_profile.hide()
            self.restore_auto.hide()
        except Exception:
            pass
        # Ensure hidden CLI fields exist before persistence restore
        try:
            self._qwen_cli_template_removed
        except Exception:
            self._qwen_cli_template_removed = QLineEdit('qwen_image --model "{model}" --prompt "{prompt}" --negative "{neg}" --w {w} --h {h} --steps {steps} --seed {seed} --out "{out}"')
            self._qwen_cli_template_removed.setVisible(False)
        try:
            self._a1111_url_removed
        except Exception:
            self._a1111_url_removed = QLineEdit("http://127.0.0.1:7860")
            self._a1111_url_removed.setVisible(False)

        # Ensure shared QSettings store exists (for backends, LoRA path, etc.)
        try:
            self._persist_settings
        except Exception:
            try:
                self._persist_settings = QSettings('FrameVision','FrameVision')
            except Exception:
                self._persist_settings = QSettings()

        # Determine default LoRA root folder (relative to app root)
        def _default_lora_root():
            try:
                from pathlib import Path as _P
                base = _P(__file__).resolve().parent.parent
            except Exception:
                from pathlib import Path as _P
                base = _P.cwd()
            return str((base / "loras" / "txt2img").resolve())

        # Restore persisted LoRA root if any; otherwise fall back to default
        try:
            _lr = self._persist_settings.value("txt2img_lora_root", None, type=str)
        except Exception:
            _lr = None
        if not _lr:
            try:
                _lr = _default_lora_root()
            except Exception:
                _lr = "./loras/txt2img"
        self._lora_root = _lr

        # Restore persisted UI values
        try:
            v = None
            if v is not None:
                            v = self._persist_settings.value("a1111_url")
            if v: self._a1111_url_removed.setText(v)
            v = self._persist_settings.value("qwen_cli_template")
            if v: self._qwen_cli_template_removed.setText(v)
        except Exception:
            pass

        # Save on change
        def _save_backend():
            try:
                self._persist_settings.setValue("backend_text", "Qwen CLI")
                self._persist_settings.setValue("a1111_url", self._a1111_url_removed.text().strip())
                self._persist_settings.setValue("qwen_cli_template", self._qwen_cli_template_removed.text().strip())
            except Exception:
                pass
        self._a1111_url_removed.textChanged.connect(lambda *_: _save_backend())
        self._qwen_cli_template_removed.textChanged.connect(lambda *_: _save_backend())

        # Model (collapsed group)
        mdl_body = QWidget()
        self._mdl_form = QFormLayout(mdl_body)
        mdl_form = self._mdl_form
        self.model_combo = QComboBox()
        try:
            _compact_combo(self.model_combo, 12)
        except Exception:
            pass

        self.model_refresh = QPushButton("Refresh")
        self.model_browse = QPushButton("Browse…")
        rowm = QHBoxLayout()
        rowm.addWidget(self.model_combo, 1)
        rowm.addWidget(self.model_refresh, 0)
        rowm.addWidget(self.model_browse, 0)
        self.model_label = QLabel("Model")
        mdl_form.addRow(self.model_label, rowm)

        # GGUF model/VAE selectors (used only for Z-Image Turbo GGUF engine)
        self.gguf_model_combo = QComboBox()
        try:
            _compact_combo(self.gguf_model_combo, 12)
        except Exception:
            pass
        self.gguf_model_refresh = QPushButton("Refresh")
        self.gguf_model_browse = QPushButton("Browse…")
        rowg = QHBoxLayout()
        rowg.addWidget(self.gguf_model_combo, 1)
        rowg.addWidget(self.gguf_model_refresh, 0)
        rowg.addWidget(self.gguf_model_browse, 0)
        self.gguf_model_label = QLabel("GGUF model")
        mdl_form.addRow(self.gguf_model_label, rowg)

        self.gguf_vae_combo = QComboBox()
        try:
            _compact_combo(self.gguf_vae_combo, 12)
        except Exception:
            pass
        self.gguf_vae_refresh = QPushButton("Refresh")
        self.gguf_vae_browse = QPushButton("Browse…")
        rowv = QHBoxLayout()
        rowv.addWidget(self.gguf_vae_combo, 1)
        rowv.addWidget(self.gguf_vae_refresh, 0)
        rowv.addWidget(self.gguf_vae_browse, 0)
        self.gguf_vae_label = QLabel("GGUF VAE")
        mdl_form.addRow(self.gguf_vae_label, rowv)

        # qwen 2.5 12B GGUF selector (diffusion model)
        self.qwen_model_combo = QComboBox()
        try:
            _compact_combo(self.qwen_model_combo, 12)
        except Exception:
            pass
        self.qwen_model_refresh = QPushButton("Refresh")
        self.qwen_model_browse = QPushButton("Browse…")
        roww = QHBoxLayout()
        roww.addWidget(self.qwen_model_combo, 1)
        roww.addWidget(self.qwen_model_refresh, 0)
        roww.addWidget(self.qwen_model_browse, 0)
        self.qwen_model_label = QLabel("qwen model")
        mdl_form.addRow(self.qwen_model_label, roww)

        # sd-cli.exe selector (used for GGUF engines: Z-Image Turbo GGUF and qwen 2.5 12B GGUF)
        self.sd_cli_path = QLineEdit()
        try:
            self.sd_cli_path.setPlaceholderText("Auto (detect bundled sd-cli.exe)")
        except Exception:
            pass
        self.sd_cli_browse = QPushButton("Browse…")
        self.sd_cli_clear = QPushButton("Clear")
        row_sdcli = QHBoxLayout()
        row_sdcli.addWidget(self.sd_cli_path, 1)
        row_sdcli.addWidget(self.sd_cli_browse, 0)
        row_sdcli.addWidget(self.sd_cli_clear, 0)
        self.sd_cli_label = QLabel("sd-cli.exe")
        mdl_form.addRow(self.sd_cli_label, row_sdcli)

        # Restore + persist sd-cli path
        def _save_sd_cli_path():
            try:
                if hasattr(self, "_persist_settings"):
                    self._persist_settings.setValue("sd_cli_path", self.sd_cli_path.text().strip())
            except Exception:
                pass

        def _browse_sd_cli_path():
            try:
                start_dir = ""
                try:
                    from pathlib import Path as _P
                    # Prefer last used folder, then ./tools, then app root
                    prev = ""
                    try:
                        prev = self.sd_cli_path.text().strip()
                    except Exception:
                        prev = ""
                    if prev:
                        try:
                            start_dir = str(_P(prev).resolve().parent)
                        except Exception:
                            start_dir = ""
                    if not start_dir:
                        cand = _P("./tools").resolve()
                        if cand.exists() and cand.is_dir():
                            start_dir = str(cand)
                    if not start_dir:
                        try:
                            start_dir = str(_P(__file__).resolve().parents[1])
                        except Exception:
                            start_dir = str(_P(".").resolve())
                except Exception:
                    start_dir = ""

                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Choose sd-cli.exe",
                    start_dir or str(Path(".").resolve()),
                    "sd-cli.exe (sd-cli.exe);;Executables (*.exe);;All files (*.*)"
                )
                if not path:
                    return
                self.sd_cli_path.setText(path)
                _save_sd_cli_path()
            except Exception as e:
                try:
                    print("[txt2img] browse sd-cli failed:", e)
                except Exception:
                    pass

        def _clear_sd_cli_path():
            try:
                self.sd_cli_path.setText("")
                _save_sd_cli_path()
            except Exception:
                pass

        try:
            self.sd_cli_browse.clicked.connect(_browse_sd_cli_path)
            self.sd_cli_clear.clicked.connect(_clear_sd_cli_path)
            self.sd_cli_path.textChanged.connect(lambda *_: _save_sd_cli_path())
        except Exception:
            pass

        try:
            if hasattr(self, "_persist_settings"):
                _saved = self._persist_settings.value("sd_cli_path", "") or ""
                if str(_saved).strip():
                    self.sd_cli_path.setText(str(_saved).strip())
        except Exception:
            pass

        # Default hidden until GGUF engine is selected
        try:
            for _w in (self.gguf_model_label, self.gguf_model_combo, self.gguf_model_refresh, self.gguf_model_browse,
                       self.gguf_vae_label, self.gguf_vae_combo, self.gguf_vae_refresh, self.gguf_vae_browse, self.qwen_model_label, self.qwen_model_combo, self.qwen_model_refresh, self.qwen_model_browse, self.sd_cli_label, self.sd_cli_path, self.sd_cli_browse, self.sd_cli_clear):
                _w.setVisible(False)
        except Exception:
            pass


        def _populate_models():
            try:
                from pathlib import Path as _P
                base = _P("./models")
                self.model_combo.blockSignals(True)
                self.model_combo.clear()
                for sub in ["SDXL","SD15"]:
                    d = base / sub
                    if d.exists():
                        for f in sorted(d.glob("*.safetensors")):
                            self.model_combo.addItem(f"{sub} — {f.name}", str(f.resolve()))
                self.model_combo.blockSignals(False)
            except Exception as e:
                print("[txt2img] model scan failed:", e)

        def _browse_model():
            try:
                path, _ = QFileDialog.getOpenFileName(self, "Choose model file", str(Path("./models").resolve()), "Safetensors (*.safetensors)")
                if path:
                    from pathlib import Path as _P
                    label = _P(path).parent.name + " — " + _P(path).name
                    idx = self.model_combo.findData(path)
                    if idx < 0:
                        self.model_combo.addItem(label, path)
                        idx = self.model_combo.findData(path)
                    if idx >= 0:
                        self.model_combo.setCurrentIndex(idx)
            except Exception as e:
                print("[txt2img] browse model failed:", e)

        _populate_models()
        self.model_refresh.clicked.connect(_populate_models)
        self.model_browse.clicked.connect(_browse_model)

        # Update LoRA visibility when switching between SDXL and SD15 models
        try:
            self.model_combo.currentIndexChanged.connect(lambda *_: self._apply_lora_visibility())
        except Exception:
            pass
        try:
            self.model_combo.currentTextChanged.connect(lambda *_: self._apply_lora_visibility())
        except Exception:
            pass

        def _populate_gguf_models():
            try:
                from pathlib import Path as _P
                ggdir = _P("./models") / "Z-Image-Turbo GGUF"
                self.gguf_model_combo.blockSignals(True)
                self.gguf_model_combo.clear()
                self.gguf_model_combo.addItem("Auto (Q5 default)", "")
                if ggdir.exists():
                    for f in sorted(ggdir.glob("*.gguf")):
                        n = f.name.lower()
                        # Only show Z-Image GGUF models (avoid unrelated .gguf files in the same folder)
                        if not (('z_image' in n) or ('zimage' in n) or ('z-image' in n) or ('zimg' in n)):
                            continue
                        self.gguf_model_combo.addItem(f.name, str(f.resolve()))
                # Restore last selection
                try:
                    saved = self._persist_settings.value("zimage_gguf_model_path", "") if hasattr(self, "_persist_settings") else ""
                except Exception:
                    saved = ""
                if saved:
                    idx = self.gguf_model_combo.findData(saved)
                    if idx < 0:
                        self.gguf_model_combo.addItem(_P(saved).name, saved)
                        idx = self.gguf_model_combo.findData(saved)
                    if idx >= 0:
                        self.gguf_model_combo.setCurrentIndex(idx)
                self.gguf_model_combo.blockSignals(False)
            except Exception as e:
                try:
                    self.gguf_model_combo.blockSignals(False)
                except Exception:
                    pass
                print("[txt2img] gguf model scan failed:", e)

        def _populate_gguf_vaes():
            try:
                from pathlib import Path as _P
                ggdir = _P("./models") / "Z-Image-Turbo GGUF"
                self.gguf_vae_combo.blockSignals(True)
                self.gguf_vae_combo.clear()
                self.gguf_vae_combo.addItem("Auto (ae.safetensors default)", "")
                if ggdir.exists():
                    for f in sorted(ggdir.glob("*.safetensors")):
                        n = f.name.lower()
                        # Only show likely VAE files (avoid unrelated safetensors in the same folder)
                        if not (('vae' in n) or n.startswith('ae') or ('_ae' in n) or ('-ae' in n) or ('ae.' in n)):
                            continue
                        self.gguf_vae_combo.addItem(f.name, str(f.resolve()))
                # Restore last selection
                try:
                    saved = self._persist_settings.value("zimage_gguf_vae_path", "") if hasattr(self, "_persist_settings") else ""
                except Exception:
                    saved = ""
                if saved:
                    idx = self.gguf_vae_combo.findData(saved)
                    if idx < 0:
                        self.gguf_vae_combo.addItem(_P(saved).name, saved)
                        idx = self.gguf_vae_combo.findData(saved)
                    if idx >= 0:
                        self.gguf_vae_combo.setCurrentIndex(idx)
                self.gguf_vae_combo.blockSignals(False)
            except Exception as e:
                try:
                    self.gguf_vae_combo.blockSignals(False)
                except Exception:
                    pass
                print("[txt2img] gguf vae scan failed:", e)

        def _browse_gguf_model():
            try:
                path, _ = QFileDialog.getOpenFileName(self, "Choose GGUF diffusion model", str(Path("./models").resolve()), "GGUF (*.gguf)")
                if not path:
                    return
                from pathlib import Path as _P
                idx = self.gguf_model_combo.findData(path)
                if idx < 0:
                    self.gguf_model_combo.addItem(_P(path).name, path)
                    idx = self.gguf_model_combo.findData(path)
                if idx >= 0:
                    self.gguf_model_combo.setCurrentIndex(idx)
            except Exception as e:
                print("[txt2img] browse gguf model failed:", e)

        def _browse_gguf_vae():
            try:
                path, _ = QFileDialog.getOpenFileName(self, "Choose GGUF VAE", str(Path("./models").resolve()), "Safetensors (*.safetensors)")
                if not path:
                    return
                from pathlib import Path as _P
                idx = self.gguf_vae_combo.findData(path)
                if idx < 0:
                    self.gguf_vae_combo.addItem(_P(path).name, path)
                    idx = self.gguf_vae_combo.findData(path)
                if idx >= 0:
                    self.gguf_vae_combo.setCurrentIndex(idx)
            except Exception as e:
                print("[txt2img] browse gguf vae failed:", e)

        # Populate once and hook up actions
        try:
            _populate_gguf_models()
            _populate_gguf_vaes()
            self.gguf_model_refresh.clicked.connect(_populate_gguf_models)
            self.gguf_vae_refresh.clicked.connect(_populate_gguf_vaes)
            self.gguf_model_browse.clicked.connect(_browse_gguf_model)
            self.gguf_vae_browse.clicked.connect(_browse_gguf_vae)

            def _save_gguf_sel():
                try:
                    if hasattr(self, "_persist_settings"):
                        self._persist_settings.setValue("zimage_gguf_model_path", self.gguf_model_combo.currentData() or "")
                        self._persist_settings.setValue("zimage_gguf_vae_path", self.gguf_vae_combo.currentData() or "")
                except Exception:
                    pass

            try:
                self.gguf_model_combo.currentIndexChanged.connect(lambda *_: _save_gguf_sel())
                self.gguf_vae_combo.currentIndexChanged.connect(lambda *_: _save_gguf_sel())
            except Exception:
                pass
        except Exception:
            pass

        # qwen 2.5 12B GGUF selector: choose diffusion GGUF from the same folder as the default qwen GGUF
        def _populate_qwen_models():
            try:
                from pathlib import Path as _P
                scan_dir = None

                # Prefer the folder that contains the default qwen diffusion model (same folder as the first qwen GGUF file)
                try:
                    from helpers import qwen2512 as _qwen
                except Exception:
                    try:
                        import qwen2512 as _qwen
                    except Exception:
                        _qwen = None

                if _qwen is not None:
                    try:
                        _root_dir = _P(__file__).resolve().parents[1]
                    except Exception:
                        _root_dir = _P(".").resolve()
                    try:
                        d0, _llm0, _vae0 = _qwen._model_paths(_root_dir)
                        if d0:
                            scan_dir = _P(str(d0)).resolve().parent
                    except Exception:
                        scan_dir = None

                if scan_dir is None:
                    # Fallback folders (optional downloads may use one of these)
                    for cand in (
                        _P("./models") / "QWEN 2.5 12B GGUF",
                        _P("./models") / "qwen2512gguf",
                        _P("./models") / "qwen2512GGUF",
                    ):
                        try:
                            if cand.exists() and cand.is_dir():
                                scan_dir = cand
                                break
                        except Exception:
                            continue

                if scan_dir is None:
                    scan_dir = _P("./models")

                self.qwen_model_combo.blockSignals(True)
                self.qwen_model_combo.clear()
                self.qwen_model_combo.addItem("Auto (default)", "")

                ggufs = []
                try:
                    if scan_dir.exists():
                        ggufs = sorted(scan_dir.glob("*.gguf"))
                except Exception:
                    ggufs = []

                def _looks_like_diffusion(name: str) -> bool:
                    n = (name or "").lower()
                    # Avoid showing obvious text/llm/tokenizer files by default
                    bad = ("llm", "qwen", "text", "t5", "clip", "encoder", "token", "vocab")
                    if any(b in n for b in bad):
                        return False
                    return True

                shown = 0
                for f in ggufs:
                    if not _looks_like_diffusion(f.name):
                        continue
                    self.qwen_model_combo.addItem(f.name, str(f.resolve()))
                    shown += 1

                # If nothing showed up due to filtering, show all ggufs in that folder
                if shown == 0:
                    for f in ggufs:
                        self.qwen_model_combo.addItem(f.name, str(f.resolve()))

                # Restore last selection (persisted)
                try:
                    saved = self._persist_settings.value("qwen2512_model_path", "") if hasattr(self, "_persist_settings") else ""
                except Exception:
                    saved = ""
                if saved:
                    idx = self.qwen_model_combo.findData(saved)
                    if idx < 0:
                        self.qwen_model_combo.addItem(_P(saved).name, saved)
                        idx = self.qwen_model_combo.findData(saved)
                    if idx >= 0:
                        self.qwen_model_combo.setCurrentIndex(idx)

                self.qwen_model_combo.blockSignals(False)
            except Exception as e:
                try:
                    self.qwen_model_combo.blockSignals(False)
                except Exception:
                    pass
                print("[txt2img] qwen model scan failed:", e)

        def _browse_qwen_model():
            try:
                path, _ = QFileDialog.getOpenFileName(self, "Choose qwen GGUF diffusion model", str(Path("./models").resolve()), "GGUF (*.gguf)")
                if not path:
                    return
                from pathlib import Path as _P
                idx = self.qwen_model_combo.findData(path)
                if idx < 0:
                    self.qwen_model_combo.addItem(_P(path).name, path)
                    idx = self.qwen_model_combo.findData(path)
                if idx >= 0:
                    self.qwen_model_combo.setCurrentIndex(idx)
            except Exception as e:
                print("[txt2img] browse qwen model failed:", e)

        # Populate once and hook up actions
        try:
            _populate_qwen_models()
            self.qwen_model_refresh.clicked.connect(_populate_qwen_models)
            self.qwen_model_browse.clicked.connect(_browse_qwen_model)

            def _save_qwen_sel():
                try:
                    if hasattr(self, "_persist_settings"):
                        self._persist_settings.setValue("qwen2512_model_path", self.qwen_model_combo.currentData() or "")
                except Exception:
                    pass

            try:
                self.qwen_model_combo.currentIndexChanged.connect(lambda *_: _save_qwen_sel())
            except Exception:
                pass
        except Exception:
            pass



        # Helper: where to scan for LoRA files
        def _get_lora_root():
            try:
                lr = getattr(self, "_lora_root", None)
            except Exception:
                lr = None
            if not lr:
                try:
                    lr = _default_lora_root()
                except Exception:
                    lr = "./loras/txt2img"
                try:
                    self._lora_root = lr
                except Exception:
                    pass
            return lr

        def _set_lora_root(path_str):
            try:
                if not path_str:
                    return
                self._lora_root = str(path_str)
                try:
                    self._persist_settings.setValue("txt2img_lora_root", self._lora_root)
                except Exception:
                    pass
                try:
                    if hasattr(self, "lora_path_btn") and self.lora_path_btn is not None:
                        self.lora_path_btn.setToolTip(self._lora_root)
                except Exception:
                    pass
            except Exception:
                pass

        def _browse_lora_root():
            try:
                start_dir = _get_lora_root()
            except Exception:
                start_dir = "./loras/txt2img"
            try:
                path = QFileDialog.getExistingDirectory(self, "Choose LoRA folder", start_dir)
            except Exception as e:
                try:
                    print("[txt2img] choose LoRA folder failed:", e)
                except Exception:
                    pass
                path = ""
            if path:
                try:
                    from pathlib import Path as _P
                    path = str(_P(path).resolve())
                except Exception:
                    pass
                _set_lora_root(path)
                try:
                    _reload_all_loras()
                except Exception:
                    try:
                        _populate_loras()
                    except Exception:
                        pass
                    try:
                        _populate_loras2()
                    except Exception:
                        pass

        # --- LoRA picker (SDXL) ---
        self.lora_combo = QComboBox()
        try:
            _compact_combo(self.lora_combo, 12)
        except Exception:
            pass

        self.lora_refresh = QPushButton("Reload")
        self.lora_path_btn = QPushButton("Choose LoRA path…")
        try:
            self.lora_path_btn.setToolTip(_get_lora_root())
        except Exception:
            pass
        rowl = QHBoxLayout()
        rowl.addWidget(self.lora_path_btn, 0)
        rowl.addWidget(self.lora_refresh, 0)
        rowl.addWidget(self.lora_combo, 1)
        self.lora_label = QLabel("LoRA")
        mdl_form.addRow(self.lora_label, rowl)
        try:
            self.lora_path_btn.clicked.connect(_browse_lora_root)
        except Exception:
            pass
        # LoRA 1 Strength (scale)
        self.lora_strength_slider = QSlider(Qt.Horizontal)
        try:
            self.lora_strength_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass
        self.lora_strength_slider.setMinimum(0)
        self.lora_strength_slider.setMaximum(150)
        self.lora_strength_slider.setValue(100)
        self.lora_strength = QDoubleSpinBox(); self.lora_strength.setRange(0.0, 1.5); self.lora_strength.setSingleStep(0.05); self.lora_strength.setDecimals(2); self.lora_strength.setValue(1.0)
        rowls = QHBoxLayout()
        self.lora_strength_label = QLabel("LoRA 1 Strength:")
        rowls.addWidget(self.lora_strength_label)
        rowls.addWidget(self.lora_strength, 0)
        rowls.addWidget(self.lora_strength_slider, 1)
        mdl_form.addRow(rowls)

        def _sync_lora1_strength_from_slider(v):
            try:
                self.lora_strength.blockSignals(True)
                self.lora_strength.setValue(v/100.0)
                self.lora_strength.blockSignals(False)
            except Exception:
                pass

        def _sync_lora1_slider_from_spin(v):
            try:
                self.lora_strength_slider.blockSignals(True)
                self.lora_strength_slider.setValue(int(round(v*100)))
                self.lora_strength_slider.blockSignals(False)
            except Exception:
                pass

        self.lora_strength_slider.valueChanged.connect(_sync_lora1_strength_from_slider)
        self.lora_strength.valueChanged.connect(_sync_lora1_slider_from_spin)

        self.lora2_combo = QComboBox()
        try:
            _compact_combo(self.lora2_combo, 12)
        except Exception:
            pass

        self.lora2_refresh = QPushButton("Reload")
        rowl2 = QHBoxLayout()
        rowl2.addWidget(self.lora2_refresh, 0)
        rowl2.addWidget(self.lora2_combo, 1)
        self.lora2_label = QLabel("LoRA 2")
        mdl_form.addRow(self.lora2_label, rowl2)

        def _populate_loras2():
            try:
                from pathlib import Path as _P
                root = _get_lora_root()
                base = _P(root)
                self.lora2_combo.blockSignals(True)
                self.lora2_combo.clear()
                self.lora2_combo.addItem("None", "")
                if base.exists():
                    for f in sorted(base.glob("*.safetensors")):
                        self.lora2_combo.addItem(f.name, str(f.resolve()))
                self.lora2_combo.blockSignals(False)
            except Exception as e:
                print("[txt2img] lora2 scan failed:", e)

        
        _populate_loras2()
        # Both reloads refresh both dropdowns for convenience. Guard duplicate connections.
        def _reload_all_loras():
            try:
                _populate_loras()
            except Exception:
                pass
            try:
                _populate_loras2()
            except Exception:
                pass
        if not hasattr(self, "_loras_reload_wired"):
            self._loras_reload_wired = True
            try:
                self.lora_refresh.clicked.connect(_reload_all_loras)
            except Exception:
                pass
            try:
                self.lora2_refresh.clicked.connect(_reload_all_loras)
            except Exception:
                pass

        # Both reloads refresh both dropdowns for convenience
        def _reload_all_loras():
            try:
                _populate_loras()
            except Exception:
                pass
            try:
                _populate_loras2()
            except Exception:
                pass
        self.lora2_refresh.clicked.connect(_reload_all_loras)
        try:
            self.lora_refresh.clicked.disconnect()
        except Exception:
            pass
        self.lora_refresh.clicked.connect(_reload_all_loras)

        # LoRA 2 strength (scale)
        self.lora2_strength_slider = QSlider(Qt.Horizontal)
        try:
            self.lora2_strength_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        except Exception:
            pass
        self.lora2_strength_slider.setMinimum(0)
        self.lora2_strength_slider.setMaximum(150)
        self.lora2_strength_slider.setValue(100)
        self.lora2_strength = QDoubleSpinBox(); self.lora2_strength.setRange(0.0, 1.5); self.lora2_strength.setSingleStep(0.05); self.lora2_strength.setDecimals(2); self.lora2_strength.setValue(1.0)
        rowls2 = QHBoxLayout()
        self.lora2_strength_label = QLabel("LoRA 2 Strength:")
        rowls2.addWidget(self.lora2_strength_label)
        rowls2.addWidget(self.lora2_strength, 0)
        rowls2.addWidget(self.lora2_strength_slider, 1)
        mdl_form.addRow(rowls2)

        def _sync_lora2_strength_from_slider(v):
            try:
                self.lora2_strength.blockSignals(True)
                self.lora2_strength.setValue(v/100.0)
                self.lora2_strength.blockSignals(False)
            except Exception:
                pass

        def _sync_lora2_slider_from_spin(v):
            try:
                self.lora2_strength_slider.blockSignals(True)
                self.lora2_strength_slider.setValue(int(round(v*100)))
                self.lora2_strength_slider.blockSignals(False)
            except Exception:
                pass

        (_sync_lora2_strength_from_slider)
        self.lora2_strength.valueChanged.connect(_sync_lora2_slider_from_spin)


        def _populate_loras():
            try:
                from pathlib import Path as _P
                root = _get_lora_root()
                base = _P(root)
                self.lora_combo.blockSignals(True)
                self.lora_combo.clear()
                self.lora_combo.addItem("None", "")
                if base.exists():
                    for f in sorted(base.glob("*.safetensors")):
                        self.lora_combo.addItem(f.name, str(f.resolve()))
                self.lora_combo.blockSignals(False)
            except Exception as e:
                print("[txt2img] lora scan failed:", e)

        _populate_loras()
        self.lora_refresh.clicked.connect(_populate_loras)

        self._model_picker = _Disclosure("Model", mdl_body, start_open=False, parent=self)
        root.addWidget(self._model_picker)
                # Engine-dependent banner and Model/LoRA visibility
        try:
            if hasattr(self, "engine_combo") and self.engine_combo is not None:
                self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        except Exception:
            pass
        try:
            self._on_engine_changed()
        except Exception:
            pass

        # Advanced (collapsed group)
        adv_body = QWidget()
        adv_form = QFormLayout(adv_body)
        self.sampler = QComboBox()
        try:
            _compact_combo(self.sampler, 10)
        except Exception:
            pass
        self.sampler.addItems(["auto","DPM++ 2M (Karras)","Euler a","Euler","Heun","UniPC","DDIM"])
        self.attn_slicing = QCheckBox("Attention slicing")
        # Qwen-only: CPU offload (stable-diffusion.cpp --offload-to-cpu)
        self.qwen_offload_cpu = QCheckBox("CPU offload")
        try:
            _off0 = False
            if hasattr(self, "_persist_settings"):
                _off0 = bool(int(self._persist_settings.value("qwen2512_offload_cpu", 0) or 0))
            self.qwen_offload_cpu.setChecked(bool(_off0))
        except Exception:
            pass
        try:
            self.qwen_offload_cpu.setToolTip("Qwen-Image: offload model to CPU (saves VRAM, slower).")
        except Exception:
            pass

        self.vae_device = QComboBox()
        try:
            _compact_combo(self.vae_device, 6)
        except Exception:
            pass
        self.vae_device.addItems(["auto","cpu","gpu"])
        self.gpu_index = QSpinBox(); self.gpu_index.setRange(0,8)
        self.threads = QSpinBox(); self.threads.setRange(1,256); self.threads.setValue(8)
        self.format_combo = QComboBox();
        try:
            _compact_combo(self.format_combo, 4)
        except Exception:
            pass
        self.format_combo.addItems(["png","jpg","webp"])
        self.filename_template = QLineEdit("IMG_{seed}.png")
        self.reset_fname = QPushButton("Reset"); self.reset_fname.clicked.connect(lambda: self.filename_template.setText(f"IMG_{{seed}}.{self.format_combo.currentText()}"))
        try:
            _on_format_changed()
        except Exception:
            pass
        self.hires_helper = QCheckBox("Hi-res helper")
        try:
            self.hires_helper.setChecked(False)
            self.hires_helper.setVisible(False)
        except Exception:
            pass
        self.fit_check = QCheckBox("Fit-check")
        adv_form.addRow("Sampler", self.sampler)
        # Qwen-only (Qwen-Image / stable-diffusion.cpp): flow-shift value
        self.qwen_flow_label = QLabel("Flow")
        self.qwen_flow_shift = QSpinBox()
        try:
            self.qwen_flow_shift.setRange(0, 20)
        except Exception:
            pass
        _flow0 = 3
        try:
            if hasattr(self, "_persist_settings"):
                _flow0 = int(self._persist_settings.value("qwen2512_flow_shift", 3))
        except Exception:
            _flow0 = 3
        try:
            self.qwen_flow_shift.setValue(int(_flow0))
        except Exception:
            pass
        try:
            self.qwen_flow_shift.setToolTip("Qwen-Image flow-shift value (Unsloth example uses 3).")
        except Exception:
            pass
        adv_form.addRow(self.qwen_flow_label, self.qwen_flow_shift)
        try:
            self.qwen_flow_label.setVisible(False)
            self.qwen_flow_shift.setVisible(False)
            try:
                self.qwen_offload_cpu.setVisible(False)
            except Exception:
                pass
        except Exception:
            pass
        try:
            def _save_qwen_flow():
                try:
                    if hasattr(self, "_persist_settings"):
                        self._persist_settings.setValue("qwen2512_flow_shift", int(self.qwen_flow_shift.value()))
                except Exception:
                    pass
            self.qwen_flow_shift.valueChanged.connect(lambda *_: _save_qwen_flow())
        except Exception:
            pass
        
        # Combine Attention slicing + Qwen CPU offload on one row (Qwen toggle auto-hides)
        self._adv_row_attn_qwen = QWidget()
        _adv_row_lay = QHBoxLayout(self._adv_row_attn_qwen)
        _adv_row_lay.setContentsMargins(0,0,0,0)
        _adv_row_lay.setSpacing(12)
        _adv_row_lay.addWidget(self.attn_slicing)
        try:
            _adv_row_lay.addWidget(self.qwen_offload_cpu)
        except Exception:
            pass
        _adv_row_lay.addStretch(1)
        adv_form.addRow(self._adv_row_attn_qwen)

        adv_form.addRow("VAE device", self.vae_device)
        adv_form.addRow("GPU index", self.gpu_index)
        adv_form.addRow("Threads", self.threads)
        adv_form.addRow("File format", self.format_combo)

        # Auto-sync filename template extension when format changes
        def _on_format_changed(*_):
            try:
                fmt = (self.format_combo.currentText() or "").strip().lower()
                name = (self.filename_template.text() or "").strip()
                import re
                if not name:
                    name = f"IMG_{{seed}}.{fmt}"
                else:
                    # Replace a known image extension at the end, or append if missing
                    if re.search(r"\.(png|jpe?g|webp|tiff?|bmp)$", name, flags=re.IGNORECASE):
                        name = re.sub(r"\.(png|jpe?g|webp|tiff?|bmp)$", f".{fmt}", name, flags=re.IGNORECASE)
                    else:
                        name = name.rstrip('.') + f".{fmt}"
                # Avoid recursive autosave storms while updating the field
                try:
                    self.filename_template.blockSignals(True)
                except Exception:
                    pass
                self.filename_template.setText(name)
                try:
                    self.filename_template.blockSignals(False)
                except Exception:
                    pass
            except Exception:
                pass
        try:
            self.format_combo.currentTextChanged.connect(_on_format_changed)
        except Exception:
            pass
        try:
            self.format_combo.currentIndexChanged.connect(lambda *_: _on_format_changed())
        except Exception:
            pass
        # Remember last user-chosen format so engine switches or presets never reset it
        try:
            self._last_user_format = self.format_combo.currentText()
            def _remember_format(*_):
                try:
                    self._last_user_format = self.format_combo.currentText()
                except Exception:
                    pass
            self.format_combo.currentIndexChanged.connect(_remember_format)
            self.format_combo.currentTextChanged.connect(_remember_format)
        except Exception:
            pass
        rowf = QHBoxLayout(); rowf.addWidget(self.filename_template, 1); rowf.addWidget(self.reset_fname)
        adv_form.addRow("Filename", rowf)
     #   adv_form.addRow(self.hires_helper)
        # fit-check hidden per request
        # adv_form.addRow(self.fit_check)
        try:
            self.fit_check.hide()
        except Exception:
            pass


        # --- Helpful tooltips for Advanced options ---
        try:
            self.sampler.setToolTip("Which sampler/scheduler the diffusion model uses. Affects look, speed and stability.\n• DPM++ 2M (Karras): balanced quality/speed\n• Euler a: fast, stylized (anime)\n• Heun: cinematic contrast\n• UniPC: stable across settings\n• DDIM: softer, fewer steps. Changing sampler can change the ideal step count.")
        except Exception:
            pass
        try:
            self.attn_slicing.setToolTip("Reduce VRAM usage by splitting attention layers into smaller chunks. Enable on low‑VRAM GPUs. Slightly slower when enabled.")
        except Exception:
            pass
        try:
            self.vae_device.setToolTip("Where to run the VAE (decoder/encoder).\n• Auto: choose GPU if memory allows\n• GPU: fastest, uses more VRAM\n• CPU: frees VRAM but slower.")
        except Exception:
            pass
        try:
            self.gpu_index.setToolTip("Select which GPU to use (0 = first device). Only relevant if you have multiple GPUs.")
        except Exception:
            pass
        try:
            self.threads.setToolTip("Only Relevant for CPU users and any CPU-side offloading steps (e.g., VAE on CPU). Higher can speed up saves/loads, but using all cores of your CPU may reduce UI responsiveness.")
        except Exception:
            pass
        try:
            self.format_combo.setToolTip("Output image format.\n• PNG: lossless (largest files)\n• JPG: smallest, lossy (choose for web)\n• WEBP: modern balance; may be slower to save.")
        except Exception:
            pass
        try:
            self.filename_template.setToolTip("Filename pattern for saved images. You can use placeholders like {seed}, {idx}, {width}, {height}, {model}. Example: sd_{seed}_{idx:03d}.png")
        except Exception:
            pass
        try:
            self.hires_helper.setToolTip("Two‑stage generate→upscale pass to add detail. Slower but sharper—great for portraits, products, and typography.")
        except Exception:
            pass
        try:
            self.fit_check.setToolTip("")
        except Exception:
            pass
        root.addLayout(form)
        self._advanced = _Disclosure("Advanced", adv_body, start_open=False, parent=self)
        root.addWidget(self._advanced)

        # Progress + actions
        prog_row = QHBoxLayout()
        self.progress = QProgressBar(); self.progress.setRange(0,100)
        self.status = QLabel("Ready")
        prog_row.addWidget(self.progress, 1); prog_row.addWidget(self.status, 0)
        outer.addLayout(prog_row)

        try:
            self._apply_queue_visibility(self.use_queue.isChecked())
        except Exception:
            pass
        try:
            QTimer.singleShot(0, lambda: self._apply_queue_visibility(self.use_queue.isChecked()))
        except Exception:
            pass

        btns = QHBoxLayout()
        self.add_to_queue = QPushButton("Add to Queue")
        self.add_and_run = QPushButton("Add & Run")
        self.add_and_run.hide()
        self.generate_now = QPushButton("Generate image(s)")
        # Batch controls moved next to Generate button on the same line
        self.batch_label_bottom = QLabel("Batch:")
        # Enlarge the main action buttons (+3 px) and keep hover style for Generate
        try:
            for _btn, _name in ((self.generate_now, "generateNowButton"), (self.add_and_run, "addAndRunButton")):
                _btn.setObjectName(_name)
                _font = _btn.font()
                try:
                    _ps = _font.pointSize()
                    if _ps and _ps > 0:
                        _font.setPointSize(_ps + 3)
                    else:
                        _px = _font.pixelSize()
                        if _px and _px > 0:
                            _font.setPixelSize(_px + 3)
                except Exception:
                    pass
                _btn.setFont(_font)
            self.generate_now.setStyleSheet("""
                QPushButton#generateNowButton:hover {
                    background-color: #a64dff;
                }
            """.strip())
        except Exception:
            pass
        btns.addWidget(self.add_to_queue)
        self.add_to_queue.hide()
        btns.addWidget(self.generate_now)
        btns.addWidget(self.batch_label_bottom)
        btns.addWidget(self.batch)
        # Open output folder button (same row as Generate/Batch)
        self.open_folder_btn = QPushButton("View results")
        try:
            self.open_folder_btn.setToolTip("Open Media Explorer on the output folder and scan for generated images.")
        except Exception:
            pass
        try:
            self.open_folder_btn.clicked.connect(self._open_output_in_media_explorer)
        except Exception:
            pass
        btns.addWidget(self.open_folder_btn)
        btns.addStretch(1)
        btns.addWidget(self.add_and_run)
        outer.addLayout(btns)

    def _apply_lora_visibility(self):
        """LoRA UI is relevant for SDXL (diffusers), Z-Image FP16, and Qwen2512."""
        try:
            key = self._engine_key_selected() if hasattr(self, "_engine_key_selected") else ""
        except Exception:
            key = ""
        try:
            key = str(key or "").lower().strip() or "diffusers"
        except Exception:
            key = "diffusers"

        show_lora = False
        try:
            if key in ("zimage", "qwen2512"):
                show_lora = True
            elif key == "diffusers":
                mp = ""
                try:
                    if hasattr(self, "model_combo") and self.model_combo is not None:
                        mp = f"{self.model_combo.currentText() or ''} {self.model_combo.currentData() or ''}"
                except Exception:
                    mp = ""
                mp = (mp or "").lower()
                show_lora = ("sdxl" in mp) or ("sd_xl" in mp) or ("sd-xl" in mp)
        except Exception:
            show_lora = False

        try:
            for _w in (
                getattr(self, "lora_label", None),
                getattr(self, "lora_path_btn", None),
                getattr(self, "lora_combo", None),
                getattr(self, "lora_refresh", None),
                getattr(self, "lora_strength_label", None),
                getattr(self, "lora_strength", None),
                getattr(self, "lora_strength_slider", None),
                getattr(self, "lora_a_strength", None),
                getattr(self, "lora_b_strength", None),
                getattr(self, "lora2_label", None),
                getattr(self, "lora2_combo", None),
                getattr(self, "lora2_refresh", None),
                getattr(self, "lora2_strength_label", None),
                getattr(self, "lora2_strength", None),
                getattr(self, "lora2_strength_slider", None),
            ):
                if _w is not None:
                    _w.setVisible(bool(show_lora))
        except Exception:
            pass


    def _on_engine_changed(self, *_):
        """
        Toggle banner text and show/hide the SD model/LoRA picker
        based on the selected engine.

        Also adjusts Steps / CFG ranges for Z-Image:
        - Z-Image: steps 1–50 (default 9), CFG 0.0–5.0 (default 0.0)
        - Diffusers (SD15/SDXL): steps 10–100 (default 25), CFG 1.0–15.0 (default 5.5)
        """

        # Preserve the currently-selected output format so engine changes never reset it
        try:
            saved_fmt = getattr(self, "_last_user_format", None)
        except Exception:
            saved_fmt = None
        try:
            if saved_fmt is None and hasattr(self, "format_combo"):
                saved_fmt = self.format_combo.currentText()
        except Exception:
            pass
        try:
            cb = getattr(self, "engine_combo", None)
        except Exception:
            cb = None
        if cb is None:
            return

        # Determine engine key: prefer itemData, fall back to text
        key = ""
        try:
            data = cb.currentData()
        except Exception:
            data = None
        try:
            if data:
                key = str(data).lower().strip()
            else:
                text = (cb.currentText() or "").lower()
                if ("qwen" in text) and (("2512" in text) or ("2.5" in text) or ("12b" in text)):
                    key = "qwen2512"
                elif "gguf" in text:
                    key = "zimage_gguf"
                elif "z-image" in text or "zimage" in text:
                    key = "zimage"
                else:
                    key = "diffusers"
        except Exception:
            key = "diffusers"
        is_zimage = key.startswith("zimage")
        is_gguf = (key == "zimage_gguf")
        is_qwen = (key == "qwen2512")

        # Z-Image-only UI: init-image (img2img) row (FP16 only; GGUF does not support init images)
        try:
            _show_i2i = bool(key == "zimage")
            for _w in (getattr(self, "zimg_i2i_label", None), getattr(self, "zimg_i2i_wrap", None)):
                if _w is not None:
                    _w.setVisible(_show_i2i)

            # Safety: if user previously enabled init-image, force-disable it for GGUF / qwen (no init-image support)
            if bool(key in ("zimage_gguf", "qwen2512")):
                try:
                    if getattr(self, "zimg_init_enable", None) is not None:
                        self.zimg_init_enable.setChecked(False)
                except Exception:
                    pass
        except Exception:
            pass

        # If switching to Z-Image / qwen, aggressively try to free CUDA VRAM
        if is_zimage or is_qwen:
            try:
                _aggressive_free_cuda_vram()
            except Exception:
                pass

        # Update banner text
        try:
            base = getattr(self, "_banner_default_text", None)
            if not base and hasattr(self, "banner") and self.banner is not None:
                base = self.banner.text()
                self._banner_default_text = base
            if hasattr(self, "banner") and self.banner is not None:
                if is_zimage:
                    self.banner.setText("Text to image with Z-image Turbo")
                elif is_qwen:
                    self.banner.setText("Text to image with qwen 2.5 12B GGUF")
                else:
                    self.banner.setText(base or "Text to Image with SDXL Loader")
        except Exception:
            pass

        # Keep Model/LoRA group visible, but adjust contents/labels per engine
        try:
            picker = getattr(self, "_model_picker", None)
        except Exception:
            picker = None
        try:
            if picker is not None:
                # Always keep the group itself visible
                picker.setVisible(True)
                # Update the disclosure title: 'Model' for SD engines, 'LoRA' for Z-Image
                try:
                    btn = getattr(picker, "_btn", None)
                    if btn is not None:
                        btn.setText("qwen models" if is_qwen else ("GGUF" if is_gguf else ("LoRA" if is_zimage else "Model")))
                except Exception:
                    pass
        except Exception:
            pass

        # Toggle SDXL model row visibility and rename LoRA labels depending on engine
        try:
            # Model row: hide for Z-Image, show for SD engines
            model_label = getattr(self, "model_label", None)
            for _w in (model_label,
                       getattr(self, "model_combo", None),
                       getattr(self, "model_refresh", None),
                       getattr(self, "model_browse", None)):
                if _w is not None:
                    _w.setVisible(not (is_zimage or is_qwen))

            # LoRA labels: show engine-specific text
            lora_label = getattr(self, "lora_label", None)
            if lora_label is not None:
                lora_label.setText("LoRA (Z-Image)" if is_zimage else ("LoRA (Qwen)" if is_qwen else "LoRA (SDXL)"))
            lora2_label = getattr(self, "lora2_label", None)
            if lora2_label is not None:
                lora2_label.setText("LoRA 2 (Z-Image)" if is_zimage else ("LoRA 2 (Qwen)" if is_qwen else "LoRA 2 (SDXL)"))

            # GGUF selectors: visible only for GGUF engine
            for _w in (
                getattr(self, "gguf_model_label", None),
                getattr(self, "gguf_model_combo", None),
                getattr(self, "gguf_model_refresh", None),
                getattr(self, "gguf_model_browse", None),
                getattr(self, "gguf_vae_label", None),
                getattr(self, "gguf_vae_combo", None),
                getattr(self, "gguf_vae_refresh", None),
                getattr(self, "gguf_vae_browse", None),
            ):
                if _w is not None:
                    _w.setVisible(bool(is_gguf))

            # qwen selector: visible only for qwen engine
            for _w in (
                getattr(self, "qwen_model_label", None),
                getattr(self, "qwen_model_combo", None),
                getattr(self, "qwen_model_refresh", None),
                getattr(self, "qwen_model_browse", None),
            ):
                if _w is not None:
                    _w.setVisible(bool(is_qwen))

            # sd-cli.exe selector: visible for GGUF engines (Z-Image GGUF + qwen GGUF)
            _need_sdcli = bool(is_gguf or is_qwen)
            for _w in (
                getattr(self, "sd_cli_label", None),
                getattr(self, "sd_cli_path", None),
                getattr(self, "sd_cli_browse", None),
                getattr(self, "sd_cli_clear", None),
            ):
                if _w is not None:
                    _w.setVisible(_need_sdcli)
            # LoRA UI: only for Z-Image FP16 and SDXL (diffusers)
            try:
                self._apply_lora_visibility()
            except Exception:
                pass
            # Qwen-only UI: show Flow setting only for qwen2512
            for _w in (
                getattr(self, "qwen_flow_label", None),
                getattr(self, "qwen_flow_shift", None),
                getattr(self, "qwen_offload_cpu", None),
            ):
                if _w is not None:
                    _w.setVisible(bool(is_qwen))
        except Exception:
            pass

        # Show/hide presets row based on engine
        try:
            lab = getattr(self, "preset_label", None)
            combo = getattr(self, "preset_combo", None)
            visible = not (is_zimage or is_qwen)
            if lab is not None:
                lab.setVisible(visible)
            if combo is not None:
                combo.setVisible(visible)
        except Exception:
            pass

        # Adjust filename template depending on engine
        try:
            fname_edit = getattr(self, "filename_template", None)
        except Exception:
            fname_edit = None
        if fname_edit is not None:
            try:
                prev_engine = getattr(self, "_last_engine_key", None)
            except Exception:
                prev_engine = None

            # Determine the active format (user choice wins)
            try:
                fmt = (saved_fmt or (self.format_combo.currentText() if hasattr(self, "format_combo") else "png") or "png")
            except Exception:
                fmt = "png"
            try:
                fmt = str(fmt).strip().lower() or "png"
            except Exception:
                fmt = "png"

            def _ensure_ext(name: str, _fmt: str) -> str:
                try:
                    import re as _re
                    if not name:
                        return name
                    if _re.search(r"\.(png|jpe?g|webp|tiff?|bmp)$", name, flags=_re.IGNORECASE):
                        return _re.sub(r"\.(png|jpe?g|webp|tiff?|bmp)$", f".{_fmt}", name, flags=_re.IGNORECASE)
                    return name.rstrip('.') + f".{_fmt}"
                except Exception:
                    return name

            try:
                if is_zimage:
                    # Store previous template once when entering Z-Image
                    if prev_engine != "zimage":
                        try:
                            self._filename_template_before_zimage = fname_edit.text()
                        except Exception:
                            pass
                    try:
                        fname_edit.blockSignals(True)
                    except Exception:
                        pass
                    fname_edit.setText(f"z_img_{{seed}}_{{idx:03d}}.{fmt}")
                    try:
                        fname_edit.blockSignals(False)
                    except Exception:
                        pass
                elif is_qwen:
                    # Store previous template once when entering qwen
                    if prev_engine != "qwen2512":
                        try:
                            self._filename_template_before_qwen = fname_edit.text()
                        except Exception:
                            pass
                    try:
                        fname_edit.blockSignals(True)
                    except Exception:
                        pass
                    fname_edit.setText(f"qwen_{{seed}}_{{idx:03d}}.{fmt}")
                    try:
                        fname_edit.blockSignals(False)
                    except Exception:
                        pass
                else:
                    # Switching back to SD15/SDXL: restore original or default (from whichever special engine we came from)
                    try:
                        if prev_engine == "qwen2512":
                            orig = getattr(self, "_filename_template_before_qwen", "") or ""
                        else:
                            orig = getattr(self, "_filename_template_before_zimage", "") or ""
                    except Exception:
                        orig = ""
                    if not orig:
                        orig = f"IMG_{{seed}}.{fmt}"
                    else:
                        # Ensure restored template extension matches current format
                        try:
                            orig = _ensure_ext(str(orig), fmt)
                        except Exception:
                            pass
                    try:
                        fname_edit.blockSignals(True)
                    except Exception:
                        pass
                    fname_edit.setText(orig)
                    try:
                        fname_edit.blockSignals(False)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if is_zimage:
                    self._last_engine_key = "zimage"
                elif is_qwen:
                    self._last_engine_key = "qwen2512"
                else:
                    self._last_engine_key = "diffusers"
            except Exception:
                pass

        # Adjust Steps / CFG ranges depending on engine
        try:
            # Steps slider
            ss = getattr(self, "steps_slider", None)
            sv = getattr(self, "steps_value", None)
            if ss is not None:
                if is_zimage:
                    # Z-Image prefers very low steps; force 9 steps on switch
                    try:
                        ss.blockSignals(True)
                    except Exception:
                        pass
                    ss.setRange(1, 50)
                    ss.setValue(9)
                    try:
                        ss.blockSignals(False)
                    except Exception:
                        pass
                    try:
                        if sv is not None:
                            sv.setText("9")
                    except Exception:
                        pass
                elif is_qwen:
                    # qwen 2.5 12B GGUF: wider range; default 25
                    try:
                        ss.blockSignals(True)
                    except Exception:
                        pass
                    ss.setRange(1, 200)
                    ss.setValue(25)
                    try:
                        ss.blockSignals(False)
                    except Exception:
                        pass
                    try:
                        if sv is not None:
                            sv.setText("25")
                    except Exception:
                        pass
                else:
                    # Diffusers SD15/SDXL: 10–100, force 25 steps on switch
                    try:
                        ss.blockSignals(True)
                    except Exception:
                        pass
                    ss.setRange(10, 100)
                    ss.setValue(25)
                    try:
                        ss.blockSignals(False)
                    except Exception:
                        pass
                    try:
                        if sv is not None:
                            sv.setText("25")
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            # CFG scale
            cs = getattr(self, "cfg_scale", None)
            if cs is not None:
                if is_zimage:
                    # Z-Image: CFG 0.0–5.0, force 0.0 on switch
                    try:
                        cs.blockSignals(True)
                    except Exception:
                        pass
                    cs.setRange(0.0, 5.0)
                    cs.setSingleStep(0.1)
                    cs.setValue(0.0)
                    try:
                        cs.blockSignals(False)
                    except Exception:
                        pass
                elif is_qwen:
                    # qwen 2.5 12B GGUF: typical low CFG; default 2.5
                    try:
                        cs.blockSignals(True)
                    except Exception:
                        pass
                    cs.setRange(0.0, 20.0)
                    cs.setSingleStep(0.1)
                    cs.setValue(2.5)
                    try:
                        cs.blockSignals(False)
                    except Exception:
                        pass
                else:
                    # Diffusers SD15/SDXL: CFG 1.0–15.0, force 5.5 on switch
                    try:
                        cs.blockSignals(True)
                    except Exception:
                        pass
                    cs.setRange(1.0, 15.0)
                    cs.setSingleStep(0.1)
                    cs.setValue(5.5)
                    try:
                        cs.blockSignals(False)
                    except Exception:
                        pass
        except Exception:
            pass
        # Adjust size presets and manual width/height range depending on engine
        try:
            combo = getattr(self, "size_combo", None)
            w_spin = getattr(self, "size_manual_w", None)
            h_spin = getattr(self, "size_manual_h", None)
            base_presets = getattr(self, "_size_presets", None)
            z_presets = getattr(self, "_zimage_size_presets", [])
            if combo is not None and base_presets:
                # Remember current requested size (if any)
                try:
                    data = combo.currentData()
                    cur_size = (int(data[0]), int(data[1])) if data else None
                except Exception:
                    cur_size = None
                try:
                    last_idx = int(getattr(self, "_last_size_index", combo.currentIndex()))
                except Exception:
                    last_idx = combo.currentIndex()
                try:
                    combo.blockSignals(True)
                except Exception:
                    pass
                try:
                    combo.clear()
                except Exception:
                    pass
                # Rebuild from base presets (SDXL-safe)
                try:
                    for lbl, wv, hv in base_presets:
                        combo.addItem(lbl, (wv, hv))
                except Exception:
                    pass
                # If Z-Image is active, append 2K–4K presets
                if is_zimage and z_presets:
                    try:
                        for lbl, wv, hv in z_presets:
                            combo.addItem(lbl, (wv, hv))
                    except Exception:
                        pass
                # Restore selection by size when possible
                sel_idx = -1
                if cur_size:
                    try:
                        for i in range(combo.count()):
                            d = combo.itemData(i)
                            if d and int(d[0]) == cur_size[0] and int(d[1]) == cur_size[1]:
                                sel_idx = i
                                break
                    except Exception:
                        sel_idx = -1
                if sel_idx < 0:
                    # Fallback: keep previous index if valid, otherwise pick a sensible default
                    if 0 <= last_idx < combo.count():
                        sel_idx = last_idx
                    else:
                        # For Z-Image, prefer the largest preset (last item); otherwise first
                        sel_idx = combo.count() - 1 if (is_zimage and combo.count() > 0) else 0
                try:
                    combo.setCurrentIndex(max(0, sel_idx))
                except Exception:
                    pass
                try:
                    combo.blockSignals(False)
                except Exception:
                    pass
                try:
                    self._last_size_index = sel_idx
                except Exception:
                    pass
            # Manual spinbox ranges:
            # - SDXL/diffusers: clamp to SDXL-safe max (1536px)
            # - Z-Image: allow up to full 4K (4096px)
            max_dim = 4096 if (is_zimage or is_qwen) else 1536
            for sp in (w_spin, h_spin):
                if sp is not None:
                    try:
                        sp.blockSignals(True)
                    except Exception:
                        pass
                    try:
                        sp.setRange(256, max_dim)
                    except Exception:
                        pass
                    try:
                        sp.blockSignals(False)
                    except Exception:
                        pass
        except Exception:
            pass

        # Restore previously-selected output format after engine change so it never jumps
        try:
            if hasattr(self, "format_combo") and saved_fmt:
                try:
                    self.format_combo.blockSignals(True)
                except Exception:
                    pass
                try:
                    if self.format_combo.currentText() != saved_fmt:
                        self.format_combo.setCurrentText(saved_fmt)
                finally:
                    try:
                        self.format_combo.blockSignals(False)
                    except Exception:
                        pass
        except Exception:
            pass

    def _on_clear_prompt_clicked(self):
        """Clear the main positive prompt box."""
        try:
            self.prompt.clear()
        except Exception:
            pass






    # ---------------- Prompt enhancer preset dropdown (Txt2Img-local) ----------------

    def _current_prompt_preset(self) -> str:

        cb = getattr(self, "combo_prompt_preset", None)

        if cb is None:

            return "Default"

        try:

            t = str(cb.currentText() or "Default").strip()

        except Exception:

            t = "Default"

        return t or "Default"


    def _rebuild_prompt_preset_combo(self, want: str | None = None) -> None:

        cb = getattr(self, "combo_prompt_preset", None)

        if cb is None:

            return

        names = []

        try:

            names = list((_PROMPT_PRESET_DEFS or {}).keys())

        except Exception:

            names = []

        if not names:

            names = ["Default"]

        names = ["Default"] + sorted([n for n in names if n != "Default"])

        try:

            cb.blockSignals(True)

        except Exception:

            pass

        try:

            cb.clear()

            cb.addItems(names)

            if want:

                try:

                    cb.setCurrentText(str(want))

                except Exception:

                    pass

            if not cb.currentText():

                cb.setCurrentText("Default")

        finally:

            try:

                cb.blockSignals(False)

            except Exception:

                pass


    def _on_prompt_preset_changed(self, *args) -> None:

        """Local preset selection changed (persist selection only)."""

        try:

            QSettings('FrameVision','FrameVision').setValue("txt2img_prompt_preset", self._current_prompt_preset())

        except Exception:

            pass


    def _prompttool_settings_path(self, app_root):

        try:

            from pathlib import Path as _P

            return _P(str(app_root)) / "presets" / "setsave" / "prompt.json"

        except Exception:

            from pathlib import Path as _P

            return _P("presets") / "setsave" / "prompt.json"


    def _prompttool_apply_overrides(self, app_root, overrides: dict):

        """Temporarily apply overrides to presets/setsave/prompt.json for this Qwen run.

        Returns restore info, to be passed into _prompttool_restore_settings().

        """

        try:

            import json as _json

            p = self._prompttool_settings_path(app_root)

            try:

                p.parent.mkdir(parents=True, exist_ok=True)

            except Exception:

                pass

            before = None

            try:

                if p.exists():

                    before = p.read_text(encoding="utf-8")

            except Exception:

                before = None

            data = {}

            try:

                if before:

                    data = _json.loads(before)

            except Exception:

                data = {}

            if not isinstance(data, dict):

                data = {}

            try:

                data.update(overrides or {})

            except Exception:

                pass

            try:

                p.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            except Exception:

                pass

            return {"path": str(p), "before": before}

        except Exception:

            return None


    def _prompttool_restore_settings(self, info):

        if not info:

            return

        try:

            from pathlib import Path as _P

            p = _P(str(info.get("path") or ""))

            before = info.get("before", None)

            if before is None:

                return

            p.write_text(str(before), encoding="utf-8")

        except Exception:

            pass


    def _prompttool_overrides_for_preset(self, st: dict) -> dict:

        """Compute PromptTool overrides so the CLI enhancer uses the selected preset."""

        preset = self._current_prompt_preset()

        pdef = (_PROMPT_PRESET_DEFS or {}).get(preset) or (_PROMPT_PRESET_DEFS or {}).get("Default") or {}


        defaults = {}

        try:

            defaults = pdef.get("defaults", {}) or {}

        except Exception:

            defaults = {}


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


        p_style = ""

        p_negs = ""

        try:

            p_style = str(pdef.get("style", "") or "").strip()

        except Exception:

            p_style = ""

        try:

            p_negs = str(pdef.get("negatives", "") or "").strip()

        except Exception:

            p_negs = ""


        overrides = {

            "target": "image",

            "preset": preset,

            "last_used_preset": preset,

        }


        if p_style:

            overrides["style"] = (f"{cur_style}, {p_style}".strip(", ") if cur_style else p_style)

        if p_negs:

            overrides["negatives"] = _merge_neg_csv(cur_negs, p_negs) if cur_negs else p_negs


        try:

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

    def _on_enhance_prompt_clicked(self):
        """Enhance the current prompt using the external Qwen helper without blocking the UI."""
        # Run the Qwen3-VL prompt helper from its own .venv and replace the prompt/negative text.
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

        # Prevent re-entry if a helper is already running
        try:
            proc = getattr(self, "_qwen_prompt_proc", None)
            if proc is not None and proc.state() != QProcess.NotRunning:
                try:
                    QMessageBox.information(self, "Prompt enhancer", "Qwen prompt enhancer is already running.")
                except Exception:
                    pass
                return
        except Exception:
            pass

        try:
            neg = (self.negative.toPlainText() or "").strip()
        except Exception:
            neg = ""

        try:
            import json, os
            from pathlib import Path as _P
        except Exception as e:
            try:
                QMessageBox.critical(self, "Prompt enhancer", f"Missing standard modules: {e}")
            except Exception:
                pass
            return

        # Resolve app root + helpers dir
        try:
            here = _P(__file__).resolve()
            helpers_dir = here.parent
            app_root = helpers_dir.parent
        except Exception:
            try:
                app_root = _P.cwd()
                helpers_dir = app_root / "helpers"
            except Exception:
                app_root = _P.cwd()


                # Temporarily apply selected PromptTool preset + force IMAGE target for this enhancement run.
        _prompttool_restore = None
        try:
            st = {}
            try:
                st_path = self._prompttool_settings_path(app_root)
                if st_path.exists():
                    st = json.loads(st_path.read_text(encoding="utf-8"))
            except Exception:
                st = {}
            overrides = self._prompttool_overrides_for_preset(st)
            _prompttool_restore = self._prompttool_apply_overrides(app_root, overrides)
        except Exception:
            _prompttool_restore = None

        # Locate dedicated .venv Python
        py_candidates = []
        try:
            venv = app_root / ".venv"
            win_py = venv / "Scripts" / "python.exe"
            nix_py = venv / "bin" / "python"
            if win_py.exists():
                py_candidates.append(win_py)
            if nix_py.exists():
                py_candidates.append(nix_py)
        except Exception:
            pass
        py_path = None
        for c in py_candidates:
            try:
                if c.exists():
                    py_path = c
                    break
            except Exception:
                continue
        if py_path is None:
            try:
                QMessageBox.critical(
                    self,
                    "Prompt enhancer",
                    "Could not find a dedicated .venv Python.\n"
                    "Expected .venv/Scripts/python.exe or .venv/bin/python next to the app folder."
                )
            except Exception:
                pass
            return

        cli_path = helpers_dir / "prompt_enhancer_cli.py"
        if not cli_path.exists():
            try:
                QMessageBox.critical(
                    self,
                    "Prompt enhancer",
                    "helpers/prompt_enhancer_cli.py is missing.\n"
                    "Please copy the helper script into the helpers/ folder."
                )
            except Exception:
                pass
            return

        # Build args (no shell) — QProcess runs async, so the UI stays responsive.
        args = [str(cli_path), "--seed", base_prompt]
        if neg:
            args += ["--neg", neg]

        # Update UI state
        try:
            self._qwen_prompt_prev_status = self.status.text()
        except Exception:
            self._qwen_prompt_prev_status = ""
        try:
            self.status.setText("Enhancing prompt (Qwen)…")
        except Exception:
            pass
        try:
            self._qwen_prompt_btn_text = self.btn_prompt_enhance.text()
            self.btn_prompt_enhance.setText("Enhancing…")
            self.btn_prompt_enhance.setEnabled(False)
        except Exception:
            pass

        # Buffers for output
        try:
            self._qwen_prompt_stdout = bytearray()
            self._qwen_prompt_stderr = bytearray()
        except Exception:
            self._qwen_prompt_stdout = None
            self._qwen_prompt_stderr = None

        # Start QProcess
        try:
            proc = QProcess(self)
            proc.setProgram(str(py_path))
            proc.setArguments(args)
            proc.setWorkingDirectory(str(app_root))

            try:
                env = QProcessEnvironment.systemEnvironment()
                env.insert("PYTHONUTF8", "1")
                proc.setProcessEnvironment(env)
            except Exception:
                pass

            def _read_out():
                try:
                    chunk = proc.readAllStandardOutput()
                    b = bytes(chunk)
                    if getattr(self, "_qwen_prompt_stdout", None) is not None:
                        self._qwen_prompt_stdout.extend(b)
                except Exception:
                    pass

            def _read_err():
                try:
                    chunk = proc.readAllStandardError()
                    b = bytes(chunk)
                    if getattr(self, "_qwen_prompt_stderr", None) is not None:
                        self._qwen_prompt_stderr.extend(b)
                except Exception:
                    pass

            def _cleanup_ui():
                try:
                    self._prompttool_restore_settings(_prompttool_restore)
                except Exception:
                    pass

                try:
                    self.btn_prompt_enhance.setEnabled(True)
                except Exception:
                    pass
                try:
                    t = getattr(self, "_qwen_prompt_btn_text", None)
                    if t:
                        self.btn_prompt_enhance.setText(t)
                except Exception:
                    pass

            def _finalize(exit_code: int):
                try:
                    out_txt = bytes(getattr(self, "_qwen_prompt_stdout", b"")).decode("utf-8", "ignore").strip()
                except Exception:
                    out_txt = ""
                try:
                    err_txt = bytes(getattr(self, "_qwen_prompt_stderr", b"")).decode("utf-8", "ignore").strip()
                except Exception:
                    err_txt = ""

                if int(exit_code) != 0:
                    msg = err_txt or out_txt or f"Exit code {exit_code}"
                    if len(msg) > 2000:
                        msg = msg[:2000] + "…"
                    try:
                        QMessageBox.critical(self, "Prompt enhancer", "Qwen prompt helper failed:\n\n" + msg)
                    except Exception:
                        pass
                    try:
                        self.status.setText("Prompt enhancer failed")
                    except Exception:
                        pass
                    return

                data = None
                try:
                    data = json.loads(out_txt)
                except Exception:
                    data = None
                if not isinstance(data, dict) or not data.get("ok"):
                    msg = out_txt or "Unexpected response from helper."
                    if len(msg) > 2000:
                        msg = msg[:2000] + "…"
                    try:
                        QMessageBox.critical(
                            self,
                            "Prompt enhancer",
                            "Qwen prompt helper returned an unexpected payload:\n\n" + msg
                        )
                    except Exception:
                        pass
                    try:
                        self.status.setText("Prompt enhancer returned bad output")
                    except Exception:
                        pass
                    return

                new_prompt = data.get("prompt") or ""
                new_neg = data.get("negatives") or ""
                if new_prompt:
                    try:
                        self.prompt.setPlainText(new_prompt)
                    except Exception:
                        pass
                if new_neg:
                    try:
                        self.negative.setPlainText(new_neg)
                    except Exception:
                        pass
                try:
                    self.status.setText("Prompt enhanced with Qwen3-VL")
                except Exception:
                    pass

            def _on_finished(exit_code, _exit_status):
                try:
                    _read_out()
                    _read_err()
                except Exception:
                    pass
                try:
                    _cleanup_ui()
                except Exception:
                    pass
                try:
                    _finalize(int(exit_code))
                finally:
                    try:
                        proc.deleteLater()
                    except Exception:
                        pass
                    try:
                        self._qwen_prompt_proc = None
                    except Exception:
                        pass

            def _on_error(_err):
                try:
                    _read_out()
                    _read_err()
                except Exception:
                    pass
                try:
                    _cleanup_ui()
                except Exception:
                    pass
                try:
                    msg = bytes(getattr(self, "_qwen_prompt_stderr", b"")).decode("utf-8", "ignore").strip()
                except Exception:
                    msg = ""
                if not msg:
                    msg = "Qwen prompt helper crashed or failed to start."
                if len(msg) > 2000:
                    msg = msg[:2000] + "…"
                try:
                    QMessageBox.critical(self, "Prompt enhancer", msg)
                except Exception:
                    pass
                try:
                    self.status.setText("Prompt enhancer failed")
                except Exception:
                    pass
                try:
                    proc.deleteLater()
                except Exception:
                    pass
                try:
                    self._qwen_prompt_proc = None
                except Exception:
                    pass

            proc.readyReadStandardOutput.connect(_read_out)
            proc.readyReadStandardError.connect(_read_err)
            proc.finished.connect(_on_finished)
            proc.errorOccurred.connect(_on_error)

            self._qwen_prompt_proc = proc
            proc.start()
        except Exception as e:
            try:
                QMessageBox.critical(self, "Prompt enhancer", f"Failed to start Qwen helper: {e}")
            except Exception:
                pass
            try:
                self.btn_prompt_enhance.setEnabled(True)
            except Exception:
                pass
            try:
                t = getattr(self, "_qwen_prompt_btn_text", None)
                if t:
                    self.btn_prompt_enhance.setText(t)
            except Exception:
                pass
            try:
                self._qwen_prompt_proc = None
            except Exception:
                pass
            return

    def _on_browse(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_path.text())
        if d:
            self.output_path.setText(d)

    def _collect_job(self) -> dict:
        seed = int(self.seed.value())
        if self.seed_policy.currentIndex() == 1:  # random
            seed = random.randint(0, 2_147_483_647)
        data = self.size_combo.currentData() if hasattr(self, "size_combo") else None
        w_ui, h_ui = (data if data else (self.size_manual_w.value() if hasattr(self, "size_manual_w") else 1024,
                                         self.size_manual_h.value() if hasattr(self, "size_manual_h") else 1024))
        # Determine engine key for defaults
        engine = (self.engine_combo.currentData() if hasattr(self, "engine_combo") else "diffusers") or "diffusers"
        engine_key = str(engine).lower().strip() or "diffusers"
        # Determine filename template; special default for Z-Image
        fname = self.filename_template.text().strip() if hasattr(self, "filename_template") else ""
        if not fname:
            try:
                fmt = (self.format_combo.currentText() if hasattr(self, "format_combo") else "png") or "png"
            except Exception:
                fmt = "png"
            try:
                fmt = str(fmt).strip().lower() or "png"
            except Exception:
                fmt = "png"
            if engine_key.startswith("zimage"):
                fname = f"z_img_{{seed}}_{{idx:03d}}.{fmt}"
            else:
                fname = f"IMG_{{seed}}.{fmt}"
        job = {
            "type": "txt2img",
            "engine": engine_key,
            "prompt": self.prompt.toPlainText().strip(),
            "negative": self.negative.toPlainText().strip(),
            "init_image_enabled": bool(getattr(self, "zimg_init_enable", None).isChecked()) if getattr(self, "zimg_init_enable", None) is not None else False,
            "init_image": (getattr(self, "zimg_init_path", None).text().strip() if getattr(self, "zimg_init_path", None) is not None else ""),
            "img2img_strength": float(getattr(self, "zimg_strength", None).value()) if getattr(self, "zimg_strength", None) is not None else 0.35,
            "seed": seed,
            "seed_policy": ["fixed","random","increment"][self.seed_policy.currentIndex()],
            "batch": int(self.batch.value()),
            "cfg_scale": float(self.cfg_scale.value()) if hasattr(self, "cfg_scale") else 7.5,
            "output": self.output_path.text().strip(),
            "show_in_player": self.show_in_player.isChecked(),
            "use_queue": bool(self.use_queue.isChecked()),
            "vram_profile": self.vram_profile.currentText(),
            "sd_cli_path": (self.sd_cli_path.text().strip() if hasattr(self, "sd_cli_path") else ""),
            "sampler": self.sampler.currentText(),
            "flow_shift": (int(self.qwen_flow_shift.value()) if hasattr(self, "qwen_flow_shift") else 3),
            "offload_cpu": (bool(self.qwen_offload_cpu.isChecked()) if hasattr(self, "qwen_offload_cpu") else False),
            "model_path": (self.model_combo.currentData() if hasattr(self, "model_combo") else "auto"),
            "lora_path": (self.lora_combo.currentData() if hasattr(self, "lora_combo") else ""),
            "lora_a_scale": (self.lora_a_strength.value() if hasattr(self, "lora_a_strength") else 1.0),
            "lora_b_scale": (self.lora_b_strength.value() if hasattr(self, "lora_b_strength") else 1.0),

            "lora_scale": (self.lora_strength.value() if hasattr(self, "lora_strength") else 1.0),
            "lora2_path": (self.lora2_combo.currentData() if hasattr(self, "lora2_combo") else ""),
            "lora2_scale": (self.lora2_strength.value() if hasattr(self, "lora2_strength") else 1.0),
            "attn_slicing": self.attn_slicing.isChecked(),
            "vae_device": self.vae_device.currentText(),
            "gpu_index": int(self.gpu_index.value()),
            "threads": int(self.threads.value()),
                        "format": self.format_combo.currentText(),
            "filename_template": fname,
            "hires_helper": (bool(self.hires_helper.isChecked()) if hasattr(self, "hires_helper") else False),
            "fit_check": self.fit_check.isChecked(),
            "steps": int(self.steps_slider.value()),
            "created_at": time.time(),
            "size_preset_index": int(self.size_combo.currentIndex()) if hasattr(self,"size_combo") else -1,
            "size_preset_label": str(self.size_combo.currentText()) if hasattr(self,"size_combo") else "",
            "width": int(w_ui),
            "height": int(h_ui),

            "preset": (self.preset_combo.currentText() if (engine_key == "diffusers" and hasattr(self, "preset_combo")) else ""),
            "preset_index": (int(self.preset_combo.currentIndex()) if (engine_key == "diffusers" and hasattr(self, "preset_combo")) else -1),
            "a1111_url": self._a1111_url_removed.text().strip() if hasattr(self, "_a1111_url_removed") else "http://127.0.0.1:7860",
                    }
        # Engine-specific queue metadata to avoid SDXL model bleed into Z-Image jobs
        try:
            ek = str(job.get("engine") or "").lower().strip()
            if ek.startswith("zimage"):
                # Z-Image uses its own internal model management; don't carry SDXL model_path
                job.pop("model_path", None)
                # If this is GGUF mode, route GGUF selections through existing LoRA args
                # so the worker does not need special-case plumbing.
                if ek == "zimage_gguf":
                    try:
                        gm = (self.gguf_model_combo.currentData() if hasattr(self, "gguf_model_combo") else "") or ""
                        gv = (self.gguf_vae_combo.currentData() if hasattr(self, "gguf_vae_combo") else "") or ""
                        # Store for debugging/JSON display
                        job["gguf_model_path"] = gm
                        job["gguf_vae_path"] = gv
                        # Reuse lora/lora2 fields as transport to zimage_cli (GGUF backend ignores actual LoRA)
                        job["lora_path"] = gm
                        job["lora2_path"] = gv
                        # GGUF backend does not support init images (img2img); force-disable to avoid checkerboards
                        try:
                            job["init_image_enabled"] = False
                            job["init_image"] = ""
                            job["img2img_strength"] = 0.35
                        except Exception:
                            pass
                    except Exception:
                        pass
                job.setdefault("model_name", "Z-Image-Turbo")
                job.setdefault("model", "Z-Image-Turbo")
            elif ek == "qwen2512":
                # qwen uses its own GGUF diffusion model; don't carry SDXL model_path
                job.pop("model_path", None)
                try:
                    wm = (self.qwen_model_combo.currentData() if hasattr(self, "qwen_model_combo") else "") or ""
                    job["qwen_model_path"] = wm
                except Exception:
                    pass
                try:
                    job["init_image_enabled"] = False
                    job["init_image"] = ""
                    job["img2img_strength"] = 0.35
                except Exception:
                    pass
                job.setdefault("model_name", "qwen 2.5 12B GGUF")
                job.setdefault("model", "qwen 2.5 12B GGUF")

        except Exception:
            pass

                # Persist settings
        try:
            self._save_settings(job)
        except Exception as e:
            print("[txt2img] warn: could not save settings:", e)
        try:
            self._save_settings(job)
        except Exception as e:
            print('[txt2img] warn: could not save settings:', e)
        return job

        # Compute per-image seeds deterministically (unchanged)...

    
    
    def _save_settings(self, job: dict):
        """Write the given dict to presets/setsave/txt2img.json."""
        try:
            p = self._settings_path()
            with open(p, "w", encoding="utf-8") as f:
                json.dump(job, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            try: print("[txt2img] save settings error:", e)
            except Exception: pass
            return False

    def _load_settings(self):
        """Load JSON from presets/setsave/txt2img.json and apply to UI."""
        try:
            p = self._settings_path()
            if not p.exists():
                return False
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            try:
                self._t2i_last_loaded = dict(data)
            except Exception:
                self._t2i_last_loaded = data
            self._apply_settings_from_dict(data)
            try:
                QTimer.singleShot(0, lambda: self._apply_settings_from_dict(getattr(self,'_t2i_last_loaded', data)))
            except Exception:
                pass
            return True
        except Exception as e:
            try: print("[txt2img] load settings error:", e)
            except Exception: pass
            return False
            s = json.loads(pth.read_text(encoding='utf-8') or '{}')
        except Exception as e:
            try:
                print('[txt2img] load settings error:', e)
            except Exception:
                pass
            return
        # fill UI (best-effort)
        try:
            self.prompt.setPlainText(s.get('prompt','') or '')
            self.negative.setPlainText(s.get('negative','') or '')
        except Exception:
            pass
        try:
            self.seed.setValue(int(s['seed'])) if 'seed' in s else None
            sp = str(s['seed_policy']).lower() if 'seed_policy' in s else None
            idx = {'fixed':0,'random':1,'increment':2}.get(sp,0)
            self.seed_policy.setCurrentIndex(idx) if sp is not None else None
            self.batch.setValue(int(s['batch'])) if 'batch' in s else None
        except Exception:
            pass
        try:
            self.output_path.setText(s['output']) if 'output' in s else None
            self.show_in_player.setChecked(bool(s['show_in_player'])) if 'show_in_player' in s else None
            if hasattr(self, 'use_queue'):
                self.use_queue.setChecked(bool(s['use_queue'])) if 'use_queue' in s else None
        except Exception:
            pass
        try:
            w = int(s['width']) if 'width' in s else None; h = int(s['height']) if 'height' in s else None
            idx = -1
            for i,(label,wv,hv) in enumerate(self._size_presets):
                if wv==w and hv==h:
                    idx = i; break
            if (w is not None and h is not None) and idx>=0 and hasattr(self, 'size_combo'):
                self.size_combo.setCurrentIndex(idx)
            elif (w is not None and h is not None):
                if hasattr(self, 'size_manual_w'): self.size_manual_w.setValue(w)
                if hasattr(self, 'size_manual_h'): self.size_manual_h.setValue(h)
        except Exception:
            pass
        try:
            self.steps_slider.setValue(int(s['steps'])) if 'steps' in s and hasattr(self,'steps_slider') else None
            self.cfg_scale.setValue(float(s['cfg_scale'])) if 'cfg_scale' in s and hasattr(self,'cfg_scale') else None
        except Exception:
            pass
        try:
            if hasattr(self, 'vram_profile'): self.vram_profile.setCurrentText(s['vram_profile']) if 'vram_profile' in s else None
            if hasattr(self, 'sampler'): self.sampler.setCurrentText(s['sampler']) if 'sampler' in s else None
            if hasattr(self, 'attn_slicing'): self.attn_slicing.setChecked(bool(s['attn_slicing'])) if 'attn_slicing' in s else None
            if hasattr(self, 'vae_device'): self.vae_device.setCurrentText(str(s['vae_device'])) if 'vae_device' in s else None
            if hasattr(self, 'gpu_index'): self.gpu_index.setValue(int(s['gpu_index'])) if 'gpu_index' in s else None
            if hasattr(self, 'threads'): self.threads.setValue(int(s['threads'])) if 'threads' in s else None
            if hasattr(self, 'format_combo'): self.format_combo.setCurrentText(str(s['format'])) if 'format' in s else None
            if hasattr(self, 'filename_template'): self.filename_template.setText(s['filename_template']) if 'filename_template' in s else None
            if hasattr(self, 'hires_helper'): self.hires_helper.setChecked(bool(s['hires_helper'])) if 'hires_helper' in s else None
            if hasattr(self, 'fit_check'): self.fit_check.setChecked(bool(s.get('fit_check', True)))
        except Exception:
            pass
        try:
            if hasattr(self, 'model_combo'):
                mp = s.get('model_path','')
                if mp:
                    i = self.model_combo.findData(mp)
                    if i >= 0: self.model_combo.setCurrentIndex(i)
            if hasattr(self, 'lora_combo'):
                lp = s.get('lora_path','')
                if lp:
                    i = self.lora_combo.findData(lp)
                    if i >= 0: self.lora_combo.setCurrentIndex(i)
            if hasattr(self, 'lora_strength'):
                try: self.lora_strength.setValue(float(s.get('lora_scale', 1.0)))
                except Exception: pass
            if hasattr(self, 'lora2_combo'):
                lp2 = s.get('lora2_path','')
                if lp2:
                    i2 = self.lora2_combo.findData(lp2)
                    if i2 >= 0: self.lora2_combo.setCurrentIndex(i2)
            if hasattr(self, 'lora2_strength'):
                try: self.lora2_strength.setValue(float(s.get('lora2_scale', 1.0)))
                except Exception: pass
        except Exception:
            pass
        return
    def _snapshot_ui(self) -> dict:
        d = {}
        try: d["prompt"] = self.prompt.toPlainText().strip()
        except Exception: pass
        try: d["negative"] = self.negative.toPlainText().strip()
        except Exception: pass
        try: d["init_image_enabled"] = bool(getattr(self, "zimg_init_enable", None).isChecked()) if getattr(self, "zimg_init_enable", None) is not None else False
        except Exception: pass
        try: d["init_image"] = getattr(self, "zimg_init_path", None).text().strip() if getattr(self, "zimg_init_path", None) is not None else ""
        except Exception: pass
        try: d["img2img_strength"] = float(getattr(self, "zimg_strength", None).value()) if getattr(self, "zimg_strength", None) is not None else 0.35
        except Exception: pass
        try: d["seed"] = int(self.seed.value())
        except Exception: pass
        try:
            idx = int(self.seed_policy.currentIndex())
            d["seed_policy"] = ["fixed","random","increment"][idx if 0 <= idx < 3 else 0]
        except Exception: pass
        try: d["batch"] = int(self.batch.value())
        except Exception: pass
        try:
            if hasattr(self,'size_combo') and self.size_combo.currentIndex() >= 0:
                _, w, h = self._size_presets[self.size_combo.currentIndex()]
                d["width"] = int(w); d["height"] = int(h)
            else:
                if hasattr(self,'size_manual_w'): d["width"] = int(self.size_manual_w.value())
                if hasattr(self,'size_manual_h'): d["height"] = int(self.size_manual_h.value())
        except Exception: pass
        try: d["steps"] = int(self.steps_slider.value())
        except Exception: pass
        try: d["cfg_scale"] = float(self.cfg_scale.value())
        except Exception: pass
        try: d["sampler"] = self.sampler.currentText()
        except Exception: pass
        try:
            if hasattr(self, "qwen_flow_shift"):
                d["flow_shift"] = int(self.qwen_flow_shift.value())
        except Exception: pass
        try:
            if hasattr(self,'model_combo'): d["model_path"] = self.model_combo.currentData()
        except Exception: pass
        try:
            if hasattr(self,'lora_combo'): d["lora_path"] = self.lora_combo.currentData()
        except Exception: pass
        try:
            if hasattr(self,'lora_strength'): d["lora_scale"] = float(self.lora_strength.value())
        except Exception: pass
        try:
            if hasattr(self,'lora2_combo'): d["lora2_path"] = self.lora2_combo.currentData()
        except Exception: pass
        try:
            if hasattr(self,'lora2_strength'): d["lora2_scale"] = float(self.lora2_strength.value())
        except Exception: pass
        try: d["attn_slicing"] = bool(self.attn_slicing.isChecked())
        except Exception: pass
        try: d["vae_device"] = self.vae_device.currentText()
        except Exception: pass
        try: d["gpu_index"] = int(self.gpu_index.value())
        except Exception: pass
        try: d["threads"] = int(self.threads.value())
        except Exception: pass
        # Remember current engine (Z-Image / SDXL etc.) so we restore it on restart
        try:
            if hasattr(self, "engine_combo"):
                eng = None
                try:
                    eng = self.engine_combo.currentData()
                except Exception:
                    eng = None
                if not eng:
                    try:
                        eng = self.engine_combo.currentText()
                    except Exception:
                        eng = None
                if eng is not None:
                    try:
                        d["engine"] = str(eng)
                    except Exception:
                        d["engine"] = eng
        except Exception:
            pass
        try: d["format"] = self.format_combo.currentText()
        except Exception: pass
        try: d["filename_template"] = self.filename_template.text().strip()
        except Exception: pass
        try: d["hires_helper"] = bool(self.hires_helper.isChecked())
        except Exception: pass
        try: d["fit_check"] = bool(self.fit_check.isChecked())
        except Exception: pass
        try: d["use_queue"] = bool(self.use_queue.isChecked())
        except Exception: pass
        try: d["show_in_player"] = bool(self.show_in_player.isChecked())
        except Exception: pass
        try: d["vram_profile"] = self.vram_profile.currentText()
        except Exception: pass
        try: d["output"] = self.output_path.text().strip()
        except Exception: pass
        try: d["a1111_url"] = self._a1111_url_removed.text().strip()
        except Exception: pass
        try: d["qwen_cli_template"] = self._qwen_cli_template_removed.text().strip()
        except Exception: pass
        # Only persist image-gen presets for SD (diffusers). Z-Image/Qwen ignore them.
        try:
            ek = None
            try:
                ek = self.engine_combo.currentData() if hasattr(self, "engine_combo") else None
            except Exception:
                ek = None
            if not ek:
                try:
                    ek = self.engine_combo.currentText() if hasattr(self, "engine_combo") else None
                except Exception:
                    ek = None
            ek = str(ek or "").lower().strip()
            is_diff = (ek == "diffusers") or ("sd" in ek and "zimage" not in ek and "qwen" not in ek)
        except Exception:
            is_diff = True
        try: d["preset"] = (self.preset_combo.currentText() if is_diff else "")
        except Exception: pass
        try: d["preset_index"] = (int(self.preset_combo.currentIndex()) if is_diff else -1)
        except Exception: pass

        try:
            d["size_preset_index"] = int(self.size_combo.currentIndex()) if hasattr(self, "size_combo") else -1
        except Exception:
            pass

        return d

    def _autosave_now(self):
        try:
            self._save_settings(self._snapshot_ui())
        except Exception as e:
            try: print("[txt2img] autosave error:", e)
            except Exception: pass

    def _connect_autosave(self):
        def connect_sig(obj, sig):
            try:
                getattr(obj, sig).connect(lambda *a, **k: self._autosave_now())
            except Exception:
                pass
        connect_sig(self.prompt, 'textChanged')
        connect_sig(self.negative, 'textChanged')
        connect_sig(self.preset_combo, 'currentIndexChanged')
        for obj, sig in [
            (self.seed, 'valueChanged'),
            (self.seed_policy, 'currentIndexChanged'),
            (self.batch, 'valueChanged'),
            (self.size_combo, 'currentIndexChanged'),
            (getattr(self,'size_manual_w', None), 'valueChanged'),
            (getattr(self,'size_manual_h', None), 'valueChanged'),
            (self.steps_slider, 'valueChanged'),
            (self.cfg_scale, 'valueChanged'),
            (self.sampler, 'currentIndexChanged'),
            (getattr(self,'qwen_flow_shift', None), 'valueChanged'),
            (getattr(self,'model_combo', None), 'currentIndexChanged'),
            (getattr(self,'lora_combo', None), 'currentIndexChanged'),
            (getattr(self,'lora_strength', None), 'valueChanged'),
            (getattr(self,'lora2_combo', None), 'currentIndexChanged'),
            (getattr(self,'lora2_strength', None), 'valueChanged'),
            (self.attn_slicing, 'toggled'),
            (self.vae_device, 'currentIndexChanged'),
            (self.gpu_index, 'valueChanged'),
            (self.threads, 'valueChanged'),
            (self.format_combo, 'currentIndexChanged'),
            (self.filename_template, 'textChanged'),
            (getattr(self,'hires_helper', None), 'toggled'),
            (self.fit_check, 'toggled'),
            (self.use_queue, 'toggled'),
            (self.show_in_player, 'toggled'),
            (self.vram_profile, 'currentIndexChanged'),
            (self.output_path, 'textChanged'),
            (getattr(self, '_a1111_url_removed', None), 'textChanged'),
            (getattr(self, '_qwen_cli_template_removed', None), 'textChanged'),
        ]:
            if obj is None: 
                continue
            connect_sig(obj, sig)
        try: print("Applied all settings")
        except Exception: pass

    # === Busy indicator (indeterminate; no ETA) ===
    def _start_busy(self, watch_dir: str = None, watch_single: bool = False, label: str = "Generating…"):
        try:
            self._busy_active = True
            self.progress.setRange(0, 0)  # indeterminate
            self.progress.setTextVisible(False)
        except Exception:
            pass
        try:
            self.status.setText(label)
        except Exception:
            pass
        try:
            self._busy_idx = 0
            self._busy_timer.start()
        except Exception:
            pass
        # Optional: stop on first image if single-image run
        try:
            self._busy_watch_dir = watch_dir
            self._busy_watch_single = bool(watch_single)
            self._busy_watch_t0 = time.time()
            if self._busy_watch_single and self._busy_watch_dir and self._busy_fs_timer:
                self._busy_fs_timer.start()
        except Exception:
            pass

    def _stop_busy(self, done: bool = False):
        try:
            self._busy_timer.stop()
        except Exception:
            pass
        try:
            if getattr(self, "_busy_fs_timer", None):
                self._busy_fs_timer.stop()
        except Exception:
            pass
        self._busy_active = False
        try:
            self.progress.setRange(0, 100)
            self.progress.setValue(100 if done else 0)
            self.progress.setTextVisible(True)
            self.progress.setFormat("Done" if done else "Stopped")
        except Exception:
            pass
        try:
            self.status.setText("Ready" if done else "Stopped")
        except Exception:
            pass

    
    def _on_busy_tick(self):
        # Stop immediately if something else already marked us done.
        try:
            if self.progress.minimum() == 0 and self.progress.maximum() != 0:
                try:
                    fmt = self.progress.format()
                except Exception:
                    fmt = ""
                try:
                    val = int(self.progress.value())
                except Exception:
                    val = 0
                if val >= 100 or ("Done" in str(fmt)):
                    self._stop_busy(done=True)
                    try:
                        self.status.setText("Ready")
                    except Exception:
                        pass
                    return
        except Exception:
            pass
        except Exception:
            pass

        if not getattr(self, "_busy_active", False):
            return
        try:
            self._busy_idx = (self._busy_idx + 1) % len(self._busy_frames)
            frame = self._busy_frames[self._busy_idx]
            self.status.setText(f"{frame} Generating…")
        except Exception:
            pass

    def _on_busy_fs_tick(self):
        try:
            if not self._busy_watch_single or not self._busy_watch_dir:
                return
            d = Path(self._busy_watch_dir)
            if not d.exists():
                return
            t0 = float(self._busy_watch_t0 or 0.0)
            for ext in ("*.png","*.jpg","*.jpeg","*.webp"):
                for f in d.glob(ext):
                    try:
                        if f.stat().st_mtime >= t0:
                            self._stop_busy(done=True)
                            return
                    except Exception:
                        continue
        except Exception:
            pass

    def _stop_busy(self, done: bool = False):
        try:
            # Return to determinate to show final state (0% or 100%)
            self.progress.setRange(0, 100)
            self.progress.setValue(100 if done else 0)
            self.progress.setTextVisible(True)
            self.progress.setFormat("Done" if done else "Stopped")
        except Exception:
            pass


    def _enqueue(self, run_now: bool):
            try:
                from helpers.queue_adapter import enqueue_txt2img
            except Exception:
                enqueue_txt2img = None
            job = self._collect_job()
            if not job or not job.get('prompt'):
                try:
                    self.status.setText('Please enter a prompt')
                except Exception:
                    pass
                return
            if not job or not job.get('prompt'):
                self.status.setText("Prompt is empty")
                return
            pass  # queue decision handled in _on_generate_clicked
            ok = False
            if enqueue_txt2img:
                ok = bool(enqueue_txt2img(job | {"run_now": bool(run_now)}))
            if ok:
                seeds = job.get("seeds") or []
                seed_info = (f" | seed {seeds[0]}" if seeds else "")
                size_info = f" | {job.get('width',1024)}x{job.get('height',1024)}"
                self.status.setText("Enqueued" + (" and running…" if run_now else "") + seed_info + size_info)
            else:
                self.status.setText("Enqueue failed")

    def _on_generate_clicked(self):
        job = self._collect_job()
        if not job or not job.get('prompt'):
            try: self.status.setText('Please enter a prompt')
            except Exception: pass
            return

        # SDXL safety: if SDXL is selected but models are missing, guide user to Optional Downloads.
        try:
            if not self._ensure_models_installed():
                return
        except Exception:
            pass
        try:
            use_q = bool(self.use_queue.isChecked())
        except Exception:
            use_q = False

        # Queue logic:
        # - If user enabled "Use queue", enqueue normally (do NOT jump ahead).
        # - If batch>1 without queue toggle, we still queue but allow "run now" priority.
        batch_n = int(job.get('batch', 1) or 1)
        should_queue = (batch_n > 1) or use_q
        run_now_flag = bool((batch_n > 1) and (not use_q))

        if should_queue:
            try:
                from helpers.queue_adapter import enqueue_txt2img
            except Exception:
                enqueue_txt2img = None
            if enqueue_txt2img and enqueue_txt2img(job | {'run_now': run_now_flag}):
                try:
                    self.status.setText('Enqueued' + (' and running…' if run_now_flag else ''))
                except Exception:
                    pass
            else:
                try: self.status.setText('Enqueue failed')
                except Exception: pass
        else:
            self._run_direct(job)


    def _run_direct(self, job: dict):
        # Ensure a sane default filename template for Diffusers runs
        if not job.get('filename_template'):
            job['filename_template'] = 'sd_{seed}_{idx:03d}.png'

        try:
            self.status.setText("Generating…")
        except Exception:
            pass
        # Start indeterminate busy animation (watch output dir if single image)
        try:
            out_dir = None
            try:
                out_dir = str(Path(self.output_path.text()).resolve())
            except Exception:
                out_dir = None
            watch_single = (int(job.get('batch',1)) == 1)
            self._start_busy(watch_dir=out_dir, watch_single=watch_single)
        except Exception:
            try:
                self._start_busy()
            except Exception:
                pass

        cancel_flag = threading.Event()

        def progress_cb(p):
            return  # no per-step UI

        def worker():
            try:
                res = generate_qwen_images(job, progress_cb=progress_cb, cancel_event=cancel_flag)
                if res.get("ok"):
                    try:
                        QTimer.singleShot(0, lambda: self._stop_busy(done=True))
                    except Exception:
                        self._stop_busy(done=True)
                    try:
                        if job.get("show_in_player") and res.get("files"):
                            self.fileReady.emit(res["files"][-1])
                    except Exception:
                        pass
                else:
                    try:
                        QTimer.singleShot(0, lambda: self._stop_busy(done=False))
                    except Exception:
                        self._stop_busy(done=False)
                    try:
                        self.status.setText("Failed")
                    except Exception:
                        pass
            except Exception as e:
                try:
                    QTimer.singleShot(0, lambda: self._stop_busy(done=False))
                except Exception:
                    self._stop_busy(done=False)
                try:
                    self.status.setText(f"Error: {e}")
                except Exception:
                    pass
            finally:
                # Safety: ensure busy stopped
                def _finalize():
                    if getattr(self, "_busy_active", False):
                        self._stop_busy(done=False)
                try:
                    QTimer.singleShot(0, _finalize)
                except Exception:
                    _finalize()

        threading.Thread(target=worker, daemon=True).start()


def _draw_text_image(text: str, size=(1024,1024), seed: int = 0) -> QImage:
    """CPU fallback placeholder: checkerboard + centered prompt text."""
    w, h = size
    img = QImage(w, h, QImage.Format_RGB32)
    p = QPainter(img)
    tile = 32
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            color = Qt.lightGray if ((x//tile + y//tile) % 2) else Qt.gray
            p.fillRect(x, y, tile, tile, color)
    p.setRenderHint(QPainter.Antialiasing, True)
    rect = QRect(40, 40, w-80, h-80) if 'QRect' in globals() else None
    try:
        from PySide6.QtCore import QRect as _QRect
        rect = _QRect(40, 40, w-80, h-80)
        font = p.font(); font.setPointSize(22); font.setBold(True); p.setFont(font)
        p.setPen(Qt.white); p.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, text or "No prompt")
    except Exception:
        pass
    p.end()
    return img



def _gen_via_diffusers(job: dict, out_dir: Path, progress_cb=None):
    """
    Diffusers backend with detailed error reporting and CPU fallback.
    """

    # Ensure PyTorch uses non-Flash SDPA kernels on Windows builds (avoid 'not compiled with flash attention' warnings)
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    try:
        import os
        from pathlib import Path as _Path
        import importlib
        diffusers = importlib.import_module("diffusers")
        torch = importlib.import_module("torch")
        from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, HeunDiscreteScheduler, UniPCMultistepScheduler, DDIMScheduler
    except Exception:
        return None  # diffusers/torch not installed

    # Choose model path: job override -> ENV -> default DreamShaper in models/SD15
    root = _Path(__file__).resolve().parent.parent
    default_model = root / "models" / "SD15" / "DreamShaper_8_pruned.safetensors"

    # Resolve model path with strong guards
    cand = job.get("model_path")
    envp = os.environ.get("FRAMEVISION_DIFFUSERS_MODEL")
    if cand is None or str(cand).strip() == "":
        cand = envp
    if cand is None or str(cand).strip() == "":
        cand = str(default_model)

    # Normalize; if this resolves to "." or a directory or a non-existing path, use default
    _cand_norm = str(_Path(cand))
    try:
        _p = _Path(_cand_norm)
        if _cand_norm in ("", ".", "./") or (_p.exists() and _p.is_dir()) or (not _p.exists() and not _cand_norm.lower().endswith(".safetensors")):
            _cand_norm = str(default_model)
    except Exception:
        _cand_norm = str(default_model)
    model_path = _cand_norm


    # Basic params
    prompt   = job.get("prompt","")
    negative = job.get("negative","")
    steps    = int(job.get("steps", 30))
    seed     = int(job.get("seed", 0))
    batch    = int(job.get("batch", 1))
    width    = int(job.get("width", 1024))
    height   = int(job.get("height", 1024))

    try:
        # Load single-file SD1.5 checkpoint
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.float16 if device == "cuda" else torch.float32
        is_sdxl = ("sdxl" in model_path.lower()) or ("sd_xl" in model_path.lower())
        if is_sdxl:
            try:
                pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=dtype, local_files_only=True)
            except Exception:
                pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=dtype, local_files_only=True)
        else:
            pipe = StableDiffusionPipeline.from_single_file(model_path, torch_dtype=dtype, local_files_only=True)
        pipe = pipe.to(device)

        
        # Optionally load up to two LoRAs for SDXL (from UI slots 1 & 2)
        lora_mode = None
        lora_names = []
        loras_to_load = []
        scales = []
        try:
            lora1 = str(job.get("lora_path") or "").strip()
            lora2 = str(job.get("lora2_path") or "").strip()
            s1 = float(job.get("lora_scale", 1.0) or 1.0)
            s2 = float(job.get("lora2_scale", 1.0) or 1.0)
            if lora1 and Path(lora1).exists():
                loras_to_load.append(lora1); scales.append(s1)
            if lora2 and Path(lora2).exists():
                loras_to_load.append(lora2); scales.append(s2)
            if loras_to_load and is_sdxl:
                try:
                    lora_names = [f"lora_{i+1}" for i in range(len(loras_to_load))]
                    for path, name in zip(loras_to_load, lora_names):
                        try:
                            pipe.load_lora_weights(path, adapter_name=name)
                        except TypeError:
                            pipe.load_lora_weights(path)
                    if hasattr(pipe, "set_adapters"):
                        pipe.set_adapters(lora_names, scales)
                        lora_mode = "adapters"
                        try:
                            print(f"[txt2img] LoRA applied via adapters: {list(zip(lora_names, scales))}")
                        except Exception:
                            pass
                    else:
                        raise RuntimeError("set_adapters missing")
                except Exception as e:
                    print("[txt2img] adapter set failed, fallback fuse", e)
                    try:
                        if hasattr(pipe, "fuse_lora"):
                            if len(loras_to_load) >= 1:
                                try:
                                    pipe.load_lora_weights(loras_to_load[0])
                                except Exception:
                                    pass
                                pipe.fuse_lora(lora_scale=scales[0])
                            if len(loras_to_load) >= 2:
                                try:
                                    pipe.load_lora_weights(loras_to_load[1])
                                except Exception:
                                    pass
                                pipe.fuse_lora(lora_scale=scales[1])
                            lora_mode = "fused"
                            try:
                                print(f"[txt2img] LoRA applied via fuse: {list(zip(loras_to_load, scales))}")
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception as _e:
            try:
                print("[txt2img] LoRA load failed:", _e)
            except Exception:
                pass

        # apply selected sampler if not auto
        try:
            name = (job.get("sampler") or "auto").lower()
            sched_map = {
                "dpm++ 2m (karras)": lambda cfg: DPMSolverMultistepScheduler.from_config(cfg, algorithm_type="dpmsolver++", use_karras_sigmas=True),
                "euler a": lambda cfg: EulerAncestralDiscreteScheduler.from_config(cfg),
                "euler": lambda cfg: EulerDiscreteScheduler.from_config(cfg),
                "heun": lambda cfg: HeunDiscreteScheduler.from_config(cfg),
                "unipc": lambda cfg: UniPCMultistepScheduler.from_config(cfg),
                "ddim": lambda cfg: DDIMScheduler.from_config(cfg),
            }
            if name in sched_map:
                pipe.scheduler = sched_map[name](pipe.scheduler.config)
        except Exception:
            pass

        # Disable safety checker for local use
        try:
            if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
                pipe.safety_checker = (lambda images, **kwargs: (images, [False]*len(images)))
            if hasattr(pipe, "requires_safety_checker"):
                pipe.requires_safety_checker = False
        except Exception:
            pass

        # Memory-friendly tweaks
        try:
            if job.get('attn_slicing'):
                pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pass
        except Exception:
            pass

        gen = torch.Generator(device=device)
        if seed:
            gen = gen.manual_seed(seed)

        files = []
        
        for i in range(batch):
            # Initialize step progress at the start of each image
            try:
                if progress_cb:
                    progress_cb({"step": i * max(1, steps), "total": max(1, steps*batch)})
            except Exception:
                pass
            _cb = None
            if progress_cb:
                def _cb(step, timestep, latents, _i=i, _steps=steps, _batch=batch):
                    try:
                        base = _i * max(1, _steps)
                        cur = base + (step + 1)
                        progress_cb({"step": cur, "total": max(1, _steps*_batch)})
                    except Exception:
                        pass
            result = pipe(
                prompt=prompt,
                negative_prompt=negative or None,
                num_inference_steps=max(1, steps),
                guidance_scale=float(job.get("cfg_scale", 7.5)),
                width=width, height=height,
                generator=gen,
                callback=_cb, callback_steps=1
            )
            img = result.images[0]
            fn_tmpl = job.get("filename_template") or "IMG_{seed}.png"
            fname = fn_tmpl.format(seed=(seed if seed else 0)+i, idx=i)
            if not fname.lower().endswith((".png",".jpg",".jpeg",".webp")):
                try:
                    _fmt = str(job.get("format","png")).strip().lower() or "png"
                except Exception:
                    _fmt = "png"
                fname += f".{_fmt}"
            fpath = out_dir / fname
            fpath = _unique_path(fpath)
            img.save(str(fpath))
            files.append(str(fpath))
            try:
                if progress_cb:
                    progress_cb({"step": (i+1)*max(1, steps), "total": max(1, steps*batch)})
            except Exception:
                pass
        # After finishing a Diffusers SD15/SDXL run, try to free as much
        # GPU memory as possible from this process so other engines (like Z-Image
        # in its own environment) can see more available VRAM.
        try:
            import torch, gc  # type: ignore
            try:
                gc.collect()
            except Exception:
                pass
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

        return {"files": files, "backend": "diffusers", "model": model_path, "actual_size": [width, height], "lora_mode": (lora_mode or ""), "loras": loras_to_load, "lora_scales": scales, "lora_names": lora_names}
    except Exception as e:
        import traceback
        err = {"backend":"diffusers","error":str(e),"trace": traceback.format_exc()}
        _record_result(job, out_dir, err)
        return None



def _run_qwen_cli(job: dict, out_dir: Path, tpl: str, progress_cb=None):
    """Run an external CLI for Qwen image generation using a user-provided command template."""
    try:
        import os, subprocess, shlex
        prompt = job.get("prompt",""); neg = job.get("negative","")
        w = int(job.get("width",1024)); h = int(job.get("height",1024))
        steps = int(job.get("steps",30)); seed = int(job.get("seed",0)); batch = int(job.get("batch",1))
        files = []
        for i in range(batch):
            fname = (job.get("filename_template") or "qwen_{seed}_{idx:03d}.png").format(seed=seed+i, idx=i)
            if not fname.lower().endswith(("png","jpg","jpeg","webp")):
                try:
                    _fmt = str(job.get("format","png")).strip().lower() or "png"
                except Exception:
                    _fmt = "png"
                fname += f".{_fmt}"
            fpath = out_dir / fname
            fpath = _unique_path(fpath)
            model = job.get("model_path","")
            cmd = tpl.format(prompt=prompt, neg=neg, w=w, h=h, steps=steps, seed=seed+i, out=str(fpath), model=model)
            try:
                args = shlex.split(cmd)
            except Exception:
                args = cmd  # let shell parse if split fails
            proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=isinstance(args, str))
            if proc.returncode != 0:
                try:
                    print("[qwen-cli] stderr:", proc.stderr.decode("utf-8","ignore"))
                except Exception:
                    pass
                return None
            files.append(str(fpath))
            if progress_cb: progress_cb((i+1)/max(1,batch))
        return {"files": files, "backend": "qwen-cli"}
    except Exception as e:
        try:
            print("[qwen-cli] failed:", e)
        except Exception:
            pass
        return None

def _gen_via_a1111(job: dict, out_dir: Path, base_url: str, progress_cb=None):
    if requests is None:
        return None
    try:
        # quick health check
        r = requests.get(base_url + "/sdapi/v1/sd-models", timeout=2)
    except Exception:
        return None
    prompt = job.get("prompt",""); neg = job.get("negative","")
    w = int(job.get("width", 1024)); h = int(job.get("height", 1024))
    steps = int(job.get("steps", 30)); seed = int(job.get("seed", 0))
    batch = int(job.get("batch", 1))
    payload = {"prompt": prompt, "negative_prompt": neg, "width": w, "height": h, "steps": steps, "seed": seed, "batch_size": batch}
    try:
        resp = requests.post(base_url + "/sdapi/v1/txt2img", json=payload, timeout=240)
        resp.raise_for_status()
        data = resp.json()
        images_b64 = data.get("images") or []
        import base64, io
        from PIL import Image
        files = []
        for i, b64 in enumerate(images_b64):
            if not isinstance(b64, str): continue
            raw = base64.b64decode(b64)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            fname = (job.get("filename_template") or "qwen_{seed}_{idx:03d}.png").format(seed=seed+i, idx=i)
            fpath = out_dir / fname
            fpath = _unique_path(fpath)
            img.save(str(fpath))
            files.append(str(fpath))
            if progress_cb: progress_cb({"step": (i+1)*max(1, steps), "total": max(1, steps*batch)})
        return {"files": files, "backend": "a1111"}
    except Exception:
        return None

def _gen_via_zimage(job: dict, out_dir: Path, progress_cb=None):
    """
    Z-Image Turbo backend that runs in its own virtualenv (.zimage_env) as an
    external process, so it can have independent deps (torch/diffusers).

    It calls helpers/zimage_cli.py using the python.exe from:
        <root>/.zimage_env/Scripts/python.exe   (Windows)
        <root>/.zimage_env/scripts/python.exe   (alt)
        <root>/.zimage_env/bin/python           (Linux/macOS)

    The CLI returns a JSON payload with a list of files.
    """
    # Before calling into the external Z-Image env, try to free
    # as much CUDA VRAM as possible from this process (e.g. Qwen VL).
    try:
        _aggressive_free_cuda_vram()
    except Exception:
        pass

    try:
        from pathlib import Path as _P
        import subprocess, json, shlex, os as _os
    except Exception:
        return None

    # Resolve root dir (app root = parent of helpers/)
    try:
        root_dir = _P(__file__).resolve().parents[1]
    except Exception:
        root_dir = Path(".").resolve()

    # Locate dedicated Z-Image env python
    candidates = [
        root_dir / ".zimage_env" / "Scripts" / "python.exe",
        root_dir / ".zimage_env" / "scripts" / "python.exe",
        root_dir / ".zimage_env" / "bin" / "python",
    ]
    pyexe = None
    for c in candidates:
        try:
            if c.exists():
                pyexe = c
                break
        except Exception:
            continue
    if pyexe is None:
        try:
            print("[txt2img] Z-Image: python.exe not found in .zimage_env; tried:", candidates)
        except Exception:
            pass
        return None

    # Helper CLI script
    cli = root_dir / "helpers" / "zimage_cli.py"
    if not cli.exists():
        try:
            print("[txt2img] Z-Image: helpers/zimage_cli.py not found at", cli)
        except Exception:
            pass
        return None

    # Basic job params
    prompt = str(job.get("prompt") or "")
    neg = str(job.get("negative") or "")
    batch = int(job.get("batch", 1) or 1)
    steps = int(job.get("steps", 9) or 9)
    cfg = float(job.get("cfg_scale", 0.0) or 0.0)
    width = int(job.get("width", 1024) or 1024)
    height = int(job.get("height", 1024) or 1024)
    seed = int(job.get("seed", 0) or 0)
    fmt = (job.get("format", "png") or "png").lower().strip()
    if fmt not in ("png", "jpg", "jpeg", "webp", "bmp"):
        fmt = "png"
    fname_tmpl = job.get("filename_template") or "zimage_{seed}_{idx:03d}.png"

    # Build command
    args = [
        str(pyexe),
        str(cli),
        "--prompt", prompt,
        "--negative", neg,
        "--height", str(height),
        "--width", str(width),
        "--steps", str(steps),
        "--guidance", str(cfg),
        "--seed", str(seed),
        "--batch", str(batch),
        "--outdir", str(out_dir),
        "--fmt", fmt,
        "--filename_template", fname_tmpl,
    ]

    # Optional: image-to-image (init image) for Z-Image (FP16 only) to keep identity/composition
    try:
        init_img = (job.get("init_image") or "").strip()
    except Exception:
        init_img = ""
    try:
        init_enabled = bool(job.get("init_image_enabled", True))
    except Exception:
        init_enabled = True
    # GGUF backend: init images are not supported; force-disable
    try:
        if str(job.get("engine") or "").strip().lower() == "zimage_gguf":
            init_enabled = False
    except Exception:
        pass
    if init_enabled and init_img:
        try:
            strength = float(job.get("img2img_strength", 0.35))
        except Exception:
            strength = 0.35
        # Clamp to sane range
        try:
            strength = max(0.0, min(1.0, strength))
        except Exception:
            strength = 0.35
        args += ["--init-image", str(init_img), "--strength", str(strength)]

    # Backend switch: Diffusers (default) vs GGUF (stable-diffusion.cpp)
    try:
        eng0 = str(job.get("engine") or "").strip().lower()
        if eng0 == "zimage_gguf":
            args += ["--backend", "gguf"]
    except Exception:
        pass


    # Pass attention slicing flag through to Z-Image CLI so low-VRAM
    # users can enable it from the Advanced settings checkbox.
    try:
        if bool(job.get("attn_slicing")):
            args.append("--attn-slicing")
    except Exception:
        pass

    # Pass LoRA selection through to Z-Image CLI (slots 1 & 2).
    # NOTE: previously LoRAs were only wired for the Diffusers SDXL backend.
    try:
        from pathlib import Path as _Path
        l1 = str(job.get("lora_path") or "").strip()
        l2 = str(job.get("lora2_path") or "").strip()
        s1 = float(job.get("lora_scale", 1.0) or 1.0)
        s2 = float(job.get("lora2_scale", 1.0) or 1.0)

        # Resolve relative paths against the project root (helpers/ -> root).
        try:
            _root = _Path(__file__).resolve().parents[1]
        except Exception:
            _root = _Path(".")

        def _resolve_existing(p: str):
            try:
                pp = _Path(p)
                if not pp.is_absolute():
                    pp = (_root / pp).resolve()
                return pp if pp.exists() else None
            except Exception:
                return None

        p1 = _resolve_existing(l1) if l1 else None
        if p1 is not None:
            args += ["--lora", str(p1), "--lora_scale", str(s1)]

        p2 = _resolve_existing(l2) if l2 else None
        if p2 is not None:
            args += ["--lora2", str(p2), "--lora2_scale", str(s2)]
    except Exception:
        pass


    # Stream CLI output so queue mode can show live progress (tqdm step X/Y).
    payload = None
    last_json_line = None
    log_file = None
    code = -1

    # Ensure out_dir exists before launching
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Best-effort log file (helps debugging in queue mode)
    try:
        base = None
        try:
            base = ROOT  # type: ignore[name-defined]
        except Exception:
            base = Path(__file__).resolve().parents[1]  # helpers/ -> project root
        log_dir = Path(base) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / (f"zimage_gguf_{stamp}.log" if str(job.get("engine","")).strip().lower()=="zimage_gguf" else f"zimage_{stamp}.log")
    except Exception:
        log_file = None

    try:
        import re as _re
        step_re = _re.compile(r"\b(\d{1,6})\s*/\s*(\d{1,6})\b")
        pct_re = _re.compile(r"\b(\d{1,3})%\b")
        last_step = -1
        last_pct = -1

        try:
            if progress_cb:
                progress_cb({"pct": 0, "status": "zimage"})
        except Exception:
            pass

        with (open(log_file, "w", encoding="utf-8", errors="replace") if log_file else None) as lf:
            try:
                if lf:
                    lf.write("CMD: " + " ".join([str(x) for x in args]) + "\n\n")
                    lf.flush()
            except Exception:
                pass

            # Optional: pass sd-cli.exe / sd.exe override to the Z-Image CLI for GGUF backend
            _env = None
            try:
                import os as _os
                eng0 = str(job.get("engine") or "").strip().lower()
                sel_cli = str(job.get("sd_cli_path") or "").strip()
                pcli = None

                if sel_cli:
                    from pathlib import Path as _P2
                    pcli = _P2(sel_cli)
                    if not pcli.is_absolute():
                        try:
                            _root = _P2(__file__).resolve().parents[1]
                        except Exception:
                            _root = _P2(".").resolve()
                        pcli = (_root / pcli).resolve()
                    if not (pcli.exists() and pcli.is_file()):
                        pcli = None

                # Auto default for GGUF engines: <root>/presets/bin/{sd-cli.exe, sd.exe}
                if pcli is None and eng0 == "zimage_gguf":
                    try:
                        pcli = _preferred_sd_cli_from_presets_bin(root_dir)
                    except Exception:
                        pcli = None

                if pcli is not None:
                    _env = _os.environ.copy()
                    _env["SD_CLI_PATH"] = str(pcli)
                    _env["SD_CLI"] = str(pcli)
            except Exception:
                _env = None

            _popen_kwargs = dict(
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                if _env is not None:
                    _popen_kwargs["env"] = _env
            except Exception:
                pass

            proc = subprocess.Popen(
                args,
                **_popen_kwargs,
            )

            if proc.stdout:
                for line in proc.stdout:
                    try:
                        if lf:
                            lf.write(line)
                            lf.flush()
                    except Exception:
                        pass

                    s = (line or "").strip()
                    if not s:
                        continue

                    # Capture JSON payload if the CLI prints it as a line (usually at the end)
                    if s.startswith("{") and s.endswith("}"):
                        last_json_line = s
                        try:
                            payload = json.loads(s)
                            continue
                        except Exception:
                            pass

                    # Parse tqdm-style progress "X/Y" and percent
                    try:
                        sm = step_re.search(s)
                        if sm and ":" not in sm.group(0):
                            step = int(sm.group(1))
                            total = int(sm.group(2))
                            if total > 0 and step != last_step:
                                last_step = step
                                if progress_cb:
                                    progress_cb({"step": step, "total": total, "status": "zimage"})
                                try:
                                    print(f"[txt2img] Z-Image {step}/{total}")
                                except Exception:
                                    pass
                        pm = pct_re.search(s)
                        if pm:
                            pct = int(pm.group(1))
                            if 0 <= pct <= 100 and pct != last_pct:
                                last_pct = pct
                                if progress_cb:
                                    progress_cb({"pct": pct, "status": "zimage"})
                    except Exception:
                        pass

            code = proc.wait()
            try:
                if lf:
                    lf.write(f"\nEXIT CODE: {code}\n")
                    lf.flush()
            except Exception:
                pass

    except Exception as e:
        try:
            print("[txt2img] Z-Image: failed to launch/stream CLI:", e)
        except Exception:
            pass
        return None

    # If CLI failed, dump a hint and abort
    if int(code) != 0:
        try:
            print("[txt2img] Z-Image CLI failed (see log):", str(log_file) if log_file else "(no log)")
        except Exception:
            pass
        return None

    # Fallback: parse last JSON line if we didn't decode it live
    if payload is None and last_json_line:
        try:
            payload = json.loads(last_json_line)
        except Exception:
            payload = None

    if not payload:
        return None

    return {
        "files": payload.get("files") or [],
        "backend": "zimage",
        "model": payload.get("model") or "Z-Image-Turbo",
        "log_file": (str(log_file) if log_file else None),
    }



def _gen_via_qwen2512(job: dict, out_dir: Path, progress_cb=None, cancel_event: Optional[threading.Event] = None):
    """qwen 2.5 12B GGUF backend (stable-diffusion.cpp sd-cli)."""
    try:
        _aggressive_free_cuda_vram()
    except Exception:
        pass

    import subprocess
    import random
    import time
    from pathlib import Path as _P

    try:
        root_dir = _P(__file__).resolve().parents[1]
    except Exception:
        root_dir = _P(".").resolve()

    try:
        from helpers import qwen2512 as _qwen
    except Exception:
        try:
            import qwen2512 as _qwen
        except Exception:
            _qwen = None
    if _qwen is None:
        return None

    try:
        _pref = _preferred_sd_cli_from_presets_bin(root_dir)
        sd_cli = _pref if _pref else _qwen._find_sd_cli(root_dir)
        # Optional UI/job override: allow selecting an explicit sd-cli.exe path
        try:
            sel_cli = str(job.get("sd_cli_path") or "").strip()
            if sel_cli:
                pcli = _P(sel_cli)
                if not pcli.is_absolute():
                    pcli = (root_dir / pcli).resolve()
                if pcli.exists() and pcli.is_file():
                    sd_cli = pcli
        except Exception:
            pass
        diffusion, llm, vae = _qwen._model_paths(root_dir)
        if not (sd_cli and diffusion and llm and vae):
            return None
        # Optional UI/job override: allow selecting an alternate qwen diffusion GGUF
        try:
            sel = str(job.get("qwen_model_path") or "").strip()
            if sel:
                psel = _P(sel)
                if not psel.is_absolute():
                    psel = (root_dir / psel).resolve()
                if psel.exists() and psel.is_file() and psel.suffix.lower() == ".gguf":
                    diffusion = psel
        except Exception:
            pass

    except Exception:
        return None

    prompt = str(job.get("prompt", "") or "")

    # Optional LoRA support: sd-cli parses <lora:NAME:W> tags when --lora-model-dir is set.
    lora_path = str(job.get("lora_path", "") or "").strip()
    lora2_path = str(job.get("lora2_path", "") or "").strip()
    lora_scale = job.get("lora_scale", None)
    lora2_scale = job.get("lora2_scale", None)
    lora_apply_mode = str(job.get("lora_apply_mode", "") or "at_runtime").strip() or "at_runtime"
    lora_model_dir = ""
    def _fmt_lora_tag(_path: str, _scale) -> str:
        if not _path:
            return ""
        try:
            _w = float(_scale) if _scale is not None else 1.0
        except Exception:
            _w = 1.0
        _name = Path(_path).stem
        # Use :g to avoid '1.0' style spam in the prompt, but keep precision when needed.
        try:
            _w_s = (f"{_w:g}")
        except Exception:
            _w_s = str(_w)
        return f"<lora:{_name}:{_w_s}>"
    try:
        _dirs = []
        for _lp in (lora_path, lora2_path):
            if _lp:
                try:
                    _dirs.append(str(Path(_lp).resolve().parent))
                except Exception:
                    _dirs.append(str(Path(_lp).parent))
        if _dirs:
            import os as _os
            _common = _os.path.commonpath(_dirs)
            if _common and Path(_common).exists():
                lora_model_dir = _common
            else:
                lora_model_dir = _dirs[0]
    except Exception:
        lora_model_dir = str(Path(lora_path).parent) if lora_path else (str(Path(lora2_path).parent) if lora2_path else "")
    try:
        _tags = []
        _t1 = _fmt_lora_tag(lora_path, lora_scale)
        _t2 = _fmt_lora_tag(lora2_path, lora2_scale)
        if _t1: _tags.append(_t1)
        if _t2: _tags.append(_t2)
        if _tags:
            prompt = " ".join(_tags) + " " + prompt
    except Exception:
        pass

    # Some sd-cli Windows builds assert/crash on non-ASCII bytes in -p.
    try:
        _sp = getattr(_qwen, "_sanitize_prompt_ascii", None)
        prompt_cli = _sp(prompt) if callable(_sp) else prompt.encode("ascii", "ignore").decode("ascii", "ignore")
    except Exception:
        prompt_cli = prompt

    w = int(job.get("width", 1024)); h = int(job.get("height", 1024))
    steps = int(job.get("steps", 40))
    cfg = float(job.get("cfg_scale", 2.5))
    seed_policy = str(job.get("seed_policy", "fixed") or "fixed").lower().strip()
    batch = int(job.get("batch", 1))
    fmt = str(job.get("format", "png") or "png").lower().strip()
    fname_tmpl = str(job.get("filename_template") or f"qwen_{{seed}}_{{idx:03d}}.{fmt}")

    ui_sampler = str(job.get("sampler", "") or "").strip()
    sampler_map = {
        "Euler a": "euler_a",
        "Euler": "euler",
        "Heun": "heun",
        "DPM++ 2M (Karras)": "dpm++2m",
        "DPM++ 2M": "dpm++2m",
        "DPM++ 2M SDE (Karras)": "dpm++2m_sde",
        "DPM++ 2M SDE": "dpm++2m_sde",
        "auto": "euler",
        "UniPC": "euler",
        "DDIM": "euler",
    }
    sampler = sampler_map.get(ui_sampler, "euler")

    try:
        flow_shift = int(job.get("flow_shift", 3) or 3)
    except Exception:
        flow_shift = 3

    try:
        # Respect explicit Qwen toggle when present; otherwise infer from VRAM profile
        if "offload_cpu" in job:
            offload_cpu = bool(job.get("offload_cpu"))
        else:
            vp = str(job.get("vram_profile", "") or "").lower()
            offload_cpu = ("low" in vp) or ("very" in vp)
    except Exception:
        offload_cpu = False

    seeds_list = job.get("seeds")
    if not seeds_list:
        seed0 = int(job.get("seed", 0) or 0)
        if seed_policy == "fixed":
            seeds_list = [seed0 for _ in range(batch)]
        elif seed_policy == "increment":
            seeds_list = [seed0 + i for i in range(batch)]
        else:
            rng = random.Random(seed0 if seed0 else int(time.time()))
            seeds_list = [rng.randint(0, 2_147_483_647) for _ in range(batch)]

    files = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(int(batch)):
        if cancel_event is not None and cancel_event.is_set():
            break
        s0 = int(seeds_list[i] if i < len(seeds_list) else job.get("seed", 0) or 0)
        try:
            fname = fname_tmpl.format(seed=s0, idx=i)
        except Exception:
            fname = f"qwen_{s0}_{i:03d}.{fmt}"
        if not str(fname).lower().endswith(("png", "jpg", "jpeg", "webp")):
            fname = str(fname).rstrip(".") + f".{fmt}"
        out_path = _unique_path(out_dir / fname)

        cmd = [
            str(sd_cli),
            "--diffusion-model", str(diffusion),
            "--vae", str(vae),
            "--llm", str(llm),
            "--sampling-method", str(sampler),
            "--cfg-scale", str(cfg),
            "--steps", str(steps),
            "-W", str(w),
            "-H", str(h),
            "--diffusion-fa",
            "--flow-shift", str(flow_shift),
            "-p", prompt_cli,
            "-o", str(out_path),
            "--seed", str(s0),
        ]
        # LoRA flags (optional)
        if lora_model_dir and (lora_path or lora2_path):
            cmd += ["--lora-model-dir", str(lora_model_dir), "--lora-apply-mode", str(lora_apply_mode)]
        if offload_cpu:
            cmd.append("--offload-to-cpu")

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        except Exception:
            return None

        try:
            if proc.stdout is not None:
                for _line in proc.stdout:
                    if cancel_event is not None and cancel_event.is_set():
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        break
        except Exception:
            pass

        try:
            code = proc.wait()
        except Exception:
            code = 1
        if int(code) != 0:
            return None
        if not out_path.exists():
            return None
        files.append(str(out_path))
        if progress_cb:
            try:
                progress_cb({"step": i + 1, "total": max(1, batch)})
            except Exception:
                pass

    if not files:
        return None
    return {"files": files, "backend": "qwen2512", "model": f"qwen 2.5 12B GGUF — {_P(str(diffusion)).name}"}

def generate_qwen_images(job: dict, progress_cb: Optional[Callable[[float], None]] = None, cancel_event: Optional[threading.Event] = None):
    """Generator that saves images and returns metadata. Uses CPU fallback to validate pipeline end-to-end."""
    out_dir = Path(job.get("output") or "./output/images"); out_dir.mkdir(parents=True, exist_ok=True)
    batch = int(job.get("batch", 1)); seed = int(job.get("seed", 0))
    seed_policy = job.get("seed_policy", "fixed"); fmt = job.get("format", "png").lower().strip()
    steps = int(job.get("steps", 30))
    req_w = int(job.get("width", 1024)); req_h = int(job.get("height", 1024))
    # Z-Image: auto-correct resolutions in the fragile band around 1080p on the short side.
    # Many turbo builds produce NaNs for non-square sizes where min(W,H) is ~1000–1150.
    # To keep things stable, we gently downscale such sizes so the short side becomes ~960px.
    try:
        engine_key = str(job.get("engine") or "").strip().lower()
        if engine_key == "zimage":
            short_side = min(req_w, req_h)
            if 960 < short_side < 1200:
                scale = 960.0 / float(short_side)
                def _snap64(v: float) -> int:
                    return max(256, int(round(v / 64.0) * 64))
                req_w = _snap64(req_w * scale)
                req_h = _snap64(req_h * scale)
    except Exception:
        pass
    # Snap to a safe bucket that typical T2I models like (multiples of 64, near known presets)
    safe_w, safe_h = _safe_snap_size(req_w, req_h)
    fname_tmpl = job.get("filename_template", "qwen_{seed}_{idx:03d}.png"); prompt = job.get("prompt", "")
    # Determine per-image seeds
    seeds_list = job.get("seeds")
    if not seeds_list:
        if seed_policy == "fixed":
            seeds_list = [seed for _ in range(batch)]
        elif seed_policy == "increment":
            seeds_list = [seed + i for i in range(batch)]
        else:
            rng = random.Random(seed if seed else int(time.time()))
            seeds_list = [rng.randint(0, 2_147_483_647) for _ in range(batch)]
    files = []
    engine = str(job.get("engine") or "").strip().lower()
    # Try Diffusers or Z-Image backend if available
    try:
        if engine in ("zimage","zimage_gguf"):
            diff = _gen_via_zimage(job, out_dir, progress_cb)
        elif engine == "qwen2512":
            # Use snapped safe sizes to keep CLI stable
            try:
                job2 = dict(job)
                job2["width"] = int(safe_w)
                job2["height"] = int(safe_h)
            except Exception:
                job2 = job
            diff = _gen_via_qwen2512(job2, out_dir, progress_cb, cancel_event)
        else:
            diff = _gen_via_diffusers(job, out_dir, progress_cb)

        if diff and diff.get('files'):
            files = diff['files']
            meta_backend = diff.get('backend') or 'diffusers'
            meta_model = diff.get('model')
            meta = {"ok": True, "prompt": prompt, "negative": job.get("negative",""),
                    "seed": seed, "seed_policy": seed_policy, "batch": batch, "files": files,
                    "created_at": time.time(), "engine": meta_backend, "model": meta_model,
                    "vram_profile": job.get("vram_profile","auto"), "steps": steps, "seeds": seeds_list,
                    "requested_size": [req_w, req_h], "actual_size": [safe_w, safe_h], "lora_mode": (diff.get("lora_mode") if diff else None), "loras": (diff.get("loras") if diff else None), "lora_scales": (diff.get("lora_scales") if diff else None), "lora_names": (diff.get("lora_names") if diff else None)}
            try:
                with open(out_dir / (Path(files[0]).stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            return meta
    except Exception:
        pass
    for i in range(batch):
        if cancel_event and cancel_event.is_set(): break
        s = seeds_list[i] if i < len(seeds_list) else seed
        # Simulate per-step progress for fallback
        for _step in range(max(1, steps)):
            try:
                if progress_cb:
                    progress_cb({"step": i*max(1, steps) + (_step+1), "total": max(1, steps*batch)})
            except Exception:
                pass
        img = _draw_text_image(prompt, size=(safe_w, safe_h), seed=s)
        fname = fname_tmpl.format(seed=s, idx=i)
        if not fname.lower().endswith(("png","jpg","jpeg","webp")): fname += "." + fmt
        fpath = out_dir / fname
        fpath = _unique_path(fpath)
        img.save(str(fpath)); files.append(str(fpath))
        if progress_cb: progress_cb(((i+1)/batch))
        time.sleep(0.02)
    meta = {"ok": True, "prompt": prompt, "negative": job.get("negative",""), "seed": seed,
            "seed_policy": seed_policy, "batch": batch, "files": files, "created_at": time.time(),
            "engine": "fallback", "vram_profile": job.get("vram_profile","auto"), "steps": steps, "seeds": seeds_list,
            "requested_size": [req_w, req_h], "actual_size": [safe_w, safe_h], "lora_mode": (diff.get("lora_mode") if diff else None), "loras": (diff.get("loras") if diff else None), "lora_scales": (diff.get("lora_scales") if diff else None), "lora_names": (diff.get("lora_names") if diff else None)}
    try:
        with open(out_dir / (Path(files[0]).stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return meta
# <<< FRAMEVISION_QWEN_END

# --- Tiny helper: open a result in the main player or OS viewer (shared with Recents) ---
try:
    from pathlib import Path as _PathForOpen
except Exception:
    _PathForOpen = None

def _play_in_player(self, p):
    """Try to open a media path in the app's internal player if available; return True on success.

    This mirrors the internal-player behavior used by the qwen2.2 tab:
    - Prefer main.video.open(...)
    - Fall back to main.open_video(...)
    - Keep main.current_path and media info in sync when possible.
    """
    # Normalize to a Path-like object when possible
    try:
        from pathlib import Path as _Path
    except Exception:
        _Path = None

    try:
        global _PathForOpen
    except Exception:
        _PathForOpen = None

    try:
        if _PathForOpen is not None and not isinstance(p, _PathForOpen):
            try:
                p = _PathForOpen(str(p))
            except Exception:
                if _Path is not None:
                    try:
                        p = _Path(str(p))
                    except Exception:
                        pass
        elif _Path is not None and not isinstance(p, _Path):
            try:
                p = _Path(str(p))
            except Exception:
                pass
    except Exception:
        pass

    # Find the main window that owns the media player
    m = None
    try:
        m = getattr(self, "_main", None) or getattr(self, "main", None)
    except Exception:
        m = None

    if m is None:
        try:
            from PySide6.QtWidgets import QApplication
            for w in QApplication.topLevelWidgets():
                if hasattr(w, "video") and hasattr(getattr(w, "video", None), "open"):
                    m = w
                    break
        except Exception:
            m = None

    if m is None:
        return False

    # 1) Preferred: main.video.open(...)
    try:
        player = getattr(m, "video", None)
        if player is not None and hasattr(player, "open"):
            try:
                player.open(p)
            except TypeError:
                # Some builds expect a plain string path
                player.open(str(p))
            try:
                if _Path is not None:
                    m.current_path = _Path(str(p))
                else:
                    m.current_path = p
            except Exception:
                pass
            try:
                from helpers.mediainfo import refresh_info_now
                refresh_info_now(p)
            except Exception:
                pass
            return True
    except Exception:
        pass

    # 2) Legacy hook: main.open_video(...)
    try:
        if hasattr(m, "open_video"):
            m.open_video(str(p))
            return True
    except Exception:
        pass

    return False

def _open_file(self, p):
    """Fallback: open a path via the OS (Explorer / Finder / xdg-open)."""
    try:
        import os, subprocess
        if _PathForOpen is not None and not isinstance(p, _PathForOpen):
            try:
                p = _PathForOpen(str(p))
            except Exception:
                pass
        if os.name == "nt":
            os.startfile(str(p))  # nosec - user initiated
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception:
        pass


# === Tiny Txt2Img deferred-load patch (minimal & safe) ===
try:
    from PySide6.QtCore import QTimer  # type: ignore
except Exception:
    QTimer = None  # type: ignore

def _t2i_store_path(self):
    # presets/setsave/txt2img.json under app root; fallback to CWD
    try:
        from pathlib import Path as _P
        app_root = _P(__file__).resolve().parent.parent
        p = app_root / 'presets' / 'setsave' / 'txt2img.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        pass
    try:
        p = _P.cwd() / 'presets' / 'setsave' / 'txt2img.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return None

def _t2i_apply_from_dict(self, s: dict):
    # Block signals while applying so handlers don't overwrite loaded values
    widgets = [
        'prompt','negative','seed','seed_policy','batch','size_manual_w','size_manual_h',
        'steps_slider','cfg_scale','sampler','model_combo','lora_combo','lora_strength',
        'lora2_combo','lora2_strength','attn_slicing','vae_device','gpu_index','threads',
        'format_combo','filename_template','hires_helper','fit_check',
        'use_queue','show_in_player','vram_profile','output_path'
    ]
    blocked = []
    for name in widgets:
        w = getattr(self, name, None)
        try:
            if w is not None and hasattr(w, 'blockSignals'):
                w.blockSignals(True); blocked.append(w)
        except Exception:
            pass
    try:
        # Text fields
        if hasattr(self, 'prompt') and 'prompt' in s: self.prompt.setPlainText(s.get('prompt') or '')
        if hasattr(self, 'negative') and 'negative' in s: self.negative.setPlainText(s.get('negative') or '')
        # Basic nums
        if hasattr(self, 'seed') and 'seed' in s: self.seed.setValue(int(s.get('seed') or 0))
        if hasattr(self, 'seed_policy') and 'seed_policy' in s:
            sp = str(s['seed_policy']).lower() if 'seed_policy' in s else None
            idx = {'fixed':0,'random':1,'increment':2}.get(sp,0)
            self.seed_policy.setCurrentIndex(idx) if sp is not None else None
        if hasattr(self, 'batch') and 'batch' in s: self.batch.setValue(int(s.get('batch') or 1))
        # Size
        # Prefer explicit preset index if present
        if 'size_preset_index' in s and hasattr(self,'size_combo'):
            try:
                idx = int(s['size_preset_index'])
                if 0 <= idx < self.size_combo.count():
                    self.size_combo.blockSignals(True)
                    self.size_combo.setCurrentIndex(idx)
                    self.size_combo.blockSignals(False)
            except Exception:
                pass
        # Fallback: match width/height to a preset
        w = s.get('width'); h = s.get('height')
        if w is not None and h is not None and hasattr(self,'size_combo'):
            try:
                idx = -1
                for i,(label,wv,hv) in enumerate(self._size_presets):
                    if int(w)==wv and int(h)==hv:
                        idx = i; break
                if idx >= 0:
                    self.size_combo.blockSignals(True)
                    self.size_combo.setCurrentIndex(idx)
                    self.size_combo.blockSignals(False)
            except Exception:
                pass
        # Always set manual boxes too (kept in sync by combo change handler)
        if hasattr(self,'size_manual_w') and w is not None: self.size_manual_w.setValue(int(w))
        if hasattr(self,'size_manual_h') and h is not None: self.size_manual_h.setValue(int(h))
        # Sampler/steps/cfg
        if hasattr(self, 'sampler') and 'sampler' in s and s.get('sampler'): self.sampler.setCurrentText(str(s.get('sampler')))
        if hasattr(self, 'steps_slider') and 'steps' in s: self.steps_slider.setValue(int(s.get('steps') or 0))
        if hasattr(self, 'cfg_scale') and 'cfg_scale' in s: self.cfg_scale.setValue(float(s.get('cfg_scale') or 0))
        # Paths & toggles
        if hasattr(self, 'output_path') and 'output' in s and s.get('output'): self.output_path.setText(str(s.get('output')))
        if hasattr(self, 'show_in_player') and 'show_in_player' in s: self.show_in_player.setChecked(bool(s.get('show_in_player')))
        if hasattr(self, 'use_queue') and 'use_queue' in s: self.use_queue.setChecked(bool(s.get('use_queue')))
    except Exception as e:
        try: print('[txt2img] apply settings warn:', e)
        except Exception: pass
    finally:
        for w in blocked:
            try: w.blockSignals(False)
            except Exception: pass

def _t2i_load_settings(self):
    self._t2i_loading = True
    try:
        p = _t2i_store_path(self)
        if p is None or not p.exists():
            return False
        import json
        with open(p, 'r', encoding='utf-8') as f:
            data = json.load(f) or {}
        if isinstance(data, dict):
            _t2i_apply_from_dict(self, data)
            return True
        return False
    except Exception as e:
        try: print('[txt2img] load settings error:', e)
        except Exception: pass
        return False
    finally:
        self._t2i_loading = False


# === Minimal persistence patch (no bloat): guard saves during load + map missing fields ===
try:
    _T2I = Txt2ImgPane

    _orig_load = _T2I._load_settings
    def _load_settings_patched(self):
        try:
            self._t2i_loading = True
        except Exception:
            pass
        try:
            return _orig_load(self)
        finally:
            try:
                self._t2i_loading = False
            except Exception:
                pass
    _T2I._load_settings = _load_settings_patched

    _orig_save = _T2I._save_settings
    def _save_settings_patched(self, job: dict):
        try:
            if getattr(self, "_t2i_loading", False):
                return
        except Exception:
            pass
        return _orig_save(self, job)
    _T2I._save_settings = _save_settings_patched

    def _set_text(w, val):
        try:
            w.setText("" if val is None else str(val))
        except Exception:
            pass
    def _set_combo_text(w, val):
        try:
            if val is None:
                return
            t = str(val)
            idx = w.findText(t) if hasattr(w, "findText") else -1
            if idx is not None and idx >= 0:
                w.setCurrentIndex(idx)
            elif hasattr(w, "setCurrentText"):
                w.setCurrentText(t)
        except Exception:
            pass
    def _set_checked(w, val):
        try:
            w.setChecked(bool(val))
        except Exception:
            pass
    def _set_value(w, val):
        try:
            w.setValue(int(val))
        except Exception:
            pass

    _orig_apply = _T2I._apply_settings_from_dict
    def _apply_settings_from_dict_patched(self, s: dict):
        # ## t2i_force_model_size_seed
        try:
            # --- Model restore ---
            mp = s.get('model_path') or s.get('model') or s.get('model_name')
            if mp and hasattr(self,'model_combo'):
                idxm = -1
                try:
                    for i in range(self.model_combo.count()):
                        if (self.model_combo.itemData(i) or '') == mp:
                            idxm = i; break
                except Exception:
                    pass
                if idxm < 0:
                    try:
                        import os
                        base = os.path.basename(str(mp)).lower()
                        for i in range(self.model_combo.count()):
                            txt = (self.model_combo.itemText(i) or '').lower()
                            if base and base in txt:
                                idxm = i; break
                    except Exception:
                        pass
                if idxm >= 0:
                    try: self.model_combo.blockSignals(True)
                    except Exception: pass
                    self.model_combo.setCurrentIndex(idxm)
                    try: self.model_combo.blockSignals(False)
                    except Exception: pass

            # --- qwen GGUF model restore ---
            wmp = s.get('qwen_model_path') or s.get('qwen2512_model_path')
            if wmp and hasattr(self, 'qwen_model_combo'):
                idxw = -1
                try:
                    for i in range(self.qwen_model_combo.count()):
                        if (self.qwen_model_combo.itemData(i) or '') == wmp:
                            idxw = i; break
                except Exception:
                    idxw = -1
                if idxw < 0:
                    try:
                        from pathlib import Path as _P
                        self.qwen_model_combo.addItem(_P(str(wmp)).name, str(wmp))
                        idxw = self.qwen_model_combo.findData(str(wmp))
                    except Exception:
                        idxw = -1
                if idxw >= 0:
                    try: self.qwen_model_combo.blockSignals(True)
                    except Exception: pass
                    self.qwen_model_combo.setCurrentIndex(idxw)
                    try: self.qwen_model_combo.blockSignals(False)
                    except Exception: pass

            # --- sd-cli.exe restore (GGUF engines) ---
            try:
                sdc = s.get('sd_cli_path') or s.get('sdcli_path') or s.get('sd_cli')
                if sdc and hasattr(self, 'sd_cli_path'):
                    try:
                        self.sd_cli_path.blockSignals(True)
                    except Exception:
                        pass
                    try:
                        self.sd_cli_path.setText(str(sdc))
                    except Exception:
                        pass
                    try:
                        self.sd_cli_path.blockSignals(False)
                    except Exception:
                        pass
            except Exception:
                pass

            # --- Seed policy restore ---
            if hasattr(self, 'seed_policy') and 'seed_policy' in s:
                sp = str(s.get('seed_policy') or '').lower().strip()
                idxs = {'fixed':0,'random':1,'increment':2}.get(sp, None)
                if idxs is None:
                    if 'rand' in sp: idxs = 1
                    elif 'inc' in sp: idxs = 2
                    else: idxs = 0
                try: self.seed_policy.blockSignals(True)
                except Exception: pass
                self.seed_policy.setCurrentIndex(int(idxs))
                try: self.seed_policy.blockSignals(False)
                except Exception: pass

            # --- Size restore ---
            w = s.get('width'); h = s.get('height')
            idxp = s.get('size_preset_index')
            if hasattr(self,'size_combo'):
                try:
                    if idxp is not None:
                        idxp = int(idxp)
                        if 0 <= idxp < self.size_combo.count():
                            self.size_combo.blockSignals(True)
                            self.size_combo.setCurrentIndex(idxp)
                            self.size_combo.blockSignals(False)
                    if w is not None and h is not None and hasattr(self,'_size_presets'):
                        idx2 = -1
                        for i,(label,wv,hv) in enumerate(self._size_presets):
                            if int(w)==int(wv) and int(h)==int(hv):
                                idx2 = i; break
                        if idx2 >= 0:
                            self.size_combo.blockSignals(True)
                            self.size_combo.setCurrentIndex(idx2)
                            self.size_combo.blockSignals(False)
                except Exception:
                    pass
            try:
                if w is not None and hasattr(self,'size_manual_w'): self.size_manual_w.setValue(int(w))
                if h is not None and hasattr(self,'size_manual_h'): self.size_manual_h.setValue(int(h))
            except Exception:
                pass
        except Exception:
            pass
    
        _orig_apply(self, s)
        try:
            for name in ("attn_slicing", "attention_slicing", "attentionSlicing"):
                w = getattr(self, name, None)
                if w is not None and "attn_slicing" in s:
                    _set_checked(w, s.get("attn_slicing"))
                    break
            for name in ("hires_helper", "hires_check", "hiresHelper"):
                w = getattr(self, name, None)
                if w is not None and "hires_helper" in s:
                    _set_checked(w, s.get("hires_helper"))
                    break
            for name in ("format_combo", "formatBox", "file_format", "formatCombo"):
                w = getattr(self, name, None)
                if w is not None and "format" in s:
                    _set_combo_text(w, s.get("format"))
                    break
            for name in ("filename_template", "filenameTemplate", "filename_edit"):
                w = getattr(self, name, None)
                if w is not None and "filename_template" in s:
                    _set_text(w, s.get("filename_template"))
                    break
            for name in ("vae_device", "vaeCombo", "vaeDevice"):
                w = getattr(self, name, None)
                if w is not None and "vae_device" in s:
                    _set_combo_text(w, s.get("vae_device"))
                    break
            for name in ("gpu_index", "gpuIndex"):
                w = getattr(self, name, None)
                if w is not None and "gpu_index" in s:
                    _set_value(w, s.get("gpu_index"))
                    break
            for name in ("threads", "num_threads", "threadsSpin"):
                w = getattr(self, name, None)
                if w is not None and "threads" in s:
                    _set_value(w, s.get("threads"))
                    break
            for name in ("fit_check", "fitCheck"):
                w = getattr(self, name, None)
                if w is not None and "fit_check" in s:
                    _set_checked(w, s.get("fit_check"))
                    break
        except Exception:
            pass
    _T2I._apply_settings_from_dict = _apply_settings_from_dict_patched

    _orig_collect = _T2I._collect_job
    def _collect_job_patched(self):
        job = _orig_collect(self)
        try:
            def _get_checked(name):
                w = getattr(self, name, None)
                try:
                    return bool(w.isChecked()) if w is not None else None
                except Exception:
                    return None
            def _get_combo_text(name):
                w = getattr(self, name, None)
                try:
                    if w is None:
                        return None
                    if hasattr(w, "currentText"):
                        return str(w.currentText())
                    if hasattr(w, "itemText"):
                        i = w.currentIndex()
                        return str(w.itemText(i))
                    return None
                except Exception:
                    return None
            def _get_text(name):
                w = getattr(self, name, None)
                try:
                    return str(w.text()) if w is not None else None
                except Exception:
                    return None
            def _get_value(name):
                w = getattr(self, name, None)
                try:
                    return int(w.value()) if w is not None else None
                except Exception:
                    return None

            attn = _get_checked("attn_slicing") or _get_checked("attention_slicing") or _get_checked("attentionSlicing")
            if attn is not None: job["attn_slicing"] = attn

            hires = _get_checked("hires_helper") or _get_checked("hires_check") or _get_checked("hiresHelper")
            if hires is not None: job["hires_helper"] = hires

            fmt = _get_combo_text("format_combo") or _get_combo_text("formatBox") or _get_combo_text("file_format") or _get_combo_text("formatCombo")
            if fmt is not None: job["format"] = fmt

            fname = _get_text("filename_template") or _get_text("filenameTemplate") or _get_text("filename_edit")
            if fname is not None: job["filename_template"] = fname

            vae = _get_combo_text("vae_device") or _get_combo_text("vaeCombo") or _get_combo_text("vaeDevice")
            if vae is not None: job["vae_device"] = vae

            gpu = _get_value("gpu_index") or _get_value("gpuIndex")
            if gpu is not None: job["gpu_index"] = gpu

            th = _get_value("threads") or _get_value("num_threads") or _get_value("threadsSpin")
            if th is not None: job["threads"] = th

            fitc = _get_checked("fit_check") or _get_checked("fitCheck")
            if fitc is not None: job["fit_check"] = fitc
        except Exception:
            pass
        return job
    _T2I._collect_job = _collect_job_patched

except Exception as _e:
    try:
        print("[txt2img] minimal persistence patch failed:", _e)
    except Exception:
        pass
# === End minimal persistence patch ===
