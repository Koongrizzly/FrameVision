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

# === QImageIO maxalloc disabled by patch ===
import os as _qt_img_os
_qt_img_os.environ["QT_IMAGEIO_MAXALLOC"] = "0"  # Disable env-based cap (0 = no limit)
try:
    from PySide6.QtGui import QImageReader as _QIR
    _QIR.setAllocationLimit(0)  # Disable runtime cap as well
except Exception as _e:
    pass
# === end patch ===

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

from PySide6.QtCore import QSettings, QTimer, Qt, Signal
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
def _rounded_pixmap(pm, radius: int = 10):
    try:
        from PySide6.QtGui import QPixmap
        from PySide6.QtCore import QRectF, Qt
        if pm is None:
            return pm
        if isinstance(pm, QPixmap) and not pm.isNull():
            w, h = pm.width(), pm.height()
            if w <= 0 or h <= 0:
                return pm
            r = max(0, int(radius))
            out = QPixmap(w, h)
            out.fill(Qt.transparent)
            p = QPainter(out)
            p.setRenderHint(QPainter.Antialiasing, True)
            p.setRenderHint(QPainter.SmoothPixmapTransform, True)
            path = QPainterPath()
            path.addRoundedRect(QRectF(0, 0, w, h), r, r)
            p.setClipPath(path)
            p.drawPixmap(0, 0, pm)
            p.end()
            return out
        return pm
    except Exception:
        return pm
# --- end helper ---
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
                    if hasattr(self,'cfg_scale') and 'cfg_scale' in s: self.cfg_scale.setValue(float(s.get('cfg_scale') or 7.5))
            except Exception:
                pass
            try:
                if hasattr(self,'vram_profile') and 'vram_profile' in s:
                    self.vram_profile.setCurrentText(s.get('vram_profile'))
                if hasattr(self,'sampler') and 'sampler' in s:
                    self.sampler.setCurrentText(s.get('sampler'))
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

            # Recent results options
            try:
                sz = int(s.get("recents_thumb_size", 100))
                if hasattr(self, "sld_recent_size"):
                    try:
                        self.sld_recent_size.setValue(sz)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                sort = s.get("recents_sort")
                cb = getattr(self, "combo_recent_sort", None)
                if cb is not None and sort is not None:
                    idx = -1
                    try:
                        for i in range(cb.count()):
                            v = cb.itemData(i)
                            if v == sort:
                                idx = i
                                break
                    except Exception:
                        idx = -1
                    if idx < 0:
                        try:
                            idx = cb.findText(str(sort))
                        except Exception:
                            idx = -1
                    if idx >= 0:
                        try:
                            cb.setCurrentIndex(idx)
                        except Exception:
                            pass
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
        try:
            self._connect_autosave()
        except Exception:
            pass

        # Check for Z-Image env; hide engine dropdown if missing
        try:
            self._update_engine_visibility_for_zimage_env()
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

        # After startup, force 'Recent results' closed once, a bit after other
        # state restorers have done their work (in case something else reopens it).
        try:
            from PySide6.QtCore import QTimer as _T2I_QTimer
        except Exception:
            _T2I_QTimer = None
        try:
            if _T2I_QTimer is not None:
                def _t2i_close_recents_later():
                    try:
                        box = getattr(self, "recents_box", None)
                        if box is not None:
                            try:
                                box._btn.setChecked(False)
                            except Exception:
                                pass
                            try:
                                box._body.setVisible(False)
                            except Exception:
                                pass
                    except Exception:
                        pass
                _T2I_QTimer.singleShot(3000, _t2i_close_recents_later)
        except Exception:
            pass


    def _update_engine_visibility_for_zimage_env(self):
        """Hide engine dropdown and force SD engine when Z-Image env is missing."""
        try:
            from pathlib import Path as _P
        except Exception:
            return
        # Determine app root (prefer ROOT if available)
        try:
            base = ROOT  # type: ignore[name-defined]
        except Exception:
            try:
                base = _P(__file__).resolve().parent.parent
            except Exception:
                base = _P.cwd()
        try:
            env_path = base / ".zimage_env" / "scripts" / "python.exe"
        except Exception:
            env_path = None

        has_zimage = False
        try:
            if env_path is not None:
                has_zimage = env_path.exists()
        except Exception:
            has_zimage = False

        cb = getattr(self, "engine_combo", None)
        lab = getattr(self, "engine_label", None)

        if not has_zimage:
            # Hide engines dropdown and force SD models (SD15/SDXL)
            try:
                if cb is not None:
                    idx = -1
                    try:
                        for i in range(cb.count()):
                            data = cb.itemData(i)
                            text = (cb.itemText(i) or "").lower()
                            if (isinstance(data, str) and data.lower() == "diffusers") or "sd models" in text:
                                idx = i
                                break
                    except Exception:
                        idx = -1
                    if idx >= 0:
                        cb.setCurrentIndex(idx)
                    cb.setVisible(False)
            except Exception:
                pass
            try:
                if lab is not None:
                    lab.setVisible(False)
            except Exception:
                pass
        else:
            # Environment present: keep engines dropdown visible
            try:
                if cb is not None:
                    cb.setVisible(True)
            except Exception:
                pass
            try:
                if lab is not None:
                    lab.setVisible(True)
            except Exception:
                pass

    def _apply_preset(self, name: str):
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


    # ===== Recent results: gather from thumbnails, output dir, and finished queue jobs =====
    def _recents_dir(self):
        """Return Path to the txt2img recent-thumbnail folder."""
        from pathlib import Path as _Path
        try:
            try:
                base = _Path(__file__).resolve().parent.parent
            except Exception:
                base = _Path.cwd()
            d = base / "output" / "last results" / "txt2img"
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return d
        except Exception:
            # Fallback to relative path
            return _Path("output") / "last results" / "txt2img"

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

    def _jobs_done_dirs(self):
        """Return a list of job result folders to scan for finished txt2img queue jobs."""
        from pathlib import Path as _Path
        try:
            # ROOT is provided by the main app in most places
            base = ROOT  # type: ignore[name-defined]
        except Exception:
            base = _Path(__file__).resolve().parent.parent
        roots = []
        for name in ("finished", "done"):
            try:
                d = base / "jobs" / name
                if d.exists() and d.is_dir():
                    roots.append(d)
            except Exception:
                continue
        return roots

    def _list_recent_txt2img_jobs(self):
        """List finished txt2img job JSON files (newest first)."""
        from pathlib import Path as _Path
        jobs = []
        try:
            for d in self._jobs_done_dirs():
                try:
                    for p in d.iterdir():
                        try:
                            if (
                                p.is_file()
                                and p.suffix.lower() == ".json"
                                and "txt2img" in p.name.lower()
                            ):
                                jobs.append(p)
                        except Exception:
                            continue
                except Exception:
                    continue
            jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return jobs[:40]
        except Exception:
            return []

    def _resolve_output_from_job(self, job_json):
        """Return (media_path, job_data) for a finished job JSON.

        This mirrors the generic logic from the Upscale tab and should also work
        for txt2img jobs.
        """
        from pathlib import Path as _Path
        import json as _json

        try:
            j = _json.loads(_Path(job_json).read_text(encoding="utf-8"))
        except Exception:
            j = {}

        def _as_path(val):
            if not val:
                return None
            try:
                p = _Path(str(val)).expanduser()
                if not p.is_absolute():
                    out_dir = j.get("out_dir") or (j.get("args") or {}).get("out_dir")
                    if out_dir:
                        p = _Path(out_dir).expanduser() / p
                return p
            except Exception:
                return None

        # Priority fields
        for k in ("produced", "outfile", "output", "result", "file", "path"):
            v = j.get(k) or (j.get("args") or {}).get(k)
            p = _as_path(v)
            if p and p.exists() and p.is_file():
                return p, j

        # List fields
        for k in ("outputs", "produced_files", "results", "files", "artifacts", "saved"):
            seq = j.get(k) or (j.get("args") or {}).get(k)
            if isinstance(seq, (list, tuple)):
                for v in seq:
                    p = _as_path(v)
                    if p and p.exists() and p.is_file():
                        return p, j

        # Fallback: newest media from out_dir
        media_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        out_dir = _as_path(j.get("out_dir") or (j.get("args") or {}).get("out_dir"))
        try:
            if out_dir and out_dir.exists():
                cand = [
                    p for p in out_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in media_exts
                ]
                cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if cand:
                    return cand[0], j
        except Exception:
            pass
        return None, j

    def _thumb_path_for_media(self, media_path, max_side: int = 120):
        """Return a Path under _recents_dir used to store a thumbnail for *media_path*."""
        from pathlib import Path as _Path
        import hashlib

        d = self._recents_dir()
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            stem = _Path(media_path).stem
        except Exception:
            stem = "img"

        try:
            key = str(media_path)
            h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        except Exception:
            h = "thumb"

        return d / f"{stem}_{h}_{max_side}.jpg"

    def _ensure_recent_thumb_for_media(self, media_path, max_side: int = 120):
        """Create (or reuse) a thumbnail in _recents_dir for the given media file."""
        from pathlib import Path as _Path
        from PySide6.QtGui import QImageReader
        from PySide6.QtCore import QSize, Qt

        media_path = _Path(media_path)
        try:
            if not (media_path.exists() and media_path.is_file()):
                return None
        except Exception:
            return None

        thumb = self._thumb_path_for_media(media_path, max_side=max_side)
        try:
            if thumb.exists() and thumb.stat().st_mtime >= media_path.stat().st_mtime:
                return thumb
        except Exception:
            # If we cannot stat, fall through and try to rebuild
            pass

        try:
            reader = QImageReader(str(media_path))
            reader.setAutoTransform(True)
            sz = reader.size()
            if sz.isValid():
                w, h = sz.width(), sz.height()
                if w > 0 and h > 0:
                    scale = max(w, h) / float(max_side or 1)
                    if scale > 1.0:
                        w = int(w / scale)
                        h = int(h / scale)
                        reader.setScaledSize(QSize(max(16, w), max(16, h)))
            img = reader.read()
            if img.isNull():
                return None
        except Exception:
            return None

        try:
            thumb.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        try:
            img.save(str(thumb), "JPG", 88)
            return thumb
        except Exception:
            return None


    def _resolve_media_for_thumb(self, thumb_path):
        """Best-effort: given a thumbnail path, return the original media path.

        Uses an in-memory thumb->media map populated when thumbnails are created,
        and falls back to searching likely output folders by filename stem.
        """
        try:
            from pathlib import Path as _P
        except Exception:
            _P = None

        if _P is None:
            return thumb_path

        try:
            t = _P(str(thumb_path))
        except Exception:
            return thumb_path

        # 1) In-memory mapping (new thumbnails in this session)
        try:
            mapping = getattr(self, "_recents_thumb_map", {}) or {}
            orig = mapping.get(str(t))
            if orig:
                p = _P(str(orig))
                if p.exists() and p.is_file():
                    return p
        except Exception:
            pass

        # 2) Parse the thumbnail filename: stem_hash_size.jpg -> stem
        try:
            name_stem = t.stem
            parts = name_stem.rsplit("_", 2)
            if len(parts) == 3 and parts[1] and parts[2]:
                base_stem = parts[0]
            else:
                base_stem = name_stem
        except Exception:
            base_stem = None

        media_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")

        # Build a list of candidate folders to search
        dirs = []
        try:
            out_dir = self._current_output_dir()
            if out_dir and out_dir.exists():
                dirs.append(out_dir)
        except Exception:
            pass

        # Include result folders from recent txt2img queue jobs
        try:
            jobs = self._list_recent_txt2img_jobs()
            for job_json in jobs:
                try:
                    media, _j = self._resolve_output_from_job(job_json)
                except Exception:
                    media = None
                if media is not None:
                    d = getattr(media, "parent", None)
                    if d is not None and d not in dirs:
                        dirs.append(d)
        except Exception:
            pass

        # 3) Try to find a matching media file by stem and known image extensions
        for d in dirs:
            try:
                if base_stem:
                    for ext in media_exts:
                        cand = d / f"{base_stem}{ext}"
                        if cand.exists() and cand.is_file():
                            return cand
                # Fallback: any file in the folder whose stem matches
                for f in d.iterdir():
                    try:
                        if f.is_file() and f.suffix.lower() in media_exts:
                            if base_stem and f.stem == base_stem:
                                return f
                    except Exception:
                        continue
            except Exception:
                continue

        # Last resort: fall back to the thumbnail itself
        return thumb_path

    def _list_recent_files(self):
        """List recent result thumbnail files (most recent first).

        This merges three sources:
        - Existing thumbnails under output/last results/txt2img
        - New images from the current txt2img output folder (non-queued runs)
        - Finished txt2img queue jobs (from jobs/finished or jobs/done)
        """
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        candidates = []

        # 1) Existing thumbs under the recents dir
        try:
            thumbs_dir = self._recents_dir()
            if thumbs_dir and thumbs_dir.exists():
                for p in thumbs_dir.iterdir():
                    try:
                        if p.is_file() and p.suffix.lower() in exts:
                            candidates.append(p)
                    except Exception:
                        continue
        except Exception:
            pass

        # Ensure we have a mapping dict available for thumb -> media
        try:
            if not hasattr(self, "_recents_thumb_map") or getattr(self, "_recents_thumb_map") is None:
                self._recents_thumb_map = {}
        except Exception:
            try:
                self._recents_thumb_map = {}
            except Exception:
                pass

        # 2) Non-queued direct output folder (current output_path)
        try:
            out_dir = self._current_output_dir()
            if out_dir and out_dir.exists():
                imgs = [
                    p for p in out_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in exts
                ]
                imgs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                for media in imgs[:40]:
                    tp = self._ensure_recent_thumb_for_media(media, max_side=120)
                    if tp:
                        candidates.append(tp)
                        try:
                            m = getattr(self, "_recents_thumb_map", {}) or {}
                            m[str(tp)] = str(media)
                            self._recents_thumb_map = m
                        except Exception:
                            pass
        except Exception:
            pass

        # 3) Finished queue jobs (txt2img jobs only)
        try:
            jobs = self._list_recent_txt2img_jobs()
            for job_json in jobs:
                media, _j = self._resolve_output_from_job(job_json)
                if not media:
                    continue
                tp = self._ensure_recent_thumb_for_media(media, max_side=120)
                if tp:
                    candidates.append(tp)
                    try:
                        m = getattr(self, "_recents_thumb_map", {}) or {}
                        m[str(tp)] = str(media)
                        self._recents_thumb_map = m
                    except Exception:
                        pass
        except Exception:
            pass


        # Deduplicate & sort by mtime (newest first)
        try:
            uniq = {}
            for p in candidates:
                try:
                    mt = p.stat().st_mtime
                except Exception:
                    mt = 0
                key = str(p)
                cur = uniq.get(key)
                if (cur is None) or (mt > cur[1]):
                    uniq[key] = (p, mt)
            files = [v[0] for v in uniq.values()]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return files[:48]
        except Exception:
            try:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return candidates[:48]
            except Exception:
                return []
    def _install_recents_poller(self):
        """Poll the txt2img recents folder every few seconds; rebuild UI on change."""
        try:
            if getattr(self, "_recents_poller", None):
                return
            from PySide6.QtCore import QTimer
            def _sig():
                try:
                    files = self._list_recent_files()
                    return tuple((p.name, int(p.stat().st_mtime)) for p in files)
                except Exception:
                    return tuple()
            def _tick():
                try:
                    cur = _sig()
                    if cur != getattr(self, "_recents_sig", None):
                        self._recents_sig = cur
                        self._rebuild_recents()
                except Exception:
                    pass
            try:
                self._recents_sig = None
                _tick()
            except Exception:
                pass
            t = QTimer(self)
            t.setInterval(5000)
            t.timeout.connect(_tick)
            t.start()
            self._recents_poller = t
        except Exception:
            pass

    def _rebuild_recents(self):
        """Rebuild the Recent results grid from thumbnail files."""
        try:
            layout = getattr(self, "_recents_row", None)
            inner = getattr(self, "_recents_inner", None)
            scroll = getattr(self, "recents_scroll", None)
            if layout is None or inner is None or scroll is None:
                return

            # Clear existing widgets
            try:
                while layout.count():
                    item = layout.takeAt(0)
                    w = item.widget()
                    if w is not None:
                        w.setParent(None)
            except Exception:
                pass

            # Thumb size from slider (clamped)
            try:
                size_slider = getattr(self, "sld_recent_size", None)
                size = int(size_slider.value()) if size_slider is not None else 100
            except Exception:
                size = 100
            if size < 40:
                size = 40
            if size > 200:
                size = 200

            files = self._list_recent_files()
            if not files:
                from PySide6.QtWidgets import QLabel as _QLabel
                lab = _QLabel("No results yet.", self)
                try:
                    lab.setStyleSheet("color:#9fb3c8;")
                except Exception:
                    pass
                layout.addWidget(lab, 0, 0)
                try:
                    inner.setMinimumHeight(lab.sizeHint().height() + 8)
                except Exception:
                    pass
                return

            from PySide6.QtGui import QIcon, QPixmap
            from PySide6.QtCore import QSize, Qt

            # Helper functions for sorting by underlying media
            def _media_for_sort(thumb_path):
                try:
                    return self._resolve_media_for_thumb(thumb_path)
                except Exception:
                    return thumb_path

            def _mtime_for(thumb_path):
                from pathlib import Path as _P
                try:
                    mp = _P(str(_media_for_sort(thumb_path)))
                    if mp.exists():
                        return mp.stat().st_mtime
                except Exception:
                    pass
                try:
                    return _P(str(thumb_path)).stat().st_mtime
                except Exception:
                    return 0

            def _name_for(thumb_path):
                from pathlib import Path as _P
                try:
                    mp = _P(str(_media_for_sort(thumb_path)))
                    return mp.name.lower()
                except Exception:
                    pass
                try:
                    return _P(str(thumb_path)).name.lower()
                except Exception:
                    return str(thumb_path)

            def _size_for(thumb_path):
                from pathlib import Path as _P
                try:
                    mp = _P(str(_media_for_sort(thumb_path)))
                    if mp.exists():
                        return mp.stat().st_size
                except Exception:
                    pass
                try:
                    return _P(str(thumb_path)).stat().st_size
                except Exception:
                    return 0

            # Determine sort mode from combo box
            try:
                mode = None
                cb = getattr(self, "combo_recent_sort", None)
                if cb is not None:
                    mode = cb.currentData()
                    if not mode:
                        mode = cb.currentText()
                if not mode:
                    mode = "newest"
            except Exception:
                mode = "newest"

            # Apply sorting
            try:
                if mode in ("newest", "oldest"):
                    files.sort(key=_mtime_for, reverse=(mode == "newest"))
                elif mode in ("az", "za"):
                    files.sort(key=_name_for, reverse=(mode == "za"))
                elif mode in ("size_small", "size_large"):
                    files.sort(key=_size_for, reverse=(mode == "size_large"))
            except Exception:
                # Fallback: newest first by thumbnail mtime
                try:
                    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                except Exception:
                    pass

            setattr(self, "_recents_idx", 0)
            for p in files:
                btn = QToolButton(self)
                btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
                try:
                    btn.setText(p.name)
                except Exception:
                    pass
                btn.setCursor(Qt.PointingHandCursor)
                btn.setAutoRaise(True)
                try:
                    btn.setStyleSheet(
                        "QToolButton { border-radius: 10px; padding: 4px 2px; }"
                        "QToolButton:hover { background: rgba(255,255,255,0.06); }"
                    )
                except Exception:
                    pass

                try:
                    pm = QPixmap(str(p))
                except Exception:
                    pm = QPixmap()
                if pm and not pm.isNull():
                    try:
                        pm2 = pm.scaled(int(size), int(size), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    except Exception:
                        pm2 = pm
                    try:
                        pm2 = _rounded_pixmap(pm2, 10)
                    except Exception:
                        pass
                    try:
                        btn.setIcon(QIcon(pm2))
                    except Exception:
                        pass
                try:
                    btn.setIconSize(QSize(int(size), int(size)))
                    btn.setFixedSize(int(size * 1.25), int(size * 1.25) + 28)
                except Exception:
                    pass

                # Tooltip with underlying media path (best effort)
                try:
                    media_tp = _media_for_sort(p)
                    from pathlib import Path as _P
                    btn.setToolTip(str(_P(str(media_tp))))
                except Exception:
                    pass

                # Left-click: open result
                def _mk_open(thumb_path):
                    def _open():
                        try:
                            media = self._resolve_media_for_thumb(thumb_path)
                        except Exception:
                            media = thumb_path
                        try:
                            # Prefer the app's internal media player; fall back to OS viewer only if needed
                            if not _play_in_player(self, media):
                                _open_file(self, media)
                        except Exception:
                            _open_file(self, media)
                    return _open
                try:
                    btn.clicked.connect(_mk_open(p))
                except Exception:
                    pass

                # Right-click: context menu with Info / Rename / Open folder / Delete
                def _mk_ctx(thumb_path, button):
                    def _on_menu(pos):
                        from pathlib import Path as _P
                        try:
                            from PySide6.QtWidgets import QMenu, QInputDialog
                        except Exception:
                            QMenu = None  # type: ignore
                            QInputDialog = None  # type: ignore
                        try:
                            from PySide6.QtGui import QDesktopServices
                        except Exception:
                            QDesktopServices = None  # type: ignore
                        try:
                            from PySide6.QtCore import QUrl
                        except Exception:
                            QUrl = None  # type: ignore

                        if QMenu is None:
                            return

                        # Resolve underlying media file once
                        try:
                            media = self._resolve_media_for_thumb(thumb_path)
                        except Exception:
                            media = thumb_path
                        try:
                            mp = _P(str(media))
                        except Exception:
                            mp = None

                        try:
                            menu = QMenu(button)
                        except Exception:
                            return
                        try:
                            act_info = menu.addAction("Info")
                        except Exception:
                            act_info = None
                        try:
                            act_rename = menu.addAction("Rename")
                        except Exception:
                            act_rename = None
                        try:
                            act_open = menu.addAction("Open folder")
                        except Exception:
                            act_open = None
                        try:
                            menu.addSeparator()
                        except Exception:
                            pass
                        try:
                            act_del = menu.addAction("Delete")
                        except Exception:
                            act_del = None

                        try:
                            global_pos = button.mapToGlobal(pos)
                        except Exception:
                            global_pos = None
                        try:
                            chosen = menu.exec(global_pos) if global_pos is not None else menu.exec()
                        except Exception:
                            chosen = None
                        if not chosen:
                            return

                        # Info
                        if act_info is not None and chosen is act_info:
                            try:
                                if mp is None or (not mp.exists()):
                                    QMessageBox.information(
                                        self,
                                        "Info",
                                        "Original image file could not be found on disk."
                                    )
                                    return
                            except Exception:
                                pass
                            try:
                                from PySide6.QtGui import QImageReader
                                reader = QImageReader(str(mp))
                                size = reader.size()
                                w = size.width() if size.isValid() else 0
                                h = size.height() if size.isValid() else 0
                                fmt_bytes = reader.format()
                                fmt = fmt_bytes.data().decode("ascii", "ignore").upper() if hasattr(fmt_bytes, "data") else (str(fmt_bytes).upper() if fmt_bytes else (mp.suffix.upper() if mp is not None else "?"))
                            except Exception:
                                w = h = 0
                                fmt = mp.suffix.upper() if mp is not None else "?"
                            try:
                                st = mp.stat() if mp is not None and mp.exists() else None
                                size_bytes = st.st_size if st is not None else 0
                            except Exception:
                                size_bytes = 0
                            try:
                                size_kib = size_bytes / 1024.0 if size_bytes else 0.0
                                mpx = (w * h) / 1_000_000.0 if w and h else 0.0
                                txt = []
                                txt.append(f"File: {mp.name if mp is not None else media}")
                                txt.append(f"Path: {mp}")
                                txt.append("")
                                txt.append(f"Resolution: {w} × {h} px" if w and h else "Resolution: unknown")
                                txt.append(f"Format: {fmt}" if fmt else "Format: unknown")
                                txt.append(f"File size: {size_kib:.1f} KiB ({size_bytes} bytes)")
                                if mpx:
                                    txt.append(f"Megapixels: {mpx:.2f} MP")
                                QMessageBox.information(self, "Image info", "\n".join(txt))
                            except Exception:
                                pass
                            return

                        # Rename
                        if act_rename is not None and chosen is act_rename:
                            try:
                                if mp is None or (not mp.exists()):
                                    QMessageBox.warning(
                                        self,
                                        "Rename failed",
                                        "Original image file could not be found on disk."
                                    )
                                    return
                            except Exception:
                                pass
                            try:
                                current_name = mp.name
                            except Exception:
                                current_name = str(media)
                            try:
                                ok = False
                                new_name, ok = QInputDialog.getText(
                                    self,
                                    "Rename image",
                                    "New filename:",
                                    text=current_name
                                )
                            except Exception:
                                new_name = ""
                                ok = False
                            if not ok:
                                return
                            try:
                                new_name = str(new_name).strip()
                            except Exception:
                                pass
                            if not new_name:
                                return
                            # Keep directory, avoid path components
                            try:
                                base_dir = mp.parent
                            except Exception:
                                base_dir = None
                            try:
                                # Drop any path fragments user may have entered
                                simple = new_name.replace("\\", "/").split("/")[-1]
                            except Exception:
                                simple = new_name
                            # Ensure extension
                            try:
                                if "." not in simple and mp is not None:
                                    simple = simple + mp.suffix
                            except Exception:
                                pass
                            try:
                                if base_dir is not None:
                                    dest = base_dir / simple
                                else:
                                    dest = _P(simple)
                            except Exception:
                                dest = None
                            if dest is None:
                                return
                            try:
                                if dest.exists():
                                    QMessageBox.warning(
                                        self,
                                        "Rename failed",
                                        "A file with that name already exists."
                                    )
                                    return
                            except Exception:
                                pass
                            try:
                                mp.rename(dest)
                            except Exception:
                                try:
                                    QMessageBox.warning(
                                        self,
                                        "Rename failed",
                                        "Could not rename the image file."
                                    )
                                except Exception:
                                    pass
                                return
                            # Update mapping + tooltip + label
                            try:
                                mapping = getattr(self, "_recents_thumb_map", {}) or {}
                                mapping[str(thumb_path)] = str(dest)
                                self._recents_thumb_map = mapping
                            except Exception:
                                pass
                            try:
                                button.setToolTip(str(dest))
                            except Exception:
                                pass
                            try:
                                button.setText(dest.name)
                            except Exception:
                                pass
                            try:
                                self._rebuild_recents()
                            except Exception:
                                pass
                            return

                        # Open folder
                        if act_open is not None and chosen is act_open:
                            try:
                                if mp is not None and mp.exists():
                                    folder = mp.parent
                                else:
                                    folder = _P(str(thumb_path)).parent
                            except Exception:
                                folder = None
                            if folder is None:
                                return
                            try:
                                if QDesktopServices is not None and QUrl is not None:
                                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
                            except Exception:
                                pass
                            return

                        # Delete (unload + remove media & thumb)
                        if act_del is not None and chosen is act_del:
                            # Confirm deletion
                            try:
                                fname = mp.name if mp is not None else str(media)
                            except Exception:
                                fname = str(media)
                            try:
                                res = QMessageBox.question(
                                    self,
                                    "Delete image?",
                                    f"Delete this generated image from disk?\n\n{fname}",
                                    QMessageBox.Yes | QMessageBox.No,
                                    QMessageBox.No,
                                )
                            except Exception:
                                res = QMessageBox.Yes
                            if res != QMessageBox.Yes:
                                return

                            # Try to unload from player if currently open
                            try:
                                m = getattr(self, "_main", None) or getattr(self, "main", None)
                            except Exception:
                                m = None
                            if m is None:
                                try:
                                    from PySide6.QtWidgets import QApplication as _QApp
                                    for w in _QApp.topLevelWidgets():
                                        if hasattr(w, "video"):
                                            m = w
                                            break
                                except Exception:
                                    m = None
                            try:
                                if m is not None and hasattr(m, "current_path"):
                                    cur = getattr(m, "current_path", None)
                                    if cur is not None and str(cur) == (str(mp) if mp is not None else str(media)):
                                        try:
                                            player = getattr(m, "video", None)
                                            if player is not None and hasattr(player, "stop"):
                                                player.stop()
                                        except Exception:
                                            pass
                                        try:
                                            m.current_path = None
                                        except Exception:
                                            pass
                            except Exception:
                                pass

                            # Delete media file
                            try:
                                if mp is not None and mp.exists():
                                    mp.unlink()
                            except Exception:
                                try:
                                    QMessageBox.warning(
                                        self,
                                        "Delete failed",
                                        f"Could not delete file:\n{fname}",
                                    )
                                except Exception:
                                    pass

                            # Delete thumbnail file
                            try:
                                tp = _P(str(thumb_path))
                                if tp.exists():
                                    tp.unlink()
                            except Exception:
                                pass

                            # Remove from in-memory mapping
                            try:
                                mapping = getattr(self, "_recents_thumb_map", {}) or {}
                                mapping.pop(str(thumb_path), None)
                                self._recents_thumb_map = mapping
                            except Exception:
                                pass

                            # Rebuild grid
                            try:
                                self._rebuild_recents()
                            except Exception:
                                pass
                            return
                    return _on_menu

                try:
                    from PySide6 import QtWidgets as _QtW  # local alias
                except Exception:
                    _QtW = None
                try:
                    btn.setContextMenuPolicy(Qt.CustomContextMenu)
                    if _QtW is not None:
                        btn.customContextMenuRequested.connect(_mk_ctx(p, btn))
                except Exception:
                    pass

                # grid placement with wrapping
                try:
                    try:
                        vpw = scroll.viewport().width()
                    except Exception:
                        vpw = inner.width()
                    if not vpw or vpw <= 1:
                        vpw = max(scroll.width(), self.width(), 600)
                except Exception:
                    vpw = 600
                try:
                    spacing = getattr(layout, "spacing", lambda: 8)()
                except Exception:
                    spacing = 8
                item_w = int(size * 1.25)
                if vpw <= item_w + spacing and len(files) > 1:
                    cols = min(len(files), 4)
                else:
                    cols = max(1, int((vpw + spacing) // (item_w + spacing)))
                idx = getattr(self, "_recents_idx", 0)
                row = idx // cols
                col = idx % cols
                setattr(self, "_recents_idx", idx + 1)
                try:
                    layout.addWidget(btn, row, col)
                except Exception:
                    pass

            # ensure the scroll area can expand vertically if needed
            try:
                try:
                    spacing = getattr(layout, "spacing", lambda: 8)()
                except Exception:
                    spacing = 8
                item_w = int(size * 1.25)
                item_h = int(size * 1.25) + 28
                try:
                    vpw = scroll.viewport().width()
                except Exception:
                    vpw = inner.width()
                if not vpw or vpw <= 1:
                    vpw = max(scroll.width(), self.width(), 600)
                cols = max(1, int((vpw + spacing) // (item_w + spacing)))
                total = layout.count()
                rows = max(1, (total + cols - 1) // cols)
                min_h = rows * item_h + max(0, rows - 1) * spacing + 12
                inner.setMinimumHeight(min_h)
            except Exception:
                pass
        except Exception as e:
            try:
                print("[txt2img] recents rebuild error:", e)
            except Exception:
                pass

    def eventFilter(self, obj, ev):
        """Forward event filtering to base class, but watch recents viewport width."""
        try:
            from PySide6.QtCore import QEvent
            if hasattr(self, "recents_scroll") and self.recents_scroll is not None:
                if obj is self.recents_scroll.viewport():
                    if ev.type() == QEvent.Resize:
                        try:
                            w = ev.size().width()
                        except Exception:
                            w = 0
                        if w and w != getattr(self, "_recents_last_w", 0):
                            self._recents_last_w = w
                            try:
                                self._rebuild_recents()
                            except Exception:
                                pass
        except Exception:
            pass
        try:
            return super().eventFilter(obj, ev)
        except Exception:
            return False

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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll, 1)
        container = QWidget(); scroll.setWidget(container)
        root = QVBoxLayout(container)

        # Engine / backend selector
        engine_row = QHBoxLayout()
        self.engine_combo = QComboBox()
        try:
            self.engine_combo.addItem("SD models (SD15/SDXL)", "diffusers")
            self.engine_combo.addItem("Z-Image Turbo", "zimage")
        except Exception:
            # Fallback without userData support
            self.engine_combo.addItem("SD models (SD15/SDXL)")
            self.engine_combo.addItem("Z-Image Turbo")
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
        form.addRow("Negative", self.negative)

        # Seed + seed policy (Batch moved next to Generate button)
        row = QHBoxLayout()
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(0)
        self.seed_policy = QComboBox(); self.seed_policy.addItems(["Fixed (use seed)", "Random", "Increment"]); self.seed_policy.setCurrentIndex(1)
        self.batch = QSpinBox(); self.batch.setRange(1, 64); self.batch.setValue(1)
        row.addWidget(QLabel("Seed:")); row.addWidget(self.seed)
        row.addWidget(QLabel("")); row.addWidget(self.seed_policy)
        form.addRow(row)

        # --- Quality / Output Size (next to seed) ---
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Output size:"))
        self.size_combo = QComboBox()
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
            ("1024x1024 (1:1, recommended)", 1024, 1024),
            ("1280x1280 (1:1, recommended)", 1280, 1280),

            # 16:9/9:16
            ("576x1024 (9:16)", 576, 1024),
            ("1024x576 (16:9)", 1024, 576),
            ("640x1136 (9:16)", 640, 1136),
            ("1136x640 (16:9)", 1136, 640),
            ("720x1280 (9:16, recommended)", 720, 1280),
            ("1280x720 (16:9, recommended)", 1280, 720),
            ("704x1280 (9:16, WAN specific size)", 704, 1280),
            ("1280x704 (16:9, WAN specific size)", 1280, 704),
            ("864x1536 (9:16, recommended)", 864, 1536),
            ("1344x768 (16:9, recommended)", 1344, 768),
            ("768x1344 (9:16, recommended)", 768, 1344),
            ("1536x864 (16:9, max advised for SDXL)", 1536, 864),
            ("864x1536 (9:16, max advised for SDXL)", 864, 1536),
            #        ("972x1728 (9:16)", 972, 1728),
            #        ("1728x972 (16:9)", 1728, 972),

            # 21:9/9:21
            ("544x1280 (9:21)", 544, 1280),
            ("1280x544 (21:9)", 1280, 544),
            ("576x1344 (9:21, recommended)", 576, 1344),
            ("1344x576 (21:9, recommended)", 1344, 576),

            # 9:7/7:9
            ("896x1152 (7:9, recommended)", 896, 1152),
            ("1152x896 (9:7, recommended)", 1152, 896),

            # 4:3/3:4
            ("480x640 (3:4)", 480, 640),
            ("640x480 (4:3)", 640, 480),
            ("600x800 (3:4)", 600, 800),
            ("800x600 (4:3)", 800, 600),
            ("896x672 (4:3)", 896, 672),
            ("768x1024 (3:4, recommended)", 768, 1024),
            ("1024x768 (4:3, recommended)", 1024, 768),
            ("832x1104 (3:4)", 832, 1104),
            ("1104x832 (4:3)", 1104, 832),
            ("1152x864 (4:3, max advised for SDXL)", 1152, 864),

            # 3:2/2:3
            ("960x640 (3:2)", 960, 640),
            ("768x1152 (2:3, recommended)", 768, 1152),
            ("1152x768 (3:2, recommended)", 1152, 768),
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
        self.size_manual_h = QSpinBox(); self.size_manual_h.setRange(256, 2560); self.size_manual_h.setSingleStep(64)
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
        size_row.addWidget(QLabel("W:"), 0)
        size_row.addWidget(self.size_manual_w, 0)
        size_row.addWidget(QLabel("H:"), 0)
        size_row.addWidget(self.size_manual_h, 0)
        size_row.addWidget(self.size_lock, 0)
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
        self.steps_slider.valueChanged.connect(lambda v: self.steps_value.setText(str(int(v))))
        steps_row.addWidget(QLabel("Steps:"))
        steps_row.addWidget(self.steps_slider, 1)
        steps_row.addWidget(self.steps_value, 0)
        steps_row.addWidget(self.steps_default, 0)
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

        # VRAM profile override
        vram_row = QHBoxLayout()
        self.vram_profile = QComboBox(); self.vram_profile.addItems(["Auto", "6 GB", "8 GB", "12 GB", "24 GB"])
        self.restore_auto = QPushButton("Restore Auto")
        self.restore_auto.hide(); self.restore_auto.clicked.connect(lambda: self.vram_profile.setCurrentIndex(0))
        vram_row.addWidget(QLabel("VRAM profile:")); vram_row.addWidget(self.vram_profile); vram_row.addWidget(self.restore_auto)
        
        vram_row.addWidget(self.show_in_player)
        vram_row.addWidget(self.use_queue)
        form.addRow(vram_row)
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
        mdl_form = QFormLayout(mdl_body)
        self.model_combo = QComboBox()
        self.model_refresh = QPushButton("Refresh")
        self.model_browse = QPushButton("Browse…")
        rowm = QHBoxLayout()
        rowm.addWidget(self.model_combo, 1)
        rowm.addWidget(self.model_refresh, 0)
        rowm.addWidget(self.model_browse, 0)
        mdl_form.addRow("Model", rowm)

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
        mdl_form.addRow("LoRA (SDXL)", rowl)
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
        rowls.addWidget(QLabel("LoRA 1 Strength:"))
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
        self.lora2_refresh = QPushButton("Reload")
        rowl2 = QHBoxLayout()
        rowl2.addWidget(self.lora2_refresh, 0)
        rowl2.addWidget(self.lora2_combo, 1)
        mdl_form.addRow("LoRA 2 (SDXL)", rowl2)

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
        rowls2.addWidget(QLabel("LoRA 2 Strength:"))
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
        self.sampler = QComboBox(); self.sampler.addItems(["auto","DPM++ 2M (Karras)","Euler a","Euler","Heun","UniPC","DDIM"])
        self.attn_slicing = QCheckBox("Attention slicing")
        self.vae_device = QComboBox(); self.vae_device.addItems(["auto","cpu","gpu"])
        self.gpu_index = QSpinBox(); self.gpu_index.setRange(0,8)
        self.threads = QSpinBox(); self.threads.setRange(1,256); self.threads.setValue(8)
        self.format_combo = QComboBox(); self.format_combo.addItems(["png","jpg","webp"])
        self.filename_template = QLineEdit("IMG_{seed}.png")
        self.reset_fname = QPushButton("Reset"); self.reset_fname.clicked.connect(lambda: self.filename_template.setText(f"IMG_{{seed}}.{self.format_combo.currentText()}"))
        try:
            _on_format_changed()
        except Exception:
            pass
        self.hires_helper = QCheckBox("Hi-res helper")
        self.fit_check = QCheckBox("Fit-check")
        adv_form.addRow("Sampler", self.sampler)
        adv_form.addRow(self.attn_slicing)
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
        rowf = QHBoxLayout(); rowf.addWidget(self.filename_template, 1); rowf.addWidget(self.reset_fname)
        adv_form.addRow("Filename", rowf)
        adv_form.addRow(self.hires_helper)
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
            self.threads.setToolTip("Max CPU threads for image I/O and any CPU-side steps (e.g., VAE on CPU). Higher can speed up saves/loads, but too high may reduce UI responsiveness.")
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

        # Recent results (sticky, above Generate button; collapsed by default)
        try:
            from PySide6.QtWidgets import QGridLayout
            rec_body = QWidget(self)
            rec_wrap = QVBoxLayout(rec_body)
            rec_wrap.setContentsMargins(6, 2, 6, 6)
            rec_wrap.setSpacing(6)

            # Size + sort row
            size_row = QHBoxLayout()

            # Sort dropdown (recent results)
            try:
                lbl_sort = QLabel("Sort:", self)
            except Exception:
                lbl_sort = None
            try:
                self.combo_recent_sort = QComboBox(self)
                self.combo_recent_sort.addItem("Newest first", "newest")
                self.combo_recent_sort.addItem("Oldest first", "oldest")
                self.combo_recent_sort.addItem("Alphabetical (A-Z)", "az")
                self.combo_recent_sort.addItem("Alphabetical (Z-A)", "za")
                self.combo_recent_sort.addItem("Size (smallest first)", "size_small")
                self.combo_recent_sort.addItem("Size (largest first)", "size_large")
            except Exception:
                self.combo_recent_sort = None

            try:
                if lbl_sort is not None:
                    size_row.addWidget(lbl_sort)
            except Exception:
                pass
            try:
                if self.combo_recent_sort is not None:
                    size_row.addWidget(self.combo_recent_sort)
            except Exception:
                pass

            size_row.addSpacing(12)
            size_row.addWidget(QLabel("Thumb size:", self))
            self.sld_recent_size = QSlider(Qt.Horizontal, self)
            self.sld_recent_size.setMinimum(50)
            self.sld_recent_size.setMaximum(180)
            self.sld_recent_size.setSingleStep(8)
            self.sld_recent_size.setPageStep(30)
            self.sld_recent_size.setValue(100)
            size_row.addWidget(self.sld_recent_size, 1)
            self.lbl_recent_size = QLabel("100 px", self)
            size_row.addWidget(self.lbl_recent_size)
            rec_wrap.addLayout(size_row)

            # Scroll area with a wrapped grid of thumbnails
            self.recents_scroll = QScrollArea(self)
            self.recents_scroll.setWidgetResizable(True)
            self.recents_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self.recents_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            try:
                self.recents_scroll.setFrameShape(QScrollArea.NoFrame)
            except Exception:
                pass

            try:
                # Watch width changes so we can re-wrap items correctly
                self.recents_scroll.viewport().installEventFilter(self)
            except Exception:
                pass

            self._recents_inner = QWidget(self)
            self._recents_row = QGridLayout(self._recents_inner)
            self._recents_row.setContentsMargins(0, 0, 0, 0)
            self._recents_row.setSpacing(8)
            self.recents_scroll.setWidget(self._recents_inner)
            rec_wrap.addWidget(self.recents_scroll)

            # Collapsible wrapper, closed by default
            self.recents_box = _Disclosure("Recent results", rec_body, start_open=False, parent=self)
            outer.addWidget(self.recents_box)

            # Initial build + poller
            try:
                QTimer.singleShot(0, self._rebuild_recents)
            except Exception:
                pass
            try:
                self._install_recents_poller()
            except Exception:
                pass

            # Wire slider to resize thumbnails
            def _on_recent_size(val):
                try:
                    self.lbl_recent_size.setText(f"{val} px")
                except Exception:
                    pass
                try:
                    self._rebuild_recents()
                except Exception:
                    pass
            try:
                self.sld_recent_size.valueChanged.connect(_on_recent_size)
            except Exception:
                pass

            # Wire sort dropdown to rebuild thumbnails
            try:
                if getattr(self, "combo_recent_sort", None) is not None:
                    def _on_recent_sort(_index):
                        try:
                            self._rebuild_recents()
                        except Exception:
                            pass
                    try:
                        self.combo_recent_sort.currentIndexChanged.connect(_on_recent_sort)
                    except Exception:
                        pass
            except Exception:
                pass
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
        btns.addStretch(1)
        btns.addWidget(self.add_and_run)
        outer.addLayout(btns)

    def _on_engine_changed(self, *_):
        """
        Toggle banner text and show/hide the SD model/LoRA picker
        based on the selected engine.

        Also adjusts Steps / CFG ranges for Z-Image:
        - Z-Image: steps 1–50 (default 9), CFG 0.0–5.0 (default 0.0)
        - Diffusers (SD15/SDXL): steps 10–100 (default 25), CFG 1.0–15.0 (default 5.5)
        """
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
                key = str(data).lower()
            else:
                text = (cb.currentText() or "").lower()
                if "z-image" in text or "zimage" in text:
                    key = "zimage"
                else:
                    key = "diffusers"
        except Exception:
            pass
        is_zimage = (key == "zimage")

        # If switching to Z-Image, aggressively try to free CUDA VRAM
        if is_zimage:
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
                else:
                    self.banner.setText(base or "Text to Image with SDXL Loader")
        except Exception:
            pass

        # Show/hide SD model + LoRA dropdown group
        try:
            if hasattr(self, "_model_picker") and self._model_picker is not None:
                self._model_picker.setVisible(not is_zimage)
        except Exception:
            pass

        # Show/hide presets row based on engine
        try:
            lab = getattr(self, "preset_label", None)
            combo = getattr(self, "preset_combo", None)
            visible = not is_zimage
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
                    fname_edit.setText("z_img_{seed}_{idx:03d}.png")
                    try:
                        fname_edit.blockSignals(False)
                    except Exception:
                        pass
                else:
                    # Switching back to SD15/SDXL: restore original or default
                    try:
                        orig = getattr(self, "_filename_template_before_zimage", "") or ""
                    except Exception:
                        orig = ""
                    if not orig:
                        orig = "IMG_{seed}.png"
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
                self._last_engine_key = ("zimage" if is_zimage else "diffusers")
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
            if engine_key == "zimage":
                fname = "z_img_{seed}_{idx:03d}.png"
            else:
                fname = "IMG_{seed}.png"
        job = {
            "type": "txt2img",
            "engine": engine,
"type": "txt2img",
            "engine": (self.engine_combo.currentData() if hasattr(self, "engine_combo") else "diffusers") or "diffusers",
            "prompt": self.prompt.toPlainText().strip(),
            "negative": self.negative.toPlainText().strip(),
            "seed": seed,
            "seed_policy": ["fixed","random","increment"][self.seed_policy.currentIndex()],
            "batch": int(self.batch.value()),
            "cfg_scale": float(self.cfg_scale.value()) if hasattr(self, "cfg_scale") else 7.5,
            "output": self.output_path.text().strip(),
            "show_in_player": self.show_in_player.isChecked(),
            "use_queue": bool(self.use_queue.isChecked()),
            "vram_profile": self.vram_profile.currentText(),
            "sampler": self.sampler.currentText(),
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

            "hires_helper": self.hires_helper.isChecked(),
            "fit_check": self.fit_check.isChecked(),
            "steps": int(self.steps_slider.value()),
            "created_at": time.time(),
            "size_preset_index": int(self.size_combo.currentIndex()) if hasattr(self,"size_combo") else -1,
            "size_preset_label": str(self.size_combo.currentText()) if hasattr(self,"size_combo") else "",
            "width": int(w_ui),
            "height": int(h_ui),

            "preset": (self.preset_combo.currentText() if hasattr(self, "preset_combo") else ""),
            "preset_index": (int(self.preset_combo.currentIndex()) if hasattr(self, "preset_combo") else -1),
            "a1111_url": self._a1111_url_removed.text().strip() if hasattr(self, "_a1111_url_removed") else "http://127.0.0.1:7860",
                    }
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
        try: d["preset"] = self.preset_combo.currentText()
        except Exception: pass
        try: d["preset_index"] = int(self.preset_combo.currentIndex())
        except Exception: pass

        # Recent results options
        try:
            d["recents_thumb_size"] = int(self.sld_recent_size.value())
        except Exception:
            pass
        try:
            sort = None
            try:
                sort = self.combo_recent_sort.currentData()
            except Exception:
                sort = None
            if sort is None:
                try:
                    if getattr(self, "combo_recent_sort", None) is not None:
                        sort = self.combo_recent_sort.currentText()
                except Exception:
                    sort = None
            if sort:
                d["recents_sort"] = str(sort)
        except Exception:
            pass

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
            (self.hires_helper, 'toggled'),
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
        try:
            use_q = bool(self.use_queue.isChecked())
        except Exception:
            use_q = False
        if (int(job.get('batch',1))>1) or use_q:
            try:
                from helpers.queue_adapter import enqueue_txt2img
            except Exception:
                enqueue_txt2img = None
            if enqueue_txt2img and enqueue_txt2img(job | {'run_now': True}):
                try: self.status.setText('Enqueued and running…')
                except Exception: pass
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
                fname += ".png"
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
                fname += ".png"
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

    try:
        # Ensure out_dir exists before launching
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        try:
            print("[txt2img] Z-Image: failed to launch CLI:", e)
        except Exception:
            pass
        return None

    if proc.returncode != 0:
        try:
            print("[txt2img] Z-Image CLI stderr:", proc.stderr)
        except Exception:
            pass
        return None

    # CLI prints JSON (possibly with other logs; take last JSON-looking line)
    out_text = (proc.stdout or "").strip()
    if not out_text and proc.stderr:
        out_text = proc.stderr.strip()

    payload = None
    if out_text:
        for line in out_text.splitlines()[::-1]:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
                break
            except Exception:
                continue
    if not isinstance(payload, dict):
        return None

    files = payload.get("files") or []
    if not files:
        return None

    # Progress callback: mark as done
    if progress_cb:
        try:
            progress_cb(batch, batch)
        except Exception:
            pass

    return {
        "files": [str(f) for f in files],
        "backend": "zimage",
        "model": payload.get("model") or "Z-Image-Turbo",
    }


def generate_qwen_images(job: dict, progress_cb: Optional[Callable[[float], None]] = None, cancel_event: Optional[threading.Event] = None):
    """Generator that saves images and returns metadata. Uses CPU fallback to validate pipeline end-to-end."""
    out_dir = Path(job.get("output") or "./output/images"); out_dir.mkdir(parents=True, exist_ok=True)
    batch = int(job.get("batch", 1)); seed = int(job.get("seed", 0))
    seed_policy = job.get("seed_policy", "fixed"); fmt = job.get("format", "png").lower().strip()
    steps = int(job.get("steps", 30))
    req_w = int(job.get("width", 1024)); req_h = int(job.get("height", 1024))
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
        if engine == "zimage":
            diff = _gen_via_zimage(job, out_dir, progress_cb)
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

    This mirrors the internal-player behavior used by the Wan2.2 tab:
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