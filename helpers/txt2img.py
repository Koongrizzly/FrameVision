from __future__ import annotations
# >>> FRAMEVISION_QWEN_BEGIN
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
            (320,320),(480,480),(640,640),(960,960),(1024,1024),
            # 16:9 landscape
            (854,480),(1280,720),(1536,864),(1920,1080),(2560,1440),
            # 9:16 portrait
            (480,854),(720,1280),(864,1536),(1080,1920),(1440,2560),
            # 640p-ish 16:9 (Apple-ish) + portrait
            (1136,640),(640,1136),
            # Cinematic + their portrait
            (1280,544),(544,1280),
            # 4:3 and 3:4
            (1104,832),(832,1104)
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

from typing import Callable, Optional
try:
    import requests
except Exception:
    requests = None

from PySide6.QtCore import Qt, Signal, QSettings
from PySide6.QtWidgets import (
    QMessageBox,
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton, QSpinBox,
    QCheckBox, QFileDialog, QComboBox, QProgressBar, QGroupBox, QFormLayout, QScrollArea, QToolButton, QSlider,
    QDoubleSpinBox
)
# Import QShortcut correctly from QtGui; fall back to no shortcut if missing
try:
    from PySide6.QtGui import QKeySequence, QImage, QPainter, QShortcut
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
        self._btn.clicked.connect(self._on_clicked)
        self._body = content; self._body.setVisible(start_open)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self._btn); lay.addWidget(self._body)
    def _on_clicked(self, checked: bool):
        self._body.setVisible(checked)
        self._btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.toggled.emit(checked)
class Txt2ImgPane(QWidget):
    fileReady = Signal(str)

    def __init__(self, app_window):
        super().__init__(parent=app_window)
        self.app_window = app_window
        self.setObjectName("Txt2ImgPane")
        self._build_ui()

        # Wire actions
        self.add_to_queue.clicked.connect(lambda: self._enqueue(run_now=False))
        self.add_and_run.clicked.connect(lambda: self._enqueue(run_now=True))
        self.generate_now.clicked.connect(self._on_generate_clicked)


        # Defaults
        self.use_queue.setChecked(False)  # default removed; will restore via UI
        self.show_in_player.setChecked(False)  # default removed; will restore via UI
        self.nsfw_toggle.setChecked(False)  # default removed; will restore via UI
        self.preset_combo.addItems(["Photoreal / Daylight","Portrait / Soft","Cinematic / Moody","Product / Clean","Landscape / Golden Hour","Neon Cyberpunk","Anime Clean","Isometric 3D","Macro / Shallow DOF","Illustration / Watercolor"])
        self.preset_combo.setCurrentIndex(0)
        try:
            self.preset_combo.currentTextChanged.connect(self._apply_preset)
        except Exception:
            pass

        # Hotkey (Ctrl+Enter) only if QShortcut available
        try:
            if QShortcut is not None:
                QShortcut(QKeySequence("Ctrl+Enter"), self, activated=self._on_generate_clicked)
        except Exception:
            pass


    def _apply_preset(self, name: str):
        presets = {
            "Photoreal / Daylight": {"sampler":"DPM++ 2M (Karras)","steps":32,"cfg":5.5,"size":(1024,1024),
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
                self.size_combo.setCurrentText(f"{w}x{h} (1:1)" if w==h else f"{w}x{h}")
            if hasattr(self, "size_manual_w"): self.size_manual_w.setValue(w)
            if hasattr(self, "size_manual_h"): self.size_manual_h.setValue(h)
        except Exception:
            pass
        try:
            if self.negative.toPlainText().strip() == "":
                self.negative.setPlainText(cfg["neg"])
        except Exception:
            pass
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10,10,10,10)
        outer.setSpacing(8)
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll, 1)
        container = QWidget(); scroll.setWidget(container)
        root = QVBoxLayout(container)

        # Presets and chips
        top = QHBoxLayout()
        self.preset_combo = QComboBox()
        self.style_builder_btn = QPushButton("Style Builder")
        top.addWidget(QLabel("Preset:"))
        top.addWidget(self.preset_combo, 1)
        top.addWidget(self.style_builder_btn, 0)
        root.addLayout(top)

        # Prompts
        form = QFormLayout()
        self.prompt = QTextEdit(); self.prompt.setPlaceholderText("Describe the image you want…"); self.prompt.setFixedHeight(64)
        self.negative = QTextEdit(); self.negative.setPlaceholderText("What to avoid (optional)…"); self.negative.setFixedHeight(48)
        form.addRow("Prompt", self.prompt)
        form.addRow("Negative", self.negative)

        # Seed, batch, seed policy
        row = QHBoxLayout()
        self.seed = QSpinBox(); self.seed.setRange(0, 2_147_483_647); self.seed.setValue(0)
        self.seed_policy = QComboBox(); self.seed_policy.addItems(["Fixed (use seed)", "Random", "Increment"]); self.seed_policy.setCurrentIndex(1)
        self.batch = QSpinBox(); self.batch.setRange(1, 64); self.batch.setValue(1)
        row.addWidget(QLabel("Seed:")); row.addWidget(self.seed)
        row.addWidget(QLabel("Policy:")); row.addWidget(self.seed_policy)
        row.addWidget(QLabel("Batch:")); row.addWidget(self.batch)
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
            ("640x640 (1:1)", 640, 640),
            ("960x960 (1:1)", 960, 960),
            ("1024x1024 (1:1)", 1024, 1024),
            # 16:9
            ("854x480 (16:9)", 854, 480),
            ("1280x720 (16:9)", 1280, 720),
            ("1536x864 (16:9)", 1536, 864),
            ("1920x1080 (16:9)", 1920, 1080),
            ("2560x1440 (16:9)", 2560, 1440),
            # 9:16
            ("480x854 (9:16)", 480, 854),
            ("720x1280 (9:16)", 720, 1280),
            ("864x1536 (9:16)", 864, 1536),
            ("1080x1920 (9:16)", 1080, 1920),
            ("1440x2560 (9:16)", 1440, 2560),
            # Alt 640p-ish
            ("1136x640 (16:9)", 1136, 640),
            ("640x1136 (9:16)", 640, 1136),
            # 21:9 and 9:21
            ("1280x544 (21:9)", 1280, 544),
            ("544x1280 (9:21)", 544, 1280),
            # 4:3
            ("1104x832 (4:3)", 1104, 832),
            ("832x1104 (3:4)", 832, 1104),
        ]
        for label, w, h in self._size_presets:
            self.size_combo.addItem(label, (w, h))
        # default selection
        idx_default = next((i for i,(lbl,_,__) in enumerate(self._size_presets) if "1280x720" in lbl), 0)  # 1280x720 default
        if 0 <= idx_default < self.size_combo.count():
            self.size_combo.setCurrentIndex(idx_default)
        # Optional manual override (advanced later; present but small)
        self.size_manual_w = QSpinBox(); self.size_manual_w.setRange(256, 4096)  # allow manual width(256, 4096); self.siself.size_manual_w.setSingleStep(64); self.size_manual_w.setValue(1280)
        self.size_manual_h = QSpinBox(); self.size_manual_h.setRange(256, 4096)  # allow manual height(256, 4096); self.sself.size_manual_h.setSingleStep(64); self.size_manual_h.setValue(720)
        self.size_lock = QCheckBox("Lock aspect")
        self.size_lock.setChecked(False)  # default removed; will restore via UI
        def _on_size_combo_changed(i):
            data = self.size_combo.itemData(i)
            if data:
                w, h = data
                self.size_manual_w.blockSignals(True); self.size_manual_h.blockSignals(True)
                self.size_manual_w.setValue(int(w)); self.size_manual_h.setValue(int(h))
                self.size_manual_w.blockSignals(False); self.size_manual_h.blockSignals(False)
        self.size_combo.currentIndexChanged.connect(_on_size_combo_changed)
        def _sync_manual_w(v):
            if not self.size_lock.isChecked(): return
            # keep aspect from current combo data
            data = self.size_combo.currentData()
            if not data: return
            w0, h0 = data
            aspect = h0 / w0 if w0 else 1.0
            new_h = max(256, int(round(v * aspect / 64) * 64))
            self.size_manual_h.blockSignals(True); self.size_manual_h.setValue(new_h); self.size_manual_h.blockSignals(False)
        def _sync_manual_h(v):
            if not self.size_lock.isChecked(): return
            data = self.size_combo.currentData()
            if not data: return
            w0, h0 = data
            aspect = w0 / h0 if h0 else 1.0
            new_w = max(256, int(round(v * aspect / 64) * 64))
            self.size_manual_w.blockSignals(True); self.size_manual_w.setValue(new_w); self.size_manual_w.blockSignals(False)
        self.size_manual_w.valueChanged.connect(_sync_manual_w)
        self.size_manual_h.valueChanged.connect(_sync_manual_h)
        size_row.addWidget(self.size_combo, 2)
        size_row.addWidget(QLabel("W:"), 0)
        size_row.addWidget(self.size_manual_w, 0)
        size_row.addWidget(QLabel("H:"), 0)
        size_row.addWidget(self.size_manual_h, 0)
        size_row.addWidget(self.size_lock, 0)
        form.addRow(size_row)

        # Steps slider (right behind seed section)
        steps_row = QHBoxLayout()
        self.steps_slider = QSlider(Qt.Horizontal)
        self.steps_slider.setRange(10, 150)
        self.steps_slider.setValue(30)
        try:
            self.steps_slider.setTickPosition(QSlider.TicksBelow)
            self.steps_slider.setTickInterval(10)
        except Exception:
            pass
        self.steps_value = QLabel("30")
        self.steps_default = QCheckBox("Lock to default (30)")
        self.steps_default.setChecked(False)
        def _on_steps_default_changed(checked: bool):
            if checked:
                self.steps_slider.setValue(30)
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
        cfg_row.addWidget(QLabel("CFG:"))
        cfg_row.addWidget(self.cfg_scale, 0)
        form.addRow(cfg_row)

        # Seed used (always visible)
        self.seed_used_label = QLabel("Seed used: —")
        form.addRow(self.seed_used_label)

        # Output path + show in player + queue toggle
        out_row = QHBoxLayout()
        self.output_path = QLineEdit(str(Path("./output/images").resolve()))
        self.browse_btn = QPushButton("Browse…"); self.browse_btn.clicked.connect(self._on_browse)
        self.show_in_player = QCheckBox("Show in Player")
        self.use_queue = QCheckBox("Use queue (Add/Run)")
        out_row.addWidget(QLabel("Output:")); out_row.addWidget(self.output_path, 1); out_row.addWidget(self.browse_btn)
        out_row.addWidget(self.show_in_player); out_row.addWidget(self.use_queue)
        form.addRow(out_row)

        # VRAM profile override
        vram_row = QHBoxLayout()
        self.vram_profile = QComboBox(); self.vram_profile.addItems(["Auto", "6 GB", "8 GB", "12 GB", "24 GB"])
        self.restore_auto = QPushButton("Restore Auto"); self.restore_auto.clicked.connect(lambda: self.vram_profile.setCurrentIndex(0))
        vram_row.addWidget(QLabel("VRAM profile:")); vram_row.addWidget(self.vram_profile); vram_row.addWidget(self.restore_auto)
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
                self.model_combo.addItem("Auto (prefer SDXL)", "auto")
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

        self._model_picker = _Disclosure("Model", mdl_body, start_open=False, parent=self)
        root.addWidget(self._model_picker)

        # Advanced (collapsed group)
        adv_body = QWidget()
        adv_form = QFormLayout(adv_body)
        self.sampler = QComboBox(); self.sampler.addItems(["auto","DPM++ 2M (Karras)","Euler a","Euler","Heun","UniPC","DDIM"])
        self.attn_slicing = QCheckBox("Attention slicing")
        self.vae_device = QComboBox(); self.vae_device.addItems(["auto","cpu","gpu"])
        self.gpu_index = QSpinBox(); self.gpu_index.setRange(0,8)
        self.threads = QSpinBox(); self.threads.setRange(1,256); self.threads.setValue(8)
        self.nsfw_toggle = QCheckBox("Allow NSFW")
        self.format_combo = QComboBox(); self.format_combo.addItems(["png","jpg","webp"])
        self.filename_template = QLineEdit("sd_{seed}_{idx:03d}.png")
        self.reset_fname = QPushButton("Reset"); self.reset_fname.clicked.connect(lambda: self.filename_template.setText("sd_{seed}_{idx:03d}.png"))
        self.hires_helper = QCheckBox("Hi-res helper")
        self.fit_check = QCheckBox("Fit-check")
        adv_form.addRow("Sampler", self.sampler)
        adv_form.addRow(self.attn_slicing)
        adv_form.addRow("VAE device", self.vae_device)
        adv_form.addRow("GPU index", self.gpu_index)
        adv_form.addRow("Threads", self.threads)
        adv_form.addRow(self.nsfw_toggle)
        adv_form.addRow("File format", self.format_combo)
        rowf = QHBoxLayout(); rowf.addWidget(self.filename_template, 1); rowf.addWidget(self.reset_fname)
        adv_form.addRow("Filename", rowf)
        adv_form.addRow(self.hires_helper)
        adv_form.addRow(self.fit_check)

        root.addLayout(form)
        self._advanced = _Disclosure("Advanced", adv_body, start_open=False, parent=self)
        root.addWidget(self._advanced)

        # Progress + actions
        prog_row = QHBoxLayout()
        self.progress = QProgressBar(); self.progress.setRange(0,100)
        self.status = QLabel("Ready")
        prog_row.addWidget(self.progress, 1); prog_row.addWidget(self.status, 0)
        outer.addLayout(prog_row)

        btns = QHBoxLayout()
        self.add_to_queue = QPushButton("Add to Queue")
        self.add_and_run = QPushButton("Add & Run")
        self.generate_now = QPushButton("Generate")
        btns.addWidget(self.add_to_queue)
        btns.addWidget(self.add_and_run)
        btns.addWidget(self.generate_now)
        outer.addLayout(btns)



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
        job = {
            "type": "txt2img",
            "prompt": self.prompt.toPlainText().strip(),
            "negative": self.negative.toPlainText().strip(),
            "seed": seed,
            "seed_policy": ["fixed","random","increment"][self.seed_policy.currentIndex()],
            "batch": int(self.batch.value()),
            "cfg_scale": float(self.cfg_scale.value()) if hasattr(self, "cfg_scale") else 7.5,
            "output": self.output_path.text().strip(),
            "show_in_player": self.show_in_player.isChecked(),
            "vram_profile": self.vram_profile.currentText(),
            "sampler": self.sampler.currentText(),
            "model_path": (self.model_combo.currentData() if hasattr(self, "model_combo") else "auto"),
            "attn_slicing": self.attn_slicing.isChecked(),
            "vae_device": self.vae_device.currentText(),
            "gpu_index": int(self.gpu_index.value()),
            "threads": int(self.threads.value()),
            "nsfw": self.nsfw_toggle.isChecked(),
            "format": self.format_combo.currentText(),
            "filename_template": self.filename_template.text().strip() or "sd_{seed}_{idx:03d}.png",
            "hires_helper": self.hires_helper.isChecked(),
            "fit_check": self.fit_check.isChecked(),
            "steps": int(self.steps_slider.value()),
            "created_at": time.time(),
            "width": int(w_ui),
            "height": int(h_ui),

            "a1111_url": self._a1111_url_removed.text().strip() if hasattr(self, "a1111_url") else "http://127.0.0.1:7860",
                    }
        # Persist settings
        try:
            self._save_settings(job)
        except Exception as e:
            print("[txt2img] warn: could not save settings:", e)
        return job

        # Compute per-image seeds deterministically (unchanged)...

    def _save_settings(self, job: dict):
        try:
            from pathlib import Path as _P, json as _json  # type: ignore
        except Exception:
            pass
        return job

    def _load_settings(self):
        try:
            return
        except Exception:
            return

    def _enqueue(self):
        try:
            if not self._settings_path.exists():
                return
            with open(self._settings_path, "r", encoding="utf-8") as f:
                s = json.load(f)
            # fill UI
            self.prompt.setPlainText(s.get("prompt",""))
            self.negative.setPlainText(s.get("negative",""))
            self.seed.setValue(int(s.get("seed", 0)))
            sp = s.get("seed_policy","fixed")
            self.seed_policy.setCurrentIndex(["fixed","random","increment"].index(sp) if sp in ["fixed","random","increment"] else 0)
            self.batch.setValue(int(s.get("batch",1)))
            self.output_path.setText(s.get("output", str(Path("./output/images").resolve())))
            self.show_in_player.setChecked(bool(s.get("show_in_player", True)))
            self.use_queue.setChecked(bool(s.get("use_queue", False)))
            # size
            w = int(s.get("width", 1280)); h = int(s.get("height", 720))
            # try to select matching preset
            idx = -1
            for i,(label,wv,hv) in enumerate(self._size_presets):
                if wv==w and hv==h:
                    idx = i; break
            if idx>=0:
                self.size_combo.setCurrentIndex(idx)
            else:
                self.size_manual_w.setValue(w); self.size_manual_h.setValue(h)
            # steps & cfg
            self.steps_slider.setValue(int(s.get("steps", 30)))
            if hasattr(self, "cfg_scale"):
                self.cfg_scale.setValue(float(s.get("cfg_scale", 7.5)))
            # advanced
            self.vram_profile.setCurrentText(s.get("vram_profile","Auto"))
            self.sampler.setCurrentText(s.get("sampler","Euler a"))
            self.attn_slicing.setChecked(bool(s.get("attn_slicing", True)))
            self.vae_device.setCurrentText(s.get("vae_device","Auto"))
            self.gpu_index.setValue(int(s.get("gpu_index",0)))
            self.threads.setValue(int(s.get("threads",0)))
            self.nsfw_toggle.setChecked(bool(s.get("nsfw", False)))
            self.format_combo.setCurrentText(s.get("format","PNG"))
            self.filename_template.setText(s.get("filename_template","sd_{seed}_{idx:03d}.png"))
            self.hires_helper.setChecked(bool(s.get("hires_helper", False)))
            self.fit_check.setChecked(bool(s.get("fit_check", True)))
        except Exception as e:
            print("[txt2img] load settings error:", e)


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
        if not self.use_queue.isChecked():
            self._run_direct(job); return
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
            try:
                self.status.setText('Please enter a prompt')
            except Exception:
                pass
            return
        if not job or not job.get('prompt'):
            self.status.setText('Prompt is empty'); return
        self._enqueue(run_now=True)

    def _run_direct(self, job: dict):
        # Ensure a sane default filename template for Diffusers runs
        if not job.get('filename_template'):
            job['filename_template'] = 'sd_{seed}_{idx:03d}.png'

        self.status.setText("Generating…")
        self.progress.setValue(0)
        cancel_flag = threading.Event()

        def progress_cb(p):
            try:
                val = int(p*100) if isinstance(p, (float,)) and p <= 1.0 else int(p)
                self.progress.setValue(max(0, min(100, val)))
            except Exception:
                pass

        def worker():
            try:
                res = generate_qwen_images(job, progress_cb=progress_cb, cancel_event=cancel_flag)
                if res.get("ok"):
                    self.status.setText("Done")
                    if job.get("show_in_player") and res.get("files"):
                        self.fileReady.emit(res["files"][-1])
                else:
                    self.status.setText("Failed")
            except Exception as e:
                self.status.setText(f"Error: {e}")

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
            if progress_cb: progress_cb(i / max(1, batch))
            result = pipe(
                prompt=prompt,
                negative_prompt=negative or None,
                num_inference_steps=max(1, steps),
                guidance_scale=float(job.get("cfg_scale", 7.5)),
                width=width, height=height,
                generator=gen
            )
            img = result.images[0]
            fn_tmpl = job.get("filename_template") or "sd_{seed}_{idx:03d}.png"
            fname = fn_tmpl.format(seed=(seed if seed else 0)+i, idx=i)
            if not fname.lower().endswith((".png",".jpg",".jpeg",".webp")):
                fname += ".png"
            fpath = out_dir / fname
            img.save(str(fpath))
            files.append(str(fpath))
            if progress_cb: progress_cb((i+1)/max(1, batch))

        return {"files": files, "backend": "diffusers", "model": model_path, "actual_size": [width, height]}
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
            img.save(str(fpath))
            files.append(str(fpath))
            if progress_cb: progress_cb((i+1)/max(1,len(images_b64)))
        return {"files": files, "backend": "a1111"}
    except Exception:
        return None
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
    # Try Diffusers backend if available
    try:
        diff = _gen_via_diffusers(job, out_dir, progress_cb)
        if diff and diff.get('files'):
            files = diff['files']
            meta_backend = diff.get('backend') or 'diffusers'
            meta_model = diff.get('model')
            meta = {"ok": True, "prompt": prompt, "negative": job.get("negative",""),
                    "seed": seed, "seed_policy": seed_policy, "batch": batch, "files": files,
                    "created_at": time.time(), "engine": meta_backend, "model": meta_model,
                    "vram_profile": job.get("vram_profile","auto"), "steps": steps, "seeds": seeds_list,
                    "requested_size": [req_w, req_h], "actual_size": [safe_w, safe_h]}
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
        img = _draw_text_image(prompt, size=(safe_w, safe_h), seed=s)
        fname = fname_tmpl.format(seed=s, idx=i)
        if not fname.lower().endswith(("png","jpg","jpeg","webp")): fname += "." + fmt
        fpath = out_dir / fname
        img.save(str(fpath)); files.append(str(fpath))
        if progress_cb: progress_cb(((i+1)/batch))
        time.sleep(0.02)
    meta = {"ok": True, "prompt": prompt, "negative": job.get("negative",""), "seed": seed,
            "seed_policy": seed_policy, "batch": batch, "files": files, "created_at": time.time(),
            "engine": "fallback", "vram_profile": job.get("vram_profile","auto"), "steps": steps, "seeds": seeds_list,
            "requested_size": [req_w, req_h], "actual_size": [safe_w, safe_h]}
    try:
        with open(out_dir / (Path(files[0]).stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return meta
# <<< FRAMEVISION_QWEN_END