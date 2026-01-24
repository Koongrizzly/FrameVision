# -*- coding: utf-8 -*-
"""
helpers/glm-image.py  (Standalone GLM-Image Tester)

This file is intentionally self-contained and robust:
- UI runs on the Python you launch it with (only needs PySide6).
- Local inference always runs inside the GLM venv python:
      environments\.glm_image\Scripts\python.exe
  via a subprocess runner script generated under: temp\glm_image\_glm_local_runner.py

Folder structure (FrameVision root):
    environments/.glm_image/
    models/glm-image/
    presets/setsave/glm.json
    presets/extra_env/glm_install.bat
    output/glm-image/

Run:
    python helpers\glm-image.py
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, asdict, field
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets


# ---------------- Root detection ----------------

def _find_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "presets").exists() and (p / "helpers").exists():
            return p
        if (p / "presets" / "extra_env").exists():
            return p
    return start

def FV_ROOT() -> Path:
    return _find_root(Path(__file__).resolve().parent)


def _resolve(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (FV_ROOT() / pp).resolve()

def _rel_if_inside_root(p: Path) -> str:
    try:
        rel = p.resolve().relative_to(FV_ROOT().resolve())
        return str(rel).replace("/", "\\")
    except Exception:
        return str(p.resolve())

def _open_folder(p: Path) -> None:
    try:
        os.startfile(str(p))  # type: ignore[attr-defined]
    except Exception:
        pass


# ---------------- Settings ----------------

@dataclass
class Settings:
    env_dir: str = r"environments\.glm_image"
    models_dir: str = r"models\glm-image"
    output_dir: str = r"output\glm-image"

    # Backend selection
    backend: str = "local"  # local | sglang
    sglang_base_url: str = "http://localhost:30000"
    sglang_model: str = "zai-org/GLM-Image"

    # Local model selection
    use_local_model: bool = True
    hf_repo_id: str = "zai-org/GLM-Image"
    local_model_subdir: str = r"model\GLM-Image"

    # Compute
    torch_dtype: str = "float16"     # float16 | bfloat16 | float32
    device_map: str = "cuda"         # cuda | balanced  (your diffusers build supports these)
    enable_cpu_offload: bool = False
    enable_seq_cpu_offload: bool = False

    # Memory helpers (best-effort)
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_attention_slicing: bool = False

    # Defaults
    width: int = 1024
    height: int = 768
    steps: int = 25
    guidance_scale: float = 1.8
    seed: int = -1
    batch_size: int = 1
    temperature: float = 0.9
    top_p: float = 0.9
    strength: float = 0.85

    open_output_folder_after_save: bool = False



    # Remember last inputs (best-effort)
    last_t2i_prompt: str = ""
    last_t2i_negative: str = ""
    last_i2i_prompt: str = ""
    last_i2i_negative: str = ""
    last_i2i_images: list[str] = field(default_factory=list)
def _settings_path() -> Path:
    return FV_ROOT() / "presets" / "setsave" / "glm.json"


def load_settings() -> Settings:
    p = _settings_path()
    s = Settings()
    if not p.exists():
        return s
    try:
        data = json.loads(p.read_text(encoding="utf-8")) or {}
        base = asdict(s)
        for k, v in data.items():
            if k in base:
                base[k] = v
        s = Settings(**base)
        # migrate unsupported device_map values
        if str(s.device_map).lower().strip() not in ("cuda", "balanced"):
            s.device_map = "balanced"
        return s
    except Exception:
        return Settings()


def save_settings(s: Settings) -> None:
    p = _settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(s), indent=2), encoding="utf-8")


def installer_bat() -> Path:
    return FV_ROOT() / "presets" / "extra_env" / "glm_install.bat"


# ---------------- Local runner (executes inside venv python) ----------------

LOCAL_RUNNER = r"""
import base64, json, sys, traceback, inspect
from pathlib import Path

def _import_pipeline():
    try:
        from diffusers.pipelines.glm_image import GlmImagePipeline
        return GlmImagePipeline
    except Exception:
        pass
    try:
        from diffusers import GlmImagePipeline
        return GlmImagePipeline
    except Exception:
        pass
    try:
        from diffusers.pipelines.glm_image.pipeline_glm_image import GlmImagePipeline
        return GlmImagePipeline
    except Exception:
        raise ModuleNotFoundError(
            "diffusers does not contain GLM-Image pipeline. Re-run installer (diffusers/transformers from source)."
        )

def _maybe_enable(pipe, name, enabled):
    if not enabled:
        return
    fn = getattr(pipe, name, None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "Missing job JSON path"}))
        return 2

    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))

    try:
        import torch
        from PIL import Image
        from io import BytesIO

        GlmImagePipeline = _import_pipeline()

        # ---- Always define everything we reference (no NameError) ----
        mode = str(job.get("mode", "t2i")).lower().strip()
        prompt = (job.get("prompt") or "").strip()
        if not prompt:
            raise RuntimeError("Missing prompt.")
        negative = (job.get("negative_prompt") or "").strip()

        width = int(job.get("width", 1024))
        height = int(job.get("height", 1024))
        steps = int(job.get("steps", 25))
        guidance = float(job.get("guidance_scale", 1.8))
        seed = int(job.get("seed", 0))
        batch = int(job.get("batch_size", 1))
        temperature = float(job.get("temperature", 0.9))
        top_p = float(job.get("top_p", 0.9))
        strength = float(job.get("strength", 0.85))

        torch_dtype_s = str(job.get("torch_dtype", "float16")).lower().strip()
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(torch_dtype_s, torch.float16)

        device_map = str(job.get("device_map", "cuda")).lower().strip()
        if device_map not in ("cuda", "balanced"):
            device_map = "balanced"

        use_local_model = bool(job.get("use_local_model"))
        repo_or_path = job.get("hf_repo_id", "zai-org/GLM-Image")

        if use_local_model:
            models_dir = Path(job.get("models_dir", ""))
            subdir = job.get("local_model_subdir", r"model\\GLM-Image")
            local_path = (models_dir / subdir)
            if local_path.exists():
                repo_or_path = str(local_path)

        # Offload flags
        enable_offload = bool(job.get("enable_cpu_offload")) or bool(job.get("enable_seq_cpu_offload"))

        # Load
        load_kwargs = {"torch_dtype": torch_dtype, "device_map": device_map}
        pipe = GlmImagePipeline.from_pretrained(repo_or_path, **load_kwargs)

        # Memory helpers (best-effort)
        _maybe_enable(pipe, "enable_vae_slicing", bool(job.get("enable_vae_slicing")))
        _maybe_enable(pipe, "enable_vae_tiling", bool(job.get("enable_vae_tiling")))
        if bool(job.get("enable_attention_slicing")):
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass

        # Offload (best-effort)
        if enable_offload:
            if bool(job.get("enable_seq_cpu_offload")) and hasattr(pipe, "enable_sequential_cpu_offload"):
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass
            elif bool(job.get("enable_cpu_offload")) and hasattr(pipe, "enable_model_cpu_offload"):
                try:
                    pipe.enable_model_cpu_offload()
                except Exception:
                    pass

        # Device consistency fix for i2i: ensure every module is on CUDA if we expect CUDA and not offloading
        if (not enable_offload) and device_map == "cuda":
            try:
                comps = getattr(pipe, "components", None)
                if isinstance(comps, dict):
                    for _k, _v in comps.items():
                        if hasattr(_v, "to"):
                            try:
                                _v.to("cuda")
                            except Exception:
                                pass
            except Exception:
                pass
            try:
                pipe.to("cuda")
            except Exception:
                pass

        # Generator
        gen_dev = "cuda" if device_map == "cuda" else "cpu"
        gen = torch.Generator(device=gen_dev).manual_seed(seed)

        # Call kwargs, only pass supported params
        sig = inspect.signature(pipe.__call__)
        supported = set(sig.parameters.keys())

        kw = dict(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
        )
        if "negative_prompt" in supported and negative:
            kw["negative_prompt"] = negative
        if "temperature" in supported:
            kw["temperature"] = temperature
        if "top_p" in supported:
            kw["top_p"] = top_p
        if "num_images_per_prompt" in supported and batch > 1:
            kw["num_images_per_prompt"] = batch

        if mode == "i2i":
            images = []
            for p in (job.get("images") or []):
                images.append(Image.open(p).convert("RGB"))
            if not images:
                raise RuntimeError("No input images for image-to-image.")
            if "image" in supported:
                kw["image"] = images
            elif "images" in supported:
                kw["images"] = images
            else:
                kw["image"] = images
            if "strength" in supported:
                kw["strength"] = strength

        out = pipe(**kw)
        imgs = out.images if hasattr(out, "images") else []
        if not isinstance(imgs, list):
            imgs = [imgs]

        b64s = []
        for im in imgs:
            bio = BytesIO()
            im.save(bio, format="PNG")
            b64s.append(base64.b64encode(bio.getvalue()).decode("ascii"))

        print(json.dumps({"ok": True, "images_b64": b64s}))
        return 0

    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e), "trace": traceback.format_exc()}))
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
"""


# ---------------- Worker ----------------

class Worker(QtCore.QObject):
    progress = QtCore.Signal(str)
    error = QtCore.Signal(str)
    finished = QtCore.Signal(object)

    def __init__(self, job: dict, s: Settings):
        super().__init__()
        self.job = job
        self.s = s

    @QtCore.Slot()
    def run(self) -> None:
        try:
            if self.s.backend == "sglang":
                out = self._run_sglang()
            else:
                out = self._run_local()
            self.finished.emit(out)
        except Exception as e:
            self.error.emit(str(e) + "\n\n" + traceback.format_exc())

    def _run_local(self) -> dict:
        env_py = _resolve(self.s.env_dir) / "Scripts" / "python.exe"
        if not env_py.exists():
            raise FileNotFoundError("Env python not found: " + str(env_py))

        temp_dir = FV_ROOT() / "temp" / "glm_image"
        temp_dir.mkdir(parents=True, exist_ok=True)

        runner_path = temp_dir / "_glm_local_runner.py"
        job_path = temp_dir / "_glm_job.json"
        runner_path.write_text(LOCAL_RUNNER, encoding="utf-8")

        payload = dict(self.job)
        payload.update({
            "use_local_model": bool(self.s.use_local_model),
            "hf_repo_id": self.s.hf_repo_id,
            "models_dir": str(_resolve(self.s.models_dir)),
            "local_model_subdir": self.s.local_model_subdir,
            "torch_dtype": self.s.torch_dtype,
            "device_map": self.s.device_map,
            "enable_cpu_offload": bool(self.s.enable_cpu_offload),
            "enable_seq_cpu_offload": bool(self.s.enable_seq_cpu_offload),
            "enable_vae_slicing": bool(self.s.enable_vae_slicing),
            "enable_vae_tiling": bool(self.s.enable_vae_tiling),
            "enable_attention_slicing": bool(self.s.enable_attention_slicing),
        })
        job_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self.progress.emit("Using env python: " + str(env_py))
        self.progress.emit("Running local inference (env subprocess)...")

        p = subprocess.run(
            [str(env_py), str(runner_path), str(job_path)],
            cwd=str(FV_ROOT()),
            capture_output=True,
            text=True,
        )

        stdout = (p.stdout or "").strip()
        stderr = (p.stderr or "").strip()

        data = None
        if stdout:
            try:
                data = json.loads(stdout)
            except Exception:
                # sometimes other output lines exist; try last line
                try:
                    data = json.loads(stdout.splitlines()[-1])
                except Exception:
                    data = None

        if p.returncode != 0 or not isinstance(data, dict) or not data.get("ok"):
            err = ""
            if isinstance(data, dict) and data.get("error"):
                err += data.get("error", "") + "\n\n" + str(data.get("trace", ""))
            else:
                err += "Local runner failed."
                if stdout:
                    err += "\n\nSTDOUT:\n" + stdout
                if stderr:
                    err += "\n\nSTDERR:\n" + stderr
            raise RuntimeError(err.strip())

        images_bytes = [base64.b64decode(b) for b in (data.get("images_b64") or [])]
        return {"backend": "local", "images_bytes": images_bytes}

    def _run_sglang(self) -> dict:
        raise RuntimeError("SGLang backend not implemented in this rebuild. Use local.")


# ---------------- UI ----------------

class Pane(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.s = load_settings()
        self._thread = None
        self._worker = None

        # Debounced autosave so prompts/negatives/images persist across restarts
        self._applying = False
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._collect)

        self._build()
        self._apply()
        self._refresh_status()

    def _build(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        title = QtWidgets.QLabel("GLM-Image — Standalone Tester (Clean rebuild)")
        f = title.font()
        f.setPointSize(max(12, f.pointSize() + 4))
        f.setBold(True)
        title.setFont(f)

        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        tr = QtWidgets.QHBoxLayout()
        tr.addWidget(title)
        tr.addStretch(1)
        tr.addWidget(self.lbl_status)
        root.addLayout(tr)

        # Folders
        gb = QtWidgets.QGroupBox("Folders")
        g = QtWidgets.QGridLayout(gb)
        g.setColumnStretch(1, 1)

        self.ed_env = QtWidgets.QLineEdit()
        self.btn_env = QtWidgets.QPushButton("Browse…")
        self.btn_env.clicked.connect(self._pick_env)

        self.ed_models = QtWidgets.QLineEdit()
        self.btn_models = QtWidgets.QPushButton("Browse…")
        self.btn_models.clicked.connect(self._pick_models)

        self.btn_open_env = QtWidgets.QPushButton("Open env")
        self.btn_open_env.clicked.connect(lambda: self._open_path(self.ed_env.text().strip()))

        self.btn_open_models = QtWidgets.QPushButton("Open models")
        self.btn_open_models.clicked.connect(lambda: self._open_path(self.ed_models.text().strip()))

        self.btn_install = QtWidgets.QPushButton("One-click install (CUDA)")
        self.btn_install.clicked.connect(self._on_install)

        g.addWidget(QtWidgets.QLabel("Env dir"), 0, 0)
        g.addWidget(self.ed_env, 0, 1)
        g.addWidget(self.btn_env, 0, 2)
        g.addWidget(self.btn_open_env, 1, 2)

        g.addWidget(QtWidgets.QLabel("Models dir"), 2, 0)
        g.addWidget(self.ed_models, 2, 1)
        g.addWidget(self.btn_models, 2, 2)
        g.addWidget(self.btn_open_models, 3, 2)

        g.addWidget(self.btn_install, 4, 0, 1, 3)
        root.addWidget(gb)

        # Local compute
        gb2 = QtWidgets.QGroupBox("Local compute")
        g2 = QtWidgets.QGridLayout(gb2)
        g2.setColumnStretch(1, 1)

        self.cmb_dtype = QtWidgets.QComboBox()
        self.cmb_dtype.addItems(["float16", "bfloat16", "float32"])
        self.cmb_devmap = QtWidgets.QComboBox()
        self.cmb_devmap.addItems(["cuda", "balanced"])

        self.chk_offload = QtWidgets.QCheckBox("Enable model CPU offload (best-effort)")
        self.chk_seq_offload = QtWidgets.QCheckBox("Enable sequential CPU offload (best-effort)")

        self.chk_vae_slicing = QtWidgets.QCheckBox("Enable VAE slicing (recommended)")
        self.chk_vae_tiling = QtWidgets.QCheckBox("Enable VAE tiling (recommended)")
        self.chk_attn_slicing = QtWidgets.QCheckBox("Enable attention slicing (slower)")

        g2.addWidget(QtWidgets.QLabel("torch dtype"), 0, 0)
        g2.addWidget(self.cmb_dtype, 0, 1)
        g2.addWidget(QtWidgets.QLabel("device_map"), 1, 0)
        g2.addWidget(self.cmb_devmap, 1, 1)
        g2.addWidget(self.chk_offload, 2, 0, 1, 2)
        g2.addWidget(self.chk_seq_offload, 3, 0, 1, 2)
        g2.addWidget(self.chk_vae_slicing, 4, 0, 1, 2)
        g2.addWidget(self.chk_vae_tiling, 5, 0, 1, 2)
        g2.addWidget(self.chk_attn_slicing, 6, 0, 1, 2)
        root.addWidget(gb2)

        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        self.tab_t2i = QtWidgets.QWidget()
        self.tab_i2i = QtWidgets.QWidget()
        self.tabs.addTab(self.tab_t2i, "Text → Image")
        self.tabs.addTab(self.tab_i2i, "Image → Image")
        root.addWidget(self.tabs, 1)

        self._build_t2i()
        self._build_i2i()

        # Output / run
        gb3 = QtWidgets.QGroupBox("Run")
        g3 = QtWidgets.QGridLayout(gb3)
        g3.setColumnStretch(1, 1)
        self.ed_out = QtWidgets.QLineEdit()
        self.btn_out = QtWidgets.QPushButton("Browse…")
        self.btn_out.clicked.connect(self._pick_out)
        self.chk_open_out = QtWidgets.QCheckBox("Open output folder after saving")
        self.btn_run = QtWidgets.QPushButton("Generate")
        self.btn_run.clicked.connect(self._on_run)

        g3.addWidget(QtWidgets.QLabel("Output dir"), 0, 0)
        g3.addWidget(self.ed_out, 0, 1)
        g3.addWidget(self.btn_out, 0, 2)
        g3.addWidget(self.chk_open_out, 1, 0, 1, 3)
        g3.addWidget(self.btn_run, 2, 0, 1, 3)
        root.addWidget(gb3)

        # Preview + logs
        gb4 = QtWidgets.QGroupBox("Preview and logs")
        v = QtWidgets.QVBoxLayout(gb4)
        self.preview = QtWidgets.QLabel("No output yet.")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(240)
        self.preview.setStyleSheet("QLabel { border: 1px solid rgba(255,255,255,0.15); }")
        v.addWidget(self.preview)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        v.addWidget(self.log, 1)
        root.addWidget(gb4, 2)

    def _build_t2i(self) -> None:
        g = QtWidgets.QGridLayout(self.tab_t2i)
        g.setColumnStretch(1, 1)

        self.t_prompt = QtWidgets.QPlainTextEdit()
        self.t_neg = QtWidgets.QPlainTextEdit()

        self.t_prompt.textChanged.connect(self._schedule_save)
        self.t_neg.textChanged.connect(self._schedule_save)

        self.t_w = QtWidgets.QSpinBox(); self.t_w.setRange(256, 4096); self.t_w.setSingleStep(32)
        self.t_h = QtWidgets.QSpinBox(); self.t_h.setRange(256, 4096); self.t_h.setSingleStep(32)
        self.t_steps = QtWidgets.QSpinBox(); self.t_steps.setRange(1, 200)
        self.t_guid = QtWidgets.QDoubleSpinBox(); self.t_guid.setRange(0.0, 20.0); self.t_guid.setSingleStep(0.1)
        self.t_seed = QtWidgets.QSpinBox(); self.t_seed.setRange(-1, 2_000_000_000)
        self.t_batch = QtWidgets.QSpinBox(); self.t_batch.setRange(1, 8)
        self.t_temp = QtWidgets.QDoubleSpinBox(); self.t_temp.setRange(0.0, 2.0); self.t_temp.setSingleStep(0.05)
        self.t_top_p = QtWidgets.QDoubleSpinBox(); self.t_top_p.setRange(0.0, 1.0); self.t_top_p.setSingleStep(0.05)

        r = 0
        g.addWidget(QtWidgets.QLabel("Prompt"), r, 0); g.addWidget(self.t_prompt, r, 1, 1, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Negative prompt"), r, 0); g.addWidget(self.t_neg, r, 1, 1, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Width"), r, 0); g.addWidget(self.t_w, r, 1)
        g.addWidget(QtWidgets.QLabel("Height"), r, 2); g.addWidget(self.t_h, r, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Steps"), r, 0); g.addWidget(self.t_steps, r, 1)
        g.addWidget(QtWidgets.QLabel("Guidance"), r, 2); g.addWidget(self.t_guid, r, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Seed (-1=random)"), r, 0); g.addWidget(self.t_seed, r, 1)
        g.addWidget(QtWidgets.QLabel("Batch"), r, 2); g.addWidget(self.t_batch, r, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Temperature"), r, 0); g.addWidget(self.t_temp, r, 1)
        g.addWidget(QtWidgets.QLabel("top_p"), r, 2); g.addWidget(self.t_top_p, r, 3); r += 1
        g.setRowStretch(r, 1)

    def _build_i2i(self) -> None:
        g = QtWidgets.QGridLayout(self.tab_i2i)
        g.setColumnStretch(1, 1)

        self.i_prompt = QtWidgets.QPlainTextEdit()
        self.i_neg = QtWidgets.QPlainTextEdit()

        self.i_prompt.textChanged.connect(self._schedule_save)
        self.i_neg.textChanged.connect(self._schedule_save)

        self.i_list = QtWidgets.QListWidget()
        self.i_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.i_list.setIconSize(QtCore.QSize(96, 96))

        self.btn_add = QtWidgets.QPushButton("Add image(s)…")
        self.btn_rm = QtWidgets.QPushButton("Remove selected")
        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_add.clicked.connect(self._i_add)
        self.btn_rm.clicked.connect(self._i_remove)
        self.btn_clear.clicked.connect(self._i_clear)

        vb = QtWidgets.QVBoxLayout()
        vb.addWidget(self.btn_add); vb.addWidget(self.btn_rm); vb.addWidget(self.btn_clear); vb.addStretch(1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.i_list, 1)
        row.addLayout(vb)

        self.i_w = QtWidgets.QSpinBox(); self.i_w.setRange(256, 4096); self.i_w.setSingleStep(32)
        self.i_h = QtWidgets.QSpinBox(); self.i_h.setRange(256, 4096); self.i_h.setSingleStep(32)
        self.i_steps = QtWidgets.QSpinBox(); self.i_steps.setRange(1, 200)
        self.i_guid = QtWidgets.QDoubleSpinBox(); self.i_guid.setRange(0.0, 20.0); self.i_guid.setSingleStep(0.1)
        self.i_seed = QtWidgets.QSpinBox(); self.i_seed.setRange(-1, 2_000_000_000)
        self.i_batch = QtWidgets.QSpinBox(); self.i_batch.setRange(1, 8)
        self.i_strength = QtWidgets.QDoubleSpinBox(); self.i_strength.setRange(0.0, 1.0); self.i_strength.setSingleStep(0.05)
        self.i_temp = QtWidgets.QDoubleSpinBox(); self.i_temp.setRange(0.0, 2.0); self.i_temp.setSingleStep(0.05)
        self.i_top_p = QtWidgets.QDoubleSpinBox(); self.i_top_p.setRange(0.0, 1.0); self.i_top_p.setSingleStep(0.05)

        r = 0
        g.addWidget(QtWidgets.QLabel("Prompt / instruction"), r, 0); g.addWidget(self.i_prompt, r, 1, 1, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Negative prompt"), r, 0); g.addWidget(self.i_neg, r, 1, 1, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Input images"), r, 0); g.addLayout(row, r, 1, 1, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Width"), r, 0); g.addWidget(self.i_w, r, 1)
        g.addWidget(QtWidgets.QLabel("Height"), r, 2); g.addWidget(self.i_h, r, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Steps"), r, 0); g.addWidget(self.i_steps, r, 1)
        g.addWidget(QtWidgets.QLabel("Guidance"), r, 2); g.addWidget(self.i_guid, r, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Seed (-1=random)"), r, 0); g.addWidget(self.i_seed, r, 1)
        g.addWidget(QtWidgets.QLabel("Batch"), r, 2); g.addWidget(self.i_batch, r, 3); r += 1
        g.addWidget(QtWidgets.QLabel("Strength"), r, 0); g.addWidget(self.i_strength, r, 1); r += 1
        g.addWidget(QtWidgets.QLabel("Temperature"), r, 0); g.addWidget(self.i_temp, r, 1)
        g.addWidget(QtWidgets.QLabel("top_p"), r, 2); g.addWidget(self.i_top_p, r, 3); r += 1
        g.setRowStretch(r, 1)

    def _apply(self) -> None:
        self._applying = True
        try:
            self.ed_env.setText(self.s.env_dir)
            self.ed_models.setText(self.s.models_dir)
            self.ed_out.setText(self.s.output_dir)
            self.chk_open_out.setChecked(bool(self.s.open_output_folder_after_save))

            self.cmb_dtype.setCurrentText(self.s.torch_dtype)
            self.cmb_devmap.setCurrentText(self.s.device_map)
            self.chk_offload.setChecked(bool(self.s.enable_cpu_offload))
            self.chk_seq_offload.setChecked(bool(self.s.enable_seq_cpu_offload))
            self.chk_vae_slicing.setChecked(bool(self.s.enable_vae_slicing))
            self.chk_vae_tiling.setChecked(bool(self.s.enable_vae_tiling))
            self.chk_attn_slicing.setChecked(bool(self.s.enable_attention_slicing))

            # Defaults both tabs
            self.t_w.setValue(int(self.s.width)); self.t_h.setValue(int(self.s.height))
            self.t_steps.setValue(int(self.s.steps)); self.t_guid.setValue(float(self.s.guidance_scale))
            self.t_seed.setValue(int(self.s.seed)); self.t_batch.setValue(int(self.s.batch_size))
            self.t_temp.setValue(float(self.s.temperature)); self.t_top_p.setValue(float(self.s.top_p))

            self.i_w.setValue(int(self.s.width)); self.i_h.setValue(int(self.s.height))
            self.i_steps.setValue(int(self.s.steps)); self.i_guid.setValue(float(self.s.guidance_scale))
            self.i_seed.setValue(int(self.s.seed)); self.i_batch.setValue(int(self.s.batch_size))
            self.i_strength.setValue(float(self.s.strength))
            self.i_temp.setValue(float(self.s.temperature)); self.i_top_p.setValue(float(self.s.top_p))

            # Remembered prompts / negatives
            self.t_prompt.setPlainText(self.s.last_t2i_prompt or "")
            self.t_neg.setPlainText(self.s.last_t2i_negative or "")
            self.i_prompt.setPlainText(self.s.last_i2i_prompt or "")
            self.i_neg.setPlainText(self.s.last_i2i_negative or "")

            # Remembered input images (only keep existing files)
            self.i_list.clear()
            for p in (self.s.last_i2i_images or []):
                try:
                    pp = Path(p)
                    if pp.exists():
                        self._i_add_item(pp)
                except Exception:
                    pass
        finally:
            self._applying = False

    def _collect(self) -> None:
        self.s.env_dir = self.ed_env.text().strip() or self.s.env_dir
        self.s.models_dir = self.ed_models.text().strip() or self.s.models_dir
        self.s.output_dir = self.ed_out.text().strip() or self.s.output_dir
        self.s.open_output_folder_after_save = self.chk_open_out.isChecked()

        self.s.torch_dtype = self.cmb_dtype.currentText().strip()
        self.s.device_map = self.cmb_devmap.currentText().strip()
        self.s.enable_cpu_offload = self.chk_offload.isChecked()
        self.s.enable_seq_cpu_offload = self.chk_seq_offload.isChecked()
        self.s.enable_vae_slicing = self.chk_vae_slicing.isChecked()
        self.s.enable_vae_tiling = self.chk_vae_tiling.isChecked()
        self.s.enable_attention_slicing = self.chk_attn_slicing.isChecked()

        # keep defaults from current tab t2i
        self.s.width = int(self.t_w.value())
        self.s.height = int(self.t_h.value())
        self.s.steps = int(self.t_steps.value())
        self.s.guidance_scale = float(self.t_guid.value())
        self.s.seed = int(self.t_seed.value())
        self.s.batch_size = int(self.t_batch.value())
        self.s.temperature = float(self.t_temp.value())
        self.s.top_p = float(self.t_top_p.value())
        self.s.strength = float(self.i_strength.value())

        # Remember last inputs
        try:
            self.s.last_t2i_prompt = self.t_prompt.toPlainText()
            self.s.last_t2i_negative = self.t_neg.toPlainText()
            self.s.last_i2i_prompt = self.i_prompt.toPlainText()
            self.s.last_i2i_negative = self.i_neg.toPlainText()
            self.s.last_i2i_images = [
                (self.i_list.item(i).data(QtCore.Qt.UserRole) or "")
                for i in range(self.i_list.count())
            ]
            self.s.last_i2i_images = [p for p in self.s.last_i2i_images if p]
        except Exception:
            pass

        save_settings(self.s)

    def _schedule_save(self) -> None:
        if getattr(self, "_applying", False):
            return
        # Debounce frequent typing / list edits
        try:
            self._save_timer.start(400)
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        self.log.appendPlainText(msg)
        sb = self.log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _refresh_status(self) -> None:
        self._collect()
        env_py = _resolve(self.s.env_dir) / "Scripts" / "python.exe"
        env_ok = env_py.exists()
        model_ok = True
        if self.s.use_local_model:
            model_ok = (_resolve(self.s.models_dir) / self.s.local_model_subdir).exists()

        self.lbl_status.setText(("ENV: OK" if env_ok else "ENV: missing") + " | " + ("MODEL: OK" if model_ok else "MODEL: missing"))

    def _pick_env(self) -> None:
        start = str(_resolve(self.ed_env.text().strip() or self.s.env_dir))
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select environment folder (.glm_image)", start)
        if p:
            self.ed_env.setText(_rel_if_inside_root(Path(p)))
            self._refresh_status()

    def _pick_models(self) -> None:
        start = str(_resolve(self.ed_models.text().strip() or self.s.models_dir))
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select models folder (glm-image)", start)
        if p:
            self.ed_models.setText(_rel_if_inside_root(Path(p)))
            self._refresh_status()

    def _pick_out(self) -> None:
        start = str(_resolve(self.ed_out.text().strip() or self.s.output_dir))
        p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if p:
            self.ed_out.setText(_rel_if_inside_root(Path(p)))
            self._collect()

    def _open_path(self, p: str) -> None:
        pp = _resolve(p)
        pp.mkdir(parents=True, exist_ok=True)
        _open_folder(pp)

    def _on_install(self) -> None:
        bat = installer_bat()
        if not bat.exists():
            QtWidgets.QMessageBox.critical(self, "Missing installer", "Installer not found:\n" + str(bat))
            return
        subprocess.Popen(["cmd.exe", "/c", "start", "cmd.exe", "/k", str(bat)], cwd=str(FV_ROOT()))
        self._log("Opened installer in a new CMD window.")


    def _i_add_item(self, p: Path) -> None:
        it = QtWidgets.QListWidgetItem(p.name)
        it.setData(QtCore.Qt.UserRole, str(p))
        try:
            img = QtGui.QImage(str(p))
            if not img.isNull():
                it.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(img)))
        except Exception:
            pass
        self.i_list.addItem(it)

    def _i_add(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Add image(s)",
            str(FV_ROOT()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;All files (*.*)",
        )
        for f in files or []:
            try:
                self._i_add_item(Path(f))
            except Exception:
                pass
        self._schedule_save()

    def _i_remove(self) -> None:
        for it in self.i_list.selectedItems():
            self.i_list.takeItem(self.i_list.row(it))
        self._schedule_save()

    def _i_clear(self) -> None:
        self.i_list.clear()
        self._schedule_save()

    def _on_run(self) -> None:
        self._collect()
        mode = "t2i" if self.tabs.currentWidget() == self.tab_t2i else "i2i"

        if mode == "t2i":
            prompt = self.t_prompt.toPlainText().strip()
            neg = self.t_neg.toPlainText().strip()
            w, h = int(self.t_w.value()), int(self.t_h.value())
            steps, gs = int(self.t_steps.value()), float(self.t_guid.value())
            seed, batch = int(self.t_seed.value()), int(self.t_batch.value())
            temp, top_p = float(self.t_temp.value()), float(self.t_top_p.value())
            images = []
            strength = None
        else:
            prompt = self.i_prompt.toPlainText().strip()
            neg = self.i_neg.toPlainText().strip()
            w, h = int(self.i_w.value()), int(self.i_h.value())
            steps, gs = int(self.i_steps.value()), float(self.i_guid.value())
            seed, batch = int(self.i_seed.value()), int(self.i_batch.value())
            temp, top_p = float(self.i_temp.value()), float(self.i_top_p.value())
            strength = float(self.i_strength.value())
            images = [self.i_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.i_list.count())]

        if not prompt:
            QtWidgets.QMessageBox.warning(self, "Missing prompt", "Please enter a prompt.")
            return
        if (w % 32) != 0 or (h % 32) != 0:
            QtWidgets.QMessageBox.warning(self, "Invalid resolution", "Width and height must be divisible by 32.")
            return
        if mode == "i2i" and not images:
            QtWidgets.QMessageBox.warning(self, "Missing input image", "Please add at least one input image.")
            return
        if seed == -1:
            seed = int(QtCore.QRandomGenerator.global_().generate()) % 2_000_000_000

        job = {
            "mode": mode,
            "prompt": prompt,
            "negative_prompt": neg,
            "width": w,
            "height": h,
            "steps": steps,
            "guidance_scale": gs,
            "seed": seed,
            "batch_size": batch,
            "temperature": temp,
            "top_p": top_p,
        }
        if images:
            job["images"] = images
        if strength is not None:
            job["strength"] = strength

        self._start(job)

    def _start(self, job: dict) -> None:
        if self._thread is not None:
            QtWidgets.QMessageBox.information(self, "Busy", "A generation is already running.")
            return

        self.btn_run.setEnabled(False)

        self._log("----")
        self._log("FrameVision root: " + str(FV_ROOT()))
        self._log("Env dir: " + str(_resolve(self.s.env_dir)))
        self._log("Models dir: " + str(_resolve(self.s.models_dir)))
        self._log("Backend: local | Mode: " + job.get("mode", "?"))

        self._thread = QtCore.QThread(self)
        self._worker = Worker(job, self.s)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._log)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._on_done)
        self._worker.finished.connect(lambda _: self._thread.quit())
        self._thread.finished.connect(self._cleanup)

        self._thread.start()

    def _cleanup(self) -> None:
        try:
            if self._worker is not None:
                self._worker.deleteLater()
            if self._thread is not None:
                self._thread.deleteLater()
        finally:
            self._worker = None
            self._thread = None
            self.btn_run.setEnabled(True)

    def _on_error(self, msg: str) -> None:
        self._log("ERROR:\n" + msg)
        QtWidgets.QMessageBox.critical(self, "GLM-Image error", msg)

    def _on_done(self, result: object) -> None:
        images_bytes = []
        if isinstance(result, dict):
            images_bytes = result.get("images_bytes", []) or []
        if not images_bytes:
            self._log("No images returned.")
            return

        out_dir = _resolve(self.s.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # filename from prompt
        base_name = (self._current_prompt() or "glm_image").strip().replace("\n", " ")
        base_name = "".join(ch for ch in base_name if ch not in '<>:"/\\|?*')[:120].strip() or "glm_image"
        stamp = QtCore.QDateTime.currentDateTime().toString("yyyyMMdd_HHmmss")

        saved = []
        for i, b in enumerate(images_bytes, start=1):
            fp = out_dir / f"{base_name}_{stamp}_{i:02d}.png"
            fp.write_bytes(b)
            saved.append(fp)

        self._log("Saved %d image(s) to: %s" % (len(saved), str(out_dir)))
        self._set_preview(saved[0])

        if self.chk_open_out.isChecked():
            _open_folder(out_dir)

    def _current_prompt(self) -> str:
        return self.t_prompt.toPlainText().strip() if self.tabs.currentWidget() == self.tab_t2i else self.i_prompt.toPlainText().strip()

    def _set_preview(self, path: Path) -> None:
        pm = QtGui.QPixmap(str(path))
        if pm.isNull():
            self.preview.setText("Preview unavailable.")
            return
        self.preview.setPixmap(pm.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        super().resizeEvent(e)
        pm = self.preview.pixmap()
        if pm is not None and not pm.isNull():
            self.preview.setPixmap(pm.scaled(self.preview.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("GLM-Image — Standalone Tester")
    pane = Pane()

    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
    scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
    scroll.setWidget(pane)

    win.setCentralWidget(scroll)
    win.resize(1150, 900)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
