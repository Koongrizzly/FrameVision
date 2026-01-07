# helpers/qwen2511.py
# Qwen Image Edit 2511 (GGUF) – PySide6 pane + standalone test runner
#
# Standalone:
#   .\\.qwen2512\\venv\\Scripts\\python.exe .\\helpers\\qwen2511.py
#
# Features:
# - Runs sd-cli (stable-diffusion.cpp) Qwen Image Edit GGUF
# - Low-VRAM options (VAE tiling / offload / mmap / VAE-on-CPU)
# - Mask support (region edit) via --mask
# - Aspect/size helper: auto-set Width/Height to match the input image aspect ratio (prevents square-crop)
#
# Still auto-detects flags from `sd-cli --help` (best-effort).

from __future__ import annotations

import os
import sys
import json
import time
import shlex
import subprocess
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QImage


APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SDCLI = os.path.join(APP_ROOT, ".qwen2512", "bin", "sd-cli.exe")

SETSAVE_PATH = os.path.join(APP_ROOT, "presets", "setsave", "qwen2511.hson")
DOWNLOAD_SCRIPT = os.path.join(APP_ROOT, "presets", "extra_env", "qwen2511_download.py")

MODELS_ROOT = os.path.join(APP_ROOT, "models", "qwen2511gguf")
UNET_DIR = os.path.join(MODELS_ROOT, "unet")
TEXTENC_DIR = os.path.join(MODELS_ROOT, "text_encoders")
VAE_DIR = os.path.join(MODELS_ROOT, "vae")

OUTPUT_DIR = os.path.join(APP_ROOT, "output", "qwen2511")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_jsonish(path: str) -> Dict:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = f.read().replace("\r\n", "\n")
            lines = []
            for line in s.split("\n"):
                if "//" in line:
                    line = line.split("//", 1)[0]
                lines.append(line)
            s = "\n".join(lines)
            s = re.sub(r",\s*([}\]])", r"\1", s)
            return json.loads(s)
        except Exception:
            return {}


def _write_json(path: str, data: Dict) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _list_files(folder: str, exts: Tuple[str, ...]) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for fn in os.listdir(folder):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and fn.lower().endswith(exts):
            out.append(p)
    out.sort(key=lambda x: os.path.basename(x).lower())
    return out


def _run_capture(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, shell=False)
        txt = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
        return p.returncode, txt
    except Exception as e:
        return 1, str(e)


def _invert_mask_to_temp(mask_path: str, out_dir: str) -> Optional[str]:
    if not mask_path or not os.path.isfile(mask_path):
        return None
    img = QImage(mask_path)
    if img.isNull():
        return None
    img = img.convertToFormat(QImage.Format_Grayscale8)
    for y in range(img.height()):
        for x in range(img.width()):
            v = img.pixelColor(x, y).red()
            inv = 255 - v
            img.setPixel(x, y, QtCore.qRgb(inv, inv, inv))
    _ensure_dir(out_dir)
    tmp = os.path.join(out_dir, f"_mask_inverted_{int(time.time())}.png")
    if img.save(tmp, "PNG"):
        return tmp
    return None


def _snap64(v: int) -> int:
    v = int(v)
    v = max(64, v)
    return int(round(v / 64.0) * 64)


@dataclass
class SdCliCaps:
    mode_flag: str = "--mode"
    mode_imggen: str = "img_gen"

    prompt: Optional[str] = "--prompt"
    negative: Optional[str] = "--negative-prompt"

    diffusion_model: Optional[str] = None
    llm: Optional[str] = None
    mmproj: Optional[str] = None
    vae: Optional[str] = None

    input_image: Optional[str] = None
    mask: Optional[str] = None
    output: Optional[str] = None

    steps: Optional[str] = None
    cfg: Optional[str] = None
    img_cfg: Optional[str] = None
    seed: Optional[str] = None
    width: Optional[str] = None
    height: Optional[str] = None
    strength: Optional[str] = None
    sampling_method: Optional[str] = None
    shift: Optional[str] = None

    # low-vram / performance flags
    vae_tiling: Optional[str] = None
    vae_tile_size: Optional[str] = None
    vae_tile_overlap: Optional[str] = None
    offload_to_cpu: Optional[str] = None
    mmap: Optional[str] = None
    vae_on_cpu: Optional[str] = None
    clip_on_cpu: Optional[str] = None
    diffusion_fa: Optional[str] = None


def detect_sdcli_caps(sdcli_path: str) -> SdCliCaps:
    caps = SdCliCaps()
    if not os.path.isfile(sdcli_path):
        return caps

    rc, help_txt = _run_capture([sdcli_path, "--help"])
    if rc != 0 and not help_txt.strip():
        return caps

    t = help_txt.lower()

    def has(flag: str) -> bool:
        return flag.lower() in t

    if "img_gen" in t:
        caps.mode_imggen = "img_gen"

    if has("--diffusion-model"):
        caps.diffusion_model = "--diffusion-model"
    elif has("--model"):
        caps.diffusion_model = "--model"
    elif has("--unet"):
        caps.diffusion_model = "--unet"

    if has("--llm"):
        caps.llm = "--llm"
    elif has("--text-encoder"):
        caps.llm = "--text-encoder"

    if has("--mmproj"):
        caps.mmproj = "--mmproj"
    elif has("--llm_vision"):
        caps.mmproj = "--llm_vision"
    elif has("--llm-vision"):
        caps.mmproj = "--llm-vision"

    if has("--vae"):
        caps.vae = "--vae"

    for cand in ["--init-img", "--init_image", "--image", "--input", "-i", "--ref", "-r", "--reference"]:
        if has(cand):
            caps.input_image = cand
            break

    if has("--mask"):
        caps.mask = "--mask"

    for cand in ["--output", "--out", "-o", "--output-path", "--output_path"]:
        if has(cand):
            caps.output = cand
            break

    if has("--negative-prompt"):
        caps.negative = "--negative-prompt"
    elif has("--neg-prompt"):
        caps.negative = "--neg-prompt"
    else:
        caps.negative = None

    if not has("--prompt"):
        caps.prompt = None

    if has("--steps"):
        caps.steps = "--steps"
    if has("--cfg-scale"):
        caps.cfg = "--cfg-scale"
    if has("--img-cfg-scale"):
        caps.img_cfg = "--img-cfg-scale"
    if has("--seed"):
        caps.seed = "--seed"
    if has("--width"):
        caps.width = "--width"
    if has("--height"):
        caps.height = "--height"
    if has("--strength"):
        caps.strength = "--strength"

    if has("--sampling-method"):
        caps.sampling_method = "--sampling-method"
    elif has("--sampler"):
        caps.sampling_method = "--sampler"

    if has("--flow-shift"):
        caps.shift = "--flow-shift"
    elif has("--shift"):
        caps.shift = "--shift"

    # Low-VRAM options
    if has("--vae-tiling"):
        caps.vae_tiling = "--vae-tiling"
    if has("--vae-tile-size"):
        caps.vae_tile_size = "--vae-tile-size"
    if has("--vae-tile-overlap"):
        caps.vae_tile_overlap = "--vae-tile-overlap"
    if has("--offload-to-cpu"):
        caps.offload_to_cpu = "--offload-to-cpu"
    if has("--mmap"):
        caps.mmap = "--mmap"
    if has("--vae-on-cpu"):
        caps.vae_on_cpu = "--vae-on-cpu"
    if has("--clip-on-cpu"):
        caps.clip_on_cpu = "--clip-on-cpu"
    if has("--diffusion-fa"):
        caps.diffusion_fa = "--diffusion-fa"

    return caps


def default_model_paths() -> Dict[str, Optional[str]]:
    unets = _list_files(UNET_DIR, (".gguf",))
    llms = _list_files(TEXTENC_DIR, (".gguf",))
    vaes = _list_files(VAE_DIR, (".safetensors", ".gguf", ".bin"))

    unet_pref = None
    for p in unets:
        if "q4_k_m" in os.path.basename(p).lower():
            unet_pref = p
            break
    if not unet_pref and unets:
        unet_pref = unets[0]

    llm_pref = None
    for p in llms:
        bn = os.path.basename(p).lower()
        if "mmproj" not in bn and "7b" in bn and "instruct" in bn:
            llm_pref = p
            break
    if not llm_pref:
        for p in llms:
            if "mmproj" not in os.path.basename(p).lower():
                llm_pref = p
                break

    mmproj_pref = None
    for p in llms:
        if "mmproj" in os.path.basename(p).lower():
            mmproj_pref = p
            break

    vae_pref = vaes[0] if vaes else None
    return {"unet": unet_pref, "llm": llm_pref, "mmproj": mmproj_pref, "vae": vae_pref}


def build_sdcli_cmd(
    sdcli_path: str,
    caps: SdCliCaps,
    init_img: str,
    mask_path: str,
    prompt: str,
    negative: str,
    unet_path: str,
    llm_path: str,
    mmproj_path: str,
    vae_path: str,
    steps: int,
    cfg: float,
    img_cfg: float,
    seed: int,
    width: int,
    height: int,
    strength: float,
    sampling_method: str,
    shift: float,
    out_file: str,
    # low vram
    use_vae_tiling: bool,
    vae_tile_size: str,
    vae_tile_overlap: float,
    use_offload: bool,
    use_mmap: bool,
    use_vae_on_cpu: bool,
    use_clip_on_cpu: bool,
    use_diffusion_fa: bool,
) -> List[str]:
    cmd: List[str] = [sdcli_path, caps.mode_flag, caps.mode_imggen]

    if caps.input_image and init_img:
        cmd += [caps.input_image, init_img]

    if mask_path and caps.mask:
        cmd += [caps.mask, mask_path]

    if caps.prompt:
        cmd += [caps.prompt, prompt]
    if caps.negative is not None:
        cmd += [caps.negative, negative or ""]

    if caps.diffusion_model:
        cmd += [caps.diffusion_model, unet_path]
    if caps.llm:
        cmd += [caps.llm, llm_path]
    if caps.mmproj:
        cmd += [caps.mmproj, mmproj_path]
    if caps.vae:
        cmd += [caps.vae, vae_path]

    if caps.steps:
        cmd += [caps.steps, str(int(steps))]
    if caps.cfg:
        cmd += [caps.cfg, str(float(cfg))]
    if caps.img_cfg:
        cmd += [caps.img_cfg, str(float(img_cfg))]
    if caps.seed:
        cmd += [caps.seed, str(int(seed))]
    if caps.width:
        cmd += [caps.width, str(int(width))]
    if caps.height:
        cmd += [caps.height, str(int(height))]
    if caps.strength:
        cmd += [caps.strength, str(float(strength))]
    if caps.sampling_method:
        cmd += [caps.sampling_method, sampling_method]
    if caps.shift:
        cmd += [caps.shift, str(float(shift))]

    # Low VRAM flags
    if use_mmap and caps.mmap:
        cmd += [caps.mmap]
    if use_offload and caps.offload_to_cpu:
        cmd += [caps.offload_to_cpu]
    if use_vae_on_cpu and caps.vae_on_cpu:
        cmd += [caps.vae_on_cpu]
    if use_clip_on_cpu and caps.clip_on_cpu:
        cmd += [caps.clip_on_cpu]
    if use_diffusion_fa and caps.diffusion_fa:
        cmd += [caps.diffusion_fa]

    if use_vae_tiling and caps.vae_tiling:
        cmd += [caps.vae_tiling]
        if caps.vae_tile_size and vae_tile_size.strip():
            cmd += [caps.vae_tile_size, vae_tile_size.strip()]
        if caps.vae_tile_overlap:
            cmd += [caps.vae_tile_overlap, str(float(vae_tile_overlap))]

    if caps.output:
        cmd += [caps.output, out_file]

    return cmd


class Qwen2511Pane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Qwen2511Pane")
        self.caps: SdCliCaps = detect_sdcli_caps(DEFAULT_SDCLI)
        self._settings = _read_jsonish(SETSAVE_PATH)

        self._build_ui()
        self._load_defaults()
        self._apply_settings_to_ui()

        self._append_log(self._caps_summary())
        self._append_log(
            "\nWhy you got a different woman + blurry output:\n"
            "- Without a mask, the model is free to rewrite the entire image.\n"
            "- If Width/Height are square, sd-cli crops your 16:9 image to a square (it logged a crop).\n"
            "\nBest practice for 'change dress color':\n"
            "1) Provide a mask for the dress area,\n"
            "2) Strength ~0.25–0.45,\n"
            "3) Keep aspect ratio (use the Auto size helper),\n"
            "4) Consider 28–40 steps and Img-CFG 6–9 to preserve the original."
        )

    def _append_log(self, s: str):
        self.log.appendPlainText(s.rstrip())

    def _caps_summary(self) -> str:
        c = self.caps
        return (
            "Detected sd-cli capabilities (best-effort):\n"
            f"  mode: {c.mode_imggen}\n"
            f"  input_image flag: {c.input_image}\n"
            f"  mask flag: {c.mask}\n"
            f"  diffusion_model flag: {c.diffusion_model}\n"
            f"  llm/text-encoder flag: {c.llm}\n"
            f"  mmproj flag: {c.mmproj}\n"
            f"  vae flag: {c.vae}\n"
            f"  output flag: {c.output}\n"
            f"  sampling flag: {c.sampling_method}\n"
            f"  supports img-cfg: {bool(c.img_cfg)}\n"
            f"  supports vae-tiling: {bool(c.vae_tiling)}\n"
            f"  supports offload-to-cpu: {bool(c.offload_to_cpu)}\n"
            f"  supports vae-on-cpu: {bool(c.vae_on_cpu)}\n"
            f"  supports mmap: {bool(c.mmap)}\n"
        )

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QtWidgets.QLabel("Qwen Image Edit 2511 (GGUF)")
        f = title.font()
        f.setPointSize(max(10, f.pointSize() + 3))
        f.setBold(True)
        title.setFont(f)
        layout.addWidget(title)

        # Paths
        g_paths = QtWidgets.QGroupBox("Paths")
        gl = QtWidgets.QFormLayout(g_paths)
        gl.setLabelAlignment(QtCore.Qt.AlignRight)

        self.ed_sdcli = QtWidgets.QLineEdit(DEFAULT_SDCLI)
        btn_sdcli = QtWidgets.QToolButton(); btn_sdcli.setText("…"); btn_sdcli.clicked.connect(self._pick_sdcli)
        h_sd = QtWidgets.QHBoxLayout(); h_sd.addWidget(self.ed_sdcli, 1); h_sd.addWidget(btn_sdcli)
        gl.addRow("sd-cli.exe", h_sd)

        self.ed_initimg = QtWidgets.QLineEdit()
        self.ed_initimg.setPlaceholderText("Select an input image…")
        btn_img = QtWidgets.QToolButton(); btn_img.setText("…"); btn_img.clicked.connect(self._pick_image)
        h_img = QtWidgets.QHBoxLayout(); h_img.addWidget(self.ed_initimg, 1); h_img.addWidget(btn_img)
        gl.addRow("Input image", h_img)

        self.lbl_imginfo = QtWidgets.QLabel("")
        self.lbl_imginfo.setStyleSheet("color: #888;")
        gl.addRow("", self.lbl_imginfo)

        self.ed_mask = QtWidgets.QLineEdit()
        self.ed_mask.setPlaceholderText("Optional: mask image (white/black) …")
        btn_mask = QtWidgets.QToolButton(); btn_mask.setText("…"); btn_mask.clicked.connect(self._pick_mask)
        h_mask = QtWidgets.QHBoxLayout(); h_mask.addWidget(self.ed_mask, 1); h_mask.addWidget(btn_mask)
        gl.addRow("Mask", h_mask)

        self.chk_invert_mask = QtWidgets.QCheckBox("Invert mask")
        self.chk_invert_mask.setToolTip("If your mask behaves backwards, enable this. It writes a temporary inverted mask PNG.")
        gl.addRow("", self.chk_invert_mask)

        layout.addWidget(g_paths)

        # Prompt
        g_prompt = QtWidgets.QGroupBox("Prompt")
        pl = QtWidgets.QVBoxLayout(g_prompt)
        self.ed_prompt = QtWidgets.QPlainTextEdit()
        self.ed_prompt.setPlaceholderText("Describe the edit… (example: keep everything the same, only change dress color to red)")
        self.ed_neg = QtWidgets.QLineEdit(); self.ed_neg.setPlaceholderText("Negative prompt (optional)")
        pl.addWidget(self.ed_prompt, 1)
        pl.addWidget(self.ed_neg)
        layout.addWidget(g_prompt, 2)

        # Models
        g_models = QtWidgets.QGroupBox("Models")
        ml = QtWidgets.QFormLayout(g_models)
        ml.setLabelAlignment(QtCore.Qt.AlignRight)
        self.cb_unet = QtWidgets.QComboBox()
        self.cb_llm = QtWidgets.QComboBox()
        self.cb_mmproj = QtWidgets.QComboBox()
        self.cb_vae = QtWidgets.QComboBox()
        ml.addRow("UNet", self.cb_unet)
        ml.addRow("Text encoder", self.cb_llm)
        ml.addRow("mmproj", self.cb_mmproj)
        ml.addRow("VAE", self.cb_vae)
        layout.addWidget(g_models)

        # Settings
        g_set = QtWidgets.QGroupBox("Settings")
        sl = QtWidgets.QGridLayout(g_set)

        self.sp_steps = QtWidgets.QSpinBox(); self.sp_steps.setRange(1, 200); self.sp_steps.setValue(28)
        self.sp_cfg = QtWidgets.QDoubleSpinBox(); self.sp_cfg.setRange(0.0, 30.0); self.sp_cfg.setDecimals(2); self.sp_cfg.setSingleStep(0.25); self.sp_cfg.setValue(4.5)

        self.sp_imgcfg = QtWidgets.QDoubleSpinBox()
        self.sp_imgcfg.setRange(0.0, 30.0)
        self.sp_imgcfg.setDecimals(2)
        self.sp_imgcfg.setSingleStep(0.25)
        self.sp_imgcfg.setValue(7.0)
        self.sp_imgcfg.setToolTip("Image guidance scale (if supported). Helps preserve the init image structure for edits.")

        self.sp_seed = QtWidgets.QSpinBox(); self.sp_seed.setRange(-1, 2**31 - 1); self.sp_seed.setValue(-1)

        self.sp_w = QtWidgets.QSpinBox(); self.sp_w.setRange(64, 4096); self.sp_w.setSingleStep(64); self.sp_w.setValue(1024)
        self.sp_h = QtWidgets.QSpinBox(); self.sp_h.setRange(64, 4096); self.sp_h.setSingleStep(64); self.sp_h.setValue(576)

        self.sp_strength = QtWidgets.QDoubleSpinBox(); self.sp_strength.setRange(0.0, 1.0); self.sp_strength.setDecimals(2); self.sp_strength.setSingleStep(0.05); self.sp_strength.setValue(0.35)

        self.cb_sampling = QtWidgets.QComboBox()
        self.cb_sampling.addItems(["euler","euler_a","heun","dpm2","dpm++2s_a","dpm++2m","dpm++2mv2","ipndm","ipndm_v","lcm","ddim_trailing","tcd"])
        self.cb_sampling.setCurrentText("euler_a")

        # Aspect helper
        self.chk_auto_aspect = QtWidgets.QCheckBox("Auto size from input aspect")
        self.chk_auto_aspect.setToolTip("Prevents square-crop. Uses the input image aspect ratio and snaps to multiples of 64.")
        self.cb_base = QtWidgets.QComboBox()
        self.cb_base.addItems(["512", "768", "1024"])
        self.cb_base.setCurrentText("1024")
        self.btn_apply_aspect = QtWidgets.QPushButton("Apply")
        self.btn_apply_aspect.clicked.connect(self._apply_aspect_now)
        aspect_row = QtWidgets.QHBoxLayout()
        aspect_row.addWidget(self.chk_auto_aspect)
        aspect_row.addWidget(QtWidgets.QLabel("Base:"))
        aspect_row.addWidget(self.cb_base)
        aspect_row.addWidget(self.btn_apply_aspect)
        aspect_row.addStretch(1)

        self.sp_shift = QtWidgets.QDoubleSpinBox(); self.sp_shift.setRange(0.0, 30.0); self.sp_shift.setDecimals(2); self.sp_shift.setSingleStep(0.25); self.sp_shift.setValue(12.5)

        # Low VRAM options
        self.chk_vae_tiling = QtWidgets.QCheckBox("VAE tiling (low VRAM)")
        self.ed_vae_tile_size = QtWidgets.QLineEdit("32x32")
        self.sp_vae_tile_overlap = QtWidgets.QDoubleSpinBox()
        self.sp_vae_tile_overlap.setRange(0.0, 0.95); self.sp_vae_tile_overlap.setDecimals(2); self.sp_vae_tile_overlap.setSingleStep(0.05); self.sp_vae_tile_overlap.setValue(0.50)

        self.chk_offload = QtWidgets.QCheckBox("Offload weights to CPU (very slow)")
        self.chk_mmap = QtWidgets.QCheckBox("mmap")
        self.chk_vae_on_cpu = QtWidgets.QCheckBox("VAE on CPU (slow)")
        self.chk_clip_on_cpu = QtWidgets.QCheckBox("CLIP on CPU (if available)")
        self.chk_diffusion_fa = QtWidgets.QCheckBox("Diffusion flash-attn (if available)")

        r = 0
        sl.addWidget(QtWidgets.QLabel("Steps"), r, 0); sl.addWidget(self.sp_steps, r, 1)
        sl.addWidget(QtWidgets.QLabel("CFG"), r, 2); sl.addWidget(self.sp_cfg, r, 3)
        r += 1
        sl.addWidget(QtWidgets.QLabel("Img-CFG"), r, 0); sl.addWidget(self.sp_imgcfg, r, 1)
        sl.addWidget(QtWidgets.QLabel("Seed"), r, 2); sl.addWidget(self.sp_seed, r, 3)
        r += 1
        sl.addWidget(QtWidgets.QLabel("Sampling"), r, 0); sl.addWidget(self.cb_sampling, r, 1)
        sl.addWidget(QtWidgets.QLabel("Strength"), r, 2); sl.addWidget(self.sp_strength, r, 3)
        r += 1
        sl.addWidget(QtWidgets.QLabel("Width"), r, 0); sl.addWidget(self.sp_w, r, 1)
        sl.addWidget(QtWidgets.QLabel("Height"), r, 2); sl.addWidget(self.sp_h, r, 3)
        r += 1
        sl.addWidget(QtWidgets.QLabel("Flow shift"), r, 0); sl.addWidget(self.sp_shift, r, 1)
        sl.addLayout(aspect_row, r, 2, 1, 2)

        r += 1
        sl.addWidget(self.chk_vae_tiling, r, 0, 1, 2)
        sl.addWidget(QtWidgets.QLabel("Tile size"), r, 2); sl.addWidget(self.ed_vae_tile_size, r, 3)

        r += 1
        sl.addWidget(self.chk_mmap, r, 0, 1, 2)
        sl.addWidget(QtWidgets.QLabel("Tile overlap"), r, 2); sl.addWidget(self.sp_vae_tile_overlap, r, 3)

        r += 1
        sl.addWidget(self.chk_offload, r, 0, 1, 2)
        sl.addWidget(self.chk_vae_on_cpu, r, 2, 1, 2)

        r += 1
        sl.addWidget(self.chk_clip_on_cpu, r, 0, 1, 2)
        sl.addWidget(self.chk_diffusion_fa, r, 2, 1, 2)

        layout.addWidget(g_set)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_download = QtWidgets.QPushButton("Download models")
        self.btn_reload = QtWidgets.QPushButton("Reload models list")
        self.btn_save = QtWidgets.QPushButton("Save settings")
        self.btn_run = QtWidgets.QPushButton("Run sd-cli")

        self.btn_download.clicked.connect(self._run_downloader)
        self.btn_reload.clicked.connect(self._reload_model_lists)
        self.btn_save.clicked.connect(self._save_settings)
        self.btn_run.clicked.connect(self._run_sdcli)

        btn_row.addWidget(self.btn_download)
        btn_row.addWidget(self.btn_reload)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_run)
        layout.addLayout(btn_row)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)
        layout.addWidget(self.log, 2)

    def _update_imginfo(self, path: str):
        if not path or not os.path.isfile(path):
            self.lbl_imginfo.setText("")
            return
        img = QImage(path)
        if img.isNull():
            self.lbl_imginfo.setText("")
            return
        self.lbl_imginfo.setText(f"Input size: {img.width()}x{img.height()}  (Tip: keep Width/Height similar aspect to avoid crop)")

    def _apply_aspect_now(self):
        path = self.ed_initimg.text().strip()
        if not path or not os.path.isfile(path):
            return
        img = QImage(path)
        if img.isNull():
            return

        w0, h0 = img.width(), img.height()
        base = int(self.cb_base.currentText())
        if w0 >= h0:
            w = base
            h = int(round(base * (h0 / float(w0))))
        else:
            h = base
            w = int(round(base * (w0 / float(h0))))

        self.sp_w.setValue(_snap64(w))
        self.sp_h.setValue(_snap64(h))

    def _pick_sdcli(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select sd-cli.exe", APP_ROOT, "Executable (*.exe)")
        if fn:
            self.ed_sdcli.setText(fn)
            self.caps = detect_sdcli_caps(fn)
            self._append_log("\n" + self._caps_summary())

    def _pick_image(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select input image", APP_ROOT, "Images (*.png *.jpg *.jpeg *.webp)")
        if fn:
            self.ed_initimg.setText(fn)
            self._update_imginfo(fn)
            if self.chk_auto_aspect.isChecked():
                self._apply_aspect_now()

    def _pick_mask(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select mask image", APP_ROOT, "Images (*.png *.jpg *.jpeg *.webp)")
        if fn:
            self.ed_mask.setText(fn)

    def _reload_model_lists(self):
        self.cb_unet.clear(); self.cb_llm.clear(); self.cb_mmproj.clear(); self.cb_vae.clear()

        for p in _list_files(UNET_DIR, (".gguf",)):
            self.cb_unet.addItem(os.path.basename(p), p)

        llms = _list_files(TEXTENC_DIR, (".gguf",))
        for p in llms:
            bn = os.path.basename(p).lower()
            if "mmproj" in bn:
                self.cb_mmproj.addItem(os.path.basename(p), p)
            else:
                self.cb_llm.addItem(os.path.basename(p), p)

        for p in _list_files(VAE_DIR, (".safetensors", ".gguf", ".bin")):
            self.cb_vae.addItem(os.path.basename(p), p)

        if self.cb_mmproj.count() == 0:
            for p in llms:
                self.cb_mmproj.addItem(os.path.basename(p), p)

    def _load_defaults(self):
        self._reload_model_lists()
        d = default_model_paths()
        self._settings.setdefault("sdcli_path", DEFAULT_SDCLI)
        self._settings.setdefault("last_init_img", "")
        self._settings.setdefault("last_mask_img", "")
        self._settings.setdefault("invert_mask", False)
        self._settings.setdefault("prompt", "")
        self._settings.setdefault("negative_prompt", "")
        self._settings.setdefault("unet_path", d["unet"] or "")
        self._settings.setdefault("llm_path", d["llm"] or "")
        self._settings.setdefault("mmproj_path", d["mmproj"] or "")
        self._settings.setdefault("vae_path", d["vae"] or "")
        self._settings.setdefault("steps", 28)
        self._settings.setdefault("cfg", 4.5)
        self._settings.setdefault("img_cfg", 7.0)
        self._settings.setdefault("seed", -1)
        self._settings.setdefault("width", 1024)
        self._settings.setdefault("height", 576)
        self._settings.setdefault("strength", 0.35)
        self._settings.setdefault("sampling_method", "euler_a")
        self._settings.setdefault("shift", 12.5)
        self._settings.setdefault("auto_aspect", True)
        self._settings.setdefault("auto_aspect_base", 1024)

        # low-vram defaults
        self._settings.setdefault("vae_tiling", True)
        self._settings.setdefault("vae_tile_size", "32x32")
        self._settings.setdefault("vae_tile_overlap", 0.50)
        self._settings.setdefault("offload_to_cpu", False)
        self._settings.setdefault("mmap", True)
        self._settings.setdefault("vae_on_cpu", False)
        self._settings.setdefault("clip_on_cpu", False)
        self._settings.setdefault("diffusion_fa", False)

    def _apply_settings_to_ui(self):
        s = self._settings
        self.ed_sdcli.setText(s.get("sdcli_path", DEFAULT_SDCLI))
        self.ed_initimg.setText(s.get("last_init_img", ""))
        self._update_imginfo(self.ed_initimg.text().strip())
        self.ed_mask.setText(s.get("last_mask_img", ""))
        self.chk_invert_mask.setChecked(bool(s.get("invert_mask", False)))

        self.ed_prompt.setPlainText(s.get("prompt", ""))
        self.ed_neg.setText(s.get("negative_prompt", ""))

        self.sp_steps.setValue(int(s.get("steps", 28)))
        self.sp_cfg.setValue(float(s.get("cfg", 4.5)))
        self.sp_imgcfg.setValue(float(s.get("img_cfg", 7.0)))
        self.sp_seed.setValue(int(s.get("seed", -1)))
        self.sp_w.setValue(int(s.get("width", 1024)))
        self.sp_h.setValue(int(s.get("height", 576)))
        self.sp_strength.setValue(float(s.get("strength", 0.35)))
        self.cb_sampling.setCurrentText(str(s.get("sampling_method", "euler_a")))
        self.sp_shift.setValue(float(s.get("shift", 12.5)))

        self.chk_auto_aspect.setChecked(bool(s.get("auto_aspect", True)))
        base = int(s.get("auto_aspect_base", 1024))
        self.cb_base.setCurrentText(str(base))

        self.chk_vae_tiling.setChecked(bool(s.get("vae_tiling", True)))
        self.ed_vae_tile_size.setText(str(s.get("vae_tile_size", "32x32")))
        self.sp_vae_tile_overlap.setValue(float(s.get("vae_tile_overlap", 0.50)))
        self.chk_offload.setChecked(bool(s.get("offload_to_cpu", False)))
        self.chk_mmap.setChecked(bool(s.get("mmap", True)))
        self.chk_vae_on_cpu.setChecked(bool(s.get("vae_on_cpu", False)))
        self.chk_clip_on_cpu.setChecked(bool(s.get("clip_on_cpu", False)))
        self.chk_diffusion_fa.setChecked(bool(s.get("diffusion_fa", False)))

        def set_combo_to_path(cb: QtWidgets.QComboBox, path: str):
            if not path:
                return
            for i in range(cb.count()):
                if cb.itemData(i) == path:
                    cb.setCurrentIndex(i)
                    return

        set_combo_to_path(self.cb_unet, s.get("unet_path", ""))
        set_combo_to_path(self.cb_llm, s.get("llm_path", ""))
        set_combo_to_path(self.cb_mmproj, s.get("mmproj_path", ""))
        set_combo_to_path(self.cb_vae, s.get("vae_path", ""))

    def _save_settings(self):
        s = {
            "sdcli_path": self.ed_sdcli.text().strip(),
            "last_init_img": self.ed_initimg.text().strip(),
            "last_mask_img": self.ed_mask.text().strip(),
            "invert_mask": bool(self.chk_invert_mask.isChecked()),
            "prompt": self.ed_prompt.toPlainText().strip(),
            "negative_prompt": self.ed_neg.text().strip(),
            "unet_path": self.cb_unet.currentData(),
            "llm_path": self.cb_llm.currentData(),
            "mmproj_path": self.cb_mmproj.currentData(),
            "vae_path": self.cb_vae.currentData(),
            "steps": int(self.sp_steps.value()),
            "cfg": float(self.sp_cfg.value()),
            "img_cfg": float(self.sp_imgcfg.value()),
            "seed": int(self.sp_seed.value()),
            "width": int(self.sp_w.value()),
            "height": int(self.sp_h.value()),
            "strength": float(self.sp_strength.value()),
            "sampling_method": str(self.cb_sampling.currentText()),
            "shift": float(self.sp_shift.value()),
            "auto_aspect": bool(self.chk_auto_aspect.isChecked()),
            "auto_aspect_base": int(self.cb_base.currentText()),
            "vae_tiling": bool(self.chk_vae_tiling.isChecked()),
            "vae_tile_size": self.ed_vae_tile_size.text().strip(),
            "vae_tile_overlap": float(self.sp_vae_tile_overlap.value()),
            "offload_to_cpu": bool(self.chk_offload.isChecked()),
            "mmap": bool(self.chk_mmap.isChecked()),
            "vae_on_cpu": bool(self.chk_vae_on_cpu.isChecked()),
            "clip_on_cpu": bool(self.chk_clip_on_cpu.isChecked()),
            "diffusion_fa": bool(self.chk_diffusion_fa.isChecked()),
        }
        self._settings.update(s)
        _write_json(SETSAVE_PATH, self._settings)
        self._append_log(f"\nSaved settings -> {SETSAVE_PATH}")

    def _run_downloader(self):
        py = os.path.join(APP_ROOT, ".qwen2512", "venv", "Scripts", "python.exe")
        if not os.path.isfile(py):
            py = sys.executable

        if not os.path.isfile(DOWNLOAD_SCRIPT):
            self._append_log(f"\nDownloader script not found: {DOWNLOAD_SCRIPT}")
            return

        cmd = [py, DOWNLOAD_SCRIPT]
        self._append_log("\nRunning downloader:\n" + " ".join(shlex.quote(x) for x in cmd))

        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out_lines = []
            while True:
                line = p.stdout.readline() if p.stdout else ""
                if not line and p.poll() is not None:
                    break
                if line:
                    out_lines.append(line.rstrip())
                    QtWidgets.QApplication.processEvents()
            rc = p.wait()
            self._append_log("\n".join(out_lines))
            self._append_log(f"\nDownloader finished (code {rc})")
            self._reload_model_lists()
        except Exception as e:
            self._append_log(f"\nDownloader error: {e}")

    def _run_sdcli(self):
        sdcli = self.ed_sdcli.text().strip()
        init_img = self.ed_initimg.text().strip()
        mask_img = self.ed_mask.text().strip()

        if not os.path.isfile(sdcli):
            self._append_log(f"\nERROR: sd-cli.exe not found: {sdcli}")
            return

        self.caps = detect_sdcli_caps(sdcli)

        if not init_img or not os.path.isfile(init_img):
            self._append_log("\nERROR: Input image missing. Pick an input image first.")
            return

        if mask_img and not os.path.isfile(mask_img):
            self._append_log("\nERROR: Mask path is set but file not found.")
            return

        if mask_img and not self.caps.mask:
            self._append_log("\nWARNING: Your sd-cli build does not advertise --mask in --help. Mask will be ignored.")
            mask_img = ""

        unet = self.cb_unet.currentData()
        llm = self.cb_llm.currentData()
        mmproj = self.cb_mmproj.currentData()
        vae = self.cb_vae.currentData()

        missing = [name for name, p in [("UNet", unet), ("Text encoder", llm), ("mmproj", mmproj), ("VAE", vae)] if not p or not os.path.isfile(p)]
        if missing:
            self._append_log("\nERROR: Missing model files: " + ", ".join(missing))
            return

        _ensure_dir(OUTPUT_DIR)
        ts = int(time.time())
        out_file = os.path.join(OUTPUT_DIR, f"qwen_image_edit_2511_{ts}.png")

        if mask_img and self.chk_invert_mask.isChecked():
            inv_tmp = _invert_mask_to_temp(mask_img, OUTPUT_DIR)
            if inv_tmp:
                self._append_log(f"\nInvert mask: wrote temp mask -> {inv_tmp}")
                mask_img = inv_tmp
            else:
                self._append_log("\nInvert mask: failed to invert mask. Using original mask.")

        cmd = build_sdcli_cmd(
            sdcli_path=sdcli,
            caps=self.caps,
            init_img=init_img,
            mask_path=mask_img,
            prompt=self.ed_prompt.toPlainText().strip(),
            negative=self.ed_neg.text().strip(),
            unet_path=unet,
            llm_path=llm,
            mmproj_path=mmproj,
            vae_path=vae,
            steps=int(self.sp_steps.value()),
            cfg=float(self.sp_cfg.value()),
            img_cfg=float(self.sp_imgcfg.value()),
            seed=int(self.sp_seed.value()),
            width=int(self.sp_w.value()),
            height=int(self.sp_h.value()),
            strength=float(self.sp_strength.value()),
            sampling_method=str(self.cb_sampling.currentText()),
            shift=float(self.sp_shift.value()),
            out_file=out_file,
            use_vae_tiling=bool(self.chk_vae_tiling.isChecked()),
            vae_tile_size=self.ed_vae_tile_size.text().strip(),
            vae_tile_overlap=float(self.sp_vae_tile_overlap.value()),
            use_offload=bool(self.chk_offload.isChecked()),
            use_mmap=bool(self.chk_mmap.isChecked()),
            use_vae_on_cpu=bool(self.chk_vae_on_cpu.isChecked()),
            use_clip_on_cpu=bool(self.chk_clip_on_cpu.isChecked()),
            use_diffusion_fa=bool(self.chk_diffusion_fa.isChecked()),
        )

        self._append_log("\nRunning sd-cli:\n" + " ".join(shlex.quote(x) for x in cmd))

        try:
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            out_lines = []
            while True:
                line = p.stdout.readline() if p.stdout else ""
                if not line and p.poll() is not None:
                    break
                if line:
                    out_lines.append(line.rstrip())
                    QtWidgets.QApplication.processEvents()
            rc = p.wait()
            self._append_log("\n".join(out_lines))
            self._append_log(f"\nsd-cli finished (code {rc})")

            if os.path.isfile(out_file):
                self._append_log(f"Output written: {out_file}")
            else:
                self._append_log("Output file not found at expected path. If it crashed, check logs above.")
        except Exception as e:
            self._append_log(f"\nsd-cli error: {e}")


def _standalone_main():
    app = QtWidgets.QApplication(sys.argv)
    w = Qwen2511Pane()
    w.resize(1020, 960)
    w.setWindowTitle("Qwen Image Edit 2511 – Standalone Test")
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _standalone_main()
