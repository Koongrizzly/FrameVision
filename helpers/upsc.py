
# helpers/upsc.py — Upscale tab with video pipeline and model-dir fixes
from __future__ import annotations
import os
import sys
import shutil
import subprocess
import json, tempfile, time
import re
from pathlib import Path
from typing import List, Tuple, Optional
from helpers.mediainfo import refresh_info_now

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QTextCursor, QPixmap, QIcon, QPainter, QPainterPath

try:
    from helpers.framevision_app import ROOT, OUT_VIDEOS, OUT_SHOTS
except Exception:
    ROOT = Path(__file__).resolve().parent.parent
    OUT_VIDEOS = ROOT / "output" / "video"
    OUT_SHOTS = ROOT / "output" / "photo"



PRESETS_BIN = ROOT / "presets" / "bin"
BIN_DIR = ROOT / "bin"
MODELS_DIR = ROOT / "models"
REALSR_DIR = MODELS_DIR / "realesrgan"
WAIFU2X_DIR = MODELS_DIR / "waifu2x"
SRMD_DIR = MODELS_DIR / "srmd-ncnn-vulkan-master"
REALSR_NCNN_DIR = MODELS_DIR / "realsr-ncnn-vulkan-20220728-windows"

GFPGAN_DIR = MODELS_DIR / "gfpgan"
GFPGAN_ENV_DIR = GFPGAN_DIR / ".GFPGAN"
GFPGAN_MODEL_V14 = GFPGAN_DIR / "GFPGANv1.4.pth"
GFPGAN_MODEL_V13 = GFPGAN_DIR / "GFPGANv1.3.pth"

# SeedVR2 (video upscaler) paths
SEEDVR2_MODELS_DIR = MODELS_DIR / "SEEDVR2"
SEEDVR2_CLI = ROOT / "presets" / "extra_env" / "seedvr2_src" / "ComfyUI-SeedVR2_VideoUpscaler" / "inference_cli.py"
if os.name == "nt":
    SEEDVR2_ENV_PY = ROOT / "environments" / ".seedvr2" / "Scripts" / "python.exe"
else:
    SEEDVR2_ENV_PY = ROOT / "environments" / ".seedvr2" / "bin" / "python"
def _seedvr2_runner_path() -> Optional[Path]:
    """Best-effort locate helpers/seedvr2_runner.py across dev + frozen layouts.

    Why: when packaged (e.g. PyInstaller), this module may live under _internal/ or
    a temp extraction folder, so Path(__file__).parent is not the on-disk helpers/.
    """
    candidates: List[Path] = []

    try:
        # 1) Next to this file (dev / non-frozen)
        candidates.append(Path(__file__).resolve().with_name("seedvr2_runner.py"))
    except Exception:
        pass

    try:
        # 2) Portable layout: <ROOT>/helpers/seedvr2_runner.py
        candidates.append((ROOT / "helpers" / "seedvr2_runner.py").resolve())
    except Exception:
        pass

    try:
        # 3) Frozen EXE layout: <exe_dir>/helpers/seedvr2_runner.py
        if getattr(sys, "frozen", False) and getattr(sys, "executable", None):
            candidates.append((Path(sys.executable).resolve().parent / "helpers" / "seedvr2_runner.py"))
    except Exception:
        pass

    try:
        # 4) PyInstaller extraction dir (if used)
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(Path(meipass) / "helpers" / "seedvr2_runner.py")
    except Exception:
        pass

    try:
        # 5) Current working dir fallback
        candidates.append(Path.cwd() / "helpers" / "seedvr2_runner.py")
    except Exception:
        pass

    for p in candidates:
        try:
            if p and p.exists():
                return p
        except Exception:
            pass
    return None


def scan_seedvr2_gguf() -> List[str]:
    """Return GGUF filenames found under models/SEEDVR2 (recursively)."""
    names: List[str] = []
    try:
        if SEEDVR2_MODELS_DIR.exists():
            for p in sorted(SEEDVR2_MODELS_DIR.rglob("*.gguf")):
                # The CLI typically expects just the filename (argparse choices)
                names.append(p.name)
    except Exception:
        pass
    # Reasonable defaults if none found yet
    return names or ["seedvr2_ema_3b-Q4_K_M.gguf"]

def _gfpgan_python() -> Optional[Path]:
    try:
        if os.name == "nt":
            p = GFPGAN_ENV_DIR / "Scripts" / "python.exe"
        else:
            p = GFPGAN_ENV_DIR / "bin" / "python"
        return p if p.exists() else None
    except Exception:
        return None

_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".gif"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _exists(p: str | Path) -> bool:
    try:
        return Path(p).exists()
    except Exception:
        return False


def _which_ffmpeg(name: str) -> str:
    exe = name + (".exe" if os.name == "nt" else "")
    for d in (PRESETS_BIN, BIN_DIR, ROOT):
        cand = d / exe
        if cand.exists():
            return str(cand)
    return name


FFMPEG = _which_ffmpeg("ffmpeg")
FFPROBE = _which_ffmpeg("ffprobe")

# UI switches
SHOW_UPSCALE_LOG = False  # hide the big log window; logging remains internal


def detect_engines() -> List[Tuple[str, str]]:
    engines: List[Tuple[str, str]] = []
    realsr = REALSR_DIR / ("realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan")
    if realsr.exists():
        engines.append(("Real-ESRGAN (ncnn)", str(realsr)))
    engines.append(("UltraSharp (ncnn)", str(realsr)))
    engines.append(("SRMD (ncnn via RealESRGAN)", str(realsr)))
    waifu = WAIFU2X_DIR / ("waifu2x-ncnn-vulkan.exe" if os.name == "nt" else "waifu2x-ncnn-vulkan")
    if waifu.exists():
        engines.append(("Waifu2x (ncnn)", str(waifu)))
    return engines



def scan_realsr_models() -> List[str]:
    names: set[str] = set()
    if REALSR_DIR.exists():
        for ext in ("*.bin", "*.param"):
            for p in sorted(REALSR_DIR.glob(ext)):
                stem = p.stem
                low = stem.lower()
                if low.startswith("srmd") or low.startswith("srmdnf") or low.startswith("srdmnf") or low.startswith("4x-ultrasharp"):
                    continue
                names.add(stem)
    if names:
        return sorted(names)
    # Fallback suggestions if no models detected (first-run scenarios)
    return [
        "realesrgan-x4plus",
        "realesrgan-x4plus-anime",
        "realesr-animevideov3-x4",
        "realesr-general-x4v3",
        "realesr-general-wdn-x4v3",
    ]




def scan_ultrasharp_models() -> List[str]:
    names: set[str] = set()
    try:
        if REALSR_DIR.exists():
            for ext in ("*.bin","*.param"):
                for p in sorted(REALSR_DIR.glob("4x-UltraSharp-*"+ext.split("*")[-1])):
                    names.add(p.stem)
    except Exception:
        pass
    return sorted(names) or ["4x-UltraSharp-fp16","4x-UltraSharp-fp32"]

def scan_srmd_realesrgan_models() -> List[str]:
    names: set[str] = set()
    try:
        if REALSR_DIR.exists():
            for ext in ("*.bin","*.param"):
                for p in sorted(REALSR_DIR.glob(ext)):
                    s = p.stem.lower()
                    if s.startswith("srmd") or s.startswith("srmdnf") or s.startswith("srdmnf"):
                        names.add(p.stem)
    except Exception:
        pass
    return sorted(names) or ["srmd_x2","srmd_x3","srmd_x4","srmdnf_x2","srmdnf_x3","srmdnf_x4"]

def scan_waifu2x_models() -> List[str]:
    names: List[str] = []
    if WAIFU2X_DIR.exists():
        for p in sorted(WAIFU2X_DIR.glob("models-*")):
            names.append(p.name)
    return names or ["models-cunet", "models-upconv_7_photo", "models-upconv_7_anime_style_art_rgb"]
def scan_srmd_models() -> List[str]:
    names: List[str] = []
    try:
        if SRMD_DIR.exists():
            for p in sorted(SRMD_DIR.glob("models-*")):
                if p.is_dir():
                    names.append(p.name)
    except Exception:
        pass
    return names or ["models-DF2K", "models-DF2K_JPEG"]

def scan_realsr_ncnn_models() -> List[str]:
    names: List[str] = []
    try:
        if REALSR_NCNN_DIR.exists():
            for p in sorted(REALSR_NCNN_DIR.glob("*.param")):
                names.append(p.stem)
    except Exception:
        pass
    return names or ["realsr-x2", "realsr-x4"]
    names: List[str] = []
    if WAIFU2X_DIR.exists():
        for p in sorted(WAIFU2X_DIR.glob("models-*")):
            names.append(p.name)
    return names or ["models-cunet", "models-upconv_7_photo", "models-upconv_7_anime_style_art_rgb"]


def _parse_fps(src: Path) -> Optional[str]:
    """Return best-effort source FPS (as an ffmpeg-friendly value).

    Why this exists:
      - r_frame_rate can be misleading (often 30/1) for VFR or certain encodes.
      - avg_frame_rate is usually closer to real playback timing.
      - As a fallback, derive FPS from nb_frames / duration when available.
    """
    def _parse_ratio(s: str) -> Optional[tuple[int,int]]:
        try:
            s = (s or "").strip()
            if not s or s in ("0/0", "0", "N/A"):
                return None
            if "/" in s:
                a, b = s.split("/", 1)
                a_i = int(float(a))
                b_i = int(float(b))
                if b_i == 0:
                    return None
                return (a_i, b_i)
            # decimal
            f = float(s)
            if f <= 0:
                return None
            # represent as ratio with 1e6 precision
            den = 1_000_000
            num = int(round(f * den))
            if num <= 0:
                return None
            return (num, den)
        except Exception:
            return None

    def _ratio_to_str(r: tuple[int,int]) -> str:
        n, d = r
        if d == 1:
            return str(n)
        return f"{n}/{d}"

    def _fps_from_ratio(r: tuple[int,int]) -> float:
        try:
            n, d = r
            return float(n) / float(d) if d else 0.0
        except Exception:
            return 0.0

    try:
        # Pull multiple fields at once (JSON) so we can pick the best.
        out = subprocess.check_output(
            [FFPROBE, "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames,duration,time_base",
             "-show_entries", "format=duration",
             "-of", "json",
             str(src)],
            cwd=str(ROOT),
            universal_newlines=True
        )
        j = json.loads(out or "{}")
        streams = j.get("streams") or []
        s0 = streams[0] if streams else {}
        fmt = j.get("format") or {}

        avg = _parse_ratio(str(s0.get("avg_frame_rate") or ""))
        rfr = _parse_ratio(str(s0.get("r_frame_rate") or ""))
        nb_frames = s0.get("nb_frames")
        try:
            nb = int(nb_frames) if nb_frames not in (None, "", "N/A") else None
        except Exception:
            nb = None

        # Prefer stream.duration, fallback to format.duration
        dur_s = s0.get("duration")
        if dur_s in (None, "", "N/A"):
            dur_s = fmt.get("duration")
        try:
            dur = float(dur_s) if dur_s not in (None, "", "N/A") else None
        except Exception:
            dur = None

        candidates: list[tuple[str, float]] = []

        if avg:
            candidates.append((_ratio_to_str(avg), _fps_from_ratio(avg)))
        if nb and dur and dur > 0:
            candidates.append((f"{(nb/dur):.6f}".rstrip("0").rstrip("."), float(nb) / float(dur)))
        if rfr:
            candidates.append((_ratio_to_str(rfr), _fps_from_ratio(rfr)))

        # Pick the first sane candidate (avg > derived > r_frame_rate).
        for val_str, fps in candidates:
            if fps and 1.0 <= fps <= 240.0:
                return val_str

        # If nothing sane, last resort: try r_frame_rate as-is.
        if rfr:
            return _ratio_to_str(rfr)

        return None
    except Exception:
        return None


class _RunThread(QtCore.QThread):
    progress = Signal(str)
    done = Signal(int, str)

    def __init__(self, cmds: List[List[str]], cwd: Optional[Path] = None, env: Optional[dict] = None, parent=None):
        super().__init__(parent)
        if cmds and isinstance(cmds[0], str):
            cmds = [cmds]
        self.cmds = cmds
        self.cwd = cwd
        self.env = env

    def run(self):
        last_cmd = ""
        try:
            total = len(self.cmds)
            for i, cmd in enumerate(self.cmds, 1):
                last_cmd = " ".join(cmd)
                self.progress.emit(f"[{i}/{total}] {last_cmd}")
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.cwd) if self.cwd else None,
                    env=self.env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                for line in proc.stdout:
                    self.progress.emit(line.rstrip())
                ret = proc.wait()
                if ret is not None and ret > 0x7FFFFFFF:
                    ret -= 0x100000000  # fix Windows -1 wrap
                if ret != 0:
                    self.done.emit(int(ret), last_cmd)
                    return
            self.done.emit(0, last_cmd)
        except Exception as e:
            self.progress.emit(f"Error: {e}")
            self.done.emit(-1, last_cmd)


class UpscPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._engines = detect_engines()
        self._realsr_models = scan_realsr_models()
        self._waifu_models = scan_waifu2x_models()
        self._srmd_models = scan_srmd_models()
        self._realsr_ncnn_models = scan_realsr_ncnn_models()
        self._ultrasharp_models = scan_ultrasharp_models()
        self._srmd_realsr_models = scan_srmd_realesrgan_models()
        self._last_outfile: Optional[Path] = None
        self._build_ui()

    def set_main(self, main):  # optional hook
        self._main = main

    def _build_ui(self):
        v_main = QtWidgets.QVBoxLayout(self)
        
        # ----- Top action bar (fixed; does not scroll) -----
        topbar = QtWidgets.QHBoxLayout()
        self.btn_upscale = QtWidgets.QPushButton("Upscale", self)
        self.btn_batch = QtWidgets.QPushButton("Batch…", self)
        self.btn_view_results = QtWidgets.QPushButton("View results", self)
        self.btn_info = QtWidgets.QPushButton("Info", self)
        topbar.addWidget(self.btn_upscale)
        topbar.addWidget(self.btn_batch)
        topbar.addWidget(self.btn_view_results)
        topbar.addWidget(self.btn_info)
        topbar.addStretch(1)
        v_main.addLayout(topbar)
        v_main.addSpacing(6)

        # Fancy green banner at the top
#        self.banner = QtWidgets.QLabel("Upscaling", self)
 #       self.banner.setObjectName("upscBanner")
  #      self.banner.setAlignment(Qt.AlignCenter)
   #     self.banner.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
    #    self.banner.setFixedHeight(45)
     #   self.banner.setStyleSheet(
      #      "#upscBanner {"
       #     " font-size: 15px;"
        #    " font-weight: 600;"
         #   " padding: 8px 17px;"
          #  " border-radius: 12px;"
#            " margin: 0 0 6px 0;"
 #           " color: white;"
  #          " background: qlineargradient("
   #         "   x1:0, y1:0, x2:1, y2:0,"
    #        "   stop:0 #1565c0,"      # deep blue
     #       "   stop:0.5 #1e88e5,"    # medium blue
      #      "   stop:1 #42a5f5"       # bright blue
       #     " );"
#            " letter-spacing: 0.5px;"
 #           "}"
  #      )
   #     v_main.addWidget(self.banner)
    #    v_main.addSpacing(4)
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        _content = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(_content)
        scroll.setWidget(_content)
        v_main.addWidget(scroll, 1)

        
        # Preferences (compact grid: avoids cramped header and wraps nicely)
        prefs_grid = QtWidgets.QGridLayout()
        prefs_grid.setHorizontalSpacing(12)
        prefs_grid.setVerticalSpacing(6)

        self.chk_remember = QtWidgets.QCheckBox('Remember settings', self)
        self.chk_remember.setChecked(True)
        self.chk_remember.hide()  # hidden; always on
        self.chk_video_thumbs = QtWidgets.QCheckBox("Video thumbnails (heavier)", self)
        self.chk_video_thumbs.setChecked(True)
        self.chk_video_thumbs.hide()
        self.chk_video_thumbs.setToolTip("Generate first-frame thumbnails for videos. Uses more CPU/RAM. Disabled while jobs run.")
        prefs_grid.addWidget(self.chk_video_thumbs, 0, 2, 1, 2)

        self.chk_streaming_lowmem = getattr(self, "chk_streaming_lowmem", None) or QtWidgets.QCheckBox("Streaming (low memory)", self)
        self.chk_streaming_lowmem.setChecked(True)
        self.chk_streaming_lowmem.setToolTip("Process in a streaming, low-memory mode using smaller buffers. Good default for stability.")
        prefs_grid.addWidget(self.chk_streaming_lowmem, 1, 2, 1, 2)

        prefs_grid.setColumnStretch(3, 1)
        v.addLayout(prefs_grid)

        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(8)

        row = QtWidgets.QHBoxLayout()
        self.edit_input = QtWidgets.QLineEdit(self)
        self.edit_input.setObjectName("edit_input")
        self.btn_browse = QtWidgets.QPushButton("Browse…", self)
        row.addWidget(self.edit_input, 1)
        row.addWidget(self.btn_browse)
        
        v.addLayout(row)

        # SeedVR2 (optional upscaler)
        self.chk_seedvr2 = QtWidgets.QCheckBox("Use seedVR2", self)
        v.addWidget(self.chk_seedvr2)

        scale_lay = QtWidgets.QHBoxLayout()
        scale_lay.addWidget(QtWidgets.QLabel("Scale:", self))
        self.spin_scale = QtWidgets.QDoubleSpinBox(self)
        self.spin_scale.setRange(1.0, 8.0)
        self.spin_scale.setSingleStep(0.5)
        self.spin_scale.setValue(2.0)
        self.spin_scale.setDecimals(1)
        self.slider_scale = QtWidgets.QSlider(Qt.Horizontal, self)
        self.slider_scale.setMinimum(10)
        self.slider_scale.setMaximum(80)
        self.slider_scale.setSingleStep(5)
        self.slider_scale.setPageStep(5)
        self.slider_scale.setValue(20)
        scale_lay.addWidget(self.spin_scale)
        scale_lay.addWidget(self.slider_scale, 1)
        # Wrap scale row so we can hide/disable it when SeedVR2 is enabled
        self._w_scale = QtWidgets.QWidget(self)
        self._w_scale.setLayout(scale_lay)
        v.addWidget(self._w_scale)

        from helpers.collapsible_compat import CollapsibleSection
        # Models (inline, no collapsible)

        # SeedVR2 settings (shown only when toggle is ON)
        self.box_seedvr2 = QtWidgets.QWidget(self)
        self.box_seedvr2.setVisible(False)
        lay_seed = QtWidgets.QVBoxLayout(self.box_seedvr2)
        lay_seed.setContentsMargins(6, 2, 6, 6)
        lay_seed.setSpacing(6)

        # Friendly warning / guidance (visible only when SeedVR2 is enabled)
        self.lbl_seedvr2_note = QtWidgets.QLabel(
            "SeedVR2 is not a fast upscale model, it can take more then an hour to upscale 1 minute of video to 1080p ! "
            "Use this model for realistic video footage, use realsrgan for animated video footage",
            self,
        )
        self.lbl_seedvr2_note.setWordWrap(True)
        self.lbl_seedvr2_note.setStyleSheet("color:#b9c3cf;font-size:12px;")
        lay_seed.addWidget(self.lbl_seedvr2_note)

        # GGUF model picker + refresh
        row_m = QtWidgets.QHBoxLayout()
        row_m.addWidget(QtWidgets.QLabel("GGUF model:", self))
        self.combo_seedvr2_gguf = QtWidgets.QComboBox(self)
        for m in scan_seedvr2_gguf():
            self.combo_seedvr2_gguf.addItem(m)
        row_m.addWidget(self.combo_seedvr2_gguf, 1)
        self.btn_seedvr2_refresh = QtWidgets.QPushButton("Refresh", self)
        self.btn_seedvr2_refresh.setFixedWidth(90)
        row_m.addWidget(self.btn_seedvr2_refresh)
        lay_seed.addLayout(row_m)

        # Resolution (Upscale to)
        row_r = QtWidgets.QHBoxLayout()
        row_r.addWidget(QtWidgets.QLabel("Upscale to:", self))
        self.combo_seedvr2_res = QtWidgets.QComboBox(self)
        for _r in ("720", "1080", "1440", "2160"):
            self.combo_seedvr2_res.addItem(_r)
        self.combo_seedvr2_res.setCurrentText("1440")
        row_r.addWidget(self.combo_seedvr2_res, 1)
        lay_seed.addLayout(row_r)

        self.lbl_seedvr2_2160_warn = QtWidgets.QLabel("⚠ 2160: use only for image upscaling (videos may be too heavy).", self)
        self.lbl_seedvr2_2160_warn.setStyleSheet("color:#ffcc66;font-size:12px;")
        self.lbl_seedvr2_2160_warn.setVisible(False)
        lay_seed.addWidget(self.lbl_seedvr2_2160_warn)

        # Temporal overlap
        self.chk_seedvr2_temporal = QtWidgets.QCheckBox("Temporal overlap", self)
        self.chk_seedvr2_temporal.setChecked(True)
        lay_seed.addWidget(self.chk_seedvr2_temporal)

        # Color correction
        row_c = QtWidgets.QHBoxLayout()
        row_c.addWidget(QtWidgets.QLabel("Color correction:", self))
        self.combo_seedvr2_color = QtWidgets.QComboBox(self)
        for _cc in ("lab", "none"):
            self.combo_seedvr2_color.addItem(_cc)
        self.combo_seedvr2_color.setCurrentText("lab")
        row_c.addWidget(self.combo_seedvr2_color, 1)
        lay_seed.addLayout(row_c)

        # Advanced settings (collapsed by default)
        self.grp_seedvr2_adv = QtWidgets.QGroupBox("Advanced settings", self)
        self.grp_seedvr2_adv.setCheckable(True)
        self.grp_seedvr2_adv.setChecked(False)
        adv = QtWidgets.QGridLayout(self.grp_seedvr2_adv)
        adv.setHorizontalSpacing(12)
        adv.setVerticalSpacing(6)

        adv.addWidget(QtWidgets.QLabel("Batch size:", self), 0, 0)
        self.spin_seedvr2_batch = QtWidgets.QSpinBox(self)
        self.spin_seedvr2_batch.setRange(1, 16)
        self.spin_seedvr2_batch.setValue(1)
        adv.addWidget(self.spin_seedvr2_batch, 0, 1)

        adv.addWidget(QtWidgets.QLabel("Chunk size:", self), 1, 0)
        self.spin_seedvr2_chunk = QtWidgets.QSpinBox(self)
        self.spin_seedvr2_chunk.setRange(1, 999)
        self.spin_seedvr2_chunk.setValue(20)
        adv.addWidget(self.spin_seedvr2_chunk, 1, 1)

        adv.addWidget(QtWidgets.QLabel("Attention mode:", self), 2, 0)
        self.combo_seedvr2_attn = QtWidgets.QComboBox(self)
        for _am in ("sdpa", "xformers", "flash_attn"):
            self.combo_seedvr2_attn.addItem(_am)
        self.combo_seedvr2_attn.setCurrentText("sdpa")
        adv.addWidget(self.combo_seedvr2_attn, 2, 1)

        self.lbl_seedvr2_attn_warn = QtWidgets.QLabel("Note: xformers / flash_attn must be installed in the SeedVR2 environment, or the run may fail.", self)
        self.lbl_seedvr2_attn_warn.setStyleSheet("color:#9fb3c8;font-size:12px;")
        self.lbl_seedvr2_attn_warn.setWordWrap(True)
        self.lbl_seedvr2_attn_warn.setVisible(False)
        adv.addWidget(self.lbl_seedvr2_attn_warn, 3, 0, 1, 2)

        # Offload options (tiled VAE)
        self.chk_seedvr2_vae_enc = QtWidgets.QCheckBox("VAE encode tiled", self)
        self.chk_seedvr2_vae_dec = QtWidgets.QCheckBox("VAE decode tiled", self)
        adv.addWidget(self.chk_seedvr2_vae_enc, 4, 0, 1, 2)
        adv.addWidget(self.chk_seedvr2_vae_dec, 5, 0, 1, 2)

        adv.addWidget(QtWidgets.QLabel("Encode tile size:", self), 6, 0)
        self.spin_seedvr2_enc_tile = QtWidgets.QSpinBox(self)
        self.spin_seedvr2_enc_tile.setRange(128, 4096)
        self.spin_seedvr2_enc_tile.setValue(1024)
        adv.addWidget(self.spin_seedvr2_enc_tile, 6, 1)

        adv.addWidget(QtWidgets.QLabel("Encode overlap:", self), 7, 0)
        self.spin_seedvr2_enc_ov = QtWidgets.QSpinBox(self)
        self.spin_seedvr2_enc_ov.setRange(0, 1024)
        self.spin_seedvr2_enc_ov.setValue(64)
        adv.addWidget(self.spin_seedvr2_enc_ov, 7, 1)

        adv.addWidget(QtWidgets.QLabel("Decode tile size:", self), 8, 0)
        self.spin_seedvr2_dec_tile = QtWidgets.QSpinBox(self)
        self.spin_seedvr2_dec_tile.setRange(128, 4096)
        self.spin_seedvr2_dec_tile.setValue(1024)
        adv.addWidget(self.spin_seedvr2_dec_tile, 8, 1)

        adv.addWidget(QtWidgets.QLabel("Decode overlap:", self), 9, 0)
        self.spin_seedvr2_dec_ov = QtWidgets.QSpinBox(self)
        self.spin_seedvr2_dec_ov.setRange(0, 1024)
        self.spin_seedvr2_dec_ov.setValue(64)
        adv.addWidget(self.spin_seedvr2_dec_ov, 9, 1)

        lay_seed.addWidget(self.grp_seedvr2_adv)
        v.addWidget(self.box_seedvr2)
        self.box_models = QtWidgets.QWidget(self)
        inner = QtWidgets.QVBoxLayout(self.box_models)
        inner.setContentsMargins(6, 2, 6, 6)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(QtWidgets.QLabel("Engine:", self))
        self.combo_engine = QtWidgets.QComboBox(self)
        for label, exe in self._engines:
            self.combo_engine.addItem(label, exe)
        if self.combo_engine.count() == 0:
            self.combo_engine.addItem("Real-ESRGAN (expected at models/realesrgan)",
                                      str(REALSR_DIR / "realesrgan-ncnn-vulkan.exe"))
        hl.addWidget(self.combo_engine, 1)
        inner.addLayout(hl)

        self.stk_models = QtWidgets.QStackedWidget(self)
        # Real-ESRGAN
        pg_r = QtWidgets.QWidget()
        lay_r = QtWidgets.QHBoxLayout(pg_r)
        lay_r.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_realsr = QtWidgets.QComboBox(self)
        for m in self._realsr_models:
            self.combo_model_realsr.addItem(m)
        lay_r.addWidget(self.combo_model_realsr, 1)
        # Badge for model category
        self.lbl_model_badge = QtWidgets.QLabel("—", self)
        self.lbl_model_badge.setStyleSheet("QLabel{padding:2px 6px;border-radius:8px;background:#394b70;color:#cfe3ff;font-size:12px;font-weight:600;}")
        
        self.stk_models.addWidget(pg_r)

        # Waifu2x
        pg_w = QtWidgets.QWidget()
        lay_w = QtWidgets.QHBoxLayout(pg_w)
        lay_w.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_w2x = QtWidgets.QComboBox(self)
        for m in self._waifu_models:
            self.combo_model_w2x.addItem(m)
        lay_w.addWidget(self.combo_model_w2x, 1)
        self.stk_models.addWidget(pg_w)
        # SRMD (ncnn)
        pg_srmd = QtWidgets.QWidget()
        lay_s = QtWidgets.QHBoxLayout(pg_srmd)
        lay_s.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_srmd = QtWidgets.QComboBox(self)
        for m in self._srmd_models:
            self.combo_model_srmd.addItem(m)
        lay_s.addWidget(self.combo_model_srmd, 1)
        self.stk_models.addWidget(pg_srmd)
        # RealSR (ncnn)
        pg_rs = QtWidgets.QWidget()
        lay_rs = QtWidgets.QHBoxLayout(pg_rs)
        lay_rs.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_realsr_ncnn = QtWidgets.QComboBox(self)
        for m in self._realsr_ncnn_models:
            self.combo_model_realsr_ncnn.addItem(m)
        lay_rs.addWidget(self.combo_model_realsr_ncnn, 1)
        self.stk_models.addWidget(pg_rs)
        # UltraSharp (via Real-ESRGAN backend)
        pg_ul = QtWidgets.QWidget()
        lay_ul = QtWidgets.QHBoxLayout(pg_ul)
        lay_ul.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_ultrasharp = QtWidgets.QComboBox(self)
        for m in self._ultrasharp_models:
            self.combo_model_ultrasharp.addItem(m)
        lay_ul.addWidget(self.combo_model_ultrasharp, 1)
        self.stk_models.addWidget(pg_ul)

        # SRMD (via Real-ESRGAN backend)
        pg_srmd_rs = QtWidgets.QWidget()
        lay_srmd_rs = QtWidgets.QHBoxLayout(pg_srmd_rs)
        lay_srmd_rs.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_srmd_realsr = QtWidgets.QComboBox(self)
        for m in self._srmd_realsr_models:
            self.combo_model_srmd_realsr.addItem(m)
        lay_srmd_rs.addWidget(self.combo_model_srmd_realsr, 1)
        self.stk_models.addWidget(pg_srmd_rs)


        # GFPGAN (faces)
        pg_gfp = QtWidgets.QWidget()
        lay_gfp = QtWidgets.QHBoxLayout(pg_gfp)
        lay_gfp.addWidget(QtWidgets.QLabel("Model:", self))
        self.combo_model_gfpgan = QtWidgets.QComboBox(self)
        # Keep labels stable; weights live in models/gfpgan/
        self.combo_model_gfpgan.addItem("GFPGANv1.4")
        self.combo_model_gfpgan.addItem("GFPGANv1.3")
        lay_gfp.addWidget(self.combo_model_gfpgan, 1)
        self.stk_models.addWidget(pg_gfp)


        inner.addWidget(self.stk_models)

        # Model hint row (hint + badge on the same line)
        self.lbl_model_hint = QtWidgets.QLabel("", self)
        self.lbl_model_hint.setWordWrap(True)
        self.lbl_model_hint.setStyleSheet("color:#9fb3c8;font-size:12px;")
        hl_hint = QtWidgets.QHBoxLayout()
        hl_hint.addWidget(self.lbl_model_hint, 1)
        hl_hint.addWidget(self.lbl_model_badge)
        inner.addLayout(hl_hint)

        hl_out = QtWidgets.QHBoxLayout()
        hl_out.addWidget(QtWidgets.QLabel("Output:", self))
        self.edit_outdir = QtWidgets.QLineEdit(str(OUT_VIDEOS))
        btn_out = QtWidgets.QPushButton("…", self)
        hl_out.addWidget(self.edit_outdir, 1)
        hl_out.addWidget(btn_out)
        inner.addLayout(hl_out)
        v.addWidget(self.box_models)

        # Encoder (inline)
        self.box_encoder = QtWidgets.QWidget(self)
        lay_enc = QtWidgets.QGridLayout(self.box_encoder); lay_enc.setContentsMargins(6,2,6,6)
        # Video codec
        lay_enc.addWidget(QtWidgets.QLabel("Video codec:", self), 0, 0)
        self.combo_vcodec = QtWidgets.QComboBox(self)
        for c in ("libx264","libx265","libsvtav1","libaom-av1"):
            self.combo_vcodec.addItem(c)
        self.combo_vcodec.setCurrentText("libx264")
        lay_enc.addWidget(self.combo_vcodec, 0, 1)
        # Rate control
        self.rad_crf = QtWidgets.QRadioButton("CRF", self); self.rad_crf.setChecked(True)
        self.spin_crf = QtWidgets.QSpinBox(self); self.spin_crf.setRange(0, 51); self.spin_crf.setValue(18)
        lay_enc.addWidget(self.rad_crf, 1, 0); lay_enc.addWidget(self.spin_crf, 1, 1)
        self.rad_bitrate = QtWidgets.QRadioButton("Bitrate (kbps)", self)
        self.spin_bitrate = QtWidgets.QSpinBox(self); self.spin_bitrate.setRange(100, 200000); self.spin_bitrate.setValue(8000)
        lay_enc.addWidget(self.rad_bitrate, 2, 0); lay_enc.addWidget(self.spin_bitrate, 2, 1)

        # Grey out the non-selected rate control
        try:
            def _update_rate_controls():
                use_crf = bool(self.rad_crf.isChecked())
                try:
                    self.spin_crf.setEnabled(use_crf)
                except Exception:
                    pass
                try:
                    self.spin_bitrate.setEnabled(bool(self.rad_bitrate.isChecked()))
                except Exception:
                    pass
            try:
                self.rad_crf.toggled.connect(_update_rate_controls)
                self.rad_bitrate.toggled.connect(_update_rate_controls)
            except Exception:
                pass
            _update_rate_controls()
        except Exception:
            pass

        # Preset + Keyint
        lay_enc.addWidget(QtWidgets.QLabel("Preset:", self), 3, 0)
        self.combo_preset = QtWidgets.QComboBox(self)
        for p in ("ultrafast","fast","medium","slow"):
            self.combo_preset.addItem(p)
        self.combo_preset.setCurrentText("fast")
        lay_enc.addWidget(self.combo_preset, 3, 1)
        lay_enc.addWidget(QtWidgets.QLabel("Keyint (GOP):", self), 4, 0)
        self.spin_keyint = QtWidgets.QSpinBox(self); self.spin_keyint.setRange(0, 1000); self.spin_keyint.setValue(0)
        self.spin_keyint.setToolTip("0 = let encoder decide; otherwise sets -g keyint")
        lay_enc.addWidget(self.spin_keyint, 4, 1)

        # Hide Keyint (auto mode). Force to 0 under the hood.
        try:
            self._lbl_keyint = lay_enc.itemAtPosition(4, 0).widget() if hasattr(lay_enc, "itemAtPosition") else None
            if self._lbl_keyint is not None:
                self._lbl_keyint.hide()
            self.spin_keyint.setValue(0)
            self.spin_keyint.hide()
            try:
                # ensure any saved setting won't override this
                self.spin_keyint.valueChanged.connect(lambda _v: self.spin_keyint.setValue(0))
            except Exception:
                pass
        except Exception:
            pass

        # Audio
        lay_enc.addWidget(QtWidgets.QLabel("Audio:", self), 5, 0)
        self.radio_a_copy = QtWidgets.QRadioButton("Keep", self); self.radio_a_copy.setChecked(True)
        self.radio_a_encode = QtWidgets.QRadioButton("Encode", self)
        self.radio_a_mute = QtWidgets.QRadioButton("Mute", self)
        arow = QtWidgets.QHBoxLayout(); arow.addWidget(self.radio_a_copy); arow.addWidget(self.radio_a_encode); arow.addWidget(self.radio_a_mute); arow.addStretch(1)
        lay_enc.addLayout(arow, 5, 1)

        # --- Ensure radio groups are independent ---
        try:
            self.grp_rate = QtWidgets.QButtonGroup(self.box_encoder)
            self.grp_rate.setExclusive(True)
            self.grp_rate.addButton(self.rad_crf)
            self.grp_rate.addButton(self.rad_bitrate)

            self.grp_audio = QtWidgets.QButtonGroup(self.box_encoder)
            self.grp_audio.setExclusive(True)
            self.grp_audio.addButton(self.radio_a_copy)
            self.grp_audio.addButton(self.radio_a_encode)
            self.grp_audio.addButton(self.radio_a_mute)
        except Exception:
            pass
        # Audio codec/bitrate
        lay_enc.addWidget(QtWidgets.QLabel("Audio codec:", self), 6, 0)
        self.combo_acodec = QtWidgets.QComboBox(self)
        for ac in ("aac","libopus","libvorbis"):
            self.combo_acodec.addItem(ac)
        lay_enc.addWidget(self.combo_acodec, 6, 1)
        lay_enc.addWidget(QtWidgets.QLabel("Audio bitrate (kbps):", self), 7, 0)
        self.spin_abitrate = QtWidgets.QSpinBox(self); self.spin_abitrate.setRange(32, 1024); self.spin_abitrate.setValue(192)
        lay_enc.addWidget(self.spin_abitrate, 7, 1)

        # Replace spinner with dropdown for audio bitrate
        try:
            self.combo_abitrate = QtWidgets.QComboBox(self)
            _abrs = [24, 36, 48, 64, 112, 128, 160, 192, 256, 320]
            for _br in _abrs:
                self.combo_abitrate.addItem(str(_br))
            try:
                self.combo_abitrate.setCurrentText("192")
            except Exception:
                pass
            lay_enc.addWidget(self.combo_abitrate, 7, 1)
            try:
                self.spin_abitrate.hide()
            except Exception:
                pass
        except Exception:
            pass


        # --- Audio controls visibility & bitrate discretization ---
        try:
            # Cached label widgets from the grid layout by row/column
            self.lbl_acodec = lay_enc.itemAtPosition(6, 0).widget() if hasattr(lay_enc, "itemAtPosition") else None
            self.lbl_abitrate = lay_enc.itemAtPosition(7, 0).widget() if hasattr(lay_enc, "itemAtPosition") else None

            # Allowed audio bitrates
            self._allowed_abitrates = [24, 36, 48, 64, 112, 128, 160, 192, 256, 320]

            def _snap_abitrate(val: int):
                try:
                    target = min(self._allowed_abitrates, key=lambda x: (abs(x - int(val)), x))
                    if target != self.spin_abitrate.value():
                        old = self.spin_abitrate.blockSignals(True)
                        self.spin_abitrate.setValue(target)
                        self.spin_abitrate.blockSignals(old)
                except Exception:
                    pass

            try:
                self.spin_abitrate.setRange(min(self._allowed_abitrates), max(self._allowed_abitrates))
                self.spin_abitrate.setSingleStep(1)
                _snap_abitrate(self.spin_abitrate.value())
                self.spin_abitrate.valueChanged.connect(_snap_abitrate)
                if hasattr(self.spin_abitrate, "editingFinished"):
                    self.spin_abitrate.editingFinished.connect(lambda: _snap_abitrate(self.spin_abitrate.value()))
            except Exception:
                pass

            def _update_audio_controls():
                enc = False
                try:
                    enc = bool(self.radio_a_encode.isChecked())
                except Exception:
                    pass
                for w in (self.lbl_acodec, self.combo_acodec, self.lbl_abitrate, self.combo_abitrate):
                    try:
                        if w is not None:
                            w.setVisible(enc)
                    except Exception:
                        pass

            try:
                self.radio_a_copy.toggled.connect(_update_audio_controls)
                self.radio_a_encode.toggled.connect(_update_audio_controls)
                self.radio_a_mute.toggled.connect(_update_audio_controls)
            except Exception:
                pass

            _update_audio_controls()
        except Exception:
            pass
        # Encoder inline: layout already attached to self.box_encoder
        # (removed collapsible content setters)
        v.addWidget(self.box_encoder)

        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        try:
            fm = self.log.fontMetrics()
            self.log.setFixedHeight(int(fm.height() * 6.5))
        except Exception:
            self.log.setFixedHeight(100)
        self.log.setMaximumBlockCount(1000)
        if SHOW_UPSCALE_LOG:
            v.addWidget(self.log)
        else:
            self.log.hide()


        # Tooltips
        try:
            self.btn_upscale.setToolTip("Load any video or image in the media player, then select an engine and model and press 'Upscale'.")
            self.btn_batch.setToolTip("Add multiple files or a full folder to the queue for upscaling with the current settings.")
            self.btn_view_results.setToolTip("Open the output folder in Media Explorer and scan it automatically.")
            self.btn_info.setToolTip("Shows detailed info (only works when you load an image or video directly in the Upscale tab).")
        except Exception:
            pass

        # -- Style tweaks: bigger bottom button fonts + Upscale hover blue
        try:
            # Make the bottom action buttons' font match the Describer actions
            for _b in (self.btn_upscale, self.btn_batch, self.btn_view_results, self.btn_info):
                try:
                    _b.setMinimumHeight(32)
                except Exception:
                    pass
                _f = _b.font()
                try:
                    _sz = _f.pointSize()
                    if _sz <= 0:
                        _sz = 10
                except Exception:
                    _sz = 10
                # About +3pt for a clearly larger label, like in Describer
                _f.setPointSize(_sz + 3)
                _b.setFont(_f)
                try:
                    from PySide6.QtWidgets import QSizePolicy as _QSP
                    _b.setSizePolicy(_QSP.Preferred, _QSP.Fixed)
                except Exception:
                    pass
            # Hover background for Upscale button only
            self.btn_upscale.setObjectName("btn_upscale_main")
            self.btn_upscale.setStyleSheet((self.btn_upscale.styleSheet() or "") + "QPushButton:hover{background-color:#0d6efd;}")
        except Exception:
            pass


        # wiring
        self.btn_browse.clicked.connect(self._pick_single)
        self.btn_info.clicked.connect(self._show_media_info)
        self.btn_view_results.clicked.connect(self._view_results)
        self.btn_batch.clicked.connect(self._pick_batch)
        self.btn_upscale.clicked.connect(self._do_single)
        btn_out.clicked.connect(self._pick_outdir)
        self.spin_scale.valueChanged.connect(self._sync_scale_from_spin)
        self.slider_scale.valueChanged.connect(self._sync_scale_from_slider)
        self.combo_engine.currentIndexChanged.connect(self._update_engine_ui)
        self._update_engine_ui()
        # SeedVR2 wiring
        try:
            self.chk_seedvr2.toggled.connect(self._update_seedvr2_mode)
        except Exception:
            pass
        try:
            self.btn_seedvr2_refresh.clicked.connect(self._seedvr2_refresh_models)
        except Exception:
            pass
        try:
            self.combo_seedvr2_res.currentTextChanged.connect(self._update_seedvr2_warnings)
        except Exception:
            pass
        try:
            self.combo_seedvr2_attn.currentTextChanged.connect(self._update_seedvr2_warnings)
        except Exception:
            pass
        try:
            self._update_seedvr2_mode(self.chk_seedvr2.isChecked())
        except Exception:
            pass

        try:
            self.chk_seedvr2.toggled.connect(self._update_seedvr2_mode)
        except Exception:
            pass
        try:
            self.btn_seedvr2_refresh.clicked.connect(self._seedvr2_refresh_models)
        except Exception:
            pass
        try:
            self.combo_seedvr2_res.currentTextChanged.connect(self._update_seedvr2_warnings)
        except Exception:
            pass
        try:
            self.combo_seedvr2_attn.currentTextChanged.connect(self._update_seedvr2_warnings)
        except Exception:
            pass
        # apply initial mode
        try:
            self._update_seedvr2_mode(bool(self.chk_seedvr2.isChecked()))
        except Exception:
            pass
        try:
            self.combo_model_realsr.currentTextChanged.connect(self._update_engine_ui)
        except Exception:
            pass


    
        try:
            self.combo_model_realsr.currentTextChanged.connect(self._update_model_hint)
        except Exception:
            pass
        try:
            self.btn_auto_tile.clicked.connect(self._auto_tile_size)
        except Exception:
            pass
        self._update_model_hint()

        # Logger buffering to avoid UI stalls when many lines arrive quickly
        try:
            self._log_buf = []
            self._log_flush_timer = QTimer(self)
            self._log_flush_timer.setSingleShot(True)
            self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        except Exception:
            pass


        # Load persisted settings (if any)
        try:
            self._load_settings()
        except Exception:
            pass

        # Ensure batch limit matches current scale
        try:
            self._update_batch_limit()
        except Exception:
            pass

        # Wire quick buttons
        try:
            self.btn_scale_2x.clicked.connect(lambda: self._set_scale_quick(2.0))
            self.btn_scale_3x.clicked.connect(lambda: self._set_scale_quick(3.0))
            self.btn_scale_4x.clicked.connect(lambda: self._set_scale_quick(4.0))
        except Exception:
            pass

        
        # Save on application exit too (best-effort)
        try:
            app = QtWidgets.QApplication.instance()
            if app:
                app.aboutToQuit.connect(self._auto_save_on_exit)
        except Exception:
            pass

        # Wire auto-save on change
        try:
            self._connect_auto_save()
        except Exception:
            pass

# UI helpers
    def _set_section_state(self, box, open_: bool):
        """Reliably set a CollapsibleSection open/closed, even if setChecked() is ignored.
        Uses toggle.click() when the current state differs, which triggers the same logic as a user click.
        """
        try:
            tb = getattr(box, "toggle", None)
            if tb is None:
                return
            want = bool(open_)
            cur = bool(tb.isChecked())
            if cur != want:
                tb.click()
        except Exception:
            pass


    def _set_scale_quick(self, val: float):
        try:
            self.spin_scale.setValue(float(val))
            self._sync_scale_from_spin(float(val))
        except Exception:
            pass

    def _settings_path(self) -> Path:
        try:
            return (ROOT / "presets" / "setsave" / "upsc_settings.json")
        except Exception:
            return Path("presets/setsave/upsc_settings.json").resolve()
    def eventFilter(self, obj, ev):
        try:
            return super().eventFilter(obj, ev)
        except Exception:
            return False



    def _connect_auto_save(self):
        """Connect change signals to auto-save, safely and individually, so a missing widget won't break others."""
        def _connect(obj, sig_name):
            try:
                if obj is None:
                    return False
                sig = getattr(obj, sig_name, None)
                if sig is None:
                    return False
                sig.connect(self._auto_save_if_enabled)
                return True
            except Exception:
                return False

        # Standard widgets
        pairs = [
            ("spin_scale", "valueChanged"),
            ("slider_scale", "valueChanged"),
            ("combo_engine", "currentIndexChanged"),
            ("combo_model_realsr", "currentTextChanged"),
            ("combo_model_w2x", "currentTextChanged"),
            ("combo_model_gfpgan", "currentTextChanged"),
            ("edit_outdir", "textChanged"),
            ("chk_play_internal", "toggled"),
            ("chk_replace_in_player", "toggled"),
            ("combo_vcodec", "currentTextChanged"),
            ("rad_crf", "toggled"),
            ("spin_crf", "valueChanged"),
            ("rad_bitrate", "toggled"),
            ("spin_bitrate", "valueChanged"),
            ("combo_preset", "currentTextChanged"),
            ("spin_keyint", "valueChanged"),
            ("radio_a_copy", "toggled"),
            ("radio_a_encode", "toggled"),
            ("radio_a_mute", "toggled"),
            ("combo_acodec", "currentTextChanged"),
            ("combo_abitrate", "currentTextChanged"),
            ("spin_tile", "valueChanged"),
            ("spin_overlap", "valueChanged"),
            ("chk_deinterlace", "toggled"),
            ("combo_range", "currentTextChanged"),
            ("chk_deblock", "toggled"),
            ("chk_denoise", "toggled"),
            ("chk_deband", "toggled"),
            ("sld_sharpen", "valueChanged"),
            ("chk_seedvr2", "toggled"),
            ("combo_seedvr2_gguf", "currentTextChanged"),
            ("combo_seedvr2_res", "currentTextChanged"),
            ("chk_seedvr2_temporal", "toggled"),
            ("combo_seedvr2_color", "currentTextChanged"),
            ("spin_seedvr2_batch", "valueChanged"),
            ("spin_seedvr2_chunk", "valueChanged"),
            ("combo_seedvr2_attn", "currentTextChanged"),
            ("chk_seedvr2_vae_enc", "toggled"),
            ("chk_seedvr2_vae_dec", "toggled"),
            ("spin_seedvr2_enc_tile", "valueChanged"),
            ("spin_seedvr2_enc_ov", "valueChanged"),
            ("spin_seedvr2_dec_tile", "valueChanged"),
            ("spin_seedvr2_dec_ov", "valueChanged"),
                    ]

        connected = 0
        for name, sig_name in pairs:
            obj = getattr(self, name, None)
            if _connect(obj, sig_name):
                connected += 1

        # Collapsible sections (their .toggle has a .toggled signal)
        for box_name in ("box_models", "box_encoder", "box_advanced"):
            box = getattr(self, box_name, None)
            toggle = getattr(box, "toggle", None) if box is not None else None
            if _connect(toggle, "toggled"):
                connected += 1

        try:
            self._append_log(f"[settings] Auto-save wired to {connected} UI change signal(s).")
        except Exception:
            pass


# UI helpers

    def _auto_save_if_enabled(self, *args, **kwargs):
        try:
            self._save_settings()
        except Exception:
            pass


    def _gather_settings(self) -> dict:
        d = {}
        # Core toggles / basics
        try: d['remember'] = True
        except Exception: pass
        try: d["scale"] = float(self.spin_scale.value())
        except Exception: pass
        try: d["engine_index"] = int(self.combo_engine.currentIndex())
        except Exception: pass
        try: d["model_realsr"] = self.combo_model_realsr.currentText()
        except Exception: pass
        try: d["model_w2x"] = self.combo_model_w2x.currentText()
        except Exception: pass
        try: d["model_gfpgan"] = getattr(self, "combo_model_gfpgan", None).currentText() if getattr(self, "combo_model_gfpgan", None) else ""
        except Exception: pass
        try: d["outdir"] = self.edit_outdir.text().strip()
        except Exception: pass
        try:
            d["play_internal"] = bool(self.chk_play_internal.isChecked())
            d["replace_in_player"] = bool(self.chk_replace_in_player.isChecked())
        except Exception: pass
        # Video encode settings
        try: d["vcodec"] = self.combo_vcodec.currentText()
        except Exception: pass
        try:
            d["use_crf"] = bool(self.rad_crf.isChecked())
            d["crf"] = int(self.spin_crf.value())
            d["use_bitrate"] = bool(self.rad_bitrate.isChecked())
            d["bitrate"] = int(self.spin_bitrate.value())
        except Exception: pass
        try: d["preset"] = self.combo_preset.currentText()
        except Exception: pass
        try: d["keyint"] = 0
        except Exception: pass
        # Audio
        try:
            if self.radio_a_mute.isChecked():
                d["audio"] = "mute"
            elif self.radio_a_encode.isChecked():
                d["audio"] = "encode"
            else:
                d["audio"] = "copy"
        except Exception: pass
        try: d["acodec"] = self.combo_acodec.currentText()
        except Exception: pass
        try: d["abitrate"] = int(self.combo_abitrate.currentText())
        except Exception: pass
        # Advanced / tiles
        try:
            d["tile"] = int(self.spin_tile.value())
            d["overlap"] = int(self.spin_overlap.value())
        except Exception: pass
        try: d["deinterlace"] = bool(self.chk_deinterlace.isChecked())
        except Exception: pass
        try: d["range"] = self.combo_range.currentText()
        except Exception: pass
        try:
            d["deblock"] = bool(self.chk_deblock.isChecked())
            d["denoise"] = bool(self.chk_denoise.isChecked())
            d["deband"] = bool(self.chk_deband.isChecked())
        except Exception: pass
        try: d["sharpen"] = int(self.sld_sharpen.value())
        except Exception: pass
                # SeedVR2
        try: d["seedvr2_enabled"] = bool(self.chk_seedvr2.isChecked())
        except Exception: pass
        try: d["seedvr2_gguf"] = self.combo_seedvr2_gguf.currentText()
        except Exception: pass
        try: d["seedvr2_res"] = self.combo_seedvr2_res.currentText()
        except Exception: pass
        try: d["seedvr2_temporal"] = bool(self.chk_seedvr2_temporal.isChecked())
        except Exception: pass
        try: d["seedvr2_color"] = self.combo_seedvr2_color.currentText()
        except Exception: pass
        try:
            d["seedvr2_batch"] = int(self.spin_seedvr2_batch.value())
            d["seedvr2_chunk"] = int(self.spin_seedvr2_chunk.value())
            d["seedvr2_attn"] = self.combo_seedvr2_attn.currentText()
            d["seedvr2_vae_enc"] = bool(self.chk_seedvr2_vae_enc.isChecked())
            d["seedvr2_vae_dec"] = bool(self.chk_seedvr2_vae_dec.isChecked())
            d["seedvr2_enc_tile"] = int(self.spin_seedvr2_enc_tile.value())
            d["seedvr2_enc_ov"] = int(self.spin_seedvr2_enc_ov.value())
            d["seedvr2_dec_tile"] = int(self.spin_seedvr2_dec_tile.value())
            d["seedvr2_dec_ov"] = int(self.spin_seedvr2_dec_ov.value())
        except Exception:
            pass

# Sections collapsed state
        d["sections"] = {}
        try:
            d["sections"]["models"] = bool(self.box_models.toggle.isChecked())
        except Exception:
            try:
                d["sections"]["models"] = bool(getattr(self.box_models, "isExpanded", lambda: True)())
            except Exception:
                d["sections"]["models"] = True
        try:
            d["sections"]["encoder"] = bool(self.box_encoder.toggle.isChecked())
        except Exception:
            try:
                d["sections"]["encoder"] = bool(getattr(self.box_encoder, "isExpanded", lambda: True)())
            except Exception:
                d["sections"]["encoder"] = True
        try:
            d["sections"]["advanced"] = bool(self.box_advanced.toggle.isChecked())
        except Exception:
            try:
                d["sections"]["advanced"] = bool(getattr(self.box_advanced, "isExpanded", lambda: False)())
            except Exception:
                d["sections"]["advanced"] = False
        # Batch / gallery options
        try: d["video_thumbs"] = bool(self.chk_video_thumbs.isChecked())
        except Exception: d["video_thumbs"] = False
        try: d["streaming_lowmem"] = bool(self.chk_streaming_lowmem.isChecked())
        except Exception: d["streaming_lowmem"] = True

        return d

    def _apply_settings(self, d: dict):
        try:
            # remember toggle removed from UI; always on
            self.chk_remember.setChecked(True)
        except Exception:
            pass
        try:
            self.spin_scale.setValue(float(d.get("scale", 2.0)))
        except Exception:
            pass
        try:
            idx = int(d.get("engine_index", 0))
            if 0 <= idx < self.combo_engine.count():
                self.combo_engine.setCurrentIndex(idx)
        except Exception:
            pass
        try:
            m = d.get("model_realsr")
            if m:
                i = self.combo_model_realsr.findText(m)
                if i >= 0: self.combo_model_realsr.setCurrentIndex(i)
        except Exception:
            pass
        try:
            m = d.get("model_w2x")
            if m:
                i = self.combo_model_w2x.findText(m)
                if i >= 0: self.combo_model_w2x.setCurrentIndex(i)
        except Exception:
            pass
        try:
            m = d.get("model_gfpgan")
            if m and getattr(self, "combo_model_gfpgan", None):
                i = self.combo_model_gfpgan.findText(m)
                if i >= 0: self.combo_model_gfpgan.setCurrentIndex(i)
        except Exception:
            pass
        try:
            out = d.get("outdir")
            if out: self.edit_outdir.setText(out)
        except Exception:
            pass

        # SeedVR2
        try:
            self.chk_seedvr2.setChecked(bool(d.get("seedvr2_enabled", False)))
        except Exception:
            pass
        try:
            gg = d.get("seedvr2_gguf")
            if gg:
                i = self.combo_seedvr2_gguf.findText(gg)
                if i >= 0: self.combo_seedvr2_gguf.setCurrentIndex(i)
        except Exception:
            pass
        try:
            rs = str(d.get("seedvr2_res") or "")
            if rs:
                i = self.combo_seedvr2_res.findText(rs)
                if i >= 0: self.combo_seedvr2_res.setCurrentIndex(i)
        except Exception:
            pass
        try:
            self.chk_seedvr2_temporal.setChecked(bool(d.get("seedvr2_temporal", True)))
        except Exception:
            pass
        try:
            cc = d.get("seedvr2_color")
            if cc:
                i = self.combo_seedvr2_color.findText(cc)
                if i >= 0: self.combo_seedvr2_color.setCurrentIndex(i)
        except Exception:
            pass
        try:
            self.spin_seedvr2_batch.setValue(int(d.get("seedvr2_batch", 1)))
            self.spin_seedvr2_chunk.setValue(int(d.get("seedvr2_chunk", 20)))
            am = d.get("seedvr2_attn")
            if am:
                i = self.combo_seedvr2_attn.findText(am)
                if i >= 0: self.combo_seedvr2_attn.setCurrentIndex(i)
            self.chk_seedvr2_vae_enc.setChecked(bool(d.get("seedvr2_vae_enc", False)))
            self.chk_seedvr2_vae_dec.setChecked(bool(d.get("seedvr2_vae_dec", False)))
            self.spin_seedvr2_enc_tile.setValue(int(d.get("seedvr2_enc_tile", 1024)))
            self.spin_seedvr2_enc_ov.setValue(int(d.get("seedvr2_enc_ov", 64)))
            self.spin_seedvr2_dec_tile.setValue(int(d.get("seedvr2_dec_tile", 1024)))
            self.spin_seedvr2_dec_ov.setValue(int(d.get("seedvr2_dec_ov", 64)))
        except Exception:
            pass
        try:
            self._update_seedvr2_mode(self.chk_seedvr2.isChecked())
        except Exception:
            pass

        try:
            self.chk_play_internal.setChecked(bool(d.get("play_internal", True)))
            self.chk_replace_in_player.setChecked(bool(d.get("replace_in_player", True)))
        except Exception:
            pass
        try:
            vc = d.get("vcodec"); 
            if vc:
                i = self.combo_vcodec.findText(vc)
                if i >= 0: self.combo_vcodec.setCurrentIndex(i)
        except Exception:
            pass
        try:
            if d.get("use_crf", True):
                self.rad_crf.setChecked(True)
            if d.get("use_bitrate", False):
                self.rad_bitrate.setChecked(True)
            self.spin_crf.setValue(int(d.get("crf", 18)))
            self.spin_bitrate.setValue(int(d.get("bitrate", 8000)))
        except Exception:
            pass
        try:
            pr = d.get("preset")
            if pr:
                i = self.combo_preset.findText(pr)
                if i >= 0: self.combo_preset.setCurrentIndex(i)
        except Exception:
            pass
        try:
            self.spin_keyint.setValue(0)
        except Exception:
            pass
        try:
            a = d.get("audio", "copy")
            if a == "mute": self.radio_a_mute.setChecked(True)
            elif a == "encode": self.radio_a_encode.setChecked(True)
            else: self.radio_a_copy.setChecked(True)
        except Exception:
            pass
        try:
            ac = d.get("acodec")
            if ac:
                i = self.combo_acodec.findText(ac)
                if i >= 0: self.combo_acodec.setCurrentIndex(i)
            (_i := self.combo_abitrate.findText(str(d.get("abitrate", 192)))); self.combo_abitrate.setCurrentIndex(_i if _i >= 0 else self.combo_abitrate.findText("192"))
        except Exception:
            pass
        try:
            self.spin_tile.setValue(int(d.get("tile", 0)))
            self.spin_overlap.setValue(int(d.get("overlap", 0)))
        except Exception:
            pass
        try:
            self.chk_deinterlace.setChecked(bool(d.get("deinterlace", False)))
            rng = d.get("range")
            if rng:
                i = self.combo_range.findText(rng)
                if i >= 0: self.combo_range.setCurrentIndex(i)
            self.chk_deblock.setChecked(bool(d.get("deblock", False)))
            self.chk_denoise.setChecked(bool(d.get("denoise", False)))
            self.chk_deband.setChecked(bool(d.get("deband", False)))
            self.sld_sharpen.setValue(int(d.get("sharpen", 0)))
        except Exception:
            pass
        try:
            secs = d.get("sections", {})
            try:
                self._append_log(f"[settings] Loaded section states: {secs}")
            except Exception:
                pass
            self._set_section_state(self.box_models, bool(secs.get("models", True)))
            self._set_section_state(self.box_encoder, bool(secs.get("encoder", True)))
            self._set_section_state(self.box_advanced, bool(secs.get("advanced", False)))
        except Exception:
            pass
            self.chk_streaming_lowmem.setChecked(bool(d.get("streaming_lowmem", True)))
        except Exception:
            pass
        except Exception:
            pass

    def _save_settings(self):
        try:
            d = self._gather_settings()
            p = self._settings_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(d, indent=2), encoding="utf-8")
            try:
                self._append_log(f"[settings] Saved to: {p}")
            except Exception:
                pass
        except Exception as e:
            try:
                self._append_log(f"[settings] Save failed: {e}")
            except Exception:
                pass

    def _load_settings(self):
        try:
            p = self._settings_path()
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                try:
                    self._append_log(f"[settings] Loaded from: {p}")
                except Exception:
                    pass
                # Apply settings, then ensure remember is True (hidden)
                self._apply_settings(d)
                try:
                    self.chk_remember.setChecked(True)
                except Exception:
                    pass
            else:
                try:
                    self._append_log(f"[settings] No file yet at: {p}")
                except Exception:
                    pass
        except Exception as e:
            try:
                self._append_log(f"[settings] Load failed: {e}")
            except Exception:
                pass

    def _update_batch_limit(self):
        """Batch limit control removed; keeping method as no-op for compatibility."""
        return

    def _apply_streaming_lowmem(self, cmd: list) -> list:
        """Inject low-memory streaming flags into FFmpeg commands when the toggle is ON."""
        try:
            if getattr(self, "chk_streaming_lowmem", None) and self.chk_streaming_lowmem.isChecked():
                if cmd and isinstance(cmd, list) and len(cmd) > 1 and cmd[0] == FFMPEG:
                    inject = ["-fflags", "nobuffer", "-probesize", "64k", "-analyzeduration", "1000"]
                    # add -threads 1 only if not already specified
                    if "-threads" not in cmd:
                        inject += ["-threads", "1"]
                    # rebuild: FFMPEG + inject + rest (excluding the first FFMPEG element)
                    cmd = [cmd[0]] + inject + cmd[1:]
        except Exception:
            pass
        return cmd




    def _make_thumb(self, p: "Path") -> QPixmap:
        try:
            ext = p.suffix.lower()
            # Skip work if a job is running
            if getattr(self, "_job_running", False):
                return QPixmap()
            # Images: load via QImageReader (scaled) to avoid huge allocations
            if ext in _IMAGE_EXTS:
                try:
                    from PySide6.QtGui import QImageReader, QPixmap
                    from PySide6.QtCore import QSize
                    reader = QImageReader(str(p))
                    reader.setAutoTransform(True)
                    # Decode to a thumbnail-sized image to keep memory low
                    try:
                        sz = reader.size()
                        max_dim = 512
                        if sz.isValid() and sz.width() > 0 and sz.height() > 0:
                            w, h = sz.width(), sz.height()
                            if w >= h:
                                scaled = QSize(max_dim, max(1, int(h * max_dim / max(1, w))))
                            else:
                                scaled = QSize(max(1, int(w * max_dim / max(1, h))), max_dim)
                        else:
                            scaled = QSize(512, 512)
                        reader.setScaledSize(scaled)
                    except Exception:
                        reader.setScaledSize(QSize(512, 512))
                    img = reader.read()
                    if img and not img.isNull():
                        pm = QPixmap.fromImage(img)
                        if pm and not pm.isNull():
                            return pm
                except Exception:
                    pass
                return QPixmap()
            # Videos: optional, heavier
            if ext in _VIDEO_EXTS:
                try:
                    from pathlib import Path as _P
                    import time, subprocess, os
                    td = _P(".").resolve() / "output" / "_thumbs"
                    td.mkdir(parents=True, exist_ok=True)
                    outp = td / (p.stem + "_thumb.jpg")
                    if not outp.exists() or (time.time() - outp.stat().st_mtime > 24*3600):
                        cmd = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-hwaccel", "none", "-threads", "1", "-v", "error",
                               "-i", str(p), "-vf", "thumbnail,scale=320:-1", "-frames:v", "1", str(outp)]
                        try:
                            cmd = self._apply_streaming_lowmem(cmd)
                        except Exception:
                            pass
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    pm = QPixmap(str(outp))
                    if pm and not pm.isNull():
                        return pm
                except Exception:
                    pass
            return QPixmap()
        except Exception:
            return QPixmap()

    def _update_engine_ui(self):
        # Keep model stack in sync with engine
        eng_txt = (self.combo_engine.currentText() or '').lower()
        page = 0
        if 'gfpgan' in eng_txt:
            page = 6
        elif 'waifu2x' in eng_txt:
            page = 1
        elif 'srmd (ncnn via realesrgan' in eng_txt:
            page = 5
        elif 'ultrasharp' in eng_txt:
            page = 4
        elif 'srmd' in eng_txt:
            page = 2
        elif 'realsr' in eng_txt:
            page = 3
        else:
            page = 0
        self.stk_models.setCurrentIndex(page)
        # GFPGAN: no scale/model auto-guiding here
        if 'gfpgan' in eng_txt:
            try:
                self._update_model_hint()
            except Exception:
                pass
            return
        # Auto-guide scale to the model's native when using Real-ESRGAN-compatible backends
        try:
            eng = (self.combo_engine.currentText() or '').lower()
        except Exception:
            eng = ''
        if ('realesrgan' in eng) or ('real-esrgan' in eng) or ('ultrasharp' in eng) or ('srmd (ncnn via realesrgan' in eng):
            try:
                model = (self.combo_model_w2x.currentText() if "Waifu2x" in engine_label else (
                    getattr(self, "combo_model_ultrasharp", self.combo_model_realsr).currentText() if "UltraSharp" in engine_label else (
                    getattr(self, "combo_model_srmd_realsr", self.combo_model_realsr).currentText() if "SRMD (ncnn via RealESRGAN)" in engine_label else self.combo_model_realsr.currentText()))) if page==0 else (
                        self.combo_model_ultrasharp.currentText() if page==4 else (
                        self.combo_model_srmd_realsr.currentText() if page==5 else self.combo_model_realsr.currentText()))
                t = (model or '').lower()
                native = 4 if ('-x4' in t or 'x4' in t) else (3 if ('-x3' in t or 'x3' in t) else (2 if ('-x2' in t or 'x2' in t) else 2))
                self.spin_scale.blockSignals(True); self.spin_scale.setValue(float(native)); self.spin_scale.blockSignals(False)
                self.slider_scale.blockSignals(True); self.slider_scale.setValue(int(native*10)); self.slider_scale.blockSignals(False)
                for name, want in (('rad_x2', 2), ('rad_x3', 3), ('rad_x4', 4)):
                    try:
                        w = getattr(self, name, None)
                        if w is not None:
                            w.blockSignals(True); w.setChecked(native == want); w.blockSignals(False)
                    except Exception:
                        pass
            except Exception:
                pass
        # Re-apply page
        self.stk_models.setCurrentIndex(page)

    # ---------------------------
    # SeedVR2 UI helpers
    # ---------------------------
    def _seedvr2_refresh_models(self):
        try:
            cur = self.combo_seedvr2_gguf.currentText() if hasattr(self, "combo_seedvr2_gguf") else ""
        except Exception:
            cur = ""
        try:
            items = scan_seedvr2_gguf()
            self.combo_seedvr2_gguf.blockSignals(True)
            self.combo_seedvr2_gguf.clear()
            for it in items:
                self.combo_seedvr2_gguf.addItem(it)
            if cur:
                i = self.combo_seedvr2_gguf.findText(cur)
                if i >= 0:
                    self.combo_seedvr2_gguf.setCurrentIndex(i)
            self.combo_seedvr2_gguf.blockSignals(False)
        except Exception:
            try:
                self.combo_seedvr2_gguf.blockSignals(False)
            except Exception:
                pass

    def _update_seedvr2_warnings(self):
        # Apply resolution-based presets for SeedVR2 advanced settings.
        # 720p: faster/lighter defaults.
        # 1080p+ (1080/1440/2160 etc): enable tiled VAE decode + heavier defaults.
        try:
            self._apply_seedvr2_resolution_preset()
        except Exception:
            pass
        try:
            res = (self.combo_seedvr2_res.currentText() or "").strip()
            self.lbl_seedvr2_2160_warn.setVisible(res == "2160")
        except Exception:
            pass
        try:
            am = (self.combo_seedvr2_attn.currentText() or "").strip().lower()
            self.lbl_seedvr2_attn_warn.setVisible(am in ("xformers", "flash_attn"))
        except Exception:
            pass

    def _apply_seedvr2_resolution_preset(self):
        """Auto-tune SeedVR2 advanced settings based on target resolution.

        Rules:
        - 720p: VAE decode tiled OFF, batch=4, chunk=40
        - 1080p or higher: VAE decode tiled ON, batch=8, chunk=80,
          decode tile size=1024, decode overlap=64
        """
        try:
            res_txt = (self.combo_seedvr2_res.currentText() or "").strip()
            res = int(res_txt) if res_txt.isdigit() else 0
        except Exception:
            res = 0

        # Guard: if widgets don't exist yet
        if not all(hasattr(self, a) for a in (
            "chk_seedvr2_vae_dec",
            "spin_seedvr2_batch",
            "spin_seedvr2_chunk",
            "spin_seedvr2_dec_tile",
            "spin_seedvr2_dec_ov",
        )):
            return

        def _set(widget, fn, value):
            try:
                widget.blockSignals(True)
            except Exception:
                pass
            try:
                fn(value)
            finally:
                try:
                    widget.blockSignals(False)
                except Exception:
                    pass

        if res >= 1080:
            _set(self.chk_seedvr2_vae_dec, self.chk_seedvr2_vae_dec.setChecked, True)
            _set(self.spin_seedvr2_batch, self.spin_seedvr2_batch.setValue, 8)
            _set(self.spin_seedvr2_chunk, self.spin_seedvr2_chunk.setValue, 80)
            _set(self.spin_seedvr2_dec_tile, self.spin_seedvr2_dec_tile.setValue, 1024)
            _set(self.spin_seedvr2_dec_ov, self.spin_seedvr2_dec_ov.setValue, 64)
        elif res == 720:
            _set(self.chk_seedvr2_vae_dec, self.chk_seedvr2_vae_dec.setChecked, False)
            _set(self.spin_seedvr2_batch, self.spin_seedvr2_batch.setValue, 4)
            _set(self.spin_seedvr2_chunk, self.spin_seedvr2_chunk.setValue, 40)

    def _update_seedvr2_mode(self, on: bool | None = None):
        try:
            enabled = bool(self.chk_seedvr2.isChecked()) if on is None else bool(on)
        except Exception:
            enabled = False
        # Show SeedVR2 panel and hide normal controls when enabled
        try:
            self.box_seedvr2.setVisible(enabled)
            self.box_seedvr2.setEnabled(enabled)
        except Exception:
            pass
        for w in (getattr(self, "_w_scale", None), getattr(self, "box_models", None), getattr(self, "box_encoder", None), getattr(self, "box_advanced", None)):
            try:
                if w is None:
                    continue
                w.setVisible(not enabled)
                w.setEnabled(not enabled)
            except Exception:
                pass
        # Keep warnings in sync
        try:
            self._update_seedvr2_warnings()
        except Exception:
            pass

    def _sync_scale_from_spin(self, v: float):
        self.slider_scale.blockSignals(True)
        self.slider_scale.setValue(int(round(v * 10)))
        self.slider_scale.blockSignals(False)
        try:
            self._update_batch_limit()
        except Exception:
            pass

    def _sync_scale_from_slider(self, v: int):
        self.spin_scale.blockSignals(True)
        self.spin_scale.setValue(v / 10.0)
        self.spin_scale.blockSignals(False)
        try:
            self._update_batch_limit()
        except Exception:
            pass


    def _pick_single(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select image or video", str(ROOT))
        if path:
            self.edit_input.setText(path)
            self._append_log(f"Selected input: {path}")

    def _pick_outdir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output folder", str(self.edit_outdir.text() or OUT_VIDEOS)
        )
        if d:
            self.edit_outdir.setText(d)
            self._append_log(f"Output set to: {d}")


    def _view_results(self):
        """Open the current output folder in Media Explorer and trigger a scan."""
        folder = None
        try:
            p = getattr(self, "_last_outfile", None)
            if p:
                pp = Path(p)
                if pp.exists():
                    folder = pp.parent
        except Exception:
            folder = None

        if folder is None:
            try:
                txt = (self.edit_outdir.text() or "").strip()
                if txt:
                    folder = Path(txt)
            except Exception:
                folder = None

        if folder is None:
            # Safe default (videos output folder)
            try:
                folder = Path(OUT_VIDEOS)
            except Exception:
                folder = None

        # Ensure it exists so Media Explorer can scan it
        try:
            if folder is not None:
                folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Prefer the centralized MainWindow handler (switch tab + scan)
        try:
            mw = self.window()
            fn = getattr(mw, "open_media_explorer_folder", None)
            if callable(fn) and folder is not None:
                try:
                    fn(str(folder))
                except TypeError:
                    # In case older signature only accepts path
                    fn(str(folder))
                return
        except Exception:
            pass

        # Fallback: open the folder in the OS file explorer
        try:
            if folder is None:
                return
            if os.name == "nt":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            try:
                QtWidgets.QMessageBox.information(self, "View results", f"Folder:\n{folder}")
            except Exception:
                pass

    def _pick_batch(self):
        # Use the shared BatchSelectDialog (helpers/batch.py); fallback to legacy prompt if missing.
        try:
            from helpers.batch import BatchSelectDialog as _BatchDialog
        except Exception:
            try:
                from helpers.vatch import BatchSelectDialog as _BatchDialog  # optional alias
            except Exception:
                _BatchDialog = None

        if _BatchDialog is not None:
            try:
                # Allow both images and videos
                _exts = sorted(set(getattr(_BatchDialog, "VIDEO_EXTS", set())) | set(getattr(_BatchDialog, "IMAGE_EXTS", set())))
                files, conflict = _BatchDialog.pick(self, title="Add Batch", exts=_exts)
            except Exception:
                files, conflict = None, None

            if files is None:
                # cancelled
                return
            if not files:
                QtWidgets.QMessageBox.information(self, "No media", "No files selected.")
                return

            # map dialog result to internal duplicate mode
            if conflict in ("version", "autorename", "auto", "ver"):
                dup_mode = "version"
            elif conflict == "overwrite":
                dup_mode = "overwrite"
            else:
                dup_mode = "skip"

            # store for _build_outfile behavior (restored at end)
            prev_dup = getattr(self, "_dup_mode", None)
            try:
                self._dup_mode = dup_mode
            except Exception:
                pass

            # If skip: pre-filter files whose default outfile already exists
            to_run = []
            skipped = 0
            try:
                sc = int(round(float(self.spin_scale.value())))
            except Exception:
                sc = 2
            for f in files:
                p = Path(f)
                try:
                    is_video = p.suffix.lower() in _VIDEO_EXTS
                except Exception:
                    try:
                        from pathlib import Path as _P
                        is_video = _P(f).suffix.lower() in _VIDEO_EXTS
                    except Exception:
                        is_video = False
                try:
                    outd = Path(self.edit_outdir.text().strip()) if getattr(self, "edit_outdir", None) and (self.edit_outdir.text().strip()) else (OUT_VIDEOS if is_video else OUT_SHOTS)
                except Exception:
                    outd = OUT_VIDEOS if is_video else OUT_SHOTS
                # Compute the *default* name (no versioning) for skip check
                default_out = outd / f"{p.stem}_x{sc}{p.suffix}"
                if dup_mode == "skip" and default_out.exists():
                    skipped += 1
                    continue
                to_run.append(p)

            if not to_run:
                QtWidgets.QMessageBox.information(self, "Nothing to do", "All selected outputs already exist.")
                # restore dup mode
                if prev_dup is not None:
                    self._dup_mode = prev_dup
                else:
                    try: delattr(self, "_dup_mode")
                    except Exception: pass
                return

            self._run_batch_files(to_run)

            # restore dup mode after queuing
            if prev_dup is not None:
                self._dup_mode = prev_dup
            else:
                try: delattr(self, "_dup_mode")
                except Exception: pass
            if skipped:
                try:
                    self._append_log(f"[batch] Skipped {skipped} existing output(s).")
                except Exception:
                    pass
            return
        # ----- Legacy fallback (original UI) -----
        dlg = QtWidgets.QMessageBox(self)
        dlg.setWindowTitle("Batch input")
        dlg.setText("Pick a folder or choose files.")
        btn_folder = dlg.addButton("Folder…", QtWidgets.QMessageBox.AcceptRole)
        btn_files = dlg.addButton("Files…", QtWidgets.QMessageBox.ActionRole)
        btn_cancel = dlg.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        dlg.setDefaultButton(btn_files)
        dlg.exec()
        if dlg.clickedButton() is btn_cancel:
            self._append_log("Batch canceled.")
            return
        if dlg.clickedButton() is btn_folder:
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder")
            if not d:
                self._append_log("Batch canceled.")
                return
            self._run_batch_from_folder(Path(d))
            return
        filt = "Media files (*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.ts *.m2ts *.wmv *.flv *.mpg *.mpeg *.3gp *.3g2 *.ogv *.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff *.gif);;All files (*)"
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Select files", "", filt)
        if files:
            self._run_batch_files([Path(f) for f in files])


    def _run_batch_from_folder(self, folder: Path):
        exts = _IMAGE_EXTS | _VIDEO_EXTS
        files = [p for p in folder.iterdir() if p.suffix.lower() in exts]
        if not files:
            QtWidgets.QMessageBox.information(self, "No media", "That folder has no supported images or videos.")
            return
        # Confirmation dialog before queuing
        count = len(files)
        resp = QtWidgets.QMessageBox.question(
            self,
            "Confirm queue",
            f"Queue {count} files?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes
        )
        if resp != QtWidgets.QMessageBox.Yes:
            self._append_log("Batch canceled.")
            return
        for p in files:
            self._run_one(p)

    def _run_batch_files(self, files: List[Path]):
        # Confirmation dialog before queuing
        count = len(files)
        if count <= 0:
            QtWidgets.QMessageBox.information(self, "No media", "No files selected.")
            return
        resp = QtWidgets.QMessageBox.question(
            self,
            "Confirm queue",
            f"Queue {count} files?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Yes
        )
        if resp != QtWidgets.QMessageBox.Yes:
            self._append_log("Batch canceled.")
            return
        for p in files:
            self._run_one(p)
    def _build_outfile(self, src: Path, outd: Path, scale: int) -> Path:
        outd.mkdir(parents=True, exist_ok=True)
        base = outd / f"{src.stem}_x{scale}{src.suffix}"
        mode = getattr(self, "_dup_mode", "overwrite")
        if mode == "version":
            i = 1
            cand = base
            while cand.exists():
                cand = outd / f"{src.stem}_x{scale} ({i}){src.suffix}"
                i += 1
            return cand
        return base
    def _sanitize_realsr_model(self, model: str, scale: int) -> str:
        """Best-effort cleanup for Real-ESRGAN model names.
        Fixes common issues:
          - model accidentally passed as a full path
          - model auto-suffixed with _x{scale} / -x{scale} when only the base file exists
        """
        n = (model or "").strip().strip('"').strip("'")
        # If it looks like a path, keep only the basename (without extension)
        try:
            if ":" in n or "/" in n or "\\" in n:
                base = os.path.basename(n.replace("\\", "/"))
                n = os.path.splitext(base)[0]
        except Exception:
            pass

        # If the selected model doesn't exist, but a stripped suffix variant does, fall back.
        try:
            cand_param = REALSR_DIR / f"{n}.param"
            cand_bin = REALSR_DIR / f"{n}.bin"
            if (not cand_param.exists()) and (not cand_bin.exists()):
                mm = re.match(r"^(.*?)(?:[_-]x([234]))$", n, flags=re.IGNORECASE)
                if mm:
                    base = (mm.group(1) or "").strip()
                    if base:
                        b_param = REALSR_DIR / f"{base}.param"
                        b_bin = REALSR_DIR / f"{base}.bin"
                        if b_param.exists() or b_bin.exists():
                            n = base
        except Exception:
            pass

        return n

    def _realsr_cmd_dir(self, exe: str, indir: Path, outdir: Path, model: str, scale: int) -> List[str]:
        n = (model or "").strip()
        n = self._sanitize_realsr_model(n, scale)
        cmd = [exe, "-i", str(indir), "-o", str(outdir), "-n", n, "-s", str(scale), "-m", str(REALSR_DIR), "-f", "png"]
        # Force GPU selection for Real-ESRGAN (ncnn-vulkan).
        # Default to GPU 0 (usually the discrete NVIDIA GPU). Override with FV_REALSR_GPU_ID.
        try:
            gid = os.environ.get("FV_REALSR_GPU_ID", "").strip() or "0"
            cmd += ["-g", str(gid)]
        except Exception:
            pass
        # Optional thread tuning: FV_REALSR_JOBS like "1:2:2" (load:proc:save).
        try:
            jobs = os.environ.get("FV_REALSR_JOBS", "").strip()
            if jobs:
                cmd += ["-j", str(jobs)]
        except Exception:
            pass
        try:
            t = int(self.spin_tile.value()) if hasattr(self, "spin_tile") else 0
            if t > 0:
                cmd += ["-t", str(t)]
        except Exception:
            pass
        return cmd

    def _realsr_cmd_file(self, exe: str, infile: Path, outfile: Path, model: str, scale: int) -> List[str]:
        n = (model or "").strip()
        n = self._sanitize_realsr_model(n, scale)
        cmd = [exe, "-i", str(infile), "-o", str(outfile), "-n", n, "-s", str(scale), "-m", str(REALSR_DIR)]
        # Force GPU selection for Real-ESRGAN (ncnn-vulkan).
        # Default to GPU 0 (usually the discrete NVIDIA GPU). Override with FV_REALSR_GPU_ID.
        try:
            gid = os.environ.get("FV_REALSR_GPU_ID", "").strip() or "0"
            cmd += ["-g", str(gid)]
        except Exception:
            pass
        # Optional thread tuning: FV_REALSR_JOBS like "1:2:2" (load:proc:save).
        try:
            jobs = os.environ.get("FV_REALSR_JOBS", "").strip()
            if jobs:
                cmd += ["-j", str(jobs)]
        except Exception:
            pass
        try:
            t = int(self.spin_tile.value()) if hasattr(self, "spin_tile") else 0
            if t > 0:
                cmd += ["-t", str(t)]
        except Exception:
            pass
        return cmd

    def _waifu_cmd_file(self, exe: str, infile: Path, outfile: Path, model_dirname: str, scale: int) -> List[str]:
        return [exe, "-i", str(infile), "-o", str(outfile), "-m", str(WAIFU2X_DIR / model_dirname), "-s", str(scale)]
    def _do_single(self):
        raw = (self.edit_input.text() or '').strip()
        src = Path(raw)
        if raw in ('.','./','..',''):
            # Try to fall back to player path if available
            try:
                p = _fv_get_input(self)
                if p:
                    src = Path(p)
            except Exception:
                pass
        if not src.exists():
            QtWidgets.QMessageBox.warning(self, "No input", "Please choose an image or video.")
            return
        if src.is_dir():
            self._run_batch_from_folder(src)
            return
        self._run_one(src)


    def _run_one(self, src: Path):
        # Make current file visible to the queue shim during batch
        try:
            self._last_infile = src
        except Exception:
            pass
        try:
            if hasattr(self, 'edit_input') and hasattr(self.edit_input, 'setText'):
                self.edit_input.setText(str(src))
        except Exception:
            pass
        self._job_running = True
        try:
            ext = src.suffix.lower()
            is_video = ext in _VIDEO_EXTS
            is_image = ext in _IMAGE_EXTS
    
            # SeedVR2 path (overrides all other engines when enabled)
            try:
                if getattr(self, "chk_seedvr2", None) is not None and self.chk_seedvr2.isChecked():
                    # Basic validation
                    if not SEEDVR2_ENV_PY.exists():
                        QtWidgets.QMessageBox.critical(self, "SeedVR2 missing", f"SeedVR2 environment python was not found:\n{SEEDVR2_ENV_PY}")
                        return
                    if not SEEDVR2_CLI.exists():
                        QtWidgets.QMessageBox.critical(self, "SeedVR2 missing", f"SeedVR2 CLI was not found:\n{SEEDVR2_CLI}")
                        return
                    if not SEEDVR2_MODELS_DIR.exists():
                        QtWidgets.QMessageBox.critical(self, "SeedVR2 missing", f"SeedVR2 model folder was not found:\n{SEEDVR2_MODELS_DIR}")
                        return

                    gguf_name = (self.combo_seedvr2_gguf.currentText() or "").strip()
                    if not gguf_name:
                        QtWidgets.QMessageBox.warning(self, "SeedVR2", "Please select a GGUF model.")
                        return
                    gguf_path = None
                    try:
                        for p in SEEDVR2_MODELS_DIR.rglob("*.gguf"):
                            if p.name == gguf_name:
                                gguf_path = p
                                break
                    except Exception:
                        gguf_path = None
                    if gguf_path is None or not gguf_path.exists():
                        QtWidgets.QMessageBox.critical(self, "SeedVR2 missing", f"Selected GGUF model was not found in models/SEEDVR2:\n{gguf_name}")
                        return

                    res = int((self.combo_seedvr2_res.currentText() or "1440").strip() or 1440)
                    if is_video and res >= 2160:
                        QtWidgets.QMessageBox.warning(self, "SeedVR2", "2160 is intended for image upscaling only. Please choose 1440 or lower for videos.")
                        return

                    outd = Path(self.edit_outdir.text().strip()) if self.edit_outdir.text().strip() else (OUT_VIDEOS if is_video else OUT_SHOTS)
                    outd.mkdir(parents=True, exist_ok=True)

                    # Output naming (no overwrites):
                    # <orig15>_seedVR2_<res>_<yymmdd>.ext
                    # If the file exists, add _01, _02, ...
                    try:
                        raw_stem = (src.stem or "video")[:15]
                        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_stem).strip("_")
                        if not safe:
                            safe = "video"
                        stamp = time.strftime("%y%m%d", time.localtime())
                        tag = f"{safe}_seedVR2_{res}_{stamp}"
                    except Exception:
                        tag = f"{src.stem}_seedVR2_{res}"

                    if is_video:
                        out_fmt = "mp4"
                        outfile = outd / f"{tag}.mp4"
                    else:
                        out_fmt = "png"
                        outfile = outd / f"{tag}.png"

                    # Ensure we never overwrite previous results
                    try:
                        if outfile.exists():
                            n = 1
                            while True:
                                cand = outd / f"{tag}_{n:02d}{outfile.suffix}"
                                if not cand.exists():
                                    outfile = cand
                                    break
                                n += 1
                    except Exception:
                        pass

                    self._last_outfile = outfile

                    temporal = 1 if bool(self.chk_seedvr2_temporal.isChecked()) else 0
                    cc = (self.combo_seedvr2_color.currentText() or "").strip() or "lab"
                    attn = (self.combo_seedvr2_attn.currentText() or "sdpa").strip() or "sdpa"

                    seed_forward = [
                           "--output", str(outfile),
                           "--output_format", out_fmt,
                           "--video_backend", "ffmpeg",
                           "--model_dir", str(SEEDVR2_MODELS_DIR),
                           "--dit_model", gguf_path.name,
                           "--resolution", str(res),
                           "--batch_size", str(int(self.spin_seedvr2_batch.value())),
                           "--chunk_size", str(int(self.spin_seedvr2_chunk.value())),
                           "--temporal_overlap", str(int(temporal)),
                           "--color_correction", str(cc),
                           "--attention_mode", str(attn),
                           ]

                    # Offload options (tiled VAE)
                    try:
                        if self.chk_seedvr2_vae_enc.isChecked():
                            seed_forward.append("--vae_encode_tiled")
                            seed_forward += ["--vae_encode_tile_size", str(int(self.spin_seedvr2_enc_tile.value()))]
                            seed_forward += ["--vae_encode_tile_overlap", str(int(self.spin_seedvr2_enc_ov.value()))]
                        if self.chk_seedvr2_vae_dec.isChecked():
                            seed_forward.append("--vae_decode_tiled")
                            seed_forward += ["--vae_decode_tile_size", str(int(self.spin_seedvr2_dec_tile.value()))]
                            seed_forward += ["--vae_decode_tile_overlap", str(int(self.spin_seedvr2_dec_ov.value()))]
                    except Exception:
                        pass

                    # Env: make sure FrameVision bins are on PATH (ffmpeg backend)
                    env = os.environ.copy()
                    try:
                        if os.name == "nt":
                            env["PATH"] = str(PRESETS_BIN) + ";" + env.get("PATH", "")
                        else:
                            env["PATH"] = str(PRESETS_BIN) + ":" + env.get("PATH", "")
                    except Exception:
                        pass
                    env.setdefault("PYTHONUTF8", "1")
                    env.setdefault("PYTHONIOENCODING", "utf-8")

                    runner = _seedvr2_runner_path()
                    if runner:
                        cmd = [str(SEEDVR2_ENV_PY), "-X", "utf8", str(runner),
                               "--cli", str(SEEDVR2_CLI),
                               "--input", str(src),
                               "--ffmpeg", str(FFMPEG),
                               "--ffprobe", str(FFPROBE),
                               "--work_root", str(ROOT),
                               "--"] + seed_forward
                    else:
                        # Fallback to direct CLI call if runner is missing.
                        cmd = [str(SEEDVR2_ENV_PY), "-X", "utf8", str(SEEDVR2_CLI), str(src)] + seed_forward

                    self._append_log("Engine: SeedVR2")
                    self._append_log(f"Python: {SEEDVR2_ENV_PY}")
                    self._append_log(f"CLI: {SEEDVR2_CLI}")
                    self._append_log(f"Runner: {runner if runner else '(missing — using CLI directly)'}")
                    self._append_log(f"Model dir: {SEEDVR2_MODELS_DIR}")
                    self._append_log(f"GGUF: {gguf_path.name}")
                    self._append_log(f"Upscale to: {res}")
                    self._append_log(f"Temporal overlap: {temporal}")
                    self._append_log(f"Color correction: {cc}")
                    self._append_log(f"Attention: {attn}")
                    self._append_log(f"Input: {src}")
                    self._append_log(f"Output: {outfile}")

                    self._run_cmd([cmd], open_on_success=True, cwd=SEEDVR2_CLI.parent, env=env)
                    return
            except Exception:
                # If SeedVR2 block fails unexpectedly, fall back to normal engines
                pass

            engine_label = self.combo_engine.currentText()
            do_gfpgan = ("gfpgan" in (engine_label or "").lower())
            engine_exe = self.combo_engine.currentData()
            if not engine_exe or not _exists(engine_exe):
                QtWidgets.QMessageBox.critical(
                    self, "Engine not found",
                    f"The selected engine executable was not found:\n{engine_exe}\n\nCheck your models folder paths."
                )
                self._append_log(f"✖ Engine missing: {engine_exe}")
                return
    
            scale = int(round(float(self.spin_scale.value())))
            if do_gfpgan:
                scale = 1
            scale = max(1, min(8, scale))
            outd = Path(self.edit_outdir.text().strip()) if self.edit_outdir.text().strip() else (OUT_VIDEOS if is_video else OUT_SHOTS)
            outfile = self._build_outfile(src, outd, scale)
            self._last_outfile = outfile
    
            if is_video:
                if "Waifu2x" in engine_label:
                    QtWidgets.QMessageBox.information(self, "Not supported", "Waifu2x (ncnn) handles images only for now. Please select Real-ESRGAN for videos.")
                    self._append_log("Waifu2x selected for a video — blocked (images only).")
                    return

                if do_gfpgan:
                    QtWidgets.QMessageBox.information(self, "Not supported", "GFPGAN is a face restorer for images only (no video pipeline here yet). Please select a Real-ESRGAN engine for videos.")
                    self._append_log("GFPGAN selected for a video — blocked (images only).")
                    return
    
                # Pick model from the currently active engine/page (important for UltraSharp/SRMD, etc.)
                try:
                    model = (self._fv_current_model_text() or "").strip()
                except Exception:
                    model = ""
                if not model:
                    try:
                        model = self.combo_model_realsr.currentText()
                    except Exception:
                        model = ""
                fps = _parse_fps(src) or "30"
                work = outd / f"{src.stem}_x{scale}_work"
                in_dir = work / "in"
                out_dir = work / "out"
                in_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                seq_in = in_dir / "f_%08d.png"
                seq_out = out_dir / "f_%08d.png"
    
                pre = self._build_pre_filters()
                cmd_extract = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(src), "-map", "0:v:0"]
                if pre: cmd_extract += ["-vf", pre]
                cmd_extract += ["-fps_mode", "vfr", str(seq_in)]
                cmd_upscale = self._realsr_cmd_dir(engine_exe, in_dir, out_dir, model, scale)
                post = self._build_post_filters()
                cmd_encode = [FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-framerate", fps, "-i", str(seq_out), "-i", str(src), "-map", "0:v:0"]
                if self.radio_a_mute.isChecked():
                    pass
                else:
                    cmd_encode += ["-map", "1:a?"]
                vcodec = self.combo_vcodec.currentText()
                cmd_encode += ["-c:v", vcodec, "-pix_fmt", "yuv420p"]
                cmd_encode += ["-vsync", "cfr"]
                if self.rad_crf.isChecked():
                    cmd_encode += ["-crf", str(self.spin_crf.value())]
                else:
                    cmd_encode += ["-b:v", f"{self.spin_bitrate.value()}k"]
                preset = self.combo_preset.currentText()
                if preset:
                    cmd_encode += ["-preset", preset]
                if int(self.spin_keyint.value() or 0) > 0:
                    cmd_encode += ["-g", str(int(self.spin_keyint.value()))]
                if post:
                    cmd_encode += ["-vf", post]
                
                
                if self.radio_a_mute.isChecked():
                    cmd_encode += ["-an"]
                elif self.radio_a_copy.isChecked():
                    cmd_encode += ["-c:a", "copy"]
                else:
                    cmd_encode += ["-c:a", self.combo_acodec.currentText(), "-b:a", f"{self.combo_abitrate.currentText()}k"]
                cmd_encode += ["-r", fps, "-shortest", str(outfile)]
    
                self._append_log(f"Engine: {engine_label}")
                self._append_log(f"Executable: {engine_exe}")
                self._append_log(f"Model: {model}")
                self._append_log(f"Model dir: {REALSR_DIR}")
                self._append_log(f"Scale: x{scale}")
                self._append_log(f"FPS: {fps}")
                self._append_log(f"Work dir: {work}")
                self._append_log(f"Output: {outfile}")
    
                self._run_cmd([cmd_extract, cmd_upscale, cmd_encode], open_on_success=True, cleanup_dirs=[work])
                return
    
            # Image path
            if do_gfpgan:
                try:
                    mname = self.combo_model_gfpgan.currentText() if hasattr(self, "combo_model_gfpgan") else "GFPGANv1.4"
                except Exception:
                    mname = "GFPGANv1.4"
                mlow = (mname or "").lower()
                weights = GFPGAN_MODEL_V13 if ("1.3" in mlow) else GFPGAN_MODEL_V14
                model = (mname or 'GFPGANv1.4').strip() or 'GFPGANv1.4'
                script = ROOT / "helpers" / "gfpgan_cli.py"
                cmd = [engine_exe, str(script), "--in", str(src), "--out", str(outfile), "--model", str(weights), "--device", "auto"]
            elif "Waifu2x" in engine_label:
                model = self.combo_model_w2x.currentText()
                cmd = self._waifu_cmd_file(engine_exe, src, outfile, model, scale)
            elif "UltraSharp" in engine_label:
                model = getattr(self, "combo_model_ultrasharp", self.combo_model_realsr).currentText()
                cmd = self._realsr_cmd_file(engine_exe, src, outfile, model, scale)
            elif "SRMD (ncnn via RealESRGAN)" in engine_label:
                model = getattr(self, "combo_model_srmd_realsr", self.combo_model_realsr).currentText()
                cmd = self._realsr_cmd_file(engine_exe, src, outfile, model, scale)
            else:
                model = self.combo_model_realsr.currentText()
                cmd = self._realsr_cmd_file(engine_exe, src, outfile, model, scale)
    
            self._append_log(f"Engine: {engine_label}")
            self._append_log(f"Executable: {engine_exe}")
            self._append_log(f"Model: {model}")
            if do_gfpgan:
                try:
                    self._append_log(f"Model dir: {GFPGAN_DIR}")
                    self._append_log(f"Weights: {weights}")
                except Exception:
                    pass
            elif "Waifu2x" not in engine_label:
                self._append_log(f"Model dir: {REALSR_DIR}")
            self._append_log(f"Scale: x{scale}")
            self._append_log(f"Input: {src}")
            self._append_log(f"Output: {outfile}")
            self._run_cmd([cmd], open_on_success=True)
        finally:
            self._job_running = False

    
    def _run_cmd(self, cmds: List[List[str]], open_on_success: bool = False, cleanup_dirs: Optional[List[Path]] = None, cwd: Optional[Path] = None, env: Optional[dict] = None):
            self.btn_upscale.setEnabled(False)
            self.btn_batch.setEnabled(False)
            self._thr = _RunThread(cmds, cwd=(cwd or ROOT), env=env, parent=self)
            self._thr.progress.connect(self._append_log)

            def on_done(code: int, last: str):
                self.btn_upscale.setEnabled(True)
                self.btn_batch.setEnabled(True)
                if code == 0:
                    self._append_log("✔ Done.")
                    if self._last_outfile and self._last_outfile.exists():
                        self._append_log(f"Saved: {self._last_outfile}")
                    if cleanup_dirs:
                        for d in cleanup_dirs:
                            try:
                                shutil.rmtree(d, ignore_errors=True)
                            except Exception:
                                pass
                else:
                    self._append_log(f"✖ Finished with code {code}")

            self._thr.done.connect(on_done)
            self._thr.start()

    @QtCore.Slot(str)
    def _append_log(self, line: str):
        try:
            # Fast append at end; avoids O(n) inserts at start
            self.log.appendPlainText(line)
            self.log.ensureCursorVisible()
        except Exception:
            try:
                cur = self.log.textCursor()
                cur.movePosition(cur.End)
                cur.insertText(line + "\n")
                self.log.setTextCursor(cur)
                self.log.ensureCursorVisible()
            except Exception:
                # Last-resort fallback (avoid expensive full setPlainText on large logs)
                pass
# Build pre-extract filters (applied on input video before frame extraction)
    def _build_pre_filters(self) -> str:
        fs = []
        try:
            if self.chk_deinterlace.isChecked():
                fs.append("yadif=mode=1:parity=-1:deint=0")
            if self.chk_deblock.isChecked():
                fs.append("pp=deblock")
            if self.chk_denoise.isChecked():
                fs.append("hqdn3d=3:3:6:6")
            if self.chk_deband.isChecked():
                fs.append("gradfun=thr=0.6")
        except Exception:
            pass
        return ",".join(fs)

    # Build post-encode filters (after upscale, on image sequence)
    def _build_post_filters(self) -> str:
        fs = []
        try:
            rng = self.combo_range.currentText()
            if rng == "Full→Limited":
                fs.append("scale=in_range=pc:out_range=tv")
            elif rng == "Limited→Full":
                fs.append("scale=in_range=tv:out_range=pc")
            val = int(self.sld_sharpen.value()) if hasattr(self, "sld_sharpen") else 0
            if val > 0:
                amt = round(1.5 * (val / 100.0), 3)
                fs.append(f"unsharp=luma_msize_x=7:luma_msize_y=7:luma_amount={amt}")
        except Exception:
            pass
        return ",".join(fs)


    def _play_in_player(self, p: Path):

        m = getattr(self, "_main", None) or getattr(self, "main", None)
        try:
            if m is not None and hasattr(m, "video") and hasattr(m.video, "open"):
                m.video.open(p)
                try:
                    # Keep MainWindow's notion of current media in sync
                    m.current_path = p
                except Exception:
                    pass
                try:
                    # Force-refresh the Info dialog (if open)
                    refresh_info_now(p)
                except Exception:
                    pass
                return True
        except Exception:
            pass
        try:
            self._open_file(p)
            return True
        except Exception:
            return False
    def _show_media_info(self):
        src_path = self.edit_input.text().strip()
        if not src_path:
            QtWidgets.QMessageBox.information(self, "Info", "No input selected.")
            return
        self._show_media_info_for(src_path)

    def _show_media_info_for(self, src_path):
        src_path = (str(src_path) or "").strip()
        if not src_path:
            QtWidgets.QMessageBox.information(self, "Info", "No media selected.")
            return
        try:
            cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", src_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = proc.communicate()
            info = out.decode("utf-8", errors="ignore") if out else "{}"
        except Exception as e:
            info = json.dumps({"error": str(e)}, indent=2)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Media info")
        lay = QtWidgets.QVBoxLayout(dlg)
        te = QtWidgets.QPlainTextEdit(dlg); te.setReadOnly(True); te.setPlainText(info)
        lay.addWidget(te)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, parent=dlg)
        btns.rejected.connect(dlg.reject); btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.resize(700, 600)
        dlg.exec()

    def _auto_tile_size(self):
        exe = self.combo_engine.currentData()
        if not exe or "realesrgan" not in str(exe).lower():
            QtWidgets.QMessageBox.information(self, "Auto tile", "Auto tile size works with Real-ESRGAN (ncnn) engine.")
            return
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="tileprobe_"))
        try:
            test_in = tmp / "in.png"; test_out = tmp / "out.png"
            subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "warning", "-y", "-f", "lavfi", "-i", "color=c=gray:s=256x256", str(test_in)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            model = self.combo_model_realsr.currentText(); scale = int(round(self.spin_scale.value()))
            candidates = [800, 600, 400, 300, 200, 100]
            ok = None
            for t in candidates:
                cmd = [exe, "-i", str(test_in), "-o", str(test_out), "-n", (model[:-3] if model.endswith(('-x2','-x3','-x4')) else model), "-s", str(scale), "-m", str(REALSR_DIR), "-t", str(t)]
                try:
                    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
                    if r.returncode == 0 and test_out.exists():
                        ok = t; break
                except Exception:
                    pass
            if ok is None:
                self.lbl_tile_result.setText("Probe failed; using engine default.")
                self.spin_tile.setValue(0)
            else:
                self.spin_tile.setValue(ok)
                self.lbl_tile_result.setText(f"Chosen tile: {ok}")
        finally:
            try:
                for p in tmp.glob("*"): p.unlink()
                tmp.rmdir()
            except Exception:
                pass

    def _update_model_hint(self):
        try:
            name = self.combo_model_realsr.currentText().lower()
        except Exception:
            return
        cat, hint = "Other", ""
        if "anime" in name:
            cat, hint = "Anime", "Optimized for line art / anime; preserves edges with minimal ringing."
        elif "ultrasharp" in name or "ui" in name or "text" in name:
            cat, hint = "UI/Text", "Very crisp edges; can halo UI/text. Good for logos and screen captures."
        elif "general" in name or "x4plus" in name:
            cat, hint = "Photo", "Balanced detail for photographs; natural textures."
        try:
            self.lbl_model_badge.setText(cat)
        except Exception:
            pass
        try:
            self.lbl_model_hint.setText(hint)
        except Exception:
            pass

def _open_file(self, p: Path):
        try:
            if os.name == "nt":
                os.startfile(str(p))  # nosec - user initiated
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception:
            pass


# === FrameVision patch: queue + player wiring (adaptive queue args) ===
from pathlib import Path as _FVPath
import inspect as _FVinspect
try:
    from PySide6 import QtWidgets as _FVQtW  # type: ignore
except Exception:  # pragma: no cover
    _FVQtW = None  # type: ignore

def _fv_is_valid_file(p):
    try:
        return p and _FVPath(p).exists() and _FVPath(p).is_file()
    except Exception:
        return False

def _fv_guess_player_path(owner):
    m = getattr(owner, "_main", None) or getattr(owner, "main", None)
    candidates = []
    if m:
        candidates += [
            getattr(m, "current_path", None),
            getattr(m, "current_file", None),
            getattr(m, "current_media", None),
            getattr(getattr(m, "player", None), "current_path", None),
            getattr(getattr(m, "player", None), "source", None),
            getattr(getattr(m, "viewer", None), "current_path", None),
        ]
    seen = set(); cand = []
    for c in candidates:
        s = str(c) if c is not None else None
        if s and s not in seen:
            seen.add(s); cand.append(s)
    for c in cand:
        if _fv_is_valid_file(c):
            return c
    return None

def _fv_get_input(self):
    # try text fields first
    for attr in ("edit_input","line_input","edit_in","input_line","editPath","linePath"):
        w = getattr(self, attr, None)
        try:
            if w and hasattr(w, "text"):
                t = (w.text() or '').strip()
                if t in ('.','./','..',''):
                    pass
                elif _fv_is_valid_file(t):
                    return t
        except Exception:
            pass
    # then stored infile
    try:
        p = getattr(self, "_last_infile", None)
        if p and _fv_is_valid_file(str(p)):
            return str(p)
    except Exception:
        pass
    # finally ask player
    return _fv_guess_player_path(self)

def _fv_push_input_to_tab(self, path_str):
    pushed = False
    for attr in ("edit_input","line_input","edit_in","input_line","editPath","linePath"):
        w = getattr(self, attr, None)
        try:
            if w and hasattr(w, "setText"):
                w.setText(path_str); pushed = True
        except Exception:
            pass
    for meth in ("set_input_path","load_single_input","set_source","set_path"):
        fn = getattr(self, meth, None)
        try:
            if callable(fn):
                fn(path_str); pushed = True
        except Exception:
            pass
    try:
        self._last_infile = _FVPath(path_str)
    except Exception:
        pass
    return pushed

def _fv_find_enqueue(self):
    """Return a callable that accepts (job_dict) OR (input_path, out_dir, factor, model), plus a label."""
    m = getattr(self, "_main", None) or getattr(self, "main", None)

    candidates = []
    # direct
    for name in ("enqueue", "enqueue_job", "enqueue_external", "enqueue_single_action", "queue_add", "add_job"):
        fn = getattr(m, name, None) if m is not None else None
        if callable(fn): candidates.append((fn, f"main.{name}"))
    # queue_adapter on main
    qa = getattr(m, "queue_adapter", None) if m is not None else None
    for name in ("enqueue", "add", "put"):
        fn = getattr(qa, name, None) if qa is not None else None
        if callable(fn): candidates.append((fn, f"main.queue_adapter.{name}"))
    # nested queue
    q = getattr(m, "queue", None) if m is not None else None
    for name in ("enqueue", "add", "put", "add_job"):
        fn = getattr(q, name, None) if q is not None else None
        if callable(fn): candidates.append((fn, f"main.queue.{name}"))
    # module-level
    try:
        import helpers.queue_adapter as _qa  # type: ignore
        for name in ("enqueue", "add", "put", "add_job"):
            fn = getattr(_qa, name, None)
            if callable(fn): candidates.append((fn, f"helpers.queue_adapter.{name}"))
    except Exception:
        pass

    # pick best (prefer queue_adapter.enqueue if it exists)
    lbl_order = ["helpers.queue_adapter.enqueue", "main.queue_adapter.enqueue", "main.enqueue", "main.enqueue_job"]
    for prefer in lbl_order:
        for fn, label in candidates:
            if label.endswith(prefer):
                return fn, label
    return candidates[0] if candidates else (None, "")

def _fv_call_enqueue(self, enq, where_label, cmds, open_on_success, **_kwargs):
    """Try to call enqueue either with a job dict, or with signature (input_path, out_dir, factor, model)."""
    sig = None
    try:
        sig = _FVinspect.signature(enq)
    except Exception:
        pass

    # Collect context
    input_path = _fv_get_input(self)
    out_dir = ""
    try:
        out_dir = (getattr(self, 'edit_outdir', None).text() if getattr(self, 'edit_outdir', None) else '') or (getattr(self, 'edit_output', None).text() if getattr(self, 'edit_output', None) else '')
    except Exception:
        pass
    # factor: try to read from UI or model name
    factor = 2
    try:
        factor = int(round(float(getattr(self, 'spin_scale', None).value())))
    except Exception:
        try:
            txt = getattr(self, 'combo_model', None).currentText()
            if isinstance(txt, str) and 'x4' in txt: factor = 4
            elif isinstance(txt, str) and 'x3' in txt: factor = 3
            elif isinstance(txt, str) and 'x2' in txt: factor = 2
        except Exception:
            pass
    # Model: prefer helper that knows which dropdown is active.
    # This prevents the "always picks the first model" bug when multiple model combos exist.
    model_name = ''
    try:
        fn = getattr(self, "_fv_current_model_text", None)
        if callable(fn):
            model_name = str(fn() or "").strip()
    except Exception:
        model_name = ''

    if not model_name:
        # Fallback: choose by stacked-models page (what the user is actually looking at)
        try:
            stk = getattr(self, "stk_models", None)
            page = int(stk.currentIndex()) if stk is not None else -1
        except Exception:
            page = -1

        def _ct(attr: str) -> str:
            w = getattr(self, attr, None)
            try:
                return str(w.currentText()).strip() if (w is not None and hasattr(w, "currentText")) else ""
            except Exception:
                return ""

        if page == 1:
            model_name = _ct("combo_model_w2x")
        elif page == 2:
            model_name = _ct("combo_model_srmd")
        elif page == 3:
            model_name = _ct("combo_model_realsr_ncnn")
        elif page == 4:
            model_name = _ct("combo_model_ultrasharp")
        elif page == 5:
            model_name = _ct("combo_model_srmd_realsr")
        elif page == 6:
            model_name = _ct("combo_model_gfpgan")
        else:
            # page 0 or unknown: use engine label mapping as best-effort
            eng_label = _ct("combo_engine")
            if "waifu2x" in eng_label.lower():
                model_name = _ct("combo_model_w2x")
            elif "ultrasharp" in eng_label.lower():
                model_name = _ct("combo_model_ultrasharp")
            elif "srmd (ncnn via realesrgan" in eng_label.lower():
                model_name = _ct("combo_model_srmd_realsr")
            elif "srmd" in eng_label.lower():
                model_name = _ct("combo_model_srmd")
            elif "realsr" in eng_label.lower():
                model_name = _ct("combo_model_realsr_ncnn")
            elif "gfpgan" in eng_label.lower():
                model_name = _ct("combo_model_gfpgan")
            else:
                model_name = _ct("combo_model_realsr") or _ct("combo_model")

        if not model_name:
            model_name = _ct("combo_model_realsr") or _ct("combo_model")


    
    # SeedVR2: when enabled, never use the 4-arg enqueue signature (it would lose the custom CLI cmd).
    # Always enqueue a job dict that contains the exact command + cwd/env.
    try:
        if getattr(self, "chk_seedvr2", None) is not None and getattr(self, "chk_seedvr2").isChecked():
            if cmds:
                c0 = cmds[0]
                txt0 = " ".join(map(str, c0)).lower()
                if "seedvr2" in txt0 or "inference_cli.py" in txt0:
                    job = {
                        "name": "Upscale (seedVR2)",
                        "category": "upscale",
                        "engine": "seedvr2",
                        "cmd": c0,
                        "cwd": str((_kwargs.get("cwd") or globals().get("ROOT", "."))),
                        "open_on_success": bool(open_on_success),
                        "output": str(getattr(self, "_last_outfile", "")),
                    }
                    # pass env through if the queue/worker supports it
                    try:
                        if _kwargs.get("env"):
                            job["env"] = _kwargs.get("env")
                    except Exception:
                        pass
                    try:
                        enq(job)
                        try:
                            self._append_log(f"Queued SeedVR2 via {where_label}.")
                        except Exception:
                            pass
                        return True
                    except Exception as e:
                        try:
                            self._append_log(f"Queue error (SeedVR2) via {where_label}: {e}")
                        except Exception:
                            pass
                        return False
    except Exception:
        pass


# If signature wants plain args, call with those
    if sig:
        params = list(sig.parameters.keys())
        if params[:4] == ["input_path", "out_dir", "factor", "model"] or set(("input_path","out_dir","factor","model")).issubset(set(params)):
            try:
                enq(job_type=('upscale_photo' if str(Path(input_path)).lower().endswith(tuple(_IMAGE_EXTS)) else 'upscale_video'), input_path=input_path, out_dir=out_dir, factor=factor, model=model_name)
                try:
                    self._append_log(f"Queued via {where_label} (kwargs).")
                except Exception: pass
                return True
            except Exception as e:
                try: self._append_log(f"Queue error via {where_label}: {e}")
                except Exception: pass
                return False

    # Else, fallback to job dicts (one per command)
    for i, c in enumerate(cmds, 1):
        job = {
            "name": "Upscale" if len(cmds) == 1 else f"Upscale ({i}/{len(cmds)})",
            "category": "upscale",
            "cmd": c,
            "cwd": str(_kwargs.get("cwd") or globals().get("ROOT", ".")),
            "env": _kwargs.get("env"),
            "open_on_success": bool(open_on_success and i == len(cmds)),
            "output": str(getattr(self, "_last_outfile", "")),
        }
        try:
            enq(job)
        except Exception as e:
            try: self._append_log(f"Queue error via {where_label}: {e}")
            except Exception: pass
            return False
    try:
        self._append_log(f"Queued {len(cmds)} job(s) via {where_label}")
    except Exception:
        pass
    return True

try:
    _UpscClass = None
    for _n, _obj in list(globals().items()):
        if isinstance(_obj, type) and hasattr(_obj, "_run_cmd"):
            _UpscClass = _obj
            break

    if _UpscClass is not None:
        _orig_run_cmd = getattr(_UpscClass, "_run_cmd", None)
        def _patched_run_cmd(self, cmds, open_on_success: bool = False, cleanup_dirs=None, **_kwargs):
            enq, where = _fv_find_enqueue(self)
            if callable(enq) and cmds:
                if _fv_call_enqueue(self, enq, where, cmds, open_on_success, **_kwargs):
                    return
            # Fallback to original implementation
            if callable(_orig_run_cmd):
                return _orig_run_cmd(self, cmds, open_on_success=open_on_success, cleanup_dirs=cleanup_dirs, **_kwargs)

        setattr(_UpscClass, "_run_cmd", _patched_run_cmd)

        _orig_set_main = getattr(_UpscClass, "set_main", None)
        def _patched_set_main(self, main):
            if callable(_orig_set_main):
                try:
                    _orig_set_main(self, main)
                except TypeError:
                    try: self._main = main
                    except Exception: pass
            else:
                try: self._main = main
                except Exception: pass
            if _FVQtW is not None and main is not None:
                try:
                    for _btn in main.findChildren(_FVQtW.QPushButton):
                        if _btn.text().strip().lower() == "upscale":
                            try: _btn.clicked.disconnect()
                            except Exception: pass
                            def _call():
                                p = _fv_guess_player_path(self) or _fv_get_input(self)
                                if not _fv_is_valid_file(p):
                                    try: self._append_log("Player Upscale: no valid file path available.")
                                    except Exception: pass
                                    return
                                _fv_push_input_to_tab(self, p)
                                try:
                                    if hasattr(self, "btn_upscale"):
                                        self.btn_upscale.click()
                                    else:
                                        self._do_single()
                                except Exception:
                                    try: self._do_single()
                                    except Exception: pass
                            _btn.clicked.connect(_call)
                            break
                except Exception:
                    pass

        setattr(_UpscClass, "set_main", _patched_set_main)

except Exception as _patch_exc:
    try:
        print(f"[upsc patch] non-fatal: {_patch_exc}")
    except Exception:
        pass



# --- FrameVision silent integration r11 ---
from typing import List, Tuple
from pathlib import Path as _P
import os as _O

def _fv_r11_find(root: _P, names) -> _P | None:
    """Fast, non-recursive lookup.

    The previous implementation used Path.rglob(), which can crawl very large model
    folders and dramatically slow down app startup.
    """
    try:
        if not root.exists():
            return None

        # 1) Direct hits (also supports nested relative paths like "Scripts/python.exe")
        for nm in names:
            try:
                q = root / nm
                if q.is_file():
                    return q
            except Exception:
                pass

        # 2) One-level deep fallback (cheap, covers common layouts)
        for nm in names:
            # Skip if it already looks like a subpath
            if isinstance(nm, str) and ("/" in nm or "\\" in nm):
                continue
            try:
                for sub in root.iterdir():
                    try:
                        if sub.is_dir():
                            q = sub / nm
                            if q.is_file():
                                return q
                    except Exception:
                        continue
            except Exception:
                pass
    except Exception:
        return None
    return None

def _fv_r11_scan_pairs(root: _P):
    # Fast, depth-limited scan for *.param + *.bin pairs (avoid rglob startup stalls).
    out = []
    try:
        if not root.exists():
            return out

        def _iter_params():
            # depth 0 and depth 1 are usually enough for packaged model folders
            for prm in root.glob("*.param"):
                yield prm
            for prm in root.glob("*/*.param"):
                yield prm

        seen = set()
        for prm in _iter_params():
            try:
                if not prm.is_file():
                    continue
                key = str(prm)
                if key in seen:
                    continue
                seen.add(key)
                base = prm.stem
                binf = prm.with_suffix(".bin")
                if not binf.exists():
                    continue
                sc = 4
                low = base.lower()
                for k in ("-8x", "-6x", "-4x", "-3x", "-2x"):
                    if k in low:
                        try:
                            sc = int(k[1])
                        except Exception:
                            sc = 4
                        break
                out.append((base, str(prm.parent), sc))
            except Exception:
                continue
    except Exception:
        pass
    return out

def _fv_r11_install_upsc_models(self):
    try:
        root = ROOT  # provided by original module
    except Exception:
        root = _P(__file__).resolve().parents[1]
    cand_dirs = [root / "models" / "upscayl", root / "models" / "upscayl" / "resources" / "models"]
    seen = set()
    for d in cand_dirs:
        for base, dstr, sc in _fv_r11_scan_pairs(d):
            key = (base, dstr)
            if key in seen:
                continue
            seen.add(key)
            label = "Upscayl - " + base
            try:
                self.combo_model_realsr.addItem(label, {"fv":"upsc-model","base":base,"dir":dstr,"scale":sc})
            except Exception:
                self.combo_model_realsr.addItem(label)

def _fv_r11_wrap_file(orig):
    def _wrapped(self, engine_exe, src, outfile, model, scale):
        from pathlib import Path as __P
        idx = getattr(self.combo_model_realsr, "currentIndex", lambda: -1)()
        data = None
        try:
            data = self.combo_model_realsr.itemData(idx)
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("fv") == "upsc-model":
            base = data["base"]; dstr = data["dir"]; sc = int(data["scale"])
            in_p = __P(src); out_p = __P(outfile)
            if in_p.exists() and in_p.is_dir():
                out_p.mkdir(parents=True, exist_ok=True)
                cmd = [engine_exe, "-i", str(in_p), "-o", str(out_p), "-s", str(sc), "-n", base, "-m", dstr]
                # Force GPU selection + tuning for Upscayl/ESRGAN models (realesrgan-ncnn-vulkan)
                try:
                    gid = os.environ.get("FV_REALSR_GPU_ID", "").strip() or "0"
                    cmd += ["-g", str(gid)]
                except Exception:
                    pass
                try:
                    jobs = os.environ.get("FV_REALSR_JOBS", "").strip()
                    if jobs:
                        cmd += ["-j", str(jobs)]
                except Exception:
                    pass
                try:
                    t = int(self.spin_tile.value()) if hasattr(self, "spin_tile") else 0
                    if t > 0:
                        cmd += ["-t", str(t)]
                except Exception:
                    pass
                return cmd
            if (not out_p.suffix) or (out_p.exists() and out_p.is_dir()):
                out_p.mkdir(parents=True, exist_ok=True)
                out_p = out_p / (in_p.stem + ".png")
            cmd = [engine_exe, "-i", str(in_p), "-o", str(out_p), "-s", str(sc), "-n", base, "-m", dstr]
            # Force GPU selection + tuning for Upscayl/ESRGAN models (realesrgan-ncnn-vulkan)
            try:
                gid = os.environ.get("FV_REALSR_GPU_ID", "").strip() or "0"
                cmd += ["-g", str(gid)]
            except Exception:
                pass
            try:
                jobs = os.environ.get("FV_REALSR_JOBS", "").strip()
                if jobs:
                    cmd += ["-j", str(jobs)]
            except Exception:
                pass
            try:
                t = int(self.spin_tile.value()) if hasattr(self, "spin_tile") else 0
                if t > 0:
                    cmd += ["-t", str(t)]
            except Exception:
                pass
            return cmd
        return orig(self, engine_exe, src, outfile, model, scale)
    return _wrapped

def _fv_r11_wrap_dir(orig):
    def _wrapped(self, engine_exe, in_dir, out_dir, model, scale):
        from pathlib import Path as __P
        idx = getattr(self.combo_model_realsr, "currentIndex", lambda: -1)()
        data = None
        try:
            data = self.combo_model_realsr.itemData(idx)
        except Exception:
            data = None
        if isinstance(data, dict) and data.get("fv") == "upsc-model":
            base = data["base"]; dstr = data["dir"]; sc = int(data["scale"])
            in_p = __P(in_dir); out_p = __P(out_dir); out_p.mkdir(parents=True, exist_ok=True)
            cmd = [engine_exe, "-i", str(in_p), "-o", str(out_p), "-s", str(sc), "-n", base, "-m", dstr]
            # Force GPU selection + tuning for Upscayl/ESRGAN models (realesrgan-ncnn-vulkan)
            try:
                gid = os.environ.get("FV_REALSR_GPU_ID", "").strip() or "0"
                cmd += ["-g", str(gid)]
            except Exception:
                pass
            try:
                jobs = os.environ.get("FV_REALSR_JOBS", "").strip()
                if jobs:
                    cmd += ["-j", str(jobs)]
            except Exception:
                pass
            try:
                t = int(self.spin_tile.value()) if hasattr(self, "spin_tile") else 0
                if t > 0:
                    cmd += ["-t", str(t)]
            except Exception:
                pass
            return cmd
        return orig(self, engine_exe, in_dir, out_dir, model, scale)
    return _wrapped

def _fv_r11_install_hooks():
    try:
        _Pane = UpscPane  # UI class from original module
    except Exception:
        return
    try:
        _old_init = _Pane.__init__
        def _new_init(self, *a, **k):
            _old_init(self, *a, **k)
            try:
                _fv_r11_install_upsc_models(self)
            except Exception:
                pass
        _Pane.__init__ = _new_init
        if hasattr(_Pane, "_realsr_cmd_file"):
            _Pane._realsr_cmd_file = _fv_r11_wrap_file(_Pane._realsr_cmd_file)
        if hasattr(_Pane, "_realsr_cmd_dir"):
            _Pane._realsr_cmd_dir = _fv_r11_wrap_dir(_Pane._realsr_cmd_dir)
    except Exception:
        pass

def _fv_r11_detect_engines() -> List[Tuple[str, str]]:
    engines: List[Tuple[str, str]] = []
    try:
        root = ROOT
    except Exception:
        root = _P(__file__).resolve().parents[1]
    models = root / "models"
    externals = root / "externals"
    def _find(root: _P, names):
        return _fv_r11_find(root, names)
    realsr = _find(models / "realesrgan", ["realesrgan-ncnn-vulkan.exe","realesrgan-ncnn-vulkan"])
    if realsr: engines.append(("Real-ESRGAN (ncnn)", str(realsr)))
    if realsr: engines.append(("UltraSharp (ncnn)", str(realsr)))
    if realsr: engines.append(("SRMD (ncnn via RealESRGAN)", str(realsr)))
    waifu = _find(models / "waifu2x", ["waifu2x-ncnn-vulkan.exe","waifu2x-ncnn-vulkan"])
    if waifu: engines.append(("Waifu2x (ncnn)", str(waifu)))
    srmd = _find(models / "srmd-ncnn-vulkan-master", ["srmd-ncnn-vulkan.exe","srmd-ncnn-vulkan"])
    if srmd: engines.append(("SRMD (ncnn)", str(srmd)))
    realsr_ncnn = _find(models / "realsr-ncnn-vulkan-20220728-windows", ["realsr-ncnn-vulkan.exe","realsr-ncnn-vulkan"])
    if realsr_ncnn: engines.append(("RealSR (ncnn)", str(realsr_ncnn)))
    swinir = _find(models / "swinir", ["swinir-ncnn-vulkan.exe","swinir-ncnn-vulkan"])
    if swinir: engines.append(("SwinIR (ncnn)", str(swinir)))
    lapsrn = _find(models / "lapsrn", ["lapsrn-ncnn-vulkan.exe","lapsrn-ncnn-vulkan"])
    if lapsrn: engines.append(("LapSRN (ncnn)", str(lapsrn)))
    # Upscayl: only show if CLI exists
    up_cli = None
    for base in (externals / "upscayl", models / "upscayl"):
        up_cli = _find(base, ["upscayl-cli.exe","upscayl-cli"])
        if up_cli:
            engines.append(("Upscayl (CLI)", str(up_cli)))
            break
    # GFPGAN (faces)
    try:
        gfp_py = _find(models / "gfpgan" / ".GFPGAN", ["python.exe","python","Scripts/python.exe","bin/python"])
        if gfp_py:
            engines.append(("GFPGAN (faces)", str(gfp_py)))
    except Exception:
        pass
    return engines

# Attach hooks and quiet detector
try:
    _fv_r11_install_hooks()
    detect_engines = _fv_r11_detect_engines
except Exception:
    pass
# --- end r11 ---


# --- FRAMEVISION_INPUT_SYNC_AND_SCALE_TOGGLE_PATCH ---
try:
    _Pane = UpscPane  # type: ignore  # noqa: F821
    if not hasattr(_Pane, "_fv_patched_input_and_scale"):
        _Pane._fv_patched_input_and_scale = True

        # ---------- Wrap _do_single to sync input from player ----------
        if not hasattr(_Pane, "_fv_orig_do_single"):
            _Pane._fv_orig_do_single = _Pane._do_single
            def _fv_wrapped_do_single(self, *a, **k):
                try:
                    p = _fv_guess_player_path(self)  # type: ignore  # noqa: F821
                    if p and _fv_is_valid_file(p):   # type: ignore  # noqa: F821
                        try:
                            cur = (self.edit_input.text() or '').strip()
                        except Exception:
                            cur = ''
                        if str(p) != cur:
                            try:
                                _fv_push_input_to_tab(self, str(p))  # type: ignore  # noqa: F821
                            except Exception:
                                try:
                                    self.edit_input.setText(str(p))
                                except Exception:
                                    pass
                except Exception:
                    pass
                return _Pane._fv_orig_do_single(self, *a, **k)
            _Pane._do_single = _fv_wrapped_do_single

        # ---------- Helpers for scale UI toggle ----------
        from PySide6 import QtWidgets  # type: ignore

        def _fv_is_realsr_engine(self) -> bool:
            try:
                t = (self.combo_engine.currentText() or '').lower()
                # Treat ALL engines as fixed-scale except Waifu2x.
                # When this returns True, the UI hides the Scale label/slider/spin.
                return ('waifu2x' not in t)
            except Exception:
                return False
        def _fv_native_scale(self) -> int:
            try:
                eng = (self.combo_engine.currentText() or '').lower()
                if 'gfpgan' in eng:
                    return 1
                t = (self.combo_model_realsr.currentText() or '').lower()
                if ('-x4' in t) or ('x4' in t): return 4
                if ('-x3' in t) or ('x3' in t): return 3
                return 2
            except Exception:
                return 2

        def _fv_find_scale_label(self):
            try:
                for lab in self.findChildren(QtWidgets.QLabel):
                    if (lab.text() or '').strip().lower() == 'scale:':
                        return lab
            except Exception:
                pass
            return None

        def _fv_apply_scale_ui(self):
            # Show/hide and sync scale value depending on engine
            isr = _fv_is_realsr_engine(self)
            lbl = getattr(self, '_fv_lbl_scale', None)
            if lbl is None:
                lbl = _fv_find_scale_label(self)
                if lbl is not None:
                    self._fv_lbl_scale = lbl
            try:
                if isr:
                    native = _fv_native_scale(self)
                    try:
                        self.spin_scale.blockSignals(True)
                        self.spin_scale.setValue(float(native))
                        self.spin_scale.blockSignals(False)
                    except Exception:
                        pass
                    try:
                        self.slider_scale.blockSignals(True)
                        self.slider_scale.setValue(int(native * 10))
                        self.slider_scale.blockSignals(False)
                    except Exception:
                        pass
                # Visibility
                vis = not isr
                try:
                    self.spin_scale.setVisible(vis)
                except Exception:
                    pass
                try:
                    self.slider_scale.setVisible(vis)
                except Exception:
                    pass
                try:
                    if lbl is not None:
                        lbl.setVisible(vis)
                except Exception:
                    pass
            except Exception:
                pass

        # ---------- Wrap __init__ to connect signals and apply on startup ----------
        if not hasattr(_Pane, "_fv_orig_init"):
            _Pane._fv_orig_init = _Pane.__init__
            def _fv_init(self, *a, **k):
                _Pane._fv_orig_init(self, *a, **k)
                try:
                    # Cache label and connect to updates
                    self._fv_lbl_scale = _fv_find_scale_label(self)
                    try:
                        self.combo_engine.currentIndexChanged.connect(lambda *_: _fv_apply_scale_ui(self))
                    except Exception:
                        pass
                    try:
                        self.combo_model_realsr.currentIndexChanged.connect(lambda *_: _fv_apply_scale_ui(self))
                    except Exception:
                        pass
                    _fv_apply_scale_ui(self)
                except Exception:
                    pass
            _Pane.__init__ = _fv_init
except Exception:
    pass


# --- FRAMEVISION_TOOLTIP_AND_MODELINFO_PATCH (2025-10-20) ---
try:
    _Pane = UpscPane  # type: ignore  # noqa: F821
    if not hasattr(_Pane, "_fv_patched_tooltips_modelinfo_20251020"):
        _Pane._fv_patched_tooltips_modelinfo_20251020 = True

        from PySide6 import QtWidgets as _FVQtW2  # type: ignore

        # Rich, sourced model info (short, single-line style)
        _FV_MODEL_HINTS = {
            # Real-ESRGAN family (primary)
            "realesrgan-x4plus": ("Photo", "General 4× model for real‑world photo/video; Can take a while to finish on longer videos."),
            "realesrnet-x4plus": ("Photo", "4× RealESRNet (no-GAN) — fewer hallucinated textures; stable on photos."),
            "realesrgan-x4plus-anime": ("Anime", "4× model tuned for anime/illustrations; preserves clean lines & flats."),
            "realesr-animevideov3-x4": ("Anime/Video", "4× anime‑video model (v3); reduces temporal artifacts on frames."),
            "realesr-general-x4v3": ("Photo (Tiny)", "4× general v3 (tiny) for blind SR; robust across varied content."),
            "realesr-general-wdn-x4v3": ("Photo+Denoise", "4× general v3 with built‑in denoise (WDN) to suppress compression noise."),
            # Popular community model
            "4x-ultrasharp": ("UI/Text", "4× UltraSharp: crisp, edge‑preserving detail; great for UI/screens; can look oversharp."),
            # Waifu2x model dirs
            "models-cunet": ("Anime", "Waifu2x CUNet model for anime‑style art with denoise levels."),
            "models-upconv_7_anime_style_art_rgb": ("Anime", "UpConv7 (anime RGB) — fast; keeps flat colors and linework."),
            "models-upconv_7_photo": ("Photo", "UpConv7 (photo) — better for natural images & gradients."),
            # SRMD (classical)
            "srmd": ("Classical SR", "SRMD classical SR; use SRMDNF for noise‑free variant; good on older photos."),
            "srmdnf": ("Classical SR", "SRMDNF (noise‑free) — removes noise prior to upscaling; stable results."),
            # RealSR (ncnn)
            "realsr-x2": ("Photo", "RealSR (x2) real‑world photo SR with degradation modeling; natural detail."),
            "realsr-x4": ("Photo", "RealSR (x4) real‑world photo SR; fewer artifacts on camera images."),
        }

        def _fv_lookup_hint(name: str, engine_label: str) -> tuple[str, str]:
            low = (name or "").lower().strip()
            elow = (engine_label or "").lower()
            # Direct keys or substring matches for known models
            for key, (cat, txt) in _FV_MODEL_HINTS.items():
                if key in low:
                    return cat, txt
            # Upscayl packaged models (label usually "Upscayl - <base>")
            if "upscayl" in low:
                if "anime" in low:
                    return "Anime", "Upscayl anime model based on ESRGAN family; strong line preservation."
                if "digital" in low or "art" in low:
                    return "Digital Art", "Upscayl digital‑art model; enhances CG and illustrations with sharp edges."
                return "Photo", "Upscayl general‑photo model (ESRGAN‑based) for everyday pictures."
            # Engine‑level fallbacks
            if "swinir" in elow:
                return "Transformer", "SwinIR (Transformer‑based SR) — strong fidelity on natural images."
            if "lapsrn" in elow:
                return "Pyramid CNN", "LapSRN (Laplacian pyramid CNN) — lightweight & fast SR."
            if "waifu2x" in elow:
                return "Anime", "Waifu2x family — best for anime/line art; choose CUNet/UpConv7 variants."
            if "realsr (" in elow:
                return "Photo", "RealSR (ncnn) for real‑world photos; good at reducing artifacts."
            if "srmd" in elow:
                return "Classical SR", "SRMD classical SR; SRMDNF variant includes denoise prior."
            # Generic
            if "ultrasharp" in low or "ui" in low or "text" in low:
                return "UI/Text", "Edge‑focused model for UI, text, and crisp graphics."
            if "anime" in low:
                return "Anime", "Optimized for anime/illustrations; preserves lines & flats."
            if "general" in low or "x4" in low or "x3" in low or "x2" in low:
                return "Photo", "Balanced detail for photographs across varied content."
            return "Other", ""

        def _fv_current_model_text(self) -> str:
            try:
                engine_label = self.combo_engine.currentText()
            except Exception:
                engine_label = ""
            try:
                page = int(self.stk_models.currentIndex())
            except Exception:
                page = 0
            # Choose the right combo for current engine/page
            if page == 0:
                if "Waifu2x" in engine_label and hasattr(self, "combo_model_w2x"):
                    return self.combo_model_w2x.currentText()
                if "UltraSharp" in engine_label and hasattr(self, "combo_model_ultrasharp"):
                    return self.combo_model_ultrasharp.currentText()
                if "SRMD (ncnn via RealESRGAN)" in engine_label and hasattr(self, "combo_model_srmd_realsr"):
                    return self.combo_model_srmd_realsr.currentText()
                return self.combo_model_realsr.currentText()
            # Other stacked pages (best‑effort)
            for nm in ("combo_model_gfpgan","combo_model_ultrasharp","combo_model_srmd_realsr","combo_model_w2x","combo_model_srmd","combo_model_realsr_ncnn","combo_model_realsr"):
                w = getattr(self, nm, None)
                if w is not None and hasattr(w, "currentText"):
                    try:
                        return w.currentText()
                    except Exception:
                        pass
            try:
                return self.combo_model_realsr.currentText()
            except Exception:
                return ""

        # Replace the simple hint updater with the richer one
        def _fv_update_model_hint(self):
            try:
                engine_label = self.combo_engine.currentText()
            except Exception:
                engine_label = ""
            name = ""
            try:
                name = _fv_current_model_text(self)
            except Exception:
                pass
            cat, hint = _fv_lookup_hint(name, engine_label)
            try:
                self.lbl_model_badge.setText(cat)
            except Exception:
                pass
            try:
                self.lbl_model_hint.setText(hint)
            except Exception:
                pass

        # Install encoder/tool tooltips in a dedicated helper
        def _fv_install_tooltips(self):
            def tip(w, text):
                try:
                    if w is not None and hasattr(w, "setToolTip"):
                        w.setToolTip(text)
                except Exception:
                    pass

            # Video codec & rate control
            tip(getattr(self, "combo_vcodec", None),
                "Video codec: H.264 (libx264) is most compatible; H.265/HEVC (libx265) smaller files; AV1 (SVT‑AV1/libaom) best compression but slower/less supported.")
            tip(getattr(self, "rad_crf", None),
                "CRF = Constant Rate Factor (quality‑based). Lower CRF → higher quality & larger file. Typical: x264 ≈ 18–23, x265 ≈ 20–28. Ignored if Bitrate is selected.")
            tip(getattr(self, "spin_crf", None),
                "Set CRF quality level for quality‑based encoding. Lower is better quality/larger size.")
            tip(getattr(self, "rad_bitrate", None),
                "Bitrate mode targets an average video bitrate (kbps). Use when you need a fixed size/bitrate. Disables CRF.")
            tip(getattr(self, "spin_bitrate", None),
                "Target video bitrate in kbps (e.g., 8000 = 8 Mbps). Rough guide: 1080p 6–12 Mbps; 4K 12–35 Mbps (content dependent).")
            tip(getattr(self, "combo_preset", None),
                "Encoder speed/quality trade‑off. Slower presets compress better (smaller files for same quality).")
            tip(getattr(self, "spin_keyint", None),
                "Keyframe interval (GOP). 0 lets the encoder choose. Otherwise sets ffmpeg -g <N>.")

            # Audio controls
            tip(getattr(self, "radio_a_copy", None), "Copy audio stream without re‑encoding (fast; keeps original quality/codec).")
            tip(getattr(self, "radio_a_encode", None), "Encode audio with chosen codec/bitrate (use if changing container/codecs).")
            tip(getattr(self, "radio_a_mute", None), "Remove/mute audio from the output.")
            tip(getattr(self, "combo_acodec", None), "Audio codec: AAC (very compatible), Opus (efficient esp. speech), Vorbis (legacy/OGG).")
            tip(getattr(self, "spin_abitrate", None), "Audio bitrate (kbps). Common: AAC 128–192, Opus 96–160.")

            # Model badge/hint hover
            tip(getattr(self, "lbl_model_badge", None), "Category of the selected model.")
            tip(getattr(self, "lbl_model_hint", None), "selected model.")

        # Wrap __init__ to attach tooltips and hook our rich hint updater
        if not hasattr(_Pane, "_fv_orig_init_tooltips"):
            _Pane._fv_orig_init_tooltips = _Pane.__init__
            def _fv_init_plus(self, *a, **k):
                _Pane._fv_orig_init_tooltips(self, *a, **k)
                try:
                    # Install tooltips
                    _fv_install_tooltips(self)
                except Exception:
                    pass
                try:
                    # Rewire model-hint updater
                    self._update_model_hint = _fv_update_model_hint.__get__(self, _Pane)  # bind
                    # Run once to refresh
                    self._update_model_hint()
                    # Connect changes
                    for nm in ("combo_model_realsr", "combo_model_w2x",
                               "combo_model_ultrasharp", "combo_model_srmd_realsr",
                               "combo_model_srmd", "combo_model_realsr_ncnn", "combo_engine"):
                        w = getattr(self, nm, None)
                        try:
                            if hasattr(w, "currentTextChanged"):
                                w.currentTextChanged.connect(self._update_model_hint)
                        except Exception:
                            pass
                        try:
                            if hasattr(w, "currentIndexChanged"):
                                w.currentIndexChanged.connect(self._update_model_hint)
                        except Exception:
                            pass
                    try:
                        if hasattr(self, "stk_models"):
                            self.stk_models.currentChanged.connect(lambda *_: self._update_model_hint())
                    except Exception:
                        pass
                except Exception:
                    pass
            _Pane.__init__ = _fv_init_plus
except Exception as _fv_patch_exc:
    try:
        print("[upsc tooltip/modelinfo patch] non-fatal:", _fv_patch_exc)
    except Exception:
        pass
# --- END FRAMEVISION_TOOLTIP_AND_MODELINFO_PATCH ---



# --- FRAMEVISION_TOOLTIP_AND_MODELINFO_PATCH v2 (visible-combo fix) ---
try:
    _Pane = UpscPane  # type: ignore  # noqa: F821
    if not hasattr(_Pane, "_fv_visible_combo_fix_20251020b"):
        _Pane._fv_visible_combo_fix_20251020b = True

        def _fv_current_model_text(self):
            # Prefer the model combo that is actually visible in the stacked widget.
            combo_names = (
                "combo_model_realsr",
                "combo_model_w2x",
                "combo_model_ultrasharp",
                "combo_model_srmd_realsr",
                "combo_model_srmd",
                "combo_model_realsr_ncnn",
            )
            visible = []
            for nm in combo_names:
                w = getattr(self, nm, None)
                if w is None or not hasattr(w, "currentText"):
                    continue
                try:
                    if hasattr(w, "isVisible") and w.isVisible():
                        visible.append(w)
                except Exception:
                    # If visibility check fails, keep going
                    pass
            try:
                if visible:
                    return visible[0].currentText()
            except Exception:
                pass

            # If none are visible (rare), pick by engine label mapping.
            try:
                engine_label = self.combo_engine.currentText().lower()
            except Exception:
                engine_label = ""

            def pick(*names):
                for n in names:
                    w = getattr(self, n, None)
                    if w is not None and hasattr(w, "currentText"):
                        try:
                            return w.currentText()
                        except Exception:
                            pass
                return ""

            if "srmd" in engine_label:
                return pick("combo_model_srmd_realsr", "combo_model_srmd")
            if "waifu2x" in engine_label:
                return pick("combo_model_w2x")
            if "ultrasharp" in engine_label:
                return pick("combo_model_ultrasharp")
            if "realsr" in engine_label or "ncnn" in engine_label:
                return pick("combo_model_realsr", "combo_model_realsr_ncnn")

            # Fallback: first available
            return pick(*combo_names)

except Exception as _fv_fix_exc:
    try:
        print("[upsc tooltip/modelinfo patch v2] non-fatal:", _fv_fix_exc)
    except Exception:
        pass
# --- END v2 (visible-combo fix) ---


# --- FRAMEVISION_CLEARREALITY_V1_ENGINE_PATCH (2025-10-24) ---
# Adds a dedicated "ClearReality V1" engine option (re-using the Real-ESRGAN backend)
# and filters the model list to only ClearReality models when selected. Also augments
# model tooltips.
try:
    from PySide6 import QtCore as _FVQtCore2, QtWidgets as _FVQtW3  # type: ignore
except Exception:
    _FVQtCore2 = None
    _FVQtW3 = None

def _fv_scan_clearreality_models() -> list[str]:
    try:
        base = REALSR_DIR  # type: ignore  # provided by module
    except Exception:
        try:
            base = Path(__file__).resolve().parents[1] / "models" / "realesrgan"  # type: ignore
        except Exception:
            return []
    names: set[str] = set()
    try:
        if base.exists():
            for ext in ("*.bin", "*.param"):
                for p in sorted(base.glob(ext)):
                    s = p.stem.lower()
                    if "clearrealityv1" in s:
                        names.add(p.stem)
    except Exception:
        pass
    # Reasonable fallbacks if nothing found
    if not names:
        return ["4x-ClearRealityV1-fp16", "4x-ClearRealityV1-fp32",
                "4x-ClearRealityV1_Soft-fp16", "4x-ClearRealityV1_Soft-fp32"]
    return sorted(names)

# 1) Extend engine detection to include "ClearReality V1" (uses realesrgan-ncnn-vulkan)
try:
    _fv_orig_detect_engines = detect_engines  # type: ignore
except Exception:
    _fv_orig_detect_engines = None

def _fv_detect_engines_plus_clearreality():
    out = []
    try:
        if callable(_fv_orig_detect_engines):
            out = list(_fv_orig_detect_engines())  # type: ignore
    except Exception:
        out = []
    # Try to find the realesrgan executable used by other engines
    realsr_exe = None
    for label, exe in out:
        l = (label or "").lower()
        if ("real-esrgan" in l) or ("realesrgan" in l) or ("ultrasharp" in l) or ("srmd (ncnn via realesrgan" in l):
            realsr_exe = exe
            break
    if realsr_exe is None:
        try:
            cand = REALSR_DIR / ("realesrgan-ncnn-vulkan.exe" if os.name == "nt" else "realesrgan-ncnn-vulkan")  # type: ignore
            if cand.exists():
                realsr_exe = str(cand)
        except Exception:
            pass
    if realsr_exe and not any((lbl or "").lower().startswith("clearreality v1") for (lbl, _x) in out):
        # Place it near the top for convenience (after Real-ESRGAN if present)
        idx = 1 if out and "real-esrgan" in (out[0][0] or "").lower() else len(out)
        out.insert(idx, ("ClearReality V1", realsr_exe))
    return out

try:
    detect_engines = _fv_detect_engines_plus_clearreality  # type: ignore
except Exception:
    pass

# 2) Filter the Real-ESRGAN model combo when the "ClearReality V1" engine is selected.
try:
    _Pane = UpscPane  # type: ignore  # noqa: F821
    if not hasattr(_Pane, "_fv_clearreality_engine_patch_20251024"):
        _Pane._fv_clearreality_engine_patch_20251024 = True

        def _fv_restore_or_filter_models(self):
            try:
                eng = (self.combo_engine.currentText() or "").lower()
            except Exception:
                eng = ""
            combo = getattr(self, "combo_model_realsr", None)
            if combo is None or not hasattr(combo, "count"):
                return
            # Cache the full model list once
            full_key = "_fv_realsr_all_items"
            if not hasattr(self, full_key):
                try:
                    all_items = []
                    for i in range(combo.count()):
                        all_items.append(combo.itemText(i))
                    setattr(self, full_key, all_items)
                except Exception:
                    setattr(self, full_key, [])
            all_items = list(getattr(self, full_key, []))

            def _repopulate(items: list[str]):
                try:
                    combo.blockSignals(True)
                    combo.clear()
                    for it in items:
                        combo.addItem(it)
                finally:
                    try:
                        combo.blockSignals(False)
                    except Exception:
                        pass
                # refresh hint, if available
                try:
                    if hasattr(self, "_update_model_hint"):
                        self._update_model_hint()
                except Exception:
                    pass

            if "clearreality v1" in eng:
                # Only show ClearReality models (scan directly from disk)
                try:
                    items = list(_fv_scan_clearreality_models())
                except Exception:
                    items = []
 #           elif "bstexty for text upscaling" in eng:
  #              # Only show BStexty models
   #             items = [s for s in all_items if "bstexty" in (s or "").lower()]
 #           elif "fatality noisetoner for sharping/denoising" in eng:
    #            # Only show Fatality NoiseToner models
     #           items = [s for s in all_items if ("fatality" in (s or "").lower()) or ("noisetoner" in (s or "").lower())]
            elif "real-esrgan" in eng:
                # Real-ESRGAN engine: hide BStexty/Fatality models (they have their own engines)
                items = [s for s in all_items if ("bstexty" not in (s or "").lower()) and ("fatality" not in (s or "").lower()) and ("noisetoner" not in (s or "").lower())]
            else:
                items = None

            if items is not None:
                # If nothing matched, fall back to the full list so the combo never ends up empty.
                if not items:
                    items = list(all_items)
                _repopulate(items)
            else:
                # Restore full list if it looks filtered
                if combo.count() != len(all_items):
                    _repopulate(all_items)

        # Hook into __init__ to wire signal + run once
        _orig_init = _Pane.__init__
        def _init_and_wire(self, *a, **k):
            _orig_init(self, *a, **k)
            try:
                if hasattr(self.combo_engine, "currentTextChanged"):
                    self.combo_engine.currentTextChanged.connect(lambda *_: _fv_restore_or_filter_models(self))
            except Exception:
                pass
            try:
                _fv_restore_or_filter_models(self)
            except Exception:
                pass
        _Pane.__init__ = _init_and_wire

except Exception as _fv_cex:
    try:
        print("[ClearReality engine patch] non-fatal:", _fv_cex)
    except Exception:
        pass

# 3) Remove ClearReality entries from the generic Real-ESRGAN scan (to avoid duplicates)
try:
    _fv_orig_scan_realsr = scan_realsr_models  # type: ignore
    def scan_realsr_models():  # type: ignore
        try:
            items = list(_fv_orig_scan_realsr())
        except Exception:
            items = []
        # filter out ClearReality models; they will live under the dedicated engine
        return [i for i in items if "clearreality" not in (i or "").lower()]
except Exception:
    pass

# 4) Extend tooltip map for ClearReality variants (short, single-line style)
try:
    _cr_hints = {
        "4x-clearrealityv1-fp16": ("Photo/Sharp", "ClearReality V1 (fp16) — crisp detail and punchy edges; fast."),
        "4x-clearrealityv1-fp32": ("Photo/Sharp", "ClearReality V1 (fp32) — higher precision; slightly slower; crisp detail."),
        "4x-clearrealityv1_soft-fp16": ("Photo/Soft", "ClearReality V1 Soft (fp16) — gentler sharpening; fewer halos."),
        "4x-clearrealityv1_soft-fp32": ("Photo/Soft", "ClearReality V1 Soft (fp32) — soft variant with fp32 precision."),
        "clearrealityv1": ("Photo", "ClearReality V1 family — Real-ESRGAN fine‑tune with clean detail. Use Soft to reduce halos."),
    }
    try:
        _FV_MODEL_HINTS.update(_cr_hints)  # type: ignore
    except Exception:
        # If the dict wasn't defined yet, create a minimal overlay and a wrapper resolver
        _FV_MODEL_HINTS = dict(_cr_hints)  # type: ignore
except Exception:
    pass
# --- END FRAMEVISION_CLEARREALITY_V1_ENGINE_PATCH ---



# --- FRAMEVISION_HIDE_ENCODER_STYLE_IN_REALESRGAN (2025-10-24) ---
# Hides 'encoder' and 'style' entries from the Real-ESRGAN model dropdown, globally.
try:
    _fv_prev_scan_realsr_2 = scan_realsr_models  # type: ignore
    def scan_realsr_models():  # type: ignore
        try:
            items = list(_fv_prev_scan_realsr_2())
        except Exception:
            items = []
        blocked = {"encoder", "style"}
        out = []
        for i in items:
            s = (i or "").strip()
            low = s.lower()
            if low in blocked:
                continue
            # Also maintain prior behavior of removing ClearReality from the generic list
            if "clearreality" in low:
                continue
            out.append(s)
        return out
except Exception:
    pass

# Also prune any pre-populated combos on construction, just in case.
try:
    _Pane2 = UpscPane  # type: ignore
    if not hasattr(_Pane2, "_fv_hide_encoder_style_runtime_20251024"):
        _orig_init_hide = _Pane2.__init__
        def _init_hide(self, *a, **k):
            _orig_init_hide(self, *a, **k)
            try:
                combo = getattr(self, "combo_model_realsr", None)
                if combo is not None and hasattr(combo, "count"):
                    blocked = {"encoder", "style"}
                    for i in reversed(range(combo.count())):
                        t = (combo.itemText(i) or "").strip().lower()
                        if t in blocked:
                            combo.removeItem(i)
            except Exception:
                pass
        _Pane2.__init__ = _init_hide
        _Pane2._fv_hide_encoder_style_runtime_20251024 = True
except Exception:
    pass
# --- END FRAMEVISION_HIDE_ENCODER_STYLE_IN_REALESRGAN ---

