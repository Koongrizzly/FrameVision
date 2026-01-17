# helpers/qwen2511.py
# Qwen Edit 2511 (GGUF) – PySide6 pane + standalone test runner
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
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QImage, QTextCursor, QPixmap



class ClickableLabel(QtWidgets.QLabel):
    """Tiny clickable thumbnail label."""
    clicked = QtCore.Signal()

    def mousePressEvent(self, event):
        try:
            if event.button() == QtCore.Qt.LeftButton:
                self.clicked.emit()
                event.accept()
                return
        except Exception:
            pass
        return super().mousePressEvent(event)


class ImagePreviewPopup(QtWidgets.QDialog):
    """Popup preview that auto-closes when clicking outside (Qt.Popup)."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(QtCore.Qt.Popup | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(14, 14, 14, 14)

        frame = QtWidgets.QFrame(self)
        frame.setObjectName("imgPreviewFrame")
        frame.setStyleSheet(
            "QFrame#imgPreviewFrame {"
            "background: rgba(20, 20, 20, 235);"
            "border: 1px solid rgba(255, 255, 255, 40);"
            "border-radius: 12px;"
            "}"
        )
        inner = QtWidgets.QVBoxLayout(frame)
        inner.setContentsMargins(12, 12, 12, 12)

        self.lbl = QtWidgets.QLabel(frame)
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl.setMinimumSize(260, 180)
        self.lbl.setStyleSheet("color: #ddd;")

        pix = QPixmap(image_path) if image_path else QPixmap()
        if pix.isNull():
            self.lbl.setText("Preview unavailable")
        else:
            try:
                scr = QtWidgets.QApplication.primaryScreen()
                geo = scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1280, 720)
                max_w = int(geo.width() * 0.82)
                max_h = int(geo.height() * 0.82)
                pix = pix.scaled(max_w, max_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            except Exception:
                pass
            self.lbl.setPixmap(pix)

        inner.addWidget(self.lbl)
        outer.addWidget(frame)

        self.adjustSize()
        self._center_on_screen()

    def _center_on_screen(self):
        try:
            scr = QtWidgets.QApplication.primaryScreen()
            geo = scr.availableGeometry() if scr else QtCore.QRect(0, 0, 1280, 720)
            self.move(geo.center() - self.rect().center())
        except Exception:
            pass

    def keyPressEvent(self, event):
        try:
            if event.key() == QtCore.Qt.Key_Escape:
                self.close()
                return
        except Exception:
            pass
        return super().keyPressEvent(event)




APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SDCLI = os.path.join(APP_ROOT, ".qwen2512", "bin", "sd-cli.exe")

SETSAVE_PATH = os.path.join(APP_ROOT, "presets", "setsave", "qwen2511.json")
DOWNLOAD_SCRIPT = os.path.join(APP_ROOT, "presets", "extra_env", "qwen2511_download.py")

MODELS_ROOT = os.path.join(APP_ROOT, "models", "qwen2511gguf")
UNET_DIR = os.path.join(MODELS_ROOT, "unet")
TEXTENC_DIR = os.path.join(MODELS_ROOT, "text_encoders")
VAE_DIR = os.path.join(MODELS_ROOT, "vae")

# LoRA
# Default folder for Qwen2511 LoRAs (user can override in UI).
DEFAULT_LORA_DIR = os.path.join(APP_ROOT, "models", "lora", "qwen2511")

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


def _list_lora_files(folder: str) -> List[str]:
    """Return LoRA filenames (not full paths) from a directory.

    Reasonable default extensions (can be expanded later):
      - .safetensors (common)
      - .pt / .bin / .ckpt / .gguf (fallback)
    """
    if not folder:
        return []
    try:
        if not os.path.isdir(folder):
            return []
    except Exception:
        return []

    exts = (".safetensors", ".pt", ".bin", ".ckpt", ".gguf")
    out: List[str] = []
    try:
        for fn in os.listdir(folder):
            p = os.path.join(folder, fn)
            if os.path.isfile(p) and fn.lower().endswith(exts):
                out.append(fn)
    except Exception:
        return []
    out.sort(key=lambda x: x.lower())
    return out


def _format_lora_tag(lora_name: str, lora_strength: float) -> str:
    """Format a LoRA tag for prompt injection.

    Default behavior (documented):
      - Use filename stem (basename without extension) as NAME.
      - Strength formatted with up to 3 decimals (trim trailing zeros).
    """
    name = str(lora_name or "").strip()
    name = os.path.basename(name)
    # Use stem by default; sd-cli typically resolves from --lora-model-dir.
    name = os.path.splitext(name)[0]
    try:
        s = float(lora_strength)
    except Exception:
        s = 1.0
    try:
        s = max(0.0, float(s))
    except Exception:
        s = 1.0
    # Trim trailing zeros for cleaner prompt text.
    s_txt = f"{s:.3f}".rstrip("0").rstrip(".")
    return f"<lora:{name}:{s_txt}>"


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
    # Use Qt's native pixel inversion to avoid per-pixel Python loops and
    # avoid QtCore.qRgb (not available in some PySide6 builds).
    img = img.convertToFormat(QImage.Format_Grayscale8)
    img.invertPixels()  # InvertRgb is fine for grayscale
    _ensure_dir(out_dir)
    tmp = os.path.join(out_dir, f"_mask_inverted_{int(time.time())}.png")
    if img.save(tmp, "PNG"):
        return tmp
    return None


def _snap64(v: int) -> int:
    v = int(v)
    v = max(64, v)
    return int(round(v / 64.0) * 64)


def _write_blank_png(path: str, w: int, h: int, rgba=(255, 255, 255, 255)) -> str:
    """Write a simple RGBA PNG without external deps."""
    import zlib
    import struct
    from pathlib import Path

    w = max(1, int(w))
    h = max(1, int(h))
    r, g, b, a = [int(x) & 255 for x in rgba]

    row = bytes([r, g, b, a]) * w
    raw = b"".join((b"\x00" + row) for _ in range(h))  # filter=0 per row
    comp = zlib.compress(raw, 9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)  # 8-bit RGBA
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")
    Path(path).write_bytes(png)
    return path


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
    ref_image: Optional[str] = None
    increase_ref_index: Optional[str] = None
    disable_auto_resize_ref_image: Optional[str] = None

    output: Optional[str] = None

    steps: Optional[str] = None
    cfg: Optional[str] = None
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

    for cand in ["--init-img", "--init_image", "--image", "--input", "-i", "--ref", "--reference"]:
        if has(cand):
            caps.input_image = cand
            break

    if has("--mask"):
        caps.mask = "--mask"

    # Multi-reference images (for "image 1 / image 2 / image 3" prompting)
    if has("--ref-image"):
        caps.ref_image = "--ref-image"
    elif has("-r, --ref-image") or has("-r,--ref-image") or has("-r"):
        caps.ref_image = "--ref-image"

    if has("--increase-ref-index"):
        caps.increase_ref_index = "--increase-ref-index"
    if has("--disable-auto-resize-ref-image"):
        caps.disable_auto_resize_ref_image = "--disable-auto-resize-ref-image"

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
    ref_images: List[str],
    use_increase_ref_index: bool,
    disable_auto_resize_ref_images: bool,
    prompt: str,
    negative: str,
    unet_path: str,
    llm_path: str,
    mmproj_path: str,
    vae_path: str,
    steps: int,
    cfg: float,
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
    # LoRA
    lora_model_dir: str = "",
    lora_name: str = "",
    lora_strength: float = 1.0,
) -> List[str]:
    cmd: List[str] = [sdcli_path, caps.mode_flag, caps.mode_imggen]

    # LoRA support (Qwen2511):
    # - Use --lora-model-dir to point sd-cli to the folder.
    # - Inject <lora:NAME:STRENGTH> at the *start* of the prompt.
    # Enabled when both dir and name are provided and name is not (None).
    use_lora = bool(str(lora_model_dir or "").strip()) and bool(str(lora_name or "").strip())
    if str(lora_name or "").strip() in ("(none)", "none"):
        use_lora = False

    if use_lora:
        cmd += ["--lora-model-dir", str(lora_model_dir)]
        try:
            tag = _format_lora_tag(str(lora_name), float(lora_strength))
            prompt = (tag + (" " + (prompt or "").strip() if (prompt or "").strip() else "")).strip()
        except Exception:
            pass

    if caps.input_image and init_img:
        cmd += [caps.input_image, init_img]

    # Optional multi-reference images (image 2, image 3, ...)
    if ref_images and caps.ref_image:
        if disable_auto_resize_ref_images and caps.disable_auto_resize_ref_image:
            cmd += [caps.disable_auto_resize_ref_image]
        for rp in ref_images:
            if rp:
                # If enabled, advance ref index per image so refs map to image 2, image 3, ...
                # (Fixes multi-ref 'all blended on top of each other' behavior.)
                if use_increase_ref_index and caps.increase_ref_index:
                    cmd += [caps.increase_ref_index]
                cmd += [caps.ref_image, rp]

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
        # Tools-tab "Remember settings" uses a generic widget snapshot/restore.
        # This pane already has its own JSON persistence; opt out to prevent cross-tool value bleed.
        try:
            self.setProperty("_fv_skip_restore", True)
            self.setProperty("_fv_skip_snapshot", True)
        except Exception:
            pass
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Prefer a reference to the main window if our parent exposes it (Tools tab does).
        try:
            self.main = getattr(parent, "main", None)
        except Exception:
            self.main = None
        self.caps: SdCliCaps = detect_sdcli_caps(DEFAULT_SDCLI)
        self._settings = _read_jsonish(SETSAVE_PATH)

        self._ui_loading = True
        self._autosave_suspended = 0
        self._autosave_timer: Optional[QtCore.QTimer] = None

        self._build_ui()
        self._load_defaults()
        self._apply_settings_to_ui()

        # Instant-save settings: persist UI changes to JSON behind the scenes (debounced).
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(lambda: self._save_settings(silent=True))
        self._ui_loading = False
        self._install_autosave_hooks()


        # Track most recent output for quick preview.
        self._last_out_file: Optional[str] = None
        # Async process runner (prevents UI freeze while sd-cli runs)
        self._proc: Optional[QtCore.QProcess] = None
        self._proc_kind: str = ""
        self._proc_buf: str = ""
        self._proc_expected_out: Optional[str] = None

        # Lightweight toast notifications (non-blocking).
        self._toast_label: Optional[QtWidgets.QLabel] = None
        self._toast_timer: Optional[QtCore.QTimer] = None

        self._append_log(self._caps_summary())
        self._append_log(
            "\nBest practice for clean edits (like 'change dress color', 'hold bottle from ref image 2' etc),\n"
            "1) flow around 2.15 and cfg around 2.35 (1 ref image) or 4.0 (2 ref images) seems a good default for starting,\n"
            "2) Strength 0.85 up to 1 ( 0.95 recommended if complete background needs to be changed also).,\n"
            "3) 'scene' image don't work well with faces on it's own, but it can be used to add a ref image twice to anchor it better in the edit,\n"            
        )

    def closeEvent(self, event):
        # Flush any pending autosave to ensure settings persist.
        try:
            self._save_settings(silent=True)
        except Exception:
            pass
        return super().closeEvent(event)


    def _append_log(self, s: str):
        # Newest-first log: insert at the top.
        if not hasattr(self, "log") or self.log is None:
            return
        txt = (s or "").rstrip()
        if not txt:
            return
        cur = self.log.textCursor()
        cur.movePosition(QTextCursor.Start)
        cur.insertText(txt + "\n")
        self.log.setTextCursor(cur)
        try:
            self.log.verticalScrollBar().setValue(0)
        except Exception:
            pass

    def _clear_log(self):
        if hasattr(self, "log") and self.log is not None:
            self.log.clear()


    def _toast(self, message: str, msec: int = 2000) -> None:
        """Show a non-blocking toast notification (best-effort).

        Priority:
          1) If the FrameVision host provides a toast API, use that.
          2) Otherwise, render a small in-window overlay label that auto-hides.
        """
        msg = (message or "").strip()
        if not msg:
            return

        # 1) Use host app toast helper if available.
        host = None
        try:
            host = getattr(self, "main", None)
        except Exception:
            host = None
        if host is None:
            try:
                host = self.window()
            except Exception:
                host = None

        if host is not None:
            for attr in ("toast", "show_toast", "notify_toast", "toast_message", "notify"):
                if hasattr(host, attr):
                    fn = getattr(host, attr, None)
                    if callable(fn):
                        try:
                            fn(msg)
                            return
                        except TypeError:
                            try:
                                fn(msg, msec)
                                return
                            except Exception:
                                pass
                        except Exception:
                            pass

        # 2) Fallback: overlay label inside the current top-level window.
        try:
            win = self.window()
            if win is None:
                return

            if self._toast_label is None:
                lbl = QtWidgets.QLabel(win)
                lbl.setObjectName("toastLabel")
                lbl.setWordWrap(True)
                lbl.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                lbl.setStyleSheet(
                    "QLabel#toastLabel {"
                    "background: rgba(30, 30, 30, 210);"
                    "color: white;"
                    "padding: 8px 10px;"
                    "border-radius: 10px;"
                    "}"
                )
                lbl.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
                lbl.hide()
                self._toast_label = lbl

            if self._toast_timer is None:
                t = QtCore.QTimer(self)
                t.setSingleShot(True)
                t.timeout.connect(lambda: self._toast_label.hide() if self._toast_label else None)
                self._toast_timer = t

            self._toast_label.setText(msg)
            self._toast_label.adjustSize()

            margin = 14
            w = win.width()
            h = win.height()
            tw = self._toast_label.width()
            th = self._toast_label.height()
            x = max(margin, w - tw - margin)
            y = max(margin, h - th - margin)
            self._toast_label.move(x, y)
            self._toast_label.raise_()
            self._toast_label.show()

            try:
                self._toast_timer.stop()
            except Exception:
                pass
            self._toast_timer.start(max(500, int(msec)))
        except Exception:
            # Last resort: write it to the log.
            try:
                self._append_log("\n" + msg)
            except Exception:
                pass

    def _open_results_folder(self):
        """Open Qwen2511 output folder in the app Media Explorer (no OS Explorer button)."""
        try:
            _ensure_dir(OUTPUT_DIR)
        except Exception:
            pass

        # Prefer Media Explorer (single shared entry-point) if available on the main window.
        main = None
        try:
            main = getattr(self, "main", None)
        except Exception:
            main = None
        if main is None:
            try:
                p = self.parent() if hasattr(self, "parent") else None
                main = getattr(p, "main", None) if p is not None else None
            except Exception:
                main = None
        if main is None:
            try:
                main = self.window() if hasattr(self, "window") else None
            except Exception:
                main = None

        if main is not None and hasattr(main, "open_media_explorer_folder"):
            try:
                # Use the most common signature in FrameVision, but fall back if it differs.
                main.open_media_explorer_folder(str(OUTPUT_DIR), preset="images", include_subfolders=False)
                return
            except TypeError:
                try:
                    main.open_media_explorer_folder(str(OUTPUT_DIR))
                    return
                except Exception:
                    pass
            except Exception:
                pass

        # No OS-Explorer fallback here (user request). Just show an info box.
        try:
            QtWidgets.QMessageBox.information(
                self,
                "View results",
                f"Media Explorer entry-point not available.\nFolder: {OUTPUT_DIR}"
            )
        except Exception:
            try:
                self._append_log(f"\nMedia Explorer entry-point not available. Folder: {OUTPUT_DIR}")
            except Exception:
                pass

    def _get_newest_output_file(self) -> Optional[str]:
        """Return newest file path in OUTPUT_DIR (best-effort)."""
        try:
            if not os.path.isdir(OUTPUT_DIR):
                return None
            best = None
            best_m = -1.0
            for name in os.listdir(OUTPUT_DIR):
                fp = os.path.join(OUTPUT_DIR, name)
                if not os.path.isfile(fp):
                    continue
                try:
                    m = os.path.getmtime(fp)
                except Exception:
                    m = 0.0
                if m > best_m:
                    best_m = m
                    best = fp
            return best
        except Exception:
            return None

    def _play_file_in_main_player(self, fp: str) -> bool:
        """Play/open a file in the internal VideoPane if running inside FrameVision."""
        if not fp:
            return False
        main = None
        try:
            main = getattr(self, "main", None)
        except Exception:
            main = None
        if main is None:
            try:
                main = self.window()
            except Exception:
                main = None
        if main is None:
            return False
        try:
            video = getattr(main, "video", None)
        except Exception:
            video = None
        if video is None or not hasattr(video, "open"):
            return False
        try:
            video.open(Path(fp))
            return True
        except Exception:
            try:
                video.open(fp)
                return True
            except Exception:
                return False

    def _play_last_result(self):
        """Play newest output in internal media player (no external player)."""
        fp = None
        try:
            fp = self._last_out_file if (hasattr(self, "_last_out_file") and self._last_out_file) else None
            if fp and not os.path.isfile(fp):
                fp = None
        except Exception:
            fp = None

        if not fp:
            fp = self._get_newest_output_file()

        if not fp:
            try:
                QtWidgets.QMessageBox.information(self, "Play last result", "No outputs found yet in output/qwen2511.")
            except Exception:
                pass
            return

        ok = self._play_file_in_main_player(fp)
        if not ok:
            try:
                QtWidgets.QMessageBox.information(
                    self,
                    "Play last result",
                    "Could not find the internal Media Player host. (This button works inside FrameVision.)",
                )
            except Exception:
                pass

    def _caps_summary(self) -> str:
        c = self.caps
        return (
            "Detected sd-cli capabilities (best-effort):\n"
            f"  mode: {c.mode_imggen}\n"
            f"  input_image flag: {c.input_image}\n"
            f"  mask flag: {c.mask}\n"
            f"  ref-image flag: {c.ref_image}\n"
            f"  increase-ref-index flag: {c.increase_ref_index}\n"
            f"  disable-auto-resize-ref-image flag: {c.disable_auto_resize_ref_image}\n"
            f"  diffusion_model flag: {c.diffusion_model}\n"
            f"  llm/text-encoder flag: {c.llm}\n"
            f"  mmproj flag: {c.mmproj}\n"
            f"  vae flag: {c.vae}\n"
            f"  output flag: {c.output}\n"
            f"  sampling flag: {c.sampling_method}\n"
            f"  supports vae-tiling: {bool(c.vae_tiling)}\n"
            f"  supports offload-to-cpu: {bool(c.offload_to_cpu)}\n"
            f"  supports vae-on-cpu: {bool(c.vae_on_cpu)}\n"
            f"  supports mmap: {bool(c.mmap)}\n"
        )

    def _build_ui(self):
        # Layout goals:
        # - Main controls + log live INSIDE the scroll area (so they scroll normally).
        # - Action buttons stick at the bottom (always visible, do not scroll).
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Scroll area for the whole pane content (except the sticky bottom buttons bar).
        scroll = QtWidgets.QScrollArea()
        scroll.setToolTip("Scroll to access all settings on smaller screens.")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        outer.addWidget(scroll, 1)

        content = QtWidgets.QWidget()
        scroll.setWidget(content)

        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)


        # Fancy banner (ported from Tools tab)
        self.banner = QtWidgets.QLabel("Qwen Edit 2511")
        self.banner.setObjectName("qwenEditBanner")
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.banner.setFixedHeight(48)
        self.banner.setStyleSheet(
            "#qwenEditBanner {"
            " font-size: 15px;"
            " font-weight: 600;"
            " padding: 8px 17px;"
            " border-radius: 12px;"
            " margin: 0 0 6px 0;"
            " color: #eef7ff;"
            " background: qlineargradient("
            "   x1:0, y1:0, x2:1, y2:0,"
            "   stop:0 #bfe9ff,"
            "   stop:0.55 #55b6ff,"
            "   stop:1 #0b3d91"
            " );"
            " letter-spacing: 0.5px;"
            "}"
        )
        layout.addWidget(self.banner)
        layout.addSpacing(4)


        # Models (collapsible)
        self.g_models = QtWidgets.QGroupBox("Models")
        self.g_models.setToolTip("Select the GGUF / VAE files used by sd-cli. Collapse this box if you want a cleaner UI.")
        self.g_models.setCheckable(True)
        self.g_models.setChecked(True)
        self.g_models.toggled.connect(self._on_models_toggled_slot)

        mv = QtWidgets.QVBoxLayout(self.g_models)
        mv.setContentsMargins(8, 8, 8, 8)

        self.models_content = QtWidgets.QWidget()
        ml = QtWidgets.QFormLayout(self.models_content)
        ml.setLabelAlignment(QtCore.Qt.AlignRight)
        ml.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        ml.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        # sd-cli.exe (moved here from Paths)
        self.ed_sdcli = QtWidgets.QLineEdit(DEFAULT_SDCLI)
        self.ed_sdcli.setToolTip("Path to sd-cli.exe (stable-diffusion.cpp). Leave default unless you installed it elsewhere.")
        btn_sdcli = QtWidgets.QToolButton(); btn_sdcli.setText("…"); btn_sdcli.clicked.connect(self._pick_sdcli)
        btn_sdcli.setToolTip("Browse for sd-cli.exe")
        h_sd = QtWidgets.QHBoxLayout(); h_sd.addWidget(btn_sdcli); h_sd.addWidget(self.ed_sdcli, 1)
        ml.addRow("sd-cli.exe", h_sd)

        self.cb_unet = QtWidgets.QComboBox()
        self.cb_unet.setToolTip("Select the UNet GGUF file. If you just downloaded models, click 'Reload models list'.")
        self.cb_llm = QtWidgets.QComboBox()
        self.cb_llm.setToolTip("Select the text encoder (LLM) GGUF file.")
        self.cb_mmproj = QtWidgets.QComboBox()
        self.cb_mmproj.setToolTip("Select the mmproj/vision projection model (GGUF).")
        self.cb_vae = QtWidgets.QComboBox()
        self.cb_vae.setToolTip("Select the VAE file used for decoding/encoding.")

        ml.addRow("UNet", self.cb_unet)
        ml.addRow("Text encoder", self.cb_llm)
        ml.addRow("mmproj", self.cb_mmproj)
        ml.addRow("VAE", self.cb_vae)

        mv.addWidget(self.models_content)

        # Models actions (kept inside the Models section)
        self.models_buttons = QtWidgets.QWidget()
        mb = QtWidgets.QHBoxLayout(self.models_buttons)
        mb.setContentsMargins(0, 6, 0, 0)
        mb.setSpacing(6)

        self.btn_download = QtWidgets.QPushButton("Download models")
        self.btn_download.setToolTip("Downloads the required Qwen2511 GGUF files into the models folder.")
        self.btn_reload = QtWidgets.QPushButton("Reload models list")
        self.btn_reload.setToolTip("Rescans the models folders and refreshes the dropdowns.")
        _dl_cb = getattr(self, "_run_downloader", None)
        if not callable(_dl_cb):
            _dl_cb = self._noop_downloader
        self.btn_download.clicked.connect(_dl_cb)
        self.btn_reload.clicked.connect(self._reload_model_lists)

        mb.addWidget(self.btn_download)
        mb.addWidget(self.btn_reload)
        mb.addStretch(1)

        mv.addWidget(self.models_buttons)
        layout.addWidget(self.g_models)

        # LoRA (collapsible)
        self.g_lora = QtWidgets.QGroupBox("LoRA")
        self.g_lora.setToolTip("Optional LoRA injection for Qwen2511. Collapse this box if you want a cleaner UI.")
        self.g_lora.setCheckable(True)
        self.g_lora.setChecked(True)
        self.g_lora.toggled.connect(self._on_lora_toggled_slot)

        lv = QtWidgets.QVBoxLayout(self.g_lora)
        lv.setContentsMargins(8, 8, 8, 8)

        self.lora_content = QtWidgets.QWidget()
        lf = QtWidgets.QFormLayout(self.lora_content)
        lf.setLabelAlignment(QtCore.Qt.AlignRight)
        lf.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        lf.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        # Folder picker + refresh
        self.ed_lora_dir = QtWidgets.QLineEdit()
        self.ed_lora_dir.setToolTip("Folder containing LoRA files for Qwen2511.")
        self.ed_lora_dir.setPlaceholderText(DEFAULT_LORA_DIR)

        self.btn_pick_lora_dir = QtWidgets.QToolButton(); self.btn_pick_lora_dir.setText("…")
        self.btn_pick_lora_dir.setToolTip("Browse for a LoRA folder")
        self.btn_pick_lora_dir.clicked.connect(self._pick_lora_dir)

        self.btn_lora_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_lora_refresh.setToolTip("Rescan the LoRA folder and refresh the dropdown.")
        self.btn_lora_refresh.clicked.connect(self._reload_lora_list)

        h_ld = QtWidgets.QHBoxLayout()
        h_ld.addWidget(self.btn_pick_lora_dir)
        h_ld.addWidget(self.ed_lora_dir, 1)
        h_ld.addWidget(self.btn_lora_refresh)
        lf.addRow("LoRA folder", h_ld)

        # LoRA dropdown
        self.cb_lora = QtWidgets.QComboBox()
        self.cb_lora.setToolTip("Select a LoRA file to enable it. Choose (None) to disable LoRA.")
        self.cb_lora.currentIndexChanged.connect(self._on_lora_choice_changed)
        lf.addRow("LoRA", self.cb_lora)

        # Strength (slider + spin)
        self.sl_lora_strength = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_lora_strength.setRange(0, 200)  # 0.00 .. 2.00
        self.sl_lora_strength.setSingleStep(1)
        self.sl_lora_strength.setPageStep(5)
        self.sl_lora_strength.setToolTip("LoRA strength")

        self.sp_lora_strength = QtWidgets.QDoubleSpinBox()
        self.sp_lora_strength.setRange(0.0, 2.0)
        self.sp_lora_strength.setSingleStep(0.05)
        self.sp_lora_strength.setDecimals(2)
        self.sp_lora_strength.setToolTip("LoRA strength")

        self.sl_lora_strength.valueChanged.connect(self._on_lora_strength_slider)
        self.sp_lora_strength.valueChanged.connect(self._on_lora_strength_spin)

        h_ls = QtWidgets.QHBoxLayout()
        h_ls.addWidget(self.sl_lora_strength, 1)
        h_ls.addWidget(self.sp_lora_strength)
        lf.addRow("Strength", h_ls)

        # Multi-angle camera helper (Qwen Image Edit 2511 Multiple-Angles LoRA)
        self.chk_multi_angle = QtWidgets.QCheckBox("Multi-angle camera (Qwen 2511)")
        self.chk_multi_angle.setToolTip(
            "Helper for Qwen-Image-Edit-2511 Multiple-Angles LoRA. Inserts <sks> camera tokens into the prompt."
        )
        try:
            self.chk_multi_angle.toggled.connect(self._on_multi_angle_toggled_slot)
        except Exception:
            pass
        lf.addRow("", self.chk_multi_angle)

        self.w_multi_angle = QtWidgets.QWidget()
        self.w_multi_angle.setVisible(False)
        ma = QtWidgets.QGridLayout(self.w_multi_angle)
        ma.setContentsMargins(0, 0, 0, 0)
        ma.setHorizontalSpacing(6)
        ma.setVerticalSpacing(6)
        ma.setColumnStretch(1, 1)
        ma.setColumnStretch(3, 1)
        ma.setColumnStretch(5, 1)

        self.cb_ma_az = QtWidgets.QComboBox()
        self.cb_ma_az.setToolTip("Azimuth (camera direction)")
        for label, tok in [
            ("0° Front", "front"),
            ("45° Front-Right", "front_right"),
            ("90° Right", "right"),
            ("135° Back-Right", "back_right"),
            ("180° Back", "back"),
            ("225° Back-Left", "back_left"),
            ("270° Left", "left"),
            ("315° Front-Left", "front_left"),
        ]:
            self.cb_ma_az.addItem(label, tok)

        self.cb_ma_el = QtWidgets.QComboBox()
        self.cb_ma_el.setToolTip("Elevation (camera height)")
        for label, tok in [
            ("-30° Low", "low"),
            ("0° Eye level", "eye"),
            ("30° Elevated", "elevated"),
            ("60° High", "high"),
        ]:
            self.cb_ma_el.addItem(label, tok)

        self.cb_ma_dist = QtWidgets.QComboBox()
        self.cb_ma_dist.setToolTip("Distance (camera zoom)")
        for label, tok in [
            ("x0.6 Close", "close"),
            ("x1.0 Medium", "medium"),
            ("x1.8 Wide", "wide"),
        ]:
            self.cb_ma_dist.addItem(label, tok)

        self.lbl_ma_preview = QtWidgets.QLabel("Will insert: <sks> front view eye-level shot medium shot")
        self.lbl_ma_preview.setToolTip("Live preview of the token block that will be inserted/replaced.")
        self.lbl_ma_preview.setStyleSheet("color: #888;")
        self.lbl_ma_preview.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        self.chk_ma_auto_apply = QtWidgets.QCheckBox("Auto-apply")
        self.chk_ma_auto_apply.setToolTip("When enabled, changes to azimuth/elevation/distance immediately update the <sks> token in the prompt.")

        self.btn_ma_apply = QtWidgets.QPushButton("Apply")
        self.btn_ma_apply.setToolTip("Insert/replace the <sks> camera token block in the main prompt.")
        self.btn_ma_apply.clicked.connect(self._apply_multi_angle_to_prompt)

        # Row 1: three dropdowns
        ma.addWidget(QtWidgets.QLabel("Azimuth"), 0, 0)
        ma.addWidget(self.cb_ma_az, 0, 1)
        ma.addWidget(QtWidgets.QLabel("Elevation"), 0, 2)
        ma.addWidget(self.cb_ma_el, 0, 3)
        ma.addWidget(QtWidgets.QLabel("Distance"), 0, 4)
        ma.addWidget(self.cb_ma_dist, 0, 5)

        # Row 2: preview + apply button
        ma.addWidget(self.lbl_ma_preview, 1, 0, 1, 5)
        ma.addWidget(self.btn_ma_apply, 1, 5)

        # Row 3: auto-apply
        ma.addWidget(self.chk_ma_auto_apply, 2, 0, 1, 6)

        try:
            self.cb_ma_az.currentIndexChanged.connect(self._on_multi_angle_combo_changed)
            self.cb_ma_el.currentIndexChanged.connect(self._on_multi_angle_combo_changed)
            self.cb_ma_dist.currentIndexChanged.connect(self._on_multi_angle_combo_changed)
        except Exception:
            pass

        lf.addRow("", self.w_multi_angle)

        lv.addWidget(self.lora_content)
        layout.addWidget(self.g_lora)

        # Paths (collapsible)
        self.g_paths = QtWidgets.QGroupBox("Paths")
        self.g_paths.setToolTip("Pick input images and other paths. Collapse this box if you want a cleaner UI.")
        self.g_paths.setCheckable(True)
        self.g_paths.setChecked(True)
        self.g_paths.toggled.connect(self._on_paths_toggled_slot)

        pv = QtWidgets.QVBoxLayout(self.g_paths)
        pv.setContentsMargins(8, 8, 8, 8)

        self.paths_content = QtWidgets.QWidget()
        gl = QtWidgets.QFormLayout(self.paths_content)
        gl.setLabelAlignment(QtCore.Qt.AlignRight)
        gl.setRowWrapPolicy(QtWidgets.QFormLayout.WrapAllRows)
        gl.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        
        self.chk_use_scene = QtWidgets.QCheckBox("Use scene image (image 1)")
        self.chk_use_scene.setToolTip("When OFF, FrameVision uses a blank canvas as image 1. Useful when you want to build a new image mainly from reference images.")
        self.chk_use_scene.setChecked(True)
        try:
            self.chk_use_scene.toggled.connect(self._on_use_scene_toggled)
        except Exception:
            pass
        gl.addRow("", self.chk_use_scene)

        self.ed_initimg = QtWidgets.QLineEdit()
        self.ed_initimg.setToolTip("Scene / base image (image 1). Tip: Use a reasonably high-res input (720p+) for cleaner edits.")
        self.ed_initimg.setPlaceholderText("Optional when 'Use scene image' is OFF…")
        self.btn_pick_initimg = QtWidgets.QToolButton(); self.btn_pick_initimg.setText("…"); self.btn_pick_initimg.clicked.connect(self._pick_image)
        self.btn_pick_initimg.setToolTip("Browse for a scene image")

        self.thumb_initimg = ClickableLabel()
        self._init_thumb_label(self.thumb_initimg, self.ed_initimg)
        try:
            self.ed_initimg.textChanged.connect(lambda t: self._update_imginfo(t.strip()))
        except Exception:
            pass

        h_img = QtWidgets.QHBoxLayout()
        h_img.addWidget(self.btn_pick_initimg)
        h_img.addWidget(self.thumb_initimg)
        h_img.addWidget(self.ed_initimg, 1)
        gl.addRow("Scene image", h_img)

        self.lbl_imginfo = QtWidgets.QLabel("")
        self.lbl_imginfo.setToolTip("Shows detected scene resolution and a reminder to keep Width/Height aspect similar to avoid crops.")
        self.lbl_imginfo.setStyleSheet("color: #888;")
        gl.addRow("", self.lbl_imginfo)

        self.chk_use_mask = QtWidgets.QCheckBox("Use mask (advanced)")
        self.chk_use_mask.setToolTip("When ON, you can provide a mask to limit where edits are allowed. When OFF, the mask fields are ignored and hidden.")
        self.chk_use_mask.setChecked(False)
        try:
            self.chk_use_mask.toggled.connect(self._on_use_mask_toggled)
        except Exception:
            pass
        gl.addRow("", self.chk_use_mask)

        self.ed_mask = QtWidgets.QLineEdit()
        self.ed_mask.setToolTip("Optional mask for targeted edits. White = editable area (typical). If it behaves backwards, enable Invert mask.")
        self.ed_mask.setPlaceholderText("Optional: mask image (white/black) …")
        self.btn_pick_mask = QtWidgets.QToolButton(); self.btn_pick_mask.setText("…"); self.btn_pick_mask.clicked.connect(self._pick_mask)
        self.btn_pick_mask.setToolTip("Browse for a mask image")

        self.thumb_mask = ClickableLabel()
        self._init_thumb_label(self.thumb_mask, self.ed_mask)

        self.w_mask_row = QtWidgets.QWidget()
        wm = QtWidgets.QHBoxLayout(self.w_mask_row)
        wm.setContentsMargins(0, 0, 0, 0)
        wm.addWidget(self.btn_pick_mask)
        wm.addWidget(self.thumb_mask)
        wm.addWidget(self.ed_mask, 1)
        gl.addRow("Mask", self.w_mask_row)

        self.chk_invert_mask = QtWidgets.QCheckBox("Invert mask")
        self.chk_invert_mask.setToolTip("If your mask behaves backwards, enable this. It writes a temporary inverted mask PNG.")
        gl.addRow("", self.chk_invert_mask)

        # Initialize visibility
        try:
            self._on_use_scene_toggled(bool(self.chk_use_scene.isChecked()), persist=False)
        except Exception:
            pass
        try:
            self._on_use_mask_toggled(bool(self.chk_use_mask.isChecked()), persist=False)
        except Exception:
            pass
        # Reference images (multi-image edit)
        self.ed_ref1 = QtWidgets.QLineEdit()
        self.ed_ref1.setToolTip("Optional reference image 1 (image 2 in prompt when scene is used).")
        self.ed_ref1.setPlaceholderText("Optional: reference image 1 (object/face/clothes)…")
        btn_ref1 = QtWidgets.QToolButton(); btn_ref1.setText("…"); btn_ref1.clicked.connect(lambda: self._pick_ref(self.ed_ref1))
        btn_ref1.setToolTip("Browse for reference image 1")

        self.thumb_ref1 = ClickableLabel()
        self._init_thumb_label(self.thumb_ref1, self.ed_ref1)

        h_ref1 = QtWidgets.QHBoxLayout()
        h_ref1.addWidget(btn_ref1)
        h_ref1.addWidget(self.thumb_ref1)
        h_ref1.addWidget(self.ed_ref1, 1)
        gl.addRow("Ref image 1", h_ref1)

        self.ed_ref2 = QtWidgets.QLineEdit()
        self.ed_ref2.setToolTip("Optional reference image 2 (image 3 in prompt when scene is used).")
        self.ed_ref2.setPlaceholderText("Optional: reference image 2 …")
        btn_ref2 = QtWidgets.QToolButton(); btn_ref2.setText("…"); btn_ref2.clicked.connect(lambda: self._pick_ref(self.ed_ref2))
        btn_ref2.setToolTip("Browse for reference image 2")

        self.thumb_ref2 = ClickableLabel()
        self._init_thumb_label(self.thumb_ref2, self.ed_ref2)

        h_ref2 = QtWidgets.QHBoxLayout()
        h_ref2.addWidget(btn_ref2)
        h_ref2.addWidget(self.thumb_ref2)
        h_ref2.addWidget(self.ed_ref2, 1)
        gl.addRow("Ref image 2", h_ref2)

        self.ed_ref3 = QtWidgets.QLineEdit()
        self.ed_ref3.setToolTip("Optional reference image 3 (image 4 in prompt when scene is used).")
        self.ed_ref3.setPlaceholderText("Optional: reference image 3 …")
        btn_ref3 = QtWidgets.QToolButton(); btn_ref3.setText("…"); btn_ref3.clicked.connect(lambda: self._pick_ref(self.ed_ref3))
        btn_ref3.setToolTip("Browse for reference image 3")

        self.thumb_ref3 = ClickableLabel()
        self._init_thumb_label(self.thumb_ref3, self.ed_ref3)

        h_ref3 = QtWidgets.QHBoxLayout()
        h_ref3.addWidget(btn_ref3)
        h_ref3.addWidget(self.thumb_ref3)
        h_ref3.addWidget(self.ed_ref3, 1)
        gl.addRow("Ref image 3", h_ref3)

        self.lbl_ref_hint = QtWidgets.QLabel("Prompt numbering: Scene=image 1, Ref1=image 2, Ref2=image 3, Ref3=image 4.")
        self.lbl_ref_hint.setStyleSheet("color: #888;")
        gl.addRow("", self.lbl_ref_hint)

        self.chk_ref_increase_index = QtWidgets.QCheckBox("Auto index refs (image 2, image 3, …)")
        self.chk_ref_increase_index.setToolTip("Adds --increase-ref-index so you can refer to refs as image 2 / image 3 / … in the prompt.")
        gl.addRow("", self.chk_ref_increase_index)

        self.chk_disable_ref_resize = QtWidgets.QCheckBox("Disable auto resize ref images")
        self.chk_disable_ref_resize.setToolTip("Adds --disable-auto-resize-ref-image (if supported).")
        gl.addRow("", self.chk_disable_ref_resize)

        pv.addWidget(self.paths_content)
        layout.addWidget(self.g_paths)

        # Prompt
        g_prompt = QtWidgets.QGroupBox("Prompt")
        g_prompt.setToolTip("Your edit instruction and an optional negative prompt.")
        pl = QtWidgets.QVBoxLayout(g_prompt)
        self.ed_prompt = QtWidgets.QPlainTextEdit()
        self.ed_prompt.setToolTip("What should change? Be explicit: 'keep everything the same, only…'. For best results: use a mask. If Strength < 1.00 behaves oddly, keep it at 1.00 and adjust CFG/steps instead.")
        self.ed_prompt.setPlaceholderText("Describe the edit… (example: keep everything the same, only change dress color to red)")
        self.ed_neg = QtWidgets.QLineEdit(); self.ed_neg.setPlaceholderText("Negative prompt (optional)")
        self.ed_neg.setToolTip("Optional: things you do NOT want. Keep short and practical (e.g., 'blurry, extra fingers').")
        pl.addWidget(self.ed_prompt, 1)
        pl.addWidget(self.ed_neg)
        layout.addWidget(g_prompt, 2)

        # Settings
        g_set = QtWidgets.QGroupBox("Settings")
        g_set.setToolTip("Quality, size, and performance options.")
        sl = QtWidgets.QGridLayout(g_set)
        sl.setColumnStretch(1, 1)
        sl.setColumnStretch(3, 1)

        self.sp_steps = QtWidgets.QSpinBox(); self.sp_steps.setRange(1, 200); self.sp_steps.setValue(30)
        self.sp_steps.setToolTip("Number of diffusion steps. More steps = slower, sometimes cleaner. 24–40 is a common range.")
        self.sp_cfg = QtWidgets.QDoubleSpinBox(); self.sp_cfg.setRange(0.0, 30.0); self.sp_cfg.setDecimals(2); self.sp_cfg.setSingleStep(0.05); self.sp_cfg.setValue(2.35)
        self.sp_cfg.setToolTip("Text guidance scale (CFG). Higher follows the prompt more, but can drift away from the source image.")
        self.sp_seed = QtWidgets.QSpinBox(); self.sp_seed.setRange(-1, 2**31 - 1); self.sp_seed.setValue(-1)
        self.sp_seed.setToolTip("Random seed. -1 = random each run. Set a fixed value to reproduce results.")

        # Strength (editable, but 1.00 recommended for reliability)
        self.sl_strength = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sl_strength.setRange(0, 100)
        self.sl_strength.setValue(35)

        self.sp_strength = QtWidgets.QDoubleSpinBox()
        self.sp_strength.setRange(0.0, 1.0); self.sp_strength.setDecimals(2); self.sp_strength.setSingleStep(0.05); self.sp_strength.setValue(0.35)

        # Backward-compat alias (older code may still look for lbl_strength)
        self.lbl_strength = self.sp_strength

        strength_tip = (
            "Strength controls how strongly the edit is applied. "
            "Warning: many Qwen2511 / sd-cli builds work reliably only at 1.00. "
            "If your edit ignores the ref image or fails, set Strength back to 1.00."
        )
        self.sl_strength.setToolTip(strength_tip)
        self.sp_strength.setToolTip(strength_tip)

        self._strength_updating = False

        def _sync_strength_from_slider(v: int):
            try:
                if self._strength_updating:
                    return
                self._strength_updating = True
                self.sp_strength.setValue(float(v) / 100.0)
            finally:
                self._strength_updating = False

        def _sync_strength_from_spin(v: float):
            try:
                if self._strength_updating:
                    return
                self._strength_updating = True
                self.sl_strength.setValue(int(round(float(v) * 100.0)))
            finally:
                self._strength_updating = False

        self.sl_strength.valueChanged.connect(_sync_strength_from_slider)
        self.sp_strength.valueChanged.connect(_sync_strength_from_spin)

        self.strength_row = QtWidgets.QWidget()
        _sr = QtWidgets.QHBoxLayout(self.strength_row)
        _sr.setContentsMargins(0, 0, 0, 0)
        _sr.setSpacing(6)
        _sr.addWidget(self.sl_strength, 1)
        _sr.addWidget(self.sp_strength, 0)

        self.sp_w = QtWidgets.QSpinBox(); self.sp_w.setRange(64, 4096); self.sp_w.setSingleStep(64); self.sp_w.setValue(1024)
        self.sp_w.setToolTip("Output width (multiple of 64). Keep aspect similar to input to avoid crop.")
        self.sp_h = QtWidgets.QSpinBox(); self.sp_h.setRange(64, 4096); self.sp_h.setSingleStep(64); self.sp_h.setValue(576)
        self.sp_h.setToolTip("Output height (multiple of 64). Keep aspect similar to input to avoid crop.")

        self.cb_sampling = QtWidgets.QComboBox()
        self.cb_sampling.setToolTip("Sampler / sampling method. If results look unstable, try euler or heun.")
        self.cb_sampling.addItems(["euler","euler_a","heun","dpm2","dpm++2s_a","dpm++2m","dpm++2mv2","ipndm","ipndm_v","lcm","ddim_trailing","tcd"])
        self.cb_sampling.setCurrentText("euler_a")

        # Aspect helper
        self.chk_auto_aspect = QtWidgets.QCheckBox("Auto size from input aspect")
        self.chk_auto_aspect.setToolTip("Prevents square-crop. Uses the input image aspect ratio and snaps to multiples of 64.")
        self.cb_base = QtWidgets.QComboBox()
        self.cb_base.setToolTip("Base size used by Auto size helper. Larger base = larger output (and more VRAM/time).")
        self.cb_base.addItems(["512", "768", "1024"])
        self.cb_base.setCurrentText("1024")
        self.btn_apply_aspect = QtWidgets.QPushButton("Apply")
        self.btn_apply_aspect.setToolTip("Apply Auto size now using the current input image.")
        self.btn_apply_aspect.clicked.connect(self._apply_aspect_now)
        aspect_row = QtWidgets.QHBoxLayout()
        aspect_row.addWidget(self.chk_auto_aspect)
        aspect_row.addWidget(QtWidgets.QLabel("Base:"))
        aspect_row.addWidget(self.cb_base)
        aspect_row.addWidget(self.btn_apply_aspect)
        aspect_row.addStretch(1)

        self.sp_shift = QtWidgets.QDoubleSpinBox(); self.sp_shift.setRange(0.0, 30.0); self.sp_shift.setDecimals(2); self.sp_shift.setSingleStep(0.05); self.sp_shift.setValue(2.15)
        self.sp_shift.setToolTip("Flow/shift parameter (if your sd-cli build supports it). Default is usually fine.")

        # Low VRAM options
        self.chk_vae_tiling = QtWidgets.QCheckBox("VAE tiling (low VRAM)")
        self.chk_vae_tiling.setToolTip("Splits VAE encode/decode into tiles to reduce VRAM. Only affects the VAE (not the diffusion model). Use this if you hit out-of-memory during VAE encode/decode. Smaller tiles use less VRAM but are slower and can create seams. Start with 256x256; if needed try 128x128 → 64x64 → 32x32 (last resort).")
        self.cb_vae_tile_size = QtWidgets.QComboBox()
        self.cb_vae_tile_size.addItems(["32x32", "64x64", "128x128", "256x256", "512x512"])
        self.cb_vae_tile_size.setCurrentText("256x256")
        self.cb_vae_tile_size.setToolTip("VAE tile size. Smaller tiles reduce VRAM usage but slow down processing and may introduce seams. Recommended: 256x256; drop to 128x128 or 64x64 if you run out of VRAM; 32x32 is last resort. 512x512 is fastest but uses more VRAM.")
        self.sp_vae_tile_overlap = QtWidgets.QDoubleSpinBox()
        self.sp_vae_tile_overlap.setToolTip("Overlap between VAE tiles. Higher overlap can reduce seams, but costs time.")
        self.sp_vae_tile_overlap.setRange(0.0, 0.95); self.sp_vae_tile_overlap.setDecimals(2); self.sp_vae_tile_overlap.setSingleStep(0.05); self.sp_vae_tile_overlap.setValue(0.50)

        self.chk_offload = QtWidgets.QCheckBox("Offload weights to CPU (very slow)")
        self.chk_offload.setToolTip("Moves model weights to CPU to save VRAM. Expect a big speed hit.")
        self.chk_mmap = QtWidgets.QCheckBox("mmap")
        self.chk_mmap.setToolTip("Memory-map model files from disk. Can reduce RAM spikes and speed up startup on some systems.")
        self.chk_vae_on_cpu = QtWidgets.QCheckBox("VAE on CPU (slow)")
        self.chk_vae_on_cpu.setToolTip("Runs VAE on CPU to save VRAM. Slower.")
        self.chk_clip_on_cpu = QtWidgets.QCheckBox("CLIP on CPU (if available)")
        self.chk_clip_on_cpu.setToolTip("Runs CLIP/text encoder parts on CPU (if supported). Saves VRAM, may be slower.")
        self.chk_diffusion_fa = QtWidgets.QCheckBox("Diffusion flash-attn (if available)")
        self.chk_diffusion_fa.setToolTip("Enables flash-attention in diffusion if your sd-cli build supports it. Can speed up.")

        r = 0
        sl.addWidget(QtWidgets.QLabel("Steps"), r, 0); sl.addWidget(self.sp_steps, r, 1)
        sl.addWidget(QtWidgets.QLabel("CFG"), r, 2); sl.addWidget(self.sp_cfg, r, 3)
        r += 1
        sl.addWidget(QtWidgets.QLabel("Strength"), r, 0); sl.addWidget(self.strength_row, r, 1)
        sl.addWidget(QtWidgets.QLabel("Seed"), r, 2); sl.addWidget(self.sp_seed, r, 3)
        r += 1
        sl.addWidget(QtWidgets.QLabel("Sampling"), r, 0); sl.addWidget(self.cb_sampling, r, 1)
        r += 1
        sl.addWidget(QtWidgets.QLabel("Width"), r, 0); sl.addWidget(self.sp_w, r, 1)
        sl.addWidget(QtWidgets.QLabel("Height"), r, 2); sl.addWidget(self.sp_h, r, 3)

        r += 1
        sl.addWidget(QtWidgets.QLabel("Flow shift"), r, 0); sl.addWidget(self.sp_shift, r, 1)

        # Advanced (collapsible): aspect helper + low-VRAM / performance flags
        r += 1
        self.g_adv = QtWidgets.QGroupBox("Advanced")
        self.g_adv.setToolTip("Low-VRAM and performance options. Collapse this box for a cleaner UI.")
        self.g_adv.setCheckable(True)
        self.g_adv.setChecked(False)
        self.g_adv.toggled.connect(self._on_adv_toggled_slot)

        av = QtWidgets.QVBoxLayout(self.g_adv)
        av.setContentsMargins(8, 8, 8, 8)

        self.adv_content = QtWidgets.QWidget()
        al = QtWidgets.QGridLayout(self.adv_content)
        al.setColumnStretch(0, 1)
        al.setColumnStretch(1, 1)
        al.setColumnStretch(2, 0)
        al.setColumnStretch(3, 1)

        # Aspect helper
        al.addLayout(aspect_row, 0, 0, 1, 4)

        ar = 1
        al.addWidget(self.chk_vae_tiling, ar, 0, 1, 2)
        al.addWidget(QtWidgets.QLabel("Tile size"), ar, 2); al.addWidget(self.cb_vae_tile_size, ar, 3)

        ar += 1
        al.addWidget(self.chk_mmap, ar, 0, 1, 2)
        al.addWidget(QtWidgets.QLabel("Tile overlap"), ar, 2); al.addWidget(self.sp_vae_tile_overlap, ar, 3)

        ar += 1
        al.addWidget(self.chk_offload, ar, 0, 1, 2)
        al.addWidget(self.chk_vae_on_cpu, ar, 2, 1, 2)

        ar += 1
        al.addWidget(self.chk_clip_on_cpu, ar, 0, 1, 2)
        al.addWidget(self.chk_diffusion_fa, ar, 2, 1, 2)

        av.addWidget(self.adv_content)
        sl.addWidget(self.g_adv, r, 0, 1, 4)

        layout.addWidget(g_set)

        # Log (collapsible) – inside the scroll area, above the sticky buttons.
        self.g_log = QtWidgets.QGroupBox("Log")
        self.g_log.setToolTip("Process log (newest messages appear at the top). Collapse this box for a cleaner UI.")
        self.g_log.setCheckable(True)
        self.g_log.setChecked(True)
        self.g_log.toggled.connect(self._on_log_toggled_slot)

        lv = QtWidgets.QVBoxLayout(self.g_log)
        lv.setContentsMargins(8, 8, 8, 8)

        self.log_content = QtWidgets.QWidget()
        lcl = QtWidgets.QVBoxLayout(self.log_content)
        lcl.setContentsMargins(0, 0, 0, 0)
        lcl.setSpacing(6)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setToolTip("Process log (newest messages appear at the top). If something fails, copy the top section into a bug report.")
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(8000)
        self.log.setMinimumHeight(220)
        # Keep the log from consuming the entire scroll area (it has its own scrollbar).
        self.log.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        lcl.addWidget(self.log)

        # Log controls (inside the Log section)
        self.btn_clear_log = QtWidgets.QPushButton("Clear log")
        self.btn_clear_log.setToolTip("Clears the log output (new messages will still appear at the top).")
        self.btn_clear_log.clicked.connect(self._clear_log)

        log_btn_row = QtWidgets.QHBoxLayout()
        log_btn_row.addStretch(1)
        log_btn_row.addWidget(self.btn_clear_log)
        lcl.addLayout(log_btn_row)
        lv.addWidget(self.log_content)
        layout.addWidget(self.g_log)

        # A bit of padding at the bottom so the last controls aren't hidden by the sticky bar.
        layout.addSpacing(8)

        # Sticky bottom buttons bar (always visible, does not scroll).
        self.button_bar = QtWidgets.QWidget()
        outer.addWidget(self.button_bar, 0)

        bar = QtWidgets.QVBoxLayout(self.button_bar)
        bar.setContentsMargins(12, 8, 12, 12)
        bar.setSpacing(6)

        top_row = QtWidgets.QHBoxLayout()
        bottom_row = QtWidgets.QHBoxLayout()

        self.btn_view_results = QtWidgets.QPushButton("View results")
        self.btn_view_results.setToolTip("Open output/qwen2511 in Media Explorer.")
        self.btn_play_last = QtWidgets.QPushButton("Play last result")
        self.btn_play_last.setToolTip("Plays the newest file in output/qwen2511 using the internal media player.")
        self.chk_auto_show_results = QtWidgets.QCheckBox("Auto play result after run")
        self.chk_auto_show_results.setToolTip("When enabled, automatically plays the produced file in the internal media player after a successful run (same as 'Play last result').")
        self.chk_use_queue = QtWidgets.QCheckBox("Use queue")
        self.chk_use_queue.setToolTip("When enabled, clicking Run will add this job to the Queue instead of running sd-cli immediately.")
        self.chk_use_queue.toggled.connect(self._on_use_queue_toggled)
        self.btn_run = QtWidgets.QPushButton("Run sd-cli")
        self.btn_run.setToolTip("Runs sd-cli with the current settings. Output will be saved to output/qwen2511/.")
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setToolTip("Stops the currently running process (downloader or sd-cli).")
        self.btn_cancel.setEnabled(False)

        self.btn_view_results.clicked.connect(self._open_results_folder)
        self.btn_play_last.clicked.connect(self._play_last_result)
        self.btn_run.clicked.connect(self._run_sdcli)
        self.btn_cancel.clicked.connect(self._cancel_running_process)

        top_row.addWidget(self.btn_view_results)
        top_row.addWidget(self.btn_play_last)
        top_row.addWidget(self.chk_auto_show_results)
        top_row.addStretch(1)

        bottom_row.addWidget(self.chk_use_queue)
        bottom_row.addWidget(self.btn_run)
        bottom_row.addWidget(self.btn_cancel)
        bottom_row.addStretch(1)

        bar.addLayout(top_row)
        bar.addLayout(bottom_row)

    def _on_models_toggled_slot(self, checked: bool):
        self._on_models_toggled(bool(checked), persist=True)

    def _on_models_toggled(self, checked: bool, persist: bool = False):
        # Collapse/expand the Models group and persist the open/close state.
        try:
            if hasattr(self, "models_content") and self.models_content is not None:
                self.models_content.setVisible(bool(checked))
            if hasattr(self, "models_buttons") and self.models_buttons is not None:
                self.models_buttons.setVisible(bool(checked))
        except Exception:
            pass


    def _on_lora_toggled_slot(self, checked: bool):
        self._on_lora_toggled(bool(checked), persist=True)

    def _on_lora_toggled(self, checked: bool, persist: bool = False):
        # Collapse/expand the LoRA group and persist the open/close state.
        try:
            if hasattr(self, "lora_content") and self.lora_content is not None:
                self.lora_content.setVisible(bool(checked))
        except Exception:
            pass

        try:
            self._settings["lora_box_open"] = bool(checked)
            if persist:
                _write_json(SETSAVE_PATH, self._settings)
        except Exception:
            pass


    def _on_multi_angle_toggled_slot(self, checked: bool):
        self._on_multi_angle_toggled(bool(checked), persist=True)

    def _on_multi_angle_toggled(self, checked: bool, persist: bool = False):
        # Show/hide helper controls.
        try:
            if hasattr(self, "w_multi_angle") and self.w_multi_angle is not None:
                self.w_multi_angle.setVisible(bool(checked))
        except Exception:
            pass
        try:
            self._update_multi_angle_preview()
        except Exception:
            pass
        # Persist via the same JSON settings system (best-effort). Autosave handles this,
        # but we also update the in-memory dict so other code can read it immediately.
        try:
            self._settings["multi_angle_camera"] = bool(checked)
            if persist:
                _write_json(SETSAVE_PATH, self._settings)
        except Exception:
            pass

    def _multi_angle_token_string(self) -> str:
        """Build the exact Qwen-Image-Edit-2511 Multiple-Angles LoRA camera token string."""
        az_key = "front"
        el_key = "eye"
        dist_key = "medium"
        try:
            az_key = str(self.cb_ma_az.currentData() or az_key)
        except Exception:
            pass
        try:
            el_key = str(self.cb_ma_el.currentData() or el_key)
        except Exception:
            pass
        try:
            dist_key = str(self.cb_ma_dist.currentData() or dist_key)
        except Exception:
            pass

        az_map = {
            "front": "front view",
            "front_right": "front-right quarter view",
            "right": "right side view",
            "back_right": "back-right quarter view",
            "back": "back view",
            "back_left": "back-left quarter view",
            "left": "left side view",
            "front_left": "front-left quarter view",
        }
        el_map = {
            "low": "low-angle shot",
            "eye": "eye-level shot",
            "elevated": "elevated shot",
            "high": "high-angle shot",
        }
        dist_map = {
            "close": "close-up",
            "medium": "medium shot",
            "wide": "wide shot",
        }

        az_phrase = az_map.get(az_key, az_key)
        el_phrase = el_map.get(el_key, el_key)
        dist_phrase = dist_map.get(dist_key, dist_key)

        return f"<sks> {az_phrase} {el_phrase} {dist_phrase}"


    def _on_multi_angle_combo_changed(self, *args):
        """Dropdown change handler for multi-angle camera helper."""
        try:
            self._update_multi_angle_preview()
        except Exception:
            pass

        # Only auto-apply in normal interactive use (never during initial UI load).
        if bool(getattr(self, "_ui_loading", False)):
            return

        try:
            chk = getattr(self, "chk_ma_auto_apply", None)
            if chk is not None and bool(chk.isChecked()):
                self._apply_multi_angle_to_prompt()
        except Exception:
            pass


    def _update_multi_angle_preview(self):
        try:
            if hasattr(self, "lbl_ma_preview") and self.lbl_ma_preview is not None:
                self.lbl_ma_preview.setText(f"Will insert: {self._multi_angle_token_string()}")
        except Exception:
            pass


    def _apply_multi_angle_to_prompt(self):
        # Insert the token string at the VERY start of the main prompt.
        # If any <sks> block exists anywhere, remove it first (do not allow multiple blocks).
        if not hasattr(self, "ed_prompt") or self.ed_prompt is None:
            return

        new_block = self._multi_angle_token_string()
        try:
            txt = self.ed_prompt.toPlainText()
        except Exception:
            txt = ""
        # Remove any existing <sks> camera token block (best-effort).
        # Replacement logic: treat the existing camera token block as a substring
        # starting with '<sks>' and continuing until the end of that line / next newline.
        try:
            pat = re.compile(r"<sks>[^\n]*", re.IGNORECASE)
            cleaned = re.sub(pat, "", txt)
        except Exception:
            cleaned = txt

        # Prefix new block.
        cleaned = cleaned if cleaned is not None else ""
        add_space = bool(cleaned) and (not cleaned[:1].isspace())
        prefix = new_block + (" " if add_space else "")
        final = prefix + cleaned

        try:
            self._autosave_pause()
        except Exception:
            pass

        try:
            self.ed_prompt.setPlainText(final)
            # Put the caret after the inserted block.
            try:
                cur = self.ed_prompt.textCursor()
                cur.setPosition(len(prefix))
                self.ed_prompt.setTextCursor(cur)
            except Exception:
                pass
        finally:
            try:
                self._autosave_resume()
                self._schedule_autosave()
            except Exception:
                pass

        try:
            self._toast("Inserted multi-angle camera tokens.")
        except Exception:
            pass




    def _on_paths_toggled_slot(self, checked: bool):
        self._on_paths_toggled(bool(checked), persist=True)

    def _on_paths_toggled(self, checked: bool, persist: bool = False):
        # Collapse/expand the Paths group and persist the open/close state.
        try:
            if hasattr(self, "paths_content") and self.paths_content is not None:
                self.paths_content.setVisible(bool(checked))
        except Exception:
            pass

        try:
            self._settings["paths_box_open"] = bool(checked)
            if persist:
                _write_json(SETSAVE_PATH, self._settings)
        except Exception:
            pass


    def _on_adv_toggled_slot(self, checked: bool):
        self._on_adv_toggled(bool(checked), persist=True)

    def _on_adv_toggled(self, checked: bool, persist: bool = False):
        # Collapse/expand the Advanced group and persist the open/close state.
        try:
            if hasattr(self, "adv_content") and self.adv_content is not None:
                self.adv_content.setVisible(bool(checked))
        except Exception:
            pass

        try:
            self._settings["advanced_box_open"] = bool(checked)
            if persist:
                _write_json(SETSAVE_PATH, self._settings)
        except Exception:
            pass


    def _on_log_toggled_slot(self, checked: bool):
        self._on_log_toggled(bool(checked), persist=True)

    def _on_log_toggled(self, checked: bool, persist: bool = False):
        # Collapse/expand the Log group and persist the open/close state.
        try:
            if hasattr(self, "log_content") and self.log_content is not None:
                self.log_content.setVisible(bool(checked))
        except Exception:
            pass

        try:
            self._settings["log_box_open"] = bool(checked)
            if persist:
                _write_json(SETSAVE_PATH, self._settings)
        except Exception:
            pass

    def _update_imginfo(self, path: str):
        if not path or not os.path.isfile(path):
            self.lbl_imginfo.setText("")
            return
        img = QImage(path)
        if img.isNull():
            self.lbl_imginfo.setText("")
            return
        self.lbl_imginfo.setText(f"Input size: {img.width()}x{img.height()}  (Tip: you can also load same ref image in here and play with strength to get different results)")



    def _init_thumb_label(self, lbl: ClickableLabel, line_edit: QtWidgets.QLineEdit) -> None:
        """Configure a tiny thumbnail next to a path box."""
        if lbl is None or line_edit is None:
            return
        try:
            lbl.setFixedSize(34, 34)
            lbl.setMinimumSize(34, 34)
            lbl.setMaximumSize(34, 34)
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setToolTip("Click to preview")
            lbl.setCursor(QtCore.Qt.PointingHandCursor)
            lbl.setStyleSheet(
                "QLabel {"
                "border: 1px solid rgba(255,255,255,40);"
                "border-radius: 6px;"
                "background: rgba(0,0,0,25);"
                "}"
            )
        except Exception:
            pass

        try:
            # Update thumbnail whenever the path changes.
            line_edit.textChanged.connect(lambda t: self._update_thumb(lbl, t))
        except Exception:
            pass

        try:
            # Click opens preview (popup auto-closes on outside click).
            lbl.clicked.connect(lambda: self._show_image_preview(line_edit.text().strip()))
        except Exception:
            pass

        # Initial render (in case settings already filled)
        try:
            self._update_thumb(lbl, line_edit.text().strip())
        except Exception:
            pass

    def _update_thumb(self, lbl: QtWidgets.QLabel, path: str) -> None:
        """Render a tiny thumbnail from an image file (best-effort)."""
        if lbl is None:
            return
        p = (path or "").strip()
        if not p or not os.path.isfile(p):
            try:
                lbl.setPixmap(QPixmap())
                lbl.setText("·")
            except Exception:
                pass
            return

        try:
            pix = QPixmap(p)
            if pix.isNull():
                lbl.setPixmap(QPixmap())
                lbl.setText("·")
                return
            s = lbl.size()
            pad = 2
            tw = max(8, s.width() - pad * 2)
            th = max(8, s.height() - pad * 2)
            pix = pix.scaled(tw, th, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            lbl.setText("")
            lbl.setPixmap(pix)
        except Exception:
            try:
                lbl.setPixmap(QPixmap())
                lbl.setText("·")
            except Exception:
                pass

    def _show_image_preview(self, path: str) -> None:
        p = (path or "").strip()
        if not p or not os.path.isfile(p):
            try:
                self._toast("No image selected.")
            except Exception:
                pass
            return
        try:
            pop = ImagePreviewPopup(p, parent=self.window())
            pop.show()
        except Exception:
            try:
                QtWidgets.QMessageBox.information(self, "Preview", "Could not open preview for this file.")
            except Exception:
                pass

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

    def _pick_lora_dir(self):
        """Pick a LoRA folder.

        Reasonable default: DEFAULT_LORA_DIR.
        """
        start = self.ed_lora_dir.text().strip() if hasattr(self, "ed_lora_dir") else ""
        if not start:
            start = DEFAULT_LORA_DIR
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select LoRA folder", start)
        if d:
            try:
                self.ed_lora_dir.setText(d)
            except Exception:
                pass
            try:
                self._reload_lora_list()
            except Exception:
                pass

    def _reload_lora_list(self):
        """Rescan LoRA folder and refresh dropdown."""
        if not hasattr(self, "cb_lora"):
            return
        try:
            lora_dir = (self.ed_lora_dir.text().strip() if hasattr(self, "ed_lora_dir") else "").strip()
        except Exception:
            lora_dir = ""
        if not lora_dir:
            lora_dir = DEFAULT_LORA_DIR

        prev = ""
        try:
            prev = str(self.cb_lora.currentText() or "").strip()
        except Exception:
            prev = ""

        try:
            self.cb_lora.blockSignals(True)
        except Exception:
            pass

        try:
            self.cb_lora.clear()
            self.cb_lora.addItem("(None)")
            for fn in _list_lora_files(lora_dir):
                self.cb_lora.addItem(fn)

            # Restore selection if possible.
            if prev:
                for i in range(self.cb_lora.count()):
                    if str(self.cb_lora.itemText(i)).strip() == prev:
                        self.cb_lora.setCurrentIndex(i)
                        break
        finally:
            try:
                self.cb_lora.blockSignals(False)
            except Exception:
                pass

        try:
            self._on_lora_choice_changed()
        except Exception:
            pass

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

    def _pick_ref(self, target: QtWidgets.QLineEdit):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select reference image", APP_ROOT, "Images (*.png *.jpg *.jpeg *.webp)")
        if fn and target is not None:
            target.setText(fn)

    def _reload_model_lists(self):
        # Reload dropdowns without accidentally overwriting saved selections.
        prev_unet = self.cb_unet.currentData() if hasattr(self, "cb_unet") else None
        prev_llm = self.cb_llm.currentData() if hasattr(self, "cb_llm") else None
        prev_mmproj = self.cb_mmproj.currentData() if hasattr(self, "cb_mmproj") else None
        prev_vae = self.cb_vae.currentData() if hasattr(self, "cb_vae") else None

        self._autosave_pause()
        try:
            self.cb_unet.blockSignals(True); self.cb_llm.blockSignals(True); self.cb_mmproj.blockSignals(True); self.cb_vae.blockSignals(True)

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

            def restore(cb: QtWidgets.QComboBox, wanted):
                if not wanted:
                    return
                for i in range(cb.count()):
                    if cb.itemData(i) == wanted:
                        cb.setCurrentIndex(i)
                        return

            restore(self.cb_unet, prev_unet)
            restore(self.cb_llm, prev_llm)
            restore(self.cb_mmproj, prev_mmproj)
            restore(self.cb_vae, prev_vae)
        finally:
            try:
                self.cb_unet.blockSignals(False); self.cb_llm.blockSignals(False); self.cb_mmproj.blockSignals(False); self.cb_vae.blockSignals(False)
            except Exception:
                pass
            self._autosave_resume()



    def _load_defaults(self):
        self._reload_model_lists()
        d = default_model_paths()
        self._settings.setdefault("sdcli_path", DEFAULT_SDCLI)
        self._settings.setdefault("models_box_open", True)
        self._settings.setdefault("paths_box_open", True)
        self._settings.setdefault("advanced_box_open", False)
        self._settings.setdefault("log_box_open", True)
        self._settings.setdefault("last_init_img", "")
        self._settings.setdefault("last_mask_img", "")
        self._settings.setdefault("invert_mask", False)
        self._settings.setdefault("ref_img_2", "")
        self._settings.setdefault("ref_img_3", "")
        self._settings.setdefault("ref_img_4", "")
        self._settings.setdefault("ref_increase_index", True)
        self._settings.setdefault("ref_disable_auto_resize", False)
        self._settings.setdefault("prompt", "")
        self._settings.setdefault("negative_prompt", "")
        self._settings.setdefault("unet_path", d["unet"] or "")
        self._settings.setdefault("llm_path", d["llm"] or "")
        self._settings.setdefault("mmproj_path", d["mmproj"] or "")
        self._settings.setdefault("vae_path", d["vae"] or "")
        self._settings.setdefault("steps", 25)
        self._settings.setdefault("cfg", 2.35)
        self._settings.setdefault("strength", 0.9)
        self._settings.setdefault("seed", -1)
        self._settings.setdefault("width", 1024)
        self._settings.setdefault("height", 576)
        self._settings.setdefault("sampling_method", "euler")
        self._settings.setdefault("shift", 2.15)
        self._settings.setdefault("auto_aspect", True)
        self._settings.setdefault("auto_aspect_base", 1024)
        self._settings.setdefault("auto_show_results", False)
        self._settings.setdefault("use_queue", False)

        # LoRA defaults
        self._settings.setdefault("lora_box_open", True)
        self._settings.setdefault("lora_dir", DEFAULT_LORA_DIR)
        self._settings.setdefault("lora_name", "")
        self._settings.setdefault("lora_strength", 1.0)

        # Multi-angle camera helper defaults
        self._settings.setdefault("multi_angle_camera", False)
        self._settings.setdefault("multi_angle_azimuth", "front")
        self._settings.setdefault("multi_angle_elevation", "eye")
        self._settings.setdefault("multi_angle_distance", "medium")
        self._settings.setdefault("multi_angle_auto_apply", False)

        # low-vram defaults
        self._settings.setdefault("vae_tiling", True)
        self._settings.setdefault("vae_tile_size", "128x128")
        self._settings.setdefault("vae_tile_overlap", 0.50)
        self._settings.setdefault("offload_to_cpu", False)
        self._settings.setdefault("mmap", True)
        self._settings.setdefault("vae_on_cpu", False)
        self._settings.setdefault("clip_on_cpu", False)
        self._settings.setdefault("diffusion_fa", False)

        # Populate LoRA list once (best-effort). Selection is applied in _apply_settings_to_ui.
        try:
            self._reload_lora_list()
        except Exception:
            pass

    def _apply_settings_to_ui(self):
        s = self._settings
        checked = bool(s.get("models_box_open", True))
        if hasattr(self, "g_models") and self.g_models is not None:
            try:
                self.g_models.blockSignals(True)
                self.g_models.setChecked(checked)
                self.g_models.blockSignals(False)
            except Exception:
                pass
            self._on_models_toggled(checked, persist=False)

        # LoRA box open state
        lora_checked = bool(s.get("lora_box_open", True))
        if hasattr(self, "g_lora") and self.g_lora is not None:
            try:
                self.g_lora.blockSignals(True)
                self.g_lora.setChecked(lora_checked)
                self.g_lora.blockSignals(False)
            except Exception:
                pass
            self._on_lora_toggled(lora_checked, persist=False)

        # LoRA values
        try:
            self.ed_lora_dir.setText(str(s.get("lora_dir", DEFAULT_LORA_DIR) or ""))
        except Exception:
            pass
        try:
            self._reload_lora_list()
        except Exception:
            pass
        try:
            target = str(s.get("lora_name", "") or "").strip()
            if target:
                # Prefer exact filename match. If only stem was saved, match by stem.
                for i in range(self.cb_lora.count()):
                    t = str(self.cb_lora.itemText(i) or "").strip()
                    if t == target or os.path.splitext(t)[0] == os.path.splitext(target)[0]:
                        self.cb_lora.setCurrentIndex(i)
                        break
            else:
                self.cb_lora.setCurrentIndex(0)
        except Exception:
            pass
        try:
            v = float(s.get("lora_strength", 1.0))
            self.sp_lora_strength.setValue(v)
            self.sl_lora_strength.setValue(int(round(v * 100.0)))
        except Exception:
            pass
        try:
            self._on_lora_choice_changed()
        except Exception:
            pass

        # Multi-angle camera helper (Qwen 2511)
        try:
            enabled = bool(s.get("multi_angle_camera", False))
            if hasattr(self, "chk_multi_angle") and self.chk_multi_angle is not None:
                self.chk_multi_angle.setChecked(enabled)

            def _set_combo_data(cb: QtWidgets.QComboBox, tok: str):
                if cb is None:
                    return
                t = str(tok or "").strip()
                if not t:
                    return
                for i in range(cb.count()):
                    if str(cb.itemData(i) or "") == t:
                        cb.setCurrentIndex(i)
                        return

            _set_combo_data(getattr(self, "cb_ma_az", None), s.get("multi_angle_azimuth", "front"))
            _set_combo_data(getattr(self, "cb_ma_el", None), s.get("multi_angle_elevation", "eye"))
            _set_combo_data(getattr(self, "cb_ma_dist", None), s.get("multi_angle_distance", "medium"))
            try:
                if hasattr(self, "chk_ma_auto_apply") and self.chk_ma_auto_apply is not None:
                    self.chk_ma_auto_apply.setChecked(bool(s.get("multi_angle_auto_apply", False)))
            except Exception:
                pass
            try:
                self._on_multi_angle_toggled(enabled, persist=False)
            except Exception:
                pass
        except Exception:
            pass



        paths_checked = bool(s.get("paths_box_open", True))
        if hasattr(self, "g_paths") and self.g_paths is not None:
            try:
                self.g_paths.blockSignals(True)
                self.g_paths.setChecked(paths_checked)
                self.g_paths.blockSignals(False)
            except Exception:
                pass
            self._on_paths_toggled(paths_checked, persist=False)


        adv_checked = bool(s.get("advanced_box_open", False))
        if hasattr(self, "g_adv") and self.g_adv is not None:
            try:
                self.g_adv.blockSignals(True)
                self.g_adv.setChecked(adv_checked)
                self.g_adv.blockSignals(False)
            except Exception:
                pass
            self._on_adv_toggled(adv_checked, persist=False)


        log_checked = bool(s.get("log_box_open", True))
        if hasattr(self, "g_log") and self.g_log is not None:
            try:
                self.g_log.blockSignals(True)
                self.g_log.setChecked(log_checked)
                self.g_log.blockSignals(False)
            except Exception:
                pass
            self._on_log_toggled(log_checked, persist=False)

        self.ed_sdcli.setText(s.get("sdcli_path", DEFAULT_SDCLI))
        self.ed_initimg.setText(s.get("last_init_img", ""))
        try:
            use_scene = bool(s.get("use_scene_image", True))
            if hasattr(self, "chk_use_scene"):
                self.chk_use_scene.setChecked(use_scene)
            self._on_use_scene_toggled(use_scene, persist=False)
        except Exception:
            pass
        self._update_imginfo(self.ed_initimg.text().strip())
        self.ed_mask.setText(s.get("last_mask_img", ""))
        try:
            use_mask = bool(s.get("use_mask", bool(s.get("last_mask_img", ""))))
            if hasattr(self, "chk_use_mask"):
                self.chk_use_mask.setChecked(use_mask)
            self._on_use_mask_toggled(use_mask, persist=False)
        except Exception:
            pass
        self.chk_invert_mask.setChecked(bool(s.get("invert_mask", False)))
        self.ed_ref1.setText(s.get("ref_img_1", s.get("ref_img_2", "")))
        self.ed_ref2.setText(s.get("ref_img_2", s.get("ref_img_3", "")))
        self.ed_ref3.setText(s.get("ref_img_3", s.get("ref_img_4", "")))
        self.chk_ref_increase_index.setChecked(bool(s.get("ref_increase_index", True)))
        self.chk_disable_ref_resize.setChecked(bool(s.get("ref_disable_auto_resize", False)))

        self.ed_prompt.setPlainText(s.get("prompt", ""))
        self.ed_neg.setText(s.get("negative_prompt", ""))

        self.sp_steps.setValue(int(s.get("steps", 30)))
        self.sp_cfg.setValue(float(s.get("cfg", 2.35)))
        if hasattr(self, "sp_strength"):
            try:
                self.sp_strength.setValue(float(s.get("strength", 0.35)))
            except Exception:
                pass
        self.sp_seed.setValue(int(s.get("seed", -1)))
        self.sp_w.setValue(int(s.get("width", 1024)))
        self.sp_h.setValue(int(s.get("height", 576)))
        self.cb_sampling.setCurrentText(str(s.get("sampling_method", "euler_a")))
        self.sp_shift.setValue(float(s.get("shift", 2.15)))

        self.chk_auto_aspect.setChecked(bool(s.get("auto_aspect", True)))
        if hasattr(self, "chk_auto_show_results"):
            try:
                self.chk_auto_show_results.setChecked(bool(s.get("auto_show_results", False)))
            except Exception:
                pass
        if hasattr(self, "chk_use_queue"):
            try:
                self.chk_use_queue.setChecked(bool(s.get("use_queue", False)))
            except Exception:
                pass
            try:
                self._on_use_queue_toggled(bool(self.chk_use_queue.isChecked()))
            except Exception:
                pass
        
        base = int(s.get("auto_aspect_base", 1024))
        self.cb_base.setCurrentText(str(base))

        self.chk_vae_tiling.setChecked(bool(s.get("vae_tiling", True)))
        self.cb_vae_tile_size.setCurrentText(str(s.get("vae_tile_size", "256x256")))
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

    def _autosave_pause(self):
        try:
            self._autosave_suspended = int(getattr(self, "_autosave_suspended", 0)) + 1
        except Exception:
            self._autosave_suspended = 1

    def _autosave_resume(self):
        try:
            self._autosave_suspended = max(0, int(getattr(self, "_autosave_suspended", 0)) - 1)
        except Exception:
            self._autosave_suspended = 0

    def _schedule_autosave(self, *args):
        # Debounced autosave to avoid writing JSON on every keystroke.
        if bool(getattr(self, "_ui_loading", False)):
            return
        if int(getattr(self, "_autosave_suspended", 0)) > 0:
            return
        t = getattr(self, "_autosave_timer", None)
        if t is None:
            return
        try:
            t.start(350)
        except Exception:
            pass

    def _install_autosave_hooks(self):
        # Connect UI change signals to autosave (best-effort).
        def hook(obj, signal_name: str):
            try:
                sig = getattr(obj, signal_name, None)
                if sig is None:
                    return
                sig.connect(self._schedule_autosave)
            except Exception:
                pass

        # Text inputs
        for w in [getattr(self, "ed_sdcli", None), getattr(self, "ed_initimg", None), getattr(self, "ed_mask", None),
                  getattr(self, "ed_ref1", None), getattr(self, "ed_ref2", None), getattr(self, "ed_ref3", None),
                  getattr(self, "ed_lora_dir", None),
                  getattr(self, "ed_neg", None)]:
            if w is not None:
                hook(w, "textChanged")
        if getattr(self, "ed_prompt", None) is not None:
            hook(self.ed_prompt, "textChanged")

        # Spin boxes
        for w in [getattr(self, "sp_steps", None), getattr(self, "sp_cfg", None), getattr(self, "sp_strength", None), getattr(self, "sp_seed", None),
                  getattr(self, "sp_w", None), getattr(self, "sp_h", None), getattr(self, "sp_shift", None),
                  getattr(self, "sp_vae_tile_overlap", None),
                  getattr(self, "sp_lora_strength", None)]:
            if w is not None:
                hook(w, "valueChanged")

        # Sliders
        for w in [getattr(self, "sl_lora_strength", None)]:
            if w is not None:
                hook(w, "valueChanged")

        # Combos
        for w in [getattr(self, "cb_unet", None), getattr(self, "cb_llm", None), getattr(self, "cb_mmproj", None),
                  getattr(self, "cb_vae", None), getattr(self, "cb_sampling", None), getattr(self, "cb_base", None),
                  getattr(self, "cb_vae_tile_size", None),
                  getattr(self, "cb_lora", None),
                  getattr(self, "cb_ma_az", None), getattr(self, "cb_ma_el", None), getattr(self, "cb_ma_dist", None)]:
            if w is not None:
                hook(w, "currentIndexChanged")

        # Checkboxes / toggles
        for w in [getattr(self, "chk_use_scene", None), getattr(self, "chk_use_mask", None), getattr(self, "chk_invert_mask", None), getattr(self, "chk_ref_increase_index", None),
                  getattr(self, "chk_disable_ref_resize", None), getattr(self, "chk_auto_aspect", None),
                  getattr(self, "chk_auto_show_results", None), getattr(self, "chk_use_queue", None),
                  getattr(self, "chk_multi_angle", None),
                  getattr(self, "chk_ma_auto_apply", None),
                  getattr(self, "chk_vae_tiling", None), getattr(self, "chk_offload", None), getattr(self, "chk_mmap", None),
                  getattr(self, "chk_vae_on_cpu", None), getattr(self, "chk_clip_on_cpu", None),
                  getattr(self, "chk_diffusion_fa", None)]:
            if w is not None:
                hook(w, "toggled")

        # Collapsible group open/close state
        for w in [getattr(self, "g_models", None), getattr(self, "g_lora", None), getattr(self, "g_paths", None),
                  getattr(self, "g_adv", None), getattr(self, "g_log", None)]:
            if w is not None:
                hook(w, "toggled")


    def _save_settings(self, silent: bool = False):
        if int(getattr(self, "_autosave_suspended", 0)) > 0:
            return
        s = {
            "sdcli_path": self.ed_sdcli.text().strip(),
            "models_box_open": bool(getattr(self, "g_models", None).isChecked() if getattr(self, "g_models", None) is not None else True),
            "lora_box_open": bool(getattr(self, "g_lora", None).isChecked() if getattr(self, "g_lora", None) is not None else True),
            "paths_box_open": bool(getattr(self, "g_paths", None).isChecked() if getattr(self, "g_paths", None) is not None else True),
            "advanced_box_open": bool(getattr(self, "g_adv", None).isChecked() if getattr(self, "g_adv", None) is not None else False),
            "log_box_open": bool(getattr(self, "g_log", None).isChecked() if getattr(self, "g_log", None) is not None else True),

            # LoRA (enabled when lora_name is not empty and not (None))
            "lora_dir": (self.ed_lora_dir.text().strip() if getattr(self, "ed_lora_dir", None) is not None else "") or DEFAULT_LORA_DIR,
            "lora_name": ("" if (str(getattr(self, "cb_lora", None).currentText() if getattr(self, "cb_lora", None) is not None else "").strip().lower() in ("(none)", "none")) else str(getattr(self, "cb_lora", None).currentText() if getattr(self, "cb_lora", None) is not None else "").strip()),
            "lora_strength": float(getattr(self, "sp_lora_strength", None).value() if getattr(self, "sp_lora_strength", None) is not None else 1.0),

            # Multi-angle camera helper (Qwen 2511)
            "multi_angle_camera": bool(getattr(self, "chk_multi_angle", None).isChecked() if getattr(self, "chk_multi_angle", None) is not None else False),
            "multi_angle_azimuth": str(getattr(self, "cb_ma_az", None).currentData() if getattr(self, "cb_ma_az", None) is not None else "front"),
            "multi_angle_elevation": str(getattr(self, "cb_ma_el", None).currentData() if getattr(self, "cb_ma_el", None) is not None else "eye"),
            "multi_angle_distance": str(getattr(self, "cb_ma_dist", None).currentData() if getattr(self, "cb_ma_dist", None) is not None else "medium"),
            "multi_angle_auto_apply": bool(getattr(self, "chk_ma_auto_apply", None).isChecked() if getattr(self, "chk_ma_auto_apply", None) is not None else False),

            "last_init_img": self.ed_initimg.text().strip(),
            "use_scene_image": bool(getattr(self, "chk_use_scene", None).isChecked() if getattr(self, "chk_use_scene", None) is not None else True),
            "last_mask_img": self.ed_mask.text().strip(),
            "use_mask": bool(getattr(self, "chk_use_mask", None).isChecked() if getattr(self, "chk_use_mask", None) is not None else False),
            "invert_mask": bool(self.chk_invert_mask.isChecked()),
            "ref_img_1": self.ed_ref1.text().strip(),
            "ref_img_2": self.ed_ref2.text().strip(),
            "ref_img_3": self.ed_ref3.text().strip(),
            "ref_increase_index": bool(self.chk_ref_increase_index.isChecked()),
            "ref_disable_auto_resize": bool(self.chk_disable_ref_resize.isChecked()),
            "prompt": self.ed_prompt.toPlainText().strip(),
            "negative_prompt": self.ed_neg.text().strip(),
            "unet_path": self.cb_unet.currentData(),
            "llm_path": self.cb_llm.currentData(),
            "mmproj_path": self.cb_mmproj.currentData(),
            "vae_path": self.cb_vae.currentData(),
            "steps": int(self.sp_steps.value()),
            "cfg": float(self.sp_cfg.value()),
            "strength": float(self.sp_strength.value()) if hasattr(self, "sp_strength") else 1.0,
            "seed": int(self.sp_seed.value()),
            "width": int(self.sp_w.value()),
            "height": int(self.sp_h.value()),
            "sampling_method": str(self.cb_sampling.currentText()),
            "shift": float(self.sp_shift.value()),
            "auto_aspect": bool(self.chk_auto_aspect.isChecked()),
            "auto_aspect_base": int(self.cb_base.currentText()),
            "auto_show_results": bool(getattr(self, "chk_auto_show_results", None).isChecked() if getattr(self, "chk_auto_show_results", None) is not None else False),
            "use_queue": bool(getattr(self, "chk_use_queue", None).isChecked() if getattr(self, "chk_use_queue", None) is not None else False),
            "vae_tiling": bool(self.chk_vae_tiling.isChecked()),
            "vae_tile_size": self.cb_vae_tile_size.currentText().strip(),
            "vae_tile_overlap": float(self.sp_vae_tile_overlap.value()),
            "offload_to_cpu": bool(self.chk_offload.isChecked()),
            "mmap": bool(self.chk_mmap.isChecked()),
            "vae_on_cpu": bool(self.chk_vae_on_cpu.isChecked()),
            "clip_on_cpu": bool(self.chk_clip_on_cpu.isChecked()),
            "diffusion_fa": bool(self.chk_diffusion_fa.isChecked()),
        }
        self._settings.update(s)
        _write_json(SETSAVE_PATH, self._settings)
        if not silent:
            self._append_log(f"\nSaved settings -> {SETSAVE_PATH}")



    def _on_use_queue_toggled(self, checked: bool):
        # Update Run button label/tooltip based on queue mode.
        try:
            useq = bool(checked)
        except Exception:
            useq = False
        try:
            self._settings["use_queue"] = useq
        except Exception:
            pass
        try:
            if useq:
                self.btn_run.setText("Add to queue")
                self.btn_run.setToolTip("Adds this Qwen2511 image edit job to the Queue (jobs/pending) to be processed by the worker.")
            else:
                self.btn_run.setText("Run sd-cli")
                self.btn_run.setToolTip("Runs sd-cli with the current settings. Output will be saved to output/qwen2511/.")
        except Exception:
            pass


    def _on_use_scene_toggled(self, checked: bool, persist: bool = True):
        """Enable/disable using the scene image (image 1).

        When OFF, we will generate a blank init image at run time.
        """
        try:
            for w in (
                getattr(self, "btn_pick_initimg", None),
                getattr(self, "thumb_initimg", None),
                getattr(self, "ed_initimg", None),
            ):
                if w is not None:
                    w.setEnabled(bool(checked))
        except Exception:
            pass

        try:
            if not bool(checked):
                if getattr(self, "lbl_imginfo", None) is not None:
                    self.lbl_imginfo.setText("Scene disabled: using a blank canvas as image 1.")
            else:
                try:
                    self._update_imginfo(self.ed_initimg.text().strip() if getattr(self, "ed_initimg", None) else "")
                except Exception:
                    pass
        except Exception:
            pass

        if persist:
            try:
                self._schedule_autosave()
            except Exception:
                pass

    def _on_use_mask_toggled(self, checked: bool, persist: bool = True):
        """Show/hide mask controls."""
        try:
            if getattr(self, "w_mask_row", None) is not None:
                self.w_mask_row.setVisible(bool(checked))
            if getattr(self, "chk_invert_mask", None) is not None:
                self.chk_invert_mask.setVisible(bool(checked))
        except Exception:
            pass

        if persist:
            try:
                self._schedule_autosave()
            except Exception:
                pass

    # --- LoRA UI helpers ---
    def _lora_is_enabled(self) -> bool:
        try:
            t = str(self.cb_lora.currentText() or "").strip()
        except Exception:
            t = ""
        if not t or t.lower() in ("(none)", "none"):
            return False
        return True

    def _on_lora_choice_changed(self, *args):
        """Enable/disable LoRA based on dropdown selection.

        Rule:
          - (None) => disabled
          - any other selection => enabled automatically
        """
        enabled = self._lora_is_enabled()
        try:
            if hasattr(self, "sl_lora_strength") and self.sl_lora_strength is not None:
                self.sl_lora_strength.setEnabled(enabled)
            if hasattr(self, "sp_lora_strength") and self.sp_lora_strength is not None:
                self.sp_lora_strength.setEnabled(enabled)
        except Exception:
            pass
        try:
            self._schedule_autosave()
        except Exception:
            pass

    def _on_lora_strength_slider(self, v: int):
        try:
            val = float(v) / 100.0
        except Exception:
            val = 1.0
        try:
            self.sp_lora_strength.blockSignals(True)
        except Exception:
            pass
        try:
            self.sp_lora_strength.setValue(val)
        finally:
            try:
                self.sp_lora_strength.blockSignals(False)
            except Exception:
                pass
        try:
            self._schedule_autosave()
        except Exception:
            pass

    def _on_lora_strength_spin(self, val: float):
        try:
            v = int(round(float(val) * 100.0))
        except Exception:
            v = 100
        try:
            self.sl_lora_strength.blockSignals(True)
        except Exception:
            pass
        try:
            self.sl_lora_strength.setValue(v)
        finally:
            try:
                self.sl_lora_strength.blockSignals(False)
            except Exception:
                pass
        try:
            self._schedule_autosave()
        except Exception:
            pass

    def _set_ui_busy(self, busy: bool, label: str = ""):
        # Disable controls while a process runs so we don't start multiple heavy jobs at once.
        self.btn_run.setEnabled(not busy)
        self.btn_download.setEnabled(not busy)
        self.btn_reload.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        try:
            self.chk_use_queue.setEnabled(not busy)
        except Exception:
            pass

        if busy:
            self._append_log(f"\n[{label}] started…")
        else:
            self._append_log(f"\n[{label}] done.")
    def _start_process(self, program: str, args: List[str], kind: str, expected_out: Optional[str]):
        # QProcess keeps the GUI responsive (no blocking stdout.readline loops).
        self._proc_kind = kind
        self._proc_expected_out = expected_out
        self._proc_buf = ""

        proc = QtCore.QProcess(self)
        proc.setWorkingDirectory(APP_ROOT)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        proc.readyReadStandardOutput.connect(self._on_proc_ready_read)
        proc.finished.connect(self._on_proc_finished)
        proc.errorOccurred.connect(self._on_proc_error)

        self._proc = proc
        self._set_ui_busy(True, kind)

        proc.start(program, args)

        if not proc.waitForStarted(1500):
            # If it cannot even start, release UI immediately.
            self._append_log(f"\nERROR: failed to start process: {program}")
            self._cleanup_process_ui()
    def _on_proc_ready_read(self):
        if not self._proc:
            return
        try:
            data = bytes(self._proc.readAllStandardOutput())
        except Exception:
            data = b""
        if not data:
            return

        chunk = data.decode("utf-8", errors="replace")
        self._proc_buf += chunk

        # Flush full lines to the log; keep partial line buffered.
        while "\n" in self._proc_buf:
            line, self._proc_buf = self._proc_buf.split("\n", 1)
            if line.strip():
                self._append_log(line.rstrip())
    def _on_proc_error(self, err: QtCore.QProcess.ProcessError):
        # Non-fatal in some cases (e.g., user cancelled), but log it.
        self._append_log(f"\nProcess error ({self._proc_kind}): {err}")
    def _on_proc_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        # Flush any remaining buffered output.
        if self._proc_buf.strip():
            self._append_log(self._proc_buf.rstrip())
        self._proc_buf = ""

        self._append_log(f"\n{self._proc_kind} finished (exit {exit_code}, status {exit_status})")

        if self._proc_kind == "downloader":
            self._reload_model_lists()

        if self._proc_kind == "sd-cli":
            out_file = self._proc_expected_out or ""
            if out_file and os.path.isfile(out_file):
                self._append_log(f"Output written: {out_file}")
                try:
                    self._last_out_file = out_file
                except Exception:
                    pass
                # Optional: auto-play result after run.
                try:
                    if hasattr(self, "chk_auto_show_results") and self.chk_auto_show_results.isChecked():
                        ok = self._play_file_in_main_player(out_file)
                        if not ok:
                            self._append_log("Auto play result: internal Media Player host not available.")
                except Exception:
                    pass
            else:
                self._append_log("Output file not found at expected path. If it crashed, check logs above.")

        self._cleanup_process_ui()
    def _cleanup_process_ui(self):
        kind = self._proc_kind or "process"
        self._proc_kind = ""
        self._proc_expected_out = None

        if self._proc is not None:
            try:
                self._proc.deleteLater()
            except Exception:
                pass
            self._proc = None

        self._set_ui_busy(False, kind)
    def _cancel_running_process(self):
        if self._proc is None or self._proc.state() == QtCore.QProcess.NotRunning:
            self._append_log("\nNo running process to cancel.")
            return

        self._append_log(f"\nCancelling {self._proc_kind}…")
        self._proc.terminate()

        # If it doesn't stop quickly, force kill.
        QtCore.QTimer.singleShot(2000, self._kill_if_still_running)
    def _kill_if_still_running(self):
        if self._proc is None:
            return
        if self._proc.state() != QtCore.QProcess.NotRunning:
            self._append_log(f"\nForce killing {self._proc_kind}…")
            self._proc.kill()
    def _noop_downloader(self):
        # Compatibility: some builds may not ship the downloader helper.
        try:
            self._append_log("\nDownloader is not available in this build.")
        except Exception:
            pass
    def _run_downloader(self):
        if self._proc is not None and self._proc.state() != QtCore.QProcess.NotRunning:
            self._append_log("\nA process is already running. Cancel it first.")
            return

        py = os.path.join(APP_ROOT, ".qwen2512", "venv", "Scripts", "python.exe")
        if not os.path.isfile(py):
            py = sys.executable

        if not os.path.isfile(DOWNLOAD_SCRIPT):
            self._append_log(f"\nDownloader script not found: {DOWNLOAD_SCRIPT}")
            return

        cmd = [py, DOWNLOAD_SCRIPT]
        self._append_log("\nRunning downloader:\n" + " ".join(shlex.quote(x) for x in cmd))

        self._start_process(program=cmd[0], args=cmd[1:], kind="downloader", expected_out=None)
    def _run_sdcli(self):
        if self._proc is not None and self._proc.state() != QtCore.QProcess.NotRunning:
            self._append_log("\nA process is already running. Cancel it first.")
            return

        sdcli = self.ed_sdcli.text().strip()
        # Queue mode: add a job to the app queue instead of running sd-cli inline.
        try:
            if hasattr(self, "chk_use_queue") and self.chk_use_queue.isChecked():
                try:
                    try:
                        from helpers.queue_adapter import enqueue_qwen2511_from_widget as _enq
                    except Exception:
                        from queue_adapter import enqueue_qwen2511_from_widget as _enq
                    jid = _enq(self)
                    try:
                        self._toast(f"Job queued to jobs/pending: {jid}")
                    except Exception:
                        pass
                    # Switch to Queue tab (best-effort)
                    try:
                        win = self.window()
                        tabs = win.findChild(QtWidgets.QTabWidget)
                        if tabs:
                            for i in range(tabs.count()):
                                if tabs.tabText(i).strip().lower() == "queue":
                                    tabs.setCurrentIndex(i)
                                    break
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        self._toast(f"Queue error: {e}", msec=3500)
                    except Exception:
                        self._append_log("\nQueue error: " + str(e))
                return
        except Exception:
            pass


        use_scene = True
        try:
            use_scene = bool(getattr(self, "chk_use_scene", None).isChecked())
        except Exception:
            use_scene = True

        init_img = self.ed_initimg.text().strip() if use_scene else ""

        use_mask = False
        try:
            use_mask = bool(getattr(self, "chk_use_mask", None).isChecked())
        except Exception:
            use_mask = False

        mask_img = self.ed_mask.text().strip() if use_mask else ""

        if not os.path.isfile(sdcli):
            self._append_log(f"\nERROR: sd-cli.exe not found: {sdcli}")
            return

        self.caps = detect_sdcli_caps(sdcli)

        if use_scene:
            if not init_img or not os.path.isfile(init_img):
                self._append_log("\nERROR: Scene image missing. Pick a scene image or turn OFF 'Use scene image' to use a blank canvas.")
                return

        if mask_img and not os.path.isfile(mask_img):
            self._append_log("\nERROR: Mask path is set but file not found.")
            return

        if mask_img and not self.caps.mask:
            self._append_log("\nWARNING: Your sd-cli build does not advertise --mask in --help. Mask will be ignored.")
            mask_img = ""

        # Reference images (image 2 / image 3 / image 4)
        ref_imgs = [self.ed_ref1.text().strip(), self.ed_ref2.text().strip(), self.ed_ref3.text().strip()]
        ref_imgs = [p for p in ref_imgs if p]
        for rp in ref_imgs:
            if not os.path.isfile(rp):
                self._append_log("\nERROR: Reference image file not found: " + rp)
                return
        if ref_imgs and not self.caps.ref_image:
            self._append_log("\nWARNING: Your sd-cli build does not advertise --ref-image in --help. Reference images will be ignored.")
            ref_imgs = []

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

        if not use_scene:
            try:
                blank_scene = os.path.join(OUTPUT_DIR, f"_blank_scene_{ts}.png")
                _write_blank_png(blank_scene, int(self.sp_w.value()), int(self.sp_h.value()), rgba=(255, 255, 255, 255))
                init_img = blank_scene
                self._append_log(f"\nScene disabled: using blank canvas -> {blank_scene}")
            except Exception as e:
                self._append_log(f"\nERROR: Failed to create blank scene image: {e}")
                return


        if mask_img and self.chk_invert_mask.isChecked():
            inv_tmp = _invert_mask_to_temp(mask_img, OUTPUT_DIR)
            if inv_tmp:
                self._append_log(f"\nInvert mask: wrote temp mask -> {inv_tmp}")
                mask_img = inv_tmp
            else:
                self._append_log("\nInvert mask: failed to invert mask. Using original mask.")

        strength = float(self.sp_strength.value()) if hasattr(self, "sp_strength") else 1.0
        if abs(strength - 1.0) > 1e-6:
            self._append_log("\nWARNING: Strength != 1.00 may not work reliably with Qwen2511. If it fails or ignores refs, set it back to 1.00.")

        cmd = build_sdcli_cmd(
            sdcli_path=sdcli,
            caps=self.caps,
            init_img=init_img,
            mask_path=mask_img,
            ref_images=ref_imgs,
            use_increase_ref_index=bool(self.chk_ref_increase_index.isChecked()),
            disable_auto_resize_ref_images=bool(self.chk_disable_ref_resize.isChecked()),
            prompt=self.ed_prompt.toPlainText().strip(),
            negative=self.ed_neg.text().strip(),
            unet_path=unet,
            llm_path=llm,
            mmproj_path=mmproj,
            vae_path=vae,
            steps=int(self.sp_steps.value()),
            cfg=float(self.sp_cfg.value()),
            seed=int(self.sp_seed.value()),
            width=int(self.sp_w.value()),
            height=int(self.sp_h.value()),
            strength=strength,
            sampling_method=str(self.cb_sampling.currentText()),
            shift=float(self.sp_shift.value()),
            out_file=out_file,
            lora_model_dir=str(getattr(self, "ed_lora_dir", None).text() if getattr(self, "ed_lora_dir", None) is not None else "").strip() or DEFAULT_LORA_DIR,
            lora_name=str(getattr(self, "cb_lora", None).currentText() if getattr(self, "cb_lora", None) is not None else "").strip(),
            lora_strength=float(getattr(self, "sp_lora_strength", None).value() if getattr(self, "sp_lora_strength", None) is not None else 1.0),
            use_vae_tiling=bool(self.chk_vae_tiling.isChecked()),
            vae_tile_size=self.cb_vae_tile_size.currentText().strip(),
            vae_tile_overlap=float(self.sp_vae_tile_overlap.value()),
            use_offload=bool(self.chk_offload.isChecked()),
            use_mmap=bool(self.chk_mmap.isChecked()),
            use_vae_on_cpu=bool(self.chk_vae_on_cpu.isChecked()),
            use_clip_on_cpu=bool(self.chk_clip_on_cpu.isChecked()),
            use_diffusion_fa=bool(self.chk_diffusion_fa.isChecked()),
        )

        self._append_log("\nRunning sd-cli:\n" + " ".join(shlex.quote(x) for x in cmd))
        self._start_process(program=cmd[0], args=cmd[1:], kind="sd-cli", expected_out=out_file)



def _on_use_scene_toggled(self, checked: bool, persist: bool = True):
    # When OFF, we will generate a blank init image at run time.
    try:
        for w in [getattr(self, "btn_pick_initimg", None), getattr(self, "thumb_initimg", None), getattr(self, "ed_initimg", None)]:
            if w is not None:
                w.setEnabled(bool(checked))
    except Exception:
        pass

    try:
        if not bool(checked):
            if getattr(self, "lbl_imginfo", None) is not None:
                self.lbl_imginfo.setText("Scene disabled: using a blank canvas as image 1.")
        else:
            try:
                self._update_imginfo(str(getattr(self.ed_initimg, "text", lambda: "")()).strip())
            except Exception:
                pass
    except Exception:
        pass

    if persist:
        try:
            self._schedule_autosave()
        except Exception:
            pass

def _on_use_mask_toggled(self, checked: bool, persist: bool = True):
    try:
        if getattr(self, "w_mask_row", None) is not None:
            self.w_mask_row.setVisible(bool(checked))
        if getattr(self, "chk_invert_mask", None) is not None:
            self.chk_invert_mask.setVisible(bool(checked))
    except Exception:
        pass
    if persist:
        try:
            self._schedule_autosave()
        except Exception:
            pass

    def _set_ui_busy(self, busy: bool, label: str = ""):
        # Disable controls while a process runs so we don't start multiple heavy jobs at once.
        self.btn_run.setEnabled(not busy)
        self.btn_download.setEnabled(not busy)
        self.btn_reload.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)
        try:
            self.chk_use_queue.setEnabled(not busy)
        except Exception:
            pass

        if busy:
            self._append_log(f"\n[{label}] started…")
        else:
            self._append_log(f"\n[{label}] done.")

    def _start_process(self, program: str, args: List[str], kind: str, expected_out: Optional[str]):
        # QProcess keeps the GUI responsive (no blocking stdout.readline loops).
        self._proc_kind = kind
        self._proc_expected_out = expected_out
        self._proc_buf = ""

        proc = QtCore.QProcess(self)
        proc.setWorkingDirectory(APP_ROOT)
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        proc.readyReadStandardOutput.connect(self._on_proc_ready_read)
        proc.finished.connect(self._on_proc_finished)
        proc.errorOccurred.connect(self._on_proc_error)

        self._proc = proc
        self._set_ui_busy(True, kind)

        proc.start(program, args)

        if not proc.waitForStarted(1500):
            # If it cannot even start, release UI immediately.
            self._append_log(f"\nERROR: failed to start process: {program}")
            self._cleanup_process_ui()

    def _on_proc_ready_read(self):
        if not self._proc:
            return
        try:
            data = bytes(self._proc.readAllStandardOutput())
        except Exception:
            data = b""
        if not data:
            return

        chunk = data.decode("utf-8", errors="replace")
        self._proc_buf += chunk

        # Flush full lines to the log; keep partial line buffered.
        while "\n" in self._proc_buf:
            line, self._proc_buf = self._proc_buf.split("\n", 1)
            if line.strip():
                self._append_log(line.rstrip())

    def _on_proc_error(self, err: QtCore.QProcess.ProcessError):
        # Non-fatal in some cases (e.g., user cancelled), but log it.
        self._append_log(f"\nProcess error ({self._proc_kind}): {err}")

    def _on_proc_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        # Flush any remaining buffered output.
        if self._proc_buf.strip():
            self._append_log(self._proc_buf.rstrip())
        self._proc_buf = ""

        self._append_log(f"\n{self._proc_kind} finished (exit {exit_code}, status {exit_status})")

        if self._proc_kind == "downloader":
            self._reload_model_lists()

        if self._proc_kind == "sd-cli":
            out_file = self._proc_expected_out or ""
            if out_file and os.path.isfile(out_file):
                self._append_log(f"Output written: {out_file}")
                try:
                    self._last_out_file = out_file
                except Exception:
                    pass
                # Optional: auto-play result after run.
                try:
                    if hasattr(self, "chk_auto_show_results") and self.chk_auto_show_results.isChecked():
                        ok = self._play_file_in_main_player(out_file)
                        if not ok:
                            self._append_log("Auto play result: internal Media Player host not available.")
                except Exception:
                    pass
            else:
                self._append_log("Output file not found at expected path. If it crashed, check logs above.")

        self._cleanup_process_ui()

    def _cleanup_process_ui(self):
        kind = self._proc_kind or "process"
        self._proc_kind = ""
        self._proc_expected_out = None

        if self._proc is not None:
            try:
                self._proc.deleteLater()
            except Exception:
                pass
            self._proc = None

        self._set_ui_busy(False, kind)

    def _cancel_running_process(self):
        if self._proc is None or self._proc.state() == QtCore.QProcess.NotRunning:
            self._append_log("\nNo running process to cancel.")
            return

        self._append_log(f"\nCancelling {self._proc_kind}…")
        self._proc.terminate()

        # If it doesn't stop quickly, force kill.
        QtCore.QTimer.singleShot(2000, self._kill_if_still_running)

    def _kill_if_still_running(self):
        if self._proc is None:
            return
        if self._proc.state() != QtCore.QProcess.NotRunning:
            self._append_log(f"\nForce killing {self._proc_kind}…")
            self._proc.kill()
    def _noop_downloader(self):
        # Compatibility: some builds may not ship the downloader helper.
        try:
            self._append_log("\nDownloader is not available in this build.")
        except Exception:
            pass



    def _run_downloader(self):
        if self._proc is not None and self._proc.state() != QtCore.QProcess.NotRunning:
            self._append_log("\nA process is already running. Cancel it first.")
            return

        py = os.path.join(APP_ROOT, ".qwen2512", "venv", "Scripts", "python.exe")
        if not os.path.isfile(py):
            py = sys.executable

        if not os.path.isfile(DOWNLOAD_SCRIPT):
            self._append_log(f"\nDownloader script not found: {DOWNLOAD_SCRIPT}")
            return

        cmd = [py, DOWNLOAD_SCRIPT]
        self._append_log("\nRunning downloader:\n" + " ".join(shlex.quote(x) for x in cmd))

        self._start_process(program=cmd[0], args=cmd[1:], kind="downloader", expected_out=None)

    def _run_sdcli(self):
        if self._proc is not None and self._proc.state() != QtCore.QProcess.NotRunning:
            self._append_log("\nA process is already running. Cancel it first.")
            return

        sdcli = self.ed_sdcli.text().strip()
        # Queue mode: add a job to the app queue instead of running sd-cli inline.
        try:
            if hasattr(self, "chk_use_queue") and self.chk_use_queue.isChecked():
                try:
                    try:
                        from helpers.queue_adapter import enqueue_qwen2511_from_widget as _enq
                    except Exception:
                        from queue_adapter import enqueue_qwen2511_from_widget as _enq
                    jid = _enq(self)
                    try:
                        self._toast(f"Job queued to jobs/pending: {jid}")
                    except Exception:
                        pass
                    # Switch to Queue tab (best-effort)
                    try:
                        win = self.window()
                        tabs = win.findChild(QtWidgets.QTabWidget)
                        if tabs:
                            for i in range(tabs.count()):
                                if tabs.tabText(i).strip().lower() == "queue":
                                    tabs.setCurrentIndex(i)
                                    break
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        self._toast(f"Queue error: {e}", msec=3500)
                    except Exception:
                        self._append_log("\nQueue error: " + str(e))
                return
        except Exception:
            pass


        use_scene = True
        try:
            use_scene = bool(getattr(self, "chk_use_scene", None).isChecked())
        except Exception:
            use_scene = True

        init_img = self.ed_initimg.text().strip() if use_scene else ""

        use_mask = False
        try:
            use_mask = bool(getattr(self, "chk_use_mask", None).isChecked())
        except Exception:
            use_mask = False

        mask_img = self.ed_mask.text().strip() if use_mask else ""

        if not os.path.isfile(sdcli):
            self._append_log(f"\nERROR: sd-cli.exe not found: {sdcli}")
            return

        self.caps = detect_sdcli_caps(sdcli)

        if use_scene:
            if not init_img or not os.path.isfile(init_img):
                self._append_log("\nERROR: Scene image missing. Pick a scene image or turn OFF 'Use scene image' to use a blank canvas.")
                return

        if mask_img and not os.path.isfile(mask_img):
            self._append_log("\nERROR: Mask path is set but file not found.")
            return

        if mask_img and not self.caps.mask:
            self._append_log("\nWARNING: Your sd-cli build does not advertise --mask in --help. Mask will be ignored.")
            mask_img = ""

        # Reference images (image 2 / image 3 / image 4)
        ref_imgs = [self.ed_ref1.text().strip(), self.ed_ref2.text().strip(), self.ed_ref3.text().strip()]
        ref_imgs = [p for p in ref_imgs if p]
        for rp in ref_imgs:
            if not os.path.isfile(rp):
                self._append_log("\nERROR: Reference image file not found: " + rp)
                return
        if ref_imgs and not self.caps.ref_image:
            self._append_log("\nWARNING: Your sd-cli build does not advertise --ref-image in --help. Reference images will be ignored.")
            ref_imgs = []

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

        if not use_scene:
            try:
                blank_scene = os.path.join(OUTPUT_DIR, f"_blank_scene_{ts}.png")
                _write_blank_png(blank_scene, int(self.sp_w.value()), int(self.sp_h.value()), rgba=(255, 255, 255, 255))
                init_img = blank_scene
                self._append_log(f"\nScene disabled: using blank canvas -> {blank_scene}")
            except Exception as e:
                self._append_log(f"\nERROR: Failed to create blank scene image: {e}")
                return


        if mask_img and self.chk_invert_mask.isChecked():
            inv_tmp = _invert_mask_to_temp(mask_img, OUTPUT_DIR)
            if inv_tmp:
                self._append_log(f"\nInvert mask: wrote temp mask -> {inv_tmp}")
                mask_img = inv_tmp
            else:
                self._append_log("\nInvert mask: failed to invert mask. Using original mask.")

        strength = float(self.sp_strength.value()) if hasattr(self, "sp_strength") else 1.0
        if abs(strength - 1.0) > 1e-6:
            self._append_log("\nWARNING: Strength != 1.00 may not work reliably with Qwen2511. If it fails or ignores refs, set it back to 1.00.")

        cmd = build_sdcli_cmd(
            sdcli_path=sdcli,
            caps=self.caps,
            init_img=init_img,
            mask_path=mask_img,
            ref_images=ref_imgs,
            use_increase_ref_index=bool(self.chk_ref_increase_index.isChecked()),
            disable_auto_resize_ref_images=bool(self.chk_disable_ref_resize.isChecked()),
            prompt=self.ed_prompt.toPlainText().strip(),
            negative=self.ed_neg.text().strip(),
            unet_path=unet,
            llm_path=llm,
            mmproj_path=mmproj,
            vae_path=vae,
            steps=int(self.sp_steps.value()),
            cfg=float(self.sp_cfg.value()),
            seed=int(self.sp_seed.value()),
            width=int(self.sp_w.value()),
            height=int(self.sp_h.value()),
            strength=strength,
            sampling_method=str(self.cb_sampling.currentText()),
            shift=float(self.sp_shift.value()),
            out_file=out_file,
            lora_model_dir=str(getattr(self, "ed_lora_dir", None).text() if getattr(self, "ed_lora_dir", None) is not None else "").strip() or DEFAULT_LORA_DIR,
            lora_name=str(getattr(self, "cb_lora", None).currentText() if getattr(self, "cb_lora", None) is not None else "").strip(),
            lora_strength=float(getattr(self, "sp_lora_strength", None).value() if getattr(self, "sp_lora_strength", None) is not None else 1.0),
            use_vae_tiling=bool(self.chk_vae_tiling.isChecked()),
            vae_tile_size=self.cb_vae_tile_size.currentText().strip(),
            vae_tile_overlap=float(self.sp_vae_tile_overlap.value()),
            use_offload=bool(self.chk_offload.isChecked()),
            use_mmap=bool(self.chk_mmap.isChecked()),
            use_vae_on_cpu=bool(self.chk_vae_on_cpu.isChecked()),
            use_clip_on_cpu=bool(self.chk_clip_on_cpu.isChecked()),
            use_diffusion_fa=bool(self.chk_diffusion_fa.isChecked()),
        )

        self._append_log("\nRunning sd-cli:\n" + " ".join(shlex.quote(x) for x in cmd))
        self._start_process(program=cmd[0], args=cmd[1:], kind="sd-cli", expected_out=out_file)


def _standalone_main():
    app = QtWidgets.QApplication(sys.argv)
    w = Qwen2511Pane()
    w.resize(1020, 960)
    w.setWindowTitle("Qwen Edit 2511 – Standalone Test")
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _standalone_main()