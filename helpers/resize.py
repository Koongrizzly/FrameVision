
# helpers/resize.py — Resize tool (rev 4)
# New: Batch pre-filter + preflight summary (queued/skipped with reasons)
#      Overwrite policy: Skip / Overwrite / Versioned filename
# Keeps: compact Images/Video toggle, image formats & quality, CRF/preset, audio copy,
#        even-dims, SAR, reporting, preset path, back-compat shim.

import os, re, json, subprocess, math
from dataclasses import dataclass, asdict
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QFormLayout, QLabel, QSlider, QSpinBox, QHBoxLayout, QVBoxLayout,
    QPushButton, QComboBox, QCheckBox, QFileDialog, QMessageBox, QRadioButton, QButtonGroup
)

# --- Safe imports for shared paths/constants ---
try:
    from helpers.framevision_app import ROOT as APP_ROOT
except Exception:
    APP_ROOT = Path('.').resolve()

# Required preset path: root /presets/setsave/resize.json
PRESET_PATH = (APP_ROOT / "presets" / "setsave" / "resize.json")

IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
VIDEO_EXTS = {".mp4",".mov",".m4v",".mkv",".avi",".mpg",".mpeg",".webm",".wmv",".flv"}

# --------------------------- Data model ---------------------------
@dataclass
class ResizePreset:
    width: int = 1280
    height:  int = 720
    ar_lock: bool = True
    scale_mode: str = "Fit"            # Fit | Fill | Letterbox

    # Media mode: "Image" or "Video"
    media_mode: str = "Video"

    # Video-specific
    crf: int = 23
    x264_preset: str = "veryfast"
    audio_copy: bool = True
    round_even: bool = True

    # Image-specific
    image_format: str = "Auto"   # Auto/JPG/PNG/WebP/TIFF/BMP
    image_quality: int = 90      # 0–100

    # Output policy
    overwrite_policy: str = "Skip existing"  # Skip existing | Overwrite | Versioned filename

    @classmethod
    def load(cls) -> "ResizePreset":
        try:
            if PRESET_PATH.exists():
                data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
                return cls(**{**asdict(cls()), **data})
        except Exception:
            pass
        return cls()

    def save(self):
        PRESET_PATH.parent.mkdir(parents=True, exist_ok=True)
        PRESET_PATH.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

# --------------------------- Helpers ---------------------------
def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1

def _round_even_dims(w: int, h: int) -> tuple[int,int]:
    return _even(max(2, w)), _even(max(2, h))

def _build_vf(src_w, src_h, out_w, out_h, mode: str, round_even: bool) -> tuple[str,int,int]:
    """Return FFmpeg -vf chain including scale/pad/crop + setsar=1 (for video)."""
    ow, oh = (out_w, out_h)
    if round_even:
        ow, oh = _round_even_dims(ow, oh)

    if mode == "Fit":
        vf = f"scale={ow}:{oh}:flags=lanczos"
    elif mode == "Fill":
        vf = f"scale={ow}:{oh}:flags=lanczos,crop={ow}:{oh}"
    elif mode == "Letterbox":
        vf = (
            f"scale=min({ow}\\,iw*{oh}/ih):min({oh}\\,ih*{ow}/iw):flags=lanczos,"
            f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2"
        )
    else:
        vf = f"scale={ow}:{oh}:flags=lanczos"

    vf += ",setsar=1"
    return vf, ow, oh

def _build_vf_image(out_w: int, out_h: int, mode: str) -> str:
    """Return -vf for images (no SAR/pix_fmt needed)."""
    ow, oh = (out_w, out_h)
    if mode == "Fit":
        vf = f"scale={ow}:{oh}:flags=lanczos"
    elif mode == "Fill":
        vf = f"scale={ow}:{oh}:flags=lanczos,crop={ow}:{oh}"
    elif mode == "Letterbox":
        vf = (
            f"scale=min({ow}\\,iw*{oh}/ih):min({oh}\\,ih*{ow}/iw):flags=lanczos,"
            f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2"
        )
    else:
        vf = f"scale={ow}:{oh}:flags=lanczos"
    return vf

def _ffprobe_size(path: Path) -> tuple[int,int]:
    """Try to get input width/height with ffprobe; fallback to 0,0 on failure."""
    try:
        import json as _json
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", str(path)
        ]
        p = subprocess.run(cmd, capture_output=True, text=True)
        data = _json.loads(p.stdout)
        s = data["streams"][0]
        return int(s.get("width", 0)), int(s.get("height", 0))
    except Exception:
        return 0, 0

def _map_jpeg_q(q_percent: int) -> int:
    """Map 0..100% to FFmpeg mjpeg -q:v (2..31; lower is better)."""
    q_percent = max(0, min(100, q_percent))
    return int(round(2 + (31 - 2) * (100 - q_percent) / 100.0))

def _map_png_compress(q_percent: int) -> int:
    """Map 0..100% to png -compression_level (0..9; higher compresses more)."""
    q_percent = max(0, min(100, q_percent))
    return int(round(9 * (100 - q_percent) / 100.0))

def _image_ext_from_choice(choice: str, src_ext: str) -> str:
    if choice == "Auto":
        return src_ext.lower()
    mapping = {
        "JPG": ".jpg",
        "PNG": ".png",
        "WebP": ".webp",
        "TIFF": ".tif",
        "BMP": ".bmp",
    }
    return mapping.get(choice, src_ext.lower())

def _versioned_path(base: Path) -> Path:
    """Return base with _v2/_v3... inserted before suffix, choosing first non-existing."""
    stem, suf = base.stem, base.suffix
    # If stem already ends with _vN, increment it
    import re as _re
    m = _re.search(r"(.*)_v(\d+)$", stem)
    if m:
        root = m.group(1)
        n = int(m.group(2))
        while True:
            n += 1
            candidate = base.with_name(f"{root}_v{n}{suf}")
            if not candidate.exists():
                return candidate
    else:
        n = 2
        while True:
            candidate = base.with_name(f"{stem}_v{n}{suf}")
            if not candidate.exists():
                return candidate

# --------------------------- UI ---------------------------
class ResizePane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.preset = ResizePreset.load()

        form = QFormLayout(self)

        # --- Compact media toggle (Images / Video) ---
        self.rad_images = QRadioButton("Images")
        self.rad_video  = QRadioButton("Video")
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        self.btn_group.addButton(self.rad_images)
        self.btn_group.addButton(self.rad_video)
        if self.preset.media_mode == "Image":
            self.rad_images.setChecked(True)
        else:
            self.rad_video.setChecked(True)

        row_toggle = QHBoxLayout()
        row_toggle.setContentsMargins(0,0,0,0)
        row_toggle.addWidget(self.rad_images)
        row_toggle.addWidget(self.rad_video)
        row_toggle.addStretch(1)
        form.addRow(QLabel("Mode"), row_toggle)

        # Wire toggle
        self.rad_images.toggled.connect(self._on_media_mode_changed)
        self.rad_video.toggled.connect(self._on_media_mode_changed)

        # Width/Height — single pair (no duplicates)
        self.spin_w = QSpinBox(); self.spin_w.setRange(16, 8192); self.spin_w.setValue(self.preset.width)
        self.spin_h = QSpinBox(); self.spin_h.setRange(16, 8192); self.spin_h.setValue(self.preset.height)
        row_wh = QHBoxLayout()
        row_wh.addWidget(QLabel("W")); row_wh.addWidget(self.spin_w)
        row_wh.addWidget(QLabel("H")); row_wh.addWidget(self.spin_h)
        form.addRow(QLabel("Target size"), row_wh)

        # AR lock
        self.chk_ar = QCheckBox("Lock aspect ratio"); self.chk_ar.setChecked(self.preset.ar_lock)
        form.addRow(self.chk_ar)

        # Mode
        self.cmb_mode = QComboBox(); self.cmb_mode.addItems(["Fit","Fill","Letterbox"])
        self.cmb_mode.setCurrentText(self.preset.scale_mode)
        form.addRow(QLabel("Scale mode"), self.cmb_mode)

        # ---------------- Image-only options ----------------
        self.cmb_imgfmt = QComboBox(); self.cmb_imgfmt.addItems(["Auto","JPG","PNG","WebP","TIFF","BMP"])
        self.cmb_imgfmt.setCurrentText(self.preset.image_format)
        form.addRow(QLabel("Image format"), self.cmb_imgfmt)

        self.slider_imgq = QSlider(Qt.Horizontal); self.slider_imgq.setRange(0,100); self.slider_imgq.setValue(self.preset.image_quality)
        self.lbl_imgq = QLabel(str(self.preset.image_quality) + "%")
        self.slider_imgq.valueChanged.connect(lambda v: self.lbl_imgq.setText(f"{v}%"))
        row_imgq = QHBoxLayout(); row_imgq.addWidget(self.slider_imgq); row_imgq.addWidget(self.lbl_imgq)
        form.addRow(QLabel("Image quality (0–100%)"), row_imgq)

        # ---------------- Video-only options ----------------
        self.slider_crf = QSlider(Qt.Horizontal); self.slider_crf.setRange(18, 28); self.slider_crf.setValue(self.preset.crf)
        self.lbl_crf = QLabel(str(self.preset.crf))
        self.slider_crf.valueChanged.connect(lambda v: self.lbl_crf.setText(str(v)))
        row_crf = QHBoxLayout(); row_crf.addWidget(self.slider_crf); row_crf.addWidget(self.lbl_crf)
        form.addRow(QLabel("CRF (18=best, 28=smaller)"), row_crf)

        self.cmb_x264 = QComboBox(); self.cmb_x264.addItems(
            ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]
        )
        self.cmb_x264.setCurrentText(self.preset.x264_preset)
        form.addRow(QLabel("Encoder preset"), self.cmb_x264)

        self.chk_acopy = QCheckBox("Copy audio (no re-encode)"); self.chk_acopy.setChecked(self.preset.audio_copy)
        form.addRow(self.chk_acopy)

        self.chk_even = QCheckBox("Round to even dimensions"); self.chk_even.setChecked(self.preset.round_even)
        form.addRow(self.chk_even)

        # Overwrite policy
        self.cmb_overwrite = QComboBox(); self.cmb_overwrite.addItems(["Skip existing","Overwrite","Versioned filename"])
        self.cmb_overwrite.setCurrentText(self.preset.overwrite_policy)
        form.addRow(QLabel("When file exists"), self.cmb_overwrite)

        # Action buttons
        self.btn_one = QPushButton("Resize a file…")
        self.btn_batch = QPushButton("Resize a batch…")
        row_btn = QHBoxLayout(); row_btn.addWidget(self.btn_one); row_btn.addWidget(self.btn_batch)
        form.addRow(row_btn)

        # Preset buttons
        self.btn_save = QPushButton("Save preset"); self.btn_load = QPushButton("Load preset")
        row_p = QHBoxLayout(); row_p.addWidget(self.btn_save); row_p.addWidget(self.btn_load)
        form.addRow(row_p)

        # Signals
        self.chk_ar.stateChanged.connect(self._on_ar_lock_changed)
        self.spin_w.valueChanged.connect(self._on_w_changed)
        self.spin_h.valueChanged.connect(self._on_h_changed)
        self.btn_one.clicked.connect(self._on_resize_single)
        self.btn_batch.clicked.connect(self._on_resize_batch)
        self.btn_save.clicked.connect(self._on_save_preset)
        self.btn_load.clicked.connect(self._on_load_preset)

        self._src_w = None
        self._src_h = None

        # apply initial mode
        self._on_media_mode_changed()

    # ----------------- AR lock behavior -----------------
    def _on_ar_lock_changed(self, _v): pass

    def _on_w_changed(self, v):
        if self.chk_ar.isChecked() and self._src_w and self._src_h:
            ratio = self._src_h / self._src_w
            self.spin_h.blockSignals(True)
            self.spin_h.setValue(max(1, int(v * ratio)))
            self.spin_h.blockSignals(False)

    def _on_h_changed(self, v):
        if self.chk_ar.isChecked() and self._src_w and self._src_h:
            ratio = self._src_w / self._src_h
            self.spin_w.blockSignals(True)
            self.spin_w.setValue(max(1, int(v * ratio)))
            self.spin_w.blockSignals(False)

    # Utility
    def _is_image_mode(self) -> bool:
        return self.rad_images.isChecked()

    # ----------------- Media mode toggle -----------------
    def _on_media_mode_changed(self, *args):
        is_image = self._is_image_mode()

        # Grey out irrelevant controls
        for w in (self.cmb_imgfmt, self.slider_imgq, self.lbl_imgq):
            w.setEnabled(is_image)

        for w in (self.slider_crf, self.lbl_crf, self.cmb_x264, self.chk_acopy, self.chk_even):
            w.setEnabled(not is_image)

    # ----------------- Presets -----------------
    def _on_save_preset(self):
        p = ResizePreset(
            width=self.spin_w.value(),
            height=self.spin_h.value(),
            ar_lock=self.chk_ar.isChecked(),
            scale_mode=self.cmb_mode.currentText(),
            media_mode="Image" if self._is_image_mode() else "Video",
            crf=self.slider_crf.value(),
            x264_preset=self.cmb_x264.currentText(),
            audio_copy=self.chk_acopy.isChecked(),
            round_even=self.chk_even.isChecked(),
            image_format=self.cmb_imgfmt.currentText(),
            image_quality=self.slider_imgq.value(),
            overwrite_policy=self.cmb_overwrite.currentText(),
        )
        try:
            p.save()
            QMessageBox.information(self, "Preset saved", f"Saved to:\n{PRESET_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    def _on_load_preset(self):
        try:
            p = ResizePreset.load()
            self.spin_w.setValue(p.width)
            self.spin_h.setValue(p.height)
            self.chk_ar.setChecked(p.ar_lock)
            self.cmb_mode.setCurrentText(p.scale_mode)

            # radio selection
            self.rad_images.setChecked(p.media_mode == "Image")
            self.rad_video.setChecked(p.media_mode != "Image")

            self.slider_crf.setValue(p.crf)
            self.cmb_x264.setCurrentText(p.x264_preset)
            self.chk_acopy.setChecked(p.audio_copy)
            self.chk_even.setChecked(p.round_even)
            self.cmb_imgfmt.setCurrentText(p.image_format)
            self.slider_imgq.setValue(p.image_quality)
            self.cmb_overwrite.setCurrentText(p.overwrite_policy)

            self._on_media_mode_changed()
            QMessageBox.information(self, "Preset loaded", f"Loaded from:\n{PRESET_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ----------------- Actions -----------------
    def _pick_input(self) -> Path:
        dlg = QFileDialog(self, "Choose media file")
        dlg.setFileMode(QFileDialog.ExistingFile)
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                return Path(files[0])
        return None

    def _pick_inputs(self) -> list[Path]:
        dlg = QFileDialog(self, "Choose media files")
        dlg.setFileMode(QFileDialog.ExistingFiles)
        if dlg.exec():
            return [Path(f) for f in dlg.selectedFiles()]
        return []

    def _suggest_out_suffix(self, inp: Path) -> str:
        if self._is_image_mode():
            fmt_ext = _image_ext_from_choice(self.cmb_imgfmt.currentText(), inp.suffix)
            return ".resized" + fmt_ext
        else:
            return ".resized.mp4"

    def _apply_overwrite_policy(self, out_path: Path) -> tuple[Path, str]:
        """Return (final_path, skip_reason or "") obeying overwrite policy."""
        policy = self.cmb_overwrite.currentText()
        if not out_path.exists():
            return out_path, ""
        if policy == "Overwrite":
            return out_path, ""
        if policy == "Skip existing":
            return out_path, "exists"
        if policy == "Versioned filename":
            return _versioned_path(out_path), ""
        # Default safe
        return out_path, "exists"

    def _pick_output(self, inp: Path) -> Path:
        base = inp.with_name(inp.stem + self._suggest_out_suffix(inp))
        dlg = QFileDialog(self, "Save as")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.selectFile(str(base))
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                chosen = Path(files[0])
                final, reason = self._apply_overwrite_policy(chosen)
                if reason == "exists":
                    QMessageBox.information(self, "Skipped", f"File exists and policy is 'Skip existing':\n{chosen}")
                    return None
                return final
        return None

    def _build_cmd_video(self, inp: Path, out: Path, src_w: int, src_h: int) -> list[str]:
        tgt_w, tgt_h = self.spin_w.value(), self.spin_h.value()
        vf, ow, oh = _build_vf(src_w, src_h, tgt_w, tgt_h, self.cmb_mode.currentText(), self.chk_even.isChecked())

        cmd = [
            "ffmpeg", "-y",
            "-i", str(inp),
            "-vf", vf,
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", self.cmb_x264.currentText(),
            "-crf", str(self.slider_crf.value())
        ]
        if self.chk_acopy.isChecked():
            cmd += ["-c:a", "copy"]
        else:
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd += [str(out)]
        return cmd

    def _build_cmd_image(self, inp: Path, out: Path) -> list[str]:
        tgt_w, tgt_h = self.spin_w.value(), self.spin_h.value()
        vf = _build_vf_image(tgt_w, tgt_h, self.cmb_mode.currentText())
        quality = self.slider_imgq.value()
        ext = out.suffix.lower()

        cmd = ["ffmpeg", "-y", "-i", str(inp), "-vf", vf]

        if ext in (".jpg", ".jpeg"):
            cmd += ["-q:v", str(_map_jpeg_q(quality))]
        elif ext == ".webp":
            cmd += ["-q:v", str(quality)]
        elif ext == ".png":
            cmd += ["-compression_level", str(_map_png_compress(quality))]
        else:
            pass

        cmd += [str(out)]
        return cmd

    def _preflight_batch(self, inputs: list[Path]):
        """Pre-filter list and show a preflight summary; return (to_process, skipped_list) or ([], []) if cancel."""
        image_mode = self._is_image_mode()
        to_process = []
        skipped = []

        for inp in inputs:
            try:
                if not inp.exists():
                    skipped.append((inp, "missing"))
                    continue

                ext = inp.suffix.lower()
                if image_mode and ext not in IMAGE_EXTS:
                    skipped.append((inp, "not image (mode=Images)"))
                    continue
                if (not image_mode) and ext not in VIDEO_EXTS:
                    skipped.append((inp, "not video (mode=Video)"))
                    continue

                # Output path & overwrite policy
                if image_mode:
                    out_ext = _image_ext_from_choice(self.cmb_imgfmt.currentText(), ext)
                    out = inp.with_name(inp.stem + ".resized" + out_ext)
                else:
                    out = inp.with_name(inp.stem + ".resized.mp4")

                final_out, reason = self._apply_overwrite_policy(out)
                if reason == "exists":
                    skipped.append((inp, "exists (policy=Skip)"))
                    continue

                to_process.append((inp, final_out))
            except Exception as e:
                skipped.append((inp, f"error: {e}"))

        # Preflight dialog
        lines = [
            f"Mode: {'Images' if image_mode else 'Video'}",
            f"Queued: {len(to_process)}",
            f"Skipped: {len(skipped)}",
        ]
        for p, r in skipped[:10]:
            lines.append(f"  - {p.name}: {r}")
        if len(skipped) > 10:
            lines.append(f"  … (+{len(skipped)-10} more)")
        lines.append("Proceed with queued items?")
        resp = QMessageBox.question(self, "Batch preflight", "\n".join(lines), QMessageBox.Yes | QMessageBox.No)
        if resp == QMessageBox.No:
            return [], []
        return to_process, skipped

    def _on_resize_single(self):
        inp = self._pick_input()
        if not inp: return
        ext = inp.suffix.lower()

        if self._is_image_mode() and ext not in IMAGE_EXTS:
            QMessageBox.warning(self, "Wrong mode", "Current mode is Images but selected file is not an image.")
            return
        if (not self._is_image_mode()) and ext not in VIDEO_EXTS:
            QMessageBox.warning(self, "Wrong mode", "Current mode is Video but selected file is not a video.")
            return

        src_w, src_h = _ffprobe_size(inp)
        self._src_w, self._src_h = src_w, src_h

        out = self._pick_output(inp)
        if not out: return

        cmd = self._build_cmd_image(inp, out) if self._is_image_mode() else self._build_cmd_video(inp, out, src_w, src_h)
        ok, err = self._run_ffmpeg(cmd)
        if ok:
            QMessageBox.information(self, "Done", f"Saved:\n{out}")
        else:
            QMessageBox.critical(self, "FFmpeg error", err or "Unknown error")
    def _on_resize_batch(self):
        # Prefer the shared BatchSelectDialog (helpers/batch.py). Fallback to classic file picker.
        inputs = []
        try:
            from helpers.batch import BatchSelectDialog as _BatchDialog
        except Exception:
            try:
                from helpers.vatch import BatchSelectDialog as _BatchDialog
            except Exception:
                _BatchDialog = None

        if _BatchDialog is not None:
            exts = tuple(IMAGE_EXTS) if self._is_image_mode() else tuple(VIDEO_EXTS)
            result = _BatchDialog.pick(self, title="Resize a batch…", exts=exts)
            if isinstance(result, tuple):
                files, conflict = result
            else:
                files, conflict = result, "skip"
            if files is None:
                return  # user cancelled
            # Sync overwrite policy combo to dialog choice
            if conflict in ("skip", "none"):
                self.cmb_overwrite.setCurrentText("Skip existing")
            elif conflict == "overwrite":
                self.cmb_overwrite.setCurrentText("Overwrite")
            elif conflict in ("version", "autorename", "auto", "ver"):
                self.cmb_overwrite.setCurrentText("Versioned filename")
            inputs = [Path(f) for f in files]
        else:
            inputs = self._pick_inputs()
            if not inputs:
                return

        # Preflight filtering & summary
        to_process, skipped = self._preflight_batch(inputs)
        if not to_process and not skipped:
            return  # user cancelled

        ok_count = 0
        fail = []

        for inp, out in to_process:
            try:
                if self._is_image_mode():
                    cmd = self._build_cmd_image(inp, out)
                else:
                    src_w, src_h = _ffprobe_size(inp)
                    self._src_w, self._src_h = src_w, src_h
                    cmd = self._build_cmd_video(inp, out, src_w, src_h)

                ok, err = self._run_ffmpeg(cmd)
                if ok:
                    ok_count += 1
                else:
                    fail.append((inp, err or "ffmpeg failed"))
            except Exception as e:
                fail.append((inp, str(e)))

        # Final summary
        lines = [
            f"Processed: {len(to_process)} / {len(inputs)} selected",
            f"Success: {ok_count}",
            f"Mode: {'Images' if self._is_image_mode() else 'Video'}",
        ]
        if skipped:
            lines.append(f"Skipped (preflight): {len(skipped)}")
            for p, r in skipped[:10]:
                lines.append(f"  - {p.name}: {r}")
            if len(skipped) > 10:
                lines.append(f"  … (+{len(skipped)-10} more)")
        if fail:
            lines.append(f"Failed during run: {len(fail)}")
            for p, reason in fail[:10]:
                lines.append(f"  - {p.name}: {reason}")
            if len(fail) > 10:
                lines.append(f"  … (+{len(fail)-10} more)")

        msg = "\n".join(lines)
        if fail:
            QMessageBox.warning(self, "Batch finished with errors", msg)
        else:
            QMessageBox.information(self, "Batch finished", msg)


    def _run_ffmpeg(self, cmd: list[str]) -> tuple[bool, str]:
        try:
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                return True, ""
            err = p.stderr.strip()
            tail = "\n".join(err.splitlines()[-15:])
            return False, tail
        except FileNotFoundError:
            return False, "ffmpeg not found. Please install it and ensure it's on PATH."
        except Exception as e:
            return False, str(e)

# Factory for host app
def create_widget(parent=None) -> QWidget:
    return ResizePane(parent)

# ---------------- Back-compat shim ----------------
def install_resize_tool(*args, **kwargs):
    """
    Back-compat entrypoint expected by older Tools panes.
    If a section-like container is passed, this function will mount the created
    widget into that container so the UI actually shows up.
    Returns the ResizePane QWidget.
    """
    parent = kwargs.get("parent")
    section = None
    for a in args:
        if hasattr(a, "setContentLayout") or hasattr(a, "setContentWidget") or hasattr(a, "layout"):
            section = a
        if parent is None and (hasattr(a, "addWidget") or hasattr(a, "layout")):
            parent = a

    widget = create_widget(parent)
    try:
        if section is not None and hasattr(section, "setContentLayout"):
            from PySide6.QtWidgets import QVBoxLayout
            lay = QVBoxLayout()
            lay.addWidget(widget)
            section.setContentLayout(lay)
            return widget
        if section is not None and hasattr(section, "setContentWidget"):
            section.setContentWidget(widget)
            return widget
        if section is not None and hasattr(section, "layout") and section.layout() is not None:
            section.layout().addWidget(widget)
            return widget
    except Exception:
        pass
    return widget
