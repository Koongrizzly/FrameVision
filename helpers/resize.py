# helpers/resize.py — Resize tool (rev 8)
# Rev 8 changes:
#   - Fixed "Lock aspect ratio" so it actually keeps W/H linked again.
#   - Internal logic now tracks last committed W/H instead of relying on a
#     stored ratio that some themes/widgets were dropping.
#
# All rev 7 features stay:
#   - "Do not resize" toggle that hides size + scale UI and keeps native size
#   - Removed "When file exists"
#   - Proper hide/show of Image-only vs Video-only rows (not greyed out)
#   - Video format conversion (MP4/MKV/WebM)
#   - Image format conversion (Auto/JPG/PNG/WebP/TIFF/BMP)
#
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
    # Keep from source | Fit | Fill | Letterbox
    scale_mode: str = "Keep from source"

    # Media mode: "Image" or "Video"
    media_mode: str = "Video"

    # Global toggle
    no_resize: bool = False  # When True we keep original size and hide size/scale UI

    # Video-specific
    video_format: str = "MP4 (H.264)"  # MP4 (H.264) | MKV (H.264) | WebM (VP9)
    crf: int = 23
    x264_preset: str = "veryfast"
    audio_copy: bool = True
    round_even: bool = True

    # Image-specific
    image_format: str = "Auto"   # Auto/JPG/PNG/WebP/TIFF/BMP
    image_quality: int = 90      # 0–100

    @classmethod
    def load(cls) -> "ResizePreset":
        try:
            if PRESET_PATH.exists():
                data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
                # Filter unknown legacy keys for safety
                defaults = asdict(cls())
                filtered = {k: v for k, v in data.items() if k in defaults}
                merged = {**defaults, **filtered}
                return cls(**merged)
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
    # "Keep from source" means: don't rescale/pad/crop, just normalize SAR.
    if mode == "Keep from source":
        ow, oh = (src_w or out_w, src_h or out_h)
        if round_even:
            ow, oh = _round_even_dims(ow, oh)
        vf = "setsar=1"
        return vf, ow, oh

    # Otherwise we build based on requested target size.
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
        # Fallback safe
        vf = f"scale={ow}:{oh}:flags=lanczos"

    vf += ",setsar=1"
    return vf, ow, oh

def _build_vf_image(out_w: int, out_h: int, mode: str) -> str:
    """Return -vf for images (no SAR/pix_fmt needed)."""
    if mode == "Keep from source":
        # no-op scale using source size
        vf = "scale=iw:ih:flags=lanczos"
    elif mode == "Fit":
        vf = f"scale={out_w}:{out_h}:flags=lanczos"
    elif mode == "Fill":
        vf = f"scale={out_w}:{out_h}:flags=lanczos,crop={out_w}:{out_h}"
    elif mode == "Letterbox":
        vf = (
            f"scale=min({out_w}\\,iw*{out_h}/ih):min({out_h}\\,ih*{out_w}/iw):flags=lanczos,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2"
        )
    else:
        vf = f"scale={out_w}:{out_h}:flags=lanczos"
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

def _video_ext_from_choice(choice: str) -> str:
    mapping = {
        "MP4 (H.264)": ".mp4",
        "MKV (H.264)": ".mkv",
        "WebM (VP9)": ".webm",
    }
    return mapping.get(choice, ".mp4")

# --------------------------- UI ---------------------------
class ResizePane(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.preset = ResizePreset.load()

        form = QFormLayout(self)

        # Small helper for consistent row creation and later show/hide.
        def _make_row_widget(h_widgets):
            roww = QWidget()
            lay = QHBoxLayout(roww)
            lay.setContentsMargins(0,0,0,0)
            for w in h_widgets:
                lay.addWidget(w)
            lay.addStretch(1)
            return roww

        def _add_form_row(key: str, label_text: str, field_widget: QWidget):
            lbl = QLabel(label_text)
            form.addRow(lbl, field_widget)
            setattr(self, f"row_{key}_label", lbl)
            setattr(self, f"row_{key}_widget", field_widget)

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

        row_toggle_widget = _make_row_widget([self.rad_images, self.rad_video])
        _add_form_row("mode", "Mode", row_toggle_widget)

        # "Do not resize" global toggle
        self.chk_noresize = QCheckBox("Do not resize (keep original size)")
        self.chk_noresize.setChecked(self.preset.no_resize)
        _add_form_row("noresize", "", self.chk_noresize)

        # Width/Height — single pair (no duplicates)
        self.spin_w = QSpinBox(); self.spin_w.setRange(16, 8192); self.spin_w.setValue(self.preset.width)
        self.spin_h = QSpinBox(); self.spin_h.setRange(16, 8192); self.spin_h.setValue(self.preset.height)
        row_wh_widget = _make_row_widget([QLabel("W"), self.spin_w, QLabel("H"), self.spin_h])
        _add_form_row("targetsize", "Target size", row_wh_widget)

        # AR lock
        self.chk_ar = QCheckBox("Lock aspect ratio"); self.chk_ar.setChecked(self.preset.ar_lock)
        _add_form_row("arlock", "", self.chk_ar)

        # Scale mode
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Keep from source","Fit","Fill","Letterbox"])
        self.cmb_mode.setCurrentText(self.preset.scale_mode)
        _add_form_row("scalemode", "Scale mode", self.cmb_mode)

        # ---------------- Image-only options ----------------
        self.cmb_imgfmt = QComboBox(); self.cmb_imgfmt.addItems(["Auto","JPG","PNG","WebP","TIFF","BMP"])
        self.cmb_imgfmt.setCurrentText(self.preset.image_format)
        _add_form_row("imgfmt", "Image format", self.cmb_imgfmt)

        self.slider_imgq = QSlider(Qt.Horizontal); self.slider_imgq.setRange(0,100); self.slider_imgq.setValue(self.preset.image_quality)
        self.lbl_imgq = QLabel(str(self.preset.image_quality) + "%")
        self.slider_imgq.valueChanged.connect(lambda v: self.lbl_imgq.setText(f"{v}%"))
        row_imgq_widget = _make_row_widget([self.slider_imgq, self.lbl_imgq])
        _add_form_row("imgq", "Image quality (0–100%)", row_imgq_widget)

        # ---------------- Video-only options ----------------
        self.cmb_vidfmt = QComboBox()
        self.cmb_vidfmt.addItems(["MP4 (H.264)", "MKV (H.264)", "WebM (VP9)"])
        self.cmb_vidfmt.setCurrentText(self.preset.video_format)
        _add_form_row("vidfmt", "Video format", self.cmb_vidfmt)

        self.slider_crf = QSlider(Qt.Horizontal); self.slider_crf.setRange(18, 28); self.slider_crf.setValue(self.preset.crf)
        self.lbl_crf = QLabel(str(self.preset.crf))
        self.slider_crf.valueChanged.connect(lambda v: self.lbl_crf.setText(str(v)))
        row_crf_widget = _make_row_widget([self.slider_crf, self.lbl_crf])
        _add_form_row("crf", "CRF (18=best, 28=smaller)", row_crf_widget)

        self.cmb_x264 = QComboBox(); self.cmb_x264.addItems(
            ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"]
        )
        self.cmb_x264.setCurrentText(self.preset.x264_preset)
        _add_form_row("x264", "Encoder preset", self.cmb_x264)

        self.chk_acopy = QCheckBox("Copy audio (no re-encode)"); self.chk_acopy.setChecked(self.preset.audio_copy)
        _add_form_row("acopy", "", self.chk_acopy)

        self.chk_even = QCheckBox("Round to even dimensions"); self.chk_even.setChecked(self.preset.round_even)
        _add_form_row("even", "", self.chk_even)

        # Action buttons
        self.btn_one = QPushButton("Resize a file…")
        self.btn_batch = QPushButton("Resize a batch…")
        row_btn_widget = _make_row_widget([self.btn_one, self.btn_batch])
        form.addRow(row_btn_widget)

        # Preset buttons
        self.btn_save = QPushButton("Save preset"); self.btn_load = QPushButton("Load preset")
        row_p_widget = _make_row_widget([self.btn_save, self.btn_load])
        form.addRow(row_p_widget)

        # Internal state
        # last committed W/H for the ratio math
        self._last_w = self.spin_w.value()
        self._last_h = self.spin_h.value()

        # historical ratio storage (kept for back-compat / debug)
        self._ar_ratio = None

        self._src_w = None
        self._src_h = None

        # Signals
        self.rad_images.toggled.connect(self._on_media_mode_changed)
        self.rad_video.toggled.connect(self._on_media_mode_changed)
        self.chk_noresize.stateChanged.connect(self._on_noresize_changed)

        self.chk_ar.stateChanged.connect(self._on_ar_lock_changed)
        self.spin_w.valueChanged.connect(self._on_w_changed)
        self.spin_h.valueChanged.connect(self._on_h_changed)
        self.btn_one.clicked.connect(self._on_resize_single)
        self.btn_batch.clicked.connect(self._on_resize_batch)
        self.btn_save.clicked.connect(self._on_save_preset)
        self.btn_load.clicked.connect(self._on_load_preset)

        # ---- Tooltips (long, new-user friendly) ----
        self.rad_images.setToolTip(
            "Images mode: export still images.\n\n"
            "Use this for JPG, PNG, WebP, TIFF, BMP and similar files. "
            "Image options (format and quality) below will be shown. "
            "Video-only options will be hidden in this mode."
        )
        self.rad_video.setToolTip(
            "Video mode: export video files.\n\n"
            "Use this for MP4, MOV, MKV and other videos. "
            "CRF, Encoder preset, Copy audio etc. will be shown. "
            "Image-only options will be hidden in this mode."
        )

        self.chk_noresize.setToolTip(
            "Do not resize:\n\n"
            "Keep the original pixel width/height.\n"
            "When this is ON, the Width/Height boxes, aspect ratio lock, "
            "Scale mode, and even-dimension setting are hidden. "
            "You can still convert formats (e.g. MP4→MKV or PNG→JPG) and adjust quality."
        )

        self.spin_w.setToolTip(
            "Target width in pixels.\n\n"
            "When 'Lock aspect ratio' is enabled, changing Width automatically "
            "updates Height to preserve the current W:H ratio."
        )
        self.spin_h.setToolTip(
            "Target height in pixels.\n\n"
            "When 'Lock aspect ratio' is enabled, changing Height automatically "
            "updates Width to preserve the current W:H ratio."
        )
        self.chk_ar.setToolTip(
            "Lock aspect ratio for the Width/Height pair.\n\n"
            "When enabled, Width and Height stay in proportion. "
            "Whichever you change, the other updates instantly using the "
            "ratio captured at the moment you turned the lock on."
        )

        # Scale mode tooltip(s)
        self._init_scale_mode_tooltips()

        self.cmb_imgfmt.setToolTip(
            "Output image format.\n\n"
            "• Auto: keep the source format.\n"
            "• JPG / PNG / WebP / TIFF / BMP: convert to that format.\n\n"
            "Tip: JPG / WebP are smaller for photos. PNG / TIFF preserve exact pixels."
        )
        self.slider_imgq.setToolTip(
            "Image quality (0–100%).\n\n"
            "Higher values = better visual quality and larger files.\n"
            "• JPG / WebP: this is visual compression quality.\n"
            "• PNG: this mostly changes compression level, not visual quality."
        )
        self.lbl_imgq.setToolTip(
            "Shows the current Image quality percentage."
        )

        self.cmb_vidfmt.setToolTip(
            "Output video format / container.\n\n"
            "• MP4 (H.264): very compatible.\n"
            "• MKV (H.264): same video codec in an MKV container.\n"
            "• WebM (VP9): smaller for some content, good for web.\n\n"
            "Note: WebM will re-encode audio to Opus."
        )

        self.slider_crf.setToolTip(
            "CRF for video quality/size (18 = best quality / biggest file, "
            "28 = most compressed / smallest file).\n\n"
            "Lower number = higher quality but bigger files. "
            "23 is a good everyday default."
        )
        self.lbl_crf.setToolTip(
            "Shows the current CRF value for video encoding."
        )
        self.cmb_x264.setToolTip(
            "Encoder preset (H.264 only) = how hard x264 works to compress.\n\n"
            "Slower presets squeeze filesize down a bit more at the same quality, "
            "but take longer to run. 'veryfast' is a solid default speed/quality tradeoff."
        )

        self.chk_acopy.setToolTip(
            "Copy the audio track without re-encoding (H.264 modes).\n\n"
            "Keeps original audio quality and avoids extra processing. "
            "Turn this OFF only if you need to force AAC audio for compatibility. "
            "For WebM / VP9 we always re-encode audio to Opus."
        )
        self.chk_even.setToolTip(
            "Round output width/height down to even numbers.\n\n"
            "Many video codecs and players expect even dimensions. "
            "Leaving this ON is safest for video playback.\n"
            "Hidden automatically if 'Do not resize' is enabled."
        )

        self.btn_one.setToolTip(
            "Resize / convert a single file.\n\n"
            "If a video or image is already loaded in the Media Player, this button "
            "will use that file immediately.\n\n"
            "If nothing is loaded yet, you'll be asked to pick a file. "
            "That chosen file will be processed using the settings above."
        )
        self.btn_batch.setToolTip(
            "Resize / convert a batch of files.\n\n"
            "Pick multiple images or videos and process them all in one go. "
            "You'll get a preflight summary of what will run and what will be skipped."
        )

        self.btn_save.setToolTip(
            "Save all current settings (size, mode, quality, etc.) into a preset "
            "so you can reuse the exact same configuration later."
        )
        self.btn_load.setToolTip(
            "Load your previously saved preset and apply it to all controls."
        )

        # capture initial lock ratio / visibility
        self._on_ar_lock_changed(self.chk_ar.checkState())
        self._apply_visibility()

    # Tooltip helpers
    def _init_scale_mode_tooltips(self):
        """
        Attach detailed tooltips to each Scale mode option and keep the combo's
        own tooltip synced with the current selection.
        """
        tips = {
            "Keep from source": (
                "Keep the source resolution.\n\n"
                "No resizing is performed; we just normalize pixel aspect ratio "
                "for video. Use this if you only want to change the format/"
                "codec without changing the pixel size."
            ),
            "Fit": (
                "Stretch/squash to exactly match the target Width × Height.\n\n"
                "This can distort the image if the aspect ratio is different."
            ),
            "Fill": (
                "Scale until the target Width × Height is completely covered, "
                "then crop any overflow.\n\n"
                "No black bars, but edges can be cut off."
            ),
            "Letterbox": (
                "Scale to fit inside the target Width × Height without cropping, "
                "then pad with borders (black bars) so the final output matches "
                "the requested size exactly."
            ),
        }

        # per-item tooltips in the dropdown
        for i in range(self.cmb_mode.count()):
            text = self.cmb_mode.itemText(i)
            if text in tips:
                self.cmb_mode.setItemData(i, tips[text], Qt.ToolTipRole)

        # live tooltip on the combo itself
        def _sync_combo_tooltip(txt):
            self.cmb_mode.setToolTip(
                tips.get(
                    txt,
                    "Choose how to scale / crop / pad the image or video to reach the target size."
                )
            )
        _sync_combo_tooltip(self.cmb_mode.currentText())
        self.cmb_mode.currentTextChanged.connect(_sync_combo_tooltip)

    # ----------------- AR lock behavior -----------------
    def _on_ar_lock_changed(self, state):
        """
        Whenever Lock aspect ratio is toggled, capture the *current* pair
        of Width/Height as the new baseline. We also store _ar_ratio for
        debugging/back-compat, but the live math uses _last_w/_last_h.
        """
        w = max(1, self.spin_w.value())
        h = max(1, self.spin_h.value())

        self._last_w = w
        self._last_h = h

        if state == Qt.Checked:
            self._ar_ratio = (h / w) if w else None
        else:
            self._ar_ratio = None

    def _on_w_changed(self, v):
        """
        User changed Width.
        If Lock AR is ON, recalc Height from our last committed ratio.
        """
        # always update local last_w so unlocked typing still sticks
        if self.chk_ar.isChecked() and self._last_w > 0:
            # ratio of H per W based on the last committed pair
            ratio_h_per_w = self._last_h / float(self._last_w)
            new_h = max(1, int(round(v * ratio_h_per_w)))

            self.spin_h.blockSignals(True)
            self.spin_h.setValue(new_h)
            self.spin_h.blockSignals(False)

            # commit the new pair
            self._last_w = v
            self._last_h = new_h
        else:
            self._last_w = v
            self._last_h = self.spin_h.value()

    def _on_h_changed(self, v):
        """
        User changed Height.
        If Lock AR is ON, recalc Width from our last committed ratio.
        """
        if self.chk_ar.isChecked() and self._last_h > 0:
            ratio_w_per_h = self._last_w / float(self._last_h)
            new_w = max(1, int(round(v * ratio_w_per_h)))

            self.spin_w.blockSignals(True)
            self.spin_w.setValue(new_w)
            self.spin_w.blockSignals(False)

            # commit the new pair
            self._last_w = new_w
            self._last_h = v
        else:
            self._last_w = self.spin_w.value()
            self._last_h = v

    # Utility
    def _is_image_mode(self) -> bool:
        return self.rad_images.isChecked()

    def _is_webm_format(self) -> bool:
        return "WebM" in self.cmb_vidfmt.currentText()

    # ----------------- Media mode / no-resize toggles -----------------
    def _apply_visibility(self):
        """
        Central show/hide logic:
        - Hide all video rows in Images mode.
        - Hide all image rows in Video mode.
        - Hide size/scale rows when "Do not resize" is ON.
        - Hide even-dim row if either Images mode OR "Do not resize".
        """
        is_image = self._is_image_mode()
        noresize = self.chk_noresize.isChecked()

        # rows tied to resize math
        resize_rows = [
            ("targetsize",),
            ("arlock",),
            ("scalemode",),
        ]
        for (key,) in resize_rows:
            getattr(self, f"row_{key}_label").setVisible(not noresize)
            getattr(self, f"row_{key}_widget").setVisible(not noresize)

        # even-dim row (video-only and also hidden for noresize)
        even_vis = (not is_image) and (not noresize)
        self.row_even_label.setVisible(even_vis)
        self.row_even_widget.setVisible(even_vis)

        # image-only rows
        img_rows = [
            ("imgfmt",),
            ("imgq",),
        ]
        for (key,) in img_rows:
            getattr(self, f"row_{key}_label").setVisible(is_image)
            getattr(self, f"row_{key}_widget").setVisible(is_image)

        # video-only rows
        vid_rows = [
            ("vidfmt",),
            ("crf",),
            ("x264",),
            ("acopy",),
        ]
        for (key,) in vid_rows:
            getattr(self, f"row_{key}_label").setVisible(not is_image)
            getattr(self, f"row_{key}_widget").setVisible(not is_image)

    def _on_media_mode_changed(self, *args):
        self._apply_visibility()

    def _on_noresize_changed(self, *args):
        self._apply_visibility()

    # ----------------- Presets -----------------
    def _on_save_preset(self):
        p = ResizePreset(
            width=self.spin_w.value(),
            height=self.spin_h.value(),
            ar_lock=self.chk_ar.isChecked(),
            scale_mode=self.cmb_mode.currentText(),
            media_mode="Image" if self._is_image_mode() else "Video",
            no_resize=self.chk_noresize.isChecked(),
            video_format=self.cmb_vidfmt.currentText(),
            crf=self.slider_crf.value(),
            x264_preset=self.cmb_x264.currentText(),
            audio_copy=self.chk_acopy.isChecked(),
            round_even=self.chk_even.isChecked(),
            image_format=self.cmb_imgfmt.currentText(),
            image_quality=self.slider_imgq.value(),
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

            # update tracking pair
            self._last_w = p.width
            self._last_h = p.height
            self._ar_ratio = (p.height / p.width) if (p.ar_lock and p.width) else None

            # radio selection
            self.rad_images.setChecked(p.media_mode == "Image")
            self.rad_video.setChecked(p.media_mode != "Image")

            self.chk_noresize.setChecked(p.no_resize)
            self.cmb_vidfmt.setCurrentText(p.video_format)
            self.slider_crf.setValue(p.crf)
            self.cmb_x264.setCurrentText(p.x264_preset)
            self.chk_acopy.setChecked(p.audio_copy)
            self.chk_even.setChecked(p.round_even)
            self.cmb_imgfmt.setCurrentText(p.image_format)
            self.slider_imgq.setValue(p.image_quality)

            # refresh vis / AR ratio
            self._apply_visibility()
            self._on_ar_lock_changed(self.chk_ar.checkState())

            QMessageBox.information(self, "Preset loaded", f"Loaded from:\n{PRESET_PATH}")
        except Exception as e:
            QMessageBox.critical(self, "Preset error", str(e))

    # ----------------- Actions -----------------
    def _pick_input(self) -> Path | None:
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

    def _video_ext_for_current(self) -> str:
        return _video_ext_from_choice(self.cmb_vidfmt.currentText())

    def _suggest_out_suffix(self, inp: Path) -> str:
        if self._is_image_mode():
            fmt_ext = _image_ext_from_choice(self.cmb_imgfmt.currentText(), inp.suffix)
            return ".resized" + fmt_ext
        else:
            fmt_ext = self._video_ext_for_current()
            return ".resized" + fmt_ext

    def _pick_output(self, inp: Path) -> Path | None:
        base = inp.with_name(inp.stem + self._suggest_out_suffix(inp))
        dlg = QFileDialog(self, "Save as")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.selectFile(str(base))
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                chosen = Path(files[0])
                # overwrite/rename/skip is now handled globally elsewhere;
                # here we just return the chosen path.
                return chosen
        return None

    def _build_cmd_video(self, inp: Path, out: Path, src_w: int, src_h: int) -> list[str]:
        tgt_w, tgt_h = self.spin_w.value(), self.spin_h.value()
        noresize = self.chk_noresize.isChecked()

        # Build scaling / SAR
        if noresize:
            vf_chain = "setsar=1"
        else:
            vf, ow, oh = _build_vf(
                src_w, src_h,
                tgt_w, tgt_h,
                self.cmb_mode.currentText(),
                self.chk_even.isChecked()
            )
            vf_chain = vf

        fmt_choice = self.cmb_vidfmt.currentText()
        is_webm = "WebM" in fmt_choice

        cmd = [
            "ffmpeg", "-y",
            "-i", str(inp),
        ]

        if vf_chain:
            cmd += ["-vf", vf_chain]

        if is_webm:
            # WebM (VP9 + Opus). audio_copy is ignored.
            cmd += [
                "-c:v", "libvpx-vp9",
                "-pix_fmt", "yuv420p",
                "-b:v", "0",
                "-crf", str(self.slider_crf.value()),
                "-c:a", "libopus",
                "-b:a", "192k",
            ]
        else:
            # MP4/MKV with H.264
            cmd += [
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
        noresize = self.chk_noresize.isChecked()
        quality = self.slider_imgq.value()
        ext = out.suffix.lower()

        cmd = ["ffmpeg", "-y", "-i", str(inp)]
        if not noresize:
            vf = _build_vf_image(tgt_w, tgt_h, self.cmb_mode.currentText())
            cmd += ["-vf", vf]

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

                # Output path
                if image_mode:
                    out_ext = _image_ext_from_choice(self.cmb_imgfmt.currentText(), ext)
                else:
                    out_ext = self._video_ext_for_current()
                out = inp.with_name(inp.stem + ".resized" + out_ext)

                to_process.append((inp, out))
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
        if not inp:
            return
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
        if not out:
            return

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
                files, _conflict = result
            else:
                files, _conflict = result, "skip"
            if files is None:
                return  # user cancelled
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
            lay.setContentsMargins(0,0,0,0)
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
