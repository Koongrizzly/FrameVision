
from __future__ import annotations

import os
import re
import tempfile
import subprocess
import shutil
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Qt
from PySide6.QtGui import QImage, QPainter, QColor, QFont, QIcon, QPixmap


class VisualThumbManager(QObject):
    """Manage tiny preview thumbnails for music-player visuals.

    Thumbnails are stored next to the visuals in presets/viz as PNG files.
    We try to create *real* previews by rendering a very short visual clip
    with viz_offline.render_visual_track and grabbing a frame via ffmpeg.

    Behaviour:
    - A light scan kicks in ~5 seconds after construction to pre-generate
      thumbnails for all known presets so they show up in the UI.
    - When the UI asks for an icon for a visual mode, we lazily ensure a
      thumbnail exists and return a QIcon built from it.
    - If anything fails (no ffmpeg, viz offline errors, etc.) we fall back
      to a simple coloured text placeholder so the UI never breaks.
    """

    def __init__(self, parent=None, ffmpeg: Optional[str] = None) -> None:
        super().__init__(parent)
        here = os.path.dirname(os.path.abspath(__file__))
        self._viz_dir = os.path.normpath(os.path.join(here, "..", "presets", "viz"))
        self._ffmpeg = ffmpeg or shutil.which("ffmpeg") or "ffmpeg"
        self._icon_cache: dict[str, QIcon] = {}

        # Run a light scan a few seconds after startup so we don't stack
        # too much work directly on app launch.
        try:
            QTimer.singleShot(5000, self._initial_scan)
        except Exception:
            # If QTimer is not available yet we simply skip the pre-scan;
            # thumbnails will still be created lazily when needed.
            pass

    # ---------------------- public API ---------------------------------

    def icon_for_mode(self, mode_id: str) -> QIcon:
        """Return a QIcon for the given visual mode, generating a thumb if needed."""
        if not mode_id:
            return QIcon()

        key = str(mode_id)
        cached = self._icon_cache.get(key)
        if cached is not None:
            return cached

        path = self._ensure_thumbnail(key)
        if path and os.path.exists(path):
            icon = QIcon(path)
        else:
            icon = QIcon()

        self._icon_cache[key] = icon
        return icon


    def preview_pixmap_for_mode(self, mode_id: str, max_width: int = 360, max_height: int = 200) -> QPixmap:
        """Return a larger QPixmap preview for the given visual mode.

        This reuses the existing thumbnail on disk and simply scales it up for
        preview purposes. It never writes new files.
        """
        if not mode_id:
            return QPixmap()

        key = str(mode_id)
        try:
            path = self._ensure_thumbnail(key)
        except Exception:
            path = None

        if not path or not os.path.exists(path):
            return QPixmap()

        pm = QPixmap(path)
        if pm.isNull():
            return QPixmap()

        # Only scale down if the image is bigger than the preview area.
        if pm.width() > max_width or pm.height() > max_height:
            pm = pm.scaled(
                max_width,
                max_height,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        return pm

    # ---------------------- internal helpers ---------------------------

    def _initial_scan(self) -> None:
        """Best-effort scan of all known visual modes to pre-generate thumbs.

        This is intentionally conservative and must never raise. If anything
        fails we simply fall back to lazy, on-demand thumbnail creation.

        NOTE: This uses viz_offline._list_visual_modes + VisualEngine just
        to discover the available mode IDs; the actual rendering is done
        via viz_offline.render_visual_track.
        """
        try:
            from .viz_offline import _list_visual_modes
            from .music import VisualEngine
        except Exception:
            return

        try:
            engine = VisualEngine(parent=None)
        except Exception:
            engine = None

        if engine is None:
            return

        try:
            modes = _list_visual_modes(engine) or []
        except Exception:
            modes = []

        for m in modes:
            try:
                if not m:
                    continue
                self._ensure_thumbnail(str(m))
            except Exception:
                # Never crash the app because of a bad preset.
                continue

    def _slug_for_mode(self, mode_id: str) -> str:
        raw = str(mode_id).strip().lower()
        # Strip the common 'viz:' prefix so filenames stay compact.
        if raw.startswith("viz:"):
            raw = raw[4:]
        # Only allow safe filename characters.
        slug = re.sub(r"[^a-z0-9_\-]+", "_", raw)
        if not slug:
            slug = "visual"
        return slug

    def _thumb_path_for_mode(self, mode_id: str) -> str:
        slug = self._slug_for_mode(mode_id)
        # Store thumbnails directly next to the visuals as <slug>_thumb.png
        name = f"{slug}_thumb.png"
        return os.path.join(self._viz_dir, name)

    def _ensure_thumbnail(self, mode_id: str) -> Optional[str]:
        path = self._thumb_path_for_mode(mode_id)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except Exception:
            return None

        if os.path.exists(path):
            return path

        ok = self._create_real_thumbnail(mode_id, path)
        if not ok:
            ok = self._create_text_placeholder_thumbnail(mode_id, path)
        if not ok:
            return None
        return path

    # ---------------------- thumbnail generation ------------------------

    def _create_real_thumbnail(self, mode_id: str, path: str) -> bool:
        """Try to render a short visual clip and grab a frame as thumbnail.

        We render ~2 seconds of silent audio, force a single section that
        uses this visual mode, and ask viz_offline.render_visual_track to
        create a tiny MP4. Then we grab a frame around t=0.8s via ffmpeg.
        """
        try:
            from .viz_offline import render_visual_track
        except Exception:
            return False

        ffmpeg = self._ffmpeg or shutil.which("ffmpeg") or "ffmpeg"

        # If ffmpeg is obviously not available, bail early.
        if not ffmpeg:
            return False

        try:
            with tempfile.TemporaryDirectory(prefix="fv_vizthumb_") as tmpdir:
                audio_path = os.path.join(tmpdir, "silence.wav")
                video_path = os.path.join(tmpdir, "preview.mp4")

                # 1) Create a tiny silent audio file (~2s)
                try:
                    cmd = [
                        ffmpeg,
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=stereo",
                        "-t",
                        "2.0",
                        audio_path,
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    return False

                # 2) Render a very small visual track using this single mode.
                try:
                    ok_viz = render_visual_track(
                        audio_path=audio_path,
                        out_video=video_path,
                        ffmpeg_bin=ffmpeg,
                        resolution=(320, 180),
                        fps=30,
                        # Strategy 2 == per-section, but we fake a single 'intro'
                        # section that spans the whole clip so we can force a
                        # particular mode via section_visual_overrides.
                        strategy=2,
                        segment_boundaries=None,
                        section_map=[(0.0, 2.0, "intro")],
                        section_visual_overrides={"intro": str(mode_id)},
                    )
                except Exception:
                    ok_viz = False

                if not ok_viz or not os.path.exists(video_path):
                    return False

                # 3) Grab a single frame around 0.8s as the thumbnail.
                try:
                    cmd = [
                        ffmpeg,
                        "-y",
                        "-ss",
                        "0.8",
                        "-i",
                        video_path,
                        "-frames:v",
                        "1",
                        path,
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return os.path.exists(path)
                except Exception:
                    return False
        except Exception:
            return False

    def _create_text_placeholder_thumbnail(self, mode_id: str, path: str) -> bool:
        """Fallback: create a small, readable text-based thumbnail.

        This is used when real rendering fails (no ffmpeg, viz errors,
        headless problems, etc.). It uses the same styled text approach
        you have already seen, just as a safety net.
        """
        try:
            width, height = 320, 180
            img = QImage(width, height, QImage.Format_RGB32)

            # Stable pseudo-random background colour based on the mode name.
            h = abs(hash(str(mode_id)))
            r = 80 + (h & 0x3F)
            g = 80 + ((h >> 6) & 0x3F)
            b = 80 + ((h >> 12) & 0x3F)
            r = max(0, min(255, r))
            g = max(0, min(255, g))
            b = max(0, min(255, b))

            painter = QPainter(img)
            painter.fillRect(0, 0, width, height, QColor(r, g, b))

            pretty = str(mode_id)
            if pretty.startswith("viz:"):
                pretty = pretty[4:]

            painter.setPen(QColor(255, 255, 255))
            font = QFont()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)

            margin = 8
            rect = img.rect().adjusted(margin, margin, -margin, -margin)
            painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, pretty)
            painter.end()

            return img.save(path, "PNG")
        except Exception:
            return False
