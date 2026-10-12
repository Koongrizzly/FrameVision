"""Isolated reference-thumbnail decoder for FrameVision Qwen Image 2.1.

Runs in a child process so even a native image decoder crash cannot terminate
FrameVision. Originals are never modified. Exits nonzero on unsupported files.
"""
import os
import sys
from pathlib import Path


def main():
    source, destination = map(Path, sys.argv[1:3])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(destination.stem + f".{os.getpid()}.tmp.png")
    try:
        try:
            from PIL import Image, ImageOps
            Image.MAX_IMAGE_PIXELS = 80_000_000
            with Image.open(source) as im:
                im = ImageOps.exif_transpose(im)
                im.thumbnail((168, 148), Image.Resampling.LANCZOS)
                im.convert("RGBA").save(temp, format="PNG")
        except ImportError:
            # Optional Pillow dependency: fallback to Qt only inside this
            # disposable worker process, never inside the GUI.
            from PySide6.QtCore import QSize, Qt
            from PySide6.QtGui import QImageReader
            reader = QImageReader(str(source))
            reader.setAutoTransform(True)
            size = reader.size()
            if not size.isValid() or size.width() * size.height() > 80_000_000:
                raise ValueError("Image dimensions unavailable or too large")
            reader.setScaledSize(size.scaled(QSize(168, 148), Qt.KeepAspectRatio))
            image = reader.read()
            if image.isNull() or not image.save(str(temp), "PNG"):
                raise ValueError("Unable to decode image")
        os.replace(temp, destination)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Preview error: {exc}", file=sys.stderr)
        sys.exit(1)
