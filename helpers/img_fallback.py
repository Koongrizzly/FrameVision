from __future__ import annotations
from pathlib import Path
from PySide6.QtGui import QPixmap, QImage
try:
    from PIL import Image
except Exception:
    Image = None

def _pil_to_qimage(im) -> QImage:
    if im.mode not in ('RGB','RGBA'):
        try: im = im.convert('RGBA')
        except Exception: im = im.convert('RGB')
    if im.mode == 'RGBA':
        fmt = QImage.Format_RGBA8888; bpl = im.width * 4
    else:
        fmt = QImage.Format_RGB888; bpl = im.width * 3
    data = im.tobytes()
    qimg = QImage(data, im.width, im.height, bpl, fmt)
    return qimg.copy()

def load_pixmap(path: str | Path) -> QPixmap:
    p = str(path)
    pm = QPixmap(p)
    if not pm.isNull():
        return pm
    if Image is None:
        return pm
    try:
        im = Image.open(p)
        try: im.seek(0)
        except Exception: pass
        qimg = _pil_to_qimage(im)
        return QPixmap.fromImage(qimg)
    except Exception:
        return pm
