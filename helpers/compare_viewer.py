# helpers/compare_viewer.py
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QPixmap, QImage
from PySide6.QtWidgets import QWidget


class CompareCanvas(QWidget):
    """
    Paints left and right images stacked with a wipe slider.
    wipe: 0..1000 (0 = left only, 1000 = right only)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._left_pix = None
        self._right_pix = None
        self._wipe = 500
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setMinimumSize(1, 1)

    def clear(self):
        self._left_pix = None
        self._right_pix = None
        self.update()

    def set_wipe(self, v: int):
        try:
            self._wipe = max(0, min(1000, int(v)))
        except Exception:
            self._wipe = 500
        self.update()

    def set_left_pixmap(self, pix: QPixmap):
        self._left_pix = pix if pix is not None and not pix.isNull() else None
        self.update()

    def set_right_pixmap(self, pix: QPixmap):
        self._right_pix = pix if pix is not None and not pix.isNull() else None
        self.update()

    def set_left_qimage(self, img: QImage):
        if img is None:
            self._left_pix = None
        else:
            try:
                self._left_pix = QPixmap.fromImage(img)
            except Exception:
                self._left_pix = None
        self.update()

    def set_right_qimage(self, img: QImage):
        if img is None:
            self._right_pix = None
        else:
            try:
                self._right_pix = QPixmap.fromImage(img)
            except Exception:
                self._right_pix = None
        self.update()

    def _fit_rect(self, pix: QPixmap, target: QRect) -> QRect:
        if pix is None or pix.isNull():
            return target
        tw, th = target.width(), target.height()
        pw, ph = pix.width(), pix.height()
        if pw <= 0 or ph <= 0 or tw <= 0 or th <= 0:
            return target

        # keep aspect ratio
        s = min(tw / pw, th / ph)
        w = int(pw * s)
        h = int(ph * s)
        x = target.x() + (tw - w) // 2
        y = target.y() + (th - h) // 2
        return QRect(x, y, w, h)

    def paintEvent(self, ev):
        p = QPainter(self)
        try:
            p.fillRect(self.rect(), self.palette().window())

            if self._left_pix is None and self._right_pix is None:
                return

            r = self.rect()

            # draw left full
            if self._left_pix is not None and not self._left_pix.isNull():
                lr = self._fit_rect(self._left_pix, r)
                p.drawPixmap(lr, self._left_pix)

            # draw right clipped by wipe
            if self._right_pix is not None and not self._right_pix.isNull():
                rr = self._fit_rect(self._right_pix, r)
                # wipe boundary in widget coords
                frac = float(self._wipe) / 1000.0
                cut_x = int(r.x() + r.width() * frac)

                p.save()
                p.setClipRect(QRect(r.x(), r.y(), max(0, cut_x - r.x()), r.height()))
                p.drawPixmap(rr, self._right_pix)
                p.restore()

                # draw divider line
                p.save()
                p.setPen(self.palette().text().color())
                p.drawLine(cut_x, r.y(), cut_x, r.y() + r.height())
                p.restore()
        finally:
            p.end()
