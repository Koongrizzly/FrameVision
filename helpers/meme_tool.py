from __future__ import annotations

import os, io, json
import math
import pathlib
from dataclasses import dataclass
from typing import List, Optional

from PySide6.QtCore import Qt, QRectF, QPointF, QSettings, Signal
from PySide6.QtGui import (QCursor, QColor, QFont, QFontDatabase, QImage, QPainter, QPainterPath, QPainterPathStroker,
                           QPen, QTransform, QPixmap, QFontMetrics)
from PySide6.QtWidgets import (QToolTip, 
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFormLayout,
    QSpinBox, QComboBox, QCheckBox, QFileDialog, QColorDialog, QInputDialog,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsPixmapItem, QGraphicsPathItem, QGraphicsDropShadowEffect, QScrollBar,
    QFontComboBox, QFontDialog, QDialog, QProgressDialog, QSlider
)

# Ensure QGraphicsRectItem is available (older stubs or packaging issues)
try:
    QGraphicsRectItem  # noqa: F821
except NameError:
    try:
        from PySide6.QtWidgets import QGraphicsRectItem as _QRectItem  # type: ignore
        QGraphicsRectItem = _QRectItem  # type: ignore
    except Exception:
        # Minimal fallback so the module still loads; behavior will be basic
        class QGraphicsRectItem(QGraphicsItem):  # type: ignore
            def __init__(self, rect=QRectF(), parent=None):
                super().__init__(parent)
                self._rect = QRectF(rect)
                self.setFlag(QGraphicsItem.ItemIsSelectable, True)
                self.setAcceptHoverEvents(True)
            def rect(self):
                return QRectF(self._rect)
            def setRect(self, r):
                self.prepareGeometryChange()
                self._rect = QRectF(r)
            def boundingRect(self):
                return QRectF(self._rect)
            def paint(self, painter, option, widget=None):
                painter.setPen(QPen(QColor(0, 200, 255, 255), 3, Qt.DashLine))
                painter.setBrush(QColor(0, 180, 255, 30))
                painter.drawRect(self._rect)


try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

@dataclass
class TextStyle:
    family: str = "Impact"
    size: int = 48
    arc: float = 0.0
    color_hex: str = "#ffffff"
    stroke_hex: str = "#000000"
    stroke_width: int = 4
    shadow: bool = True
    all_caps: bool = True
    align: str = "center"   # left / center / right
    auto_fit: bool = True
    min_size: int = 18
    max_size: int = 160
    extrude: int = 0
    letter_spacing: float = 0.0
    glow: int = 0
    @property
    def color(self): return QColor(self.color_hex)
    @property
    def stroke_color(self): return QColor(self.stroke_hex)

class MemeTextItem(QGraphicsPathItem):
    def shape(self):
        try:
            if self._path.isEmpty():
                return QGraphicsPathItem.shape(self)
            sw = float(max(8, getattr(self._style, 'stroke_width', 4) + 12))
            stroker = QPainterPathStroker(); stroker.setWidth(sw)
            return stroker.createStroke(self._path).united(self._path)
        except Exception:
            return QGraphicsPathItem.shape(self)

    def apply_transform(self, rot_deg: int, shear_x: float = 0.0):
        try:
            br = self.boundingRect().center()
            t = QTransform()
            t.translate(br.x(), br.y())
            if shear_x:
                t.shear(shear_x, 0.0)
            if rot_deg:
                t.rotate(rot_deg)
            t.translate(-br.x(), -br.y())
            self.setTransform(t, False)
        except Exception:
            pass

    """Movable/selectable text rendered as outlined glyph path; crisp stroke + fill."""
    def __init__(self, text: str, rect: QRectF, style: TextStyle, parent=None):
        super().__init__(parent)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self._rect = QRectF(rect)
        self._text = text
        self._style = style
        if style.shadow:
            eff = QGraphicsDropShadowEffect()
            eff.setBlurRadius(6)
            eff.setOffset(1, 1)
            eff.setColor(QColor(0,0,0,160))
            self.setGraphicsEffect(eff)
        self._path = QPainterPath()
        self._build_path()
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)

    def setRect(self, r: QRectF):
        self._rect = QRectF(r); self._build_path()
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)

    def rect(self) -> QRectF:
        return QRectF(self._rect)

    def text(self) -> str:
        return self._text

    def setText(self, t: str):
        self._text = t; self._build_path()
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)

    def setAlign(self, a: str):
        self._style.align = a; self._build_path()
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)

    def setStyle(self, style: TextStyle):
        self._style = style
        if style.shadow and not self.graphicsEffect():
            eff = QGraphicsDropShadowEffect(); eff.setBlurRadius(6); eff.setOffset(1,1); eff.setColor(QColor(0,0,0,160))
            self.setGraphicsEffect(eff)
        if not style.shadow and self.graphicsEffect():
            self.setGraphicsEffect(None)
        self._build_path()
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.OpenHandCursor)

    def _resolve_font(self) -> QFont:
        fam = self._style.family
        db = QFontDatabase()
        if fam not in db.families():
            for alt in ("Impact", "Arial Black", "DejaVu Sans", "Arial", "Noto Sans"):
                if alt in db.families():
                    fam = alt; break
        f = QFont(fam, self._style.size)
        f.setBold(True)
        try:
            f.setLetterSpacing(QFont.AbsoluteSpacing, float(getattr(self._style,'letter_spacing',0.0)))
        except Exception:
            pass
        return f

    def _build_path(self):
        r = self._rect
        self._path = QPainterPath()
        if r.width() <= 2 or r.height() <= 2:
            self.setPath(self._path); return
        txt = (self._text or "")
        if self._style.all_caps: txt = txt.upper()
        f = self._resolve_font()

        # Auto-fit across width using actual glyph bounds
        lines = (txt.splitlines() or [""])
        if self._style.auto_fit:
            widest = 1.0
            for ln in lines:
                test = QPainterPath(); test.addText(0, 0, f, ln)
                br = test.boundingRect()
                widest = max(widest, br.width())
            scale = min((r.width() * 0.98) / max(1.0, widest), 1.0)
            new_size = max(self._style.min_size, min(self._style.max_size, int(round(f.pointSize() * scale))))
            f.setPointSize(new_size)

        
        metrics = QFontMetrics(f)
        line_h = metrics.height() * 1.1
        y = 0.0
        for ln in lines:
            if abs(getattr(self._style, 'arc', 0.0)) > 0.1 and ln:
                A = float(self._style.arc)
                W = sum(max(1, metrics.horizontalAdvance(ch)) for ch in ln)
                if W <= 1:
                    W = 1
                Arad = math.radians(abs(A))
                R = (W / 2.0) / max(1e-3, math.sin(max(1e-7, Arad / 2.0))) if Arad > 1e-6 else 1e9
                sign = -1.0 if A >= 0 else 1.0

                # x alignment baseline
                test = QPainterPath(); test.addText(0, 0, f, ln)
                br_test = test.boundingRect()
                if self._style.align == "left":
                    xoff = -br_test.left()
                elif self._style.align == "right":
                    xoff = r.width() - br_test.right()
                else:
                    xoff = (r.width() - br_test.width()) / 2.0 - br_test.left()

                baseX = r.left() + xoff + br_test.center().x()
                baseY = r.top() + y + metrics.ascent()
                cursor = 0.0
                for ch in ln:
                    adv = float(max(1, metrics.horizontalAdvance(ch)))
                    xc = (cursor + adv / 2.0) - W / 2.0
                    ang = (xc / max(1.0, W / 2.0)) * (abs(A) / 2.0)
                    ang_rad = math.radians(ang)
                    px = R * math.sin(ang_rad)
                    py = sign * (R - R * math.cos(ang_rad))
                    chpth = QPainterPath(); chpth.addText(0, 0, f, ch)
                    brc = chpth.boundingRect()
                    t = QTransform()
                    t.translate(baseX + px, baseY + py)
                    t.rotate(-sign * ang)
                    t.translate(-brc.left() - brc.width() / 2.0, 0)
                    self._path.addPath(t.map(chpth))
                    cursor += adv
            else:
                pth = QPainterPath(); pth.addText(0, 0, f, ln)
                br = pth.boundingRect()
                if self._style.align == "left":
                    xoff = -br.left()
                elif self._style.align == "right":
                    xoff = r.width() - br.right()
                else:
                    xoff = (r.width() - br.width()) / 2.0 - br.left()
                m = QTransform().translate(r.left() + xoff, r.top() + y + metrics.ascent())
                self._path.addPath(m.map(pth))
            y += line_h

        self.setPath(self._path)

    def hoverEnterEvent(self, ev):
        self.setCursor(Qt.OpenHandCursor)
        super().hoverEnterEvent(ev)
    def hoverLeaveEvent(self, ev):
        self.setCursor(Qt.ArrowCursor)
        super().hoverLeaveEvent(ev)
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            try:
                self.setSelected(True)
                self.setFocus()
                self.setCursor(Qt.ClosedHandCursor)
            except Exception:
                pass
        super().mousePressEvent(ev)
    
    def _install_shade(self):
        try:
            from PySide6.QtWidgets import QGraphicsPathItem
            self._shade = QGraphicsPathItem(self)
            self._shade.setZValue(self.zValue() - 1)
            self._shade.setBrush(QColor(0, 0, 0, 120))
            self._shade.setPen(Qt.NoPen)
            self._update_shade()
        except Exception:
            self._shade = None

    def _update_shade(self):
        if not self._shade:
            return
        try:
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            path.addRect(self.bounds)
            path.addRect(self.rect())
            self._shade.setPath(path)
        except Exception:
            pass

    def _update_hud(self):
        try:
            if not getattr(self, '_hud', None):
                return
            r = self.rect(); w = int(round(r.width())); h = int(round(r.height()))
            ar = (w / float(h)) if h else 0
            txt = f"{w}×{h}  AR={ar:.3f}"
            self._hud.setText(txt)
            pos = QPointF(r.left()+6, r.top()+6)
            self._hud.setPos(pos)
            if getattr(self, '_hud_bg', None):
                br = self._hud.boundingRect()
                self._hud_bg.setRect(QRectF(pos.x()-4, pos.y()-2, br.width()+8, br.height()+4))
        except Exception:
            pass

    def paint(self, painter: QPainter, option, widget=None):
        if self._path.isEmpty(): return
        painter.setRenderHint(QPainter.Antialiasing, True)
        depth = int(getattr(self._style, 'extrude', 0) or 0)
        if depth>0:
            painter.save(); painter.setPen(Qt.NoPen); painter.setBrush(QColor(0,0,0,120))
            for i in range(depth,0,-1):
                painter.save(); painter.translate(i*0.7, i*0.7); painter.drawPath(self._path); painter.restore()
            painter.restore()
        # Glow under stroke/fill (fast: constant rings)
        try:
            glw = int(getattr(self._style, 'glow', 0) or 0)
            if glw>0:
                base = QColor(self._style.color)
                rings = max(3, min(10, glw))  # 3..10 rings only, regardless of value
                scale = max(2.0, float(glw) * 2.0)
                for i in range(rings, 0, -1):
                    t = i / float(rings)
                    w = max(1.0, scale * t)
                    c = QColor(base); c.setAlpha(int(140 * (t*t)))
                    painter.setPen(QPen(c, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(self._path)
        except Exception:
            pass
        pen = QPen(self._style.stroke_color, float(max(1, self._style.stroke_width)), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen); painter.setBrush(Qt.NoBrush); painter.drawPath(self._path)
        painter.setPen(Qt.NoPen); painter.setBrush(self._style.color); painter.drawPath(self._path)

    def mouseReleaseEvent(self, ev):
        """Snap to image edges on release (small padding)."""
        super().mouseReleaseEvent(ev)
        s = self.scene(); img = getattr(s, "_img_item", None)
        if not s or not isinstance(img, QGraphicsPixmapItem): return
        bounds = img.sceneBoundingRect(); pad = 8.0
        r = self.sceneBoundingRect(); pos = self.pos(); newpos = QPointF(pos)
        if abs(r.top() - bounds.top()) < pad: newpos.setY(pos.y() + (bounds.top() - r.top()))
        if abs(r.bottom() - bounds.bottom()) < pad: newpos.setY(pos.y() + (bounds.bottom() - r.bottom()))
        if abs(r.left() - bounds.left()) < pad: newpos.setX(pos.x() + (bounds.left() - r.left()))
        if abs(r.right() - bounds.right()) < pad: newpos.setX(pos.x() + (bounds.right() - r.right()))
        self.setPos(newpos)

class MemeGraphicsView(QGraphicsView):
    def mousePressEvent(self, ev):
        try:
            if ev.button() == Qt.LeftButton:
                for it in self.items(ev.pos()):
                    from helpers.meme_tool import MemeTextItem as _MItem
                    if isinstance(it, _MItem):
                        it.setSelected(True)
                        it.setFocus()
                        break
        except Exception:
            pass
        super().mousePressEvent(ev)

    nudged = Signal(int, int)
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setDragMode(QGraphicsView.RubberBandDrag)
    def resizeEvent(self, e):
        super().resizeEvent(e)
        try:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        except Exception:
            pass
    def keyPressEvent(self, e):
        step = 1 if not (e.modifiers() & Qt.ShiftModifier) else 10
        dx=dy=0
        if e.key()==Qt.Key_Left: dx=-step
        elif e.key()==Qt.Key_Right: dx=step
        elif e.key()==Qt.Key_Up: dy=-step
        elif e.key()==Qt.Key_Down: dy=step
        if dx or dy: self.nudged.emit(dx,dy); return
        if e.key()==Qt.Key_Escape:
            try:
                pane = getattr(self, 'pane', None)
                if pane and getattr(pane, '_crop_overlay', None) is not None:
                    pane._clear_crop_overlay(); return
            except Exception:
                pass
        super().keyPressEvent(e)


class CropRectItem(QGraphicsRectItem):
    """Interactive crop rectangle with movable/resizable handles and optional fixed aspect ratio.
       Coordinates are in scene space (we set pos=(0,0) and use absolute rects)."""
    Handle_None, Handle_Move, Handle_L, Handle_R, Handle_T, Handle_B, Handle_TL, Handle_TR, Handle_BL, Handle_BR = range(10)
    def __init__(self, rect: QRectF, bounds: QRectF, aspect: Optional[float] = None, parent=None):
        super().__init__(rect, parent)
        self.setZValue(9999)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.aspect = float(aspect) if aspect else None
        self.bounds = QRectF(bounds)
        self.handle_size = 18.0
        self._mode = self.Handle_None
        self._start_rect = QRectF(rect)
        self._last_pos = QPointF()
        self.apply_cb = None
        self.cancel_cb = None

        pen = QPen(QColor(0, 200, 255, 255), 3, Qt.DashLine, Qt.SquareCap, Qt.MiterJoin)
        self.setPen(pen)
        self._shade = None  # optional outside shade
        self._install_shade()
        # live HUD
        try:
            self._hud = QGraphicsSimpleTextItem("", self)
            f = self._hud.font(); f.setPointSize(max(8, f.pointSize()))
            self._hud.setFont(f)
            self._hud_bg = QGraphicsRectItem(self)
            self._hud_bg.setPen(Qt.NoPen)
            self._hud_bg.setBrush(QColor(0,0,0,160))
            self._hud.setZValue(self.zValue()+2)
            self._hud_bg.setZValue(self.zValue()+1)
            self._update_hud()
        except Exception:
            self._hud = None; self._hud_bg = None

    def _handles(self):
        r = self.rect()
        s = self.handle_size
        return {
            self.Handle_TL: QRectF(r.left()-s/2,  r.top()-s/2,     s, s),
            self.Handle_TR: QRectF(r.right()-s/2, r.top()-s/2,     s, s),
            self.Handle_BL: QRectF(r.left()-s/2,  r.bottom()-s/2,  s, s),
            self.Handle_BR: QRectF(r.right()-s/2, r.bottom()-s/2,  s, s),
            self.Handle_L:  QRectF(r.left()-s/2,  r.center().y()-s/2, s, s),
            self.Handle_R:  QRectF(r.right()-s/2, r.center().y()-s/2, s, s),
            self.Handle_T:  QRectF(r.center().x()-s/2, r.top()-s/2, s, s),
            self.Handle_B:  QRectF(r.center().x()-s/2, r.bottom()-s/2, s, s),
        }

    def _hit_test(self, pos: QPointF):
        # pos is in item coordinates; since we keep pos=(0,0), it's scene-space
        for k, rc in self._handles().items():
            if rc.contains(pos):
                return k
        if self.rect().contains(pos):
            return self.Handle_Move
        return self.Handle_None

    def hoverMoveEvent(self, ev):
        h = self._hit_test(ev.pos())
        curs = Qt.ArrowCursor
        if h in (self.Handle_L, self.Handle_R):
            curs = Qt.SizeHorCursor
        elif h in (self.Handle_T, self.Handle_B):
            curs = Qt.SizeVerCursor
        elif h in (self.Handle_TL, self.Handle_BR):
            curs = Qt.SizeFDiagCursor
        elif h in (self.Handle_TR, self.Handle_BL):
            curs = Qt.SizeBDiagCursor
        elif h == self.Handle_Move:
            curs = Qt.SizeAllCursor
        self.setCursor(curs)
        super().hoverMoveEvent(ev)

    def mousePressEvent(self, ev):
        self._mode = self._hit_test(ev.pos())
        self._start_rect = QRectF(self.rect())
        self._last_pos = ev.pos()
        super().mousePressEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        try:
            cb = getattr(self, 'apply_cb', None)
            if callable(cb): cb(); return
        except Exception:
            pass
        super().mouseDoubleClickEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._mode == self.Handle_None:
            super().mouseMoveEvent(ev); return
        p = ev.pos()
        r0 = QRectF(self._start_rect)
        b = QRectF(self.bounds)
        def clamp_rect(rc: QRectF):
            if rc.width() < 2: rc.setWidth(2)
            if rc.height() < 2: rc.setHeight(2)
            # keep within bounds
            dx = 0.0; dy = 0.0
            if rc.left() < b.left(): dx = b.left() - rc.left()
            if rc.right() > b.right(): dx = b.right() - rc.right()
            if rc.top() < b.top(): dy = b.top() - rc.top()
            if rc.bottom() > b.bottom(): dy = b.bottom() - rc.bottom()
            rc.translate(dx, dy)
            return rc

        if self._mode == self.Handle_Move:
            delta = p - self._last_pos
            r = QRectF(r0); r.translate(delta)
            r = clamp_rect(r)
            self.setRect(r)
            self._update_shade()
            self._update_hud()
            self._maybe_autoscroll(ev)
            return

        # Resizing
        anchor = QPointF(r0.right(), r0.bottom())
        moving_corner = QPointF(r0.left(), r0.top())
        # set based on which handle
        if self._mode == self.Handle_TL:
            anchor = QPointF(r0.right(), r0.bottom()); moving_corner = p
        elif self._mode == self.Handle_TR:
            anchor = QPointF(r0.left(), r0.bottom()); moving_corner = p
        elif self._mode == self.Handle_BL:
            anchor = QPointF(r0.right(), r0.top()); moving_corner = p
        elif self._mode == self.Handle_BR:
            anchor = QPointF(r0.left(), r0.top()); moving_corner = p
        elif self._mode == self.Handle_L:
            anchor = QPointF(r0.right(), r0.center().y()); moving_corner = QPointF(p.x(), anchor.y())
        elif self._mode == self.Handle_R:
            anchor = QPointF(r0.left(), r0.center().y()); moving_corner = QPointF(p.x(), anchor.y())
        elif self._mode == self.Handle_T:
            anchor = QPointF(r0.center().x(), r0.bottom()); moving_corner = QPointF(anchor.x(), p.y())
        elif self._mode == self.Handle_B:
            anchor = QPointF(r0.center().x(), r0.top()); moving_corner = QPointF(anchor.x(), p.y())

        ax, ay = anchor.x(), anchor.y()
        px, py = moving_corner.x(), moving_corner.y()

        # enforce aspect if set
        if self.aspect and self.aspect > 0:
            dx = px - ax; dy = py - ay
            if dx == 0 and dy == 0:
                pass
            else:
                # choose the dominant direction while preserving sign
                if abs(dx) / max(1e-6, abs(dy)) >= self.aspect:
                    dy = (abs(dx) / self.aspect) * (1 if dy >= 0 else -1)
                else:
                    dx = (abs(dy) * self.aspect) * (1 if dx >= 0 else -1)
                px = ax + dx; py = ay + dy

        new_rect = QRectF(anchor, QPointF(px, py)).normalized()
        if ev.modifiers() & Qt.AltModifier:
            c = self._start_rect.center()
            half_w = abs(new_rect.width())/2.0
            half_h = abs(new_rect.height())/2.0
            if self.aspect and self.aspect>0:
                half_h = half_w / self.aspect
            new_rect = QRectF(QPointF(c.x()-half_w, c.y()-half_h), QPointF(c.x()+half_w, c.y()+half_h)).normalized()
        new_rect = clamp_rect(new_rect)
        # If aspect was requested, ensure it still matches (clamping can violate)
        if self.aspect and self.aspect > 0:
            # adjust height to match ratio with width
            w = new_rect.width()
            h = w / self.aspect
            # anchor at anchor point
            if anchor.x() <= new_rect.center().x():
                # anchor at left side
                x1 = anchor.x(); x2 = anchor.x() + w
            else:
                x1 = anchor.x() - w; x2 = anchor.x()
            if anchor.y() <= new_rect.center().y():
                y1 = anchor.y(); y2 = anchor.y() + h
            else:
                y1 = anchor.y() - h; y2 = anchor.y()
            new_rect = QRectF(QPointF(min(x1,x2), min(y1,y2)), QPointF(max(x1,x2), max(y1,y2)))
            new_rect = clamp_rect(new_rect)

        self.setRect(new_rect)
        self._update_shade()
        self._update_hud()

    
    def _install_shade(self):
        try:
            from PySide6.QtWidgets import QGraphicsPathItem
            self._shade = QGraphicsPathItem(self)
            self._shade.setZValue(self.zValue() - 1)
            self._shade.setBrush(QColor(0, 0, 0, 120))
            self._shade.setPen(Qt.NoPen)
            self._update_shade()
        except Exception:
            self._shade = None

    def _update_shade(self):
        if not self._shade:
            return
        try:
            path = QPainterPath()
            path.setFillRule(Qt.OddEvenFill)
            path.addRect(self.bounds)
            path.addRect(self.rect())
            self._shade.setPath(path)
        except Exception:
            pass

    def _update_hud(self):
        try:
            if not getattr(self, '_hud', None):
                return
            r = self.rect(); w = int(round(r.width())); h = int(round(r.height()))
            ar = (w / float(h)) if h else 0
            txt = f"{w}×{h}  AR={ar:.3f}"
            self._hud.setText(txt)
            pos = QPointF(r.left()+6, r.top()+6)
            self._hud.setPos(pos)
            if getattr(self, '_hud_bg', None):
                br = self._hud.boundingRect()
                self._hud_bg.setRect(QRectF(pos.x()-4, pos.y()-2, br.width()+8, br.height()+4))
        except Exception:
            pass

    def paint(self, painter: QPainter, option, widget=None):
        super().paint(painter, option, widget)
        # draw handles
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(QColor(255, 255, 255, 200))
        painter.setPen(QPen(QColor(0, 0, 0, 255), 1))
        for rc in self._handles().values():
            painter.drawRect(rc)

class CollapsibleBox(QWidget):
    """Small collapsible container used for Font & style."""
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._toggle = QPushButton(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)  # default removed; will restore via UI
        try:
            self._toggle.setStyleSheet("QPushButton{background:#0e2036;color:#a8d7ff;border:1px solid #1c4a76;border-radius:8px;padding:6px 10px;text-align:left} QPushButton:checked{background:#113257}")
        except Exception:
            pass
        self._content = QWidget()
        self._content.setVisible(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(6)
        lay.addWidget(self._toggle); lay.addWidget(self._content)
        self._toggle.toggled.connect(self._content.setVisible)
    def setContentLayout(self, layout):
        self._content.setLayout(layout)
class MemeToolPane(QWidget):
    """Meme / Caption tool widget."""
    def __init__(self, main=None, parent=None):
        super().__init__(parent)
        self.main = main
        self.settings = QSettings("FrameVision","FrameVision")
        self._last_dir = self.settings.value("meme/last_dir","") or ""
        self._pending_batch: List[str] = []
        self._init_defaults()
        self._load_extra_fonts_from_settings()
        self._build_ui()
        self._connect()

    # ---- persistence ----
    def _init_defaults(self):
        g = self.settings
        self.def_family = g.value("meme/font_family","Impact")
        self.def_size = int(g.value("meme/font_size", 48))
        self.def_fill = QColor(g.value("meme/fill","#ffffff"))
        self.def_stroke = QColor(g.value("meme/stroke","#000000"))
        self.def_stroke_w = int(g.value("meme/stroke_w", 4))
        self.def_shadow = bool(int(g.value("meme/shadow", 1)))
        self.def_allcaps = bool(int(g.value("meme/allcaps", 1)))
        self.def_align = g.value("meme/align","center")
        self.def_auto = bool(int(g.value("meme/auto_fit", 1)))
        self.def_track = float(g.value("meme/track", 0.0))
        self.def_glow = int(g.value("meme/glow", 0))
        self.out_fmt = g.value("meme/out_fmt","png")
        self.out_quality = int(g.value("meme/out_quality", 92))
        self.keep_meta = bool(int(g.value("meme/keep_meta", 1)))

    def _save_defaults(self):
        g = self.settings
        g.setValue("meme/font_family", self.def_family)
        g.setValue("meme/font_size", self.def_size)
        g.setValue("meme/fill", self.def_fill.name())
        g.setValue("meme/stroke", self.def_stroke.name())
        g.setValue("meme/stroke_w", self.def_stroke_w)
        g.setValue("meme/shadow", int(self.def_shadow))
        g.setValue("meme/allcaps", int(self.def_allcaps))
        g.setValue("meme/align", self.def_align)
        g.setValue("meme/auto_fit", int(self.def_auto))
        g.setValue("meme/track", float(self.def_track))
        g.setValue("meme/glow", int(self.def_glow))
        g.setValue("meme/out_fmt", self.out_fmt)
        g.setValue("meme/out_quality", self.out_quality)
        g.setValue("meme/keep_meta", int(self.keep_meta))
        g.setValue("meme/last_dir", self._last_dir)

    def _load_extra_fonts_from_settings(self):
        try:
            paths_json = self.settings.value("meme/extra_fonts","[]")
            paths = json.loads(paths_json) if isinstance(paths_json,str) else (paths_json or [])
        except Exception:
            paths = []
        for p in paths:
            try:
                fid = QFontDatabase.addApplicationFont(p)
                _ = QFontDatabase.applicationFontFamilies(fid)
            except Exception:
                pass

    def _persist_extra_fonts(self, new_paths: List[str]):
        try:
            paths_json = self.settings.value("meme/extra_fonts","[]")
            paths = json.loads(paths_json) if isinstance(paths_json,str) else (paths_json or [])
            for p in new_paths:
                if p not in paths: paths.append(p)
            self.settings.setValue("meme/extra_fonts", json.dumps(paths))
        except Exception:
            pass

    # ---- UI ----
    def _build_ui(self):
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0); lay.setSpacing(8)

        # Row 1
        ctrl = QHBoxLayout(); ctrl.setSpacing(8)
        self.btn_use_current = QPushButton("Use current image"); self._update_use_current_enabled()
        self.btn_open = QPushButton("Open image…")
        self.btn_big = QPushButton("Big preview")
        ctrl.addWidget(self.btn_use_current); ctrl.addWidget(self.btn_open); ctrl.addStretch(1); ctrl.addWidget(self.btn_big)
        lay.addLayout(ctrl)

        # Row 2
        ctrl2 = QHBoxLayout(); ctrl2.setSpacing(8)
        self.btn_add = QPushButton("Add text"); self.btn_edit = QPushButton("Edit selected…")
        self.btn_remove = QPushButton("Remove selected"); self.btn_reset = QPushButton("Reset positions")
        ctrl2.addWidget(self.btn_add); ctrl2.addWidget(self.btn_edit); ctrl2.addWidget(self.btn_remove); ctrl2.addWidget(self.btn_reset)
        lay.addLayout(ctrl2)

        # Crop quick presets
        crop_row = QHBoxLayout()
        self.btn_crop_169 = QPushButton("Crop 16:9")
        self.btn_crop_916 = QPushButton("Crop 9:16")
        self.btn_crop_reset = QPushButton("Reset crop")
        crop_row.addWidget(self.btn_crop_169); crop_row.addWidget(self.btn_crop_916); crop_row.addWidget(self.btn_crop_reset);
        self.btn_crop_apply = QPushButton("Apply crop")
        crop_row.addWidget(self.btn_crop_apply); crop_row.addStretch(1)

        crop_set_box = CollapsibleBox("Crop overlay")
        lay.addWidget(crop_set_box)
        crop_set_form = QFormLayout(); crop_set_box.setContentLayout(crop_set_form)
        self.sl_handle = QSlider(Qt.Horizontal); self.sl_handle.setRange(8, 40); self.sl_handle.setValue(int(self.settings.value('meme/crop_handle', 18)))
        self.sl_opacity = QSlider(Qt.Horizontal); self.sl_opacity.setRange(0, 255); self.sl_opacity.setValue(int(self.settings.value('meme/crop_opacity', 120)))
        crop_set_form.addRow("Handle size", self.sl_handle)
        crop_set_form.addRow("Overlay opacity", self.sl_opacity)
        lay.addLayout(crop_row)

        # Output (moved to top)
        out = QHBoxLayout()
        self.cmb_fmt = QComboBox(); self.cmb_fmt.addItems(["png","jpg","webp"]); self.cmb_fmt.setCurrentText(self.out_fmt)
        self.sp_quality = QSpinBox(); self.sp_quality.setRange(1,100); self.sp_quality.setValue(self.out_quality)
        self.chk_keepmeta = QCheckBox("Keep metadata (EXIF)"); self.chk_keepmeta.setChecked(self.keep_meta)
        self.btn_export = QPushButton("Export → Save as…"); self.btn_batch = QPushButton("Batch…")
        out.addWidget(QLabel("Format")); out.addWidget(self.cmb_fmt)
        out.addWidget(QLabel("Quality")); out.addWidget(self.sp_quality)
        out.addWidget(self.chk_keepmeta); out.addStretch(1); out.addWidget(self.btn_export); out.addWidget(self.btn_batch)
        lay.addLayout(out)

        # Preview
        self.view = MemeGraphicsView()
        self.view.pane = self
        self.view.setMinimumHeight(300)
        self.scene = QGraphicsScene(self); self.view.setScene(self.scene)
        self.view.setStyleSheet("QGraphicsView{border:1px solid rgba(255,255,255,30);}")
        lay.addWidget(self.view, 1)

        


        # Crop overlay state
        self._crop_overlay = None
        self._crop_aspect = None
        self._view_prev_drag = self.view.dragMode()
        # Transform (sliders) — placed between preview and fonts
        form_xform = QFormLayout()
        self.sl_rot = QSlider(Qt.Horizontal); self.sl_rot.setRange(0, 360); self.sl_rot.setValue(0); self.sl_rot.setTracking(True)
        self.sl_shear = QSlider(Qt.Horizontal); self.sl_shear.setRange(-40, 40); self.sl_shear.setValue(0); self.sl_shear.setTracking(True)
        self.sl_glow = QSlider(Qt.Horizontal); self.sl_glow.setRange(0, 40); self.sl_glow.setValue(int(self.def_glow)); self.sl_glow.setTracking(True)
        self.sl_track = QSlider(Qt.Horizontal); self.sl_track.setRange(-30, 60); self.sl_track.setValue(int(self.def_track)); self.sl_track.setTracking(True)
        self.sl_arc = QSlider(Qt.Horizontal); self.sl_arc.setRange(-170, 170); self.sl_arc.setValue(0)
        self.sl_arc.installEventFilter(self)
        self._rot_row_widget = QWidget(); _rot_row_lay = QHBoxLayout(self._rot_row_widget); _rot_row_lay.setContentsMargins(0,0,0,0); _rot_row_lay.setSpacing(8)
        self.sp_rot = QSpinBox(); self.sp_rot.setRange(0,360); self.sp_rot.setFixedWidth(64)
        _rot_row_lay.addWidget(self.sl_rot, 1); _rot_row_lay.addWidget(self.sp_rot)
        form_xform.addRow("Rotate", self._rot_row_widget)
        self._shear_row_widget = QWidget(); _shear_row_lay = QHBoxLayout(self._shear_row_widget); _shear_row_lay.setContentsMargins(0,0,0,0); _shear_row_lay.setSpacing(8)
        self.sp_shear = QSpinBox(); self.sp_shear.setRange(-40,40); self.sp_shear.setFixedWidth(64)
        _shear_row_lay.addWidget(self.sl_shear, 1); _shear_row_lay.addWidget(self.sp_shear)
        form_xform.addRow("Tilt/Skew", self._shear_row_widget)
        self._glow_row_widget = QWidget(); _glow_row_lay = QHBoxLayout(self._glow_row_widget); _glow_row_lay.setContentsMargins(0,0,0,0); _glow_row_lay.setSpacing(8)
        self.sp_glow = QSpinBox(); self.sp_glow.setRange(0,40); self.sp_glow.setFixedWidth(64)
        _glow_row_lay.addWidget(self.sl_glow, 1); _glow_row_lay.addWidget(self.sp_glow)
        form_xform.addRow("Outer glow", self._glow_row_widget)
        self._track_row_widget = QWidget(); _track_row_lay = QHBoxLayout(self._track_row_widget); _track_row_lay.setContentsMargins(0,0,0,0); _track_row_lay.setSpacing(8)
        self.sp_track = QSpinBox(); self.sp_track.setRange(-30,60); self.sp_track.setFixedWidth(64)
        _track_row_lay.addWidget(self.sl_track, 1); _track_row_lay.addWidget(self.sp_track)
        form_xform.addRow("Letter spacing", self._track_row_widget)
        self._arc_row_widget = QWidget(self)

        _arc_row_lay = QHBoxLayout(self._arc_row_widget)

        _arc_row_lay.setContentsMargins(0,0,0,0)

        self.sp_arc = QSpinBox(self)

        self.sp_arc.setRange(-170, 170)

        self.sp_arc.setFixedWidth(64)

        self.sp_arc.setValue(self.sl_arc.value())
        self.sp_arc.valueChanged.connect(self.sl_arc.setValue)
        self.sl_arc.valueChanged.connect(self.sp_arc.setValue)

        _arc_row_lay.addWidget(self.sl_arc, 1)

        _arc_row_lay.addWidget(self.sp_arc, 0)

        form_xform.addRow("Arc curvature", self._arc_row_widget)
        lay.addLayout(form_xform)
# Form
        form = QFormLayout()

        self.font_combo = QFontComboBox()
        try:
            self.font_combo.setStyleSheet('QFontComboBox{background:#0e2036;color:#cfe9ff;border:1px solid #1c4a76;border-radius:8px;padding:4px 6px}')
        except Exception:
            pass
        try: self.font_combo.setCurrentFont(QFont(self.def_family))
        except Exception: pass
        self.btn_font_add = QPushButton("Add fonts…")
        self.btn_font_folder = QPushButton("Load folder…")
        self.btn_font_dialog = QPushButton("Font…")
        row_font = QHBoxLayout(); row_font.addWidget(self.font_combo); row_font.addWidget(self.btn_font_add); row_font.addWidget(self.btn_font_folder); row_font.addWidget(self.btn_font_dialog)
        form.addRow("Font", row_font)

        self.cmb_align = QComboBox(); self.cmb_align.addItems(["left","center","right"]); self.cmb_align.setCurrentText(self.def_align)
        self.sp_size = QSpinBox(); self.sp_size.setRange(8, 400); self.sp_size.setValue(self.def_size)
        self.sp_stroke = QSpinBox(); self.sp_stroke.setRange(0, 20); self.sp_stroke.setValue(self.def_stroke_w)
        self.chk_auto = QCheckBox("Auto-fit"); self.chk_auto.setChecked(self.def_auto)
        self.chk_shadow = QCheckBox("Shadow"); self.chk_shadow.setChecked(self.def_shadow)
        self.chk_allcaps = QCheckBox("ALL CAPS"); self.chk_allcaps.setChecked(self.def_allcaps)
        self.btn_fill = QPushButton("Text color…"); self.btn_stroke = QPushButton("Outline color…")
        form.addRow("Align", self.cmb_align)
        form.addRow("Font size", self.sp_size)
        form.addRow("Outline width", self.sp_stroke)
        
        form.addRow(self.btn_fill); form.addRow(self.btn_stroke)
        form.addRow(self.chk_auto); form.addRow(self.chk_shadow); form.addRow(self.chk_allcaps)
        fonts_box = CollapsibleBox("Font & style"); lay.addWidget(fonts_box); fonts_box.setContentLayout(form)

        # state
        self._img_item: Optional[QGraphicsPixmapItem] = None
        self._items: List[MemeTextItem] = []

        self._update_quality_enabled()

    def _connect(self):

        # live slider bindings
        try:
            self.sp_rot.valueChanged.connect(self.sl_rot.setValue)
            self.sl_rot.valueChanged.connect(self._on_rot_changed)
            self.sl_rot.valueChanged.connect(self.sp_rot.setValue)
            self.sp_shear.valueChanged.connect(self.sl_shear.setValue)
            self.sl_shear.valueChanged.connect(self._on_shear_changed)
            self.sl_shear.valueChanged.connect(self.sp_shear.setValue)
            self.sp_glow.valueChanged.connect(self.sl_glow.setValue)
            self.sl_glow.valueChanged.connect(self._on_glow_changed)
            self.sl_glow.valueChanged.connect(self.sp_glow.setValue)
            self.sp_track.valueChanged.connect(self.sl_track.setValue)
            self.sl_track.valueChanged.connect(self._on_track_changed)
            self.sl_track.valueChanged.connect(self.sp_track.setValue)
            self.sl_arc.valueChanged.connect(self._on_arc_changed)
        except Exception:
            pass
        self.view.nudged.connect(self._nudge_selected)
        self.btn_use_current.clicked.connect(self._use_current)
        self.btn_open.clicked.connect(self._open_image)
        self.btn_big.clicked.connect(self._open_big_preview)
        self.btn_add.clicked.connect(self._add_text)
        self.btn_edit.clicked.connect(self._edit_selected_text)
        self.btn_remove.clicked.connect(self._remove_selected)
        self.btn_reset.clicked.connect(self._reset_positions)
        self.cmb_align.currentTextChanged.connect(self._apply_to_selected)
        self.sp_size.valueChanged.connect(self._apply_to_selected)
        self.sp_stroke.valueChanged.connect(self._apply_to_selected)
        self.chk_auto.toggled.connect(self._apply_to_selected)
        self.chk_shadow.toggled.connect(self._apply_to_selected)
        self.chk_allcaps.toggled.connect(self._apply_to_selected)
        self.btn_fill.clicked.connect(lambda: self._pick_color(True))
        self.btn_stroke.clicked.connect(lambda: self._pick_color(False))
        self.cmb_fmt.currentTextChanged.connect(lambda _: self._update_quality_enabled())
        self.btn_export.clicked.connect(self._export_single)
        self.btn_batch.clicked.connect(self._export_batch)
        self.font_combo.currentFontChanged.connect(self._on_font_changed)
        self.btn_font_add.clicked.connect(self._on_add_fonts)
        self.btn_font_folder.clicked.connect(self._on_add_font_folder)
        self.btn_font_dialog.clicked.connect(self._on_font_dialog)
        self.btn_crop_169.clicked.connect(lambda: self._begin_crop(16,9))
        self.btn_crop_916.clicked.connect(lambda: self._begin_crop(9,16))
        self.btn_crop_reset.clicked.connect(self._reset_crop)
        self.btn_crop_apply.clicked.connect(self._apply_crop)
        self.sl_handle.valueChanged.connect(self._on_crop_ui_changed)
        self.sl_opacity.valueChanged.connect(self._on_crop_ui_changed)

    def _apply_transform_to_selection(self):
        try:
            rot = int(self.sl_rot.value())
            shear = float(self.sl_shear.value()) / 100.0
            items = list(getattr(self, "_items", []))
            sel = [it for it in items if getattr(it, "isSelected", lambda: False)()]
            targets = sel if sel else items
            for it in targets:
                try:
                    it.apply_transform(rot, shear_x=shear); it.update()
                except Exception:
                    pass
        except Exception:
            pass


    def _apply_style_depth_to_selection(self):
        try:
            depth = int(self.sl_depth.value())
            items = list(getattr(self, "_items", []))
            sel = [it for it in items if getattr(it, "isSelected", lambda: False)()]
            targets = sel if sel else items
            for it in targets:
                try:
                    st = getattr(it, "_style", None)
                    if st is None:
                        continue
                    st.extrude = depth
                    it.setStyle(st)
                except Exception:
                    pass
        except Exception:
            pass


    def _apply_arc_to_selection(self):
        try:
            arc = float(self.sl_arc.value())
            items = list(getattr(self, "_items", []))
            sel = [it for it in items if getattr(it, "isSelected", lambda: False)()]
            if not sel:
                return  # only change selected text
            for it in sel:
                try:
                    st = getattr(it, "_style", None)
                    if st is None:
                        continue
                    st.arc = arc
                    it.setStyle(st)
                except Exception:
                    pass
        except Exception:
            pass


    def _on_rot_changed(self, val):
        try: QToolTip.showText(QCursor.pos(), f"{int(val)}°")
        except Exception: pass
        try:
            self._apply_transform_to_selection()
            self.scene.update()
        except Exception:
            pass

    def _on_shear_changed(self, val):
        try: QToolTip.showText(QCursor.pos(), f"{int(val)}%")
        except Exception: pass
        try:
            self._apply_transform_to_selection()
            self.scene.update()
        except Exception:
            pass

    def _on_glow_changed(self, val):
        try: QToolTip.showText(QCursor.pos(), f"{int(val)} px")
        except Exception: pass
        try:
            self.def_glow = int(val); self.settings.setValue("meme/glow", int(val))
            items = list(getattr(self, "_items", []))
            sel = [it for it in items if getattr(it, "isSelected", lambda: False)()]
            if not sel:
                return  # only change selected text; leave others alone
            for it in sel:
                st = getattr(it, "_style", None)
                if st is None: continue
                st.glow = int(val); it.setStyle(st); it.update()
        except Exception: pass


    def _on_track_changed(self, val):
        try: QToolTip.showText(QCursor.pos(), f"{int(val)} px")
        except Exception: pass
        try:
            self.def_track = float(val); self.settings.setValue("meme/track", float(val))
            items = list(getattr(self, "_items", []))
            sel = [it for it in items if getattr(it, "isSelected", lambda: False)()]
            targets = sel if sel else items
            for it in targets:
                st = getattr(it, "_style", None)
                if st is None: continue
                st.letter_spacing = float(val); it.setStyle(st); it.update()
        except Exception: pass


    def _on_arc_changed(self, val):
        try:
            QToolTip.showText(QCursor.pos(), f"{int(val)}°")
        except Exception:
            pass
        try:
            self._apply_arc_to_selection()
        except Exception:
            pass

    def _update_quality_enabled(self):
        self.sp_quality.setEnabled(self.cmb_fmt.currentText().lower() in ("jpg","webp"))

    def _update_use_current_enabled(self):
        path = self._current_path()
        ok = bool(path and str(path).lower().endswith((".png",".jpg",".jpeg",".webp",".mp4",".mkv",".mov",".webm",".avi",".m4v",".mpg",".mpeg",".ts",".m2ts")))
        self.btn_use_current.setEnabled(ok)

    def _current_path(self) -> Optional[str]:
        try:
            m = self.main
            path = getattr(m, "current_path", None) or getattr(m, "current_image", None)
            if path and isinstance(path, str) and os.path.exists(path):
                return path
        except Exception:
            pass
        return None

    def _load_image_to_scene(self, path: str):
        if not path: return
        pm = QPixmap(path)
        if pm.isNull():
            QMessageBox.warning(self,"Meme","Could not open image."); return
        self.scene.clear()
        self._orig_pm = pm
        self._img_item = QGraphicsPixmapItem(pm); self._img_item.setAcceptedMouseButtons(Qt.NoButton); self.scene.addItem(self._img_item)
        self.scene._img_item = self._img_item
        self.scene.setSceneRect(pm.rect())
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self._items.clear()
        R = self.scene.sceneRect()
        pad = 12
        top_rect = QRectF(R.left()+pad, R.top()+pad, R.width()-2*pad, R.height()*0.25)
        bot_rect = QRectF(R.left()+pad, R.bottom()-R.height()*0.25-pad, R.width()-2*pad, R.height()*0.25)
        st = self._current_style_defaults()
        self._items.append(self._make_item("TOP TEXT", top_rect, st))
        self._items.append(self._make_item("BOTTOM TEXT", bot_rect, st))

    
    def _clear_crop_overlay(self):
        try:
            if self._crop_overlay is not None:
                self.scene.removeItem(self._crop_overlay)
        except Exception:
            pass
        self._crop_overlay = None
        self._crop_aspect = None
        try:
            # restore drag mode
            if hasattr(self, "view") and hasattr(self.view, "setDragMode"):
                self.view.setDragMode(QGraphicsView.RubberBandDrag)
        except Exception:
            pass

    def _begin_crop(self, aspect_w: int, aspect_h: int):
        """Start interactive crop with fixed aspect ratio; user can move/resize then Apply Crop."""
        try:
            pm = getattr(self, "_img_item", None)
            if not isinstance(pm, QGraphicsPixmapItem):
                QMessageBox.information(self, "Meme", "Open an image first."); return
            img_rc = pm.sceneBoundingRect()
            target = float(aspect_w) / float(aspect_h)
            self._crop_aspect = target
            # initial rect: centered box covering ~80% of the shortest side while keeping aspect
            w, h = img_rc.width(), img_rc.height()
            if w / h >= target:
                # image wider than target
                box_h = h * 0.8
                box_w = box_h * target
            else:
                box_w = w * 0.8
                box_h = box_w / target
            x = img_rc.center().x() - box_w / 2.0
            y = img_rc.center().y() - box_h / 2.0
            rc = QRectF(x, y, box_w, box_h)
            # clear existing overlay
            self._clear_crop_overlay()
            # disable rubberband while cropping
            try:
                self.view.setDragMode(QGraphicsView.NoDrag)
            except Exception:
                pass
            self._crop_overlay = CropRectItem(rc, img_rc, aspect=target)
            self._crop_overlay.apply_cb = self._apply_crop
            self._crop_overlay.cancel_cb = self._clear_crop_overlay
            try:
                self._crop_overlay.handle_size = float(self.settings.value('meme/crop_handle', 18))
                alpha = int(self.settings.value('meme/crop_opacity', 120))
                self._crop_overlay.set_overlay_opacity(alpha)
            except Exception:
                pass
            self.scene.addItem(self._crop_overlay)
            self._crop_overlay.setZValue(9999)
        except Exception:
            pass

    def _on_crop_ui_changed(self, *a):
        try:
            self.settings.setValue('meme/crop_handle', int(self.sl_handle.value()))
            self.settings.setValue('meme/crop_opacity', int(self.sl_opacity.value()))
            if getattr(self, '_crop_overlay', None) is not None:
                self._crop_overlay.handle_size = float(self.sl_handle.value())
                self._crop_overlay.set_overlay_opacity(int(self.sl_opacity.value()))
                self._crop_overlay.update()
        except Exception:
            pass

    def _apply_crop(self):
        """Apply the currently selected overlay crop rectangle to the image and shift text items."""
        if self._crop_overlay is None:
            QMessageBox.information(self, "Meme", "No crop box. Click a crop preset first."); return
        try:
            pm_item = self._img_item
            if not isinstance(pm_item, QGraphicsPixmapItem):
                return
            base = pm_item.pixmap()
            if base.isNull():
                return
            rc = self._crop_overlay.rect().toRect()
            rc = rc.intersected(base.rect())
            if rc.width() < 2 or rc.height() < 2:
                QMessageBox.information(self,"Meme","Crop box too small."); return
            cropped = base.copy(rc)
            # Shift text items so they stay in place relative to the new image
            dx, dy = -rc.left(), -rc.top()
            for it in list(getattr(self, "_items", [])):
                try:
                    r = it.rect(); r.translate(dx, dy); it.setRect(r)
                except Exception:
                    pass
            self._img_item.setPixmap(cropped)
            self.scene.setSceneRect(cropped.rect())
            try:
                self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            self._clear_crop_overlay()

    def _center_crop(self, aspect_w: int, aspect_h: int):
        try:
            pm = getattr(self, "_orig_pm", None)
            if pm is None or pm.isNull():
                return
            w, h = pm.width(), pm.height()
            target = aspect_w / float(aspect_h)
            cur = w / float(h) if h else 1.0
            if cur > target:
                new_w, new_h = int(h * target), h; x = (w - new_w)//2; y = 0
            else:
                new_w, new_h = w, int(w / target); x = 0; y = (h - new_h)//2
            cropped = pm.copy(x, y, new_w, new_h)
            self._img_item.setPixmap(cropped)
            self.scene.setSceneRect(cropped.rect())
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        except Exception:
            pass

    def _make_item(self, text: str, rect: QRectF, style: TextStyle) -> MemeTextItem:
        it = MemeTextItem(text, rect, style)
        try:
            it.setZValue(10 + len(self._items))
        except Exception:
            pass
        self.scene.addItem(it)
        return it
        return it

    def _current_style_defaults(self) -> TextStyle:
        return TextStyle(
            family=self.def_family, size=self.def_size, color_hex=self.def_fill.name(),
            stroke_hex=self.def_stroke.name(), stroke_width=self.def_stroke_w,
            shadow=self.def_shadow, all_caps=self.def_allcaps, align=self.def_align, auto_fit=self.def_auto, letter_spacing=float(self.def_track), glow=int(self.def_glow)
        )

    # Actions
    def _use_current(self):
        path = self._current_path()
        if not path:
            QMessageBox.information(self, "Meme", "No current image available."); return
        low = str(path).lower()
        is_video = low.endswith((".mp4",".mkv",".mov",".webm",".avi",".m4v",".mpg",".mpeg",".ts",".m2ts",".wmv"))
        if is_video:
            try:
                vid = getattr(self.main, "video", None)
                from PySide6.QtGui import QPixmap
                tmp_path = None
                qimg = getattr(vid, "currentFrame", None) if vid is not None else None
                if qimg is not None and (not hasattr(qimg, "isNull") or not qimg.isNull()):
                    pm = QPixmap.fromImage(qimg)
                    if pm is not None and (not hasattr(pm, "isNull") or not pm.isNull()):
                        import tempfile
                        f = tempfile.NamedTemporaryFile(prefix="meme_frame_", suffix=".png", delete=False)
                        tmp_path = f.name; f.close(); pm.save(tmp_path, "PNG")
                if tmp_path is None and vid is not None and hasattr(vid, "label") and hasattr(vid.label, "pixmap"):
                    pm = vid.label.pixmap()
                    if pm is not None and (not hasattr(pm, "isNull") or not pm.isNull()):
                        import tempfile
                        f = tempfile.NamedTemporaryFile(prefix="meme_frame_", suffix=".png", delete=False)
                        tmp_path = f.name; f.close(); pm.save(tmp_path, "PNG")
                if tmp_path is None:
                    QMessageBox.warning(self, "Meme", "Could not open image."); return
                self._load_image_to_scene(tmp_path)
                try: self._tmp_last_path = tmp_path
                except Exception: pass
                return
            except Exception:
                QMessageBox.warning(self, "Meme", "Could not open image."); return
        self._load_image_to_scene(path)
    def _open_image(self):
        exts = "Images (*.png *.jpg *.jpeg *.webp)"
        d = self._last_dir or ""
        files, _ = QFileDialog.getOpenFileNames(self,"Open image(s)", d, exts)
        if not files: return
        self._last_dir = os.path.dirname(files[0])
        self._load_image_to_scene(files[0])
        self._pending_batch = files[1:]
        self._save_defaults()

    def _add_text(self):
        if not self._img_item:
            QMessageBox.information(self,"Meme","Open an image first."); return
        R = self.scene.sceneRect(); pad=12
        rect = QRectF(R.left()+pad, R.center().y()-R.height()*0.08, R.width()-2*pad, R.height()*0.16)
        it = self._make_item("NEW TEXT", rect, self._current_style_defaults()); self._items.append(it); it.setSelected(True)

    def _edit_selected_text(self):
        sel = [it for it in self._items if it.isSelected()]
        if not sel:
            QMessageBox.information(self,"Meme","Select a text box first."); return
        it = sel[0]
        txt, ok = QInputDialog.getMultiLineText(self,"Edit text","Text:", it.text())
        if not ok: return
        it.setText(txt)

    def _remove_selected(self):
        for it in list(self._items):
            if it.isSelected():
                self.scene.removeItem(it); self._items.remove(it)

    def _reset_positions(self):
        if not self._img_item: return
        R = self.scene.sceneRect(); pad=12
        if len(self._items)>=1: self._items[0].setRect(QRectF(R.left()+pad, R.top()+pad, R.width()-2*pad, R.height()*0.25))
        if len(self._items)>=2: self._items[-1].setRect(QRectF(R.left()+pad, R.bottom()-R.height()*0.25-pad, R.width()-2*pad, R.height()*0.25))

    def _apply_to_selected(self, *a):
        st = self._current_style_defaults()
        st.size = self.sp_size.value()
        st.stroke_width = self.sp_stroke.value()
        st.auto_fit = self.chk_auto.isChecked()
        st.shadow = self.chk_shadow.isChecked()
        st.all_caps = self.chk_allcaps.isChecked()
        st.align = self.cmb_align.currentText()
        st.color_hex = self.def_fill.name()
        st.stroke_hex = self.def_stroke.name()
        for it in self._items:
            if it.isSelected():
                it.setStyle(st); it.setAlign(st.align)

    def _pick_color(self, fill: bool):
        col = QColorDialog.getColor(self.def_fill if fill else self.def_stroke, self, "Pick color")
        if not col.isValid(): return
        if fill: self.def_fill = col
        else: self.def_stroke = col
        self._apply_to_selected(); self._save_defaults()

    def _nudge_selected(self, dx:int, dy:int):
        for it in self._items:
            if it.isSelected(): it.moveBy(dx,dy)

    # Fonts
    def _on_font_changed(self, qfont: QFont):
        self.def_family = qfont.family()
        self._apply_to_selected(); self._save_defaults()

    def _on_add_fonts(self):
        files, _ = QFileDialog.getOpenFileNames(self,"Add fonts…", self._last_dir or "", "Fonts (*.ttf *.otf *.ttc *.otc)")
        if not files: return
        ok=0; fams=[]
        for f in files:
            try:
                fid = QFontDatabase.addApplicationFont(f)
                fams += QFontDatabase.applicationFontFamilies(fid) or []
                ok += 1
            except Exception:
                pass
        if fams:
            self._persist_extra_fonts(files)
            try: self.font_combo.setCurrentFont(QFont(fams[0]))
            except Exception: pass
            self.def_family = fams[0]; self._save_defaults()
        QMessageBox.information(self,"Fonts", f"Loaded {ok} font file(s).")

    def _on_add_font_folder(self):
        d = QFileDialog.getExistingDirectory(self,"Load font folder…", self._last_dir or "")
        if not d: return
        exts={'.ttf','.otf','.ttc','.otc'}
        fps=[os.path.join(d,n) for n in os.listdir(d) if os.path.splitext(n)[1].lower() in exts]
        if not fps:
            QMessageBox.information(self,"Fonts","No font files found in that folder."); return
        ok=0; fams=[]
        for f in fps:
            try:
                fid = QFontDatabase.addApplicationFont(f)
                fams += QFontDatabase.applicationFontFamilies(fid) or []
                ok += 1
            except Exception:
                pass
        if fams:
            self._persist_extra_fonts(fps)
            try: self.font_combo.setCurrentFont(QFont(fams[0]))
            except Exception: pass
            self.def_family = fams[0]; self._save_defaults()
        QMessageBox.information(self,"Fonts", f"Loaded {ok} font file(s).")

    def _on_font_dialog(self):
        res = QFontDialog.getFont(QFont(self.def_family, self.def_size), self, "Pick font")
        if isinstance(res, tuple):
            f, ok = res
        else:
            # PySide6 should return a tuple, but be defensive
            f, ok = res, True
        if not ok:
            return
        self.def_family = f.family(); self.def_size = max(8, min(400, f.pointSize()))
        try: self.font_combo.setCurrentFont(f)
        except Exception: pass
        self.sp_size.setValue(self.def_size)
        self._apply_to_selected(); self._save_defaults()

    # Big preview
    def _open_big_preview(self):
        dlg = QDialog(self); dlg.setWindowTitle("Meme Preview"); dlg.resize(1100, 800)
        lay = QVBoxLayout(dlg)
        view = MemeGraphicsView(); view.setScene(self.scene)  # live sync
        lay.addWidget(view)
        btns = QHBoxLayout(); _use = QPushButton("Use current frame"); _close = QPushButton("Close")
        btns.addWidget(_use); btns.addStretch(1); btns.addWidget(_close); lay.addLayout(btns)
        _use.clicked.connect(lambda: self._try_grab_player_into_scene())
        _close.clicked.connect(dlg.accept)
        try: dlg.setModal(False)
        except Exception: pass
        dlg.show()
        self._big_preview_ref = dlg

    def _try_grab_player_into_scene(self):
        if not self.main: return False
        candidates = ["video_view","video_label","preview_label","player_view","player_widget","viewer","main_video","videoArea","image_view"]
        for name in candidates:
            w = getattr(self.main, name, None)
            if w is not None:
                try:
                    pm = w.grab()
                    if not pm.isNull():
                        if getattr(self, "_img_item", None):
                            try: self.scene.removeItem(self._img_item)
                            except Exception: pass
                        self._orig_pm = pm
                        self._img_item = QGraphicsPixmapItem(pm)
                        self.scene.addItem(self._img_item)
                        self.scene._img_item = self._img_item
                        self.scene.setSceneRect(pm.rect())
                        return True
                except Exception:
                    pass
        return False
    def _reset_crop(self):
        self._clear_crop_overlay()
        try:
            pm = getattr(self, "_orig_pm", None)
            if pm is None or pm.isNull():
                QMessageBox.information(self,"Meme","Nothing to reset."); return
            if getattr(self, "_img_item", None):
                try: self.scene.removeItem(self._img_item)
                except Exception: pass
            self._img_item = QGraphicsPixmapItem(pm)
            self.scene.addItem(self._img_item)
            self.scene._img_item = self._img_item
            self.scene.setSceneRect(pm.rect())
            try: self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
            except Exception: pass
        except Exception:
            pass

    # Export
    def _export_single(self):
        if not self._img_item:
            QMessageBox.information(self,"Meme","Open an image first."); return
        fmt = self.cmb_fmt.currentText().lower()
        d = self._last_dir or ""
        path, _ = QFileDialog.getSaveFileName(self,"Save as…", d, f"{fmt.upper()} (*.{fmt});;PNG (*.png);;JPG (*.jpg);;WebP (*.webp)")
        if not path: return
        self._last_dir = os.path.dirname(path)
        self._render_scene_to_path(path, fmt); self._save_defaults()

    def _export_batch(self):
        if not self._img_item:
            QMessageBox.information(self,"Meme","Open an image first."); return
        d = self._last_dir or ""
        files, _ = QFileDialog.getOpenFileNames(self,"Batch images…", d, "Images (*.png *.jpg *.jpeg *.webp)")
        if not files: return
        base = self._img_item.pixmap()
        base_w, base_h = base.width(), base.height()
        layout = []
        for it in self._items:
            r = it.rect()
            style = it._style  # snapshot
            layout.append((it.text(), r.x()/base_w, r.y()/base_h, r.width()/base_w, r.height()/base_h, style))
        ok = 0
        prog = QProgressDialog("Exporting…","Cancel",0,len(files), self); prog.setWindowModality(Qt.ApplicationModal); prog.setMinimumDuration(0)
        for i,f in enumerate(files):
            prog.setValue(i); prog.setLabelText(os.path.basename(f))
            if prog.wasCanceled(): break
            img = QImage(f)
            if img.isNull(): continue
            fmt = self.cmb_fmt.currentText().lower()
            out = os.path.join(os.path.dirname(f), os.path.splitext(os.path.basename(f))[0] + "_meme." + fmt)
            self._render_layout_to_qimage(img, layout, out, fmt); ok += 1
        prog.setValue(len(files))
        QMessageBox.information(self,"Batch", f"Exported {ok} image(s).")

    def _render_layout_to_qimage(self, img: QImage, layout, out_path: str, fmt: str):
        W,H = img.width(), img.height()
        canvas = QImage(img)  # keep alpha
        p = QPainter(canvas); p.setRenderHint(QPainter.Antialiasing, True)
        for text, nx, ny, nw, nh, style in layout:
            rect = QRectF(nx*W, ny*H, nw*W, nh*H)
            it = MemeTextItem(text, rect, style)
            it.paint(p, None)
        p.end()
        self._write_qimage(canvas, out_path, fmt)

    def _render_scene_to_path(self, out_path: str, fmt: str):
        R = self.scene.sceneRect().toRect()
        if R.width()<=0 or R.height()<=0:
            QMessageBox.warning(self,"Meme","Nothing to export."); return
        target = QImage(R.size(), QImage.Format_ARGB32_Premultiplied); target.fill(QColor(0,0,0,0))
        p = QPainter(target); p.setRenderHint(QPainter.Antialiasing, True)
        self.scene.render(p); p.end()
        self._write_qimage(target, out_path, fmt)

    def _write_qimage(self, qimg: QImage, out_path: str, fmt: str):
        fmt = (fmt or "png").lower()
        q = int(self.sp_quality.value())
        if PIL_OK:
            try:
                buf = io.BytesIO(); qimg.save(buf, "PNG"); buf.seek(0)
                im = Image.open(buf)
                params = {}
                if fmt == "jpg":
                    if im.mode in ("RGBA","LA"):
                        bg = Image.new("RGBA", im.size, (0,0,0,255))
                        im = Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB")
                    else:
                        im = im.convert("RGB")
                    params.update(dict(quality=q, optimize=True))
                elif fmt == "webp":
                    params.update(dict(quality=q, method=4))
                elif fmt == "png":
                    params.update(dict(compress_level=max(0,min(9,int(round((100-q)/100.0*9))))))
                if self.chk_keepmeta.isChecked() and hasattr(im, "info") and "exif" in im.info:
                    params["exif"] = im.info["exif"]
                im.save(out_path, format=fmt.upper(), **params)
                return
            except Exception:
                pass
        if fmt == "jpg":
            qimg = qimg.convertToFormat(QImage.Format_RGB888)
        qimg.save(out_path, fmt.upper(), q if fmt in ("jpg","webp") else -1)

# === FrameVision: Creator feature extension (non-invasive monkeypatch) ===
try:
    _MP = MemeToolPane
    _MTI = MemeTextItem
except NameError:
    _MP = None
    _MTI = None

if _MTI:
    # Curvature-aware build: wrap original _build_path to support arc text.
    try:
        _orig_build_path = _MTI._build_path
    except Exception:
        _orig_build_path = None

    def _fv_build_path(self):
        # Call original builder first
        if _orig_build_path:
            _orig_build_path(self)
        curv = float(getattr(self, "_fv_curvature", 0.0) or 0.0)
        if abs(curv) < 0.1:
            return  # straight text already set by original
        try:
            r = self._rect
            if r.width() <= 2 or r.height() <= 2:
                return
            txt = (self._text or "")
            if self._style.all_caps: txt = txt.upper()
            f = self._resolve_font()
            metrics = QFontMetricsF(f)
            line_h = max(metrics.height() * 1.15, float(self._style.size))

            # clear and rebuild
            out = QPainterPath()
            lines = (txt.splitlines() or [""])
            y = 0.0
            for ln in lines:
                # glyph metrics
                br_test = QPainterPath(); br_test.addText(0, 0, f, ln or " ")
                br = br_test.boundingRect()

                # horizontal placement (approximate original)
                if self._style.align == "left":
                    xoff = -br.left()
                elif self._style.align == "right":
                    xoff = r.width() - br.right()
                else:
                    xoff = (r.width() - br.width()) / 2.0 - br.left()

                # curvature mapping
                # map slider [-100..100] to arc sweep (degrees) of ~[10..170]
                sweep_deg = max(10.0, min(170.0, abs(curv) * 1.6 + 10.0))
                sweep_rad = math.radians(sweep_deg)
                width = max(1.0, br.width())
                radius = width / sweep_rad

                # center along baseline
                center_x = r.left() + xoff + br.center().x()
                base_y = r.top() + y + metrics.ascent()
                if curv > 0:
                    center_y = base_y + radius  # arc bulges upward
                    sign = -1.0
                else:
                    center_y = base_y - radius  # arc bulges downward
                    sign = 1.0

                # lay out each character
                acc = 0.0
                for ch in ln:
                    w = max(1.0, metrics.horizontalAdvance(ch))
                    acc_mid = acc + w * 0.5
                    x_rel = (acc_mid - br.center().x())
                    ang = x_rel / radius  # radians
                    sin_a = math.sin(ang); cos_a = math.cos(ang)
                    x = center_x + sin_a * radius
                    ypix = center_y + sign * (cos_a * radius - radius)

                    # Build glyph path at origin, then transform
                    gp = QPainterPath(); gp.addText(0, 0, f, ch)
                    tr = QTransform()
                    tr.translate(x, ypix)
                    tr.rotateRadians(-sign * ang)
                    tr.translate(-w * 0.5, 0)
                    out.addPath(tr.map(gp))
                    acc += w

                y += line_h

            self._path = out
            self.setPath(self._path)
        except Exception:
            # In case of failure, keep original straight layout
            pass

    try:
        _MTI._build_path = _fv_build_path
    except Exception:
        pass

if _MP:
    from PySide6.QtCore import QObject, QEvent, QPoint, QRect
    from PySide6.QtWidgets import QFormLayout, QSlider, QLabel, QPushButton, QHBoxLayout

    # ---------- Functional helpers (existing) ----------
    def _fv_current_path(self):
        try:
            m = getattr(self, "main", None)
            path = getattr(m, "current_path", None) or getattr(m, "current_image", None)
            import os as _os
            if path and _os.path.exists(str(path)):
                return str(path)
        except Exception:
            pass
        return None

    def _fv_update_use_current_enabled(self):
        try:
            self.btn_use_current.setEnabled(True)
        except Exception:
            pass

    def _fv_apply_transform_to_selection(self, *a):
        try:
            rot = int(self.sl_rot.value())
            shear = float(self.sl_shear.value()) / 100.0
            for it in getattr(self, "_items", []):
                try:
                    if it.isSelected():
                        it.apply_transform(rot, shear)
                except Exception:
                    pass
        except Exception:
            pass

    def _fv_apply_style_depth_to_selection(self, *a):
        try:
            depth = int(self.sl_depth.value())
            for it in getattr(self, "_items", []):
                try:
                    if it.isSelected():
                        st = getattr(it, "_style", None)
                        if st is None:
                            try:
                                st = self._current_style_defaults()
                            except Exception:
                                st = None
                        if st is not None:
                            try:
                                st.extrude = max(0, depth)
                            except Exception:
                                pass
                            try:
                                it.setStyle(st); it.update()
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass

    # Apply helpers
    try: _MP._current_path = _fv_current_path
    except Exception: pass
    try: _MP._update_use_current_enabled = _fv_update_use_current_enabled
    except Exception: pass
    try:
        _MP._apply_transform_to_selection = _fv_apply_transform_to_selection
        _MP._apply_style_depth_to_selection = _fv_apply_style_depth_to_selection
    except Exception: pass

    # ---------- Curvature slider UI ----------
    def _fv_install_curvature_ui(self):
        if getattr(self, "_fv_curve_ui", False): return
        try:
            forms = self.findChildren(QFormLayout)
            target_form = forms[-1] if forms else None
            if target_form is None:
                return
            self.sl_curve = QSlider(Qt.Horizontal, self); self.sl_curve.setMinimum(-100); self.sl_curve.setMaximum(100); self.sl_curve.setValue(0)
            target_form.addRow("Arc curvature", self.sl_curve)
            def _on_curve(val):
                try:
                    for it in getattr(self, "_items", []):
                        if it.isSelected():
                            setattr(it, "_fv_curvature", float(val))
                            it._build_path(); it.update()
                except Exception:
                    pass
            self.sl_curve.valueChanged.connect(_on_curve)
            self._fv_curve_ui = True
        except Exception:
            pass

    # ---------- Interactive crop tools ----------
        def __init__(self, pane):
            super().__init__(pane)
            self.pane = pane
            self.dragging = False
            self.start = QPoint()
            self.rect_item = None

        def eventFilter(self, obj, ev):
            t = ev.type()
            if t == QEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                self.dragging = True
                self.start = ev.pos()
                v = self.pane.view
                scene_p = v.mapToScene(self.start)
                self._ensure_rect_item(scene_p)
                return True
            if t == QEvent.MouseMove and self.dragging:
                v = self.pane.view
                scene_start = v.mapToScene(self.start)
                scene_curr = v.mapToScene(ev.pos())
                rect = QRectF(scene_start, scene_curr).normalized()
                self.rect_item.setRect(rect)
                return True
            if t == QEvent.MouseButtonRelease and self.dragging:
                self.dragging = False
                return True
            return False

        def _ensure_rect_item(self, scene_pos):
            if self.rect_item is None:
                from PySide6.QtGui import QPen, QColor
                from PySide6.QtCore import QRectF
                self.rect_item = self.pane.scene.addRect(QRectF(scene_pos, scene_pos), QPen(QColor(255,0,0), 2, Qt.DashLine))
                self.rect_item.setZValue(9999)

        def rect(self):
            if self.rect_item:
                return self.rect_item.rect()
            return None

        def clear(self):
            if self.rect_item:
                self.pane.scene.removeItem(self.rect_item)
                self.rect_item = None

    def _fv_add_crop_ui(self):
        # No-op: Free/Apply crop buttons removed
        try:
            self._fv_crop_ui = True
        except Exception:
            pass
    def set_overlay_opacity(self, alpha:int):
        try:
            if not getattr(self, '_shade', None):
                return
            b = self._shade.brush().color(); b.setAlpha(max(0,min(255,int(alpha))))
            brush = self._shade.brush(); brush.setColor(b); self._shade.setBrush(brush)
        except Exception:
            pass

    def _maybe_autoscroll(self, ev):
        try:
            sc = self.scene(); views = sc.views() if sc else []
            if not views: return
            v = views[0]
            scene_p = self.mapToScene(ev.pos())
            view_p = v.mapFromScene(scene_p)
            vr = v.viewport().rect(); margin = 16
            dx=dy=0
            if view_p.x() < vr.left()+margin: dx = -20
            elif view_p.x() > vr.right()-margin: dx = 20
            if view_p.y() < vr.top()+margin: dy = -20
            elif view_p.y() > vr.bottom()-margin: dy = 20
            if dx:
                sb = v.horizontalScrollBar(); sb.setValue(sb.value()+dx)
            if dy:
                sb = v.verticalScrollBar(); sb.setValue(sb.value()+dy)
        except Exception:
            pass
