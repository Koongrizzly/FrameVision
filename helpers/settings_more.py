
from __future__ import annotations
from typing import Optional, List, Dict, Union, Sequence
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QSettings, QUrl

import os, sys, time, json, random

# ==========================================================================================
# Easter Egg system — standalone module
# ==========================================================================================

# ---------------------------- Usage tracking (persisted) ----------------------------------
_USAGE_SETTINGS_SCOPE = ("FrameVision", "FrameVision")
_USAGE_TOTAL_KEY = "usage_total_seconds"
_USAGE_TRACKER_APP_PROP = "_fv_usage_tracker_singleton"
_SECRET_LISTENER_PROP = "_fv_secret_editorpy_listener"


def _usage_settings() -> QSettings:
    return QSettings(*_USAGE_SETTINGS_SCOPE)


def _get_usage_total_seconds() -> int:
    s = _usage_settings()
    try:
        val = s.value(_USAGE_TOTAL_KEY, 0, type=int)
    except Exception:
        try:
            val = int(s.value(_USAGE_TOTAL_KEY, 0))
        except Exception:
            val = 0
    return int(val or 0)


def _set_usage_total_seconds(v: int) -> None:
    _usage_settings().setValue(_USAGE_TOTAL_KEY, int(max(0, v)))


def _egg_unlocked(egg_id: str) -> bool:
    s = _usage_settings()
    return bool(s.value(f"egg_{egg_id}_unlocked", False, type=bool))


def _mark_egg_unlocked(egg_id: str) -> None:
    s = _usage_settings()
    s.setValue(f"egg_{egg_id}_unlocked", True)
    s.setValue(f"egg_{egg_id}_unlocked_at", int(time.time()))


def _unlock_popup_once(egg_id: str, message: str) -> None:
    s = _usage_settings()
    if s.value(f"egg_{egg_id}_popup_shown", False, type=bool):
        return
    s.setValue(f"egg_{egg_id}_popup_shown", True)
    try:
        app = QtWidgets.QApplication.instance()
        parent = app.activeWindow() if app else None
        m = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information,
                                  "Easter egg unlocked",
                                  message,
                                  QtWidgets.QMessageBox.Ok,
                                  parent)
        if app is not None:
            # prevent GC
            if not hasattr(app, "_fv_unlock_popups"):
                app._fv_unlock_popups = []  # type: ignore[attr-defined]
            app._fv_unlock_popups.append(m)  # type: ignore[attr-defined]
            def _cleanup(_res=None, _m=m):
                try:
                    app._fv_unlock_popups.remove(_m)  # type: ignore[attr-defined]
                except Exception:
                    pass
            m.finished.connect(_cleanup)
        m.show()
    except Exception:
        pass


class _UsageTracker(QtCore.QObject):
    """Lightweight singleton that increments a persisted counter while the app runs."""
    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._last = time.time()
        self._tm = QtCore.QTimer(self)
        self._tm.setInterval(10_000)  # 10s cadence
        self._tm.timeout.connect(self._tick)
        self._tm.start()

    def _tick(self) -> None:
        now = time.time()
        delta = max(0, int(now - self._last))
        self._last = now
        if not delta:
            return
        total = _get_usage_total_seconds() + delta
        _set_usage_total_seconds(total)
        _check_for_unlocks(total)


def ensure_usage_tracker() -> None:
    """Public: ensure the usage tracker is running (idempotent)."""
    try:
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        if getattr(app, _USAGE_TRACKER_APP_PROP, None):
            return
        tracker = _UsageTracker(app)
        setattr(app, _USAGE_TRACKER_APP_PROP, tracker)
    except Exception:
        pass


# ----------------------------- Icons -------------------------------------------------------
def _make_tetris_icon(size: int = 16) -> QtGui.QIcon:
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        s = size // 2 - 1
        colors = [QtGui.QColor('#4CC9F0'), QtGui.QColor('#4361EE'),
                  QtGui.QColor('#F72585'), QtGui.QColor('#4CAF50')]
        idx = 0
        for r in range(2):
            for c in range(2):
                p.fillRect(c*s + c+1, r*s + r+1, s, s, colors[idx % len(colors)])
                idx += 1
                pen = QtGui.QPen(QtGui.QColor(20,20,20)); pen.setWidth(1); p.setPen(pen)
                p.drawRect(c*s + c+1, r*s + r+1, s, s)
    finally:
        p.end()
    return QtGui.QIcon(pm)


def _make_pong_icon(size: int = 16) -> QtGui.QIcon:
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        pen = QtGui.QPen(QtGui.QColor('#ffffff')); pen.setWidth(2); p.setPen(pen)
        p.drawLine(3, 4, 3, size-4)
        p.drawLine(size-3, 4, size-3, size-8)
        brush = QtGui.QBrush(QtGui.QColor('#ffffff')); p.setBrush(brush)
        p.drawEllipse(QtCore.QPoint(size//2, size//2), 2, 2)
    finally:
        p.end()
    return QtGui.QIcon(pm)


def _make_snake_icon(size: int = 16) -> QtGui.QIcon:
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        pen = QtGui.QPen(QtGui.QColor('#7CFC00')); pen.setWidth(2); p.setPen(pen)
        # simple zig-zag snake
        for i in range(3, size-3, 4):
            p.drawLine(i, size//2 - 3, i+2, size//2 + 3)
        # head
        brush = QtGui.QBrush(QtGui.QColor('#7CFC00')); p.setBrush(brush)
        p.drawEllipse(QtCore.QPoint(size-4, size//2), 2, 2)
    finally:
        p.end()
    return QtGui.QIcon(pm)


def _make_dad_face_icon(size: int = 18) -> QtGui.QIcon:
    """Simple 'dad-like' smiley with moustache and glasses."""
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # face
        face = QtGui.QColor('#FFD28A')
        pen = QtGui.QPen(QtGui.QColor('#5a4a2c')); pen.setWidth(1); p.setPen(pen)
        p.setBrush(face)
        p.drawEllipse(1, 1, size-2, size-2)
        # glasses
        gpen = QtGui.QPen(QtGui.QColor('#2f2f2f')); gpen.setWidth(2); p.setPen(gpen)
        r = size // 4
        p.drawEllipse(QtCore.QPoint(size//3, size//3), r//2, r//2)
        p.drawEllipse(QtCore.QPoint(2*size//3, size//3), r//2, r//2)
        p.drawLine(size//3 + r//2, size//3, 2*size//3 - r//2, size//3)
        # moustache
        mp = QtGui.QPainterPath()
        mp.moveTo(size//2 - r//2, size//2 + 1)
        mp.cubicTo(size//2 - r, size//2 + r//3, size//2 - r//4, size//2 + r//2, size//2, size//2 + r//3)
        mp.cubicTo(size//2 + r//4, size//2 + r//2, size//2 + r, size//2 + r//3, size//2 + r//2, size//2 + 1)
        p.setBrush(QtGui.QColor('#3b2a20'))
        p.drawPath(mp)
        # smile
        spen = QtGui.QPen(QtGui.QColor('#7a4a2c')); spen.setWidth(2); p.setPen(spen)
        p.drawArc(size//3, size//2 + r//3, size//3, r//2, 0, -180*16)
    finally:
        p.end()
    return QtGui.QIcon(pm)


def _icon_thumbs_up(size: int = 14) -> QtGui.QIcon:
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setBrush(QtGui.QColor('#4CAF50')); p.setPen(Qt.NoPen)
        pts = [QtCore.QPoint(3,8), QtCore.QPoint(6,5), QtCore.QPoint(8,3),
               QtCore.QPoint(10,5), QtCore.QPoint(9,11), QtCore.QPoint(4,11)]
        poly = QtGui.QPolygon(pts)
        p.drawPolygon(poly)
    finally:
        p.end()
    return QtGui.QIcon(pm)


def _icon_thumbs_down(size: int = 14) -> QtGui.QIcon:
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        p.setBrush(QtGui.QColor('#E53935')); p.setPen(Qt.NoPen)
        pts = [QtCore.QPoint(3,6), QtCore.QPoint(6,9), QtCore.QPoint(8,11),
               QtCore.QPoint(10,9), QtCore.QPoint(9,3), QtCore.QPoint(4,3)]
        poly = QtGui.QPolygon(pts)
        p.drawPolygon(poly)
    finally:
        p.end()
    return QtGui.QIcon(pm)


def _make_breakout_icon(size: int = 18) -> QtGui.QIcon:
    """Breakout-style icon: rows of bricks, a ball, and a paddle."""
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # bricks (top row)
        brick_h = max(3, size//6)
        gap = 1
        colors = [QtGui.QColor('#F94144'), QtGui.QColor('#F3722C'), QtGui.QColor('#F8961E'),
                  QtGui.QColor('#90BE6D'), QtGui.QColor('#577590')]
        x = 1
        w = (size - 2 - 2*gap) // 3
        for i in range(3):
            p.fillRect(x, 1, w, brick_h, colors[i % len(colors)])
            pen = QtGui.QPen(QtGui.QColor(20,20,20)); pen.setWidth(1); p.setPen(pen)
            p.drawRect(x, 1, w, brick_h)
            x += w + gap
        # ball
        ball_r = max(2, size//8)
        p.setBrush(QtGui.QBrush(QtGui.QColor('#ffffff'))); p.setPen(Qt.NoPen)
        p.drawEllipse(QtCore.QPoint(size//2, size//2), ball_r, ball_r)
        # paddle
        pad_w = size//2
        pad_h = max(2, size//10)
        p.setBrush(QtGui.QBrush(QtGui.QColor('#00BCD4'))); p.setPen(Qt.NoPen)
        p.drawRoundedRect((size - pad_w)//2, size - pad_h - 2, pad_w, pad_h, 2, 2)
    finally:
        p.end()
    return QtGui.QIcon(pm)


def _make_racecar_icon(size: int = 18) -> QtGui.QIcon:
    """Small racecar icon for 'FrameRacing'."""
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # body
        p.setBrush(QtGui.QBrush(QtGui.QColor('#D32F2F'))); p.setPen(Qt.NoPen)
        body_rect = QtCore.QRect(3, size//3, size-6, size//3)
        p.drawRoundedRect(body_rect, 3, 3)
        # cockpit
        p.setBrush(QtGui.QBrush(QtGui.QColor('#90CAF9')))
        p.drawRoundedRect(QtCore.QRect(size//3, size//3 - 2, size//3, size//3 - 2), 2, 2)
        # spoiler
        p.setBrush(QtGui.QBrush(QtGui.QColor('#B71C1C')))
        p.drawRect(2, size//3 - 1, 3, size//3 + 2)
        # wheels
        p.setBrush(QtGui.QBrush(QtGui.QColor('#212121')))
        wheel_r = max(2, size//8)
        p.drawEllipse(QtCore.QPoint(5, size - 3), wheel_r, wheel_r)
        p.drawEllipse(QtCore.QPoint(size - 5, size - 3), wheel_r, wheel_r)
        # stripe
        p.setBrush(QtGui.QBrush(QtGui.QColor('#FFFFFF')))
        p.drawRect(size//2 - 1, size//3 + 1, 2, size//3 - 2)
    finally:
        p.end()
    return QtGui.QIcon(pm)


def _make_donkey_kong_icon(size: int = 18) -> QtGui.QIcon:
    """Tiny Donkey Kong-inspired icon: brown ape + barrel silhouette."""
    pm = QtGui.QPixmap(size, size); pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    try:
        p.setRenderHint(QtGui.QPainter.Antialiasing, True)
        # ape body
        p.setBrush(QtGui.QBrush(QtGui.QColor('#6D4C41'))); p.setPen(Qt.NoPen)
        p.drawEllipse(QtCore.QRect(size//6, size//4, size//3, size//3))   # head
        p.drawRoundedRect(QtCore.QRect(size//6, size//2, size//2, size//3), 3, 3)  # torso
        # arm
        p.drawRoundedRect(QtCore.QRect(size//2, size//2, size//3, size//6), 2, 2)
        # barrel
        p.setBrush(QtGui.QBrush(QtGui.QColor('#8D6E63')))
        barrel = QtCore.QRect(size//2 + 1, size//2 + 1, size//3, size//4)
        p.drawRoundedRect(barrel, 2, 2)
        pen = QtGui.QPen(QtGui.QColor('#4E342E')); pen.setWidth(1); p.setPen(pen)
        p.drawLine(barrel.left()+2, barrel.top()+3, barrel.right()-2, barrel.top()+3)
        p.drawLine(barrel.left()+2, barrel.center().y(), barrel.right()-2, barrel.center().y())
    finally:
        p.end()
    return QtGui.QIcon(pm)


# ---------------------------- Dad jokes logic ---------------------------------------------
_DAD_FILE_REL = os.path.join("assets", "dad_jokes.txt")
_DAD_BAG_KEY = "dad_jokes_bag"
_DAD_DISLIKED_KEY = "dad_jokes_disliked"

def _project_root() -> str:
    try:
        here = os.path.abspath(os.path.dirname(__file__))
        return os.path.abspath(os.path.join(here, ".."))
    except Exception:
        return os.getcwd()

def _read_dad_jokes() -> List[str]:
    root = _project_root()
    path = os.path.join(root, _DAD_FILE_REL)
    jokes: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                t = (line.strip()).strip("\\uFEFF")
                if t:
                    jokes.append(t)
    except Exception:
        jokes = [
            "I would tell you a joke about construction, but I'm still working on it.",
            "I used to hate facial hair… but then it grew on me.",
            "I’m reading a book on anti-gravity. It’s impossible to put down!",
            "How do you organize a space party? You planet.",
        ]
    return jokes

def _load_disliked() -> set:
    s = _usage_settings()
    try:
        raw = s.value(_DAD_DISLIKED_KEY, "[]", type=str)
        if isinstance(raw, list):
            return set(str(x) for x in raw)
        data = json.loads(raw or "[]")
        return set(str(x) for x in data)
    except Exception:
        return set()

def _save_disliked(disliked: set) -> None:
    s = _usage_settings()
    try:
        s.setValue(_DAD_DISLIKED_KEY, json.dumps(sorted(disliked)))
    except Exception:
        pass

def _load_bag() -> List[str]:
    s = _usage_settings()
    try:
        raw = s.value(_DAD_BAG_KEY, "[]", type=str)
        if isinstance(raw, list):
            bag = [str(x) for x in raw]
        else:
            bag = [str(x) for x in json.loads(raw or "[]")]
    except Exception:
        bag = []
    return bag

def _save_bag(bag: List[str]) -> None:
    s = _usage_settings()
    try:
        s.setValue(_DAD_BAG_KEY, json.dumps(list(bag)))
    except Exception:
        pass

def _refill_bag_if_needed() -> List[str]:
    jokes = _read_dad_jokes()
    disliked = _load_disliked()
    allowed = [j for j in jokes if j not in disliked]
    bag = _load_bag()
    bag = [j for j in bag if j in allowed]
    if not bag:
        rand = list(allowed)
        random.shuffle(rand)
        bag = rand
        _save_bag(bag)
    return bag

def _pick_next_joke() -> Optional[str]:
    bag = _refill_bag_if_needed()
    if not bag:
        return None
    j = bag.pop(0)
    _save_bag(bag)
    return j

def _dad_joke_popup(parent: Optional[QtWidgets.QWidget] = None) -> None:
    try:
        joke = _pick_next_joke()
        if not joke:
            QtWidgets.QMessageBox.information(parent, "Random Dad Joke",
                                              "No more jokes available. Add more in assets/dad_jokes.txt.")
            return

        dlg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.NoIcon, "Random Dad Joke", joke, parent=parent)
        nice_btn = dlg.addButton("nice !", QtWidgets.QMessageBox.AcceptRole)
        nah_btn = dlg.addButton("Nah...", QtWidgets.QMessageBox.DestructiveRole)
        nice_btn.setIcon(_icon_thumbs_up())
        nah_btn.setIcon(_icon_thumbs_down())
        dlg.setIconPixmap(_make_dad_face_icon(48).pixmap(48,48))
        dlg.exec()

        clicked = dlg.clickedButton()
        if clicked is nah_btn:
            disliked = _load_disliked()
            disliked.add(joke)
            _save_disliked(disliked)
            bag = _load_bag()
            bag = [x for x in bag if x != joke]
            _save_bag(bag)
        else:
            pass
    except Exception:
        pass


# ---------------------------- Helpers ------------------------------------------------------
def _project_root_for_open() -> str:
    return _project_root()

def _open_file_default_app(abs_path: str) -> None:
    try:
        if not os.path.isfile(abs_path):
            QtWidgets.QMessageBox.warning(None, "Open file", f"File not found:\\n{abs_path}")
            return
        QtGui.QDesktopServices.openUrl(QUrl.fromLocalFile(abs_path))
    except Exception:
        pass

def _spawn_helper_script(rel_path: Union[str, Sequence[str]]) -> None:
    """
    Launch a helper .py file in a detached process.
    Accepts a single relative path or a list/tuple of candidate relative paths; uses the first that exists.
    """
    try:
        root = _project_root()
        candidates: List[str] = []
        if isinstance(rel_path, (list, tuple)):
            candidates = list(rel_path)
        else:
            candidates = [str(rel_path)]
        for pth in candidates:
            norm = pth.replace("\\\\", os.sep).replace("/", os.sep)
            abs_path = os.path.join(root, norm)
            if os.path.isfile(abs_path):
                py = sys.executable or "python"
                QtCore.QProcess.startDetached(py, [abs_path])
                return
        # none found
        short_list = ", ".join(candidates)
        QtWidgets.QMessageBox.information(None, "Easter egg not found",
                                          f"Could not locate: {short_list}\\nLooked in /helpers/.")
    except Exception:
        pass


# ----------------------------- Secret Easter Egg (not in menu) -----------------------------
class _SecretKeyListener(QtCore.QObject):
    """
    Global key listener: when user types 'editor.py' then presses Enter,
    open <root>/helpers/editor.py in the default system editor.
    """
    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._buf = ""
        self._last_t = time.time()
        self._timeout = 2.5  # seconds to reset typing buffer

    def eventFilter(self, obj, event):
        try:
            if event.type() == QtCore.QEvent.KeyPress:
                now = time.time()
                if now - self._last_t > self._timeout:
                    self._buf = ""
                self._last_t = now

                key = event.key()
                txt = event.text() or ""
                # capture printable characters for the buffer
                if txt and (31 < ord(txt) < 127):  # basic ASCII
                    self._buf += txt
                    if len(self._buf) > 64:
                        self._buf = self._buf[-64:]
                # on Enter/Return, check the trigger
                if key in (Qt.Key_Return, Qt.Key_Enter):
                    if self._buf.strip().lower().endswith("editor.py"):
                        self._trigger_open()
                    # always reset on enter to avoid accidental repeats
                    self._buf = ""
            return False
        except Exception:
            return False

    def _trigger_open(self) -> None:
        try:
            root = _project_root_for_open()
            path = os.path.join(root, "helpers", "editor.py")
            _open_file_default_app(path)
        except Exception:
            pass

def install_secret_editor_egg() -> None:
    """Install the event filter once, app-wide."""
    try:
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        if getattr(app, _SECRET_LISTENER_PROP, None):
            return
        listener = _SecretKeyListener(app)
        app.installEventFilter(listener)
        setattr(app, _SECRET_LISTENER_PROP, listener)
    except Exception:
        pass


# ----------------------------- Easter Eggs registry ---------------------------------------
EASTER_EGGS: List[Dict] = [
    {
        "id": "tetris",
        "label": "Play Tetris",
        "icon_fn": lambda: _make_tetris_icon(18),
        "script": r"helpers\\tetris_game.py",
        "unlock_seconds": 60 * 60,  # 1 hour
        "message": "new easter egg unlocked, check settings tab",
    },
    {
        "id": "pong",
        "label": "Play Pong",
        "icon_fn": lambda: _make_pong_icon(18),
        "script": r"helpers\\colorful_pong.py",
        "unlock_seconds": 60 * 60 * 10,  # 10 hours
        "message": "way to go, you unlocked another easter egg, check settings tab",
    },
    {
        "id": "framie_snake",
        "label": "Play Framie Snake",
        "icon_fn": lambda: _make_snake_icon(18),
        "script": r"helpers\\Framie_snake.py",
        "unlock_seconds": 60 * 60 * 6,  # 6 hours
        "message": "new easter egg unlocked, check settings tab",
    },
    # Visible eggs
    {
        "id": "dad_joke",
        "label": "Random Dad Joke",
        "icon_fn": lambda: _make_dad_face_icon(18),
        "callback": _dad_joke_popup,
        "unlock_seconds": 60 * 5,  # 5 minutes
        "message": "Great, You just unlocked your first easter egg, check Settings tab !",
    },
    {
        "id": "framebreaker",
        "label": "FrameBreaker",
        "icon_fn": lambda: _make_breakout_icon(18),
        "script": r"helpers\\framebreaker.py",
        "unlock_seconds": 60 * 60 * 4,  # 4 hours
        "message": "Great, another easter egg unlocked, check Settings tab !",
    },
    # UPDATED: FrameRacing supports multiple filenames
    {
        "id": "frameracing",
        "label": "FrameRacing",
        "icon_fn": lambda: _make_racecar_icon(18),
        "scripts": [
            r"helpers\\frameracers.py",
            r"helpers\\frameracing.py",
            r"helpers\\FrameRacing.py",
            r"helpers\\FrameRacers.py",
        ],
        "unlock_seconds": 60 * 60 * 14,  # 14 hours
        "message": "Great, another easter egg unlocked, check Settings tab !",
    },
    # Donkey Kong Classic (36h)
    {
        "id": "dk_classic",
        "label": "Donkey Kong Classic",
        "icon_fn": lambda: _make_donkey_kong_icon(18),
        "scripts": [
            r"helpers\\DonkeyKongClassic.py",
            r"helpers\\donkey_kong_classic.py",
            r"helpers\\donkey_kong.py",
            r"helpers\\donkeykong.py",
            r"helpers\\dk_classic.py",
        ],
        "unlock_seconds": 60 * 60 * 30,  # 30 hours
        "message": "Great, another easter egg unlocked, check Settings tab !",
    },
]


def _check_for_unlocks(total_seconds: Optional[int] = None) -> None:
    if total_seconds is None:
        total_seconds = _get_usage_total_seconds()
    try:
        for egg in EASTER_EGGS:
            if _egg_unlocked(egg["id"]):
                continue
            if total_seconds >= int(egg.get("unlock_seconds", 0)):
                _mark_egg_unlocked(egg["id"])
                _unlock_popup_once(egg["id"], egg.get("message", "New easter egg unlocked!"))
    except Exception:
        pass


# schedule setup shortly after import
try:
    QtCore.QTimer.singleShot(1000, ensure_usage_tracker)
except Exception:
    pass

def _populate_easter_menu(menu: QtWidgets.QMenu, parent: QtWidgets.QWidget) -> None:
    try:
        menu.clear()
        header = QtWidgets.QWidget(menu)
        v = QtWidgets.QVBoxLayout(header); v.setContentsMargins(10,6,10,6); v.setSpacing(2)
        title = QtWidgets.QLabel("Easter Eggs & Fun stuff", header); title.setStyleSheet("font-weight:600;")
        sub = QtWidgets.QLabel("Use the app to unlock more", header); sub.setStyleSheet("opacity:0.7; font-size:11px;")
        v.addWidget(title); v.addWidget(sub)
        wa = QtWidgets.QWidgetAction(menu); wa.setDefaultWidget(header); menu.addAction(wa)
        menu.addSeparator()
        any_added = False
        for egg in EASTER_EGGS:
            if not _egg_unlocked(egg["id"]):
                continue
            any_added = True
            act = QtGui.QAction(egg["icon_fn"](), egg["label"], menu)
            cb = egg.get("callback")
            if callable(cb):
                act.triggered.connect(lambda _=False, _cb=cb: _cb(parent))
            else:
                target = egg.get("scripts") or egg.get("script")
                act.triggered.connect(lambda _=False, t=target: _spawn_helper_script(t))
            menu.addAction(act)
        if not any_added:
            stub = QtGui.QAction("Keep using FrameVision to unlock secrets…", menu)
            stub.setEnabled(False)
            menu.addAction(stub)
    except Exception:
        pass


def install_social_bottom_runtime() -> None:
    """Injects the Easter Eggs button at the bottom of the Settings content."""
    try:
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        root = None
        for w in app.allWidgets():
            try:
                if (w.objectName() or "") == "FvSettingsContent":
                    root = w; break
            except Exception:
                pass
        if not root or getattr(root, "_fv_social_bottom_installed", False):
            return
        v = root.layout()
        if not isinstance(v, QtWidgets.QVBoxLayout):
            return
        # Add a thin separator
        sep = QtWidgets.QFrame(root)
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep.setObjectName("FvSocialSeparator")
        v.addWidget(sep)
        # Row with buttons aligned right
        row = QtWidgets.QWidget(root)
        h = QtWidgets.QHBoxLayout(row); h.setContentsMargins(0,6,0,0); h.setSpacing(8)
        h.addStretch(1)

        btn_ee = QtWidgets.QPushButton("Easter Eggs", row)
        btn_ee.setMinimumWidth(120); btn_ee.setMinimumHeight(24)
        menu = QtWidgets.QMenu(btn_ee)
        _populate_easter_menu(menu, btn_ee)
        menu.aboutToShow.connect(lambda: _populate_easter_menu(menu, btn_ee))
        btn_ee.setMenu(menu)
        h.addWidget(btn_ee)

        # Optional hidden socials preserved (not shown)
        btn_gh = QtWidgets.QPushButton("GitHub", row); btn_gh.hide()
        btn_gh.setMinimumWidth(120); btn_gh.setMinimumHeight(24)
        btn_yt = QtWidgets.QPushButton("YouTube", row); btn_yt.hide()
        btn_yt.setMinimumWidth(120); btn_yt.setMinimumHeight(24)
        h.addWidget(btn_gh); h.addWidget(btn_yt)

        v.addWidget(row)
        root._fv_social_bottom_installed = True
    except Exception:
        pass

try:
    QtCore.QTimer.singleShot(800, install_social_bottom_runtime)
except Exception:
    pass

# Secret egg install
try:
    QtCore.QTimer.singleShot(600, lambda: install_secret_editor_egg())
except Exception:
    pass
