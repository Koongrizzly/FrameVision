
import os, sys, io, json, time, traceback, datetime, platform, pathlib, atexit, threading
from contextlib import suppress

LOG_DIR = None
LOG_PATH = None
LOG_FH = None
_HEARTBEAT = True

def _now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def _ensure_logs():
    global LOG_DIR, LOG_PATH, LOG_FH
    if LOG_FH:
        return LOG_DIR, LOG_PATH
    base = pathlib.Path.cwd() / "logs"
    base.mkdir(parents=True, exist_ok=True)
    LOG_DIR = str(base)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = str(base / f"diag_{ts}.log")
    LOG_FH = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
    return LOG_DIR, LOG_PATH

def _writeln(line):
    _ensure_logs()
    try:
        LOG_FH.write(line + "\n")
        LOG_FH.flush()
    except Exception:
        pass

def _json_safe(val):
    try:
        import base64
        if isinstance(val, (bytes, bytearray)):
            return {"__bytes_b64__": base64.b64encode(bytes(val)).decode("ascii")}
        tname = type(val).__name__
        if tname == "QByteArray":
            try:
                b = bytes(val)
            except Exception:
                b = str(val).encode("utf-8", "ignore")
            return {"__qbytearray_b64__": base64.b64encode(b).decode("ascii")}
        if isinstance(val, (int, float, str, bool)) or val is None:
            return val
        if isinstance(val, (list, tuple)):
            return [_json_safe(x) for x in val]
        if isinstance(val, dict):
            return {str(k): _json_safe(v) for k, v in val.items()}
        return str(val)
    except Exception:
        return str(val)

def _dump_header():
    import getpass
    _writeln("="*80)
    _writeln(f"[{_now()}] FrameVision diagnostics started")
    _writeln(f"User: {getpass.getuser()} | PID: {os.getpid()} | Python: {platform.python_version()} | {platform.platform()}")
    _writeln(f"cwd: {os.getcwd()}")
    with suppress(Exception):
        import PySide6
        _writeln(f"PySide6: {PySide6.__version__}")
    # Dump QSettings snapshot (safe)
    try:
        from PySide6.QtCore import QSettings
        s = QSettings('FrameVision','FrameVision')
        keys = s.allKeys()
        snap = {k: _json_safe(s.value(k)) for k in keys}
        _writeln("QSettings snapshot: " + json.dumps(snap, ensure_ascii=False))
    except Exception as e:
        _writeln(f"QSettings snapshot failed: {e!r}")
    _writeln("="*80)

def _install_excepthooks():
    old_hook = sys.excepthook
    def hook(exc_type, exc, tb):
        _writeln(f"[{_now()}] UNHANDLED EXCEPTION: {exc_type.__name__}: {exc}")
        _writeln("".join(traceback.format_exception(exc_type, exc, tb)))
        if old_hook:
            with suppress(Exception):
                old_hook(exc_type, exc, tb)
    sys.excepthook = hook

    # Threading excepthook (3.8+)
    def thook(args):
        _writeln(f"[{_now()}] THREAD EXCEPTION in {getattr(args, 'thread', None)}: {args.exc_type.__name__}: {args.exc_value}")
        _writeln("".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)))
    with suppress(Exception):
        threading.excepthook = thook  # type: ignore

def _install_faulthandler():
    try:
        import faulthandler
        ld, lp = _ensure_logs()
        crash_path = os.path.join(ld, "crash_{}.log".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        fh = open(crash_path, "w", encoding="utf-8")
        faulthandler.enable(file=fh, all_threads=True)
        _writeln(f"[{_now()}] faulthandler enabled -> {crash_path}")
    except Exception as e:
        _writeln(f"[{_now()}] faulthandler enable failed: {e!r}")

def _install_qt_message_handler():
    try:
        from PySide6.QtCore import qInstallMessageHandler, QtMsgType
        def handler(mode, ctx, msg):
            try:
                kind = {QtMsgType.QtDebugMsg:'D', QtMsgType.QtInfoMsg:'I', QtMsgType.QtWarningMsg:'W', QtMsgType.QtCriticalMsg:'E', QtMsgType.QtFatalMsg:'F'}.get(mode, str(mode))
            except Exception:
                kind = str(mode)
            loc = ""
            try:
                if ctx and getattr(ctx, 'file', None):
                    loc = f" ({ctx.file}:{ctx.line})"
            except Exception:
                pass
            _writeln(f"[{_now()}] [Qt{kind}]{loc} {msg}")
        qInstallMessageHandler(handler)
        _writeln(f"[{_now()}] Qt message handler installed")
    except Exception as e:
        _writeln(f"[{_now()}] Qt message handler failed: {e!r}")

def _install_app_hooks():
    try:
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance()
        if app is not None:
            try:
                app.aboutToQuit.connect(lambda: _writeln(f"[{_now()}] Qt aboutToQuit received"))
            except Exception as e:
                _writeln(f"[{_now()}] aboutToQuit hook failed: {e!r}")
    except Exception as e:
        _writeln(f"[{_now()}] app hook failure: {e!r}")

def _start_heartbeat():
    def beat():
        i = 0
        while _HEARTBEAT:
            time.sleep(1.0)
            _writeln(f"[{_now()}] heartbeat {i}")
            i += 1
    t = threading.Thread(target=beat, name="diag_heartbeat", daemon=True)
    t.start()

def mark_event(name, **kw):
    try:
        payload = " ".join(f"{k}={kw[k]!r}" for k in sorted(kw))
    except Exception:
        payload = repr(kw)
    _writeln(f"[{_now()}] EVENT {name}: {payload}")

def wire_to_videopane(VideoPaneClass):
    # Wrap VideoPane.open to log transitions and guard exceptions.
    try:
        orig = VideoPaneClass.open
    except Exception as e:
        _writeln(f"[{_now()}] wire_to_videopane failed (no .open): {e!r}")
        return
    if getattr(VideoPaneClass, "_diag_open_wrapped", False):
        return
    def open_wrapper(self, path, *args, **kwargs):
        try:
            kind = "unknown"
            p = str(path)
            ext = (os.path.splitext(p)[1] or "").lower()
            if ext in (".mp3",".wav",".aac",".flac",".ogg",".m4a"):
                kind = "audio"
            elif ext in (".png",".jpg",".jpeg",".bmp",".gif",".webp"):
                kind = "image"
            else:
                kind = "video/other"
            mark_event("open.begin", path=p, kind=kind)
        except Exception:
            pass
        try:
            res = orig(self, path, *args, **kwargs)
            mark_event("open.end", result=type(res).__name__ if res is not None else "None")
            return res
        except SystemExit as e:
            mark_event("open.system_exit", code=getattr(e, 'code', None))
            raise
        except Exception as e:
            mark_event("open.exception", error=str(e))
            _writeln("".join(traceback.format_exc()))
            raise
    VideoPaneClass.open = open_wrapper
    VideoPaneClass._diag_open_wrapped = True
    _writeln(f"[{_now()}] VideoPane.open wrapped for diagnostics")

def init_diagnostics():
    _ensure_logs()
    _dump_header()
    _install_excepthooks()
    _install_faulthandler()
    _install_qt_message_handler()
    _install_app_hooks()
    _start_heartbeat()
    atexit.register(lambda: _writeln(f"[{_now()}] diagnostics exit"))
    return LOG_PATH
