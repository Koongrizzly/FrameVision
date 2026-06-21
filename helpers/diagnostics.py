# helpers/diagnostics.py — diagnostics with timer/handle probe (drop-in)
from __future__ import annotations
import os, sys, time, platform, subprocess, json, threading, traceback, weakref, ctypes
from typing import Optional, Dict
from PySide6 import QtCore
from PySide6.QtCore import QSettings, QTimer, Qt
from PySide6.QtWidgets import QApplication

APP_ORG = "FrameVision"; APP_NAME = "FrameVision"
LOG_DIR = os.path.join(os.getcwd(), "logs")
LOG_FILE = os.path.join(LOG_DIR, "framevision.log")  # keep existing path for compatibility

# ----------------- logging core -----------------
def _ensure_logfile() -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8"): pass
    except Exception: pass

def diag_enabled() -> bool:
    try:
        s = QSettings(APP_ORG, APP_NAME)
        legacy_val = s.value("diagnostics_enabled", None)
        val = s.value("diag_probe_enabled", legacy_val if legacy_val is not None else "true")
        return str(val).lower() in ("1","true","yes","on")
    except Exception:
        return True

def _fmt(*args) -> str:
    parts = []
    for a in args:
        try: parts.append(str(a))
        except Exception:
            try: parts.append(repr(a))
            except Exception: parts.append("<unprintable>")
    return " ".join(parts)

def log(*args) -> None:
    if not diag_enabled(): return
    _ensure_logfile()
    line = f"[FrameVision DIAG] {time.strftime('%Y-%m-%d %H:%M:%S')} " + _fmt(*args)
    try: print(line, flush=True)
    except Exception: pass
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f: f.write(line + "\n")
    except Exception: pass

# ----------------- Windows handle helpers -----------------
try:
    _user32 = ctypes.windll.user32
    _kernel32 = ctypes.windll.kernel32
    _GetGuiResources = _user32.GetGuiResources
    _GetCurrentProcess = _kernel32.GetCurrentProcess
    def _get_handle_counts():
        h = _GetCurrentProcess()
        gdi = int(_GetGuiResources(h, 0))  # 0=GDI
        usr = int(_GetGuiResources(h, 1))  # 1=USER
        return gdi, usr
except Exception:
    def _get_handle_counts():
        return -1, -1

# ----------------- timer guard (optional) -----------------
_GUARD_INSTALLED = False
_LIVE_TIMERS = weakref.WeakSet()
_TIMER_ORIGINS: Dict[int, str] = {}
_TIMER_LOCK = threading.Lock()

def _origin_from_stack()->str:
    try:
        for fr in traceback.extract_stack(limit=80)[:-2]:
            fn = str(fr.filename).replace("\\", "/")
            if "PySide6" in fn or "site-packages" in fn:
                continue
            return f"{fr.filename}:{fr.lineno} in {fr.name}"
    except Exception:
        pass
    return "<unknown origin>"

def _install_timer_guard(threshold:int=4000):
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    QTimerCls = QtCore.QTimer
    orig_init = QTimerCls.__init__

    def wrapped_init(self, *a, **kw):
        orig_init(self, *a, **kw)
        try:
            with _TIMER_LOCK:
                _LIVE_TIMERS.add(self)
                _TIMER_ORIGINS[id(self)] = _origin_from_stack()
                n = len(_LIVE_TIMERS)
                if n % 500 == 0 or n > threshold:
                    counts: Dict[str,int] = {}
                    for t in list(_LIVE_TIMERS):
                        o = _TIMER_ORIGINS.get(id(t), "?")
                        counts[o] = counts.get(o, 0) + 1
                    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    log(f"[timer_guard] Live QTimers={n}  top={top}")
        except Exception as e:
            log("[timer_guard] wrap error:", e)

    QTimerCls.__init__ = wrapped_init  # type: ignore[assignment]
    _GUARD_INSTALLED = True
    log(f"[timer_guard] Installed; threshold={threshold}")

# ----------------- handle logger (optional; no Qt timers used) -----------------
_HANDLE_THREAD = None
_HANDLE_STOP = threading.Event()

def _handle_loop(period_ms:int):
    while not _HANDLE_STOP.is_set():
        g,u = _get_handle_counts()
        log(f"[handles] GDI={g} USER={u}")
        _HANDLE_STOP.wait(max(0.2, period_ms/1000.0))

def _start_handle_logger(period_ms:int=1000):
    global _HANDLE_THREAD
    if _HANDLE_THREAD and _HANDLE_THREAD.is_alive():
        return
    _HANDLE_THREAD = threading.Thread(target=_handle_loop, name="fv_handle_logger", args=(period_ms,), daemon=True)
    _HANDLE_THREAD.start()
    log(f"[handles] logger started @{period_ms}ms")


# ----------------- resource usage probe (manual report + optional live logger) -----------------
_RESOURCE_THREAD = None
_RESOURCE_STOP = threading.Event()
_LAST_INTERNAL_STACK_DUMP = 0.0

_RESOURCE_KNOWN_TOKENS = (
    "framevision", "python", "pythonw", "ffmpeg", "ffprobe", "nvidia-smi",
    "rife", "realesrgan", "waifu2x", "upscayl", "ltx", "wan", "hunyuan",
    "qwen", "zimage", "z-image", "comfy", "sd.exe", "uv.exe", "worker.py",
    "planner", "media", "preview", "thumbnail", "thumb",
)


def _mb(value) -> str:
    try:
        return f"{float(value) / (1024.0 * 1024.0):.1f} MB"
    except Exception:
        return "? MB"


def _short_cmd(cmd, limit: int = 420) -> str:
    try:
        if isinstance(cmd, (list, tuple)):
            txt = " ".join(str(x) for x in cmd)
        else:
            txt = str(cmd or "")
        return _safe_text(txt, limit)
    except Exception:
        return ""


def _nvidia_smi_resource_snapshot() -> tuple[list[str], dict[int, dict]]:
    """Return GPU summary lines and GPU-app info by pid. Never raises."""
    lines: list[str] = []
    gpu_by_pid: dict[int, dict] = {}

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        )
        raw = (out.stdout or out.stderr or "").strip()
        if raw:
            lines.append("GPU summary:")
            for ln in raw.splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) >= 5:
                    lines.append(f" - GPU {parts[0]}: util={parts[1]}% mem={parts[2]}/{parts[3]} MB temp={parts[4]}C")
                else:
                    lines.append(" - " + _safe_text(ln, 600))
        else:
            lines.append("GPU summary: <no nvidia-smi output>")
    except FileNotFoundError:
        lines.append("GPU summary: nvidia-smi not found on PATH")
    except Exception as e:
        lines.append(f"GPU summary failed: {e}")

    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
        )
        raw = (out.stdout or out.stderr or "").strip()
        if raw:
            for ln in raw.splitlines():
                parts = [x.strip() for x in ln.split(",")]
                if len(parts) >= 3:
                    try:
                        pid = int(parts[0])
                    except Exception:
                        continue
                    try:
                        used = float(parts[2])
                    except Exception:
                        used = 0.0
                    gpu_by_pid[pid] = {"process_name": parts[1], "used_mb": used}
    except Exception:
        pass

    return lines, gpu_by_pid


def _proc_resource_row(item: dict, gpu_by_pid: dict[int, dict] | None = None, *, include_cmd: bool = False) -> str:
    gpu_by_pid = gpu_by_pid or {}
    pid = item.get("pid", "?")
    name = item.get("name") or "?"
    cpu = item.get("cpu", 0.0)
    rss = item.get("rss", 0)
    threads = item.get("threads", "?")
    handles = item.get("handles", None)
    flags = item.get("flags", "")
    io_read = item.get("io_read", None)
    io_write = item.get("io_write", None)

    bits = [f"pid={pid}", str(name), f"cpu={float(cpu or 0.0):.1f}%", f"rss={_mb(rss)}", f"threads={threads}"]
    if handles is not None:
        bits.append(f"handles={handles}")
    gp = gpu_by_pid.get(int(pid)) if isinstance(pid, int) or str(pid).isdigit() else None
    if gp:
        bits.append(f"gpu={gp.get('used_mb', 0):.0f} MB")
    if io_read is not None or io_write is not None:
        try:
            bits.append(f"io=R{_mb(io_read or 0)}/W{_mb(io_write or 0)}")
        except Exception:
            pass
    if flags:
        bits.append(f"scope={flags}")
    if include_cmd:
        cmd = _short_cmd(item.get("cmd"), 900)
        if cmd:
            bits.append(f"cmd={cmd}")
    return " ".join(bits)


def _frame_brief(fr) -> str:
    """Format one frame without dumping huge paths or secrets."""
    try:
        fn = str(getattr(fr, "filename", "") or "").replace("\\", "/")
        try:
            root = os.getcwd().replace("\\", "/").rstrip("/")
            if root and fn.lower().startswith(root.lower() + "/"):
                fn = fn[len(root) + 1:]
        except Exception:
            pass
        return _safe_text(f"{fn}:{getattr(fr, 'lineno', '?')} in {getattr(fr, 'name', '?')} | {getattr(fr, 'line', '') or ''}", 950)
    except Exception:
        return "<unprintable frame>"


def _python_thread_stack_report_lines(*, sample_seconds: float = 0.35, top_n: int = 8, stack_limit: int = 10) -> list[str]:
    """Sample Python thread CPU deltas and show stack snippets.

    This is the useful follow-up after the process probe says "FrameVision itself"
    is hot. It can point at a timer callback, media refresh loop, monitor poller,
    animation update, folder scanner, or other Python-side code path.
    """
    lines: list[str] = []
    lines.append("[Internal Python/Qt thread sampler]")
    lines.append(f"Sample window: {float(sample_seconds):.2f}s | top_n={int(top_n)}")

    try:
        import psutil  # type: ignore
    except Exception as e:
        lines.append(f"psutil unavailable: {e}")
        return lines

    try:
        proc = psutil.Process(os.getpid())
        before = {int(t.id): float(t.user_time + t.system_time) for t in proc.threads()}
        time.sleep(max(0.10, min(1.5, float(sample_seconds))))
        after = {int(t.id): float(t.user_time + t.system_time) for t in proc.threads()}

        by_native = {}
        by_ident = {}
        for th in threading.enumerate():
            try:
                if getattr(th, "native_id", None) is not None:
                    by_native[int(th.native_id)] = th
                if getattr(th, "ident", None) is not None:
                    by_ident[int(th.ident)] = th
            except Exception:
                pass

        frames = sys._current_frames()
        rows = []
        for tid, after_time in after.items():
            delta = float(after_time - before.get(tid, after_time))
            th = by_native.get(tid)
            name = getattr(th, "name", None) or "<non-Python/native thread>"
            ident = getattr(th, "ident", None) if th is not None else None
            frame = frames.get(int(ident)) if ident is not None else None
            rows.append((delta, tid, name, th, frame))
        rows.sort(key=lambda x: x[0], reverse=True)

        if not rows:
            lines.append(" - <no thread data>")
            return lines

        shown = 0
        for delta, tid, name, th, frame in rows:
            # Keep idle native thread noise down, but still show a few rows when all deltas are 0.
            if delta <= 0.0005 and shown >= 3:
                continue
            daemon = getattr(th, "daemon", None) if th is not None else None
            py_ident = getattr(th, "ident", None) if th is not None else None
            lines.append(f" - native_tid={tid} name={_safe_text(name, 180)} cpu_time_delta={delta:.4f}s daemon={daemon} py_ident={py_ident}")
            if frame is None:
                lines.append("     stack=<native/no Python frame visible>")
            else:
                try:
                    stack = traceback.extract_stack(frame, limit=max(4, int(stack_limit)))
                    # Last frames are where the thread currently is. Filter only extremely unhelpful internals.
                    for fr in stack[-max(3, int(stack_limit)):]:
                        lines.append("     " + _frame_brief(fr))
                except Exception as e:
                    lines.append(f"     stack extract failed: {e}")
            shown += 1
            if shown >= max(1, int(top_n)):
                break

        try:
            with _TIMER_LOCK:
                live = list(_LIVE_TIMERS)
                if live:
                    counts: Dict[str, int] = {}
                    active_counts: Dict[str, int] = {}
                    for t in live:
                        o = _TIMER_ORIGINS.get(id(t), "?")
                        counts[o] = counts.get(o, 0) + 1
                        try:
                            if bool(t.isActive()):
                                active_counts[o] = active_counts.get(o, 0) + 1
                        except Exception:
                            pass
                    lines.append("QTimer origin summary:")
                    for origin, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:10]:
                        active = active_counts.get(origin, 0)
                        lines.append(f" - timers={count} active={active} origin={_safe_text(origin, 950)}")
                else:
                    lines.append("QTimer origin summary: <timer guard not enabled or no timers captured>")
        except Exception as e:
            lines.append(f"QTimer origin summary failed: {e}")
    except Exception as e:
        lines.append(f"Internal thread sampler failed: {e}")
        try:
            lines.append(traceback.format_exc(limit=8))
        except Exception:
            pass
    return lines


def _should_log_internal_stack(compact: bool, force: bool = False) -> bool:
    """Throttle automatic stack dumps during the live resource probe."""
    global _LAST_INTERNAL_STACK_DUMP
    if force or not compact:
        return True
    now = time.time()
    if now - float(_LAST_INTERNAL_STACK_DUMP or 0.0) >= 25.0:
        _LAST_INTERNAL_STACK_DUMP = now
        return True
    return False


# ----------------- Qt event activity sampler -----------------
def _qt_event_type_name(event_type) -> str:
    try:
        # Qt 6 enum pretty name when available.
        name = getattr(event_type, "name", None)
        if name:
            return str(name)
    except Exception:
        pass
    try:
        val = int(event_type)
    except Exception:
        return str(event_type)
    try:
        for nm in dir(QtCore.QEvent.Type):
            if nm.startswith("_"):
                continue
            try:
                if int(getattr(QtCore.QEvent.Type, nm)) == val:
                    return nm
            except Exception:
                continue
    except Exception:
        pass
    return f"Type{val}"


def _qt_object_label(obj) -> str:
    try:
        cls = obj.metaObject().className() if hasattr(obj, "metaObject") else obj.__class__.__name__
    except Exception:
        cls = obj.__class__.__name__ if obj is not None else "<none>"
    try:
        name = obj.objectName() if hasattr(obj, "objectName") else ""
    except Exception:
        name = ""
    try:
        parent = obj.parent() if hasattr(obj, "parent") else None
        pcls = parent.metaObject().className() if parent is not None and hasattr(parent, "metaObject") else (parent.__class__.__name__ if parent is not None else "")
    except Exception:
        pcls = ""
    bits = [str(cls or "?")]
    if name:
        bits.append(f"#{name}")
    if pcls:
        bits.append(f"parent={pcls}")
    return _safe_text(" ".join(bits), 500)


class _QtEventActivityProbe(QtCore.QObject):
    """Global event filter used only during an explicit diagnostic sample."""
    def __init__(self):
        super().__init__()
        self.total = 0
        self.by_type: Dict[str, int] = {}
        self.by_source: Dict[tuple[str, str], int] = {}
        self.watch_names = {
            "Timer", "Paint", "UpdateRequest", "UpdateLater", "LayoutRequest", "Resize", "Move",
            "MouseMove", "HoverMove", "HoverEnter", "HoverLeave", "Polish", "PolishRequest",
            "ChildAdded", "ChildRemoved", "DynamicPropertyChange", "MetaCall", "DeferredDelete",
        }

    def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
        try:
            typ = _qt_event_type_name(event.type())
            # Count all events by type, but source-detail only for noisy/render/timer types.
            self.total += 1
            self.by_type[typ] = self.by_type.get(typ, 0) + 1
            if typ in self.watch_names:
                lab = _qt_object_label(obj)
                key = (typ, lab)
                self.by_source[key] = self.by_source.get(key, 0) + 1
        except Exception:
            pass
        return False


def _qt_event_activity_report_lines(*, sample_seconds: float = 2.0, top_n: int = 20) -> list[str]:
    """Sample Qt event traffic to find native/Qt-side churn.

    The Python thread sampler can miss work done inside Qt/C++ paint/timer/update handling.
    This event filter tells us whether idle CPU is coming from constant Timer/Paint/Update
    events and which widget/object class receives them.
    """
    lines: list[str] = []
    lines.append("[Qt event activity sampler]")
    lines.append(f"Sample window: {float(sample_seconds):.2f}s | top_n={int(top_n)}")
    try:
        app = QApplication.instance()
        if app is None:
            lines.append("No QApplication instance available.")
            return lines

        probe = _QtEventActivityProbe()
        app.installEventFilter(probe)
        try:
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(max(200, int(float(sample_seconds) * 1000)), loop.quit)
            loop.exec()
        finally:
            try:
                app.removeEventFilter(probe)
            except Exception:
                pass

        lines.append(f"Total Qt events seen: {probe.total}")
        if probe.by_type:
            lines.append("Top event types:")
            for typ, count in sorted(probe.by_type.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top_n))]:
                lines.append(f" - {typ}: {count}")
        else:
            lines.append("Top event types: <none>")

        if probe.by_source:
            lines.append("Top watched event sources:")
            for (typ, lab), count in sorted(probe.by_source.items(), key=lambda kv: kv[1], reverse=True)[:max(1, int(top_n))]:
                lines.append(f" - {typ}: {count} -> {lab}")
        else:
            lines.append("Top watched event sources: <none>")

        # A tiny hint section, not final blame.
        hints = []
        timer_count = probe.by_type.get("Timer", 0)
        paint_count = probe.by_type.get("Paint", 0) + probe.by_type.get("UpdateRequest", 0) + probe.by_type.get("UpdateLater", 0)
        hover_count = probe.by_type.get("HoverMove", 0) + probe.by_type.get("MouseMove", 0)
        if timer_count >= 80:
            hints.append(f"High timer traffic during idle sample: Timer={timer_count}")
        if paint_count >= 80:
            hints.append(f"High paint/update traffic during idle sample: Paint/Update={paint_count}")
        if hover_count >= 80:
            hints.append(f"High mouse/hover traffic during sample: Mouse/Hover={hover_count}")
        if hints:
            lines.append("Qt activity hints:")
            for h in hints:
                lines.append(" - " + h)
    except Exception as e:
        lines.append(f"Qt event activity sampler failed: {e}")
        try:
            lines.append(traceback.format_exc(limit=8))
        except Exception:
            pass
    return lines


def _resource_snapshot_report_lines(*, sample_seconds: float = 0.45, top_n: int = 10, compact: bool = False, internal_stack: bool | None = None) -> list[str]:
    """Rank CPU/RAM/GPU users so stutter has something concrete to blame."""
    lines: list[str] = []
    lines.append("[Resource usage snapshot]")
    lines.append(f"Sample window: {float(sample_seconds):.2f}s | top_n={int(top_n)}")

    gpu_lines, gpu_by_pid = _nvidia_smi_resource_snapshot()

    try:
        import psutil  # type: ignore
    except Exception as e:
        lines.append(f"psutil unavailable: {e}")
        lines.extend(gpu_lines)
        if gpu_by_pid:
            lines.append("GPU compute apps:")
            for pid, gp in sorted(gpu_by_pid.items(), key=lambda kv: kv[1].get("used_mb", 0), reverse=True):
                lines.append(f" - pid={pid} {gp.get('process_name','?')} gpu={gp.get('used_mb',0):.0f} MB")
        return lines

    try:
        cur_pid = os.getpid()
        cur = psutil.Process(cur_pid)
        try:
            children = cur.children(recursive=True)
        except Exception:
            children = []
        tree_pids = {cur_pid}
        for ch in children:
            try:
                tree_pids.add(int(ch.pid))
            except Exception:
                pass

        procs = []
        for pr in psutil.process_iter(["pid", "ppid", "name", "exe", "cmdline", "status", "create_time"]):
            try:
                # prime process CPU measurement
                pr.cpu_percent(None)
                procs.append(pr)
            except Exception:
                continue

        time.sleep(max(0.15, min(2.0, float(sample_seconds))))

        items: list[dict] = []
        for pr in procs:
            try:
                info = pr.info or {}
                pid = int(info.get("pid") or pr.pid)
                name = str(info.get("name") or "")
                cmdline = info.get("cmdline") or []
                exe = str(info.get("exe") or "")
                cmd = " ".join(cmdline)
                hay = (name + " " + exe + " " + cmd).lower().replace("/", "\\")
                cpu = float(pr.cpu_percent(None) or 0.0)
                try:
                    rss = int(pr.memory_info().rss or 0)
                except Exception:
                    rss = 0
                try:
                    threads = int(pr.num_threads())
                except Exception:
                    threads = "?"
                handles = None
                try:
                    if hasattr(pr, "num_handles"):
                        handles = int(pr.num_handles())
                except Exception:
                    handles = None
                io_read = io_write = None
                try:
                    io = pr.io_counters()
                    io_read = int(getattr(io, "read_bytes", 0) or 0)
                    io_write = int(getattr(io, "write_bytes", 0) or 0)
                except Exception:
                    pass

                flags = []
                if pid == cur_pid:
                    flags.append("FrameVision-main")
                elif pid in tree_pids:
                    flags.append("FrameVision-child")
                if any(tok in hay for tok in _RESOURCE_KNOWN_TOKENS):
                    flags.append("known-tool")
                if pid in gpu_by_pid:
                    flags.append("GPU-app")

                items.append({
                    "pid": pid,
                    "ppid": int(info.get("ppid") or 0),
                    "name": name,
                    "exe": exe,
                    "cmd": cmd,
                    "cpu": cpu,
                    "rss": rss,
                    "threads": threads,
                    "handles": handles,
                    "io_read": io_read,
                    "io_write": io_write,
                    "flags": "+".join(flags),
                    "is_tree": pid in tree_pids,
                    "is_known": any(tok in hay for tok in _RESOURCE_KNOWN_TOKENS),
                })
            except Exception:
                continue

        items_by_pid = {int(x.get("pid")): x for x in items if str(x.get("pid", "")).isdigit()}
        top_cpu = sorted(items, key=lambda x: float(x.get("cpu") or 0.0), reverse=True)[:max(1, int(top_n))]
        top_mem = sorted(items, key=lambda x: int(x.get("rss") or 0), reverse=True)[:max(1, int(top_n))]
        related = [x for x in items if x.get("is_tree") or x.get("is_known") or int(x.get("pid") or 0) in gpu_by_pid]
        related_sorted = sorted(related, key=lambda x: (float(x.get("cpu") or 0.0), int(x.get("rss") or 0)), reverse=True)[:max(1, int(top_n))]

        if compact:
            cpu_short = "; ".join(_proc_resource_row(x, gpu_by_pid) for x in top_cpu[:5] if float(x.get("cpu") or 0.0) >= 0.1) or "<idle/no clear CPU offender>"
            mem_short = "; ".join(_proc_resource_row(x, gpu_by_pid) for x in top_mem[:5])
            fv_short = "; ".join(_proc_resource_row(x, gpu_by_pid) for x in related_sorted[:5]) or "<none>"
            lines.append("Top CPU: " + cpu_short)
            lines.append("Top RAM: " + mem_short)
            lines.append("FrameVision/known/GPU related: " + fv_short)
            if gpu_lines:
                lines.extend(gpu_lines[:4])
        else:
            gdi, usr = _get_handle_counts()
            cur_item = items_by_pid.get(cur_pid)
            lines.append(f"Current GUI handles: GDI={gdi} USER={usr}")
            if cur_item:
                lines.append("Current FrameVision process:")
                lines.append(" - " + _proc_resource_row(cur_item, gpu_by_pid, include_cmd=True))
            lines.append(f"FrameVision child process count: {len(children)}")
            if children:
                for ch in children[:20]:
                    try:
                        item = items_by_pid.get(int(ch.pid))
                        if item:
                            lines.append(" - child " + _proc_resource_row(item, gpu_by_pid, include_cmd=True))
                    except Exception:
                        pass
                if len(children) > 20:
                    lines.append(f" - <trimmed {len(children) - 20} more child process(es)>")

            lines.append("")
            lines.append("FrameVision / known tool / GPU related processes:")
            if related_sorted:
                for item in related_sorted:
                    lines.append(" - " + _proc_resource_row(item, gpu_by_pid, include_cmd=True))
            else:
                lines.append(" - <none>")

            lines.append("")
            lines.append("Top CPU processes:")
            for item in top_cpu:
                lines.append(" - " + _proc_resource_row(item, gpu_by_pid))

            lines.append("")
            lines.append("Top RAM processes:")
            for item in top_mem:
                lines.append(" - " + _proc_resource_row(item, gpu_by_pid))

            lines.append("")
            lines.extend(gpu_lines)
            if gpu_by_pid:
                lines.append("GPU compute apps:")
                for pid, gp in sorted(gpu_by_pid.items(), key=lambda kv: kv[1].get("used_mb", 0), reverse=True):
                    item = items_by_pid.get(int(pid))
                    if item:
                        lines.append(" - " + _proc_resource_row(item, gpu_by_pid, include_cmd=True))
                    else:
                        lines.append(f" - pid={pid} {gp.get('process_name','?')} gpu={gp.get('used_mb',0):.0f} MB")
            else:
                lines.append("GPU compute apps: <none reported>")

        # Simple suspect notes. They are intentionally conservative: evidence first, blame second.
        notes: list[str] = []
        high_framevision_cpu = False
        for item in items:
            try:
                pid = int(item.get("pid") or 0)
                name = str(item.get("name") or "").lower()
                cpu = float(item.get("cpu") or 0.0)
                rss = int(item.get("rss") or 0)
                is_tree = bool(item.get("is_tree"))
                is_known = bool(item.get("is_known"))
                gp = gpu_by_pid.get(pid)
                if is_tree and cpu >= 20.0:
                    high_framevision_cpu = True
                    notes.append(f"FrameVision process high CPU: {_proc_resource_row(item, gpu_by_pid)}")
                elif is_tree and cpu >= 8.0:
                    high_framevision_cpu = True
                if is_tree and rss >= 4 * 1024 * 1024 * 1024:
                    notes.append(f"FrameVision process high RAM: {_proc_resource_row(item, gpu_by_pid)}")
                if ("ffmpeg" in name or "ffprobe" in name) and (cpu >= 5.0 or is_tree):
                    notes.append(f"FFmpeg/ffprobe active during sample: {_proc_resource_row(item, gpu_by_pid)}")
                if is_known and gp and float(gp.get("used_mb") or 0.0) >= 1000:
                    notes.append(f"Known tool using VRAM: {_proc_resource_row(item, gpu_by_pid)}")
                if gp and not is_tree and not is_known and float(gp.get("used_mb") or 0.0) >= 500:
                    notes.append(f"External GPU process using VRAM: {_proc_resource_row(item, gpu_by_pid)}")
            except Exception:
                continue
        try:
            if cur_item:
                th = cur_item.get("threads")
                if isinstance(th, int) and th >= 180:
                    notes.append(f"FrameVision has many threads: {th}")
                hd = cur_item.get("handles")
                if isinstance(hd, int) and hd >= 8000:
                    notes.append(f"FrameVision has many Windows handles: {hd}")
            gdi, usr = _get_handle_counts()
            if gdi >= 8000 or usr >= 8000:
                notes.append(f"GUI handle count is high: GDI={gdi} USER={usr}")
        except Exception:
            pass

        if notes:
            lines.append("")
            lines.append("Potential suspects from this sample:")
            seen_notes = set()
            for note in notes[:20]:
                if note in seen_notes:
                    continue
                seen_notes.add(note)
                lines.append(" - " + note)
        elif not compact:
            lines.append("")
            lines.append("Potential suspects from this sample: <no obvious high offender in this short sample>")

        try:
            if internal_stack is None:
                try:
                    ss = QSettings(APP_ORG, APP_NAME)
                    internal_stack = _flag_true(ss.value("diagnostics_internal_stack_on_high_cpu", "true"))
                except Exception:
                    internal_stack = True
            if bool(internal_stack) and high_framevision_cpu and _should_log_internal_stack(compact):
                lines.append("")
                lines.extend(_python_thread_stack_report_lines(sample_seconds=1.75, top_n=8, stack_limit=9))
        except Exception as e:
            lines.append(f"Internal stack sampler trigger failed: {e}")
    except Exception as e:
        lines.append(f"Resource snapshot failed: {e}")
        try:
            lines.append(traceback.format_exc(limit=8))
        except Exception:
            pass
        lines.extend(gpu_lines)

    return lines


def _resource_loop(period_ms: int, top_n: int):
    try:
        period_ms = int(period_ms)
    except Exception:
        period_ms = 3000
    if period_ms < 1000:
        period_ms = 1000
    if period_ms > 60000:
        period_ms = 60000
    sample_seconds = max(0.20, min(1.0, period_ms / 10000.0))
    while not _RESOURCE_STOP.is_set():
        try:
            lines = _resource_snapshot_report_lines(sample_seconds=sample_seconds, top_n=int(top_n), compact=True)
            for ln in lines:
                log("[resource_probe]", ln)
        except Exception as e:
            log("[resource_probe] failed:", e)
        _RESOURCE_STOP.wait(period_ms / 1000.0)


def _start_resource_logger(period_ms: int = 3000, top_n: int = 8):
    global _RESOURCE_THREAD
    if _RESOURCE_THREAD and _RESOURCE_THREAD.is_alive():
        return
    _RESOURCE_STOP.clear()
    _RESOURCE_THREAD = threading.Thread(target=_resource_loop, name="fv_resource_probe", args=(period_ms, top_n), daemon=True)
    _RESOURCE_THREAD.start()
    log(f"[resource_probe] logger started @{period_ms}ms top_n={top_n}")

# ----------------- Qt message hook (stack dump on registerTimer; suppress QPainter spam) -----------------
_prev_qt_handler = None
_logged_qpainter_once = False
def _qt_message_handler(mode, context, message):
    try: m = str(message)
    except Exception: m = message

    # Suppress noisy QPainter spew entirely (configurable to log once)
    if isinstance(m, str) and "QPainter::" in m:
        try:
            s = QSettings(APP_ORG, APP_NAME)
            if str(s.value("diagnostics_log_qpainter","false")).lower() in ("1","true","yes","on"):
                global _logged_qpainter_once
                if not _logged_qpainter_once:
                    log(f"Qt(QPainter suppressed): {m}")
                    _logged_qpainter_once = True
        except Exception: pass
        return

    # Suppress specific harmless messages
    try:
        if isinstance(m, str):
            if "Could not parse stylesheet of object" in m and "FvSettingsContent" in m:
                return
            if "QObject::startTimer: Timers can only be used with threads started with QThread" in m:
                return
    except Exception:
        pass

    # Mirror to previous handler/stdout for normal flow
    try:
        if _prev_qt_handler is not None: _prev_qt_handler(mode, context, message)
        else: 
            try: print(m)
            except Exception: pass
    except Exception:
        pass

    # When timer creation fails, dump stack + handle counts
    try:
        if isinstance(m, str) and ("registerTimer" in m or "Failed to create a timer" in m):
            stack = "".join(traceback.format_stack(limit=80))
            gdi, usr = _get_handle_counts()
            log(f"[QtTimer] registerTimer warning; USER={usr} GDI={gdi}\n{stack}\n{'-'*70}")
    except Exception as e:
        log("registerTimer diagnostic failed:", e)

# Install the message handler ASAP (module import time), but it's cheap
try:
    _prev_qt_handler = QtCore.qInstallMessageHandler(_qt_message_handler)
except Exception:
    pass

# ----------------- helpers -----------------
NOISY_KEYS = {"intro_cached_urls","intro_gallery_queue","dad_jokes_queue","dad_jokes_seen","dad_jokes_used"}
def _summarize_value(v):
    try:
        s = v if isinstance(v, str) else str(v)
        try:
            obj = json.loads(s)
            if isinstance(obj,(list,tuple)): return f"[redacted list: {len(obj)} items]"
            if isinstance(obj,dict): return f"[redacted dict: {len(obj)} keys]"
        except Exception: pass
        if "http" in s or len(s) > 120: return f"[redacted string: len={len(s)}]"
        return "[redacted]"
    except Exception: return "[redacted]"
def _maybe_redact(k,v):
    k=str(k); lk=k.lower()
    return _summarize_value(v) if (k in NOISY_KEYS or any(p in lk for p in ("urls","gallery","jokes"))) else v

def dump_qsettings(prefix: Optional[str]=None) -> None:
    try:
        s = QSettings(APP_ORG, APP_NAME); keys = sorted(list(s.allKeys()))
        log(f"QSettings dump ({len(keys)} keys){' — ' + prefix if prefix else ''}:")
        for k in keys:
            try: v = _maybe_redact(k, s.value(k))
            except Exception: v = "<error>"
            log(f" - {k} = {v!r}")
    except Exception as e: log("QSettings dump failed:", e)

def log_environment() -> None:
    try:
        log(f"Python: {sys.version.split()[0]} @ {sys.executable}")
        log(f"Platform: {platform.platform()}")
        log(f"Qt/PySide6: Qt={QtCore.qVersion()} PySide6={getattr(QtCore,'__version__','unknown')}")
        in_venv = hasattr(sys,"real_prefix") or (getattr(sys,"base_prefix",sys.prefix) != sys.prefix)
        log(f"Virtualenv: {'yes' if in_venv else 'no'} prefix={sys.prefix}")
    except Exception as e: log("Environment log failed:", e)

def log_packages() -> None:
    def _try(name, attr="__version__"):
        try:
            m = __import__(name, fromlist=['*']); return True, getattr(m, attr, "<no __version__>")
        except Exception as e: return False, f"<missing: {e.__class__.__name__}>"
    for mod in ["torch","transformers","onnxruntime_genai","onnxruntime","onnxruntime_directml","huggingface_hub","safetensors"]:
        ok, ver = _try(mod); log(f"Package {mod}: {'OK' if ok else 'MISSING'} {ver}")

def log_ffmpeg() -> None:
    try:
        out = subprocess.run(["ffmpeg","-hide_banner","-version"], capture_output=True, text=True, timeout=5)
        line1 = (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "<no output>"
        log("ffmpeg:", line1)
    except FileNotFoundError: log("ffmpeg: not found on PATH")
    except Exception as e: log("ffmpeg: error probing:", e)


# ----------------- manual app report helpers -----------------
def _safe_text(value, limit: int = 600) -> str:
    """Small text sanitizer for manual reports."""
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = repr(value)
    try:
        home = os.path.expanduser("~")
        if home and home not in ("~", os.path.sep):
            text = text.replace(home, "<USER_HOME>")
    except Exception:
        pass
    # Avoid dumping obvious secret-like values into a report.
    lowered = text.lower()
    if any(x in lowered for x in ("token=", "apikey=", "api_key=", "password=", "passwd=", "secret=")):
        text = "<redacted secret-like text>"
    if limit and len(text) > limit:
        return text[:limit] + f" ... <trimmed {len(text) - limit} chars>"
    return text


def _append_report_lines(lines: list[str]) -> None:
    _ensure_logfile()
    try:
        with open(LOG_FILE, "a", encoding="utf-8", errors="replace") as f:
            for line in lines:
                try:
                    f.write(str(line).rstrip("\n") + "\n")
                except Exception:
                    f.write("<unprintable line>\n")
    except Exception:
        pass


def _metadata_version(pkg_name: str) -> str:
    try:
        try:
            from importlib import metadata
        except Exception:
            import importlib_metadata as metadata  # type: ignore
        return metadata.version(pkg_name)
    except Exception as e:
        return f"<missing: {e.__class__.__name__}>"


def _system_snapshot_report_lines() -> list[str]:
    lines: list[str] = []
    lines.append("[System snapshot]")
    try:
        lines.append(f"Python: {sys.version.split()[0]} @ {_safe_text(sys.executable)}")
        lines.append(f"Platform: {platform.platform()}")
        lines.append(f"CWD: {_safe_text(os.getcwd())}")
        lines.append(f"Log file: {_safe_text(LOG_FILE)}")
        lines.append(f"ARGV: {_safe_text(' '.join(sys.argv), 1200)}")
    except Exception as e:
        lines.append(f"Environment basic info failed: {e}")

    try:
        lines.append(f"Qt/PySide6: Qt={QtCore.qVersion()} PySide6={getattr(QtCore, '__version__', 'unknown')}")
    except Exception as e:
        lines.append(f"Qt info failed: {e}")

    try:
        in_venv = hasattr(sys, "real_prefix") or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
        lines.append(f"Virtualenv: {'yes' if in_venv else 'no'} prefix={_safe_text(sys.prefix)}")
    except Exception as e:
        lines.append(f"Virtualenv info failed: {e}")

    lines.append("")
    lines.append("[Packages]")
    for mod in ["torch", "torchvision", "transformers", "diffusers", "onnxruntime_genai", "onnxruntime", "onnxruntime_directml", "huggingface_hub", "safetensors", "numpy", "Pillow", "PySide6"]:
        lines.append(f"{mod}: {_metadata_version(mod)}")

    lines.append("")
    lines.append("[FFmpeg]")
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True, timeout=5)
        first = (out.stdout or out.stderr or "").splitlines()
        lines.append(first[0] if first else "<no output>")
    except FileNotFoundError:
        lines.append("ffmpeg: not found on PATH")
    except Exception as e:
        lines.append(f"ffmpeg probe failed: {e}")

    lines.append("")
    lines.append("[QSettings snapshot]")
    try:
        s = QSettings(APP_ORG, APP_NAME)
        keys = sorted(list(s.allKeys()))
        lines.append(f"QSettings dump ({len(keys)} keys):")
        for k in keys:
            try:
                v = _maybe_redact(k, s.value(k))
            except Exception:
                v = "<error>"
            lines.append(f" - {k} = {_safe_text(repr(v), 1200)}")
    except Exception as e:
        lines.append(f"QSettings dump failed: {e}")
    return lines


def _normalize_proc_text(*parts) -> str:
    try:
        return " ".join(str(x or "") for x in parts).replace("/", "\\").lower()
    except Exception:
        return ""


def _python_runtime_kind(exe: str, cmd: str) -> str:
    """Coarse runtime label from the visible process command only."""
    hay = _normalize_proc_text(exe, cmd)
    if r".venv\scripts\python" in hay or r"environments\." in hay:
        return "project-env"
    if r"\python" in hay and "program files" in hay:
        return "system-python"
    if "python" in hay:
        return "python-other"
    return "other"


def _guess_entry_label(cmdline: list[str]) -> str:
    try:
        for part in cmdline or []:
            low = str(part).lower().replace("\\", "/")
            if low.endswith("framevision_run.py"):
                return "framevision_run.py"
            if low.endswith("worker.py"):
                return "helpers/worker.py"
            if low.endswith("start.bat"):
                return "start.bat"
        for part in cmdline or []:
            low = str(part).lower()
            if low.endswith(".py") or low.endswith(".bat") or low.endswith(".exe"):
                return os.path.basename(str(part))
    except Exception:
        pass
    return "<unknown>"


def _proc_basic_info(pr):
    try:
        return pr.as_dict(attrs=["pid", "ppid", "name", "exe", "cmdline", "status", "create_time"])
    except Exception:
        return getattr(pr, "info", {}) or {}


def _proc_cmd_text(info: dict) -> str:
    try:
        return " ".join(info.get("cmdline") or [])
    except Exception:
        return ""


def _parent_text(pr) -> tuple[object | None, str, str]:
    parent = None
    p_exe = ""
    p_cmd = ""
    try:
        parent = pr.parent()
    except Exception:
        parent = None
    if parent is not None:
        try:
            p_exe = str(parent.exe() or "")
        except Exception:
            p_exe = ""
        try:
            p_cmd = " ".join(parent.cmdline() or [])
        except Exception:
            p_cmd = ""
    return parent, p_exe, p_cmd


def _runtime_kind_for_process(pr, info: dict | None = None) -> str:
    """Smarter runtime label.

    On Windows, a venv python.exe can appear to spawn the base interpreter executable.
    If the base-python child clearly comes from a project-env parent, label that as a
    venv redirect instead of a real PATH/system Python launch.
    """
    try:
        info = info or _proc_basic_info(pr)
        exe = str(info.get("exe") or "")
        cmd = _proc_cmd_text(info)
        base_kind = _python_runtime_kind(exe, cmd)
        if base_kind != "system-python":
            return base_kind

        parent, p_exe, p_cmd = _parent_text(pr)
        parent_kind = _python_runtime_kind(p_exe, p_cmd)
        child_entry = _guess_entry_label(info.get("cmdline") or [])
        parent_entry = ""
        try:
            if parent is not None:
                parent_entry = _guess_entry_label(parent.cmdline() or [])
        except Exception:
            parent_entry = ""
        if parent_kind == "project-env" and (not parent_entry or parent_entry == child_entry):
            return "venv-redirected-base-python"

        # Extra sanity: if env inspection is allowed and VIRTUAL_ENV points into this
        # project, the process is not a plain PATH/system Python run.
        try:
            env = pr.environ()
            venv = str((env or {}).get("VIRTUAL_ENV") or "")
            if venv and ".venv" in venv.lower():
                return "venv-env-base-python"
        except Exception:
            pass
        return base_kind
    except Exception:
        try:
            return _python_runtime_kind(str((info or {}).get("exe") or ""), _proc_cmd_text(info or {}))
        except Exception:
            return "unknown"


def _current_python_context_lines() -> list[str]:
    lines: list[str] = []
    lines.append("Current Python context:")
    try:
        lines.append(f" - sys.executable={_safe_text(sys.executable, 1200)}")
        lines.append(f" - sys.prefix={_safe_text(sys.prefix, 1200)}")
        lines.append(f" - sys.base_prefix={_safe_text(getattr(sys, 'base_prefix', ''), 1200)}")
        lines.append(f" - in_venv={bool(getattr(sys, 'base_prefix', sys.prefix) != sys.prefix)}")
        lines.append(f" - VIRTUAL_ENV={_safe_text(os.environ.get('VIRTUAL_ENV', ''), 1200)}")
        lines.append(f" - argv={_safe_text(' '.join(sys.argv), 1400)}")
    except Exception as e:
        lines.append(f" - failed: {e}")
    return lines


def _proc_detail_lines(pr, *, prefix: str = " - ") -> list[str]:
    lines: list[str] = []
    info = _proc_basic_info(pr)
    pid = info.get("pid", getattr(pr, "pid", "?"))
    ppid = info.get("ppid", getattr(pr, "ppid", lambda: "?")())
    name = str(info.get("name") or "")
    exe = str(info.get("exe") or "")
    status = str(info.get("status") or "")
    cmdline = info.get("cmdline") or []
    cmd = _proc_cmd_text(info)
    kind = _runtime_kind_for_process(pr, info)
    entry = _guess_entry_label(cmdline)
    parent, _p_exe, _p_cmd = _parent_text(pr)
    parent_desc = "<none>"
    if parent is not None:
        try:
            parent_desc = f"pid={parent.pid} name={parent.name()} cmd={_safe_text(_p_cmd, 900)}"
        except Exception:
            try:
                parent_desc = f"pid={parent.pid} name={parent.name()}"
            except Exception:
                parent_desc = "<unavailable>"
    try:
        cwd = pr.cwd()
    except Exception:
        cwd = "<unavailable>"
    started = "unknown"
    try:
        ct = info.get("create_time")
        if ct:
            started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(ct)))
    except Exception:
        pass
    lines.append(f"{prefix}pid={pid} ppid={ppid} name={name} status={status} kind={kind} entry={entry}")
    if exe:
        lines.append(f"{prefix}  exe={_safe_text(exe, 1200)}")
    lines.append(f"{prefix}  cmd={_safe_text(cmd, 1400)}")
    lines.append(f"{prefix}  cwd={_safe_text(cwd, 1200)}")
    lines.append(f"{prefix}  parent={parent_desc}")
    lines.append(f"{prefix}  started={started}")
    return lines


def _process_report_lines() -> list[str]:
    lines: list[str] = []
    lines.append("[Active / known running processes]")
    lines.extend(_current_python_context_lines())
    lines.append("")
    lines.append("Runtime labels: project-env = direct local venv/env python; venv-redirected-base-python = Windows venv handoff through base python; system-python = likely real PATH/system Python.")
    lines.append("")
    known_tokens = (
        "framevision", "python", "pythonw", "ffmpeg", "ffprobe", "nvidia-smi",
        "rife", "realesrgan", "waifu2x", "upscayl", "ltx", "wan", "hunyuan",
        "qwen", "zimage", "z-image", "comfy", "sd.exe", "uv.exe",
    )
    seen: set[int] = set()
    try:
        import psutil  # type: ignore
        try:
            cur = psutil.Process(os.getpid())
            try:
                lineage = list(reversed(cur.parents())) + [cur]
            except Exception:
                lineage = [cur]
            lines.append(f"Current launcher chain ({len(lineage)} process(es)):")
            for pr in lineage:
                try:
                    seen.add(int(pr.pid))
                except Exception:
                    pass
                lines.extend(_proc_detail_lines(pr))

            children = cur.children(recursive=True)
            lines.append("")
            lines.append(f"Current child process tree ({len(children)} process(es)):")
            if not children:
                lines.append(" - <no child processes>")
            for pr in children:
                try:
                    seen.add(int(pr.pid))
                except Exception:
                    pass
                lines.extend(_proc_detail_lines(pr))
        except Exception as e:
            lines.append(f"Current process scan failed: {e}")

        lines.append("")
        lines.append("Known matching processes on this machine:")
        matches = []
        for pr in psutil.process_iter(["pid", "ppid", "name", "exe", "cmdline", "status", "create_time"]):
            try:
                info = pr.info or {}
                pid = int(info.get("pid") or 0)
                if pid in seen:
                    continue
                name = str(info.get("name") or "")
                cmd = " ".join(info.get("cmdline") or [])
                hay = (name + " " + cmd).lower()
                if any(tok in hay for tok in known_tokens):
                    matches.append(pr)
            except Exception:
                continue

        summary = {}
        for pr in matches:
            try:
                info = pr.info or {}
                cmdline = info.get("cmdline") or []
                entry = _guess_entry_label(cmdline)
                kind = _runtime_kind_for_process(pr, info)
                key = (entry, kind)
                summary[key] = summary.get(key, 0) + 1
            except Exception:
                continue
        if summary:
            lines.append("Summary by entry/runtime:")
            for (entry, kind), count in sorted(summary.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                lines.append(f" - {entry} [{kind}] x{count}")
            dupes = [((entry, kind), count) for (entry, kind), count in summary.items() if count > 1]
            mixed_entries = {}
            for (entry, kind), count in summary.items():
                mixed_entries.setdefault(entry, []).append((kind, count))
            mixed = {entry: vals for entry, vals in mixed_entries.items() if len(vals) > 1}
            if dupes or mixed:
                lines.append("Potential warnings:")
                for (entry, kind), count in dupes:
                    lines.append(f" - duplicate: {entry} has {count} process(es) under runtime {kind}")
                for entry, vals in sorted(mixed.items()):
                    # Do not warn just because Windows venv has a base-python handoff.
                    kinds = {k for k, _count in vals}
                    non_redirect = {k for k in kinds if k != "venv-redirected-base-python"}
                    if kinds == {"project-env", "venv-redirected-base-python"}:
                        lines.append(f" - note: {entry} shows project-env + venv-redirected-base-python; this can be normal Windows venv behavior.")
                    elif len(non_redirect) > 1 or "system-python" in kinds:
                        desc = ", ".join(f"{kind} x{count}" for kind, count in vals)
                        lines.append(f" - mixed runtimes: {entry} appears under multiple runtimes: {desc}")
            lines.append("")

        if not matches:
            lines.append(" - <none found>")
        else:
            matches.sort(key=lambda pr: ((str((pr.info or {}).get("name") or "")).lower(), int((pr.info or {}).get("pid") or 0)))
            for pr in matches[:80]:
                lines.extend(_proc_detail_lines(pr))
            if len(matches) > 80:
                lines.append(f" - <trimmed {len(matches) - 80} more matching processes>")
    except Exception as e:
        lines.append(f"psutil process scan unavailable: {e}")
        if platform.system().lower().startswith("win"):
            try:
                out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=8)
                raw = out.stdout or out.stderr or ""
                for line in raw.splitlines():
                    low = line.lower()
                    if any(tok in low for tok in known_tokens):
                        lines.append(" - " + _safe_text(line, 1000))
            except Exception as e2:
                lines.append(f"tasklist fallback failed: {e2}")
    return lines


def _python_modules_report_lines() -> list[str]:
    lines: list[str] = []
    lines.append("[Deep Python module log]")
    try:
        items = sorted(sys.modules.items(), key=lambda kv: kv[0].lower())
        lines.append(f"Loaded modules: {len(items)}")
        for name, mod in items:
            try:
                path = getattr(mod, "__file__", "") or "<built-in/namespace>"
            except Exception:
                path = "<unknown>"
            lines.append(f" - {name}: {_safe_text(path, 1200)}")
    except Exception as e:
        lines.append(f"Python module scan failed: {e}")
    return lines


def _tail_text_file(path: str, max_lines: int = 800, max_bytes: int = 2_000_000) -> list[str]:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(max(0, size - max_bytes))
            data = f.read()
        text = data.decode("utf-8", "replace")
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
            lines.insert(0, f"<tail only: last {max_lines} lines>")
        return [_safe_text(x, 3000) for x in lines]
    except Exception as e:
        return [f"<failed to read tail: {e}>"]



def _recent_error_summary_report_lines() -> list[str]:
    lines: list[str] = []
    lines.append("[Recent error summary]")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        logs = []
        for name in os.listdir(LOG_DIR):
            p = os.path.join(LOG_DIR, name)
            try:
                if os.path.isfile(p) and name.lower().endswith(".log"):
                    logs.append((os.path.getmtime(p), p))
            except Exception:
                continue
        logs.sort(reverse=True)
        if not logs:
            lines.append("No .log files found.")
            return lines

        # Common real-problem markers. Keep this summary smaller than the full log tail.
        markers = (
            "traceback", "exception", "error", "failed", "failure",
            "runtimeerror", "nameerror", "importerror", "modulenotfounderror",
            "oserror", "winerror", "cuda out of memory", "outofmemory",
            "return code", "killed", "crash", "fatal",
        )
        ignore_markers = (
            "0 error", "errors=0", "no error", "no errors",
            "selected:", "recent error summary",
            "framevision app report",
            "error summary failed",
        )

        total_hits = 0
        ignored_success = 0
        ignored_cancel = 0
        max_hits_total = 80
        max_hits_per_file = 18

        def _exit_code_from_line(line: str):
            try:
                low = str(line).strip().lower()
                if not low.startswith("exit code:"):
                    return None
                raw = low.split(":", 1)[1].strip().split()[0]
                return int(raw)
            except Exception:
                return None

        def _is_ignored_exit_line(line: str) -> bool:
            nonlocal ignored_success, ignored_cancel
            code = _exit_code_from_line(line)
            if code == 0:
                ignored_success += 1
                return True
            if code == 130:
                ignored_cancel += 1
                return True
            return False

        for _mtime, path in logs[:24]:
            if total_hits >= max_hits_total:
                break
            try:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
                size = os.path.getsize(path)
            except Exception:
                stamp = "unknown"
                size = 0

            tail_lines = _tail_text_file(path, max_lines=1200, max_bytes=2_500_000)
            hits = []
            used_indexes = set()
            file_seen_signatures = set()

            for idx, line in enumerate(tail_lines):
                low = str(line).lower()
                if any(ig in low for ig in ignore_markers):
                    continue

                # Ignore normal success and user-cancel/interrupted command results.
                code = _exit_code_from_line(line)
                if code in (0, 130):
                    _is_ignored_exit_line(line)
                    continue

                has_marker = any(m in low for m in markers)
                if code is not None and code not in (0, 130):
                    has_marker = True
                if not has_marker:
                    continue

                # Keep traceback blocks useful: include a few surrounding lines and following stack lines.
                start = max(0, idx - 2)
                end = min(len(tail_lines), idx + 10)
                if "traceback" in low:
                    end = min(len(tail_lines), idx + 18)

                block = []
                for j in range(start, end):
                    if j in used_indexes:
                        continue
                    item = str(tail_lines[j])
                    if _is_ignored_exit_line(item):
                        continue
                    used_indexes.add(j)
                    block.append(_safe_text(item, 2200))

                if not block:
                    continue

                # Avoid repeating the same traceback/error block from nearby marker lines.
                signature = " | ".join(block[-4:]).lower()
                if signature in file_seen_signatures:
                    continue
                file_seen_signatures.add(signature)

                hits.append((idx + 1, block))
                if len(hits) >= max_hits_per_file:
                    break

            if not hits:
                continue

            lines.append("")
            lines.append(f"--- {_safe_text(path)} | modified={stamp} | size={size} bytes | hits={len(hits)} ---")
            for line_no, block in hits:
                if total_hits >= max_hits_total:
                    break
                lines.append(f"<hit near tail line {line_no}>")
                for item in block:
                    lines.append("  " + item)
                total_hits += 1

        lines.insert(1, f"Hits shown: {total_hits} | ignored EXIT CODE 0: {ignored_success} | ignored EXIT CODE 130: {ignored_cancel}")
        if total_hits <= 0:
            lines.append("No obvious recent errors found in recent log tails.")
        elif total_hits >= max_hits_total:
            lines.append(f"<trimmed after {max_hits_total} error hit(s)>")
    except Exception as e:
        lines.append(f"Recent error summary failed: {e}")
    return lines


def _recent_log_tail_report_lines() -> list[str]:
    lines: list[str] = []
    lines.append("[Full recent log tail]")
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        logs = []
        for name in os.listdir(LOG_DIR):
            p = os.path.join(LOG_DIR, name)
            try:
                if os.path.isfile(p) and name.lower().endswith(".log"):
                    logs.append((os.path.getmtime(p), p))
            except Exception:
                continue
        logs.sort(reverse=True)
        if not logs:
            lines.append("No .log files found.")
            return lines
        for _mtime, path in logs[:20]:
            try:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
                size = os.path.getsize(path)
            except Exception:
                stamp = "unknown"
                size = 0
            lines.append("")
            lines.append(f"--- {_safe_text(path)} | modified={stamp} | size={size} bytes ---")
            lines.extend(_tail_text_file(path))
        if len(logs) > 20:
            lines.append(f"<trimmed {len(logs) - 20} older log file(s)>")
    except Exception as e:
        lines.append(f"Recent log tail failed: {e}")
    return lines


def append_app_report(
    *,
    system_snapshot: bool = True,
    running_processes: bool = True,
    resource_snapshot: bool = False,
    internal_stack_sampler: bool = False,
    qt_event_sampler: bool = False,
    recent_error_summary: bool = True,
    deep_python_modules: bool = False,
    full_recent_log_tail: bool = False,
) -> str:
    """Append the selected manual report sections to logs/framevision.log."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 88)
    lines.append(f"FrameVision App Report — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 88)
    lines.append(f"Selected: system_snapshot={bool(system_snapshot)}, running_processes={bool(running_processes)}, resource_usage_report={bool(resource_snapshot)}, recent_error_summary={bool(recent_error_summary)}, deep_python_modules={bool(deep_python_modules)}, full_recent_log_tail={bool(full_recent_log_tail)}")
    lines.append("")

    if system_snapshot:
        lines.extend(_system_snapshot_report_lines())
        lines.append("")
    if running_processes:
        lines.extend(_process_report_lines())
        lines.append("")
    if resource_snapshot:
        # One UI checkbox: Resource usage report. Keep all deeper resource evidence behind it.
        lines.extend(_resource_snapshot_report_lines(sample_seconds=0.55, top_n=12, compact=False, internal_stack=True))
        lines.append("")
        lines.extend(_qt_event_activity_report_lines(sample_seconds=2.0, top_n=20))
        lines.append("")
    else:
        # Backwards compatibility for older callers; the Settings UI no longer exposes these separately.
        if internal_stack_sampler:
            lines.extend(_python_thread_stack_report_lines(sample_seconds=0.55, top_n=12, stack_limit=10))
            lines.append("")
        if qt_event_sampler:
            lines.extend(_qt_event_activity_report_lines(sample_seconds=2.0, top_n=20))
            lines.append("")
    if recent_error_summary:
        lines.extend(_recent_error_summary_report_lines())
        lines.append("")
    if deep_python_modules:
        lines.extend(_python_modules_report_lines())
        lines.append("")
    if full_recent_log_tail:
        # Read tails before appending this new report block to avoid duplicating itself.
        lines.extend(_recent_log_tail_report_lines())
        lines.append("")

    lines.append("End FrameVision App Report")
    lines.append("=" * 88)
    _append_report_lines(lines)
    return LOG_FILE

# ----------------- installer (post-start dumps + optional probes) -----------------
def _flag_true(x) -> bool:
    return str(x).lower() in ("1","true","yes","on")

def install_all(main_window=None) -> None:
    try:
        # Optional probes (controlled by env or QSettings)
        s = QSettings(APP_ORG, APP_NAME)
        want_guard = _flag_true(os.environ.get("PROBE_QT_TIMERS", "0")) or _flag_true(s.value("diagnostics_timer_guard","false"))
        want_handles = _flag_true(os.environ.get("PROBE_QT_HANDLES", "0")) or _flag_true(s.value("diagnostics_handle_log","false"))
        want_resources = _flag_true(os.environ.get("PROBE_RESOURCES", "0")) or _flag_true(s.value("diagnostics_resource_probe","false"))
        thr = int(os.environ.get("PROBE_QT_TIMERS_THRESHOLD", s.value("diagnostics_probe_threshold", 4000)))
        period = int(os.environ.get("PROBE_QT_HANDLES_MS", s.value("diagnostics_handle_log_ms", 1000)))
        res_period = int(os.environ.get("PROBE_RESOURCES_MS", s.value("diagnostics_resource_probe_ms", 3000)))
        res_top_n = int(os.environ.get("PROBE_RESOURCES_TOP_N", s.value("diagnostics_resource_probe_top_n", 8)))
        if want_guard:
            _install_timer_guard(threshold=thr)
        if want_handles:
            _start_handle_logger(period)
        if want_resources:
            _start_resource_logger(res_period, res_top_n)

        # Delay dumps a bit to avoid flashing intro
        QTimer.singleShot(1500, log_environment)
        QTimer.singleShot(2000, log_packages)
        QTimer.singleShot(2300, log_ffmpeg)
        QTimer.singleShot(2600, lambda: dump_qsettings("startup"))
        log("Diagnostics installed (post-start dumps).")
    except Exception as e:
        log("Diagnostics install failed:", e)

# Kick installer once after import
try:
    QTimer.singleShot(300, install_all)
except Exception:
    pass
