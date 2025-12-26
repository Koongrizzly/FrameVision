
# helpers/startuplog.py — prints a compact startup diagnostics block
from __future__ import annotations
from pathlib import Path as _Path
import os as _os, shutil as _shutil, subprocess as _subprocess, sys as _sys, re as _re, ctypes as _ctypes, datetime as _dt

def _safe_print(line: str) -> None:
    try:
        print(line)
    except Exception:
        pass

def print_startup_block(root: "str|_Path", app_files: int, mb: float) -> None:
    """Prints the [fv] startup info block.
    Parameters:
        root: project root path (str or Path)
        app_files: total number of files under root
        mb: total size in MB under root
    """
    try:
        # make sure we have a Path
        root = _Path(str(root))
    except Exception:
        try:
            _safe_print(f"[fv] size scan: {root} -> {app_files} files, {mb:.1f} MB")
        finally:
            return

    # add a leading blank line for readability
    _safe_print("")

    # ----- size scan + drive free space -----
    try:
        _root_str = str(root)
        _drive = root.anchor or (root.drive + "\") if root.drive else None
        _free_line = ""
        _disk_line = ""
        try:
            if _drive:
                _usage = _shutil.disk_usage(_drive)
                _total_gb = _usage.total / (1024**3)
                _free_gb  = _usage.free  / (1024**3)
                _used_gb  = (_usage.total - _usage.free) / (1024**3)
                _pct_free = (100.0 * _usage.free / max(1, _usage.total))
                _free_line = f"; free on {_drive}: {_free_gb:.1f} GB ({_pct_free:.0f}% free)"
                _disk_line = f"[fv] disk: {_drive} used {_used_gb:.1f} GB / {_total_gb:.1f} GB ({_pct_free:.0f}% free)"
        except Exception:
            pass
        _safe_print(f"[fv] size scan: {root} -> {app_files:,} files, {mb:.1f} MB{_free_line}")
        if _disk_line:
            _safe_print(_disk_line)
    except Exception:
        # fallback minimal line
        _safe_print(f"[fv] size scan: {root} -> {app_files} files, {mb:.1f} MB")

    # ----- system snapshot -----
    # RAM via GlobalMemoryStatusEx (no external deps)
    try:
        class _MEMSTAT(_ctypes.Structure):
            _fields_ = [
                ("dwLength", _ctypes.c_ulong),
                ("dwMemoryLoad", _ctypes.c_ulong),
                ("ullTotalPhys", _ctypes.c_ulonglong),
                ("ullAvailPhys", _ctypes.c_ulonglong),
                ("ullTotalPageFile", _ctypes.c_ulonglong),
                ("ullAvailPageFile", _ctypes.c_ulonglong),
                ("ullTotalVirtual", _ctypes.c_ulonglong),
                ("ullAvailVirtual", _ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", _ctypes.c_ulonglong),
            ]
        _ms = _MEMSTAT(); _ms.dwLength = _ctypes.sizeof(_MEMSTAT)
        if _ctypes.windll.kernel32.GlobalMemoryStatusEx(_ctypes.byref(_ms)):
            _tot_gb = _ms.ullTotalPhys / (1024**3)
            _fre_gb = _ms.ullAvailPhys / (1024**3)
            _safe_print(f"[fv] RAM: {_tot_gb:.1f} GB total, {_fre_gb:.1f} GB free")
    except Exception:
        pass

    # CPU cores (+ optional AVX2 if trivial)
    try:
        _cores = _os.cpu_count() or 1
        _cpu_line = f"[fv] CPU: {_cores} cores"
        try:
            # PF_AVX2_INSTRUCTIONS_AVAILABLE = 36
            if hasattr(_ctypes, "windll") and _ctypes.windll.kernel32.IsProcessorFeaturePresent(36):
                _cpu_line += " | AVX2: yes"
        except Exception:
            pass
        _safe_print(_cpu_line)
    except Exception:
        pass

    # GPU (NVIDIA) via nvidia-smi if available quickly
    try:
        _res = _subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=0.6
        )
        if _res.returncode == 0 and _res.stdout.strip():
            _drv = _res.stdout.strip().splitlines()[0].strip()
            _safe_print(f"[fv] GPU: NVIDIA detected (driver {_drv})")
    except Exception:
        pass

    # ---- Tools & runtimes ----
    # ffmpeg version and features
    try:
        _ff = _subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=0.7)
        if _ff.stdout:
            _v = "unknown"
            _m = _re.search(r"ffmpeg version\s+([^\s]+)", _ff.stdout, _re.I)
            if _m: _v = _m.group(1)
            _cfg = ""
            _m2 = _re.search(r"configuration:\s*(.+)", _ff.stdout, _re.I)
            if _m2: _cfg = _m2.group(1).lower()
            _nv = ("nvenc" in _cfg) or ("enable-nvenc" in _cfg) or ("--enable-nvenc" in _cfg)
            _vk = ("vulkan" in _cfg) or ("enable-vulkan" in _cfg)
            _safe_print(f"[fv] ffmpeg: {_v} (nvenc: {'yes' if _nv else 'no'}, vulkan: {'yes' if _vk else 'no'})")
    except Exception:
        pass

    # ncnn models count/size; runtime ok if Vulkan DLL present
    try:
        _models = ( root / "models" )
        _files = 0; _bytes = 0
        if _models.exists():
            for _p in _models.rglob("*"):
                if _p.is_file():
                    _files += 1
                    try: _bytes += int(_p.stat().st_size)
                    except Exception: pass
        _gb = _bytes / (1024**3)
        _runtime_ok = False
        try:
            _ctypes.windll.LoadLibrary("vulkan-1.dll")
            _runtime_ok = True
        except Exception:
            _runtime_ok = False
        _safe_print(f"[fv] ncnn: {'runtime ok' if _runtime_ok else 'runtime ?'} | models: {_files} files, {_gb:.1f} GB")
    except Exception:
        pass

    # Python & PySide6 versions
    try:
        import PySide6 as _PySide6
        _py = f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}"
        _qt = getattr(_PySide6, '__version__', 'unknown')
        _safe_print(f"[fv] python: {_py} | PySide6: {_qt}")
    except Exception:
        pass

    # App state (best-effort)
    try:
        # Queue counts if worker provides status (optional)
        try:
            from helpers.worker import JobQueue as _JobQueue
            if hasattr(_JobQueue, "counts"):
                _pend, _run = _JobQueue.counts()
                _safe_print(f"[fv] queue: {_pend} pending, {_run} running")
        except Exception:
            pass
        # Last session (placeholder best-effort)
        try:
            from PySide6.QtCore import QSettings as _QSettings
            _s = _QSettings("FrameVision", "FrameVision")
            _last = _s.value("session/last_restored", "0", type=str)
            if _last and str(_last).isdigit() and int(_last) > 0:
                _safe_print(f"[fv] last session: restored {int(_last)} file")
        except Exception:
            pass
    except Exception:
        pass

    # Crashlog presence (logs/crash.log)
    try:
        _logdir = root / "logs"
        _crash = _logdir / "crash.log"
        if _crash.exists():
            _ts = None
            try: _ts = _dt.datetime.fromtimestamp(_crash.stat().st_mtime)
            except Exception: pass
            if _ts:
                _safe_print(f"[fv] crashlog: last { _ts.strftime('%Y-%m-%d %H:%M') }")
            else:
                _safe_print(f"[fv] crashlog: present")
        else:
            _safe_print("[fv] crashlog: none")
    except Exception:
        pass
