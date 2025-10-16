# helpers/sysmon.py — System monitor with versions, sizes, uptimes, tools check + UI tweaks
from __future__ import annotations
import os, shutil, subprocess, platform, re, time
from typing import Optional, List, Tuple, Dict

from PySide6.QtCore import Qt, QTimer, QSettings, Signal, QObject, QThread, QUrl
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QSizePolicy, QToolButton, QFrame, QStyle, QScrollArea
)
from PySide6.QtGui import QDesktopServices

# Temperature formatter respects config['temp_units']
def _format_temp(celsius_val: int | None) -> str:
    if celsius_val is None:
        return ""
    try:
        from helpers.framevision_app import config
        units = (config.get('temp_units','C') or 'C').upper()
    except Exception:
        units = 'C'
    try:
        c = float(celsius_val)
    except Exception:
        return f"{celsius_val}°C"
    if units == 'F':
        f = int(round((c * 9/5) + 32))
        return f"{f}°F"
    return f"{int(round(c))}°C"


try:
    import psutil  # type: ignore
except Exception:
    psutil = None

_pynvml = None
try:
    import pynvml  # type: ignore
    _pynvml = pynvml
except Exception:
    _pynvml = None

# Optional cpuinfo (nicer CPU brand)
_cpuinfo = None
try:
    import cpuinfo  # type: ignore
    _cpuinfo = cpuinfo
except Exception:
    _cpuinfo = None

ORG="FrameVision"; APP="FrameVision"
ROOT = os.path.dirname(os.path.dirname(__file__))
K_SYSMON_ENABLED = "sysmon_enabled"
K_SYSMON_LOW     = "sysmon_low_impact"
K_HUD_ENABLED    = "hud_enabled"
K_FIRST_RUN_TS   = "first_run_epoch"

def _fmt_pair_gib(a_bytes: float, b_bytes: float) -> str:
    a = a_bytes / (1024**3)
    b = b_bytes / (1024**3)
    return f"{a:.1f}/{b:.0f} GiB"

def _fmt_dur(secs: float) -> str:
    try: s = int(max(0, secs))
    except Exception: return "—"
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m, s   = divmod(rem, 60)
    parts = []
    if d: parts.append(f"{d}d")
    if h or d: parts.append(f"{h}h")
    if m or h or d: parts.append(f"{m:02d}m")
    parts.append(f"{s:02d}s")
    return " ".join(parts)

def _disk_free_total(path: str) -> Tuple[int,int]:
    try:
        if psutil:
            du = psutil.disk_usage(path)
            return int(du.free), int(du.total)
        s = os.statvfs(path)
        free = s.f_bavail * s.f_frsize
        total = s.f_blocks * s.f_frsize
        return int(free), int(total)
    except Exception:
        return (0, 0)

def _get_cpu_percent() -> float:
    try:
        if psutil:
            return float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    return 0.0

def _get_ram_used_total() -> Tuple[int,int]:
    try:
        if psutil:
            vm = psutil.virtual_memory()
            return int(vm.used), int(vm.total)
    except Exception:
        pass
    return (0, 0)

def _read_gpu_info() -> List[Dict]:
    gpus: List[Dict] = []
    if _pynvml:
        try:
            _pynvml.nvmlInit()
            count = _pynvml.nvmlDeviceGetCount()
            for i in range(count):
                h = _pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = _pynvml.nvmlDeviceGetMemoryInfo(h)
                try: util = _pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                except Exception: util = None
                try: temp = _pynvml.nvmlDeviceGetTemperature(h, _pynvml.NVML_TEMPERATURE_GPU)
                except Exception: temp = None
                try: name = _pynvml.nvmlDeviceGetName(h).decode('utf-8','ignore')
                except Exception: name = f"GPU {i}"
                gpus.append({"index": i, "name": name, "used": int(mem.used), "total": int(mem.total),
                             "util": int(util) if util is not None else None,
                             "temp": int(temp) if temp is not None else None})
            _pynvml.nvmlShutdown()
            if gpus: return gpus
        except Exception:
            pass
    try:
        out = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,name",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.DEVNULL, text=True, timeout=1.5)
        for i, line in enumerate(out.strip().splitlines()):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                used = int(float(parts[0]) * 1024**2)
                total = int(float(parts[1]) * 1024**2)
                util = int(parts[2]) if parts[2].isdigit() else None
                temp = int(parts[3]) if parts[3].isdigit() else None
                name = parts[4] if len(parts) > 4 else f"GPU {i}"
                gpus.append({"index": i, "name": name, "used": used, "total": total, "util": util, "temp": temp})
    except Exception:
        pass
    return gpus

# ---- CPU static info (brand + core counts) ----------------------------------
def _read_cpu_brand() -> str:
    # Try python-cpuinfo
    if _cpuinfo:
        try:
            info = _cpuinfo.get_cpu_info()
            name = info.get("brand_raw") or info.get("brand") or ""
            if name: return name
        except Exception:
            pass
    # OS-specific fallbacks
    try:
        sys = platform.system().lower()
        if "windows" in sys:
            try:
                out = subprocess.check_output(["wmic", "cpu", "get", "name"], text=True, timeout=1.5)
                lines = [l.strip() for l in out.splitlines() if l.strip() and "name" not in l.lower()]
                if lines: return lines[0]
            except Exception: pass
        elif "darwin" in sys or "mac" in sys:
            try:
                out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=1.0)
                if out.strip(): return out.strip()
            except Exception: pass
        else:  # Linux
            try:
                with open("/proc/cpuinfo","r",encoding="utf-8",errors="ignore") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":",1)[1].strip()
            except Exception: pass
    except Exception:
        pass
    # Python platform fallback
    name = platform.processor() or platform.uname().processor or ""
    return name or "CPU"

def _core_counts() -> Tuple[int,int]:
    try:
        phys = psutil.cpu_count(logical=False) if psutil else None
        logi = psutil.cpu_count(logical=True) if psutil else None
        return int(phys or 0), int(logi or 0)
    except Exception:
        return (0, 0)

# ---- Versions line ----------------------------------------------------------
def _versions_line() -> str:
    cuda = None; torch_v = None; tv = None; tf = None
    try:
        import torch as _torch  # type: ignore
        torch_v = getattr(_torch, "__version__", None)
        cuda = getattr(_torch.version, "cuda", None)
    except Exception:
        pass
    if not cuda:
        try:
            out = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.DEVNULL, timeout=1.5)
            m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
            if m:
                cuda = m.group(1)
        except Exception:
            pass
    try:
        import torchvision as _tv  # type: ignore
        tv = getattr(_tv, "__version__", None)
    except Exception:
        pass
    try:
        import transformers as _tf  # type: ignore
        tf = getattr(_tf, "__version__", None)
    except Exception:
        pass
    cuda = cuda or "N/A"; torch_v = torch_v or "N/A"; tv = tv or "N/A"; tf = tf or "N/A"
    return f"CUDA: {cuda}  •  Torch: {torch_v}  •  TVision: {tv}  •  Transformers: {tf}"

# ---- Tools ready check ------------------------------------------------------
def _exists_any(paths: List[str]) -> Optional[str]:
    for p in paths:
        try:
            if p and (os.path.isfile(p) or os.path.isdir(p)):
                return p
        except Exception:
            continue
    return None

def _tool_ready_status(models_dir: str) -> Dict[str, Optional[str]]:
    # Executable names
    rife_names = ["rife-ncnn-vulkan.exe", "rife-ncnn-vulkan"]
    realesr_names = ["realesrgan-ncnn-vulkan.exe", "realesrgan-ncnn-vulkan"]
    waifu_names = ["waifu2x-ncnn-vulkan.exe", "waifu2x-ncnn-vulkan"]
    upscayl_names = ["upscayler.exe", "upscaler.exe", "upscayl.exe"]

    # FFmpeg
    ff = shutil.which("ffmpeg") or _exists_any([
        os.path.join(ROOT, "bin", "ffmpeg.exe"),
        os.path.join(ROOT, "ffmpeg.exe"),
        os.path.join(ROOT, "externals", "ffmpeg", "ffmpeg.exe"),
        os.path.join(ROOT, "presets", "bin", "ffmpeg.exe"),
    ])

    # Qwen2 (image model path)
    qwen = _exists_any([
        os.path.join(models_dir, "describe", "default", "qwen2-vl-2b-instruct"),
    ])

    # RIFE (models tree + presets\bin + externals + root/bin)
    rife = None
    for base in [
        os.path.join(models_dir, "rife-ncnn-vulkan"),
        os.path.join(ROOT, "presets", "bin"),
        os.path.join(ROOT, "externals", "rife"),
        os.path.join(ROOT, "bin"),
        ROOT,
    ]:
        for n in rife_names:
            if not rife:
                rife = _exists_any([os.path.join(base, n)])

    # Real-ESRGAN (models tree + externals + root/bin)
    realsr = None
    for base in [
        os.path.join(models_dir, "realesrgan"),
        os.path.join(ROOT, "externals", "realesrgan"),
        os.path.join(ROOT, "bin"),
        ROOT,
    ]:
        for n in realesr_names:
            if not realsr:
                realsr = _exists_any([os.path.join(base, n)])

    # Waifu2x (models tree + externals + root/bin)
    waifu = None
    for base in [
        os.path.join(models_dir, "waifu2x"),
        os.path.join(ROOT, "externals", "waifu2x"),
        os.path.join(ROOT, "bin"),
        ROOT,
    ]:
        for n in waifu_names:
            if not waifu:
                waifu = _exists_any([os.path.join(base, n)])


    # UpScayl(er) — info-only (accept exe OR models folder)
    upscayl = None
    upscayl_bases = [
        os.path.join(models_dir, "upscayl"),
        os.path.join(models_dir, "upscayler"),
        os.path.join(models_dir, "upscaler"),
        os.path.join(ROOT, "externals", "upscayl"),
    ]
    for base in upscayl_bases:
        if upscayl:
            break
        # 1) Prefer an executable in common places
        for n in upscayl_names:
            if not upscayl:
                upscayl = _exists_any([os.path.join(base, n)])
        # 2) Otherwise, treat a populated models folder as "present"
        if not upscayl and os.path.isdir(base):
            has_models = False
            # Typical structure: .../upscayl/models/*
            check_path = os.path.join(base, "models") if os.path.isdir(os.path.join(base, "models")) else base
            try:
                for _r, _d, _f in os.walk(check_path):
                    if _f:
                        has_models = True
                        break
            except Exception:
                pass
            if has_models:
                upscayl = base

    return {"FFmpeg": ff, "Qwen2-VL 2B": qwen, "RIFE": rife, "Real-ESRGAN": realsr, "Waifu2x": waifu, "UpScayl": upscayl}

def _tools_line(status: Dict[str, Optional[str]]) -> str:
    order = ["FFmpeg","Qwen2-VL 2B","RIFE","Real-ESRGAN","Waifu2x","UpScayl"]
    parts = []
    for name in order:
        ok = bool(status.get(name))
        check = "✅" if ok else "❌"
        parts.append(f"{name} {check}")
    return " • ".join(parts)

# ---- HUD visibility helpers --------------------------------------------------
def _apply_hud_visibility(enable: bool):
    app = QApplication.instance()
    if not app: return
    try:
        from PySide6.QtWidgets import QLabel
        for w in app.allWidgets():
            try:
                if isinstance(w, QLabel):
                    t = (w.text() or "")
                    if "GPU" in t and "DDR" in t and "CPU" in t:
                        if enable: w.show()
                        else: w.hide()
            except Exception:
                continue
    except Exception:
        pass

def _load_hud_setting_and_apply():
    s = QSettings(ORG, APP)
    enable = s.value(K_HUD_ENABLED, True)
    if isinstance(enable, bool):
        en = enable
    else:
        try:
            en = str(enable).strip().lower() in ("1","true","yes","on")
        except Exception:
            en = True
    _apply_hud_visibility(bool(en))

# ----------------------------------------------------------------------------

class DirSizerWorker(QObject):
    result = Signal(dict)
    def __init__(self, targets: Dict[str, str]):
        super().__init__()
        self.targets = targets
        self._stop = False
    def stop(self): self._stop = True
    def _dir_size(self, path: str) -> int:
        total = 0
        for root, dirs, files in os.walk(path):
            if self._stop: break
            for f in files:
                try: total += os.path.getsize(os.path.join(root, f))
                except Exception: continue
        return total
    def run(self):
        out = {}
        for key, path in self.targets.items():
            try:
                if path and os.path.isdir(path): out[key] = self._dir_size(path)
                else: out[key] = 0
            except Exception: out[key] = 0
        self.result.emit(out)

class CollapsibleSection(QWidget):
    def __init__(self, title: str, parent: QWidget | None = None, expanded: bool = False):
        super().__init__(parent)
        self._toggle = QToolButton(text=title, checkable=True, checked=expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._toggle.clicked.connect(self._on_toggled)

        self._content = QFrame()
        self._content.setFrameShape(QFrame.NoFrame)
        self._content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._content.setVisible(expanded)
        self._content.setAttribute(Qt.WA_StyledBackground, True)
        self._content.setStyleSheet("background: palette(window);")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0); lay.setSpacing(4)
        lay.addWidget(self._toggle)
        lay.addWidget(self._content)

    def _on_toggled(self, checked: bool):
        self._toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self._content.setVisible(checked)
        self._content.adjustSize()
        self.adjustSize()

    def setContentLayout(self, layout: QVBoxLayout):
        self._content.setLayout(layout)

def _settings_paths() -> Dict[str,str]:
    s = QSettings(ORG, APP)
    models = s.value("models_dir", "", str) or os.path.join(ROOT, "models")
    output = s.value("output_dir", "", str) or os.path.join(ROOT, "output")
    appdir = ROOT
    venvdir = os.path.join(ROOT, ".venv")
    return {"app": appdir, "venv": venvdir, "models": models, "output": output}

def _format_sizes_block(sizes: Dict[str,int]) -> List[str]:
    def line(key, label):
        b = sizes.get(key, 0); g = b/(1024**3)
        return f"{label}: {g:.1f} GiB"
    return [line("app","App folder"), line("venv","Venv"), line("models","Models"), line("output","Output")]

class SysMonPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SysMonPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#SysMonPanel { background: palette(window); }")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        s = QSettings(ORG, APP)
        self.enabled = bool(s.value(K_SYSMON_ENABLED, False))
        self.low_impact = bool(s.value(K_SYSMON_LOW, True))
        self.hud_enabled = bool(s.value(K_HUD_ENABLED, True))

        # First-run timestamp
        first = s.value(K_FIRST_RUN_TS, None)
        now = time.time()
        if first is None:
            s.setValue(K_FIRST_RUN_TS, now)
            self.first_run_ts = now
        else:
            try:
                self.first_run_ts = float(first)
            except Exception:
                self.first_run_ts = now

        self.session_start_ts = now

        self.toggle = QCheckBox("System monitor")
        self.toggle.setChecked(self.enabled)
        self.toggle.setToolTip("<b>System monitor</b><br/>Enable the live system monitor in this panel.")
        self.toggle.toggled.connect(self._on_toggle)

        self.low = QCheckBox("Low-impact refresh")
        self.low.setChecked(self.low_impact)
        self.low.setToolTip("<b>Low-impact refresh</b><br/>Refresh every ≈2.5s to reduce overhead.")
        self.low.toggled.connect(self._on_low)

        self.hud = QCheckBox("HUD on/off")
        self.hud.setChecked(self.hud_enabled)
        self.hud.setToolTip("<b>HUD on/off</b><br/>Show or hide the usage HUD above the tabs.")
        self.hud.toggled.connect(self._on_hud)

        top = QHBoxLayout(); top.setContentsMargins(0,0,0,0); top.setSpacing(12)
        top.addWidget(self.toggle); top.addWidget(self.low); top.addWidget(self.hud); top.addStretch(1)

        # --- Static CPU info (brand + cores) ---
        brand = _read_cpu_brand()
        phys, logi = _core_counts()
        cpu_info_line = f"CPU: {brand}  —  Cores: {phys or '?'} physical / {logi or '?'} threads"

        self.lbl_cpu_info = QLabel(cpu_info_line); self.lbl_cpu_info.setWordWrap(True)
        # Live numbers
        self.lbl_cpu = QLabel("CPU load: — %"); self.lbl_cpu.setTextFormat(Qt.PlainText)
        self.lbl_ram = QLabel("DDR: —/— GiB");   self.lbl_ram.setTextFormat(Qt.PlainText)

        self.gpu_box = QVBoxLayout(); self.gpu_box.setSpacing(2); self.gpu_box.setContentsMargins(0,0,0,0)
        self.lbl_gpu_load = QLabel("GPU load: — %"); self.lbl_gpu_load.setTextFormat(Qt.PlainText)

        # Versions + key libs on one line
        self.lbl_versions = QLabel(_versions_line())
        self.lbl_versions.setTextFormat(Qt.PlainText)

        # Tools ready line
        paths = _settings_paths()
        self._tools_status = _tool_ready_status(paths["models"])
        self.lbl_tools = QLabel(" " + _tools_line(self._tools_status))
        self.lbl_tools.setTextFormat(Qt.PlainText)
        self.lbl_tools.setToolTip("Checks FFmpeg, Qwen2-VL (describe), RIFE, Real-ESRGAN, Waifu2x, UpScayl(er)")

        # Uptime lines
        self.lbl_uptime_session = QLabel("Time online (this session): —")
        self.lbl_uptime_total = QLabel("Time online (since first run): —")

        self.lbl_disk = QLabel("Available space on installation disk: —/— GiB free"); self.lbl_disk.setTextFormat(Qt.PlainText)

        self.lbl_sizes = QLabel("App/Venv/Models/Output sizes: scanning…"); self.lbl_sizes.setWordWrap(True)

        self.btn_rescan = QPushButton("Rescan")
        self.btn_rescan.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        self.btn_rescan.setToolTip("<b>Rescan</b><br/>Recalculate folder sizes and refresh tools check & disk space.")
        self.btn_rescan.setMinimumWidth(100); self.btn_rescan.setMinimumHeight(22)
        self.btn_rescan.setAutoDefault(False); self.btn_rescan.setFlat(False)
        self.btn_rescan.clicked.connect(self._kick_dir_scan)
        self.btn_rescan.clicked.connect(self._refresh_tools)
        self.btn_rescan.clicked.connect(self._refresh_disk)

        # Reset session timer button
        self.btn_reset = QPushButton("Reset session timer")
        self.btn_reset.setToolTip("Reset the 'Time online (this session)' counter to 0.")
        self.btn_reset.setMinimumHeight(22)
        self.btn_reset.clicked.connect(self._reset_session_timer)

        # Bug report button
        self.btn_bug = QPushButton("Bug report")
        try:
            self.btn_bug.setIcon(self.style().standardIcon(QStyle.SP_MessageBoxWarning))
        except Exception:
            pass
        self.btn_bug.setToolTip("<b>Bug report</b><br/>Open a window to gather logs and type a report.")
        self.btn_bug.setMinimumWidth(120); self.btn_bug.setMinimumHeight(22)
        self.btn_bug.setAutoDefault(False); self.btn_bug.setFlat(False)
        def _open_bug():
            try:
                from helpers.bug_reporter import open_bug_reporter
                open_bug_reporter(self)
            except Exception as e:
                try:
                    from PySide6.QtWidgets import QMessageBox
                    QMessageBox.information(self, "Bug report", f"Bug reporter not available:\n{e}")
                except Exception:
                    pass
        self.btn_bug.clicked.connect(_open_bug)

        # Social links row buttons
        self.btn_gh = QPushButton("GitHub", self)
        try: self.btn_gh.setIcon(self.style().standardIcon(QStyle.SP_DirLinkIcon))
        except Exception: pass
        self.btn_gh.setToolTip("Open project GitHub (Koongrizzly)")
        self.btn_gh.setMinimumWidth(120); self.btn_gh.setMinimumHeight(24)
        self.btn_gh.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://github.com/Koongrizzly")))

        row2 = QHBoxLayout(); row2.setContentsMargins(0,0,0,0); row2.setSpacing(8)
        row2.addWidget(self.btn_gh); row2.addStretch(1)
        row2.addWidget(self.btn_rescan); row2.addWidget(self.btn_reset); row2.addWidget(self.btn_bug)

        v = QVBoxLayout(self); v.setContentsMargins(8,8,8,8); v.setSpacing(6)
        v.addLayout(top)
        v.addWidget(self.lbl_cpu_info)
        v.addWidget(self.lbl_cpu)
        v.addWidget(self.lbl_ram)
        self.gpu_container = QWidget(); self.gpu_container.setLayout(self.gpu_box)
        v.addWidget(self.gpu_container)
        v.addWidget(self.lbl_gpu_load)
        v.addWidget(self.lbl_versions)
        v.addWidget(self.lbl_tools)
        v.addWidget(self.lbl_uptime_session)
        v.addWidget(self.lbl_uptime_total)
        v.addWidget(self.lbl_disk)
        v.addWidget(self.lbl_sizes)
        v.addLayout(row2)

        self.seconds_timer = QTimer(self)
        self.seconds_timer.setTimerType(Qt.PreciseTimer)
        self.seconds_timer.setInterval(1000)
        self.seconds_timer.timeout.connect(self._tick_seconds)
        self.seconds_timer.start()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer); self.timer.timeout.connect(self._tick)
        self._apply_interval()
        if self.enabled: self.timer.start()

        self._dir_thread: Optional[QThread] = None
        self._dir_worker: Optional[DirSizerWorker] = None
        self._last_sizes: Dict[str,int] = {}
        if self.enabled: self._kick_dir_scan()

        QTimer.singleShot(0, _load_hud_setting_and_apply)

    def _apply_interval(self):
        self.timer.setInterval(2500 if self.low_impact else  1000)

    def _on_low(self, b: bool):
        self.low_impact = bool(b)
        QSettings(ORG, APP).setValue(K_SYSMON_LOW, self.low_impact)
        self._apply_interval()

    def _on_toggle(self, b: bool):
        self.enabled = bool(b)
        QSettings(ORG, APP).setValue(K_SYSMON_ENABLED, self.enabled)
        if self.enabled:
            self.session_start_ts = time.time()
            self.timer.start(); self._kick_dir_scan(); self._refresh_tools(); self._refresh_disk()
        else:
            self.timer.stop()

    def _on_hud(self, b: bool):
        self.hud_enabled = bool(b)
        QSettings(ORG, APP).setValue(K_HUD_ENABLED, self.hud_enabled)
        _apply_hud_visibility(self.hud_enabled)

    def _kick_dir_scan(self):
        if self._dir_thread:
            try:
                self._dir_worker.stop(); self._dir_thread.quit(); self._dir_thread.wait(50)
            except Exception: pass
        paths = _settings_paths()
        self.lbl_sizes.setText("App/Venv/Models/Output sizes: scanning…")
        self._dir_thread = QThread(self)
        self._dir_worker = DirSizerWorker(paths)
        self._dir_worker.moveToThread(self._dir_thread)
        self._dir_thread.started.connect(self._dir_worker.run)
        self._dir_worker.result.connect(self._on_sizes_ready)
        self._dir_thread.start()

    def _on_sizes_ready(self, sizes: Dict[str,int]):
        self._last_sizes = sizes or {}
        self.lbl_sizes.setText(" • ".join(_format_sizes_block(self._last_sizes)))
        try:
            self._dir_thread.quit(); self._dir_thread.wait(50)
        except Exception: pass
        self._dir_thread = None; self._dir_worker = None

    def _refresh_tools(self):
        models_dir = _settings_paths()["models"]
        self._tools_status = _tool_ready_status(models_dir)
        self.lbl_tools.setText(" " + _tools_line(self._tools_status))

    def _refresh_disk(self):
        free_d, total_d = _disk_free_total(ROOT)
        self.lbl_disk.setText(f"Available space on installation disk: {_fmt_pair_gib(free_d, total_d)} free")

    def _reset_session_timer(self):
        self.session_start_ts = time.time()
        self.lbl_uptime_session.setText("Time online (this session): 0m")

    def _tick_seconds(self):
        if not self.enabled: return
        now = time.time()
        try: self.lbl_uptime_session.setText(f"Time online (this session): {_fmt_dur(now - self.session_start_ts)}")
        except Exception: pass
        try: self.lbl_uptime_total.setText(f"Time online (since first run): {_fmt_dur(now - self.first_run_ts)}")
        except Exception: pass

    def _tick(self):
        if not self.enabled: return
        cpu = _get_cpu_percent(); self.lbl_cpu.setText(f"CPU load: {cpu:.0f}%")
        used, total = _get_ram_used_total(); self.lbl_ram.setText(f"DDR: {used/(1024**3):.1f}/{total/(1024**3):.0f} GiB")
        while self.gpu_box.count():
            it = self.gpu_box.takeAt(0); w = it.widget()
            if w: w.deleteLater()
        gpus = _read_gpu_info()
        if gpus:
            for g in gpus:
                used = g.get("used",0); total = g.get("total",0)
                util = g.get("util", None); temp = g.get("temp", None); name = g.get("name", "GPU")
                txt = f"{name}: VRAM {used/(1024**3):.1f}/{total/(1024**3):.0f} GiB"
                extras = []
                if util is not None: extras.append(f"{util}%")
                if temp is not None: extras.append(_format_temp(temp))
                if extras: txt += "  (" + ", ".join(extras) + ")"
                lab = QLabel(txt); lab.setTextFormat(Qt.PlainText)
                self.gpu_box.addWidget(lab)
        else:
            self.gpu_box.addWidget(QLabel("GPU: N/A"))
        try:
            utils = [g.get("util") for g in (gpus or []) if isinstance(g, dict)]
            util_val = None
            for u in utils:
                if isinstance(u, (int, float)):
                    util_val = max(util_val if util_val is not None else 0, int(u))
            self.lbl_gpu_load.setText("GPU load: — %" if util_val is None else f"GPU load: {util_val}%")
        except Exception:
            self.lbl_gpu_load.setText("GPU load: — %")

        # Disk
        self._refresh_disk()

        # Uptimes
        now = time.time()

        self._tick_seconds()
# ---------- Settings installer ----------
def _find_settings_content() -> Optional[QWidget]:
    app = QApplication.instance()
    if not app: return None
    for w in app.allWidgets():
        try:
            if w.objectName() == "FvSettingsContent":
                return w
        except Exception: continue
    try:
        from PySide6.QtWidgets import QTabWidget
        for tl in app.topLevelWidgets():
            for tw in tl.findChildren(QTabWidget):
                for i in range(tw.count()):
                    try:
                        if tw.tabText(i).strip().lower() == "settings":
                            page = tw.widget(i)
                            sa = page.findChild(QScrollArea)
                            if sa and sa.widget(): return sa.widget()
                            return page
                    except Exception: continue
    except Exception: pass
    return None

def auto_install_sysmon_settings(retries: int = 14, delay_ms:int = 350):
    content = _find_settings_content()
    if content is None:
        if retries <= 0: return
        QTimer.singleShot(delay_ms, lambda: auto_install_sysmon_settings(retries-1, delay_ms))
        return
    if getattr(content, "_fv_sysmon_wired_v14", False):
        return
    content._fv_sysmon_wired_v14 = True

    section = CollapsibleSection("System monitor", content, expanded=False)
    inner = QVBoxLayout(); inner.setContentsMargins(8,8,8,8); inner.setSpacing(6)
    inner.addWidget(SysMonPanel(section))
    section.setContentLayout(inner)

    lay = getattr(content, "layout", lambda: None)()
    if lay:
        lay.insertWidget(0, section)
    else:
        section.setParent(content); section.show()

    QTimer.singleShot(0, _load_hud_setting_and_apply)