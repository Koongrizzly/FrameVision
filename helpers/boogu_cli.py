#!/usr/bin/env python3
# helpers/boogu_cli.py
#
# FrameVision Boogu Image sd-cli runner + VRAM/RAM/pagefile logger.
#
# This wrapper intentionally does not modify stable-diffusion.cpp internals.
# sd-cli is a separate C++ process, so Python forward hooks cannot see its
# internal blocks. Instead this runner records the exact command, model files,
# sd-cli output, GPU/process memory, system RAM, pagefile pressure, and phase
# transitions in one timestamped live report.
#
# Usage from Boogu UI:
#   python helpers/boogu_cli.py --job <root>/logs/boogu_jobs/boogu_job_....json
#
# The report is written while the run is still active:
#   <FrameVision root>/logs/boogu_vram_report_YYYYMMDD_HHMMSS.txt

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _now_line() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fmt_bytes(value: Any) -> str:
    try:
        n = int(float(value or 0))
    except Exception:
        n = 0
    units = ("B", "KB", "MB", "GB", "TB")
    v = float(n)
    for unit in units:
        if abs(v) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(v)} B"
            return f"{v:.2f} {unit}"
        v /= 1024.0
    return f"{n} B"


def _gb(value: Any) -> float:
    try:
        return float(value or 0) / (1024.0 ** 3)
    except Exception:
        return 0.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def _quote_cmd(cmd: Iterable[str]) -> str:
    items = list(cmd)
    if os.name == "nt":
        try:
            return subprocess.list2cmdline(items)
        except Exception:
            pass
    return " ".join(shlex.quote(str(x)) for x in items)


def _find_app_root(job: Mapping[str, Any]) -> Path:
    root = str(job.get("app_root") or "").strip()
    if root:
        return Path(root).resolve()
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "helpers":
        return here.parent.parent.resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "presets").exists() or (cwd / "models").exists():
        return cwd
    return here.parent.parent.resolve()


def _resolve_path(path_text: str, root: Path) -> Path:
    p = Path(str(path_text or "").strip().strip('"'))
    if not p.is_absolute():
        p = root / p
    return p


def _try_import_psutil():
    try:
        import psutil  # type: ignore
        return psutil
    except Exception:
        return None


class LiveReport:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8", buffering=1)
        self._lock = threading.RLock()

    def write(self, line: str = "") -> None:
        with self._lock:
            self._fh.write(str(line).rstrip("\n") + "\n")
            self._fh.flush()

    def section(self, title: str) -> None:
        self.write("")
        self.write("=" * 88)
        self.write(str(title))
        self.write("=" * 88)

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
            finally:
                self._fh.close()


class GpuProbe:
    """Low-dependency NVIDIA sampler.

    Uses nvidia-smi because this helper must work from FrameVision's normal Python
    environment without requiring pynvml. If nvidia-smi is unavailable, the report
    still contains RAM/process/sd-cli output.
    """

    def __init__(self) -> None:
        self.available = False
        self.error = ""
        self._checked = False

    def _run(self, args: List[str], timeout: float = 2.5) -> str:
        out = subprocess.check_output(["nvidia-smi"] + args, stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", errors="replace").strip()

    def check(self) -> bool:
        if self._checked:
            return self.available
        self._checked = True
        try:
            self._run(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], timeout=2.5)
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = f"{type(exc).__name__}: {exc}"
        return self.available

    def gpu_static(self) -> Dict[str, Any]:
        if not self.check():
            return {"available": False, "error": self.error}
        try:
            text = self._run(["--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"])
            first = text.splitlines()[0] if text else ""
            parts = [p.strip() for p in first.split(",")]
            return {
                "available": True,
                "index": parts[0] if len(parts) > 0 else "0",
                "name": parts[1] if len(parts) > 1 else "unknown",
                "driver": parts[2] if len(parts) > 2 else "unknown",
                "memory_total_mib": _safe_int(parts[3], 0) if len(parts) > 3 else 0,
            }
        except Exception as exc:
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def sample(self, pid: Optional[int] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "gpu_available": False,
            "gpu_used_mib": 0,
            "gpu_free_mib": 0,
            "gpu_total_mib": 0,
            "gpu_util_pct": 0,
            "gpu_proc_used_mib": 0,
            "gpu_proc_found": False,
            "gpu_error": "",
        }
        if not self.check():
            data["gpu_error"] = self.error
            return data
        try:
            text = self._run([
                "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ], timeout=2.5)
            first = text.splitlines()[0] if text else ""
            parts = [p.strip() for p in first.split(",")]
            data.update({
                "gpu_available": True,
                "gpu_used_mib": _safe_int(parts[0], 0) if len(parts) > 0 else 0,
                "gpu_free_mib": _safe_int(parts[1], 0) if len(parts) > 1 else 0,
                "gpu_total_mib": _safe_int(parts[2], 0) if len(parts) > 2 else 0,
                "gpu_util_pct": _safe_int(parts[3], 0) if len(parts) > 3 else 0,
            })
        except Exception as exc:
            data["gpu_error"] = f"gpu query failed: {type(exc).__name__}: {exc}"

        if pid:
            try:
                text = self._run([
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ], timeout=2.5)
                total = 0
                found = False
                for line in text.splitlines():
                    row = [x.strip() for x in line.split(",")]
                    if len(row) >= 2 and _safe_int(row[0], -1) == int(pid):
                        total += _safe_int(row[1], 0)
                        found = True
                data["gpu_proc_used_mib"] = total
                data["gpu_proc_found"] = found
            except Exception:
                # Some drivers return an error when no compute process exists yet.
                pass
        return data


class RuntimeMonitor:
    def __init__(self, report: LiveReport, pid: int, sample_interval: float = 1.0) -> None:
        self.report = report
        self.pid = int(pid)
        self.interval = max(0.25, float(sample_interval or 1.0))
        self.psutil = _try_import_psutil()
        self.gpu = GpuProbe()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.phase = "process_start"
        self.t0 = time.perf_counter()
        self.samples = 0
        self.start_snapshot: Dict[str, Any] = {}
        self.peaks: Dict[str, Any] = {
            "gpu_used_mib": 0,
            "gpu_proc_used_mib": 0,
            "gpu_util_pct": 0,
            "lowest_gpu_free_mib": 0,
            "ram_used_bytes": 0,
            "ram_percent": 0.0,
            "swap_used_bytes": 0,
            "swap_percent": 0.0,
            "proc_rss_bytes": 0,
            "proc_vms_bytes": 0,
            "lowest_ram_available_bytes": 0,
        }

    def set_phase(self, phase: str) -> None:
        if phase and phase != self.phase:
            self.phase = phase
            self.report.write(f"[{_now_line()}] [PHASE] {phase}")

    def _process_memory(self) -> Dict[str, Any]:
        data = {
            "proc_alive": False,
            "proc_rss_bytes": 0,
            "proc_vms_bytes": 0,
            "proc_threads": 0,
            "proc_children": 0,
        }
        if not self.psutil:
            return data
        try:
            proc = self.psutil.Process(self.pid)
            info = proc.memory_info()
            data.update({
                "proc_alive": proc.is_running(),
                "proc_rss_bytes": int(getattr(info, "rss", 0) or 0),
                "proc_vms_bytes": int(getattr(info, "vms", 0) or 0),
                "proc_threads": int(proc.num_threads()),
                "proc_children": len(proc.children(recursive=True)),
            })
        except Exception:
            pass
        return data

    def _system_memory(self) -> Dict[str, Any]:
        data = {
            "ram_total_bytes": 0,
            "ram_available_bytes": 0,
            "ram_used_bytes": 0,
            "ram_percent": 0.0,
            "swap_total_bytes": 0,
            "swap_used_bytes": 0,
            "swap_free_bytes": 0,
            "swap_percent": 0.0,
        }
        if not self.psutil:
            return data
        try:
            vm = self.psutil.virtual_memory()
            sm = self.psutil.swap_memory()
            data.update({
                "ram_total_bytes": int(getattr(vm, "total", 0) or 0),
                "ram_available_bytes": int(getattr(vm, "available", 0) or 0),
                "ram_used_bytes": int(getattr(vm, "used", 0) or 0),
                "ram_percent": float(getattr(vm, "percent", 0.0) or 0.0),
                "swap_total_bytes": int(getattr(sm, "total", 0) or 0),
                "swap_used_bytes": int(getattr(sm, "used", 0) or 0),
                "swap_free_bytes": int(getattr(sm, "free", 0) or 0),
                "swap_percent": float(getattr(sm, "percent", 0.0) or 0.0),
            })
        except Exception:
            pass
        return data

    def _sample(self) -> Dict[str, Any]:
        elapsed = time.perf_counter() - self.t0
        data: Dict[str, Any] = {
            "t_s": elapsed,
            "phase": self.phase,
        }
        data.update(self.gpu.sample(self.pid))
        data.update(self._system_memory())
        data.update(self._process_memory())
        return data

    def _update_peaks(self, s: Mapping[str, Any]) -> None:
        self.peaks["gpu_used_mib"] = max(_safe_int(self.peaks.get("gpu_used_mib")), _safe_int(s.get("gpu_used_mib")))
        self.peaks["gpu_proc_used_mib"] = max(_safe_int(self.peaks.get("gpu_proc_used_mib")), _safe_int(s.get("gpu_proc_used_mib")))
        self.peaks["gpu_util_pct"] = max(_safe_int(self.peaks.get("gpu_util_pct")), _safe_int(s.get("gpu_util_pct")))
        free = _safe_int(s.get("gpu_free_mib"), 0)
        if free > 0:
            old = _safe_int(self.peaks.get("lowest_gpu_free_mib"), 0)
            self.peaks["lowest_gpu_free_mib"] = free if old <= 0 else min(old, free)
        self.peaks["ram_used_bytes"] = max(_safe_int(self.peaks.get("ram_used_bytes")), _safe_int(s.get("ram_used_bytes")))
        self.peaks["ram_percent"] = max(_safe_float(self.peaks.get("ram_percent")), _safe_float(s.get("ram_percent")))
        self.peaks["swap_used_bytes"] = max(_safe_int(self.peaks.get("swap_used_bytes")), _safe_int(s.get("swap_used_bytes")))
        self.peaks["swap_percent"] = max(_safe_float(self.peaks.get("swap_percent")), _safe_float(s.get("swap_percent")))
        self.peaks["proc_rss_bytes"] = max(_safe_int(self.peaks.get("proc_rss_bytes")), _safe_int(s.get("proc_rss_bytes")))
        self.peaks["proc_vms_bytes"] = max(_safe_int(self.peaks.get("proc_vms_bytes")), _safe_int(s.get("proc_vms_bytes")))
        avail = _safe_int(s.get("ram_available_bytes"), 0)
        if avail > 0:
            old = _safe_int(self.peaks.get("lowest_ram_available_bytes"), 0)
            self.peaks["lowest_ram_available_bytes"] = avail if old <= 0 else min(old, avail)

    def _format_sample(self, s: Mapping[str, Any]) -> str:
        return (
            f"[{_now_line()}] [MONITOR] "
            f"t={_safe_float(s.get('t_s')):8.1f}s phase={s.get('phase')} | "
            f"gpu_used={_safe_int(s.get('gpu_used_mib'))} MiB "
            f"gpu_free={_safe_int(s.get('gpu_free_mib'))} MiB "
            f"gpu_proc={_safe_int(s.get('gpu_proc_used_mib'))} MiB "
            f"gpu_util={_safe_int(s.get('gpu_util_pct'))}% | "
            f"ram_used={_fmt_bytes(s.get('ram_used_bytes'))} "
            f"ram_avail={_fmt_bytes(s.get('ram_available_bytes'))} "
            f"ram={_safe_float(s.get('ram_percent')):.1f}% | "
            f"pagefile/swap_used={_fmt_bytes(s.get('swap_used_bytes'))} "
            f"swap={_safe_float(s.get('swap_percent')):.1f}% | "
            f"proc_rss={_fmt_bytes(s.get('proc_rss_bytes'))} "
            f"proc_vms={_fmt_bytes(s.get('proc_vms_bytes'))} "
            f"threads={_safe_int(s.get('proc_threads'))}"
        )

    def start(self) -> None:
        self.start_snapshot = self._sample()
        self._update_peaks(self.start_snapshot)
        self.report.write(self._format_sample(self.start_snapshot))
        self.thread = threading.Thread(target=self._loop, name="boogu-vram-monitor", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval):
            s = self._sample()
            self.samples += 1
            self._update_peaks(s)
            self.report.write(self._format_sample(s))

    def stop(self) -> Dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        final = self._sample()
        self._update_peaks(final)
        self.report.write(self._format_sample(final))
        return final

    def summary_lines(self, final: Mapping[str, Any]) -> List[str]:
        start = self.start_snapshot or {}
        peaks = self.peaks
        swap_delta = _safe_int(peaks.get("swap_used_bytes")) - _safe_int(start.get("swap_used_bytes"))
        ram_pressure = _safe_float(peaks.get("ram_percent")) >= 90.0 or (
            _safe_int(peaks.get("lowest_ram_available_bytes")) > 0
            and _safe_int(peaks.get("lowest_ram_available_bytes")) < 4 * 1024 ** 3
        )
        page_pressure = swap_delta > 512 * 1024 ** 2 or _safe_float(peaks.get("swap_percent")) >= max(80.0, _safe_float(start.get("swap_percent")) + 5.0)
        shared_hint = "unknown"
        if page_pressure:
            shared_hint = "possible/likely pagefile pressure observed"
        elif _safe_int(peaks.get("gpu_used_mib")) > 0 and _safe_int(peaks.get("gpu_proc_used_mib")) > 0:
            shared_hint = "no pagefile pressure signal from this external sampler"

        return [
            f"Samples recorded: {self.samples + 2}",
            f"Peak total GPU used: {_safe_int(peaks.get('gpu_used_mib'))} MiB",
            f"Peak sd-cli process GPU used from nvidia-smi compute-apps: {_safe_int(peaks.get('gpu_proc_used_mib'))} MiB",
            f"Lowest total GPU free: {_safe_int(peaks.get('lowest_gpu_free_mib'))} MiB",
            f"Peak GPU utilization: {_safe_int(peaks.get('gpu_util_pct'))}%",
            f"Peak system RAM used: {_fmt_bytes(peaks.get('ram_used_bytes'))} ({_safe_float(peaks.get('ram_percent')):.1f}%)",
            f"Lowest system RAM available: {_fmt_bytes(peaks.get('lowest_ram_available_bytes'))}",
            f"Peak process RSS: {_fmt_bytes(peaks.get('proc_rss_bytes'))}",
            f"Peak process VMS/commit proxy: {_fmt_bytes(peaks.get('proc_vms_bytes'))}",
            f"Peak pagefile/swap used: {_fmt_bytes(peaks.get('swap_used_bytes'))} ({_safe_float(peaks.get('swap_percent')):.1f}%)",
            f"Pagefile/swap delta during run: {_fmt_bytes(swap_delta)}",
            f"RAM pressure signal: {'YES' if ram_pressure else 'NO'}",
            f"Pagefile pressure signal: {'YES' if page_pressure else 'NO'}",
            f"Shared GPU/pagefile inference: {shared_hint}",
        ]


class PhaseDetector:
    def __init__(self, monitor: RuntimeMonitor, report: LiveReport):
        self.monitor = monitor
        self.report = report
        self.current = "process_start"
        self.events: List[str] = []

    def _set(self, phase: str, reason: str) -> None:
        if phase != self.current:
            self.current = phase
            self.monitor.set_phase(phase)
            event = f"{phase}: {reason}"
            self.events.append(event)
            self.report.write(f"[{_now_line()}] [EVENT] {event}")

    def feed(self, line: str) -> None:
        low = line.lower()
        # Keep these broad. Different sd-cli builds print different text.
        if any(x in low for x in ("load", "loading", "mmap", "gguf", "safetensors", "checkpoint", "tensor")):
            if any(x in low for x in ("vae", "autoencoder")):
                self._set("vae_load_or_decode", line[:220])
            elif any(x in low for x in ("qwen", "llm", "mmproj", "clip", "text encoder", "vision")):
                self._set("text_encoder_or_mmproj_load", line[:220])
            elif any(x in low for x in ("diffusion", "dit", "boogu", "model")):
                self._set("diffusion_model_load", line[:220])
            else:
                self._set("loading", line[:220])
        if any(x in low for x in ("sampling", "sample", "denois", "step ", "steps:", "it/s", "%|")):
            self._set("denoise_sampling", line[:220])
        if any(x in low for x in ("decode", "decoding", "vae")) and not any(x in low for x in ("load", "loading")):
            self._set("vae_decode", line[:220])
        if any(x in low for x in ("save", "saving", "output", ".png")):
            self._set("saving_output", line[:220])
        if any(x in low for x in ("error", "failed", "exception", "traceback", "cuda out of memory", "out of memory")):
            self._set("error_or_warning", line[:220])


class SdCliAllocationStats:
    """Parse sd-cli verbose graph-cut / streaming allocation lines.

    This is the part that matters for Boogu VRAM control. The wrapper cannot see
    C++ tensors directly, but sd-cli verbose output exposes graph-cut segments,
    streamed parameter chunks, compute buffer sizes, and cache buffer sizes.
    """

    GRAPH_RE = re.compile(r"([A-Za-z0-9_]+) graph cut max_vram=([0-9.]+) MB merged (\d+) segments -> (\d+) segments", re.I)
    BUDGET_RE = re.compile(r"([A-Za-z0-9_]+) streaming budget =\s*([0-9.]+) MB", re.I)
    SEGMENT_RE = re.compile(r"([A-Za-z0-9_]+) graph cut executing segment (\d+)/(\d+):\s*(.*?)\s*\(residency=([^\)]+)\)", re.I)
    MMAP_RE = re.compile(r"memory-mapped\s+(\d+)\s+tensors.*?\(([0-9.]+) MB\)", re.I)
    LOAD_RE = re.compile(r"loading\s+(\d+)/(\d+)\s+tensors\s+from\s+(.+)$", re.I)
    STAGED_RE = re.compile(r"staged compute params\s*\(\s*([0-9.]+) MB,\s*(\d+) tensors\)\s*to\s*([^,]+),\s*taking\s*([0-9.]+)s", re.I)
    RELEASE_RE = re.compile(r"releasing compute params\s*\(\s*([0-9.]+) MB,\s*(\d+) tensors\)\s*from\s*(.+)$", re.I)
    COMPUTE_RE = re.compile(r"([A-Za-z0-9_]+) compute buffer size:\s*([0-9.]+) MB\(VRAM\)", re.I)
    CACHE_RE = re.compile(r"([A-Za-z0-9_]+) cache backend buffer size\s*=\s*([0-9.]+) MB\(VRAM\)\s*\((\d+) tensors\)", re.I)
    QWEN_COMPUTE_RE = re.compile(r"qwen3vl compute buffer size:\s*([0-9.]+) MB\(VRAM\)", re.I)

    def __init__(self, report: Optional[LiveReport] = None) -> None:
        self.report = report
        self.graph_cuts: List[Dict[str, Any]] = []
        self.streaming_budgets: List[Dict[str, Any]] = []
        self.segments: List[Dict[str, Any]] = []
        self.current_segment: Optional[Dict[str, Any]] = None
        self.pending_mmap: Optional[Dict[str, Any]] = None
        self.pending_load: Optional[Dict[str, Any]] = None
        self.compute_buffers: List[Dict[str, Any]] = []
        self.cache_buffers: List[Dict[str, Any]] = []
        self.staged_chunks: List[Dict[str, Any]] = []
        self.released_chunks: List[Dict[str, Any]] = []

    def _emit(self, text: str) -> None:
        if self.report is not None:
            self.report.write(f"[{_now_line()}] [ALLOC] {text}")

    def feed(self, line: str) -> None:
        text = str(line or "")
        m = self.GRAPH_RE.search(text)
        if m:
            rec = {
                "model": m.group(1),
                "max_vram_mb": _safe_float(m.group(2)),
                "input_segments": _safe_int(m.group(3)),
                "output_segments": _safe_int(m.group(4)),
            }
            self.graph_cuts.append(rec)
            self._emit(f"graph_cut model={rec['model']} max_vram={rec['max_vram_mb']:.2f} MB merged {rec['input_segments']} -> {rec['output_segments']} segments")
            return

        m = self.BUDGET_RE.search(text)
        if m:
            rec = {"model": m.group(1), "budget_mb": _safe_float(m.group(2))}
            self.streaming_budgets.append(rec)
            self._emit(f"streaming_budget model={rec['model']} budget={rec['budget_mb']:.2f} MB")
            return

        m = self.SEGMENT_RE.search(text)
        if m:
            rec = {
                "model": m.group(1),
                "index": _safe_int(m.group(2)),
                "total": _safe_int(m.group(3)),
                "name": m.group(4).strip(),
                "residency": m.group(5).strip(),
                "mmap_mb": 0.0,
                "mmap_tensors": 0,
                "load_tensors": 0,
                "load_total_tensors": 0,
                "staged_mb": 0.0,
                "staged_tensors": 0,
                "staged_backend": "",
                "staged_seconds": 0.0,
                "compute_buffer_mb": 0.0,
                "cache_buffer_mb": 0.0,
                "cache_tensors": 0,
            }
            self.current_segment = rec
            self.segments.append(rec)
            self._emit(f"segment {rec['index']}/{rec['total']} {rec['name']} residency={rec['residency']}")
            return

        m = self.MMAP_RE.search(text)
        if m:
            rec = {"tensors": _safe_int(m.group(1)), "mb": _safe_float(m.group(2))}
            self.pending_mmap = rec
            if self.current_segment is not None:
                self.current_segment["mmap_tensors"] = rec["tensors"]
                self.current_segment["mmap_mb"] = rec["mb"]
            return

        m = self.LOAD_RE.search(text)
        if m:
            rec = {"load_tensors": _safe_int(m.group(1)), "total_tensors": _safe_int(m.group(2)), "path": m.group(3).strip()}
            self.pending_load = rec
            if self.current_segment is not None:
                self.current_segment["load_tensors"] = rec["load_tensors"]
                self.current_segment["load_total_tensors"] = rec["total_tensors"]
            return

        m = self.STAGED_RE.search(text)
        if m:
            rec = {
                "mb": _safe_float(m.group(1)),
                "tensors": _safe_int(m.group(2)),
                "backend": m.group(3).strip(),
                "seconds": _safe_float(m.group(4)),
                "segment": self.current_segment.get("name") if self.current_segment else "none",
                "segment_index": self.current_segment.get("index") if self.current_segment else 0,
            }
            self.staged_chunks.append(rec)
            if self.current_segment is not None:
                self.current_segment["staged_mb"] = rec["mb"]
                self.current_segment["staged_tensors"] = rec["tensors"]
                self.current_segment["staged_backend"] = rec["backend"]
                self.current_segment["staged_seconds"] = rec["seconds"]
            self._emit(f"staged segment={rec['segment_index']} {rec['segment']} params={rec['mb']:.2f} MB tensors={rec['tensors']} backend={rec['backend']} time={rec['seconds']:.2f}s")
            return

        m = self.RELEASE_RE.search(text)
        if m:
            rec = {
                "mb": _safe_float(m.group(1)),
                "tensors": _safe_int(m.group(2)),
                "backend": m.group(3).strip(),
                "segment": self.current_segment.get("name") if self.current_segment else "none",
                "segment_index": self.current_segment.get("index") if self.current_segment else 0,
            }
            self.released_chunks.append(rec)
            return

        m = self.COMPUTE_RE.search(text)
        if m:
            rec = {
                "model": m.group(1),
                "mb": _safe_float(m.group(2)),
                "segment": self.current_segment.get("name") if self.current_segment else "none",
                "segment_index": self.current_segment.get("index") if self.current_segment else 0,
            }
            self.compute_buffers.append(rec)
            if self.current_segment is not None:
                self.current_segment["compute_buffer_mb"] = rec["mb"]
            self._emit(f"compute_buffer segment={rec['segment_index']} {rec['segment']} model={rec['model']} size={rec['mb']:.2f} MB")
            return

        m = self.QWEN_COMPUTE_RE.search(text)
        if m:
            rec = {"model": "qwen3vl", "mb": _safe_float(m.group(1)), "segment": "conditioning", "segment_index": 0}
            self.compute_buffers.append(rec)
            self._emit(f"compute_buffer conditioning model=qwen3vl size={rec['mb']:.2f} MB")
            return

        m = self.CACHE_RE.search(text)
        if m:
            rec = {
                "model": m.group(1),
                "mb": _safe_float(m.group(2)),
                "tensors": _safe_int(m.group(3)),
                "segment": self.current_segment.get("name") if self.current_segment else "none",
                "segment_index": self.current_segment.get("index") if self.current_segment else 0,
            }
            self.cache_buffers.append(rec)
            if self.current_segment is not None:
                self.current_segment["cache_buffer_mb"] = rec["mb"]
                self.current_segment["cache_tensors"] = rec["tensors"]
            return

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        lines.append("sd-cli allocation / graph-cut summary:")
        if not (self.graph_cuts or self.streaming_budgets or self.segments or self.compute_buffers):
            lines.append("  no graph-cut or allocation lines parsed from sd-cli output")
            return lines
        for rec in self.graph_cuts:
            lines.append(
                f"  graph cut: {rec['model']} max_vram={rec['max_vram_mb']:.2f} MB, "
                f"merged {rec['input_segments']} -> {rec['output_segments']} segments"
            )
        for rec in self.streaming_budgets:
            lines.append(f"  streaming budget: {rec['model']} {rec['budget_mb']:.2f} MB")
        if self.segments:
            total = max(_safe_int(x.get("total")) for x in self.segments)
            streamed = sum(1 for x in self.segments if str(x.get("residency", "")).upper() == "STREAMED")
            resident = len(self.segments) - streamed
            lines.append(f"  parsed segments: {len(self.segments)}/{total} (streamed={streamed}, non-streamed={resident})")
        if self.compute_buffers:
            biggest = max(self.compute_buffers, key=lambda x: _safe_float(x.get("mb")))
            lines.append(
                f"  biggest compute buffer: {biggest['mb']:.2f} MB "
                f"model={biggest.get('model')} segment={biggest.get('segment_index')} {biggest.get('segment')}"
            )
        if self.cache_buffers:
            biggest_cache = max(self.cache_buffers, key=lambda x: _safe_float(x.get("mb")))
            lines.append(
                f"  biggest cache backend buffer: {biggest_cache['mb']:.2f} MB "
                f"model={biggest_cache.get('model')} segment={biggest_cache.get('segment_index')} {biggest_cache.get('segment')}"
            )
        if self.staged_chunks:
            biggest_stage = max(self.staged_chunks, key=lambda x: _safe_float(x.get("mb")))
            total_stage = sum(_safe_float(x.get("mb")) for x in self.staged_chunks)
            total_stage_time = sum(_safe_float(x.get("seconds")) for x in self.staged_chunks)
            lines.append(
                f"  biggest streamed/staged params chunk: {biggest_stage['mb']:.2f} MB "
                f"segment={biggest_stage.get('segment_index')} {biggest_stage.get('segment')}"
            )
            lines.append(f"  total staged params across logged chunks: {total_stage:.2f} MB, staging time={total_stage_time:.2f}s")
        if self.segments:
            ranked = sorted(self.segments, key=lambda x: _safe_float(x.get("compute_buffer_mb")), reverse=True)[:10]
            lines.append("  top compute-buffer segments:")
            for rec in ranked:
                lines.append(
                    f"    {rec.get('index')}/{rec.get('total')} {rec.get('name')} | "
                    f"compute={_safe_float(rec.get('compute_buffer_mb')):.2f} MB, "
                    f"params={_safe_float(rec.get('staged_mb')):.2f} MB, "
                    f"cache={_safe_float(rec.get('cache_buffer_mb')):.2f} MB, "
                    f"residency={rec.get('residency')}"
                )
        return lines


def _parse_arg_map(cmd: List[str]) -> Dict[str, List[str]]:
    """Parse sd-cli args for reporting.

    Some sd-cli switches are valueless flags. The old parser treated the next
    short option, such as -v, as the value for the previous flag. Keep an
    explicit flag list so reports show --stream-layers: true instead of
    --stream-layers: -v.
    """
    flag_options = {
        "--diffusion-fa",
        "--fa",
        "--offload-to-cpu",
        "--mmap",
        "--eager-load",
        "--vae-tiling",
        "--stream-layers",
        "--diffusion-conv-direct",
        "--vae-conv-direct",
        "--disable-image-metadata",
        "--increase-ref-index",
        "--disable-auto-resize-ref-image",
        "--canny",
        "--color",
        "--preview-noisy",
        "--metadata-raw",
        "--metadata-brief",
        "--metadata-all",
        "-v",
        "--verbose",
    }
    out: Dict[str, List[str]] = {}
    i = 0
    while i < len(cmd):
        token = str(cmd[i])
        if token in flag_options:
            out.setdefault(token, []).append("true")
            i += 1
        elif token.startswith("--"):
            if i + 1 < len(cmd) and not str(cmd[i + 1]).startswith("-"):
                out.setdefault(token, []).append(str(cmd[i + 1]))
                i += 2
            else:
                out.setdefault(token, []).append("true")
                i += 1
        elif token.startswith("-") and len(token) > 1:
            if i + 1 < len(cmd) and not str(cmd[i + 1]).startswith("-"):
                out.setdefault(token, []).append(str(cmd[i + 1]))
                i += 2
            else:
                out.setdefault(token, []).append("true")
                i += 1
        else:
            i += 1
    return out


def _file_record(label: str, path_text: str, root: Path) -> Dict[str, Any]:
    p = _resolve_path(path_text, root)
    rec: Dict[str, Any] = {
        "label": label,
        "path": str(p),
        "exists": p.exists(),
        "extension": p.suffix.lower(),
        "size_bytes": 0,
        "size": "missing",
        "modified": "missing",
    }
    try:
        if p.exists() and p.is_file():
            st = p.stat()
            rec["size_bytes"] = int(st.st_size)
            rec["size"] = _fmt_bytes(st.st_size)
            rec["modified"] = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


def _write_job_metadata(report: LiveReport, job: Mapping[str, Any], root: Path, cmd: List[str]) -> None:
    argmap = _parse_arg_map(cmd)
    report.section("BOOGU RUN METADATA")
    report.write(f"Report started: {_now_line()}")
    report.write(f"App root: {root}")
    report.write(f"Mode: {job.get('mode', 'unknown')}")
    report.write(f"Python: {sys.executable}")
    report.write(f"Platform: {platform.platform()}")
    report.write(f"Working directory: {os.getcwd()}")
    report.write("")
    report.write("Command:")
    report.write(_quote_cmd(cmd))
    report.write("")
    report.write("Important sd-cli arguments:")
    important = [
        "--diffusion-model", "--llm", "--llm_vision", "--vae", "--vae-format",
        "--width", "--height", "--steps", "--cfg-scale", "--img-cfg-scale",
        "--guidance", "--strength", "--seed", "--batch-count", "--sampling-method",
        "--scheduler", "--rng", "--threads", "--diffusion-fa", "--offload-to-cpu",
        "--max-vram", "--stream-layers", "--backend", "--params-backend",
        "--diffusion-conv-direct", "--vae-conv-direct",
        "--mmap", "--eager-load", "--vae-tiling", "--disable-image-metadata",
        "--output", "--preview", "--preview-path", "--preview-interval", "--ref-image",
    ]
    for key in important:
        if key in argmap:
            report.write(f"  {key}: {', '.join(argmap[key])}")

    report.section("MODEL / FILE INVENTORY")
    files: List[Dict[str, Any]] = []
    if "--diffusion-model" in argmap:
        files.append(_file_record("diffusion_model", argmap["--diffusion-model"][-1], root))
    if "--llm" in argmap:
        files.append(_file_record("qwen_llm", argmap["--llm"][-1], root))
    if "--llm_vision" in argmap:
        files.append(_file_record("vision_mmproj", argmap["--llm_vision"][-1], root))
    if "--vae" in argmap:
        files.append(_file_record("vae", argmap["--vae"][-1], root))
    for idx, ref in enumerate(argmap.get("--ref-image", []), start=1):
        files.append(_file_record(f"reference_image_{idx}", ref, root))
    for rec in files:
        report.write(
            f"{rec['label']}: exists={rec['exists']} ext={rec['extension']} "
            f"size={rec['size']} modified={rec['modified']} path={rec['path']}"
        )
    report.write("")
    report.write("Logger note:")
    report.write(
        "Python forward hooks are not attached here because sd-cli is an external C++ process. "
        "This runner measures exact files, sd-cli output, process memory, NVIDIA memory, RAM, "
        "and pagefile/swap pressure around that process."
    )


def _write_system_static(report: LiveReport) -> None:
    gpu = GpuProbe()
    report.section("SYSTEM BASELINE")
    static = gpu.gpu_static()
    if static.get("available"):
        report.write(
            f"GPU {static.get('index')}: {static.get('name')} | "
            f"driver={static.get('driver')} | total={static.get('memory_total_mib')} MiB"
        )
    else:
        report.write(f"nvidia-smi unavailable: {static.get('error')}")
    psutil = _try_import_psutil()
    if psutil:
        try:
            vm = psutil.virtual_memory()
            sm = psutil.swap_memory()
            report.write(f"System RAM total={_fmt_bytes(vm.total)} available={_fmt_bytes(vm.available)} used={_fmt_bytes(vm.used)} percent={vm.percent:.1f}%")
            report.write(f"Pagefile/swap total={_fmt_bytes(sm.total)} used={_fmt_bytes(sm.used)} free={_fmt_bytes(sm.free)} percent={sm.percent:.1f}%")
        except Exception as exc:
            report.write(f"psutil memory baseline failed: {type(exc).__name__}: {exc}")
    else:
        report.write("psutil not available; process/RAM/pagefile detail will be limited.")


def _stream_process(proc: subprocess.Popen, report: LiveReport, detector: PhaseDetector, allocation: Optional[SdCliAllocationStats] = None) -> None:
    assert proc.stdout is not None
    buffer = ""
    while True:
        chunk = proc.stdout.read(1)
        if chunk == "" and proc.poll() is not None:
            break
        if not chunk:
            time.sleep(0.02)
            continue
        if chunk in ("\n", "\r"):
            line = buffer.strip()
            buffer = ""
            if line:
                report.write(f"[{_now_line()}] [SD-CLI] {line}")
                print(line, flush=True)
                detector.feed(line)
                if allocation is not None:
                    allocation.feed(line)
        else:
            buffer += chunk
            if len(buffer) > 2000:
                line = buffer.strip()
                buffer = ""
                if line:
                    report.write(f"[{_now_line()}] [SD-CLI] {line}")
                    print(line, flush=True)
                    detector.feed(line)
                if allocation is not None:
                    allocation.feed(line)
    if buffer.strip():
        line = buffer.strip()
        report.write(f"[{_now_line()}] [SD-CLI] {line}")
        print(line, flush=True)
        detector.feed(line)
        if allocation is not None:
            allocation.feed(line)



_WINDOWS_JOB_HANDLES: List[Any] = []


def _attach_windows_kill_on_close_job(pid: int, report: LiveReport) -> None:
    """Put sd-cli in a Windows Job Object so stopping the Python wrapper also stops sd-cli.

    QProcess kills the Python wrapper, not automatically its child process. This keeps
    the old UI Stop behavior safe when the UI launches boogu_cli.py instead of sd-cli.exe
    directly.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class LARGE_INTEGER(ctypes.Union):
            _fields_ = [("QuadPart", ctypes.c_longlong)]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", LARGE_INTEGER),
                ("PerJobUserTimeLimit", LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        JobObjectExtendedLimitInformation = 9
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        PROCESS_SET_QUOTA = 0x0100
        PROCESS_TERMINATE = 0x0001

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

        h_job = kernel32.CreateJobObjectW(None, None)
        if not h_job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            h_job,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            err = ctypes.get_last_error()
            kernel32.CloseHandle(h_job)
            raise OSError(err, "SetInformationJobObject failed")

        h_proc = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
        if not h_proc:
            err = ctypes.get_last_error()
            kernel32.CloseHandle(h_job)
            raise OSError(err, "OpenProcess failed")

        ok = kernel32.AssignProcessToJobObject(h_job, h_proc)
        kernel32.CloseHandle(h_proc)
        if not ok:
            err = ctypes.get_last_error()
            kernel32.CloseHandle(h_job)
            raise OSError(err, "AssignProcessToJobObject failed")

        _WINDOWS_JOB_HANDLES.append(h_job)
        report.write(f"[{_now_line()}] [RUNNER] Windows kill-on-close job object attached to sd-cli pid={pid}")
    except Exception as exc:
        report.write(f"[{_now_line()}] [RUNNER-WARNING] Could not attach Windows kill-on-close job object: {type(exc).__name__}: {exc}")


def run_job(job_path: Path, sample_interval: float = 1.0) -> int:
    job = json.loads(job_path.read_text(encoding="utf-8"))
    root = _find_app_root(job)
    cmd = [str(x) for x in job.get("cmd", [])]
    if not cmd:
        raise RuntimeError("Job JSON does not contain cmd list.")

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = _now_stamp()
    report_path = logs_dir / f"boogu_vram_report_{stamp}.txt"

    report = LiveReport(report_path)
    exit_code = 9999
    t0 = time.perf_counter()
    monitor: Optional[RuntimeMonitor] = None
    final_snapshot: Dict[str, Any] = {}
    detector: Optional[PhaseDetector] = None
    allocation: Optional[SdCliAllocationStats] = None

    try:
        _write_job_metadata(report, job, root, cmd)
        _write_system_static(report)

        report.section("LIVE PROCESS OUTPUT AND MEMORY TIMELINE")
        report.write(f"[{_now_line()}] [RUNNER] starting sd-cli")
        report.write(f"[{_now_line()}] [RUNNER] live report path: {report_path}")
        print(f"[Boogu CLI] live report: {report_path}", flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        report.write(f"[{_now_line()}] [RUNNER] sd-cli pid={proc.pid}")
        _attach_windows_kill_on_close_job(proc.pid, report)

        monitor = RuntimeMonitor(report, proc.pid, sample_interval=sample_interval)
        detector = PhaseDetector(monitor, report)
        allocation = SdCliAllocationStats(report)
        monitor.start()

        _stream_process(proc, report, detector, allocation)
        exit_code = int(proc.wait())
        if monitor:
            monitor.set_phase("process_finished")
            final_snapshot = monitor.stop()

        duration = time.perf_counter() - t0
        report.section("SUMMARY")
        report.write(f"Exit code: {exit_code}")
        report.write(f"Total runtime: {duration:.2f} seconds")
        if monitor:
            for line in monitor.summary_lines(final_snapshot):
                report.write(line)
        if allocation:
            report.write("")
            for line in allocation.summary_lines():
                report.write(line)
        if detector:
            report.write("")
            report.write("Detected phase/event transitions:")
            if detector.events:
                for ev in detector.events:
                    report.write(f"  - {ev}")
            else:
                report.write("  none detected from sd-cli output")
        report.write("")
        report.write("Interpretation guide:")
        report.write("- Peak total GPU used is the whole GPU, including other processes.")
        report.write("- Peak sd-cli process GPU is reported by nvidia-smi compute-apps when available.")
        report.write("- Windows shared GPU memory is not directly exposed by nvidia-smi; pagefile/swap and process commit are logged as pressure proxies.")
        report.write("- If pagefile/swap delta grows while GPU is near full, high-resolution Boogu is spilling beyond clean VRAM/RAM headroom.")
        report.write("- For Boogu stream-layers tests, compare streaming budget, segment count, biggest compute buffer, and shared/pagefile pressure. The compute buffer can remain larger than --max-vram.")
        report.write(f"[{_now_line()}] [RUNNER] report complete")
    except KeyboardInterrupt:
        report.write(f"[{_now_line()}] [RUNNER] interrupted by user")
        exit_code = 130
    except Exception as exc:
        report.write(f"[{_now_line()}] [RUNNER-ERROR] {type(exc).__name__}: {exc}")
        exit_code = 1
    finally:
        try:
            if monitor and not monitor.stop_event.is_set():
                final_snapshot = monitor.stop()
        except Exception:
            pass
        try:
            report.close()
        except Exception:
            pass

    print(f"[Boogu CLI] report written: {report_path}", flush=True)
    return exit_code


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FrameVision Boogu sd-cli runner with live VRAM/RAM/pagefile report.")
    parser.add_argument("--job", required=True, help="Path to Boogu job JSON created by boogu_ui.py")
    parser.add_argument("--sample-interval", type=float, default=1.0, help="Monitor sample interval in seconds.")
    args = parser.parse_args(argv)
    return run_job(Path(args.job), sample_interval=args.sample_interval)


if __name__ == "__main__":
    raise SystemExit(main())
