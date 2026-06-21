#!/usr/bin/env python3
"""FrameVision VRAM Lab runtime helpers.

This module is intentionally small and reusable:
- forward-time block hooks for generation/sampling
- post-step decode/finalize guard for the hidden output stage
- boundary tracing for wrappers that must keep model repos untouched

No model repo code is imported here. Model helpers call this module from outside
and pass normal Python objects/modules into it.
"""
from __future__ import annotations

import gc
import os
import re
import time
import threading
import types
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


def _fmt_bytes(value: Any) -> str:
    try:
        n = int(value or 0)
    except Exception:
        n = 0
    units = ("B", "KB", "MB", "GB", "TB")
    v = float(n)
    for unit in units:
        if abs(v) < 1024.0 or unit == units[-1]:
            return f"{v:.2f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024.0
    return f"{n} B"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")




# Shared VRAM Lab residency constants. Keep tuning here so wrappers do not each
# invent their own residency numbers.
#
# 2026-05 profile cleanup v2:
# - User-facing runtime now has one active profile only: safe/off.
# - "balanced", "edge", and "aggressive" are accepted as legacy aliases for safe
#   so old commands do not break, but they no longer select different residency logic.
# - 8/12/16/24 GB tables remain as placeholders for later auto-detection work.
VALID_VRAM_LAB_MODES = {"off", "safe"}
LEGACY_VRAM_LAB_MODE_ALIASES = {
    "balanced": "safe",
    "edge": "safe",
    "aggressive": "safe",
}
VRAM_LAB_RESIDENCY_PROFILES = {
    8: {},   # placeholder for later
    12: {},  # placeholder for later
    16: {},  # placeholder for later
    24: {
        "safe": {
            "hot_window_gb": 13.4,
            "driver_free_floor_gb": 1.5,
            "note": "single 24GB VRAM Lab profile: keep about 13.4 GB of LTX denoiser blocks resident",
        },
    },
}
ACTIVE_VRAM_LAB_PROFILE_GB = 24


def _profile_value(mode: str, key: str, default: float = 0.0) -> float:
    try:
        return float(VRAM_LAB_RESIDENCY_PROFILES[ACTIVE_VRAM_LAB_PROFILE_GB][mode][key])
    except Exception:
        return float(default)


SAFE_HOT_WINDOW_GB = _profile_value("safe", "hot_window_gb", 13.4)
SAFE_DRIVER_FREE_FLOOR_GB = _profile_value("safe", "driver_free_floor_gb", 1.5)
SAFE_HOT_WINDOW_BYTES = int(SAFE_HOT_WINDOW_GB * 1024 ** 3)
SAFE_DRIVER_FREE_FLOOR_BYTES = int(SAFE_DRIVER_FREE_FLOOR_GB * 1024 ** 3)
# Backwards-compatible report names only. Legacy mode names map to the single safe profile.
EDGE_HOT_WINDOW_GB = 0.0
EDGE_HOT_WINDOW_BYTES = 0
EDGE_DRIVER_FREE_FLOOR_GB = SAFE_DRIVER_FREE_FLOOR_GB
EDGE_DRIVER_FREE_FLOOR_BYTES = SAFE_DRIVER_FREE_FLOOR_BYTES
BALANCED_HOT_WINDOW_GB = SAFE_HOT_WINDOW_GB
BALANCED_HOT_WINDOW_BYTES = SAFE_HOT_WINDOW_BYTES
BALANCED_DRIVER_FREE_FLOOR_GB = SAFE_DRIVER_FREE_FLOOR_GB
BALANCED_DRIVER_FREE_FLOOR_BYTES = SAFE_DRIVER_FREE_FLOOR_BYTES
AGGRESSIVE_HOT_WINDOW_GB = 0.0
AGGRESSIVE_HOT_WINDOW_BYTES = 0
AGGRESSIVE_DRIVER_FREE_FLOOR_GB = SAFE_DRIVER_FREE_FLOOR_GB
AGGRESSIVE_DRIVER_FREE_FLOOR_BYTES = SAFE_DRIVER_FREE_FLOOR_BYTES
AGGRESSIVE_EXTRA_GB = 0.0
AGGRESSIVE_EXTRA_BYTES = 0


def normalize_vram_lab_mode(mode: Any, default: str = "safe") -> str:
    default_value = str(default or "safe").lower().strip()
    default_value = LEGACY_VRAM_LAB_MODE_ALIASES.get(default_value, default_value)
    if default_value not in VALID_VRAM_LAB_MODES:
        default_value = "safe"
    value = str(mode or default_value or "safe").lower().strip()
    value = LEGACY_VRAM_LAB_MODE_ALIASES.get(value, value)
    if value not in VALID_VRAM_LAB_MODES:
        return default_value
    return value


def vram_lab_profile_note(mode: Any) -> str:
    mode = normalize_vram_lab_mode(mode, "safe")
    if mode == "safe":
        note = VRAM_LAB_RESIDENCY_PROFILES[ACTIVE_VRAM_LAB_PROFILE_GB]["safe"].get("note")
        return f"{note}; emergency trim below {SAFE_DRIVER_FREE_FLOOR_GB:.1f} GB driver-free"
    return "off: VRAM Lab disabled"


def apply_vram_lab_profile_defaults(policy: Optional[Mapping[str, Any]] = None, mode: Any = None) -> Dict[str, Any]:
    """Return a shared step-phase hook policy for the active 24 GB profile.

    0.7.4 keeps the successful LTX denoiser/X0Model hook target, but the user-facing
    runtime now has one active profile only. Legacy Balanced/Edge/Aggressive inputs
    are accepted as aliases for safe so older wrappers do not fail.
    """
    out = dict(policy or {})
    selected = normalize_vram_lab_mode(mode if mode is not None else out.get("mode", "safe"), "safe")
    out["mode"] = selected
    out["profile_note"] = vram_lab_profile_note(selected)
    out["safe_hot_window_gb"] = SAFE_HOT_WINDOW_GB if selected == "safe" else 0.0
    out["edge_hot_window_gb"] = 0.0
    out["safe_hot_window_bytes"] = SAFE_HOT_WINDOW_BYTES if selected == "safe" else 0
    out["edge_hot_window_bytes"] = 0
    # Legacy report keys preserved only so older wrapper/report code does not break.
    out["balanced_hot_window_gb"] = SAFE_HOT_WINDOW_GB if selected == "safe" else 0.0
    out["balanced_hot_window_bytes"] = SAFE_HOT_WINDOW_BYTES if selected == "safe" else 0
    out["aggressive_extra_gb"] = 0.0
    out["aggressive_extra_bytes"] = 0
    if selected == "safe":
        # Former high-residency baseline; this is not the old crawl-safe/low-VRAM behavior.
        out["release_after_forward"] = False
        out["unload_other_blocks_before_load"] = False
        out.setdefault("synchronize_after_unload", False)
        out.setdefault("empty_cache_after_unload", False)
        out.setdefault("empty_cache_every", 0)
        out["hot_block_budget_bytes"] = SAFE_HOT_WINDOW_BYTES
        out.setdefault("emergency_driver_free_floor_bytes", SAFE_DRIVER_FREE_FLOOR_BYTES)
    else:  # off fallback for callers that only attach when not off
        out.setdefault("release_after_forward", True)
        out.setdefault("unload_other_blocks_before_load", True)
        out.setdefault("synchronize_after_unload", True)
        out.setdefault("empty_cache_after_unload", True)
        out.setdefault("empty_cache_every", 0)
        out.setdefault("hot_block_budget_bytes", 0)
        out.setdefault("emergency_driver_free_floor_bytes", 0)
    return out


def _module_bytes(module: Any) -> int:
    total = 0
    try:
        items = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
    except Exception:
        items = []
    for tensor in items:
        try:
            if getattr(tensor, "is_meta", False):
                continue
            total += int(tensor.numel()) * int(tensor.element_size())
        except Exception:
            pass
    return int(total)


def _module_device(module: Any) -> str:
    try:
        for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
            try:
                if getattr(tensor, "is_meta", False):
                    continue
                return str(tensor.device)
            except Exception:
                pass
    except Exception:
        pass
    return "unknown"


def _module_is_cuda(module: Any) -> bool:
    return _module_device(module).startswith("cuda")


def _safe_to(module: Any, device: str) -> bool:
    if module is None:
        return False
    try:
        if hasattr(module, "to"):
            module.to(device)
            return True
    except Exception:
        return False
    return False


def _torch_cuda_snapshot(torch_mod: Any) -> Dict[str, int]:
    out = {
        "allocated": 0,
        "reserved": 0,
        "max_allocated": 0,
        "max_reserved": 0,
        "driver_free": 0,
        "driver_total": 0,
    }
    try:
        if torch_mod is None or not torch_mod.cuda.is_available():
            return out
        out["allocated"] = int(torch_mod.cuda.memory_allocated())
        out["reserved"] = int(torch_mod.cuda.memory_reserved())
        try:
            out["max_allocated"] = int(torch_mod.cuda.max_memory_allocated())
            out["max_reserved"] = int(torch_mod.cuda.max_memory_reserved())
        except Exception:
            pass
        try:
            free, total = torch_mod.cuda.mem_get_info()
            out["driver_free"] = int(free)
            out["driver_total"] = int(total)
        except Exception:
            try:
                out["driver_total"] = int(torch_mod.cuda.get_device_properties(0).total_memory)
            except Exception:
                pass
    except Exception:
        pass
    return out


def _snapshot_string(snapshot: Mapping[str, int]) -> str:
    return (
        f"allocated={_fmt_bytes(snapshot.get('allocated', 0))}, "
        f"reserved={_fmt_bytes(snapshot.get('reserved', 0))}, "
        f"driver_free={_fmt_bytes(snapshot.get('driver_free', 0))}, "
        f"driver_total={_fmt_bytes(snapshot.get('driver_total', 0))}"
    )

_ACTIVE_STEP_SPEED_PROFILER: Any = None


def _set_active_step_speed_profiler(profiler: Any) -> None:
    global _ACTIVE_STEP_SPEED_PROFILER
    _ACTIVE_STEP_SPEED_PROFILER = profiler


def _get_active_step_speed_profiler() -> Any:
    return _ACTIVE_STEP_SPEED_PROFILER


class StepSpeedProfiler:
    """Low-overhead LTX denoise step profiler.

    It patches tqdm just before the official LTX module runs. The patch only
    profiles progress loops whose total length matches the requested inference
    step count, so loading/finalize progress bars are ignored. VRAM hook timing
    is fed in from VramForwardHookRuntime to split step wall time into block
    load/unload/trim/emergency time versus the remaining compute/kernel path.
    """

    def __init__(
        self,
        ctx: Optional[Dict[str, Any]] = None,
        torch_module: Any = None,
        steps_hint: int = 0,
        echo: bool = False,
        live_path: Optional[str] = None,
        max_step_events: int = 256,
    ) -> None:
        self.ctx = ctx if isinstance(ctx, dict) else {}
        self.torch = torch_module
        self.steps_hint = int(steps_hint or 0)
        self.echo = bool(echo)
        self.live_path = Path(live_path) if live_path else None
        self.max_step_events = int(max_step_events or 256)
        self.enabled = False
        self.current_step_index = -1
        self.current_loop_index = 0
        self.profiled_loop_count = 0
        self._orig_tqdm = None
        self._orig_auto_tqdm = None
        self._live_fh = None
        self._lock = threading.RLock()
        self.steps: List[Dict[str, Any]] = []
        self.failures: List[str] = []
        self.ctx["step_speed_profiler_enabled"] = "NO"
        self.ctx["step_speed_profiler_status"] = "not started"

    def _open_live(self) -> None:
        if self.live_path is None:
            return
        try:
            self.live_path.parent.mkdir(parents=True, exist_ok=True)
            self._live_fh = self.live_path.open("w", encoding="utf-8")
            self._live_fh.write("FrameVision LTX step-speed profiler\n")
            self._live_fh.flush()
        except Exception as exc:
            self.failures.append(f"live log open failed: {type(exc).__name__}: {exc}")
            self._live_fh = None

    def _write_live(self, text: str) -> None:
        if self._live_fh is None:
            return
        try:
            self._live_fh.write(text.rstrip() + "\n")
            self._live_fh.flush()
        except Exception as exc:
            if len(self.failures) < 20:
                self.failures.append(f"live log write failed: {type(exc).__name__}: {exc}")

    def _snapshot(self) -> Dict[str, int]:
        try:
            return _torch_cuda_snapshot(self.torch)
        except Exception:
            return {}

    def _infer_total(self, iterable: Any, kwargs: Mapping[str, Any]) -> int:
        try:
            if kwargs.get("total", None) is not None:
                return int(kwargs.get("total") or 0)
        except Exception:
            pass
        try:
            return int(len(iterable))
        except Exception:
            return 0

    def _should_profile_loop(self, iterable: Any, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
        total = self._infer_total(iterable, kwargs)
        if self.steps_hint > 0 and total == self.steps_hint:
            return True
        desc = str(kwargs.get("desc", "") or "").lower()
        if self.steps_hint <= 0 and any(word in desc for word in ("denois", "infer", "sampling", "sample")):
            return True
        return False

    def start(self) -> None:
        if self.enabled:
            return
        try:
            import tqdm as tqdm_mod  # type: ignore
            self._orig_tqdm = getattr(tqdm_mod, "tqdm")
            parent = self

            def profiled_tqdm(iterable=None, *args, **kwargs):
                # Preserve normal tqdm behavior for non-denoise progress bars.
                if iterable is None or not parent._should_profile_loop(iterable, args, kwargs):
                    return parent._orig_tqdm(iterable, *args, **kwargs)
                bar = parent._orig_tqdm(iterable, *args, **kwargs)

                def iterator():
                    parent.profiled_loop_count += 1
                    loop_index = parent.profiled_loop_count
                    for step_index, item in enumerate(bar):
                        parent.begin_step(loop_index, step_index)
                        try:
                            yield item
                        finally:
                            parent.end_step(loop_index, step_index)

                return iterator()

            setattr(profiled_tqdm, "_framevision_step_speed_profiler", True)
            setattr(tqdm_mod, "tqdm", profiled_tqdm)
            try:
                import tqdm.auto as tqdm_auto_mod  # type: ignore
                self._orig_auto_tqdm = getattr(tqdm_auto_mod, "tqdm", None)
                setattr(tqdm_auto_mod, "tqdm", profiled_tqdm)
            except Exception:
                self._orig_auto_tqdm = None
            self.enabled = True
            _set_active_step_speed_profiler(self)
            self._open_live()
            self.ctx["step_speed_profiler_enabled"] = "YES"
            self.ctx["step_speed_profiler_status"] = f"installed; waiting for tqdm loop with total={self.steps_hint or 'unknown'}"
            self.ctx["step_speed_profiler_live_path"] = str(self.live_path) if self.live_path else "none"
        except Exception as exc:
            self.ctx["step_speed_profiler_enabled"] = f"FAILED: {type(exc).__name__}: {exc}"
            self.failures.append(f"install failed: {type(exc).__name__}: {exc}")

    def stop(self) -> None:
        try:
            if self._orig_tqdm is not None:
                import tqdm as tqdm_mod  # type: ignore
                if getattr(getattr(tqdm_mod, "tqdm", None), "_framevision_step_speed_profiler", False):
                    setattr(tqdm_mod, "tqdm", self._orig_tqdm)
            if self._orig_auto_tqdm is not None:
                import tqdm.auto as tqdm_auto_mod  # type: ignore
                if getattr(getattr(tqdm_auto_mod, "tqdm", None), "_framevision_step_speed_profiler", False):
                    setattr(tqdm_auto_mod, "tqdm", self._orig_auto_tqdm)
        except Exception as exc:
            self.failures.append(f"restore tqdm failed: {type(exc).__name__}: {exc}")
        try:
            if _get_active_step_speed_profiler() is self:
                _set_active_step_speed_profiler(None)
        except Exception:
            pass
        try:
            if self._live_fh is not None:
                self._live_fh.close()
        except Exception:
            pass

    def begin_step(self, loop_index: int, step_index: int) -> None:
        with self._lock:
            self.current_loop_index = int(loop_index)
            self.current_step_index = int(step_index)
            snap = self._snapshot()
            rec: Dict[str, Any] = {
                "loop": int(loop_index),
                "step": int(step_index),
                "start": time.perf_counter(),
                "wall_s": 0.0,
                "pre_hook_s": 0.0,
                "post_hook_s": 0.0,
                "load_s": 0.0,
                "pre_unload_s": 0.0,
                "post_unload_s": 0.0,
                "trim_s": 0.0,
                "emergency_s": 0.0,
                "load_count": 0,
                "unload_count": 0,
                "forced_unload_count": 0,
                "trim_count": 0,
                "emergency_trim_count": 0,
                "pre_calls": 0,
                "post_calls": 0,
                "bytes_loaded": 0,
                "cuda_begin": snap,
                "cuda_end": {},
            }
            self.steps.append(rec)
            if len(self.steps) > self.max_step_events:
                self.steps = self.steps[-self.max_step_events:]
            self.ctx["step_speed_profiler_status"] = f"profiling loop {loop_index}, step {step_index + 1}"

    def end_step(self, loop_index: int, step_index: int) -> None:
        with self._lock:
            rec = self._find_step(loop_index, step_index)
            if rec is None:
                return
            rec["wall_s"] = max(0.0, time.perf_counter() - float(rec.get("start", time.perf_counter())))
            rec["cuda_end"] = self._snapshot()
            hook_s = float(rec.get("pre_hook_s", 0.0)) + float(rec.get("post_hook_s", 0.0))
            transfer_s = float(rec.get("load_s", 0.0)) + float(rec.get("pre_unload_s", 0.0)) + float(rec.get("post_unload_s", 0.0)) + float(rec.get("trim_s", 0.0)) + float(rec.get("emergency_s", 0.0))
            remaining_s = max(0.0, float(rec.get("wall_s", 0.0)) - hook_s)
            rec["hook_total_s"] = hook_s
            rec["transfer_guard_s"] = transfer_s
            rec["remaining_compute_or_pipeline_s"] = remaining_s
            line = (
                f"loop={loop_index} step={step_index + 1}/{self.steps_hint or '?'} "
                f"wall={rec['wall_s']:.3f}s hook={hook_s:.3f}s load={float(rec.get('load_s',0.0)):.3f}s "
                f"unload={(float(rec.get('pre_unload_s',0.0))+float(rec.get('post_unload_s',0.0))+float(rec.get('trim_s',0.0))):.3f}s "
                f"loads={int(rec.get('load_count',0))} unloads={int(rec.get('unload_count',0))} "
                f"driver_free_end={_fmt_bytes((rec.get('cuda_end') or {}).get('driver_free', 0))}"
            )
            self._write_live(line)
            if self.echo:
                print(f"[vram-lab-ltx-step] {line}", flush=True)
            self.current_step_index = -1

    def _find_step(self, loop_index: int, step_index: int) -> Optional[Dict[str, Any]]:
        for rec in reversed(self.steps):
            if int(rec.get("loop", -1)) == int(loop_index) and int(rec.get("step", -1)) == int(step_index):
                return rec
        return None

    def _current_rec(self) -> Optional[Dict[str, Any]]:
        if self.current_step_index < 0:
            return None
        return self._find_step(self.current_loop_index, self.current_step_index)

    def record_hook_timing(
        self,
        phase: str,
        block_name: str,
        block_bytes: int = 0,
        total_s: float = 0.0,
        load_s: float = 0.0,
        pre_unload_s: float = 0.0,
        post_unload_s: float = 0.0,
        trim_s: float = 0.0,
        emergency_s: float = 0.0,
        loaded: bool = False,
        unloaded: bool = False,
        forced_unloads: int = 0,
        trim_count: int = 0,
        emergency_trim_count: int = 0,
    ) -> None:
        with self._lock:
            rec = self._current_rec()
            if rec is None:
                return
            if phase == "pre":
                rec["pre_calls"] = int(rec.get("pre_calls", 0)) + 1
                rec["pre_hook_s"] = float(rec.get("pre_hook_s", 0.0)) + max(0.0, float(total_s or 0.0))
            else:
                rec["post_calls"] = int(rec.get("post_calls", 0)) + 1
                rec["post_hook_s"] = float(rec.get("post_hook_s", 0.0)) + max(0.0, float(total_s or 0.0))
            rec["load_s"] = float(rec.get("load_s", 0.0)) + max(0.0, float(load_s or 0.0))
            rec["pre_unload_s"] = float(rec.get("pre_unload_s", 0.0)) + max(0.0, float(pre_unload_s or 0.0))
            rec["post_unload_s"] = float(rec.get("post_unload_s", 0.0)) + max(0.0, float(post_unload_s or 0.0))
            rec["trim_s"] = float(rec.get("trim_s", 0.0)) + max(0.0, float(trim_s or 0.0))
            rec["emergency_s"] = float(rec.get("emergency_s", 0.0)) + max(0.0, float(emergency_s or 0.0))
            if loaded:
                rec["load_count"] = int(rec.get("load_count", 0)) + 1
                rec["bytes_loaded"] = int(rec.get("bytes_loaded", 0)) + int(block_bytes or 0)
            if unloaded:
                rec["unload_count"] = int(rec.get("unload_count", 0)) + 1
            rec["forced_unload_count"] = int(rec.get("forced_unload_count", 0)) + int(forced_unloads or 0)
            rec["trim_count"] = int(rec.get("trim_count", 0)) + int(trim_count or 0)
            rec["emergency_trim_count"] = int(rec.get("emergency_trim_count", 0)) + int(emergency_trim_count or 0)

    def update_context(self, ctx: Optional[Dict[str, Any]] = None) -> None:
        target = ctx if isinstance(ctx, dict) else self.ctx
        completed = [r for r in self.steps if float(r.get("wall_s", 0.0) or 0.0) > 0.0]
        target["step_speed_profiler_enabled"] = target.get("step_speed_profiler_enabled", "YES" if self.enabled else "NO")
        target["step_speed_profiled_loop_count"] = str(self.profiled_loop_count)
        target["step_speed_profiled_step_count"] = str(len(completed))
        target["step_speed_profiler_failures"] = "none" if not self.failures else " | ".join(self.failures[-12:])
        if not completed:
            target["step_speed_profiler_status"] = target.get("step_speed_profiler_status", "installed, no matching step loop observed")
            target["step_speed_summary"] = "no completed denoise steps recorded"
            target["step_speed_per_step"] = "none"
            return
        total_wall = sum(float(r.get("wall_s", 0.0) or 0.0) for r in completed)
        total_hook = sum(float(r.get("hook_total_s", 0.0) or 0.0) for r in completed)
        total_load = sum(float(r.get("load_s", 0.0) or 0.0) for r in completed)
        total_unload = sum(float(r.get("pre_unload_s", 0.0) or 0.0) + float(r.get("post_unload_s", 0.0) or 0.0) + float(r.get("trim_s", 0.0) or 0.0) for r in completed)
        total_emergency = sum(float(r.get("emergency_s", 0.0) or 0.0) for r in completed)
        total_remaining = sum(float(r.get("remaining_compute_or_pipeline_s", 0.0) or 0.0) for r in completed)
        total_loads = sum(int(r.get("load_count", 0) or 0) for r in completed)
        total_unloads = sum(int(r.get("unload_count", 0) or 0) for r in completed)
        total_forced = sum(int(r.get("forced_unload_count", 0) or 0) for r in completed)
        total_trim = sum(int(r.get("trim_count", 0) or 0) for r in completed)
        total_em_trim = sum(int(r.get("emergency_trim_count", 0) or 0) for r in completed)
        avg = total_wall / max(1, len(completed))
        pct = (lambda x: (100.0 * float(x) / total_wall) if total_wall > 0 else 0.0)
        target["step_speed_profiler_status"] = "completed step profiling"
        target["step_speed_total_wall_s"] = f"{total_wall:.3f}"
        target["step_speed_avg_step_s"] = f"{avg:.3f}"
        target["step_speed_total_hook_s"] = f"{total_hook:.3f} ({pct(total_hook):.1f}%)"
        target["step_speed_total_load_s"] = f"{total_load:.3f} ({pct(total_load):.1f}%)"
        target["step_speed_total_unload_trim_s"] = f"{total_unload:.3f} ({pct(total_unload):.1f}%)"
        target["step_speed_total_emergency_s"] = f"{total_emergency:.3f} ({pct(total_emergency):.1f}%)"
        target["step_speed_remaining_compute_or_pipeline_s"] = f"{total_remaining:.3f} ({pct(total_remaining):.1f}%)"
        target["step_speed_total_block_loads"] = str(total_loads)
        target["step_speed_total_block_unloads"] = str(total_unloads)
        target["step_speed_total_forced_unloads"] = str(total_forced)
        target["step_speed_total_hot_trims"] = str(total_trim)
        target["step_speed_total_emergency_trims"] = str(total_em_trim)
        target["step_speed_summary"] = (
            f"{len(completed)} steps, avg {avg:.2f}s/step; "
            f"hook {pct(total_hook):.1f}%, load {pct(total_load):.1f}%, unload/trim {pct(total_unload):.1f}%, "
            f"emergency {pct(total_emergency):.1f}%, remaining compute/pipeline {pct(total_remaining):.1f}%; "
            f"loads {total_loads}, unloads {total_unloads}, hot trims {total_trim}, emergency trims {total_em_trim}"
        )
        lines = []
        for r in completed[-32:]:
            lines.append(
                f"loop {int(r.get('loop',0))} step {int(r.get('step',0))+1}: "
                f"wall={float(r.get('wall_s',0.0)):.3f}s, "
                f"hook={float(r.get('hook_total_s',0.0)):.3f}s, "
                f"load={float(r.get('load_s',0.0)):.3f}s, "
                f"unload/trim={(float(r.get('pre_unload_s',0.0))+float(r.get('post_unload_s',0.0))+float(r.get('trim_s',0.0))):.3f}s, "
                f"remaining={float(r.get('remaining_compute_or_pipeline_s',0.0)):.3f}s, "
                f"loads={int(r.get('load_count',0))}, unloads={int(r.get('unload_count',0))}, "
                f"driver_free_end={_fmt_bytes((r.get('cuda_end') or {}).get('driver_free', 0))}"
            )
        target["step_speed_per_step"] = " | ".join(lines) if lines else "none"


def make_step_speed_profiler(
    ctx: Optional[Dict[str, Any]] = None,
    torch_module: Any = None,
    steps_hint: int = 0,
    echo: bool = False,
    live_path: Optional[str] = None,
    max_step_events: int = 256,
) -> StepSpeedProfiler:
    return StepSpeedProfiler(
        ctx=ctx,
        torch_module=torch_module,
        steps_hint=steps_hint,
        echo=echo,
        live_path=live_path,
        max_step_events=max_step_events,
    )



@dataclass
class _HookedBlock:
    name: str
    module_ref: Any
    bytes: int = 0
    pre_handle: Any = None
    post_handle: Any = None

    def module(self) -> Any:
        try:
            return self.module_ref()
        except Exception:
            return None


class VramForwardHookRuntime:
    """Move only the currently executing block to CUDA, then release it.

    This is deliberately generic. It knows nothing about Wan/Hunyuan/LTX; it only
    receives component modules and a regex telling it which submodules are heavy
    blocks worth controlling.
    """

    def __init__(self, component_map: Mapping[str, Any], policy: Optional[Mapping[str, Any]] = None, torch_module: Any = None) -> None:
        self.component_map = dict(component_map or {})
        self.policy = apply_vram_lab_profile_defaults(policy or {}, (policy or {}).get("mode", "safe"))
        self.torch = torch_module
        self.device = str(self.policy.get("device", "cuda"))
        self.release_after_forward = bool(self.policy.get("release_after_forward", True))
        self.unload_other_blocks_before_load = bool(self.policy.get("unload_other_blocks_before_load", True))
        self.synchronize_after_unload = bool(self.policy.get("synchronize_after_unload", False))
        self.empty_cache_after_unload = bool(self.policy.get("empty_cache_after_unload", False))
        self.empty_cache_every = int(self.policy.get("empty_cache_every", 0) or 0)
        self.max_blocks = int(self.policy.get("max_blocks", 999999) or 999999)
        self.hot_block_budget_bytes = int(self.policy.get("hot_block_budget_bytes", 0) or 0)
        self.emergency_driver_free_floor_bytes = int(self.policy.get("emergency_driver_free_floor_bytes", 0) or 0)
        # Generic residency strategy. This is deliberately model-agnostic: the
        # runtime only knows about an ordered list of hooked blocks and a memory
        # budget. In planned_hotset mode, a stable prefix of blocks is kept hot
        # across repeated forwards and transient blocks are released after use.
        # That avoids the worst rolling-window pattern where the end of one
        # forward keeps the last blocks hot, but the next forward starts again
        # at block 0 and reloads almost everything.
        self.residency_strategy = str(
            self.policy.get("residency_strategy")
            or os.environ.get("FRAMEVISION_VRAM_RESIDENCY_STRATEGY")
            or "planned_hotset"
        ).strip().lower()
        if self.residency_strategy in {"planned", "stable", "stable_prefix", "planned_stable"}:
            self.residency_strategy = "planned_hotset"
        if self.residency_strategy not in {"rolling", "planned_hotset"}:
            self.residency_strategy = "planned_hotset"
        try:
            self.stable_hotset_fraction = float(
                self.policy.get("stable_hotset_fraction")
                or os.environ.get("FRAMEVISION_VRAM_STABLE_HOTSET_FRACTION")
                or 0.82
            )
        except Exception:
            self.stable_hotset_fraction = 0.82
        self.stable_hotset_fraction = max(0.10, min(2.00, float(self.stable_hotset_fraction)))
        self.stable_hotset_budget_bytes = int(self.policy.get("stable_hotset_budget_bytes", 0) or 0)
        if self.stable_hotset_budget_bytes <= 0:
            try:
                budget_gb = float(os.environ.get("FRAMEVISION_VRAM_STABLE_HOTSET_BUDGET_GB", "0.0") or 0.0)
            except Exception:
                budget_gb = 0.0
            if budget_gb > 0.0:
                self.stable_hotset_budget_bytes = int(budget_gb * 1024 ** 3)
        self._stable_hotset_names: set[str] = set()
        self._stable_hotset_order: List[str] = []
        self._stable_hotset_bytes = 0
        self._transient_unload_count = 0
        self.profile_note = str(self.policy.get("profile_note", "n/a"))
        self.edge_hot_window_gb = float(self.policy.get("edge_hot_window_gb", 0.0) or 0.0)
        self.safe_hot_window_gb = float(self.policy.get("safe_hot_window_gb", 0.0) or 0.0)
        # Legacy aliases kept for older report readers.
        self.aggressive_extra_gb = float(self.policy.get("aggressive_extra_gb", 0.0) or 0.0)
        self.balanced_hot_window_gb = float(self.policy.get("balanced_hot_window_gb", 0.0) or 0.0)
        self._resident_order: List[str] = []
        self.emergency_cleanup_count = 0
        self.hot_block_trim_count = 0
        self.emergency_trim_count = 0
        self.lowest_driver_free = 0
        self.highest_driver_used = 0
        self.pattern = str(self.policy.get("hook_name_regex", r".*"))
        self.regex = re.compile(self.pattern)
        self.blocks: List[_HookedBlock] = []
        self.failures: List[str] = []
        self.pre_calls = 0
        self.post_calls = 0
        self.block_load_count = 0
        self.block_unload_count = 0
        self.forced_unload_count = 0
        self.cache_cleanup_count = 0
        self.active_block_name = "n/a"
        self.peak_allocated = 0
        self.peak_reserved = 0
        self._detached = False
        self._discover_blocks()
        self._plan_stable_hotset()
        self._install_hooks()
        self._park_hooked_blocks_on_cpu()

    def _discover_blocks(self) -> None:
        seen: set[int] = set()
        for comp_name, component in self.component_map.items():
            if component is None or not hasattr(component, "named_modules"):
                continue
            try:
                named_modules = list(component.named_modules())
            except Exception as exc:
                self.failures.append(f"{comp_name}.named_modules failed: {type(exc).__name__}: {exc}")
                continue
            for mod_name, module in named_modules:
                full = f"{comp_name}.{mod_name}" if mod_name else str(comp_name)
                if not self.regex.search(full):
                    continue
                ident = id(module)
                if ident in seen:
                    continue
                seen.add(ident)
                self.blocks.append(_HookedBlock(full, weakref.ref(module), _module_bytes(module)))
                if len(self.blocks) >= self.max_blocks:
                    return

    def _plan_stable_hotset(self) -> None:
        """Choose a reusable stable hot-set from the ordered hooked blocks.

        This is generic and does not know model names. It assumes many diffusion
        style models repeatedly execute the same ordered block list. Keeping a
        stable subset resident across forwards avoids reloading those blocks on
        every step, while leaving budget for the currently executing transient
        block and emergency cleanup.
        """
        self._stable_hotset_names = set()
        self._stable_hotset_order = []
        self._stable_hotset_bytes = 0
        if self.residency_strategy != "planned_hotset":
            return
        if not self.blocks or self.hot_block_budget_bytes <= 0:
            return
        largest = max((int(b.bytes or 0) for b in self.blocks), default=0)
        if largest <= 0:
            largest = max((_module_bytes(b.module()) for b in self.blocks if b.module() is not None), default=0)
        # Reserve enough room for at least one transient block plus allocator
        # slack. The hot-set itself should not consume the whole budget.
        requested = self.stable_hotset_budget_bytes
        if requested <= 0:
            requested = int(float(self.hot_block_budget_bytes) * float(self.stable_hotset_fraction))
        transient_reserve = int(largest * 1.25) if largest > 0 else int(1024 ** 3)
        budget = max(0, min(int(requested), int(self.hot_block_budget_bytes) - transient_reserve))
        if budget <= 0:
            return
        total = 0
        for block in self.blocks:
            size = int(block.bytes or 0)
            if size <= 0:
                size = _module_bytes(block.module())
            if size <= 0:
                continue
            if total + size > budget:
                break
            self._stable_hotset_names.add(block.name)
            self._stable_hotset_order.append(block.name)
            total += size
        self._stable_hotset_bytes = int(total)

    def _is_stable_hotset_block(self, name: str) -> bool:
        return self.residency_strategy == "planned_hotset" and name in self._stable_hotset_names

    def _evict_block_name(self, victim_name: str, reason: str = "trim") -> bool:
        block = self._block_by_name(victim_name)
        module = block.module() if block is not None else None
        if module is not None and _module_is_cuda(module):
            try:
                if _safe_to(module, "cpu"):
                    self.block_unload_count += 1
                    if reason == "transient":
                        self._transient_unload_count += 1
                    elif reason == "emergency":
                        self.forced_unload_count += 1
                        self.emergency_trim_count += 1
                    else:
                        self.hot_block_trim_count += 1
                    try:
                        if victim_name in self._resident_order:
                            self._resident_order.remove(victim_name)
                    except Exception:
                        pass
                    return True
            except Exception as exc:
                self.failures.append(f"{reason} unload failed for {victim_name}: {type(exc).__name__}: {exc}")
        return False

    def _install_hooks(self) -> None:
        for block in self.blocks:
            module = block.module()
            if module is None:
                continue
            try:
                block.pre_handle = module.register_forward_pre_hook(self._make_pre_hook(block.name))
                block.post_handle = module.register_forward_hook(self._make_post_hook(block.name))
            except Exception as exc:
                self.failures.append(f"hook install failed for {block.name}: {type(exc).__name__}: {exc}")

    def _park_hooked_blocks_on_cpu(self) -> None:
        # This is the core residency behavior: do not keep every hooked block on CUDA.
        for block in self.blocks:
            module = block.module()
            if module is None:
                continue
            try:
                if _module_is_cuda(module):
                    if _safe_to(module, "cpu"):
                        self.block_unload_count += 1
            except Exception as exc:
                self.failures.append(f"initial CPU park failed for {block.name}: {type(exc).__name__}: {exc}")
        self._cleanup_cuda("after_initial_cpu_park")

    def _cleanup_cuda(self, reason: str = "") -> None:
        try:
            if self.torch is not None and self.torch.cuda.is_available():
                try:
                    self.torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    self.torch.cuda.empty_cache()
                    self.cache_cleanup_count += 1
                except Exception:
                    pass
                try:
                    self.torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception as exc:
            self.failures.append(f"cuda cleanup failed {reason}: {type(exc).__name__}: {exc}")

    def _snapshot_peak(self) -> None:
        snap = _torch_cuda_snapshot(self.torch)
        self.peak_allocated = max(self.peak_allocated, int(snap.get("allocated", 0)))
        self.peak_reserved = max(self.peak_reserved, int(snap.get("reserved", 0)))

    def _unload_other_blocks(self, keep_name: str) -> None:
        for block in self.blocks:
            if block.name == keep_name:
                continue
            if self._is_stable_hotset_block(block.name):
                continue
            module = block.module()
            if module is None:
                continue
            try:
                if _module_is_cuda(module):
                    if _safe_to(module, "cpu"):
                        self.forced_unload_count += 1
                        self.block_unload_count += 1
                        try:
                            if block.name in self._resident_order:
                                self._resident_order.remove(block.name)
                        except Exception:
                            pass
            except Exception as exc:
                self.failures.append(f"forced unload failed for {block.name}: {type(exc).__name__}: {exc}")
        if self.synchronize_after_unload or self.empty_cache_after_unload:
            self._cleanup_cuda("after_forced_unload")

    def _mark_resident(self, name: str) -> None:
        try:
            if name in self._resident_order:
                self._resident_order.remove(name)
            self._resident_order.append(name)
        except Exception:
            pass

    def _hot_cuda_bytes(self) -> int:
        total = 0
        for block in self.blocks:
            module = block.module()
            if module is not None and _module_is_cuda(module):
                total += int(block.bytes or _module_bytes(module))
        return int(total)

    def _block_by_name(self, name: str) -> Optional[_HookedBlock]:
        for block in self.blocks:
            if block.name == name:
                return block
        return None

    def _trim_hot_blocks(self, keep_name: str = "") -> None:
        if self.hot_block_budget_bytes <= 0:
            return
        # Evict oldest resident blocks until the hot-block window fits. In
        # planned_hotset mode, transient blocks are evicted before stable blocks.
        while self._hot_cuda_bytes() > self.hot_block_budget_bytes and self._resident_order:
            victim_name = ""
            if self.residency_strategy == "planned_hotset":
                for candidate in list(self._resident_order):
                    if candidate != keep_name and not self._is_stable_hotset_block(candidate):
                        victim_name = candidate
                        break
                if not victim_name:
                    for candidate in list(self._resident_order):
                        if candidate != keep_name:
                            victim_name = candidate
                            break
                if not victim_name:
                    break
                try:
                    self._resident_order.remove(victim_name)
                except Exception:
                    pass
            else:
                victim_name = self._resident_order.pop(0)
                if victim_name == keep_name and self._resident_order:
                    self._resident_order.append(victim_name)
                    continue
            self._evict_block_name(victim_name, reason="trim")
            if victim_name == keep_name:
                break

    def _emergency_cleanup_if_needed(self) -> None:
        if self.emergency_driver_free_floor_bytes <= 0:
            return
        snap = _torch_cuda_snapshot(self.torch)
        free = int(snap.get("driver_free", 0) or 0)
        total = int(snap.get("driver_total", 0) or 0)
        if total > 0 and free > 0:
            if self.lowest_driver_free <= 0:
                self.lowest_driver_free = free
            else:
                self.lowest_driver_free = min(self.lowest_driver_free, free)
            self.highest_driver_used = max(self.highest_driver_used, total - free)
        if total <= 0 or free <= 0 or free >= self.emergency_driver_free_floor_bytes:
            return

        # Do not dump every resident block. Trim only until the requested
        # real-VRAM headroom is back. Planned hot-set mode tries transient blocks
        # first, then stable blocks only if this is a real emergency.
        trimmed = 0
        while free < self.emergency_driver_free_floor_bytes and self._resident_order:
            victim_name = ""
            if self.residency_strategy == "planned_hotset":
                for candidate in list(self._resident_order):
                    if not self._is_stable_hotset_block(candidate):
                        victim_name = candidate
                        break
                if not victim_name:
                    victim_name = self._resident_order[0]
                try:
                    self._resident_order.remove(victim_name)
                except Exception:
                    pass
            else:
                victim_name = self._resident_order.pop(0)
            if self._evict_block_name(victim_name, reason="emergency"):
                trimmed += 1
            try:
                if self.torch is not None and self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()
                    self.cache_cleanup_count += 1
                    free, total = self.torch.cuda.mem_get_info()
                    free = int(free or 0)
            except Exception:
                break
        if trimmed:
            self.emergency_cleanup_count += 1

    def _make_pre_hook(self, name: str):
        def pre_hook(module: Any, inputs: Tuple[Any, ...]) -> None:
            self.pre_calls += 1
            self.active_block_name = name
            hook_start = time.perf_counter()
            unload_s = 0.0
            load_s = 0.0
            emergency_s = 0.0
            loaded = False
            forced_before = int(self.forced_unload_count)
            emergency_trim_before = int(self.emergency_trim_count)
            if self.unload_other_blocks_before_load:
                t0 = time.perf_counter()
                self._unload_other_blocks(name)
                unload_s += time.perf_counter() - t0
            try:
                if not _module_is_cuda(module):
                    t0 = time.perf_counter()
                    if _safe_to(module, self.device):
                        self.block_load_count += 1
                        loaded = True
                    else:
                        self.failures.append(f"load to {self.device} returned false for {name}")
                    load_s += time.perf_counter() - t0
                if _module_is_cuda(module):
                    self._mark_resident(name)
                    t0 = time.perf_counter()
                    self._emergency_cleanup_if_needed()
                    emergency_s += time.perf_counter() - t0
            except Exception as exc:
                self.failures.append(f"pre-hook load failed for {name}: {type(exc).__name__}: {exc}")
            total_s = time.perf_counter() - hook_start
            profiler = _get_active_step_speed_profiler()
            if profiler is not None:
                try:
                    block = self._block_by_name(name)
                    profiler.record_hook_timing(
                        "pre",
                        name,
                        block_bytes=int((block.bytes if block is not None else 0) or 0),
                        total_s=total_s,
                        load_s=load_s,
                        pre_unload_s=unload_s,
                        emergency_s=emergency_s,
                        loaded=loaded,
                        forced_unloads=max(0, int(self.forced_unload_count) - forced_before),
                        emergency_trim_count=max(0, int(self.emergency_trim_count) - emergency_trim_before),
                    )
                except Exception:
                    pass
            self._snapshot_peak()
        return pre_hook

    def _make_post_hook(self, name: str):
        def post_hook(module: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
            self.post_calls += 1
            hook_start = time.perf_counter()
            unload_s = 0.0
            trim_s = 0.0
            emergency_s = 0.0
            unloaded = False
            forced_before = int(self.forced_unload_count)
            trim_before = int(self.hot_block_trim_count)
            emergency_trim_before = int(self.emergency_trim_count)
            try:
                if self.residency_strategy == "planned_hotset" and _module_is_cuda(module):
                    if self._is_stable_hotset_block(name):
                        self._mark_resident(name)
                        t0 = time.perf_counter()
                        self._trim_hot_blocks(keep_name=name)
                        trim_s += time.perf_counter() - t0
                    else:
                        # Transient blocks are released after use so the stable
                        # hot-set is not rolled out by the tail of each forward.
                        t0 = time.perf_counter()
                        unloaded = self._evict_block_name(name, reason="transient")
                        unload_s += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    self._emergency_cleanup_if_needed()
                    emergency_s += time.perf_counter() - t0
                elif self.release_after_forward and _module_is_cuda(module):
                    t0 = time.perf_counter()
                    if _safe_to(module, "cpu"):
                        self.block_unload_count += 1
                        unloaded = True
                    unload_s += time.perf_counter() - t0
                    try:
                        if name in self._resident_order:
                            self._resident_order.remove(name)
                    except Exception:
                        pass
                    if self.synchronize_after_unload:
                        try:
                            self.torch.cuda.synchronize()
                        except Exception:
                            pass
                    if self.empty_cache_after_unload or (self.empty_cache_every and self.post_calls % self.empty_cache_every == 0):
                        self._cleanup_cuda("post_hook")
                elif _module_is_cuda(module):
                    self._mark_resident(name)
                    t0 = time.perf_counter()
                    self._trim_hot_blocks(keep_name=name)
                    trim_s += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    self._emergency_cleanup_if_needed()
                    emergency_s += time.perf_counter() - t0
            except Exception as exc:
                self.failures.append(f"post-hook unload failed for {name}: {type(exc).__name__}: {exc}")
            total_s = time.perf_counter() - hook_start
            profiler = _get_active_step_speed_profiler()
            if profiler is not None:
                try:
                    profiler.record_hook_timing(
                        "post",
                        name,
                        total_s=total_s,
                        post_unload_s=unload_s,
                        trim_s=trim_s,
                        emergency_s=emergency_s,
                        unloaded=unloaded,
                        forced_unloads=max(0, int(self.forced_unload_count) - forced_before),
                        trim_count=max(0, int(self.hot_block_trim_count) - trim_before),
                        emergency_trim_count=max(0, int(self.emergency_trim_count) - emergency_trim_before),
                    )
                except Exception:
                    pass
            self._snapshot_peak()
            return output
        return post_hook

    def update_context(self, ctx: Optional[Dict[str, Any]]) -> None:
        if not isinstance(ctx, dict):
            return
        current_cuda = []
        for block in self.blocks:
            module = block.module()
            if module is not None and _module_is_cuda(module):
                current_cuda.append(block.name)
        largest = max((b.bytes for b in self.blocks), default=0)
        sample = ", ".join(b.name for b in self.blocks[:12])
        ctx["vram_forward_hooks_attached"] = "YES" if self.blocks else "NO"
        ctx["vram_hooked_component_names"] = ", ".join(str(k) for k in self.component_map.keys()) or "n/a"
        ctx["vram_hooked_block_count"] = str(len(self.blocks))
        ctx["vram_hooked_block_count_int"] = int(len(self.blocks))
        ctx["vram_pre_forward_calls"] = str(self.pre_calls)
        ctx["vram_pre_forward_calls_int"] = int(self.pre_calls)
        ctx["vram_post_forward_calls"] = str(self.post_calls)
        ctx["vram_post_forward_calls_int"] = int(self.post_calls)
        ctx["vram_block_load_count"] = str(self.block_load_count)
        ctx["vram_block_unload_count"] = str(self.block_unload_count)
        ctx["vram_forced_unload_count"] = str(self.forced_unload_count)
        ctx["vram_cache_cleanup_count"] = str(self.cache_cleanup_count)
        ctx["vram_emergency_cleanup_count"] = str(self.emergency_cleanup_count)
        ctx["vram_hot_block_trim_count"] = str(self.hot_block_trim_count)
        ctx["vram_emergency_trim_count"] = str(self.emergency_trim_count)
        ctx["vram_profile_note"] = self.profile_note
        ctx["vram_safe_hot_window_gb"] = f"{self.safe_hot_window_gb:.1f}"
        ctx["vram_edge_hot_window_gb"] = f"{self.edge_hot_window_gb:.1f}"
        # Legacy aliases kept for older report readers.
        ctx["vram_aggressive_extra_gb"] = f"{self.aggressive_extra_gb:.1f}"
        ctx["vram_balanced_hot_window_gb"] = f"{self.balanced_hot_window_gb:.1f}"
        ctx["vram_hot_block_budget"] = _fmt_bytes(self.hot_block_budget_bytes)
        ctx["vram_residency_strategy"] = self.residency_strategy
        ctx["vram_stable_hotset_count"] = str(len(self._stable_hotset_names))
        ctx["vram_stable_hotset_bytes"] = _fmt_bytes(self._stable_hotset_bytes)
        ctx["vram_stable_hotset_fraction"] = f"{self.stable_hotset_fraction:.2f}"
        ctx["vram_transient_unload_count"] = str(self._transient_unload_count)
        ctx["vram_stable_hotset_sample"] = ", ".join(self._stable_hotset_order[:12]) if self._stable_hotset_order else "none"
        ctx["vram_emergency_driver_free_floor"] = _fmt_bytes(self.emergency_driver_free_floor_bytes)
        ctx["vram_lowest_driver_free_during_hooks"] = _fmt_bytes(self.lowest_driver_free)
        ctx["vram_highest_driver_used_during_hooks"] = _fmt_bytes(self.highest_driver_used)
        ctx["vram_hook_block_pattern"] = self.pattern
        ctx["vram_blocks_currently_cuda"] = ", ".join(current_cuda[:20]) if current_cuda else "none"
        ctx["vram_retained_cuda_refs"] = "none" if not current_cuda else f"{len(current_cuda)} hooked block(s) still CUDA"
        ctx["vram_sample_hooked_block_names"] = sample or "none"
        ctx["vram_active_block_name"] = self.active_block_name
        ctx["vram_largest_hooked_block"] = _fmt_bytes(largest)
        ctx["vram_peak_cuda_during_hooked_execution"] = f"allocated={_fmt_bytes(self.peak_allocated)}, reserved={_fmt_bytes(self.peak_reserved)}"
        if self.hot_block_budget_bytes and self.peak_reserved and self.peak_reserved < int(self.hot_block_budget_bytes * 0.45):
            ctx["vram_policy_warning"] = (
                "active CUDA stayed far below the configured residency window; "
                "policy may still be too conservative or blocks are smaller than expected"
            )
        else:
            ctx["vram_policy_warning"] = "none"
        ctx["vram_hook_failures"] = "none" if not self.failures else " | ".join(self.failures[-10:])

    def detach_vram_hooks(self) -> None:
        if self._detached:
            return
        self._detached = True
        for block in self.blocks:
            for handle_name in ("pre_handle", "post_handle"):
                handle = getattr(block, handle_name, None)
                if handle is not None:
                    try:
                        handle.remove()
                    except Exception:
                        pass
                    try:
                        setattr(block, handle_name, None)
                    except Exception:
                        pass
        for block in self.blocks:
            module = block.module()
            if module is not None and _module_is_cuda(module):
                _safe_to(module, "cpu")
        gc.collect()
        self._cleanup_cuda("detach_vram_hooks")


def attach_vram_hooks(component_map: Mapping[str, Any], policy: Optional[Mapping[str, Any]] = None, torch_module: Any = None) -> VramForwardHookRuntime:
    return VramForwardHookRuntime(component_map, policy=policy, torch_module=torch_module)


class LtxHookDiscoveryRuntime:
    """Controlled LTX hook-point discovery and first safe hook attachment.

    This stays generic from VRAM Lab's point of view: it does not import LTX or
    know LTX internals. It watches PyTorch module calls during the LTX process,
    records likely transformer/denoiser/diffusion candidates, and when a module
    with block-like children appears it promotes that object to the existing
    attach_vram_hooks(...) runtime.
    """

    PRIMARY_WORDS = ("transformer", "denoiser", "diffusion", "dit")
    SECONDARY_WORDS = ("ltx", "video", "spatial", "temporal")
    VAE_WORDS = ("vae", "decoder", "autoencoder")
    BLOCK_NAME_RE = re.compile(
        r"(^|\.)(transformer_blocks|blocks|layers|double_blocks|single_blocks|temporal_blocks|spatial_blocks)\.\d+$"
        r"|(^|\.)(temporal|spatial).*blocks?\.\d+$",
        re.IGNORECASE,
    )
    CHILD_HINT_RE = re.compile(
        r"transformer_blocks|double_blocks|single_blocks|(^|\.)blocks\.|(^|\.)layers\.|attention|attn|temporal|spatial|decoder|vae",
        re.IGNORECASE,
    )

    def __init__(
        self,
        ctx: Optional[Dict[str, Any]] = None,
        label: str = "ltx_hook_discovery",
        torch_module: Any = None,
        mode: str = "safe",
        echo: bool = False,
        max_candidates: int = 80,
        max_samples_per_class: int = 3,
    ) -> None:
        self.ctx = ctx if isinstance(ctx, dict) else {}
        self.label = str(label or "ltx_hook_discovery")
        self.torch = torch_module
        self.mode = normalize_vram_lab_mode(mode, "safe")
        self.echo = bool(echo)
        self.max_candidates = int(max_candidates or 80)
        self.max_samples_per_class = int(max_samples_per_class or 3)
        self.started = False
        self.stopped = False
        self.total_calls = 0
        self.failures: List[str] = []
        self.candidates: Dict[str, Dict[str, Any]] = {}
        self._seen_ids: set[int] = set()
        self._checked_ids: set[int] = set()
        self._orig_call_impl: Any = None
        self._runtime: Any = None
        self._restored_call_impl = False
        self._attach_policy = "ltx_denoiser_first"
        self._first_seen_memory: List[str] = []
        self._attached_component_name = ""
        self._attached_reason = ""
        self._attach_attempts: List[str] = []
        self._hook_block_pattern = (
            r"(^|\.)(transformer_blocks|blocks|layers|double_blocks|single_blocks|temporal_blocks|spatial_blocks)\.\d+$"
            r"|(^|\.)(temporal|spatial).*blocks?\.\d+$"
        )
        self.ctx["hook_discovery_ran"] = "NO"
        self.ctx["hook_attachment_attempted"] = "NO"
        self.ctx["hook_attachment_status"] = "not started"

    def _class_key(self, module: Any) -> str:
        cls = type(module)
        return f"{getattr(cls, '__module__', 'unknown')}.{getattr(cls, '__name__', type(module).__name__)}"

    def _class_name_only(self, class_key: str) -> str:
        return str(class_key).rsplit(".", 1)[-1].lower()

    def _is_ltx_denoiser_class(self, class_key: str) -> bool:
        low = str(class_key).lower()
        name = self._class_name_only(class_key)
        return (
            low.startswith("ltx_core.model.transformer.")
            or low.startswith("ltx_core.models.transformer.")
            or name in {"ltxmodel", "x0model", "basicavtransformerblock"}
        )

    def _is_text_encoder_class(self, class_key: str) -> bool:
        low = str(class_key).lower()
        return (
            "gemma" in low
            or low.startswith("transformers.models.")
            or "text_encoder" in low
            or "text_encoders" in low
        )

    def _is_vae_class(self, class_key: str) -> bool:
        low = str(class_key).lower()
        return any(w in low for w in self.VAE_WORDS)

    def _candidate_kind(self, class_key: str) -> str:
        if self._is_ltx_denoiser_class(class_key):
            return "primary-ltx-transformer-denoiser"
        if self._is_text_encoder_class(class_key):
            return "secondary-text-encoder-conditioning"
        if self._is_vae_class(class_key):
            return "secondary-vae-decoder"
        low = class_key.lower()
        # Do not classify the third-party package name ``transformers`` as the
        # actual video transformer. That mistake made Gemma look like the primary
        # denoiser in 0.7.2.
        name = self._class_name_only(class_key)
        if any(w in name for w in self.PRIMARY_WORDS) or any(w in low for w in ("diffusion", "denoiser", ".dit")):
            return "primary-transformer-denoiser-diffusion"
        if any(w in low for w in self.SECONDARY_WORDS):
            return "secondary-ltx-video-temporal-spatial"
        return "other"

    def _score_class(self, class_key: str) -> int:
        if str(class_key).startswith("torch.nn"):
            return -100
        if self._is_ltx_denoiser_class(class_key):
            return 220
        if self._is_text_encoder_class(class_key):
            return 45
        if self._is_vae_class(class_key):
            return 35
        low = class_key.lower()
        name = self._class_name_only(class_key)
        score = 0
        for word in self.PRIMARY_WORDS:
            if word in name:
                score += 100
        if "diffusion" in low or "denoiser" in low or ".dit" in low:
            score += 100
        for word in self.SECONDARY_WORDS:
            if word in low:
                score += 20
        return score

    def _record_candidate(self, module: Any, child_blocks: Optional[List[str]] = None) -> None:
        key = self._class_key(module)
        score = self._score_class(key)
        if score <= 0 and child_blocks is None:
            return
        item = self.candidates.setdefault(
            key,
            {
                "class": key,
                "kind": self._candidate_kind(key),
                "score": score,
                "calls": 0,
                "object_samples": [],
                "block_samples": [],
            },
        )
        item["calls"] = int(item.get("calls", 0)) + 1
        item["score"] = max(int(item.get("score", 0)), score)
        samples = item.setdefault("object_samples", [])
        if len(samples) < self.max_samples_per_class:
            samples.append(f"id={id(module)}")
        if child_blocks:
            block_samples = item.setdefault("block_samples", [])
            for name in child_blocks[:10]:
                if name not in block_samples and len(block_samples) < 20:
                    block_samples.append(name)

    def _find_block_children(self, module: Any) -> List[str]:
        if id(module) in self._checked_ids:
            return []
        self._checked_ids.add(id(module))
        out: List[str] = []
        try:
            for name, child in module.named_modules():
                if not name:
                    continue
                if self.BLOCK_NAME_RE.search(name):
                    out.append(name)
                elif len(out) < 2 and self.CHILD_HINT_RE.search(name):
                    # Secondary info only; useful in the report when exact block
                    # regex does not match an unknown LTX class yet.
                    out.append(name)
                if len(out) >= 48:
                    break
        except Exception as exc:
            self.failures.append(f"named_modules failed for {self._class_key(module)}: {type(exc).__name__}: {exc}")
        return out

    def _should_try_attach(self, module: Any, class_key: str, child_blocks: List[str]) -> Tuple[bool, str]:
        if self._runtime is not None:
            return False, "already attached"
        try:
            if int(str(self.ctx.get("ltx_batch_split_early_guard_attached", "0") or "0")) > 0:
                # The FrameVision LTX wrapper attached/parked the transformer
                # blocks at BatchSplitAdapter construction time, which is
                # earlier than X0Model's first call. Keep discovery records,
                # but do not install a duplicate hook stack on the same blocks.
                self._restore_call_impl_after_attach()
                return False, "early BatchSplitAdapter residency already attached; skip duplicate X0Model hooks"
        except Exception:
            pass
        exact_block_count = sum(1 for n in child_blocks if self.BLOCK_NAME_RE.search(n))
        # 0.7.3 rule: do not attach to Gemma/HF text-encoder classes as the
        # primary LTX target. They are conditioning, not the video denoiser, and
        # hooking them made the tiny test slow while not controlling the real step
        # pressure. Keep them in the report only.
        if self._is_text_encoder_class(class_key):
            return False, f"text encoder / conditioning only; not primary LTX denoiser (exact_block_count={exact_block_count})"
        if self._is_vae_class(class_key):
            return False, f"VAE/decode candidate only; save for finalize guard (exact_block_count={exact_block_count})"
        if self._is_ltx_denoiser_class(class_key) and exact_block_count >= 2:
            return True, f"LTX denoiser/transformer class with {exact_block_count} block-like children"
        return False, f"not selected: waiting for LTXModel/X0Model/BasicAVTransformerBlock style target; exact_block_count={exact_block_count}"

    def _restore_call_impl_after_attach(self) -> None:
        if self._restored_call_impl:
            return
        try:
            if self._orig_call_impl is not None and self.torch is not None:
                setattr(self.torch.nn.Module, "_call_impl", self._orig_call_impl)
                self._restored_call_impl = True
                self.ctx["module_call_tracer_runtime_state"] = "restored immediately after LTX denoiser hook attach"
        except Exception as exc:
            self.failures.append(f"restore after attach failed: {type(exc).__name__}: {exc}")

    def _record_first_seen_memory(self, class_key: str) -> None:
        kind = self._candidate_kind(class_key)
        stage = "other"
        if kind.startswith("primary-ltx") or self._is_ltx_denoiser_class(class_key):
            stage = "ltx_denoiser_first_seen"
        elif kind.startswith("secondary-text"):
            stage = "text_encoder_first_seen"
        elif kind.startswith("secondary-vae"):
            stage = "vae_or_decoder_first_seen"
        if any(str(x).startswith(stage + ":") for x in self._first_seen_memory):
            return
        snap = _torch_cuda_snapshot(self.torch)
        line = f"{stage}: {class_key} | {_snapshot_string(snap)}"
        self._first_seen_memory.append(line)
        self.ctx.setdefault("ltx_component_first_seen_stages", []).append(line)

    def _try_attach(self, module: Any, class_key: str, reason: str) -> None:
        self.ctx["hook_attachment_attempted"] = "YES"
        component_name = class_key.split(".")[-1] or "ltx_component"
        self._attach_attempts.append(f"{class_key}: {reason}")
        try:
            runtime = attach_vram_hooks(
                {component_name: module},
                policy={
                    "mode": self.mode,
                    "device": "cuda",
                    "hook_name_regex": self._hook_block_pattern,
                    "max_blocks": 256,
                },
                torch_module=self.torch,
            )
            if int(getattr(runtime, "blocks", []) and len(runtime.blocks) or 0) > 0:
                self._runtime = runtime
                self._attached_component_name = component_name
                self._attached_reason = reason
                self.ctx["hook_attachment_status"] = f"attached to {component_name}: {reason}"
                self.ctx["hooked_component_names"] = component_name
                self.ctx["hooked_block_count"] = str(len(runtime.blocks))
                self.ctx["vram_hook_block_pattern"] = self._hook_block_pattern
                if self.echo:
                    print(f"[vram-lab-ltx-hooks] attached to {class_key}: {reason}", flush=True)
                # The broad Module._call_impl tracer is only for finding the real
                # target. Once hooks are installed on the LTX denoiser, restore the
                # original call path so the actual steps are not slowed by 50k+
                # discovery calls. Registered hooks still fire normally.
                self._restore_call_impl_after_attach()
            else:
                try:
                    runtime.detach_vram_hooks()
                except Exception:
                    pass
                self.ctx["hook_attachment_status"] = f"attempted {component_name} but no block names matched hook regex"
        except Exception as exc:
            msg = f"attach failed for {class_key}: {type(exc).__name__}: {exc}"
            self.failures.append(msg)
            self.ctx["hook_attachment_status"] = msg

    def start(self) -> None:
        if self.started:
            return
        self.started = True
        self.ctx["hook_discovery_ran"] = "YES"
        self.ctx["module_call_tracer_enabled"] = f"YES: {self.label}"
        self.ctx["hook_attachment_status"] = "discovery active; waiting for stable LTX component"
        try:
            nn_mod = self.torch.nn.Module if self.torch is not None else None
            if nn_mod is None:
                raise RuntimeError("torch.nn.Module unavailable")
            self._orig_call_impl = nn_mod._call_impl
            parent = self
            orig = self._orig_call_impl

            def wrapped_call_impl(mod: Any, *args: Any, **kwargs: Any) -> Any:
                parent.total_calls += 1
                try:
                    key = parent._class_key(mod)
                    score = parent._score_class(key)
                    if score > 0:
                        parent._record_first_seen_memory(key)
                        child_blocks: Optional[List[str]] = None
                        ident = id(mod)
                        # Only scan each object once. This keeps discovery controlled
                        # even though Module._call_impl can fire thousands of times.
                        if ident not in parent._seen_ids and len(parent.candidates) < parent.max_candidates:
                            parent._seen_ids.add(ident)
                            child_blocks = parent._find_block_children(mod)
                            parent._record_candidate(mod, child_blocks=child_blocks)
                            ok, reason = parent._should_try_attach(mod, key, child_blocks)
                            if ok:
                                parent._try_attach(mod, key, reason)
                        else:
                            parent._record_candidate(mod, child_blocks=None)
                except Exception as exc:
                    if len(parent.failures) < 20:
                        parent.failures.append(f"wrapped call discovery failed: {type(exc).__name__}: {exc}")
                return orig(mod, *args, **kwargs)

            setattr(nn_mod, "_call_impl", wrapped_call_impl)
        except Exception as exc:
            self.failures.append(f"start failed: {type(exc).__name__}: {exc}")
            self.ctx["hook_attachment_status"] = f"discovery start failed: {type(exc).__name__}: {exc}"

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            if (not self._restored_call_impl) and self._orig_call_impl is not None and self.torch is not None:
                setattr(self.torch.nn.Module, "_call_impl", self._orig_call_impl)
                self._restored_call_impl = True
        except Exception as exc:
            self.failures.append(f"restore Module._call_impl failed: {type(exc).__name__}: {exc}")

    def update_context(self, ctx: Optional[Dict[str, Any]] = None) -> None:
        target = ctx if isinstance(ctx, dict) else self.ctx
        sorted_items = sorted(
            self.candidates.values(),
            key=lambda d: (int(d.get("score", 0)), int(d.get("calls", 0))),
            reverse=True,
        )
        primary = [d for d in sorted_items if str(d.get("kind", "")).startswith("primary")]
        secondary = [d for d in sorted_items if not str(d.get("kind", "")).startswith("primary")]

        def fmt_item(d: Mapping[str, Any]) -> str:
            blocks = d.get("block_samples") or []
            block_text = ", blocks=" + ", ".join(str(x) for x in blocks[:8]) if blocks else ""
            return f"{d.get('class')} calls={d.get('calls')} kind={d.get('kind')} score={d.get('score')}{block_text}"

        target["hook_discovery_ran"] = "YES" if self.started else "NO"
        target["module_call_total_calls"] = str(self.total_calls)
        target["module_call_unique_classes"] = str(len(self.candidates))
        target["module_call_candidate_class_count"] = str(len(primary) + len(secondary))
        target["candidate_module_classes_found"] = " | ".join(fmt_item(d) for d in sorted_items[:18]) or "none"
        target["candidate_module_paths_names"] = " | ".join(str(d.get("class")) for d in sorted_items[:24]) or "none"
        target["module_call_likely_hook_candidates"] = " | ".join(fmt_item(d) for d in primary[:10]) or "none"
        target["module_call_top_classes"] = " | ".join(fmt_item(d) for d in sorted_items[:20]) or "none"
        target["module_call_candidate_summary"] = " | ".join(fmt_item(d) for d in (primary[:12] + secondary[:8])) or "none"
        target["module_call_tracer_failures"] = "none" if not self.failures else " | ".join(self.failures[-12:])
        target["module_call_tracer_runtime_state"] = target.get("module_call_tracer_runtime_state") or ("restored" if self._restored_call_impl else ("active until stop" if self.started and not self.stopped else "stopped"))
        target["ltx_component_first_seen_summary"] = " | ".join(self._first_seen_memory) if self._first_seen_memory else "none"
        target["hook_attachment_attempts"] = " | ".join(self._attach_attempts[-8:]) or "none"
        if self._runtime is not None:
            try:
                self._runtime.update_context(target)
            except Exception as exc:
                self.failures.append(f"runtime update_context failed: {type(exc).__name__}: {exc}")
            target["hook_attachment_attempted"] = "YES"
            target["hook_attachment_status"] = f"attached to {self._attached_component_name}: {self._attached_reason}"
            target["hooked_component_names"] = target.get("vram_hooked_component_names", self._attached_component_name)
            target["hooked_block_count"] = target.get("vram_hooked_block_count", "0")
            target["pre_forward_hook_calls"] = target.get("vram_pre_forward_calls", "0")
            target["post_forward_hook_calls"] = target.get("vram_post_forward_calls", "0")
            target["block_load_count"] = target.get("vram_block_load_count", "0")
            target["block_unload_count"] = target.get("vram_block_unload_count", "0")
            target["peak_cuda_during_hooked_execution"] = target.get("vram_peak_cuda_during_hooked_execution", "n/a")
        else:
            if target.get("hook_attachment_attempted") != "YES":
                target["hook_attachment_attempted"] = "NO: no safe stable component reached"
            target.setdefault("hooked_component_names", "none")
            target.setdefault("hooked_block_count", "0")
            target.setdefault("pre_forward_hook_calls", "0")
            target.setdefault("post_forward_hook_calls", "0")
            target.setdefault("block_load_count", "0")
            target.setdefault("block_unload_count", "0")
            target.setdefault("peak_cuda_during_hooked_execution", "n/a")

    def detach(self) -> None:
        try:
            if self._runtime is not None:
                self._runtime.detach_vram_hooks()
        except Exception as exc:
            self.failures.append(f"detach failed: {type(exc).__name__}: {exc}")


def make_module_call_tracer(
    ctx: Optional[Dict[str, Any]] = None,
    label: str = "ltx_hook_discovery",
    torch_module: Any = None,
    echo: bool = False,
    mode: str = "safe",
    max_candidates: int = 80,
    sample_every: int = 0,
) -> LtxHookDiscoveryRuntime:
    # Kept with the existing name used by the first LTX wrapper, but upgraded
    # from passive tracing to controlled discovery + safe hook promotion.
    return LtxHookDiscoveryRuntime(
        ctx=ctx,
        label=label,
        torch_module=torch_module,
        mode=mode,
        echo=echo,
        max_candidates=max_candidates,
    )


class PostStepFinalizeGuard:
    """Shared API for decode/finalize memory control after visible steps finish."""

    def __init__(self, ctx: Optional[Dict[str, Any]] = None, label: str = "finalize", torch_module: Any = None) -> None:
        self.ctx = ctx if isinstance(ctx, dict) else {}
        self.label = str(label or "finalize")
        self.torch = torch_module
        self.start = time.perf_counter()
        self.snapshots: Dict[str, Dict[str, int]] = {}
        self.ctx["finalize_guard_enabled"] = f"YES: {self.label}"
        self.ctx.setdefault("finalize_guard_stages", [])
        self.ctx.setdefault("finalize_guard_notes", [])
        self.stage("finalize_guard_start", f"{self.label} created")

    def _record_snapshot(self, key: str, note: str = "") -> Dict[str, int]:
        snap = _torch_cuda_snapshot(self.torch)
        self.snapshots[key] = snap
        self.ctx[key] = _snapshot_string(snap)
        self.ctx[f"{key}_time"] = _now()
        for k, v in snap.items():
            self.ctx[f"{key}_{k}"] = _fmt_bytes(v)
            self.ctx[f"{key}_{k}_int"] = int(v)
        self.ctx.setdefault("finalize_guard_stages", []).append(
            f"{_now()} | {key}: {note or 'n/a'} | {_snapshot_string(snap)}"
        )
        return snap

    def stage(self, key: str, note: str = "") -> None:
        self._record_snapshot(str(key), str(note or ""))
        if "before_vae_decode" in key or "before_decode" in key:
            self.ctx["decode_finalize_start_time"] = self.ctx.get(f"{key}_time", _now())
            self.ctx["decode_finalize_start_perf"] = time.perf_counter()
        if "after_vae_decode" in key or "after_decode" in key:
            if self.ctx.get("decode_finalize_start_perf") is not None and not self.ctx.get("decode_finalize_duration"):
                try:
                    self.ctx["decode_finalize_duration"] = f"{time.perf_counter() - float(self.ctx['decode_finalize_start_perf']):.3f}s"
                except Exception:
                    pass

    def cleanup_cuda(self, label: str = "cleanup") -> None:
        gc.collect()
        try:
            if self.torch is not None and self.torch.cuda.is_available():
                try:
                    self.torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    self.torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    self.torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception as exc:
            self.ctx.setdefault("finalize_guard_notes", []).append(
                f"CUDA cleanup failed at {label}: {type(exc).__name__}: {exc}"
            )
        self.stage(f"cuda_cleanup_{label}", f"cleanup after {label}")

    def detach_runtimes(self, runtimes: Iterable[Any], clear_context_key: Optional[str] = "_vram_runtime") -> None:
        ok = 0
        failed = []
        for rt in list(runtimes or []):
            if rt is None:
                continue
            try:
                if hasattr(rt, "update_context"):
                    rt.update_context(self.ctx)
            except Exception as exc:
                failed.append(f"update failed: {type(exc).__name__}: {exc}")
            try:
                if hasattr(rt, "detach_vram_hooks"):
                    rt.detach_vram_hooks()
                    ok += 1
            except Exception as exc:
                failed.append(f"detach failed: {type(exc).__name__}: {exc}")
        if clear_context_key:
            try:
                self.ctx.pop(clear_context_key, None)
            except Exception:
                pass
        self.ctx["finalize_hooks_detached"] = f"YES: {ok} runtime(s)" if ok else "NO: no runtime detached"
        if failed:
            self.ctx.setdefault("finalize_guard_notes", []).extend(failed[-10:])
        self.cleanup_cuda("after_detach_runtimes")

    def move_components_to_cpu(self, components: Iterable[Any], label: str = "components_to_cpu") -> None:
        moved = 0
        skipped = 0
        failed = []
        names = []
        for comp in list(components or []):
            if comp is None:
                skipped += 1
                continue
            try:
                names.append(type(comp).__name__)
                if hasattr(comp, "maybe_free_model_hooks"):
                    try:
                        comp.maybe_free_model_hooks()
                    except Exception:
                        pass
                if hasattr(comp, "to"):
                    comp.to("cpu")
                    moved += 1
                elif hasattr(comp, "cpu"):
                    comp.cpu()
                    moved += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed.append(f"{type(comp).__name__}: {type(exc).__name__}: {exc}")
        self.ctx["finalize_components_to_cpu"] = (
            f"moved={moved}, skipped={skipped}; components={', '.join(names[:12]) or 'none'}"
        )
        if failed:
            self.ctx.setdefault("finalize_guard_notes", []).append("component CPU move failures: " + " | ".join(failed[-8:]))
        self.cleanup_cuda(label)

    def tensor_to_cpu(self, obj: Any, label: str = "output_to_cpu") -> Any:
        try:
            if self.torch is not None and hasattr(self.torch, "Tensor") and isinstance(obj, self.torch.Tensor):
                try:
                    out = obj.detach().to("cpu")
                except Exception:
                    out = obj.to("cpu")
                self.ctx["finalize_output_to_cpu"] = f"YES: tensor via {label}"
                return out
        except Exception:
            pass
        try:
            if isinstance(obj, list):
                return [self.tensor_to_cpu(x, label=label) for x in obj]
            if isinstance(obj, tuple):
                return tuple(self.tensor_to_cpu(x, label=label) for x in obj)
            if isinstance(obj, dict):
                return {k: self.tensor_to_cpu(v, label=label) for k, v in obj.items()}
        except Exception as exc:
            self.ctx.setdefault("finalize_guard_notes", []).append(
                f"output CPU move recursion failed at {label}: {type(exc).__name__}: {exc}"
            )
        return obj

    def detect_vae_low_memory_support(self, vae: Any) -> str:
        if vae is None:
            return "n/a: no VAE object"
        methods = []
        for name in (
            "enable_tiling", "enable_slicing", "enable_vae_tiling", "enable_vae_slicing",
            "decode_tiled", "tiled_decode", "decode_sliced", "sliced_decode",
        ):
            try:
                if hasattr(vae, name):
                    methods.append(name)
            except Exception:
                pass
        text = ", ".join(methods) if methods else "none detected externally"
        self.ctx["finalize_vae_low_memory_support"] = text
        return text

    def prepare_for_decode(
        self,
        runtimes: Iterable[Any] = (),
        components: Iterable[Any] = (),
        label: str = "before_decode",
        vae: Any = None,
    ) -> None:
        self.ctx["post_step_decode_finalize_api"] = "YES: PostStepFinalizeGuard.prepare_for_decode"
        self.stage("post_step_decode_finalize_start", "visible steps finished; preparing decode/finalize")
        try:
            self.detect_vae_low_memory_support(vae)
        except Exception:
            pass
        self.detach_runtimes(list(runtimes or []), clear_context_key=None)
        self.move_components_to_cpu(list(components or []), label=f"{label}_components_to_cpu")
        self.cleanup_cuda(label)
        self.stage("cuda_before_vae_decode", "before original VAE decode")

    def _infer_spill_likely(self) -> str:
        before = self.snapshots.get("cuda_before_vae_decode") or self.snapshots.get("post_step_decode_finalize_start") or {}
        after = self.snapshots.get("cuda_after_vae_decode") or self.snapshots.get("cuda_after_vae_decode_raw") or {}
        total = int(after.get("driver_total") or before.get("driver_total") or 0)
        free_values = [int(x.get("driver_free") or 0) for x in (before, after) if int(x.get("driver_free") or 0) > 0]
        reserved_values = [int(x.get("reserved") or 0) for x in (before, after) if int(x.get("reserved") or 0) > 0]
        if not total or not free_values:
            return "UNKNOWN: driver free/total unavailable"
        min_free = min(free_values)
        max_reserved = max(reserved_values or [0])
        if min_free < 768 * 1024**2:
            return f"POSSIBLE: driver headroom dropped below 768 MB ({_fmt_bytes(min_free)})"
        if max_reserved and max_reserved > int(total * 0.94):
            return f"POSSIBLE: CUDA reserved near total VRAM ({_fmt_bytes(max_reserved)} / {_fmt_bytes(total)})"
        if min_free > 1536 * 1024**2 and (not max_reserved or max_reserved < int(total * 0.90)):
            return f"NO: driver headroom stayed above 1.5 GB ({_fmt_bytes(min_free)} minimum)"
        return f"UNKNOWN/WATCH: limited headroom ({_fmt_bytes(min_free)} minimum)"

    def finish(self) -> None:
        self.cleanup_cuda("final_cleanup")
        self.ctx["finalize_guard_total_duration"] = f"{time.perf_counter() - self.start:.3f}s"
        self.ctx["finalize_guard_end_memory"] = self.ctx.get("cuda_cleanup_final_cleanup", "n/a")
        inferred = self._infer_spill_likely()
        existing = str(self.ctx.get("finalize_shared_memory_spill_likely", "") or "")
        # The save/re-encode guard may finish after the decode guard but does not
        # own decode snapshots. Do not let an UNKNOWN save-stage inference erase a
        # more useful decode-stage inference.
        if existing and not existing.upper().startswith("UNKNOWN") and inferred.upper().startswith("UNKNOWN"):
            self.ctx["finalize_shared_memory_spill_likely"] = existing
        else:
            self.ctx["finalize_shared_memory_spill_likely"] = inferred


def make_finalize_guard(ctx: Optional[Dict[str, Any]] = None, label: str = "finalize", torch_module: Any = None) -> PostStepFinalizeGuard:
    return PostStepFinalizeGuard(ctx=ctx, label=label, torch_module=torch_module)



class DeepLifecycleLogger:
    """Optional LTX lifecycle logger for finding load/decode/finalize VRAM spikes.

    This logger is intentionally opt-in. It does not change residency policy,
    model loading, hook behavior, or CUDA allocation behavior. It records:
    - periodic CUDA memory samples
    - selected torch.nn.Module.to(...) calls
    - selected torch.nn.Module._apply(...) calls
    - torch.cuda.empty_cache() calls
    - manual lifecycle marks from the wrapper

    Keep it out of normal benchmark runs because it adds overhead.
    """

    _MILESTONES = {1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000}

    def __init__(
        self,
        ctx: Optional[Dict[str, Any]] = None,
        label: str = "ltx_deep",
        torch_module: Any = None,
        echo: bool = False,
        interval: float = 1.0,
        max_events: int = 4000,
        live_path: Optional[Any] = None,
    ) -> None:
        self.ctx = ctx if isinstance(ctx, dict) else {}
        self.label = str(label or "ltx_deep")
        self.torch = torch_module
        self.echo = bool(echo)
        self.interval = max(0.10, float(interval or 1.0))
        self.max_events = max(100, int(max_events or 4000))
        self.events: List[str] = []
        self.counts: Dict[str, int] = {}
        self.live_path: Optional[Path] = Path(live_path) if live_path else None
        self._live_fh: Optional[Any] = None
        self.started_at = time.perf_counter()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._orig_module_to: Any = None
        self._orig_module_apply: Any = None
        self._orig_empty_cache: Any = None
        self._installed = False
        self._failures: List[str] = []
        self.ctx["deep_lifecycle_logger_enabled"] = f"YES: {self.label}"
        self.ctx["deep_lifecycle_live_path"] = str(self.live_path) if self.live_path else "none"
        self.ctx.setdefault("deep_lifecycle_events", [])
        self.ctx.setdefault("deep_lifecycle_failures", [])

    def _elapsed(self) -> str:
        return f"{time.perf_counter() - self.started_at:9.3f}s"

    def _open_live_file(self) -> None:
        if self.live_path is None or self._live_fh is not None:
            return
        try:
            self.live_path.parent.mkdir(parents=True, exist_ok=True)
            self._live_fh = open(self.live_path, "w", encoding="utf-8", errors="replace", buffering=1)
            self._live_fh.write("==============================================================================\n")
            self._live_fh.write("FrameVision LTX deep lifecycle live log\n")
            self._live_fh.write("==============================================================================\n")
            self._live_fh.write(f"started: {_now()}\n")
            self._live_fh.write(f"label: {self.label}\n")
            self._live_fh.write("mode: live flush after every event; useful if the process exits before report writing\n")
            self._live_fh.write("------------------------------------------------------------------------------\n")
            self._live_fh.flush()
        except Exception as exc:
            self._failures.append(f"live log open failed: {type(exc).__name__}: {exc}")
            self._live_fh = None

    def _write_live(self, line: str) -> None:
        if self.live_path is None:
            return
        try:
            self._open_live_file()
            if self._live_fh is not None:
                self._live_fh.write(f"[vram-lab-ltx-deep] {line}\n")
                self._live_fh.flush()
                try:
                    os.fsync(self._live_fh.fileno())
                except Exception:
                    pass
        except Exception as exc:
            # Record only a bounded number of write failures so logging cannot
            # become the reason generation fails.
            if len(self._failures) < 20:
                self._failures.append(f"live log write failed: {type(exc).__name__}: {exc}")

    def _close_live_file(self, final_note: str = "") -> None:
        try:
            if self._live_fh is not None:
                if final_note:
                    self._live_fh.write(f"[vram-lab-ltx-deep] {self._elapsed()} | {final_note}\n")
                self._live_fh.write("------------------------------------------------------------------------------\n")
                self._live_fh.write(f"closed: {_now()}\n")
                self._live_fh.flush()
                try:
                    os.fsync(self._live_fh.fileno())
                except Exception:
                    pass
                self._live_fh.close()
        except Exception as exc:
            if len(self._failures) < 20:
                self._failures.append(f"live log close failed: {type(exc).__name__}: {exc}")
        finally:
            self._live_fh = None

    def _snap_text(self) -> str:
        snap = _torch_cuda_snapshot(self.torch)
        allocated = int(snap.get("allocated", 0) or 0)
        reserved = int(snap.get("reserved", 0) or 0)
        driver_free = int(snap.get("driver_free", 0) or 0)
        driver_total = int(snap.get("driver_total", 0) or 0)
        driver_used = max(0, driver_total - driver_free) if driver_total else 0
        return (
            f"allocated={_fmt_bytes(allocated)}, reserved={_fmt_bytes(reserved)}, "
            f"driver_free={_fmt_bytes(driver_free)}, driver_used={_fmt_bytes(driver_used)}, "
            f"driver_total={_fmt_bytes(driver_total)}"
        )

    def _record(self, event: str, detail: str = "") -> None:
        text = f"{self._elapsed()} | {event}: {detail or ''} | {self._snap_text()}"
        self.events.append(text)
        if len(self.events) > self.max_events:
            # Keep the newest events and a clear marker that earlier lines were trimmed.
            self.events = ["... deep lifecycle events trimmed to latest entries ..."] + self.events[-self.max_events:]
        self._write_live(text)
        if self.echo:
            try:
                print(f"[vram-lab-ltx-deep] {text}", flush=True)
            except Exception:
                pass

    def mark(self, event: str, detail: str = "") -> None:
        self._record(str(event), str(detail or ""))

    def _should_log_call(self, key: str, cls_name: str, arg_text: str = "") -> bool:
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        if count in self._MILESTONES:
            return True
        interesting = (
            "cuda" in arg_text
            or "cpu" in arg_text
            or "meta" in arg_text
            or "Gemma" in cls_name
            or "LTXModel" in cls_name
            or "X0Model" in cls_name
            or "BatchSplit" in cls_name
            or "VideoEncoder" in cls_name
            or "VideoDecoder" in cls_name
            or "VAE" in cls_name
            or "BasicAVTransformerBlock" in cls_name
            or "EmbeddingsProcessor" in cls_name
        )
        return bool(interesting and count <= 20)

    def _format_to_args(self, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
        parts: List[str] = []
        for a in list(args)[:3]:
            try:
                parts.append(str(a))
            except Exception:
                parts.append(type(a).__name__)
        for k in ("device", "dtype", "non_blocking"):
            if k in kwargs:
                try:
                    parts.append(f"{k}={kwargs[k]}")
                except Exception:
                    parts.append(f"{k}=<unprintable>")
        return ", ".join(parts)

    def start(self) -> None:
        if self._installed:
            return
        try:
            self._open_live_file()
            torch = self.torch
            if torch is None:
                import torch as torch  # type: ignore
                self.torch = torch
            module_cls = torch.nn.Module
            self._orig_module_to = module_cls.to
            self._orig_module_apply = module_cls._apply
            logger = self

            def wrapped_to(module_self: Any, *args: Any, **kwargs: Any) -> Any:
                cls_name = f"{module_self.__class__.__module__}.{module_self.__class__.__name__}"
                arg_text = logger._format_to_args(args, kwargs)
                key = f"Module.to:{cls_name}:{arg_text}"
                if logger._should_log_call(key, cls_name, arg_text):
                    logger._record("Module.to", f"{cls_name} call={logger.counts.get(key, 0)} args={arg_text}")
                return logger._orig_module_to(module_self, *args, **kwargs)

            def wrapped_apply(module_self: Any, fn: Any, *args: Any, **kwargs: Any) -> Any:
                cls_name = f"{module_self.__class__.__module__}.{module_self.__class__.__name__}"
                key = f"Module._apply:{cls_name}"
                if logger._should_log_call(key, cls_name, ""):
                    logger._record("Module._apply", f"{cls_name} call={logger.counts.get(key, 0)}")
                return logger._orig_module_apply(module_self, fn, *args, **kwargs)

            module_cls.to = wrapped_to
            module_cls._apply = wrapped_apply

            try:
                self._orig_empty_cache = torch.cuda.empty_cache

                def wrapped_empty_cache(*args: Any, **kwargs: Any) -> Any:
                    logger._record("torch.cuda.empty_cache", "called")
                    return logger._orig_empty_cache(*args, **kwargs)

                torch.cuda.empty_cache = wrapped_empty_cache
            except Exception as exc:
                self._failures.append(f"empty_cache wrap failed: {type(exc).__name__}: {exc}")

            self._installed = True
            self._record("logger_installed", "Module.to / Module._apply / cuda memory polling")
            self._thread = threading.Thread(target=self._poll_loop, name="FrameVisionLTXDeepLifecycleLogger", daemon=True)
            self._thread.start()
        except Exception as exc:
            self._failures.append(f"start failed: {type(exc).__name__}: {exc}")
            self.ctx["deep_lifecycle_logger_enabled"] = f"FAILED: {type(exc).__name__}: {exc}"

    def _poll_loop(self) -> None:
        self._record("logger_start", "memory polling started")
        while not self._stop.wait(self.interval):
            self._record("memory_sample", "")

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            torch = self.torch
            if torch is not None and self._orig_module_to is not None:
                torch.nn.Module.to = self._orig_module_to
            if torch is not None and self._orig_module_apply is not None:
                torch.nn.Module._apply = self._orig_module_apply
            if torch is not None and self._orig_empty_cache is not None:
                torch.cuda.empty_cache = self._orig_empty_cache
        except Exception as exc:
            self._failures.append(f"restore failed: {type(exc).__name__}: {exc}")
        self._record("logger_stop", "restored patched methods")
        self.update_context()
        self._close_live_file("logger_closed: stop() completed")

    def update_context(self, ctx: Optional[Dict[str, Any]] = None) -> None:
        target = ctx if isinstance(ctx, dict) else self.ctx
        target["deep_lifecycle_event_count"] = str(len(self.events))
        target["deep_lifecycle_live_path"] = str(self.live_path) if self.live_path else "none"
        target["deep_lifecycle_logger_failures"] = "none" if not self._failures else " | ".join(self._failures[-20:])
        target["deep_lifecycle_events"] = list(self.events)
        target["deep_lifecycle_top_call_counts"] = " | ".join(
            f"{k}={v}" for k, v in sorted(self.counts.items(), key=lambda kv: kv[1], reverse=True)[:30]
        ) or "none"


def make_deep_lifecycle_logger(
    ctx: Optional[Dict[str, Any]] = None,
    label: str = "ltx_deep",
    torch_module: Any = None,
    echo: bool = False,
    interval: float = 1.0,
    max_events: int = 4000,
    live_path: Optional[Any] = None,
) -> DeepLifecycleLogger:
    return DeepLifecycleLogger(
        ctx=ctx,
        label=label,
        torch_module=torch_module,
        echo=echo,
        interval=interval,
        max_events=max_events,
        live_path=live_path,
    )


class BoundaryTracer:
    def __init__(self, ctx: Optional[Dict[str, Any]] = None, label: str = "boundary", torch_module: Any = None, echo: bool = False) -> None:
        self.ctx = ctx if isinstance(ctx, dict) else {}
        self.label = str(label or "boundary")
        self.torch = torch_module
        self.echo = bool(echo)
        self.ctx["boundary_finder_enabled"] = f"YES: {self.label}"
        self.ctx.setdefault("boundary_trace_stages", [])
        self.ctx.setdefault("boundary_trace_notes", [])

    def mark(self, key: str, note: str = "") -> None:
        snap = _torch_cuda_snapshot(self.torch)
        text = f"{_now()} | {key}: {note or 'n/a'} | {_snapshot_string(snap)}"
        self.ctx[str(key)] = text
        self.ctx.setdefault("boundary_trace_stages", []).append(text)
        for k, v in snap.items():
            self.ctx[f"{key}_{k}"] = _fmt_bytes(v)
            self.ctx[f"{key}_{k}_int"] = int(v)
        if self.echo:
            try:
                print(f"[vram-lab-boundary] {text}", flush=True)
            except Exception:
                pass

    def wrap_bound_method(self, obj: Any, method_name: str, label: Optional[str] = None) -> bool:
        try:
            orig = getattr(obj, method_name)
        except Exception as exc:
            self.ctx.setdefault("boundary_trace_notes", []).append(f"{method_name} missing: {exc}")
            return False
        if not callable(orig) or getattr(orig, "_framevision_boundary_wrapped", False):
            return False
        lbl = str(label or method_name)
        tracer = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            tracer.mark(f"{lbl}_enter", f"enter {lbl}")
            started = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                duration = time.perf_counter() - started
                tracer.ctx[f"{lbl}_duration"] = f"{duration:.3f}s"
                tracer.mark(f"{lbl}_exit", f"exit {lbl}; duration={duration:.3f}s")

        try:
            wrapped._framevision_boundary_wrapped = True  # type: ignore[attr-defined]
            setattr(obj, method_name, wrapped)
            return True
        except Exception as exc:
            self.ctx.setdefault("boundary_trace_notes", []).append(f"wrap {lbl} failed: {type(exc).__name__}: {exc}")
            return False


def make_boundary_tracer(ctx: Optional[Dict[str, Any]] = None, label: str = "boundary", torch_module: Any = None, echo: bool = False) -> BoundaryTracer:
    return BoundaryTracer(ctx=ctx, label=label, torch_module=torch_module, echo=echo)
