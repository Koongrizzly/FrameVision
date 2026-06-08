#!/usr/bin/env python3
"""FrameVision Wan 2.2 VRAM Lab wrapper CLI.

This helper keeps models/wan22 untouched. It prepares the shared VRAM Lab
runtime, monkey-patches Wan pipeline constructors from the outside, then runs the
normal Wan generate.py with its original arguments.

Wan remains the runner. VRAM Lab remains the memory/runtime hook layer.
"""
from __future__ import annotations

import argparse
import gc
import os
import runpy
import subprocess
import sys
import time
import traceback
import threading
import types
from pathlib import Path
from typing import Any, Dict, List

APP_ROOT = Path(__file__).resolve().parents[1]
VRAM_LAB_DIR = APP_ROOT / "tools" / "vram_lab"
REPORT_PATH = VRAM_LAB_DIR / "wan_vram_lab_integration_report.txt"
STAGE_LOG_PATH = VRAM_LAB_DIR / "wan_vram_lab_stage_log.txt"

RUNTIMES: List[Any] = []
CONTEXT: Dict[str, Any] = {}
BOUNDARY_TRACER: Any = None
STAGE_EVENTS: List[Dict[str, Any]] = []
_LAST_STAGE_EVENT: Dict[str, Any] | None = None
_TILE_SUMMARY_SEEN: set[str] = set()

# Normal runs should not spam thousands of decoder leaf/tile events into the UI
# log. Set FV_WAN_VRAM_VERBOSE=1 when deep decoder tracing is needed again.
WAN_STAGE_VERBOSE = str(os.environ.get("FV_WAN_VRAM_VERBOSE", "")).lower().strip() in ("1", "true", "yes", "on")

WAN_VRAM_PROFILE_LIMITS_GB = {
    "24": 22.5,
    "16": 14.5,
    "12": 11.0,
}

# Denoise/sampling uses the same soft-guard idea as the working Wan VAE
# decoder guard, but with two separate thresholds:
# 1) the selected profile limit is the cleanup/start-correcting point;
# 2) the selected profile's near-full-card ceiling gets the 5 second abort timer.
# The runtime driver-free floor is used as an urgent cleanup signal here, not an
# instant abort, because short spikes near full VRAM can recover cleanly.
WAN_DENOISE_SOFT_OVER_LIMIT_SECONDS = 5.0
WAN_DENOISE_FULL_CEILING_GB = {
    "24": 24.0,
    "16": 16.0,
    "12": 12.0,
}
WAN_DENOISE_CEILING_MARGIN_GB = 0.10


def _detect_cuda_card_for_vram_auto() -> tuple[str, float, str]:
    """Best-effort GPU/VRAM detection for --vram-profile auto.

    Returns (gpu_name, total_vram_gb, source). Detection stays local: try
    torch first because this CLI already runs in the Wan environment, then fall
    back to nvidia-smi if torch/CUDA is not ready yet.
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)
            # CUDA_VISIBLE_DEVICES remaps visible device 0, so use 0 for torch.
            try:
                prop = torch.cuda.get_device_properties(0)
            except Exception:
                prop = torch.cuda.get_device_properties(idx)
            name = str(getattr(prop, "name", "CUDA GPU") or "CUDA GPU")
            total = float(getattr(prop, "total_memory", 0) or 0) / (1024.0 ** 3)
            if total > 0:
                return name, total, "torch.cuda"
    except Exception as exc:
        try:
            CONTEXT["wan_vram_auto_torch_error"] = f"{type(exc).__name__}: {exc}"
        except Exception:
            pass

    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if cp.returncode == 0:
            lines = [ln.strip() for ln in (cp.stdout or "").splitlines() if ln.strip()]
            if lines:
                first = lines[0]
                parts = [p.strip() for p in first.split(",")]
                name = parts[0] if parts else "NVIDIA GPU"
                mib = float(parts[1]) if len(parts) > 1 else 0.0
                gb = mib / 1024.0
                if gb > 0:
                    return name, gb, "nvidia-smi"
    except Exception as exc:
        try:
            CONTEXT["wan_vram_auto_nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
        except Exception:
            pass

    return "Unknown GPU", 0.0, "unavailable"


def _resolve_wan_vram_profile(requested: str) -> str:
    """Resolve manual/auto WAN VRAM profile to one of 12/16/24."""
    req = str(requested or "24").strip().lower()
    CONTEXT["wan_vram_profile_requested"] = req
    if req in WAN_VRAM_PROFILE_LIMITS_GB:
        CONTEXT["wan_vram_profile_auto"] = "NO"
        return req
    if req != "auto":
        CONTEXT["wan_vram_profile_auto"] = f"NO: invalid request {req!r}; fallback 24"
        return "24"

    name, total_gb, source = _detect_cuda_card_for_vram_auto()
    if total_gb <= 0:
        resolved = "24"
        note = "AUTO detection failed; fallback 24 profile"
    elif total_gb < 16.0:
        resolved = "12"
        note = "AUTO rule: under 16GB uses 12GB profile"
    # Consumer 24GB cards can report slightly below 24.0 GiB. Treat anything
    # at/above 23.0 GiB as the 24GB class so Auto does not accidentally select
    # the 16GB profile on cards like an RTX 3090.
    elif total_gb < 23.0:
        resolved = "16"
        note = "AUTO rule: 16GB through 22.9GB uses 16GB profile"
    else:
        resolved = "24"
        note = "AUTO rule: 23GB-class and higher uses 24GB profile"

    CONTEXT["wan_vram_profile_auto"] = "YES"
    CONTEXT["wan_vram_auto_gpu_name"] = name
    CONTEXT["wan_vram_auto_total_gb"] = f"{total_gb:.2f}" if total_gb > 0 else "unknown"
    CONTEXT["wan_vram_auto_source"] = source
    CONTEXT["wan_vram_auto_note"] = note
    CONTEXT["wan_vram_auto_resolved_profile"] = resolved
    try:
        print(
            f"[WAN22][VRAM Lab] Auto VRAM profile: {name} ({total_gb:.2f} GB via {source}) -> {resolved} GB profile",
            flush=True,
        )
    except Exception:
        pass
    return resolved


def _wan_profile_name() -> str:
    profile = str(CONTEXT.get("wan_vram_profile", "24") or "24").strip()
    return profile if profile in WAN_VRAM_PROFILE_LIMITS_GB else "24"


def _wan_frame_num_int(default: int = 0) -> int:
    try:
        return int(str(CONTEXT.get("wan_frame_num", default) or default).strip())
    except Exception:
        return int(default)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, str(default)) or str(default)).strip())
    except Exception:
        return float(default)


def _wan_profile_reserved_limit_gb() -> float:
    return float(WAN_VRAM_PROFILE_LIMITS_GB.get(_wan_profile_name(), 22.5))


def _wan_profile_full_ceiling_gb() -> float:
    return float(WAN_DENOISE_FULL_CEILING_GB.get(_wan_profile_name(), 24.0))


def _wan_profile_danger_start_gb() -> float:
    # Start the 5s danger timer very close to the profile/card ceiling.
    # This allows brief 23.9/24.0GB-style spikes on a 24GB card, but prevents
    # sitting at the absolute edge long enough to turn into shared-memory crawl.
    return max(0.0, _wan_profile_full_ceiling_gb() - float(WAN_DENOISE_CEILING_MARGIN_GB))


def _wan_profile_tile_size(out_h: int, out_w: int) -> tuple[int, int]:
    """Return spatial tile sizes for the selected VRAM profile.

    Match the faster post-sampling tiled VAE decode behavior already proven in
    the Wan Turbo VRAM Lab CLI: larger tiles reduce final decode/finalization
    overhead, while 12GB/16GB still keep smaller limits than the 24GB profile.
    """
    profile = _wan_profile_name()
    if out_h >= 352 or out_w >= 624:
        if profile == "12":
            return 24, 32
        if profile == "16":
            return 32, 48
        return 64, 96
    if out_h >= 256 or out_w >= 384:
        if profile == "12":
            return 32, 48
        if profile == "16":
            return 48, 64
        return 96, 128
    if profile == "12":
        return 48, 64
    if profile == "16":
        return 64, 96
    return 128, 160


def _keep_stage_event(key: str) -> bool:
    if WAN_STAGE_VERBOSE:
        return True
    k = str(key or "")
    if "error" in k or "abort" in k:
        return True
    if k.startswith("wan_decoder_layer_"):
        return False
    if k.startswith("wan_denoise_internal_block_detail"):
        return False
    # Normal queue logs should say that tiled VAE decode is active, not print
    # every repeated Conv tile/module call. Full detail is still available with
    # FV_WAN_VRAM_VERBOSE=1.
    if k in (
        "wan_tiled_causalconv3d_start",
        "wan_tiled_causalconv3d_done",
        "wan_tiled_conv2d_start",
        "wan_tiled_conv2d_done",
    ):
        return False
    if k.startswith("wan_internal_vae_frame_"):
        return False
    if k in ("wan_internal_vae_frame_enter", "wan_internal_vae_frame_exit_cpu", "wan_internal_vae_frame_cleanup"):
        return False
    if k.endswith("_tile_cleanup"):
        return False
    return True


def _tile_summary_once(kind: str, module_id: str, shape: Any, tile_h: int, tile_w: int, torch_mod: Any = None) -> None:
    """Emit one readable tile summary per unique tiled decode shape.

    The detailed start/done events are still collected when FV_WAN_VRAM_VERBOSE=1.
    Normal queue logs only need to show when a new tiled decode stage appears.
    """
    try:
        if WAN_STAGE_VERBOSE:
            return
        sig = f"{kind}|{shape}|{tile_h}x{tile_w}"
        if sig in _TILE_SUMMARY_SEEN:
            return
        _TILE_SUMMARY_SEEN.add(sig)
        short_module = str(module_id or "?")
        _stage_event(
            f"wan_{kind}_summary",
            f"active; first use for shape={shape}; tile={tile_h}x{tile_w}; example_module={short_module}",
            torch_mod,
        )
    except Exception:
        pass


def _fmt_bytes(n: int | None) -> str:
    try:
        n = int(n or 0)
    except Exception:
        n = 0
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if abs(v) < 1024.0 or u == units[-1]:
            return f"{v:.2f} {u}" if u != "B" else f"{int(v)} B"
        v /= 1024.0
    return f"{n} B"


def _cuda_snapshot(torch_mod: Any) -> str:
    try:
        if torch_mod is not None and torch_mod.cuda.is_available():
            free = total = 0
            try:
                free, total = torch_mod.cuda.mem_get_info()
            except Exception:
                pass
            return (
                f"allocated={_fmt_bytes(torch_mod.cuda.memory_allocated())}, "
                f"reserved={_fmt_bytes(torch_mod.cuda.memory_reserved())}, "
                f"driver_free={_fmt_bytes(free)}, driver_total={_fmt_bytes(total)}"
            )
    except Exception:
        pass
    return "n/a"


def _cuda_numbers(torch_mod: Any) -> Dict[str, int]:
    """Return raw CUDA memory numbers for delta/peak analysis."""
    data = {"allocated": 0, "reserved": 0, "driver_free": 0, "driver_total": 0, "driver_used": 0}
    try:
        if torch_mod is not None and torch_mod.cuda.is_available():
            try:
                free, total = torch_mod.cuda.mem_get_info()
            except Exception:
                free, total = 0, 0
            allocated = int(torch_mod.cuda.memory_allocated())
            reserved = int(torch_mod.cuda.memory_reserved())
            data.update({
                "allocated": allocated,
                "reserved": int(torch_mod.cuda.memory_reserved()),
                "driver_free": int(free or 0),
                "driver_total": int(total or 0),
                "driver_used": int((total or 0) - (free or 0)),
            })
    except Exception:
        pass
    return data


def _fmt_delta(n: int | None) -> str:
    try:
        n = int(n or 0)
    except Exception:
        n = 0
    sign = "+" if n >= 0 else "-"
    return sign + _fmt_bytes(abs(n))


def _tensor_summary(obj: Any, limit: int = 10) -> str:
    """Small, safe summary of tensor devices/shapes in args/returns."""
    out: List[str] = []
    seen: set[int] = set()

    def visit(x: Any, path: str, depth: int = 0) -> None:
        if len(out) >= limit or depth > 3:
            return
        try:
            oid = id(x)
            if oid in seen:
                return
            seen.add(oid)
        except Exception:
            pass
        try:
            if hasattr(x, "shape") and hasattr(x, "device"):
                shape = tuple(getattr(x, "shape", ()))
                device = getattr(x, "device", "?")
                dtype = getattr(x, "dtype", "?")
                out.append(f"{path}: tensor shape={shape} dtype={dtype} device={device}")
                return
        except Exception:
            pass
        try:
            if isinstance(x, dict):
                for k, v in list(x.items())[:6]:
                    visit(v, f"{path}.{k}", depth + 1)
            elif isinstance(x, (list, tuple)):
                for i, v in enumerate(list(x)[:6]):
                    visit(v, f"{path}[{i}]", depth + 1)
        except Exception:
            pass

    visit(obj, "obj")
    return "; ".join(out) if out else "no tensors detected"




def _append_stage_event_live(event: Dict[str, Any]) -> None:
    """Best-effort live stage log so results survive even if final report writing fails."""
    try:
        STAGE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not STAGE_LOG_PATH.exists():
            STAGE_LOG_PATH.write_text(
                "FrameVision Wan 2.2 VRAM Lab Stage Log\n"
                f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                encoding="utf-8",
            )
        dur = event.get("duration", "n/a")
        peak_driver = _fmt_bytes(event.get("peak_driver_used")) if event.get("peak_driver_used") else "n/a"
        peak_reserved = _fmt_bytes(event.get("peak_reserved")) if event.get("peak_reserved") else "n/a"
        min_free = _fmt_bytes(event.get("min_driver_free")) if event.get("min_driver_free") else "n/a"
        with STAGE_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(
                f"#{int(event.get('idx', 0)):03d} {event.get('time')} {event.get('key')} | {event.get('note')} | "
                f"cuda={event.get('cuda')} | "
                f"delta_alloc={_fmt_delta(event.get('delta_allocated'))} | "
                f"delta_reserved={_fmt_delta(event.get('delta_reserved'))} | "
                f"delta_driver_used={_fmt_delta(event.get('delta_driver_used'))} | "
                f"duration={dur} | peak_driver_used={peak_driver} | "
                f"peak_reserved={peak_reserved} | min_driver_free={min_free}\n"
            )
    except Exception:
        pass

def _stage_event(key: str, note: str = "", torch_mod: Any = None, extra: Dict[str, Any] | None = None) -> None:
    """Append a robust stage event and mirror the most important values into CONTEXT."""
    global _LAST_STAGE_EVENT
    now = time.perf_counter()
    nums = _cuda_numbers(torch_mod)
    prev = _LAST_STAGE_EVENT or {}
    prev_nums = prev.get("cuda_numbers") if isinstance(prev, dict) else None
    if not isinstance(prev_nums, dict):
        prev_nums = nums
    event = {
        "idx": len(STAGE_EVENTS) + 1,
        "time": time.strftime("%H:%M:%S"),
        "perf": now,
        "key": str(key),
        "note": str(note or ""),
        "cuda": _cuda_snapshot(torch_mod),
        "cuda_numbers": nums,
        "delta_allocated": nums.get("allocated", 0) - int(prev_nums.get("allocated", 0) or 0),
        "delta_reserved": nums.get("reserved", 0) - int(prev_nums.get("reserved", 0) or 0),
        "delta_driver_used": nums.get("driver_used", 0) - int(prev_nums.get("driver_used", 0) or 0),
    }
    if extra:
        event.update(extra)

    if not _keep_stage_event(str(key)):
        try:
            CONTEXT["wan_stage_logger_enabled"] = "YES: summary mode (set FV_WAN_VRAM_VERBOSE=1 for decoder/tile trace)"
            CONTEXT["wan_stage_suppressed_detail_events"] = str(int(CONTEXT.get("wan_stage_suppressed_detail_events", "0") or 0) + 1)
            CONTEXT["wan_stage_last_suppressed_event"] = str(key)
        except Exception:
            pass
        return

    STAGE_EVENTS.append(event)
    # Do not let non-CUDA bookkeeping marks reset the CUDA delta baseline.
    try:
        if any(int(nums.get(k, 0) or 0) for k in ("allocated", "reserved", "driver_free", "driver_total", "driver_used")):
            _LAST_STAGE_EVENT = event
    except Exception:
        _LAST_STAGE_EVENT = event
    try:
        CONTEXT["wan_stage_logger_enabled"] = "YES: internal CLI logger"
        CONTEXT["wan_stage_event_count"] = str(len(STAGE_EVENTS))
        CONTEXT["wan_stage_last_event"] = f"#{event['idx']} {event['key']}"
        CONTEXT[f"stage_{key}_cuda"] = event["cuda"]
    except Exception:
        pass
    try:
        _append_stage_event_live(event)
    except Exception:
        pass
    # Console echo makes this visible in the normal UI log too.
    try:
        print(
            f"[wan-vram-stage] #{event['idx']:03d} {event['key']} | {event['note']} | "
            f"{event['cuda']} | Δalloc={_fmt_delta(event['delta_allocated'])}, "
            f"Δreserved={_fmt_delta(event['delta_reserved'])}, Δdriver_used={_fmt_delta(event['delta_driver_used'])}",
            flush=True,
        )
    except Exception:
        pass


class _StageWatch:
    """Context manager that records entry/exit and peak CUDA memory inside a stage."""
    def __init__(self, key: str, note: str = "", torch_mod: Any = None, interval: float = 0.25):
        self.key = str(key)
        self.note = str(note or "")
        self.torch_mod = torch_mod
        self.interval = float(interval)
        self.started = 0.0
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.peak_reserved = 0
        self.peak_allocated = 0
        self.peak_driver_used = 0
        self.min_driver_free: int | None = None
        self.samples = 0

    def _sample_loop(self) -> None:
        while not self.stop.wait(self.interval):
            nums = _cuda_numbers(self.torch_mod)
            self.samples += 1
            self.peak_reserved = max(self.peak_reserved, nums.get("reserved", 0))
            self.peak_allocated = max(self.peak_allocated, nums.get("allocated", 0))
            self.peak_driver_used = max(self.peak_driver_used, nums.get("driver_used", 0))
            free = nums.get("driver_free", 0)
            if free:
                self.min_driver_free = free if self.min_driver_free is None else min(self.min_driver_free, free)

    def __enter__(self):
        self.started = time.perf_counter()
        nums = _cuda_numbers(self.torch_mod)
        self.peak_reserved = nums.get("reserved", 0)
        self.peak_allocated = nums.get("allocated", 0)
        self.peak_driver_used = nums.get("driver_used", 0)
        self.min_driver_free = nums.get("driver_free", 0) or None
        _stage_event(self.key + ":enter", self.note, self.torch_mod)
        try:
            self.thread = threading.Thread(target=self._sample_loop, name=f"wan-vram-watch-{self.key}", daemon=True)
            self.thread.start()
        except Exception:
            self.thread = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=1.0)
        except Exception:
            pass
        duration = time.perf_counter() - self.started
        status = "error" if exc_type is not None else "ok"
        extra = {
            "duration": duration,
            "status": status,
            "peak_reserved": self.peak_reserved,
            "peak_allocated": self.peak_allocated,
            "peak_driver_used": self.peak_driver_used,
            "min_driver_free": self.min_driver_free or 0,
            "samples": self.samples,
        }
        CONTEXT[f"stage_{self.key}_duration"] = f"{duration:.3f}s"
        CONTEXT[f"stage_{self.key}_peak_reserved"] = _fmt_bytes(self.peak_reserved)
        CONTEXT[f"stage_{self.key}_peak_allocated"] = _fmt_bytes(self.peak_allocated)
        CONTEXT[f"stage_{self.key}_peak_driver_used"] = _fmt_bytes(self.peak_driver_used)
        CONTEXT[f"stage_{self.key}_min_driver_free"] = _fmt_bytes(self.min_driver_free or 0)
        _stage_event(self.key + ":exit", f"{self.note}; duration={duration:.3f}s; status={status}", self.torch_mod, extra)
        return False


def _stage_watch(key: str, note: str = "", torch_mod: Any = None) -> _StageWatch:
    return _StageWatch(key, note, torch_mod)



def _wan_denoise_sampling_active() -> bool:
    return str(CONTEXT.get("wan_denoise_sampling_active", "NO") or "NO").upper() == "YES"


def _bump_context_counter(key: str, amount: int = 1) -> None:
    try:
        CONTEXT[key] = str(int(CONTEXT.get(key, "0") or 0) + int(amount or 0))
    except Exception:
        CONTEXT[key] = str(amount or 1)



def _detect_wan_attention_backend() -> str:
    """Best-effort probe for the Wan attention backend used by the local repo."""
    try:
        import importlib
        attn = importlib.import_module("wan.modules.attention")
        path = str(getattr(attn, "__file__", "n/a"))
        disabled = bool(getattr(attn, "_DISABLE_FLASH", False))
        fa3 = bool(getattr(attn, "FLASH_ATTN_3_AVAILABLE", False))
        fa2 = bool(getattr(attn, "FLASH_ATTN_2_AVAILABLE", False))
        if disabled:
            backend = "SDPA fallback (FlashAttention disabled by FV_WAN_DISABLE_FLASH_ATTENTION)"
        elif fa3:
            backend = "FlashAttention 3"
        elif fa2:
            backend = "FlashAttention 2"
        else:
            backend = "SDPA fallback (FlashAttention not available)"
        CONTEXT["wan_attention_backend"] = backend
        CONTEXT["wan_attention_backend_file"] = path
        return f"{backend} | {path}"
    except Exception as e:
        msg = f"unknown: {type(e).__name__}: {e}"
        CONTEXT["wan_attention_backend"] = msg
        CONTEXT["wan_attention_backend_file"] = "n/a"
        return msg


def _log_wan_attention_backend(torch_mod: Any = None) -> None:
    """Log the Wan attention backend to console, stage log and report context."""
    try:
        msg = _detect_wan_attention_backend()
        _stage_event("wan_attention_backend", msg, torch_mod)
        try:
            print(f"[WAN22][VRAM Lab] Attention backend: {msg}", flush=True)
        except Exception:
            pass
    except Exception as e:
        CONTEXT["wan_attention_backend"] = f"probe failed: {type(e).__name__}: {e}"
        CONTEXT["wan_attention_backend_file"] = "n/a"

def _protect_runtime_block_eviction_during_wan_denoise(runtime: Any) -> None:
    """Prevent lower-level VRAM Lab hooks from CPU-moving Wan blocks mid-forward.

    Wan denoise can briefly touch the full VRAM ceiling. During that live forward,
    moving a hooked `model.blocks.*` module to CPU can split attention inputs and
    weights across CPU/CUDA and crash. The Wan-specific denoise guard therefore
    owns the abort timing; runtime block eviction is blocked while sampling is
    active. Allocator-only cleanup is still allowed.
    """
    if runtime is None or getattr(runtime, "_framevision_wan_denoise_eviction_protected", False):
        return

    def _allocator_only(rt: Any, reason: str) -> None:
        try:
            cleanup = getattr(rt, "_cleanup_cuda", None)
            if callable(cleanup):
                cleanup(reason)
        except Exception:
            pass

    try:
        orig_unload_other = getattr(runtime, "_unload_other_blocks", None)
        if callable(orig_unload_other):
            setattr(runtime, "_framevision_orig_unload_other_blocks", orig_unload_other)

            def safe_unload_other(self, keep_name: str = "") -> None:
                if _wan_denoise_sampling_active():
                    _bump_context_counter("wan_denoise_protected_runtime_eviction_skips")
                    CONTEXT["wan_denoise_protected_runtime_eviction_last"] = f"_unload_other_blocks skipped; keep={keep_name or 'n/a'}"
                    return
                return orig_unload_other(keep_name)

            runtime._unload_other_blocks = types.MethodType(safe_unload_other, runtime)  # type: ignore[attr-defined]
    except Exception as exc:
        CONTEXT["wan_denoise_protected_runtime_eviction_error"] = f"_unload_other_blocks: {type(exc).__name__}: {exc}"

    try:
        orig_trim_hot = getattr(runtime, "_trim_hot_blocks", None)
        if callable(orig_trim_hot):
            setattr(runtime, "_framevision_orig_trim_hot_blocks", orig_trim_hot)

            def safe_trim_hot(self, keep_name: str = "") -> None:
                if _wan_denoise_sampling_active():
                    _bump_context_counter("wan_denoise_protected_runtime_eviction_skips")
                    CONTEXT["wan_denoise_protected_runtime_eviction_last"] = f"_trim_hot_blocks skipped; keep={keep_name or 'n/a'}"
                    return
                return orig_trim_hot(keep_name)

            runtime._trim_hot_blocks = types.MethodType(safe_trim_hot, runtime)  # type: ignore[attr-defined]
    except Exception as exc:
        CONTEXT["wan_denoise_protected_runtime_eviction_error"] = f"_trim_hot_blocks: {type(exc).__name__}: {exc}"

    try:
        orig_emergency = getattr(runtime, "_emergency_cleanup_if_needed", None)
        if callable(orig_emergency):
            setattr(runtime, "_framevision_orig_emergency_cleanup_if_needed", orig_emergency)

            def safe_emergency(self) -> None:
                if _wan_denoise_sampling_active():
                    _bump_context_counter("wan_denoise_protected_runtime_eviction_skips")
                    CONTEXT["wan_denoise_protected_runtime_eviction_last"] = "_emergency_cleanup_if_needed skipped; allocator-only cleanup"
                    _allocator_only(self, "wan_denoise_protected_emergency_allocator_only")
                    return
                return orig_emergency()

            runtime._emergency_cleanup_if_needed = types.MethodType(safe_emergency, runtime)  # type: ignore[attr-defined]
    except Exception as exc:
        CONTEXT["wan_denoise_protected_runtime_eviction_error"] = f"_emergency_cleanup_if_needed: {type(exc).__name__}: {exc}"

    try:
        runtime._framevision_wan_denoise_eviction_protected = True
    except Exception:
        pass
    CONTEXT["wan_denoise_protected_runtime_eviction"] = "YES: block-moving cleanup disabled while Wan sampling is active"
    CONTEXT.setdefault("wan_denoise_protected_runtime_eviction_skips", "0")
    CONTEXT.setdefault("wan_denoise_protected_runtime_eviction_last", "none")


def _protect_all_runtimes_for_wan_denoise() -> None:
    for rt in list(RUNTIMES):
        try:
            _protect_runtime_block_eviction_during_wan_denoise(rt)
        except Exception as exc:
            CONTEXT["wan_denoise_protected_runtime_eviction_error"] = f"protect_all: {type(exc).__name__}: {exc}"


WAN_DENOISE_STREAMED_BLOCKS: List[Dict[str, Any]] = []
WAN_DENOISE_INTERNAL_PROFILE_RECORDS: List[Dict[str, Any]] = []
WAN_DENOISE_INTERNAL_PROFILE_AGG: Dict[str, Dict[str, Any]] = {}


def _wan_internal_profiler_enabled() -> bool:
    """True when the deeper active-block profiler should run.

    This is diagnostic-only. It does not change Wan math, residency, or VAE
    decode. Default: enabled for the 12 GB profile because that is where the
    remaining peak comes from one active block forward.
    """
    flag = str(os.environ.get("FV_WAN_INTERNAL_BLOCK_PROFILER", "") or "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    return _wan_profile_name() == "12"


def _wan_internal_profiler_target_blocks() -> set[int]:
    raw = str(os.environ.get("FV_WAN_INTERNAL_BLOCK_PROFILE_BLOCKS", "0,15,29") or "0,15,29")
    out: set[int] = set()
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            pass
    return out or {0, 15, 29}


def _wan_internal_profiler_max_calls() -> int:
    try:
        return max(1, int(os.environ.get("FV_WAN_INTERNAL_BLOCK_PROFILE_MAX_CALLS", "240") or "240"))
    except Exception:
        return 240


def _wan_internal_profiler_record(label: str, duration: float, before: Dict[str, int], after: Dict[str, int], peak_alloc: int, peak_reserved: int) -> None:
    """Store a compact record and update aggregate offender summaries."""
    try:
        rec = {
            "label": str(label),
            "duration": float(duration or 0.0),
            "start_allocated": int(before.get("allocated", 0) or 0),
            "start_reserved": int(before.get("reserved", 0) or 0),
            "start_driver_used": int(before.get("driver_used", 0) or 0),
            "start_driver_free": int(before.get("driver_free", 0) or 0),
            "end_allocated": int(after.get("allocated", 0) or 0),
            "end_reserved": int(after.get("reserved", 0) or 0),
            "end_driver_used": int(after.get("driver_used", 0) or 0),
            "end_driver_free": int(after.get("driver_free", 0) or 0),
            "peak_allocated": int(peak_alloc or 0),
            "peak_reserved": int(peak_reserved or 0),
        }
        rec["delta_allocated"] = rec["end_allocated"] - rec["start_allocated"]
        rec["delta_reserved"] = rec["end_reserved"] - rec["start_reserved"]
        rec["delta_driver_used"] = rec["end_driver_used"] - rec["start_driver_used"]
        WAN_DENOISE_INTERNAL_PROFILE_RECORDS.append(rec)
        max_records = _wan_internal_profiler_max_calls()
        if len(WAN_DENOISE_INTERNAL_PROFILE_RECORDS) > max_records:
            del WAN_DENOISE_INTERNAL_PROFILE_RECORDS[: len(WAN_DENOISE_INTERNAL_PROFILE_RECORDS) - max_records]

        agg = WAN_DENOISE_INTERNAL_PROFILE_AGG.setdefault(str(label), {
            "calls": 0,
            "total_duration": 0.0,
            "max_peak_reserved": 0,
            "max_peak_allocated": 0,
            "max_delta_reserved": 0,
            "max_delta_allocated": 0,
            "max_delta_driver_used": 0,
        })
        agg["calls"] = int(agg.get("calls", 0) or 0) + 1
        agg["total_duration"] = float(agg.get("total_duration", 0.0) or 0.0) + float(duration or 0.0)
        agg["max_peak_reserved"] = max(int(agg.get("max_peak_reserved", 0) or 0), rec["peak_reserved"])
        agg["max_peak_allocated"] = max(int(agg.get("max_peak_allocated", 0) or 0), rec["peak_allocated"])
        agg["max_delta_reserved"] = max(int(agg.get("max_delta_reserved", 0) or 0), rec["delta_reserved"])
        agg["max_delta_allocated"] = max(int(agg.get("max_delta_allocated", 0) or 0), rec["delta_allocated"])
        agg["max_delta_driver_used"] = max(int(agg.get("max_delta_driver_used", 0) or 0), rec["delta_driver_used"])

        CONTEXT["wan_denoise_internal_profiler_calls"] = str(sum(int(v.get("calls", 0) or 0) for v in WAN_DENOISE_INTERNAL_PROFILE_AGG.values()))
        CONTEXT["wan_denoise_internal_profiler_records"] = str(len(WAN_DENOISE_INTERNAL_PROFILE_RECORDS))

        worst_reserved = max(WAN_DENOISE_INTERNAL_PROFILE_AGG.items(), key=lambda kv: int(kv[1].get("max_peak_reserved", 0) or 0))
        worst_alloc = max(WAN_DENOISE_INTERNAL_PROFILE_AGG.items(), key=lambda kv: int(kv[1].get("max_peak_allocated", 0) or 0))
        CONTEXT["wan_denoise_internal_profiler_worst_reserved"] = f"{worst_reserved[0]} peak_reserved={_fmt_bytes(worst_reserved[1].get('max_peak_reserved'))} calls={worst_reserved[1].get('calls')}"
        CONTEXT["wan_denoise_internal_profiler_worst_allocated"] = f"{worst_alloc[0]} peak_allocated={_fmt_bytes(worst_alloc[1].get('max_peak_allocated'))} calls={worst_alloc[1].get('calls')}"
        CONTEXT["wan_denoise_internal_profiler_last"] = (
            f"{label}: duration={duration:.4f}s peak_reserved={_fmt_bytes(peak_reserved)} "
            f"peak_allocated={_fmt_bytes(peak_alloc)} Δreserved={_fmt_delta(rec['delta_reserved'])}"
        )
    except Exception as exc:
        CONTEXT["wan_denoise_internal_profiler_error"] = f"record failed: {type(exc).__name__}: {exc}"


def _wan_internal_profiler_summary_lines(limit: int = 18) -> List[str]:
    lines: List[str] = []
    try:
        ranked = sorted(
            WAN_DENOISE_INTERNAL_PROFILE_AGG.items(),
            key=lambda kv: int(kv[1].get("max_peak_reserved", 0) or 0),
            reverse=True,
        )[: int(limit or 18)]
        for label, agg in ranked:
            calls = int(agg.get("calls", 0) or 0)
            total = float(agg.get("total_duration", 0.0) or 0.0)
            avg = total / calls if calls else 0.0
            lines.append(
                f"{label}: calls={calls}; avg={avg:.4f}s; "
                f"peak_reserved={_fmt_bytes(agg.get('max_peak_reserved'))}; "
                f"peak_allocated={_fmt_bytes(agg.get('max_peak_allocated'))}; "
                f"max_Δreserved={_fmt_delta(agg.get('max_delta_reserved'))}; "
                f"max_Δallocated={_fmt_delta(agg.get('max_delta_allocated'))}; "
                f"max_Δdriver={_fmt_delta(agg.get('max_delta_driver_used'))}"
            )
    except Exception as exc:
        lines.append(f"summary failed: {type(exc).__name__}: {exc}")
    return lines



# ---------------------------------------------------------------------------
# Wan profile-aware denoise FFN chunking
# ---------------------------------------------------------------------------

def _parse_bytes_context(value: Any) -> int:
    """Best-effort parser for strings ending in a formatted byte value."""
    try:
        s = str(value or "").strip()
        if not s or s == "n/a":
            return 0
        parts = s.split()
        if len(parts) >= 2:
            num = float(parts[-2])
            unit = parts[-1].upper()
            mult = 1
            if unit.startswith("KB"):
                mult = 1024
            elif unit.startswith("MB"):
                mult = 1024 ** 2
            elif unit.startswith("GB"):
                mult = 1024 ** 3
            return int(num * mult)
    except Exception:
        pass
    return 0


def _wan_ffn_chunking_enabled() -> bool:
    """True when the FFN inside each active Wan block should be token-chunked."""
    flag = str(os.environ.get("FV_WAN_FFN_CHUNKING", "") or "").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    if flag in ("1", "true", "yes", "on"):
        return True
    # FFN chunking is now profile-aware and enabled for all Wan VRAM Lab profiles.
    # It solved the 12 GB active-block spike and has very low overhead on the 3090,
    # while also reducing random long-run spikes for 16/24 GB users.
    return _wan_profile_name() in ("12", "16", "24")


def _wan_ffn_default_chunk_size() -> int:
    profile = _wan_profile_name()
    if profile == "12":
        return 2048
    if profile == "16":
        return 4096
    if profile == "24":
        return 8192
    return 4096


def _wan_ffn_chunk_size() -> int:
    default = _wan_ffn_default_chunk_size()
    try:
        return max(1, int(os.environ.get("FV_WAN_FFN_CHUNK_SIZE", str(default)) or str(default)))
    except Exception:
        return default


def _wan_ffn_chunk_dim() -> int:
    try:
        return int(os.environ.get("FV_WAN_FFN_CHUNK_DIM", "1") or "1")
    except Exception:
        return 1


def _tensor_like_first_arg(args: Any) -> Any:
    try:
        if args and hasattr(args[0], "shape") and hasattr(args[0], "narrow"):
            return args[0]
    except Exception:
        pass
    return None


def _install_wan_denoise_ffn_chunking(component_map: Dict[str, Any], torch_mod: Any) -> None:
    """Chunk direct block.ffn modules over the token/sequence dimension.

    The internal profiler identified block.ffn as the top reserved-memory
    offender in the 12 GB profile. This wrapper is active only during Wan
    denoise sampling. It is profile-aware and now defaults ON for 12/16/24 GB
    profiles, using larger chunk sizes for larger VRAM profiles.
    """
    try:
        if not _wan_ffn_chunking_enabled():
            CONTEXT["wan_denoise_ffn_chunking"] = f"NO: disabled by env/profile {_wan_profile_name()}"
            return
        if not component_map:
            CONTEXT["wan_denoise_ffn_chunking"] = "NO: no Wan model components found"
            return

        chunk_size = _wan_ffn_chunk_size()
        chunk_dim = _wan_ffn_chunk_dim()
        wrapped = 0
        errors: List[str] = []

        for comp_name, comp in list(component_map.items()):
            blocks = getattr(comp, "blocks", None)
            if blocks is None:
                continue
            for idx, block in enumerate(list(blocks)):
                try:
                    ffn = getattr(block, "ffn", None)
                    if ffn is None or getattr(ffn, "_framevision_wan_ffn_chunk_wrapped", False):
                        continue
                    orig_forward = ffn.forward
                    label = f"{comp_name}.blocks.{idx}.ffn"

                    def make_ffn_forward(orig: Any, module_label: str):
                        def chunked_ffn_forward(*args, **kwargs):
                            if not _wan_denoise_sampling_active():
                                return orig(*args, **kwargs)
                            x = _tensor_like_first_arg(args)
                            if x is None:
                                return orig(*args, **kwargs)
                            try:
                                dim = chunk_dim
                                if dim < 0:
                                    dim = int(x.dim()) + dim
                                if dim < 0 or dim >= int(x.dim()):
                                    return orig(*args, **kwargs)
                                total = int(x.shape[dim])
                                if total <= chunk_size:
                                    return orig(*args, **kwargs)
                            except Exception:
                                return orig(*args, **kwargs)

                            _bump_context_counter("wan_denoise_ffn_chunking_calls")
                            CONTEXT["wan_denoise_ffn_chunking_last"] = f"{module_label}: total={total}, chunk={chunk_size}, dim={dim}"
                            out = None
                            pieces = []
                            used_prealloc = False
                            peak_reserved = 0
                            peak_alloc = 0
                            try:
                                if torch_mod is not None and torch_mod.cuda.is_available():
                                    try:
                                        torch_mod.cuda.reset_peak_memory_stats()
                                    except Exception:
                                        pass
                                for start in range(0, total, chunk_size):
                                    length = min(chunk_size, total - start)
                                    x_chunk = x.narrow(dim, start, length).contiguous()
                                    y = orig(*((x_chunk,) + tuple(args[1:])), **kwargs)
                                    if not hasattr(y, "shape"):
                                        CONTEXT["wan_denoise_ffn_chunking_last"] = f"{module_label}: non-tensor output fallback"
                                        _bump_context_counter("wan_denoise_ffn_chunking_fallbacks")
                                        return orig(*args, **kwargs)
                                    if out is None:
                                        try:
                                            expected = list(y.shape)
                                            expected[dim] = total
                                            if list(expected) == list(x.shape):
                                                out = x.new_empty(tuple(expected))
                                                used_prealloc = True
                                        except Exception:
                                            out = None
                                    if out is not None:
                                        out.narrow(dim, start, length).copy_(y)
                                    else:
                                        pieces.append(y)
                                    try:
                                        del y, x_chunk
                                    except Exception:
                                        pass
                                    try:
                                        if torch_mod is not None and torch_mod.cuda.is_available():
                                            peak_alloc = max(peak_alloc, int(torch_mod.cuda.max_memory_allocated()))
                                            peak_reserved = max(peak_reserved, int(torch_mod.cuda.max_memory_reserved()))
                                    except Exception:
                                        pass
                                result = out if out is not None else torch_mod.cat(pieces, dim=dim)
                                chunks = (total + chunk_size - 1) // chunk_size
                                _bump_context_counter("wan_denoise_ffn_chunking_chunks", chunks)
                                if used_prealloc:
                                    _bump_context_counter("wan_denoise_ffn_chunking_prealloc_outputs")
                                CONTEXT["wan_denoise_ffn_chunking_last"] = (
                                    f"{module_label}: chunks={chunks}; peak_reserved={_fmt_bytes(peak_reserved)}; "
                                    f"peak_allocated={_fmt_bytes(peak_alloc)}"
                                )
                                old_peak = _parse_bytes_context(CONTEXT.get("wan_denoise_ffn_chunking_worst_reserved", "0 B"))
                                if peak_reserved > old_peak:
                                    CONTEXT["wan_denoise_ffn_chunking_worst_reserved"] = f"{module_label} {_fmt_bytes(peak_reserved)}"
                                old_alloc = _parse_bytes_context(CONTEXT.get("wan_denoise_ffn_chunking_worst_allocated", "0 B"))
                                if peak_alloc > old_alloc:
                                    CONTEXT["wan_denoise_ffn_chunking_worst_allocated"] = f"{module_label} {_fmt_bytes(peak_alloc)}"
                                return result
                            except Exception as exc:
                                CONTEXT["wan_denoise_ffn_chunking_error"] = f"{module_label}: {type(exc).__name__}: {exc}"
                                _bump_context_counter("wan_denoise_ffn_chunking_fallbacks")
                                return orig(*args, **kwargs)
                        try:
                            chunked_ffn_forward._framevision_wan_ffn_chunk_wrapped = True  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        return chunked_ffn_forward

                    ffn.forward = make_ffn_forward(orig_forward, label)  # type: ignore[method-assign]
                    try:
                        ffn._framevision_wan_ffn_chunk_wrapped = True
                        ffn._framevision_wan_ffn_chunk_label = label
                    except Exception:
                        pass
                    wrapped += 1
                except Exception as exc:
                    if len(errors) < 10:
                        errors.append(f"{comp_name}.blocks.{idx}.ffn: {type(exc).__name__}: {exc}")

        CONTEXT["wan_denoise_ffn_chunking"] = f"YES: installed for profile {_wan_profile_name()} (profile-aware)" if wrapped else "NO: no FFN modules wrapped"
        CONTEXT["wan_denoise_ffn_chunking_wrapped_modules"] = str(wrapped)
        CONTEXT["wan_denoise_ffn_chunking_chunk_size"] = str(chunk_size)
        CONTEXT["wan_denoise_ffn_chunking_dim"] = str(chunk_dim)
        CONTEXT.setdefault("wan_denoise_ffn_chunking_calls", "0")
        CONTEXT.setdefault("wan_denoise_ffn_chunking_chunks", "0")
        CONTEXT.setdefault("wan_denoise_ffn_chunking_fallbacks", "0")
        CONTEXT.setdefault("wan_denoise_ffn_chunking_prealloc_outputs", "0")
        CONTEXT.setdefault("wan_denoise_ffn_chunking_worst_reserved", "n/a")
        CONTEXT.setdefault("wan_denoise_ffn_chunking_worst_allocated", "n/a")
        CONTEXT.setdefault("wan_denoise_ffn_chunking_last", "none")
        CONTEXT.setdefault("wan_denoise_ffn_chunking_error", "n/a")
        if errors:
            CONTEXT["wan_denoise_ffn_chunking_error"] = " | ".join(errors)
        _stage_event(
            "wan_denoise_ffn_chunking_installed",
            f"wrapped={wrapped}; chunk={chunk_size}; dim={chunk_dim}; profile={_wan_profile_name()}",
            torch_mod,
        )
    except Exception as exc:
        CONTEXT["wan_denoise_ffn_chunking"] = f"FAILED: {type(exc).__name__}: {exc}"
        CONTEXT["wan_denoise_ffn_chunking_error"] = f"{type(exc).__name__}: {exc}"


def _install_wan_denoise_internal_block_profiler(component_map: Dict[str, Any], torch_mod: Any) -> None:
    """Wrap direct child modules inside selected Wan blocks to locate active-block peaks.

    This is a diagnostic profiler for the 12 GB profile. Whole-block streaming
    already proved the full model is not resident. This profiler identifies
    whether the remaining peak is self-attn, cross-attn, FFN/MLP, or another
    direct child of an active block. It intentionally avoids console spam and
    writes compact summaries to the report/stage log.
    """
    try:
        if not _wan_internal_profiler_enabled():
            CONTEXT["wan_denoise_internal_profiler"] = "NO: disabled for this profile/env"
            return
        if not component_map:
            CONTEXT["wan_denoise_internal_profiler"] = "NO: no Wan model components found"
            return
        target_blocks = _wan_internal_profiler_target_blocks()
        max_calls = _wan_internal_profiler_max_calls()
        wrapped = 0
        blocks_seen = 0
        errors: List[str] = []
        for comp_name, comp in list(component_map.items()):
            blocks = getattr(comp, "blocks", None)
            if blocks is None:
                continue
            for idx, block in enumerate(list(blocks)):
                if idx not in target_blocks:
                    continue
                blocks_seen += 1
                try:
                    for child_name, child in list(block.named_children()):
                        try:
                            if getattr(child, "_framevision_wan_internal_profile_wrapped", False):
                                continue
                            orig_forward = getattr(child, "forward", None)
                            if not callable(orig_forward):
                                continue
                            label = f"{comp_name}.blocks.{idx}.{child_name}({child.__class__.__name__})"

                            def make_child_forward(orig: Any, module_label: str):
                                def profiled_child_forward(*args, **kwargs):
                                    if (not _wan_denoise_sampling_active()) or int(CONTEXT.get("wan_denoise_internal_profiler_calls", "0") or 0) >= max_calls:
                                        return orig(*args, **kwargs)
                                    before = _cuda_numbers(torch_mod)
                                    start = time.perf_counter()
                                    try:
                                        if torch_mod is not None and torch_mod.cuda.is_available():
                                            try:
                                                torch_mod.cuda.reset_peak_memory_stats()
                                            except Exception:
                                                pass
                                        out = orig(*args, **kwargs)
                                        return out
                                    finally:
                                        duration = time.perf_counter() - start
                                        peak_alloc = 0
                                        peak_reserved = 0
                                        try:
                                            if torch_mod is not None and torch_mod.cuda.is_available():
                                                peak_alloc = int(torch_mod.cuda.max_memory_allocated())
                                                peak_reserved = int(torch_mod.cuda.max_memory_reserved())
                                        except Exception:
                                            pass
                                        after = _cuda_numbers(torch_mod)
                                        _wan_internal_profiler_record(module_label, duration, before, after, peak_alloc, peak_reserved)
                                try:
                                    profiled_child_forward._framevision_wan_internal_profile_wrapped = True  # type: ignore[attr-defined]
                                except Exception:
                                    pass
                                return profiled_child_forward

                            child.forward = make_child_forward(orig_forward, label)  # type: ignore[method-assign]
                            try:
                                child._framevision_wan_internal_profile_wrapped = True
                                child._framevision_wan_internal_profile_label = label
                            except Exception:
                                pass
                            wrapped += 1
                        except Exception as exc:
                            if len(errors) < 8:
                                errors.append(f"{comp_name}.blocks.{idx}.{child_name}: {type(exc).__name__}: {exc}")
                except Exception as exc:
                    if len(errors) < 8:
                        errors.append(f"{comp_name}.blocks.{idx}: {type(exc).__name__}: {exc}")
        CONTEXT["wan_denoise_internal_profiler"] = "YES: direct child module profiler installed" if wrapped else "NO: no child modules wrapped"
        CONTEXT["wan_denoise_internal_profiler_target_blocks"] = ",".join(str(x) for x in sorted(target_blocks))
        CONTEXT["wan_denoise_internal_profiler_target_blocks_seen"] = str(blocks_seen)
        CONTEXT["wan_denoise_internal_profiler_wrapped_modules"] = str(wrapped)
        CONTEXT["wan_denoise_internal_profiler_max_calls"] = str(max_calls)
        CONTEXT.setdefault("wan_denoise_internal_profiler_calls", "0")
        CONTEXT.setdefault("wan_denoise_internal_profiler_records", "0")
        CONTEXT.setdefault("wan_denoise_internal_profiler_error", "n/a")
        if errors:
            CONTEXT["wan_denoise_internal_profiler_error"] = " | ".join(errors)
        _stage_event(
            "wan_denoise_internal_profiler_installed",
            f"wrapped={wrapped}; target_blocks={CONTEXT.get('wan_denoise_internal_profiler_target_blocks')}; max_calls={max_calls}",
            torch_mod,
        )
    except Exception as exc:
        CONTEXT["wan_denoise_internal_profiler"] = f"FAILED: {type(exc).__name__}: {exc}"
        CONTEXT["wan_denoise_internal_profiler_error"] = f"{type(exc).__name__}: {exc}"



def _wan_denoise_streaming_enabled() -> bool:
    """True when Wan denoise blocks should be externally CPU/GPU streamed.

    Earlier builds streamed blocks only for 12/16 GB. That made 121-frame
    diagnostics work, but left 24 GB high-frame runs using normal resident
    denoise blocks; long clips could then fill dedicated VRAM and crawl into
    shared memory during step 1. VRAM Lab should protect the denoise phase too,
    so block streaming is now ON by default for every Wan VRAM Lab profile.

    Overrides:
      FV_WAN_DENOISE_STREAM_BLOCKS=0  -> disable all Wan block streaming
      FV_WAN_DENOISE_STREAM_24=0      -> allow old fast resident 24 GB path
      FV_WAN_DENOISE_STREAM_24=1      -> force 24 GB streaming explicitly
    """
    profile = _wan_profile_name()
    if not _env_flag("FV_WAN_DENOISE_STREAM_BLOCKS", True):
        CONTEXT["wan_denoise_streaming_strategy"] = "disabled by FV_WAN_DENOISE_STREAM_BLOCKS=0"
        return False
    if profile in ("12", "16"):
        CONTEXT["wan_denoise_streaming_strategy"] = f"profile {profile}: always stream denoise blocks"
        return True
    raw24 = str(os.environ.get("FV_WAN_DENOISE_STREAM_24", "") or "").strip().lower()
    if raw24 in ("0", "false", "no", "off"):
        CONTEXT["wan_denoise_streaming_strategy"] = "profile 24: old resident path forced by FV_WAN_DENOISE_STREAM_24=0"
        return False
    CONTEXT["wan_denoise_streaming_strategy"] = (
        f"profile 24: stream denoise blocks by default; frames={_wan_frame_num_int(0)}"
    )
    return True


def _first_tensor_device(obj: Any) -> Any:
    try:
        if hasattr(obj, "device"):
            return obj.device
        if isinstance(obj, (list, tuple)):
            for x in obj:
                dev = _first_tensor_device(x)
                if dev is not None:
                    return dev
        if isinstance(obj, dict):
            for x in obj.values():
                dev = _first_tensor_device(x)
                if dev is not None:
                    return dev
    except Exception:
        pass
    return None


def _module_param_device(module: Any) -> str:
    try:
        for p in module.parameters(recurse=True):
            return str(getattr(p, "device", "unknown"))
    except Exception:
        pass
    return "unknown"


def _cleanup_after_streamed_block(torch_mod: Any) -> None:
    try:
        if torch_mod is not None and torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()
    except Exception:
        pass


def _install_wan_denoise_block_streaming(component_map: Dict[str, Any], torch_mod: Any) -> None:
    """Install Wan-specific denoise block CPU/GPU streaming wrappers.

    This is intentionally external to the Wan repo. It wraps the already-built
    Wan transformer blocks and is active by default for all Wan VRAM Lab
    profiles. This prevents high-frame 24 GB runs from keeping all denoise
    blocks resident and spilling into shared memory during step 1. It avoids the
    previous CPU/CUDA crash by never unloading a block until its own forward has
    returned.
    """
    try:
        if not _wan_denoise_streaming_enabled():
            CONTEXT["wan_denoise_block_streaming"] = f"NO: profile {_wan_profile_name()} uses normal resident denoise blocks"
            return
        if not component_map:
            CONTEXT["wan_denoise_block_streaming"] = "NO: no Wan model components found"
            return

        installed = 0
        errors: List[str] = []
        for comp_name, comp in list(component_map.items()):
            try:
                blocks = getattr(comp, "blocks", None)
                if blocks is None:
                    continue
                for idx, block in enumerate(list(blocks)):
                    try:
                        if getattr(block, "_framevision_wan_stream_wrapped", False):
                            continue
                        block_name = f"{comp_name}.blocks.{idx}"
                        orig_forward = block.forward

                        def make_forward(orig: Any, module: Any, name: str):
                            def streamed_forward(*args, **kwargs):
                                CONTEXT["wan_denoise_block_streaming_last"] = f"{name}: enter"
                                _bump_context_counter("wan_denoise_block_streaming_calls")
                                target_device = _first_tensor_device({"args": args, "kwargs": kwargs})
                                if target_device is None:
                                    target_device = "cuda"
                                try:
                                    if str(_module_param_device(module)) != str(target_device):
                                        module.to(target_device)
                                        _bump_context_counter("wan_denoise_block_streaming_cuda_loads")
                                except Exception as exc:
                                    CONTEXT["wan_denoise_block_streaming_error"] = f"{name} cuda load failed: {type(exc).__name__}: {exc}"
                                    raise
                                try:
                                    return orig(*args, **kwargs)
                                finally:
                                    try:
                                        module.to("cpu")
                                        _bump_context_counter("wan_denoise_block_streaming_cpu_offloads")
                                        CONTEXT["wan_denoise_block_streaming_last"] = f"{name}: offloaded to CPU"
                                        _cleanup_after_streamed_block(torch_mod)
                                    except Exception as exc:
                                        CONTEXT["wan_denoise_block_streaming_error"] = f"{name} cpu offload failed: {type(exc).__name__}: {exc}"
                            try:
                                streamed_forward._framevision_wan_stream_wrapped = True  # type: ignore[attr-defined]
                            except Exception:
                                pass
                            return streamed_forward

                        block.forward = make_forward(orig_forward, block, block_name)  # type: ignore[method-assign]
                        try:
                            block._framevision_wan_stream_wrapped = True
                            block._framevision_wan_stream_name = block_name
                        except Exception:
                            pass
                        WAN_DENOISE_STREAMED_BLOCKS.append({"name": block_name, "module": block})
                        installed += 1
                    except Exception as exc:
                        if len(errors) < 10:
                            errors.append(f"{comp_name}.blocks.{idx}: {type(exc).__name__}: {exc}")
            except Exception as exc:
                if len(errors) < 10:
                    errors.append(f"{comp_name}: {type(exc).__name__}: {exc}")

        CONTEXT["wan_denoise_block_streaming"] = f"YES: installed for profile {_wan_profile_name()}" if installed else "NO: no blocks wrapped"
        CONTEXT["wan_denoise_block_streaming_wrapped_blocks"] = str(installed)
        CONTEXT.setdefault("wan_denoise_block_streaming_cuda_loads", "0")
        CONTEXT.setdefault("wan_denoise_block_streaming_cpu_offloads", "0")
        CONTEXT.setdefault("wan_denoise_block_streaming_calls", "0")
        if errors:
            CONTEXT["wan_denoise_block_streaming_error"] = "; ".join(errors)
    except Exception as exc:
        CONTEXT["wan_denoise_block_streaming"] = f"FAILED: {type(exc).__name__}: {exc}"


def _prepare_wan_denoise_blocks_for_sampling(torch_mod: Any) -> None:
    """Move wrapped Wan denoise blocks to CPU before the sampling loop starts."""
    if not _wan_denoise_streaming_enabled() or not WAN_DENOISE_STREAMED_BLOCKS:
        return
    moved = 0
    total = 0
    last = "none"
    try:
        for item in list(WAN_DENOISE_STREAMED_BLOCKS):
            total += 1
            name = str(item.get("name", "block"))
            block = item.get("module")
            if block is None:
                continue
            try:
                if _module_param_device(block) != "cpu":
                    block.to("cpu")
                    moved += 1
                    last = name
            except Exception as exc:
                CONTEXT["wan_denoise_block_streaming_error"] = f"prepare {name}: {type(exc).__name__}: {exc}"
        try:
            gc.collect()
            if torch_mod is not None and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            pass
        CONTEXT["wan_denoise_block_streaming_prepare"] = f"moved={moved}, total={total}, last={last}"
        CONTEXT["wan_denoise_block_streaming_last"] = f"sampling start prepare: moved {moved}/{total} blocks to CPU"
        _stage_event("wan_denoise_block_streaming_prepare", CONTEXT["wan_denoise_block_streaming_prepare"], torch_mod)
        try:
            print(f"[WAN22][VRAM Lab] denoise block streaming active: moved {moved}/{total} blocks to CPU before sampling", flush=True)
        except Exception:
            pass
    except Exception as exc:
        CONTEXT["wan_denoise_block_streaming_error"] = f"prepare failed: {type(exc).__name__}: {exc}"



class _WanDenoiseStepProfiler:
    """Low-overhead Wan sampling-step memory logger.

    This does not change Wan generation. It wraps tqdm loops that match the
    requested sample-step count and records CUDA/hook deltas around each visible
    denoise step. Normal console output is one readable line per step; deeper
    detail is written to the stage log/report, with extra console detail only
    when FV_WAN_VRAM_VERBOSE=1.
    """

    def __init__(self, torch_mod: Any, steps_hint: int = 0, live_path: Path | None = None, echo_verbose: bool = False):
        self.torch = torch_mod
        self.steps_hint = int(steps_hint or 0)
        self.live_path = live_path
        self.echo_verbose = bool(echo_verbose)
        self.enabled = False
        self._orig_tqdm = None
        self._orig_auto_tqdm = None
        self._patched_tqdm_attrs: List[tuple[Any, str, Any]] = []
        self._lock = threading.RLock()
        self.profiled_loop_count = 0
        self.current_loop_index = 0
        self.current_step_index = -1
        self.steps: List[Dict[str, Any]] = []
        self.failures: List[str] = []
        self._live_header_written = False
        self._profile_limit_bytes = int(_wan_profile_reserved_limit_gb() * (1024 ** 3))
        self._profile_ceiling_bytes = int(_wan_profile_full_ceiling_gb() * (1024 ** 3))
        self._danger_start_bytes = int(_wan_profile_danger_start_gb() * (1024 ** 3))
        self._soft_over_limit_seconds = float(WAN_DENOISE_SOFT_OVER_LIMIT_SECONDS)
        self._shared_guard_enabled = _env_flag("FV_WAN_SHARED_MEM_GUARD", True)
        self._shared_guard_free_floor = int(_env_float("FV_WAN_SHARED_MEM_FREE_FLOOR_GB", 0.50) * (1024 ** 3))
        self._shared_guard_seconds = max(0.5, _env_float("FV_WAN_SHARED_MEM_GUARD_SECONDS", 6.0))
        self._shared_guard_exit_code = int(_env_float("FV_WAN_SHARED_MEM_EXIT_CODE", 88))
        self._driver_floor_bytes = 0
        self._sampling_started = False
        self._abort_reason = ""
        self._correction_count = 0
        self._soft_abort_count = 0
        self._sampling_start_perf = 0.0
        self._pending_runtime_cleanups: List[str] = []
        CONTEXT["wan_denoise_step_profiler_enabled"] = "NO"
        CONTEXT["wan_denoise_step_profiler_status"] = "not started"

    def _write_live(self, text: str) -> None:
        try:
            path = self.live_path or STAGE_LOG_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                if not self._live_header_written:
                    f.write("\nWan denoise/sampling step profiler\n")
                    f.write("------------------------------------------------------------------------------\n")
                    self._live_header_written = True
                f.write(text.rstrip() + "\n")
        except Exception as exc:
            if len(self.failures) < 20:
                self.failures.append(f"live step log write failed: {type(exc).__name__}: {exc}")

    def _runtime_snapshot(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "pre": 0,
            "post": 0,
            "loads": 0,
            "unloads": 0,
            "forced": 0,
            "cache": 0,
            "emergency": 0,
            "hot_trim": 0,
            "emergency_trim": 0,
            "hooked": 0,
            "cuda_blocks": 0,
            "active": "n/a",
        }
        active: List[str] = []
        cuda_blocks = 0
        try:
            for rt in list(RUNTIMES):
                try:
                    rt.update_context(CONTEXT)
                except Exception:
                    pass
                data["pre"] += int(getattr(rt, "pre_calls", 0) or 0)
                data["post"] += int(getattr(rt, "post_calls", 0) or 0)
                data["loads"] += int(getattr(rt, "block_load_count", 0) or 0)
                data["unloads"] += int(getattr(rt, "block_unload_count", 0) or 0)
                data["forced"] += int(getattr(rt, "forced_unload_count", 0) or 0)
                data["cache"] += int(getattr(rt, "cache_cleanup_count", 0) or 0)
                data["emergency"] += int(getattr(rt, "emergency_cleanup_count", 0) or 0)
                data["hot_trim"] += int(getattr(rt, "hot_block_trim_count", 0) or 0)
                data["emergency_trim"] += int(getattr(rt, "emergency_trim_count", 0) or 0)
                blocks = list(getattr(rt, "blocks", []) or [])
                data["hooked"] += len(blocks)
                for block in blocks:
                    try:
                        module = block.module() if callable(getattr(block, "module", None)) else None
                        if module is not None:
                            is_cuda = False
                            try:
                                for param in module.parameters(recurse=False):
                                    if getattr(param, "is_cuda", False):
                                        is_cuda = True
                                        break
                            except Exception:
                                pass
                            if not is_cuda:
                                try:
                                    for buf in module.buffers(recurse=False):
                                        if getattr(buf, "is_cuda", False):
                                            is_cuda = True
                                            break
                                except Exception:
                                    pass
                            if is_cuda:
                                cuda_blocks += 1
                                active.append(str(getattr(block, "name", "?")))
                    except Exception:
                        pass
                active_name = str(getattr(rt, "active_block_name", "n/a") or "n/a")
                if active_name and active_name != "n/a":
                    data["active"] = active_name
        except Exception as exc:
            if len(self.failures) < 20:
                self.failures.append(f"runtime snapshot failed: {type(exc).__name__}: {exc}")
        data["cuda_blocks"] = cuda_blocks
        data["cuda_block_names"] = active[:12]
        return data

    def _driver_floor_from_context(self) -> int:
        if self._driver_floor_bytes > 0:
            return self._driver_floor_bytes
        # Prefer the runtime policy value when present; it is already bytes.
        try:
            for rt in list(RUNTIMES):
                floor = int(getattr(rt, "emergency_driver_free_floor_bytes", 0) or 0)
                if floor > 0:
                    self._driver_floor_bytes = floor
                    return floor
        except Exception:
            pass
        return 0

    def _request_runtime_cleanup(self, reason: str, active_name: str = "") -> None:
        """Ask CUDA/runtime cleanup to recover headroom without breaking live Wan forwards.

        Important: this method is called by a monitor thread while Wan may be
        inside model.blocks.N.forward(). Moving any hooked block to CPU during
        that active forward can produce mixed CPU/CUDA tensors in attention/MLP
        layers. Therefore the live-step path is allocator-only. Runtime block
        eviction is deferred until the step boundary, after the current forward
        has returned.
        """
        self._correction_count += 1
        CONTEXT["wan_denoise_soft_guard_corrections"] = str(self._correction_count)

        in_live_step = int(getattr(self, "current_step_index", -1) or -1) >= 0
        active_name = str(active_name or "")
        if in_live_step:
            pending = list(getattr(self, "_pending_runtime_cleanups", []) or [])
            pending.append(f"{reason}; active={active_name or 'n/a'}")
            self._pending_runtime_cleanups = pending[-20:]
            CONTEXT["wan_denoise_deferred_runtime_cleanups"] = str(len(self._pending_runtime_cleanups))
            try:
                self._write_live(
                    f"soft guard cleanup deferred | reason={reason} | active={active_name or 'n/a'} | "
                    f"allocator-only during active forward | cuda={_cuda_snapshot(self.torch)}"
                )
            except Exception:
                pass
        elif not _wan_denoise_sampling_active():
            self._run_runtime_eviction_cleanup(reason, keep_name=active_name)
        else:
            # Sampling is active even if the profiler is between explicit step
            # records. Keep this allocator-only; the runtime wrappers protect the
            # lower-level forward hooks too.
            CONTEXT["wan_denoise_deferred_runtime_cleanup_last"] = f"{reason}; sampling-active allocator-only"

        # Safe allocator-only cleanup. This does not move module parameters, so it
        # cannot create mixed-device tensors inside the active Wan block.
        self._allocator_cleanup(reason)

    def _allocator_cleanup(self, reason: str = "") -> None:
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
            if len(self.failures) < 20:
                self.failures.append(f"denoise cuda cleanup failed {reason}: {type(exc).__name__}: {exc}")

    def _run_runtime_eviction_cleanup(self, reason: str = "", keep_name: str = "") -> None:
        """Run block-moving runtime cleanup only at a safe step boundary."""
        try:
            for rt in list(RUNTIMES):
                try:
                    if hasattr(rt, "_trim_hot_blocks"):
                        rt._trim_hot_blocks(keep_name or "")  # type: ignore[attr-defined]
                except Exception as exc:
                    if len(self.failures) < 20:
                        self.failures.append(f"deferred denoise soft trim failed: {type(exc).__name__}: {exc}")
                try:
                    if hasattr(rt, "_emergency_cleanup_if_needed"):
                        rt._emergency_cleanup_if_needed()  # type: ignore[attr-defined]
                except Exception as exc:
                    if len(self.failures) < 20:
                        self.failures.append(f"deferred denoise emergency cleanup failed: {type(exc).__name__}: {exc}")
                try:
                    if hasattr(rt, "_cleanup_cuda"):
                        rt._cleanup_cuda(reason)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception as exc:
            if len(self.failures) < 20:
                self.failures.append(f"deferred runtime cleanup loop failed: {type(exc).__name__}: {exc}")

    def _flush_deferred_runtime_cleanup(self, reason: str = "step_boundary") -> None:
        pending = list(getattr(self, "_pending_runtime_cleanups", []) or [])
        if not pending:
            return
        if _wan_denoise_sampling_active():
            CONTEXT["wan_denoise_deferred_runtime_cleanups"] = str(len(pending))
            CONTEXT["wan_denoise_deferred_runtime_cleanup_last"] = f"{pending[-1]} | held; sampling still active"
            self._allocator_cleanup(reason)
            return
        self._pending_runtime_cleanups = []
        CONTEXT["wan_denoise_deferred_runtime_cleanups"] = "0"
        CONTEXT["wan_denoise_deferred_runtime_cleanup_last"] = pending[-1]
        try:
            self._write_live(f"soft guard deferred runtime cleanup running | count={len(pending)} | reason={reason} | cuda={_cuda_snapshot(self.torch)}")
        except Exception:
            pass
        if _wan_denoise_sampling_active():
            CONTEXT["wan_denoise_deferred_runtime_cleanup_last"] = f"{pending[-1]} | held until sampling end; allocator-only at {reason}"
            self._allocator_cleanup(reason)
            return
        self._run_runtime_eviction_cleanup(reason, keep_name="")
        self._allocator_cleanup(reason)

    def _set_abort_reason(self, reason: str) -> None:
        if not self._abort_reason:
            self._abort_reason = str(reason)
            CONTEXT["wan_denoise_soft_guard_abort_reason"] = self._abort_reason
            try:
                print(f"[WAN22][VRAM Lab] ABORT: {self._abort_reason}", flush=True)
            except Exception:
                pass
            self._write_live(f"ABORT | {self._abort_reason} | cuda={_cuda_snapshot(self.torch)}")
            _stage_event("wan_denoise_soft_guard_abort", self._abort_reason, self.torch)

    def _raise_if_abort_requested(self) -> None:
        if self._abort_reason:
            raise RuntimeError("VRAM Lab Wan denoise guard abort: " + self._abort_reason)

    def _infer_total(self, iterable: Any, kwargs: Dict[str, Any]) -> int:
        try:
            if kwargs.get("total", None) is not None:
                return int(kwargs.get("total") or 0)
        except Exception:
            pass
        try:
            return int(len(iterable))
        except Exception:
            return 0

    def _should_profile_loop(self, iterable: Any, kwargs: Dict[str, Any]) -> bool:
        total = self._infer_total(iterable, kwargs)
        desc = str(kwargs.get("desc", "") or "").lower()
        if self.steps_hint > 0 and total == self.steps_hint:
            return True
        if any(word in desc for word in ("sampling", "sample", "denois", "diffusion")):
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
                if iterable is None or not parent._should_profile_loop(iterable, kwargs):
                    return parent._orig_tqdm(iterable, *args, **kwargs)
                bar = parent._orig_tqdm(iterable, *args, **kwargs)

                def iterator():
                    parent.profiled_loop_count += 1
                    loop_index = parent.profiled_loop_count
                    parent._sampling_loop_start(loop_index, kwargs)
                    try:
                        for step_index, item in enumerate(bar):
                            parent.begin_step(loop_index, step_index)
                            try:
                                yield item
                            finally:
                                parent.end_step(loop_index, step_index)
                    finally:
                        parent._sampling_loop_end(loop_index)

                return iterator()

            setattr(profiled_tqdm, "_framevision_wan_step_profiler", True)
            setattr(tqdm_mod, "tqdm", profiled_tqdm)
            try:
                import tqdm.auto as tqdm_auto_mod  # type: ignore
                self._orig_auto_tqdm = getattr(tqdm_auto_mod, "tqdm", None)
                setattr(tqdm_auto_mod, "tqdm", profiled_tqdm)
            except Exception:
                self._orig_auto_tqdm = None
            self._patch_existing_wan_tqdm_refs(profiled_tqdm)
            self.enabled = True
            CONTEXT["wan_denoise_step_profiler_enabled"] = "YES"
            CONTEXT["wan_denoise_step_profiler_status"] = f"installed; waiting for sampling tqdm loop with total={self.steps_hint or 'unknown'}"
            CONTEXT["wan_denoise_step_profiler_verbose_console"] = "YES" if self.echo_verbose else "NO"
            CONTEXT["wan_denoise_step_profiler_live_path"] = str(self.live_path or STAGE_LOG_PATH)
        except Exception as exc:
            CONTEXT["wan_denoise_step_profiler_enabled"] = f"FAILED: {type(exc).__name__}: {exc}"
            self.failures.append(f"install failed: {type(exc).__name__}: {exc}")

    def _patch_existing_wan_tqdm_refs(self, replacement: Any) -> None:
        """Patch Wan modules that imported tqdm as a direct function before us."""
        try:
            originals = [x for x in (self._orig_tqdm, self._orig_auto_tqdm) if x is not None]
            for mod_name, mod in list(sys.modules.items()):
                if mod is None or not (str(mod_name).startswith("wan") or "wan" in str(mod_name).lower()):
                    continue
                try:
                    cur = getattr(mod, "tqdm", None)
                except Exception:
                    continue
                if cur in originals and cur is not replacement:
                    try:
                        self._patched_tqdm_attrs.append((mod, "tqdm", cur))
                        setattr(mod, "tqdm", replacement)
                    except Exception:
                        pass
            CONTEXT["wan_denoise_existing_tqdm_refs_patched"] = str(len(self._patched_tqdm_attrs))
        except Exception as exc:
            if len(self.failures) < 20:
                self.failures.append(f"existing tqdm ref patch failed: {type(exc).__name__}: {exc}")

    def stop(self) -> None:
        try:
            for mod, attr, original in list(self._patched_tqdm_attrs):
                try:
                    if getattr(getattr(mod, attr, None), "_framevision_wan_step_profiler", False):
                        setattr(mod, attr, original)
                except Exception:
                    pass
            self._patched_tqdm_attrs.clear()
            if self._orig_tqdm is not None:
                import tqdm as tqdm_mod  # type: ignore
                if getattr(getattr(tqdm_mod, "tqdm", None), "_framevision_wan_step_profiler", False):
                    setattr(tqdm_mod, "tqdm", self._orig_tqdm)
            if self._orig_auto_tqdm is not None:
                import tqdm.auto as tqdm_auto_mod  # type: ignore
                if getattr(getattr(tqdm_auto_mod, "tqdm", None), "_framevision_wan_step_profiler", False):
                    setattr(tqdm_auto_mod, "tqdm", self._orig_auto_tqdm)
        except Exception as exc:
            self.failures.append(f"restore tqdm failed: {type(exc).__name__}: {exc}")
        self.update_context(CONTEXT)

    def _sampling_loop_start(self, loop_index: int, kwargs: Dict[str, Any]) -> None:
        if not self._sampling_started:
            self._sampling_started = True
            self._sampling_start_perf = time.perf_counter()
        try:
            for rt in list(RUNTIMES):
                try:
                    rt.update_context(CONTEXT)
                except Exception:
                    pass
        except Exception:
            pass
        CONTEXT["wan_denoise_sampling_active"] = "YES"
        _protect_all_runtimes_for_wan_denoise()
        _prepare_wan_denoise_blocks_for_sampling(self.torch)
        rt = self._runtime_snapshot()
        snap = _cuda_numbers(self.torch)
        floor = self._driver_floor_from_context()
        note = (
            f"loop={loop_index}; profile={CONTEXT.get('wan_vram_profile','n/a')} "
            f"limit={_wan_profile_reserved_limit_gb():.1f}GB; frames={CONTEXT.get('wan_frame_num','n/a')}; "
            f"size={CONTEXT.get('wan_size','n/a')}; steps={CONTEXT.get('wan_sample_steps','n/a')}; "
            f"guidance={CONTEXT.get('wan_sample_guide_scale','n/a')}; seed={CONTEXT.get('wan_base_seed','n/a')}; "
            f"hooks={'YES' if int(rt.get('hooked',0) or 0) else 'NO'} hooked_blocks={rt.get('hooked',0)}; "
            f"hot_budget={CONTEXT.get('vram_hot_block_budget','n/a')}; driver_floor={_fmt_bytes(floor)}; "
            f"cleanup_limit={_fmt_bytes(self._profile_limit_bytes)}; full_ceiling={_fmt_bytes(self._profile_ceiling_bytes)}; "
            f"danger_timer_start={_fmt_bytes(self._danger_start_bytes)}; soft_over_limit={self._soft_over_limit_seconds:.1f}s"
        )
        CONTEXT["wan_denoise_sampling_start"] = note
        CONTEXT["wan_denoise_soft_guard_limit"] = _fmt_bytes(self._profile_limit_bytes)
        CONTEXT["wan_denoise_soft_guard_full_ceiling"] = _fmt_bytes(self._profile_ceiling_bytes)
        CONTEXT["wan_denoise_soft_guard_danger_start"] = _fmt_bytes(self._danger_start_bytes)
        CONTEXT["wan_denoise_soft_guard_seconds"] = f"{self._soft_over_limit_seconds:.1f}s"
        CONTEXT["wan_denoise_shared_memory_guard"] = "YES" if self._shared_guard_enabled else "NO"
        CONTEXT["wan_denoise_shared_memory_guard_free_floor"] = _fmt_bytes(self._shared_guard_free_floor)
        CONTEXT["wan_denoise_shared_memory_guard_seconds"] = f"{self._shared_guard_seconds:.1f}s"
        CONTEXT["wan_denoise_soft_guard_abort_reason"] = "none"
        CONTEXT["wan_denoise_soft_guard_recovered_aborts"] = "0"
        CONTEXT["wan_denoise_soft_guard_last_recovered_abort"] = "none"
        CONTEXT["wan_denoise_soft_guard_corrections"] = "0"
        CONTEXT["wan_denoise_deferred_runtime_cleanups"] = "0"
        CONTEXT["wan_denoise_deferred_runtime_cleanup_last"] = "none"
        CONTEXT["wan_denoise_sampling_start_cuda"] = _cuda_snapshot(self.torch)
        _stage_event("wan_denoise_sampling_start", note, self.torch)
        self._write_live(f"sampling start | {note} | cuda={_cuda_snapshot(self.torch)}")
        try:
            print(f"[WAN22][VRAM Lab] sampling steps started: {CONTEXT.get('wan_sample_steps','?')} steps, profile {CONTEXT.get('wan_vram_profile','?')} (cleanup above {_wan_profile_reserved_limit_gb():.1f} GB, abort only near {_wan_profile_full_ceiling_gb():.1f} GB for {self._soft_over_limit_seconds:.0f}s)", flush=True)
        except Exception:
            pass

    def _sampling_loop_end(self, loop_index: int) -> None:
        CONTEXT["wan_denoise_sampling_active"] = "NO"
        try:
            self._flush_deferred_runtime_cleanup(f"wan_denoise_loop_{loop_index}_end")
        except Exception as exc:
            if len(self.failures) < 20:
                self.failures.append(f"loop-end deferred cleanup failed: {type(exc).__name__}: {exc}")
        self.update_context(CONTEXT)
        note = (
            f"loop={loop_index}; total_step_time={CONTEXT.get('wan_denoise_total_step_time','n/a')}; "
            f"decision={CONTEXT.get('wan_denoise_profile_decision','n/a')}"
        )
        _stage_event("wan_denoise_sampling_end", note, self.torch)
        self._write_live(f"sampling end | {note} | cuda={_cuda_snapshot(self.torch)}")
        try:
            print(f"[WAN22][VRAM Lab] sampling steps finished: {CONTEXT.get('wan_denoise_profile_decision','n/a')}", flush=True)
        except Exception:
            pass

    def begin_step(self, loop_index: int, step_index: int) -> None:
        self._raise_if_abort_requested()
        with self._lock:
            self.current_loop_index = int(loop_index)
            self.current_step_index = int(step_index)
            try:
                if self.torch is not None and self.torch.cuda.is_available():
                    self.torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            start_cuda = _cuda_numbers(self.torch)
            start_rt = self._runtime_snapshot()
            rec: Dict[str, Any] = {
                "loop": int(loop_index),
                "step": int(step_index),
                "started": time.perf_counter(),
                "start_cuda": start_cuda,
                "end_cuda": {},
                "start_runtime": start_rt,
                "end_runtime": {},
                "peak_allocated": start_cuda.get("allocated", 0),
                "peak_reserved": start_cuda.get("reserved", 0),
                "min_driver_free": start_cuda.get("driver_free", 0),
                "stop_event": threading.Event(),
                "monitor_samples": 0,
                "soft_guard_corrections": 0,
                "soft_guard_over_seconds": 0.0,
                "soft_guard_abort": "",
                "summary": "",
            }
            self.steps.append(rec)
            CONTEXT["wan_denoise_step_profiler_status"] = f"profiling loop {loop_index}, step {step_index + 1}"

            def monitor() -> None:
                over_since = 0.0
                shared_low_since = 0.0
                cleanup_after = 0.0
                while not rec["stop_event"].wait(0.10):
                    nums = _cuda_numbers(self.torch)
                    now = time.perf_counter()
                    rec["monitor_samples"] = int(rec.get("monitor_samples", 0) or 0) + 1
                    rec["peak_allocated"] = max(int(rec.get("peak_allocated", 0) or 0), nums.get("allocated", 0))
                    rec["peak_reserved"] = max(int(rec.get("peak_reserved", 0) or 0), nums.get("reserved", 0))
                    free = nums.get("driver_free", 0)
                    if free:
                        old = int(rec.get("min_driver_free", 0) or 0)
                        rec["min_driver_free"] = free if old <= 0 else min(old, free)

                    # Real shared-memory crawl guard. The soft profile guard can
                    # wait until a step boundary, but that is too late when a
                    # high-frame step has already fallen into Windows shared GPU
                    # memory and is crawling. If actual driver-free memory stays
                    # near zero for several seconds, stop the helper process fast.
                    if self._shared_guard_enabled and free and self._shared_guard_free_floor > 0 and free <= self._shared_guard_free_floor:
                        if shared_low_since <= 0.0:
                            shared_low_since = now
                            CONTEXT["wan_denoise_shared_memory_guard_last"] = (
                                f"low driver-free entered at step {step_index + 1}: free={_fmt_bytes(free)} "
                                f"<= floor={_fmt_bytes(self._shared_guard_free_floor)}"
                            )
                            self._write_live(
                                f"shared-memory guard low-free | loop={loop_index} step={step_index + 1} | "
                                f"free={_fmt_bytes(free)} <= floor={_fmt_bytes(self._shared_guard_free_floor)} | cuda={_snapshot_text(nums)}"
                            )
                        low_for = max(0.0, now - shared_low_since)
                        rec["shared_low_seconds"] = max(float(rec.get("shared_low_seconds", 0.0) or 0.0), low_for)
                        if low_for >= self._shared_guard_seconds:
                            reason = (
                                f"shared-memory crawl guard: driver_free stayed <= {_fmt_bytes(self._shared_guard_free_floor)} "
                                f"for {low_for:.2f}s during sampling step {step_index + 1}; "
                                f"cuda={_snapshot_text(nums)}"
                            )
                            rec["soft_guard_abort"] = reason
                            CONTEXT["wan_denoise_shared_memory_guard_abort"] = reason
                            self._set_abort_reason(reason)
                            try:
                                self._write_live(f"shared-memory guard hard exit | code={self._shared_guard_exit_code} | {reason}")
                            except Exception:
                                pass
                            os._exit(int(self._shared_guard_exit_code))
                    else:
                        shared_low_since = 0.0

                    floor = int(self._driver_floor_from_context() or 0)
                    limit = int(self._profile_limit_bytes or 0)
                    ceiling = int(self._profile_ceiling_bytes or 0)
                    danger_start = int(self._danger_start_bytes or ceiling or 0)
                    reserved = int(nums.get("reserved", 0) or 0)
                    driver_used = int(nums.get("driver_used", 0) or 0)
                    pressure = max(reserved, driver_used)

                    # Cleanup starts at the selected profile guard (22.5/14.5/10.5GB),
                    # or when the runtime driver-free floor is crossed. Crossing the
                    # floor is no longer an instant abort; it is an urgent correction
                    # signal. Actual abort waits until the near-full profile/card
                    # ceiling persists for the configured soft window.
                    needs_cleanup = bool((limit and pressure > limit) or (floor and free and free <= floor))
                    in_danger_zone = bool(danger_start and pressure >= danger_start)

                    if needs_cleanup and now >= cleanup_after:
                        rec["soft_guard_corrections"] = int(rec.get("soft_guard_corrections", 0) or 0) + 1
                        active_name = str((self._runtime_snapshot() or {}).get("active", "") or "")
                        self._request_runtime_cleanup(f"wan_denoise_soft_guard_step_{step_index + 1}", active_name)
                        cleanup_after = now + 1.0
                        if not rec.get("soft_guard_first_cleanup"):
                            rec["soft_guard_first_cleanup"] = _snapshot_text(nums)
                            self._write_live(
                                f"soft guard cleanup | loop={loop_index} step={step_index + 1} | "
                                f"pressure={_fmt_bytes(pressure)} > cleanup={_fmt_bytes(limit)} "
                                f"or free={_fmt_bytes(free)} <= floor={_fmt_bytes(floor)} | cuda={_snapshot_text(nums)}"
                            )

                    if in_danger_zone:
                        if over_since <= 0.0:
                            over_since = now
                            rec["soft_guard_first_over"] = _snapshot_text(nums)
                            self._write_live(
                                f"soft guard danger zone | loop={loop_index} step={step_index + 1} | "
                                f"pressure={_fmt_bytes(pressure)} >= danger={_fmt_bytes(danger_start)} "
                                f"(ceiling={_fmt_bytes(ceiling)}) | cuda={_snapshot_text(nums)}"
                            )
                        over_for = max(0.0, now - over_since)
                        rec["soft_guard_over_seconds"] = max(float(rec.get("soft_guard_over_seconds", 0.0) or 0.0), over_for)
                        if over_for >= self._soft_over_limit_seconds:
                            # Do not abort while Wan is inside a live transformer
                            # forward. Windows/NVML can report a short full-card
                            # pressure spike during the kernel, and the step can
                            # still recover cleanly by the time the forward returns.
                            # Mark this as a pending abort candidate and decide at
                            # step boundary using fresh memory numbers.
                            reason = (
                                f"sampling step {step_index + 1} stayed near full profile/card ceiling for {over_for:.2f}s: "
                                f"pressure={_fmt_bytes(pressure)} >= danger={_fmt_bytes(danger_start)} "
                                f"(ceiling={_fmt_bytes(ceiling)})"
                            )
                            rec["soft_guard_pending_abort"] = reason
                            rec["soft_guard_pending_abort_snapshot"] = _snapshot_text(nums)
                            CONTEXT["wan_denoise_soft_guard_pending_abort"] = reason
                            try:
                                self._write_live(
                                    f"soft guard pending abort | loop={loop_index} step={step_index + 1} | "
                                    f"will re-check at step boundary | {reason} | cuda={_snapshot_text(nums)}"
                                )
                            except Exception:
                                pass
                            # Keep monitoring but do not spam this message.
                            over_since = now + 999999.0
                    else:
                        over_since = 0.0

            try:
                rec["monitor_thread"] = threading.Thread(target=monitor, name=f"wan-step-watch-{loop_index}-{step_index+1}", daemon=True)
                rec["monitor_thread"].start()
            except Exception:
                rec["monitor_thread"] = None

    def end_step(self, loop_index: int, step_index: int) -> None:
        with self._lock:
            rec = self._find_step(loop_index, step_index)
            if rec is None:
                return
            try:
                rec["stop_event"].set()
                t = rec.get("monitor_thread")
                if t is not None:
                    t.join(timeout=0.5)
            except Exception:
                pass
            try:
                self._flush_deferred_runtime_cleanup(f"wan_denoise_step_{step_index + 1}_boundary")
            except Exception as exc:
                if len(self.failures) < 20:
                    self.failures.append(f"deferred cleanup flush failed: {type(exc).__name__}: {exc}")
            try:
                if self.torch is not None and self.torch.cuda.is_available():
                    rec["peak_allocated"] = max(int(rec.get("peak_allocated", 0) or 0), int(self.torch.cuda.max_memory_allocated()))
                    rec["peak_reserved"] = max(int(rec.get("peak_reserved", 0) or 0), int(self.torch.cuda.max_memory_reserved()))
            except Exception:
                pass
            rec["duration"] = max(0.0, time.perf_counter() - float(rec.get("started", time.perf_counter())))
            rec["end_cuda"] = _cuda_numbers(self.torch)
            rec["end_runtime"] = self._runtime_snapshot()

            # If the monitor saw full-card pressure for the soft window, decide
            # only now, after the active Wan forward returned and memory numbers
            # had a chance to recover. This prevents killing good jobs that spike
            # to full VRAM during a step but return to a safe state at the step
            # boundary.
            pending_abort = str(rec.get("soft_guard_pending_abort", "") or "")
            if pending_abort:
                danger = int(self._danger_start_bytes or self._profile_ceiling_bytes or 0)
                end = rec.get("end_cuda") or {}
                end_pressure = max(int(end.get("reserved", 0) or 0), int(end.get("driver_used", 0) or 0))
                if danger and end_pressure >= danger:
                    reason = (
                        f"{pending_abort}; still near full profile/card ceiling at step boundary: "
                        f"end_pressure={_fmt_bytes(end_pressure)} >= danger={_fmt_bytes(danger)}"
                    )
                    rec["soft_guard_abort"] = reason
                    self._soft_abort_count += 1
                    CONTEXT["wan_denoise_soft_guard_aborts"] = str(self._soft_abort_count)
                    self._set_abort_reason(reason)
                else:
                    rec["soft_guard_recovered_abort"] = pending_abort
                    CONTEXT["wan_denoise_soft_guard_recovered_aborts"] = str(
                        int(CONTEXT.get("wan_denoise_soft_guard_recovered_aborts", "0") or 0) + 1
                    )
                    CONTEXT["wan_denoise_soft_guard_last_recovered_abort"] = (
                        f"step {step_index + 1}: recovered at boundary; end_pressure={_fmt_bytes(end_pressure)} < danger={_fmt_bytes(danger)}"
                    )
                    try:
                        self._write_live(
                            f"soft guard recovered | loop={loop_index} step={step_index + 1} | "
                            f"end_pressure={_fmt_bytes(end_pressure)} < danger={_fmt_bytes(danger)} | cuda={_snapshot_text(end)}"
                        )
                    except Exception:
                        pass

            self._classify_step(rec)
            self._emit_step(rec)
            self.current_step_index = -1
            self.update_context(CONTEXT)
        self._raise_if_abort_requested()

    def _find_step(self, loop_index: int, step_index: int) -> Dict[str, Any] | None:
        for rec in reversed(self.steps):
            if int(rec.get("loop", -1)) == int(loop_index) and int(rec.get("step", -1)) == int(step_index):
                return rec
        return None

    def _classify_step(self, rec: Dict[str, Any]) -> None:
        limit = int(self._profile_limit_bytes or 0)
        danger = int(self._danger_start_bytes or self._profile_ceiling_bytes or 0)
        peak_reserved = int(rec.get("peak_reserved", 0) or 0)
        end_reserved = int((rec.get("end_cuda") or {}).get("reserved", 0) or 0)
        peak_driver_used = 0
        try:
            start = rec.get("start_cuda") or {}
            end = rec.get("end_cuda") or {}
            total = int(end.get("driver_total", 0) or start.get("driver_total", 0) or 0)
            min_free = int(rec.get("min_driver_free", 0) or 0)
            if total and min_free:
                peak_driver_used = max(0, total - min_free)
        except Exception:
            peak_driver_used = 0
        peak_pressure = max(peak_reserved, peak_driver_used)
        end_pressure = max(end_reserved, int((rec.get("end_cuda") or {}).get("driver_used", 0) or 0))
        if rec.get("soft_guard_abort"):
            rec["decision"] = "FAIL/RISK: denoise soft guard requested abort"
        elif rec.get("soft_guard_recovered_abort"):
            rec["decision"] = "WARNING: full-card pressure persisted during step but recovered at boundary"
        elif danger and peak_pressure >= danger:
            rec["decision"] = "WARNING: touched near-full profile/card ceiling but recovered"
        elif limit and end_pressure > limit:
            rec["decision"] = "WARNING: ended above cleanup threshold"
        elif limit and peak_pressure > limit:
            rec["decision"] = "WARNING: crossed cleanup threshold briefly but recovered"
        else:
            rec["decision"] = "PASS: stayed under profile"

    def _emit_step(self, rec: Dict[str, Any]) -> None:
        step_no = int(rec.get("step", 0)) + 1
        total = self.steps_hint or CONTEXT.get("wan_sample_steps", "?")
        start_rt = rec.get("start_runtime") or {}
        end_rt = rec.get("end_runtime") or {}
        loads = int(end_rt.get("loads", 0) or 0) - int(start_rt.get("loads", 0) or 0)
        unloads = int(end_rt.get("unloads", 0) or 0) - int(start_rt.get("unloads", 0) or 0)
        emergency = int(end_rt.get("emergency", 0) or 0) - int(start_rt.get("emergency", 0) or 0)
        hot_trim = int(end_rt.get("hot_trim", 0) or 0) - int(start_rt.get("hot_trim", 0) or 0)
        forced = int(end_rt.get("forced", 0) or 0) - int(start_rt.get("forced", 0) or 0)
        active = str(end_rt.get("active", "n/a") or "n/a")
        cuda_blocks = int(end_rt.get("cuda_blocks", 0) or 0)
        start_reserved = int((rec.get("start_cuda") or {}).get("reserved", 0) or 0)
        end_reserved = int((rec.get("end_cuda") or {}).get("reserved", 0) or 0)
        growth = end_reserved - start_reserved
        unexpected = "YES" if cuda_blocks > 0 else "NO"
        corrections = int(rec.get("soft_guard_corrections", 0) or 0)
        over_seconds = float(rec.get("soft_guard_over_seconds", 0.0) or 0.0)
        guard_note = f", corrections {corrections}" if corrections else ""
        line = (
            f"step {step_no}/{total} done: {float(rec.get('duration', 0.0)):.1f}s, "
            f"peak reserved {_fmt_bytes(rec.get('peak_reserved'))}, min free {_fmt_bytes(rec.get('min_driver_free'))}, "
            f"loads {loads}, unloads {unloads}{guard_note}"
        )
        try:
            print(f"[WAN22][VRAM Lab] {line}", flush=True)
        except Exception:
            pass
        detail = (
            f"loop={rec.get('loop')} step={step_no}/{total} | duration={float(rec.get('duration', 0.0)):.3f}s | "
            f"start={_snapshot_text(rec.get('start_cuda'))} | end={_snapshot_text(rec.get('end_cuda'))} | "
            f"peak_allocated={_fmt_bytes(rec.get('peak_allocated'))} | peak_reserved={_fmt_bytes(rec.get('peak_reserved'))} | "
            f"min_driver_free={_fmt_bytes(rec.get('min_driver_free'))} | loads={loads} unloads={unloads} "
            f"forced={forced} hot_trims={hot_trim} emergency={emergency} | active={active} | "
            f"cuda_blocks_after={cuda_blocks} unexpected_cuda_residency={unexpected} | reserved_growth={_fmt_delta(growth)} | "
            f"soft_guard_corrections={corrections} soft_guard_over={over_seconds:.2f}s | {rec.get('decision')}"
        )
        rec["summary"] = detail
        self._write_live(detail)
        if self.echo_verbose:
            try:
                print(f"[WAN22][VRAM Lab][step-detail] {detail}", flush=True)
            except Exception:
                pass

    def update_context(self, ctx: Dict[str, Any]) -> None:
        completed = [r for r in self.steps if float(r.get("duration", 0.0) or 0.0) > 0.0]
        ctx["wan_denoise_profile_limit"] = f"{_wan_profile_reserved_limit_gb():.1f} GB"
        ctx["wan_denoise_profiled_loop_count"] = str(self.profiled_loop_count)
        ctx["wan_denoise_profiled_step_count"] = str(len(completed))
        ctx["wan_denoise_step_profiler_failures"] = "none" if not self.failures else " | ".join(self.failures[-10:])
        ctx["wan_denoise_soft_guard_limit"] = _fmt_bytes(self._profile_limit_bytes)
        ctx["wan_denoise_soft_guard_full_ceiling"] = _fmt_bytes(self._profile_ceiling_bytes)
        ctx["wan_denoise_soft_guard_danger_start"] = _fmt_bytes(self._danger_start_bytes)
        ctx["wan_denoise_soft_guard_seconds"] = f"{self._soft_over_limit_seconds:.1f}s"
        ctx["wan_denoise_soft_guard_corrections"] = str(self._correction_count)
        ctx["wan_denoise_soft_guard_aborts"] = str(self._soft_abort_count)
        ctx["wan_denoise_soft_guard_abort_reason"] = self._abort_reason or ctx.get("wan_denoise_soft_guard_abort_reason", "none")
        if not completed:
            ctx["wan_denoise_step_summary"] = "no completed sampling steps recorded yet"
            return
        total = sum(float(r.get("duration", 0.0) or 0.0) for r in completed)
        worst_peak = max(completed, key=lambda r: int(r.get("peak_reserved", 0) or 0))
        worst_free = min(completed, key=lambda r: int(r.get("min_driver_free", 1 << 60) or (1 << 60)))
        high_alloc = max(int(r.get("peak_allocated", 0) or 0) for r in completed)
        high_reserved = max(int(r.get("peak_reserved", 0) or 0) for r in completed)
        grows = []
        for r in completed:
            start_reserved = int((r.get("start_cuda") or {}).get("reserved", 0) or 0)
            end_reserved = int((r.get("end_cuda") or {}).get("reserved", 0) or 0)
            grows.append(end_reserved - start_reserved)
        positive_growth_steps = sum(1 for g in grows if g > 256 * 1024 * 1024)
        decisions = [str(r.get("decision", "")) for r in completed]
        if any(d.startswith("FAIL/RISK") for d in decisions):
            decision = "FAIL/RISK: steps stayed over profile or approached driver-free danger"
        elif any(d.startswith("WARNING") for d in decisions):
            decision = "WARNING: steps crossed cleanup/danger threshold but recovered"
        else:
            decision = "PASS: steps stayed under profile"
        ctx["wan_denoise_total_step_time"] = f"{total:.3f}s"
        ctx["wan_denoise_avg_step_time"] = f"{(total / max(1, len(completed))):.3f}s"
        ctx["wan_denoise_worst_peak_reserved_step"] = f"step {int(worst_peak.get('step',0))+1}, {_fmt_bytes(worst_peak.get('peak_reserved'))}"
        ctx["wan_denoise_worst_min_driver_free_step"] = f"step {int(worst_free.get('step',0))+1}, {_fmt_bytes(worst_free.get('min_driver_free'))}"
        ctx["wan_denoise_highest_allocated"] = _fmt_bytes(high_alloc)
        ctx["wan_denoise_highest_reserved"] = _fmt_bytes(high_reserved)
        ctx["wan_denoise_reserved_growth_steps"] = f"{positive_growth_steps}/{len(completed)} steps grew by >256MB"
        ctx["wan_denoise_profile_decision"] = decision
        ctx["wan_denoise_step_summary"] = (
            f"{len(completed)} steps; total {total:.2f}s; avg {(total / max(1, len(completed))):.2f}s; "
            f"highest reserved {_fmt_bytes(high_reserved)}; lowest driver-free {_fmt_bytes(worst_free.get('min_driver_free'))}; {decision}"
        )
        ctx["wan_denoise_per_step"] = " | ".join(str(r.get("summary", "")) for r in completed[-64:])


def _snapshot_text(nums: Any) -> str:
    if not isinstance(nums, dict):
        return "n/a"
    return (
        f"alloc={_fmt_bytes(nums.get('allocated'))}, reserved={_fmt_bytes(nums.get('reserved'))}, "
        f"driver_free={_fmt_bytes(nums.get('driver_free'))}, driver_used={_fmt_bytes(nums.get('driver_used'))}"
    )


def _install_wan_denoise_step_profiler(torch_mod: Any) -> _WanDenoiseStepProfiler | None:
    try:
        steps_hint = 0
        try:
            steps_hint = int(str(CONTEXT.get("wan_sample_steps", "0") or "0"))
        except Exception:
            steps_hint = 0
        profiler = _WanDenoiseStepProfiler(
            torch_mod=torch_mod,
            steps_hint=steps_hint,
            live_path=STAGE_LOG_PATH,
            echo_verbose=WAN_STAGE_VERBOSE,
        )
        profiler.start()
        return profiler
    except Exception as exc:
        CONTEXT["wan_denoise_step_profiler_enabled"] = f"FAILED: {type(exc).__name__}: {exc}"
        return None

def _install_method_stage_loggers(instance: Any, class_name: str, torch_mod: Any) -> None:
    """Wrap Wan public generation methods with timing/memory logging only."""
    for method_name in ("generate", "t2v", "i2v"):
        try:
            bound = getattr(instance, method_name, None)
            if not callable(bound) or getattr(bound, "_framevision_stage_logger_wrapped", False):
                continue

            def make_wrapper(orig: Any, mname: str):
                def wrapped(*args, **kwargs):
                    label = f"{class_name}.{mname}"
                    _stage_event(label + ":args", _tensor_summary({"args": args, "kwargs": kwargs}), torch_mod)
                    with _stage_watch(label, "Wan public generation method", torch_mod):
                        out = orig(*args, **kwargs)
                    _stage_event(label + ":return", _tensor_summary(out), torch_mod)
                    return out
                try:
                    wrapped._framevision_stage_logger_wrapped = True  # type: ignore[attr-defined]
                except Exception:
                    pass
                return wrapped

            setattr(instance, method_name, make_wrapper(bound, method_name))
            CONTEXT[f"{class_name}_{method_name}_stage_logger_installed"] = "YES"
        except Exception as e:
            CONTEXT.setdefault("stage_logger_notes", []).append(
                f"failed to wrap {class_name}.{method_name}: {type(e).__name__}: {e}"
            )



def _make_finalize_guard(torch_mod: Any, label: str = "wan_finalize"):
    try:
        if str(VRAM_LAB_DIR) not in sys.path:
            sys.path.insert(0, str(VRAM_LAB_DIR))
        import vram_forward_hooks as vfh  # type: ignore
        if hasattr(vfh, "make_finalize_guard"):
            return vfh.make_finalize_guard(ctx=CONTEXT, label=label, torch_module=torch_mod)
    except Exception as e:
        CONTEXT["finalize_guard_enabled"] = f"FAILED: {type(e).__name__}: {e}"
        CONTEXT.setdefault("finalize_guard_notes", []).append(f"finalize guard import/create failed: {e}")
    return None




def _make_decode_controller(torch_mod: Any, label: str = "wan_decode_controller"):
    try:
        if str(VRAM_LAB_DIR) not in sys.path:
            sys.path.insert(0, str(VRAM_LAB_DIR))
        import vram_forward_hooks as vfh  # type: ignore
        if hasattr(vfh, "make_decode_controller"):
            return vfh.make_decode_controller(ctx=CONTEXT, torch_module=torch_mod, label=label)
    except Exception as e:
        CONTEXT["decode_controller_enabled"] = f"FAILED: {type(e).__name__}: {e}"
        CONTEXT.setdefault("decode_controller_notes", []).append(f"decode controller import/create failed: {e}")
    return None


def _make_boundary_tracer(torch_mod: Any, label: str = "wan_boundary"):
    global BOUNDARY_TRACER
    if BOUNDARY_TRACER is not None:
        return BOUNDARY_TRACER
    try:
        if str(VRAM_LAB_DIR) not in sys.path:
            sys.path.insert(0, str(VRAM_LAB_DIR))
        import vram_forward_hooks as vfh  # type: ignore
        if hasattr(vfh, "make_boundary_tracer"):
            BOUNDARY_TRACER = vfh.make_boundary_tracer(ctx=CONTEXT, label=label, torch_module=torch_mod, echo=True)
            return BOUNDARY_TRACER
    except Exception as e:
        CONTEXT["boundary_finder_enabled"] = f"FAILED: {type(e).__name__}: {e}"
        CONTEXT.setdefault("boundary_trace_notes", []).append(f"boundary tracer import/create failed: {e}")
    return None


def _boundary_mark(key: str, note: str = "") -> None:
    _stage_event(key, note, None)
    tracer = BOUNDARY_TRACER
    if tracer is not None:
        try:
            tracer.mark(key, note)
            return
        except Exception:
            pass
    # fallback so report still shows ordering even if tracer import failed
    try:
        CONTEXT.setdefault("boundary_trace_stages", []).append(f"{key}: {note}; tracer unavailable")
    except Exception:
        pass


def _move_obj_to_cpu(obj: Any) -> Any:
    try:
        if hasattr(obj, "detach") and hasattr(obj, "to"):
            try:
                return obj.detach().to("cpu")
            except Exception:
                return obj.to("cpu")
        if isinstance(obj, list):
            return [_move_obj_to_cpu(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_move_obj_to_cpu(x) for x in obj)
        if isinstance(obj, dict):
            return {k: _move_obj_to_cpu(v) for k, v in obj.items()}
    except Exception:
        pass
    return obj


def _install_wan_finalize_save_guard(torch_mod: Any) -> None:
    """Patch Wan's save_video import source from outside the Wan repo.

    generate.py imports save_video from wan.utils.utils. By patching that module
    before runpy executes generate.py, the repo remains untouched while the final
    save stage gets VRAM Lab cleanup/telemetry.
    """
    try:
        import wan.utils.utils as wan_utils  # type: ignore
    except Exception as e:
        CONTEXT["finalize_guard_enabled"] = f"NO: could not import wan.utils.utils ({e})"
        return

    orig_save = getattr(wan_utils, "save_video", None)
    if not callable(orig_save):
        CONTEXT["finalize_guard_enabled"] = "NO: wan.utils.utils.save_video not callable"
        return

    if getattr(orig_save, "_framevision_vram_finalize_wrapped", False):
        CONTEXT["finalize_guard_enabled"] = "YES: save_video already wrapped"
        return

    def wrapped_save_video(*args, **kwargs):
        _stage_event("wan_save_video_enter", _tensor_summary({"args": args, "kwargs": kwargs}), torch_mod)
        guard = _make_finalize_guard(torch_mod, "wan_finalize")
        if guard is not None:
            try:
                guard.stage("cuda_after_denoise_generation", "Wan generation returned tensor; save_video called")
                # Update report counters and then detach/release the hook runtime
                # before the heavy save/finalization path.
                for rt in list(RUNTIMES):
                    try:
                        if hasattr(rt, "update_context"):
                            rt.update_context(CONTEXT)
                    except Exception:
                        pass
                guard.detach_runtimes(list(RUNTIMES), clear_context_key=None)
                try:
                    RUNTIMES.clear()
                except Exception:
                    pass

                # Ensure the video tensor passed to save_video is CPU-backed.
                if "tensor" in kwargs:
                    kwargs["tensor"] = guard.tensor_to_cpu(kwargs.get("tensor"), label="wan_save_tensor_to_cpu")
                    CONTEXT["finalize_output_to_cpu"] = "YES: save_video tensor kwarg"
                elif args:
                    args_list = list(args)
                    args_list[0] = guard.tensor_to_cpu(args_list[0], label="wan_save_tensor_to_cpu")
                    args = tuple(args_list)
                    CONTEXT["finalize_output_to_cpu"] = "YES: save_video first positional arg"
                else:
                    CONTEXT["finalize_output_to_cpu"] = "n/a: no tensor argument detected"

                gc.collect()
                guard.cleanup_cuda("before_save_reencode")
                guard.stage("cuda_before_save_reencode", "before Wan save_video")
            except Exception as e:
                CONTEXT.setdefault("finalize_guard_notes", []).append(f"Wan finalize pre-save guard failed: {type(e).__name__}: {e}")

        started = time.perf_counter()
        try:
            with _stage_watch("wan_save_video", "original Wan save_video", torch_mod):
                return orig_save(*args, **kwargs)
        finally:
            duration = time.perf_counter() - started
            CONTEXT["finalize_save_reencode_duration"] = f"{duration:.3f}s"
            CONTEXT["finalize_save_fast_or_slow"] = "FAST" if duration < 30.0 else "SLOW/WATCH"
            if guard is not None:
                try:
                    guard.stage("cuda_after_save_reencode", "after Wan save_video")
                    guard.finish()
                except Exception as e:
                    CONTEXT.setdefault("finalize_guard_notes", []).append(f"Wan finalize post-save guard failed: {type(e).__name__}: {e}")
            _stage_event("wan_save_video_exit", f"duration={duration:.3f}s", torch_mod)

    try:
        wrapped_save_video._framevision_vram_finalize_wrapped = True  # type: ignore[attr-defined]
    except Exception:
        pass
    wan_utils.save_video = wrapped_save_video
    CONTEXT["finalize_guard_enabled"] = "YES: save_video wrapped"






def _replace_decode_first_arg(args: tuple, kwargs: Dict[str, Any], new_value: Any) -> tuple[tuple, Dict[str, Any]]:
    """Return decode args with the first positional/list latent replaced."""
    new_kwargs = dict(kwargs or {})
    if args:
        lst = list(args)
        lst[0] = new_value
        return tuple(lst), new_kwargs
    # Fallback for rare keyword-based decoders.
    for key in ("z", "zs", "x", "latents", "features"):
        if key in new_kwargs:
            new_kwargs[key] = new_value
            return tuple(args), new_kwargs
    return tuple(args), new_kwargs


def _get_decode_latent_list(args: tuple, kwargs: Dict[str, Any]) -> tuple[Any, list[Any]] | tuple[None, list[Any]]:
    """Find Wan VAE decode's latent list without depending on Wan source changes."""
    src = args[0] if args else None
    if src is None:
        for key in ("z", "zs", "x", "latents", "features"):
            if key in kwargs:
                src = kwargs.get(key)
                break
    if isinstance(src, (list, tuple)):
        vals = list(src)
        if vals and all(hasattr(v, "shape") and hasattr(v, "device") for v in vals):
            return src, vals
    if hasattr(src, "shape") and hasattr(src, "device"):
        return [src], [src]
    return None, []


def _decode_output_first_tensor(out: Any) -> Any:
    try:
        if isinstance(out, (list, tuple)) and out:
            return out[0]
    except Exception:
        pass
    return out


def _cudnn_state_note(torch_mod: Any) -> str:
    try:
        backends = getattr(torch_mod, "backends", None)
        cudnn = getattr(backends, "cudnn", None) if backends is not None else None
        cuda = getattr(backends, "cuda", None) if backends is not None else None
        matmul = getattr(cuda, "matmul", None) if cuda is not None else None
        return (
            f"cudnn.enabled={getattr(cudnn, 'enabled', 'n/a')}; "
            f"cudnn.benchmark={getattr(cudnn, 'benchmark', 'n/a')}; "
            f"cudnn.deterministic={getattr(cudnn, 'deterministic', 'n/a')}; "
            f"cudnn.allow_tf32={getattr(cudnn, 'allow_tf32', 'n/a')}; "
            f"matmul.allow_tf32={getattr(matmul, 'allow_tf32', 'n/a')}"
        )
    except Exception as e:
        return f"cudnn state unavailable: {type(e).__name__}: {e}"


class _LowWorkspaceVaeDecodeContext:
    """Temporarily force a lower-workspace backend only around Wan VAE decode.

    The stage log proved the post-steps spike happens inside the VAE decoder
    convolution stack even when decoding one latent frame at a time.  That points
    at CUDA/cuDNN choosing a very large convolution workspace.  This context keeps
    the rest of Wan untouched, but for VAE decode it disables cuDNN benchmarking
    and, by default, disables cuDNN entirely so PyTorch avoids those huge
    workspaces.  It restores the user's backend flags immediately afterwards.
    """
    def __init__(self, torch_mod: Any, disable_cudnn: bool = True):
        self.torch_mod = torch_mod
        self.disable_cudnn = bool(disable_cudnn)
        self.saved: Dict[str, Any] = {}

    def __enter__(self):
        tm = self.torch_mod
        try:
            _stage_event("wan_low_workspace_decode_before", _cudnn_state_note(tm), tm)
            backends = getattr(tm, "backends", None)
            cudnn = getattr(backends, "cudnn", None) if backends is not None else None
            cuda = getattr(backends, "cuda", None) if backends is not None else None
            matmul = getattr(cuda, "matmul", None) if cuda is not None else None
            if cudnn is not None:
                for name in ("enabled", "benchmark", "deterministic", "allow_tf32"):
                    if hasattr(cudnn, name):
                        try:
                            self.saved[f"cudnn.{name}"] = getattr(cudnn, name)
                        except Exception:
                            pass
                try:
                    cudnn.benchmark = False
                except Exception:
                    pass
                try:
                    cudnn.deterministic = True
                except Exception:
                    pass
                try:
                    cudnn.allow_tf32 = False
                except Exception:
                    pass
                if self.disable_cudnn:
                    try:
                        cudnn.enabled = False
                    except Exception:
                        pass
            if matmul is not None and hasattr(matmul, "allow_tf32"):
                try:
                    self.saved["matmul.allow_tf32"] = matmul.allow_tf32
                    matmul.allow_tf32 = False
                except Exception:
                    pass
            CONTEXT["wan_low_workspace_vae_decode"] = "YES: cudnn benchmark off, deterministic on, cudnn disabled during VAE decode"
            _stage_event("wan_low_workspace_decode_active", _cudnn_state_note(tm), tm)
        except Exception as e:
            CONTEXT["wan_low_workspace_vae_decode"] = f"FAILED: {type(e).__name__}: {e}"
        return self

    def __exit__(self, exc_type, exc, tb):
        tm = self.torch_mod
        try:
            backends = getattr(tm, "backends", None)
            cudnn = getattr(backends, "cudnn", None) if backends is not None else None
            cuda = getattr(backends, "cuda", None) if backends is not None else None
            matmul = getattr(cuda, "matmul", None) if cuda is not None else None
            if cudnn is not None:
                mapping = {
                    "cudnn.enabled": "enabled",
                    "cudnn.benchmark": "benchmark",
                    "cudnn.deterministic": "deterministic",
                    "cudnn.allow_tf32": "allow_tf32",
                }
                for key, name in mapping.items():
                    if key in self.saved and hasattr(cudnn, name):
                        try:
                            setattr(cudnn, name, self.saved[key])
                        except Exception:
                            pass
            if matmul is not None and "matmul.allow_tf32" in self.saved and hasattr(matmul, "allow_tf32"):
                try:
                    matmul.allow_tf32 = self.saved["matmul.allow_tf32"]
                except Exception:
                    pass
            _stage_event("wan_low_workspace_decode_restored", _cudnn_state_note(tm), tm)
        except Exception:
            pass
        return False



def _wan_vram_tiled_conv3d_forward(module: Any, args: tuple, kwargs: Dict[str, Any], torch_mod: Any, module_id: str) -> tuple[bool, Any]:
    """Low-memory spatial tiled replacement for Wan CausalConv3d.forward.

    Wan's VAE decoder can ask CUDA for a massive workspace when a late high-res
    CausalConv3d sees tensors such as (1, 512, 1, 352, 624).  This helper
    mirrors CausalConv3d.forward, but computes the spatial result in smaller
    H/W tiles after applying the same causal padding/cache logic.  It is only
    used from the external wrapper; the Wan repo is not edited.
    """
    try:
        if not args:
            return False, None
        x = args[0]
        if not hasattr(x, "shape") or len(tuple(x.shape)) != 5:
            return False, None
        shape = tuple(int(v) for v in x.shape)
        b, c, t, h, w = shape
        if h < 192 and w < 192:
            return False, None
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "shape") or len(tuple(weight.shape)) != 5:
            return False, None
        padding_src = getattr(module, "_padding", None)
        if padding_src is None:
            return False, None
        # Keep small layers on the original path. The problem is the late,
        # high-resolution VAE CausalConv3d workspace.
        if (h * w) < (176 * 312):
            return False, None

        cache_x = None
        if len(args) >= 2:
            cache_x = args[1]
        if "cache_x" in kwargs:
            cache_x = kwargs.get("cache_x")

        padding = list(padding_src)
        if cache_x is not None and padding[4] > 0:
            try:
                cache_x = cache_x.to(x.device)
                x = torch_mod.cat([cache_x, x], dim=2)
                padding[4] -= int(cache_x.shape[2])
            except Exception as e:
                raise RuntimeError(f"tiled CausalConv3d cache handling failed for {module_id}: {e}") from e

        try:
            F = torch_mod.nn.functional
        except Exception:
            import torch.nn.functional as F  # type: ignore

        x_pad = F.pad(x, padding)
        stride = getattr(module, "stride", (1, 1, 1))
        dilation = getattr(module, "dilation", (1, 1, 1))
        groups = int(getattr(module, "groups", 1))
        bias = getattr(module, "bias", None)
        k_t, k_h, k_w = (int(v) for v in tuple(weight.shape[2:]))
        s_t, s_h, s_w = (int(v) for v in tuple(stride))
        d_t, d_h, d_w = (int(v) for v in tuple(dilation))
        in_t, in_h, in_w = int(x_pad.shape[2]), int(x_pad.shape[3]), int(x_pad.shape[4])
        out_h = (in_h - d_h * (k_h - 1) - 1) // s_h + 1
        out_w = (in_w - d_w * (k_w - 1) - 1) // s_w + 1
        out_t = (in_t - d_t * (k_t - 1) - 1) // s_t + 1
        if out_h <= 0 or out_w <= 0 or out_t <= 0:
            return False, None

        # Smaller tiles + preallocated output.  The previous tiled version still
        # used row/list CUDA cats, which lowered the monster 31GB+ spike but still
        # crossed the guard at upsamples.2.upsamples.0.residual.2.  Preallocating
        # the final output and copying each tile into place avoids holding all
        # tile outputs plus concat work buffers on CUDA at the same time.
        # Final Wan VAE stages (352x624 and above) need smaller tiles.
        # The previous 48x64 final-stage tiles avoided the 30GB+ explosion, but
        # still crossed the probe guard around 20GB reserved.  Smaller tiles keep
        # tile workspaces lower while the decoder's own live residual tensors stay
        # resident.
        tile_h, tile_w = _wan_profile_tile_size(out_h, out_w)
        out_channels = int(weight.shape[0])
        CONTEXT["wan_tiled_causalconv3d_active"] = "YES"
        CONTEXT["wan_tiled_causalconv3d_last_module"] = module_id
        CONTEXT["wan_tiled_causalconv3d_last_shape"] = str(shape)
        CONTEXT["wan_tiled_causalconv3d_tile"] = f"{tile_h}x{tile_w}"
        CONTEXT["wan_tiled_causalconv3d_strategy"] = "preallocate-copy; no row/column cuda cat"
        _tile_summary_once("tiled_causalconv3d", module_id, shape, tile_h, tile_w, torch_mod)
        _stage_event(
            "wan_tiled_causalconv3d_start",
            f"module={module_id}; input_shape={shape}; padded_shape={tuple(x_pad.shape)}; output_shape={(b, out_channels, out_t, out_h, out_w)}; tile={tile_h}x{tile_w}; strategy=preallocate_copy",
            torch_mod,
        )

        out = x.new_empty((b, out_channels, out_t, out_h, out_w))
        tile_count = 0
        for oh0 in range(0, out_h, tile_h):
            oh1 = min(out_h, oh0 + tile_h)
            ih0 = oh0 * s_h
            ih1 = (oh1 - 1) * s_h + d_h * (k_h - 1) + 1
            for ow0 in range(0, out_w, tile_w):
                ow1 = min(out_w, ow0 + tile_w)
                iw0 = ow0 * s_w
                iw1 = (ow1 - 1) * s_w + d_w * (k_w - 1) + 1
                tile_count += 1
                x_tile = x_pad[:, :, :, ih0:ih1, iw0:iw1]
                y_tile = F.conv3d(x_tile, weight, bias, stride, (0, 0, 0), dilation, groups)
                # Shape sanity: the tile result should exactly match the target
                # output window.  If not, fail loudly rather than producing a
                # subtly corrupted frame.
                expected_h = oh1 - oh0
                expected_w = ow1 - ow0
                try:
                    got = tuple(int(v) for v in y_tile.shape)
                except Exception:
                    got = ()
                if len(got) != 5 or got[2] != out_t or got[3] != expected_h or got[4] != expected_w:
                    raise RuntimeError(
                        f"tiled CausalConv3d produced unexpected tile shape for {module_id}: "
                        f"got={got}, expected=(*,*,{out_t},{expected_h},{expected_w})"
                    )
                out[:, :, :, oh0:oh1, ow0:ow1].copy_(y_tile)
                try:
                    del x_tile
                    del y_tile
                except Exception:
                    pass
                # Do not call empty_cache on every tile by default; that can make
                # decode extremely slow.  We only clean when the allocator is
                # getting close to the guard.
                try:
                    nums = _cuda_numbers(torch_mod)
                    reserved_now = int(nums.get("reserved", 0) or 0)
                    driver_free_now = int(nums.get("driver_free", 0) or 0)
                    if reserved_now > int(16.0 * 1024 ** 3) or (driver_free_now and driver_free_now < int(2.5 * 1024 ** 3)):
                        torch_mod.cuda.empty_cache()
                        torch_mod.cuda.ipc_collect()
                        _stage_event(
                            "wan_tiled_causalconv3d_tile_cleanup",
                            f"module={module_id}; tile={tile_count}; reserved_before={_fmt_bytes(reserved_now)}; driver_free_before={_fmt_bytes(driver_free_now)}",
                            torch_mod,
                        )
                except Exception:
                    pass
        CONTEXT["wan_tiled_causalconv3d_tiles"] = str(tile_count)
        _stage_event(
            "wan_tiled_causalconv3d_done",
            f"module={module_id}; tiles={tile_count}; output={_tensor_summary(out)}",
            torch_mod,
        )
        return True, out
    except Exception as e:
        CONTEXT["wan_tiled_causalconv3d_error"] = f"{type(e).__name__}: {e}"
        _stage_event("wan_tiled_causalconv3d_error", f"module={module_id}; error={type(e).__name__}: {e}", torch_mod)

        raise


def _as_pair(v: Any) -> tuple[int, int]:
    try:
        if isinstance(v, (list, tuple)):
            if len(v) >= 2:
                return int(v[-2]), int(v[-1])
            if len(v) == 1:
                return int(v[0]), int(v[0])
        return int(v), int(v)
    except Exception:
        return 1, 1


def _wan_vram_tiled_conv2d_forward(module: Any, args: tuple, kwargs: Dict[str, Any], torch_mod: Any, module_id: str) -> tuple[bool, Any]:
    """Low-memory spatial tiled replacement for large Wan Conv2d calls.

    After CausalConv3d tiling, the next guard trip moved to Wan's high-resolution
    resample Conv2d.  This helper mirrors normal Conv2d for large 4D tensors,
    preallocates the final output once, and writes each H/W tile directly into
    that output so CUDA does not allocate one huge convolution workspace.
    """
    try:
        if not args:
            return False, None
        x = args[0]
        if not hasattr(x, "shape") or len(tuple(x.shape)) != 4:
            return False, None
        shape = tuple(int(v) for v in x.shape)
        b, c, h, w = shape
        if (h * w) < (176 * 312):
            return False, None
        weight = getattr(module, "weight", None)
        if weight is None or not hasattr(weight, "shape") or len(tuple(weight.shape)) != 4:
            return False, None
        try:
            F = torch_mod.nn.functional
        except Exception:
            import torch.nn.functional as F  # type: ignore

        stride = _as_pair(getattr(module, "stride", (1, 1)))
        dilation = _as_pair(getattr(module, "dilation", (1, 1)))
        padding = getattr(module, "padding", (0, 0))
        if isinstance(padding, str):
            return False, None
        pad_h, pad_w = _as_pair(padding)
        groups = int(getattr(module, "groups", 1))
        bias = getattr(module, "bias", None)
        k_h, k_w = (int(v) for v in tuple(weight.shape[2:]))
        s_h, s_w = stride
        d_h, d_w = dilation

        if pad_h or pad_w:
            x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
        else:
            x_pad = x
        in_h, in_w = int(x_pad.shape[2]), int(x_pad.shape[3])
        out_h = (in_h - d_h * (k_h - 1) - 1) // s_h + 1
        out_w = (in_w - d_w * (k_w - 1) - 1) // s_w + 1
        if out_h <= 0 or out_w <= 0:
            return False, None

        # Use smaller tiles for the final/high-res stages.  Conv2d is less nasty
        # than CausalConv3d, but the resample Conv2d still crossed the 18GB guard
        # at 704p output, so keep tiles conservative.
        tile_h, tile_w = _wan_profile_tile_size(out_h, out_w)
        out_channels = int(weight.shape[0])
        CONTEXT["wan_tiled_conv2d_active"] = "YES"
        CONTEXT["wan_tiled_conv2d_last_module"] = module_id
        CONTEXT["wan_tiled_conv2d_last_shape"] = str(shape)
        CONTEXT["wan_tiled_conv2d_tile"] = f"{tile_h}x{tile_w}"
        CONTEXT["wan_tiled_conv2d_strategy"] = "preallocate-copy; no row/column cuda cat"
        _tile_summary_once("tiled_conv2d", module_id, shape, tile_h, tile_w, torch_mod)
        _stage_event(
            "wan_tiled_conv2d_start",
            f"module={module_id}; input_shape={shape}; padded_shape={tuple(x_pad.shape)}; output_shape={(b, out_channels, out_h, out_w)}; tile={tile_h}x{tile_w}; strategy=preallocate_copy",
            torch_mod,
        )

        out = x.new_empty((b, out_channels, out_h, out_w))
        tile_count = 0
        for oh0 in range(0, out_h, tile_h):
            oh1 = min(out_h, oh0 + tile_h)
            ih0 = oh0 * s_h
            ih1 = (oh1 - 1) * s_h + d_h * (k_h - 1) + 1
            for ow0 in range(0, out_w, tile_w):
                ow1 = min(out_w, ow0 + tile_w)
                iw0 = ow0 * s_w
                iw1 = (ow1 - 1) * s_w + d_w * (k_w - 1) + 1
                tile_count += 1
                x_tile = x_pad[:, :, ih0:ih1, iw0:iw1]
                y_tile = F.conv2d(x_tile, weight, bias, stride, (0, 0), dilation, groups)
                expected_h = oh1 - oh0
                expected_w = ow1 - ow0
                try:
                    got = tuple(int(v) for v in y_tile.shape)
                except Exception:
                    got = ()
                if len(got) != 4 or got[2] != expected_h or got[3] != expected_w:
                    raise RuntimeError(
                        f"tiled Conv2d produced unexpected tile shape for {module_id}: "
                        f"got={got}, expected=(*,*,{expected_h},{expected_w})"
                    )
                out[:, :, oh0:oh1, ow0:ow1].copy_(y_tile)
                try:
                    del x_tile
                    del y_tile
                except Exception:
                    pass
                try:
                    nums = _cuda_numbers(torch_mod)
                    reserved_now = int(nums.get("reserved", 0) or 0)
                    driver_free_now = int(nums.get("driver_free", 0) or 0)
                    if reserved_now > int(16.0 * 1024 ** 3) or (driver_free_now and driver_free_now < int(2.5 * 1024 ** 3)):
                        torch_mod.cuda.empty_cache()
                        torch_mod.cuda.ipc_collect()
                        _stage_event(
                            "wan_tiled_conv2d_tile_cleanup",
                            f"module={module_id}; tile={tile_count}; reserved_before={_fmt_bytes(reserved_now)}; driver_free_before={_fmt_bytes(driver_free_now)}",
                            torch_mod,
                        )
                except Exception:
                    pass
        CONTEXT["wan_tiled_conv2d_tiles"] = str(tile_count)
        _stage_event(
            "wan_tiled_conv2d_done",
            f"module={module_id}; tiles={tile_count}; output={_tensor_summary(out)}",
            torch_mod,
        )
        return True, out
    except Exception as e:
        CONTEXT["wan_tiled_conv2d_error"] = f"{type(e).__name__}: {e}"
        _stage_event("wan_tiled_conv2d_error", f"module={module_id}; error={type(e).__name__}: {e}", torch_mod)
        raise


class _WanDecoderLayerProbe:
    """Temporary per-module decoder probe for Wan VAE.

    This does not try to fix memory by itself. It proves exactly which internal
    decoder layer/module is responsible for the post-steps VRAM/shared-memory
    explosion, then aborts as soon as a configured ceiling is crossed.
    """
    def __init__(
        self,
        decoder: Any,
        torch_mod: Any,
        *,
        reserved_limit_gb: float = 20.0,
        driver_free_floor_gb: float = 1.0,
    ):
        self.decoder = decoder
        self.torch_mod = torch_mod
        self.reserved_limit = int(float(reserved_limit_gb) * 1024 ** 3)
        self.driver_free_floor = int(float(driver_free_floor_gb) * 1024 ** 3)
        self.originals: list[tuple[Any, str, Any]] = []
        self.installed = False
        self.call_index = 0
        # Reserved-memory guard is allowed to spike briefly. Some decoder
        # operations reserve above the selected profile limit for a moment and
        # then return memory after cleanup. Hard-abort only if the profile limit
        # stays crossed for more than this grace period. Driver-free floor is
        # still treated as an immediate safety abort.
        self.reserved_grace_sec = 5.0
        self._reserved_over_since: float | None = None
        self._reserved_over_peak = 0
        try:
            CONTEXT["wan_decoder_probe_reserved_grace_sec"] = f"{self.reserved_grace_sec:.1f}"
        except Exception:
            pass

    def _should_wrap(self, module: Any) -> bool:
        try:
            # Wrapping leaf modules gives the exact offending Conv/Norm/Resample
            # without wrapping the whole decoder container repeatedly.
            if list(module.children()):
                return False
        except Exception:
            pass
        return callable(getattr(module, "forward", None))

    def install(self) -> None:
        if self.installed:
            return
        wrapped = 0
        try:
            named = list(self.decoder.named_modules())
        except Exception:
            named = []
        for name, module in named:
            if not name:
                continue
            if not self._should_wrap(module):
                continue
            try:
                orig_forward = getattr(module, "forward", None)
                if not callable(orig_forward) or getattr(orig_forward, "_framevision_wan_decoder_probe_wrapped", False):
                    continue
                label = str(name)
                cls_name = module.__class__.__name__

                def make_wrapper(_module, _orig_forward, _label, _cls_name):
                    def _wrapped_forward(*args, **kwargs):
                        self.call_index += 1
                        call_id = self.call_index
                        module_id = f"{_label} ({_cls_name})"
                        CONTEXT["wan_decoder_probe_last_module"] = module_id
                        CONTEXT["wan_decoder_probe_last_call"] = str(call_id)
                        _stage_event(
                            "wan_decoder_layer_enter",
                            f"call={call_id}; module={module_id}; input={_tensor_summary({'args': args, 'kwargs': kwargs}, limit=4)}",
                            self.torch_mod,
                        )
                        started = time.perf_counter()
                        try:
                            used_tiled = False
                            out = None
                            if _cls_name == "CausalConv3d":
                                used_tiled, out = _wan_vram_tiled_conv3d_forward(_module, tuple(args or ()), dict(kwargs or {}), self.torch_mod, module_id)
                                if used_tiled:
                                    CONTEXT["wan_decoder_probe_tiled_module"] = module_id
                            elif _cls_name == "Conv2d":
                                used_tiled, out = _wan_vram_tiled_conv2d_forward(_module, tuple(args or ()), dict(kwargs or {}), self.torch_mod, module_id)
                                if used_tiled:
                                    CONTEXT["wan_decoder_probe_tiled_conv2d_module"] = module_id
                            if not used_tiled:
                                out = _orig_forward(*args, **kwargs)

                            # VRAM Lab Wan VAE decode runs in FP16, but Wan's RMS_norm
                            # can upcast activations back to FP32. The reports show the
                            # final high-res CausalConv3d is still receiving FP32 tensors,
                            # so force decoder normalization outputs back to FP16 before
                            # the following SiLU/Conv layers keep expanding high-res
                            # activations in FP32.
                            try:
                                if _cls_name == "RMS_norm" and self.torch_mod is not None:
                                    if hasattr(out, "is_cuda") and bool(out.is_cuda) and hasattr(out, "dtype"):
                                        if out.dtype == self.torch_mod.float32:
                                            out = out.to(dtype=self.torch_mod.float16)
                                            CONTEXT["wan_decoder_norm_fp16_cast_active"] = "YES"
                                            CONTEXT["wan_decoder_norm_fp16_cast_last_module"] = module_id
                            except Exception as _cast_e:
                                CONTEXT["wan_decoder_norm_fp16_cast_error"] = f"{module_id}: {type(_cast_e).__name__}: {_cast_e}"
                        except BaseException as e:
                            CONTEXT["wan_decoder_probe_failed_module"] = module_id
                            CONTEXT["wan_decoder_probe_failed_call"] = str(call_id)
                            CONTEXT["wan_decoder_probe_failed_error"] = f"{type(e).__name__}: {e}"
                            _stage_event(
                                "wan_decoder_layer_error",
                                f"call={call_id}; module={module_id}; error={type(e).__name__}: {e}",
                                self.torch_mod,
                            )
                            raise

                        nums = _cuda_numbers(self.torch_mod)
                        reserved = int(nums.get("reserved", 0) or 0)
                        driver_free = int(nums.get("driver_free", 0) or 0)
                        duration = time.perf_counter() - started
                        note = (
                            f"call={call_id}; module={module_id}; duration={duration:.3f}s; "
                            f"output={_tensor_summary(out, limit=4)}"
                        )
                        _stage_event("wan_decoder_layer_exit", note, self.torch_mod)

                        # Driver-free floor is the true danger condition: abort immediately.
                        if driver_free and driver_free <= self.driver_free_floor:
                            CONTEXT["wan_decoder_probe_abort_module"] = module_id
                            CONTEXT["wan_decoder_probe_abort_call"] = str(call_id)
                            CONTEXT["wan_decoder_probe_abort_memory"] = (
                                f"reserved={_fmt_bytes(reserved)}, driver_free={_fmt_bytes(driver_free)}, "
                                f"limit={_fmt_bytes(self.reserved_limit)}, floor={_fmt_bytes(self.driver_free_floor)}"
                            )
                            _stage_event(
                                "wan_decoder_layer_guard_abort",
                                (
                                    f"call={call_id}; module={module_id}; "
                                    f"driver_free={_fmt_bytes(driver_free)} <= {_fmt_bytes(self.driver_free_floor)}"
                                ),
                                self.torch_mod,
                            )
                            raise RuntimeError(
                                "VRAM Lab Wan decoder probe abort: "
                                f"{module_id} crossed driver-free guard "
                                f"(reserved={_fmt_bytes(reserved)}, driver_free={_fmt_bytes(driver_free)})."
                            )

                        # Reserved limit is a soft profile ceiling. It may spike briefly,
                        # especially during tiled VAE decoder layers. Try cleanup first and
                        # only abort if it stays above the profile limit for >5 seconds.
                        if reserved >= self.reserved_limit:
                            now_over = time.perf_counter()
                            try:
                                self._reserved_over_peak = max(int(self._reserved_over_peak or 0), int(reserved))
                            except Exception:
                                self._reserved_over_peak = int(reserved)
                            try:
                                if self.torch_mod is not None and hasattr(self.torch_mod, "cuda"):
                                    self.torch_mod.cuda.empty_cache()
                                    try:
                                        self.torch_mod.cuda.ipc_collect()
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            nums2 = _cuda_numbers(self.torch_mod)
                            reserved2 = int(nums2.get("reserved", 0) or reserved)
                            driver_free2 = int(nums2.get("driver_free", 0) or driver_free)

                            if driver_free2 and driver_free2 <= self.driver_free_floor:
                                CONTEXT["wan_decoder_probe_abort_module"] = module_id
                                CONTEXT["wan_decoder_probe_abort_call"] = str(call_id)
                                CONTEXT["wan_decoder_probe_abort_memory"] = (
                                    f"reserved={_fmt_bytes(reserved2)}, driver_free={_fmt_bytes(driver_free2)}, "
                                    f"limit={_fmt_bytes(self.reserved_limit)}, floor={_fmt_bytes(self.driver_free_floor)}"
                                )
                                _stage_event(
                                    "wan_decoder_layer_guard_abort",
                                    (
                                        f"call={call_id}; module={module_id}; "
                                        f"driver_free_after_cleanup={_fmt_bytes(driver_free2)} <= {_fmt_bytes(self.driver_free_floor)}"
                                    ),
                                    self.torch_mod,
                                )
                                raise RuntimeError(
                                    "VRAM Lab Wan decoder probe abort: "
                                    f"{module_id} crossed driver-free guard after cleanup "
                                    f"(reserved={_fmt_bytes(reserved2)}, driver_free={_fmt_bytes(driver_free2)})."
                                )

                            if reserved2 < self.reserved_limit:
                                self._reserved_over_since = None
                                CONTEXT["wan_decoder_probe_reserved_over_active"] = "NO"
                                _stage_event(
                                    "wan_decoder_layer_reserved_spike_recovered",
                                    (
                                        f"call={call_id}; module={module_id}; "
                                        f"reserved_before={_fmt_bytes(reserved)}; reserved_after_cleanup={_fmt_bytes(reserved2)}; "
                                        f"driver_free_after_cleanup={_fmt_bytes(driver_free2)}"
                                    ),
                                    self.torch_mod,
                                )
                            else:
                                if self._reserved_over_since is None:
                                    self._reserved_over_since = now_over
                                over_for = max(0.0, now_over - float(self._reserved_over_since or now_over))
                                CONTEXT["wan_decoder_probe_reserved_over_active"] = "YES"
                                CONTEXT["wan_decoder_probe_reserved_over_for_sec"] = f"{over_for:.2f}"
                                CONTEXT["wan_decoder_probe_reserved_over_peak"] = _fmt_bytes(self._reserved_over_peak)
                                _stage_event(
                                    "wan_decoder_layer_reserved_over_grace",
                                    (
                                        f"call={call_id}; module={module_id}; "
                                        f"reserved={_fmt_bytes(reserved2)} >= {_fmt_bytes(self.reserved_limit)}; "
                                        f"over_for={over_for:.2f}s/{self.reserved_grace_sec:.1f}s; "
                                        f"driver_free={_fmt_bytes(driver_free2)}"
                                    ),
                                    self.torch_mod,
                                )
                                if over_for >= self.reserved_grace_sec:
                                    CONTEXT["wan_decoder_probe_abort_module"] = module_id
                                    CONTEXT["wan_decoder_probe_abort_call"] = str(call_id)
                                    CONTEXT["wan_decoder_probe_abort_memory"] = (
                                        f"reserved={_fmt_bytes(reserved2)}, driver_free={_fmt_bytes(driver_free2)}, "
                                        f"limit={_fmt_bytes(self.reserved_limit)}, floor={_fmt_bytes(self.driver_free_floor)}, "
                                        f"over_for={over_for:.2f}s"
                                    )
                                    _stage_event(
                                        "wan_decoder_layer_guard_abort",
                                        (
                                            f"call={call_id}; module={module_id}; "
                                            f"reserved={_fmt_bytes(reserved2)} >= {_fmt_bytes(self.reserved_limit)} "
                                            f"for {over_for:.2f}s"
                                        ),
                                        self.torch_mod,
                                    )
                                    raise RuntimeError(
                                        "VRAM Lab Wan decoder probe abort: "
                                        f"{module_id} stayed over reserved profile guard for {over_for:.2f}s "
                                        f"(reserved={_fmt_bytes(reserved2)}, driver_free={_fmt_bytes(driver_free2)})."
                                    )
                        else:
                            self._reserved_over_since = None
                            CONTEXT["wan_decoder_probe_reserved_over_active"] = "NO"
                        return out
                    try:
                        _wrapped_forward._framevision_wan_decoder_probe_wrapped = True  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    return _wrapped_forward

                setattr(module, "forward", make_wrapper(module, orig_forward, label, cls_name))
                self.originals.append((module, "forward", orig_forward))
                wrapped += 1
            except Exception as e:
                CONTEXT.setdefault("wan_decoder_probe_install_notes", []).append(
                    f"could not wrap decoder module {name}: {type(e).__name__}: {e}"
                )
        self.installed = True
        CONTEXT["wan_decoder_probe_enabled"] = f"YES: wrapped {wrapped} decoder leaf module(s)"
        CONTEXT["wan_decoder_probe_reserved_limit"] = _fmt_bytes(self.reserved_limit)
        CONTEXT["wan_decoder_probe_driver_free_floor"] = _fmt_bytes(self.driver_free_floor)
        _stage_event(
            "wan_decoder_probe_installed",
            f"wrapped={wrapped}; reserved_limit={_fmt_bytes(self.reserved_limit)}; driver_free_floor={_fmt_bytes(self.driver_free_floor)}",
            self.torch_mod,
        )

    def restore(self) -> None:
        restored = 0
        for module, attr, orig in reversed(self.originals):
            try:
                setattr(module, attr, orig)
                restored += 1
            except Exception:
                pass
        self.originals.clear()
        self.installed = False
        CONTEXT["wan_decoder_probe_restored"] = f"YES: restored {restored} decoder module(s)"
        _stage_event("wan_decoder_probe_restored", f"restored={restored}", self.torch_mod)



def _maybe_chunked_wan_vae_decode(vae: Any, orig_decode: Any, args: tuple, kwargs: Dict[str, Any], torch_mod: Any) -> tuple[bool, Any]:
    """Decode Wan VAE by driving the internal decoder one latent frame at a time.

    The previous helper-side temporal chunking still called the public
    ``vae.decode()`` method. The log proved that even a tiny public decode chunk
    could reserve ~31GB because Wan's internal ``WanVAE_.decode`` builds/cats the
    decoded CUDA video inside the model. This path goes one level deeper:

    * use the live Wan VAE object's ``model.conv2`` + ``model.decoder`` directly;
    * keep Wan's own causal feature cache alive between latent frames;
    * unpatchify each decoder result immediately;
    * move that small decoded piece to CPU immediately;
    * concatenate only on CPU.

    This keeps models/wan22 untouched while avoiding the giant CUDA-side decoded
    video/cat path. If the expected Wan VAE internals are not present, return
    False so the caller can fall back to the old controller/original decode.
    """
    src, latents = _get_decode_latent_list(tuple(args or ()), dict(kwargs or {}))
    if not latents:
        CONTEXT["wan_chunked_decode_used"] = "NO: no latent tensor list found"
        return False, None

    model = getattr(vae, "model", None)
    if model is None:
        CONTEXT["wan_chunked_decode_used"] = "NO: vae.model missing"
        return False, None
    for attr in ("conv2", "decoder", "clear_cache"):
        if not hasattr(model, attr):
            CONTEXT["wan_chunked_decode_used"] = f"NO: vae.model.{attr} missing"
            return False, None

    try:
        mod = sys.modules.get(model.__class__.__module__)
        unpatchify_fn = getattr(mod, "unpatchify", None) if mod is not None else None
    except Exception:
        unpatchify_fn = None
    if not callable(unpatchify_fn):
        CONTEXT["wan_chunked_decode_used"] = "NO: could not locate Wan unpatchify function"
        return False, None

    try:
        first_shape = tuple(getattr(latents[0], "shape", ()))
        if len(first_shape) != 4:
            CONTEXT["wan_chunked_decode_used"] = f"NO: unsupported latent shape {first_shape}"
            return False, None
        t_len = int(first_shape[1])
    except Exception:
        CONTEXT["wan_chunked_decode_used"] = "NO: could not read latent time dimension"
        return False, None

    CONTEXT["wan_chunked_decode_used"] = "YES: internal streaming decoder"
    CONTEXT["wan_chunked_decode_chunk_latents"] = "1 internal latent frame at a time"
    CONTEXT["wan_chunked_decode_overlap_latents"] = "Wan causal cache, no outer overlap"
    CONTEXT["decode_controller_backend"] = "helper-side internal Wan VAE streaming decode"
    CONTEXT["decode_mode_used"] = "internal conv2/decoder loop; CPU concat only"
    CONTEXT["wan_internal_stream_latent_frames"] = str(t_len)

    decoded_full: list[Any] = []
    frame_notes: list[str] = []
    total_frames = 0

    # Use a forced low-memory decode dtype for Wan VAE.
    # Earlier reports proved the remaining spike is no longer full-video concat;
    # it is live high-resolution VAE decoder activations around upsamples.3.
    # Keeping those activations in float32 still pushed driver_free to 0 even
    # with spatial tiling. For VRAM Lab runs we force the helper-side internal
    # VAE decode to FP16 on CUDA, then convert the final decoded pieces back to
    # float32 on CPU before returning to Wan save_video.
    dtype = getattr(vae, "dtype", None)
    try:
        if torch_mod is not None and torch_mod.cuda.is_available():
            dtype = torch_mod.float16
            CONTEXT["wan_internal_vae_decode_dtype"] = "forced float16 for VRAM Lab VAE decode"
        else:
            CONTEXT["wan_internal_vae_decode_dtype"] = str(dtype)
    except Exception:
        CONTEXT["wan_internal_vae_decode_dtype"] = str(dtype)
    scale = getattr(vae, "scale", None)

    def _cleanup_cuda(label: str) -> None:
        gc.collect()
        try:
            if torch_mod is not None and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
                torch_mod.cuda.ipc_collect()
        except Exception:
            pass
        _stage_event(label, "cleanup after internal VAE streaming step", torch_mod)

    def _scale_latent(z: Any) -> Any:
        if not isinstance(scale, (list, tuple)) or len(scale) < 2:
            return z
        s0, s1 = scale[0], scale[1]
        try:
            if hasattr(s0, "to"):
                s0 = s0.to(device=z.device, dtype=z.dtype)
            if hasattr(s1, "to"):
                s1 = s1.to(device=z.device, dtype=z.dtype)
            if hasattr(s0, "view") and hasattr(s1, "view"):
                z_dim = int(getattr(model, "z_dim", z.shape[1]))
                return z / s1.view(1, z_dim, 1, 1, 1) + s0.view(1, z_dim, 1, 1, 1)
            return z / s1 + s0
        except Exception as e:
            raise RuntimeError(f"internal Wan VAE latent scale failed: {e}") from e

    for latent_index, latent in enumerate(latents):
        try:
            shape = tuple(getattr(latent, "shape", ()))
            if len(shape) != 4:
                raise RuntimeError(f"unsupported latent shape for item {latent_index}: {shape}")
            local_t = int(shape[1])
        except Exception as e:
            raise RuntimeError(f"Wan internal streaming decode could not inspect latent {latent_index}: {e}") from e

        pieces: list[Any] = []
        try:
            model.clear_cache()
            z = latent.unsqueeze(0)
            try:
                if dtype is not None and hasattr(z, "to"):
                    z = z.to(dtype=dtype)
                if dtype is not None and hasattr(model, "to"):
                    model.to(dtype=dtype)
                    CONTEXT["wan_internal_vae_model_dtype"] = str(dtype)
            except Exception as e:
                CONTEXT["wan_internal_vae_model_dtype"] = f"could not cast VAE model/input: {type(e).__name__}: {e}"
            _stage_event(
                "wan_internal_vae_stream_start",
                f"latent={latent_index}; z={_tensor_summary(z)}; decode_dtype={dtype}",
                torch_mod,
            )

            autocast_cm = None
            try:
                if torch_mod is not None and getattr(torch_mod, "cuda", None) is not None and hasattr(torch_mod.cuda, "amp"):
                    autocast_cm = torch_mod.cuda.amp.autocast(dtype=dtype)
            except Exception:
                autocast_cm = None

            class _NullContext:
                def __enter__(self):
                    return None
                def __exit__(self, exc_type, exc, tb):
                    return False

            infer_cm = None
            try:
                infer_cm = torch_mod.inference_mode() if torch_mod is not None and hasattr(torch_mod, "inference_mode") else None
            except Exception:
                infer_cm = None

            with (infer_cm if infer_cm is not None else _NullContext()):
              with _LowWorkspaceVaeDecodeContext(torch_mod, disable_cudnn=True):
               with (autocast_cm if autocast_cm is not None else _NullContext()):
                z = _scale_latent(z)
                with _stage_watch("wan_internal_vae_conv2", "Wan VAE conv2 before frame streaming", torch_mod):
                    x_all = model.conv2(z)
                _stage_event("wan_internal_vae_conv2_output", _tensor_summary(x_all), torch_mod)

                # Decode one latent frame at a time. Wan's own decoder/cache logic
                # expands this to 1 output frame for the first latent and normally
                # 4 output frames for each following latent.
                #
                # The previous reports proved the OOM is now inside decoder layers
                # (Conv3d in vae2_2.py), not in public decode/fallback/save. This
                # probe wraps decoder leaf modules so the next report names the
                # exact internal layer that crosses the memory guard.
                probe = _WanDecoderLayerProbe(
                    model.decoder,
                    torch_mod,
                    reserved_limit_gb=_wan_profile_reserved_limit_gb(),
                    driver_free_floor_gb=1.25,
                )
                probe.install()
                try:
                    for i in range(local_t):
                        total_frames += 1
                        model._conv_idx = [0]
                        x_slice = x_all[:, :, i:i + 1, :, :]
                        _stage_event(
                            "wan_internal_vae_frame_enter",
                            f"latent={latent_index}; frame={i + 1}/{local_t}; x_slice={_tensor_summary(x_slice)}",
                            torch_mod,
                        )
                        started = time.perf_counter()
                        with _stage_watch(
                            f"wan_internal_vae_frame_{total_frames}",
                            f"Wan internal VAE decoder latent frame {total_frames}",
                            torch_mod,
                        ):
                            if i == 0:
                                part = model.decoder(
                                    x_slice,
                                    feat_cache=model._feat_map,
                                    feat_idx=model._conv_idx,
                                    first_chunk=True,
                                )
                            else:
                                part = model.decoder(
                                    x_slice,
                                    feat_cache=model._feat_map,
                                    feat_idx=model._conv_idx,
                                )
                            part = unpatchify_fn(part, patch_size=2)
                            part = part.float().clamp_(-1, 1).squeeze(0)

                        # Move the small decoded piece off CUDA before the next latent
                        # frame can grow allocator residency.
                        part_cpu = part.detach().to("cpu")
                        pieces.append(part_cpu)
                        frame_notes.append(
                            f"latent {latent_index} frame {i + 1}/{local_t}: "
                            f"cpu_shape={tuple(getattr(part_cpu, 'shape', ()))}, "
                            f"duration={time.perf_counter() - started:.3f}s"
                        )
                        _stage_event(
                            "wan_internal_vae_frame_exit_cpu",
                            f"latent={latent_index}; frame={i + 1}/{local_t}; cpu_shape={tuple(getattr(part_cpu, 'shape', ()))}",
                            torch_mod,
                        )
                        try:
                            del part
                        except Exception:
                            pass
                        try:
                            del part_cpu
                        except Exception:
                            pass
                        try:
                            del x_slice
                        except Exception:
                            pass
                        _cleanup_cuda("wan_internal_vae_frame_cleanup")
                finally:
                    probe.restore()

            try:
                del x_all
            except Exception:
                pass
            try:
                del z
            except Exception:
                pass
        finally:
            try:
                model.clear_cache()
            except Exception:
                pass
            _cleanup_cuda("wan_internal_vae_stream_end_cleanup")

        if not pieces:
            raise RuntimeError(f"Wan internal streaming decode produced no pieces for latent {latent_index}")
        try:
            full = torch_mod.cat(pieces, dim=1) if torch_mod is not None else None
            if full is None:
                raise RuntimeError("torch module unavailable for CPU concat")
        except Exception as e:
            raise RuntimeError(f"Wan internal streaming decode CPU concat failed: {e}") from e
        decoded_full.append(full)

    CONTEXT["wan_chunked_decode_chunks"] = str(total_frames)
    CONTEXT["wan_chunked_decode_notes"] = frame_notes[-120:]
    _stage_event("wan_internal_vae_stream_complete", f"frames={total_frames}; outputs={_tensor_summary(decoded_full)}", torch_mod)
    return True, decoded_full

def _install_instance_vae_decode_guard(instance: Any, class_name: str, torch_mod: Any) -> None:
    """Wrap this Wan pipeline instance's VAE decode from outside the Wan repo.

    The previous finalize guard proved save_video itself is fast, but the slow
    shared-memory crawl happens after tqdm sampling reaches 100% and before
    generate.py logs "Saving generated video". In Wan this is inside
    pipeline.generate(), usually at self.vae.decode(x0). This wrapper releases
    VRAM Lab transformer/block residency *before* VAE decode starts and records
    the real post-denoise/decode timing without editing models/wan22.
    """
    try:
        vae = getattr(instance, "vae", None)
        orig_decode = getattr(vae, "decode", None)
    except Exception:
        vae = None
        orig_decode = None
    if vae is None or not callable(orig_decode):
        CONTEXT["finalize_vae_decode_guard"] = f"NO: {class_name}.vae.decode not callable"
        _boundary_mark("wan_vae_decode_guard_missing", f"{class_name}.vae.decode not callable")
        return
    if getattr(orig_decode, "_framevision_vram_vae_decode_wrapped", False):
        CONTEXT["finalize_vae_decode_guard"] = f"YES: {class_name}.vae.decode already wrapped"
        _boundary_mark("wan_vae_decode_guard_already_wrapped", f"{class_name}.vae.decode already wrapped")
        return

    # Explicit wrapper: on the common successful path it returns the CPU-backed
    # decoded video object, while still preserving exception telemetry.
    def wrapped_decode_returning_cpu(*args, **kwargs):
        _boundary_mark("wan_vae_decode_enter", f"{class_name}.vae.decode entered")
        _stage_event("wan_vae_decode_input", _tensor_summary({"args": args, "kwargs": kwargs}), torch_mod)
        guard = _make_finalize_guard(torch_mod, "wan_vae_decode")
        if guard is not None:
            try:
                CONTEXT["finalize_vae_decode_guard"] = f"YES: wrapped {class_name}.vae.decode"
                guard.stage("cuda_after_denoise_generation", "Wan denoise loop finished; VAE decode called")
                comps = []
                for attr in ("model", "low_noise_model", "high_noise_model", "text_encoder"):
                    try:
                        comp = getattr(instance, attr, None)
                        if attr == "text_encoder" and getattr(comp, "model", None) is not None:
                            comp = getattr(comp, "model")
                        if comp is not None:
                            comps.append(comp)
                    except Exception:
                        pass
                # Reusable VRAM Lab post-step API. This releases the known-good
                # step hooks and moves non-decode components off CUDA before VAE
                # decode. The new decode controller below then tries the live
                # Wan VAE native tiled path from outside the repo.
                if hasattr(guard, "prepare_for_decode"):
                    guard.prepare_for_decode(
                        runtimes=list(RUNTIMES),
                        components=comps,
                        label="before_vae_decode",
                        vae=vae,
                    )
                else:
                    guard.detach_runtimes(list(RUNTIMES), clear_context_key=None)
                    if comps:
                        guard.move_components_to_cpu(comps, label="before_vae_decode_components_to_cpu")
                    else:
                        CONTEXT["finalize_components_to_cpu"] = "none"
                        guard.cleanup_cuda("before_vae_decode_components_to_cpu")
                    gc.collect()
                    guard.cleanup_cuda("before_vae_decode")
                    guard.stage("cuda_before_vae_decode", "before Wan VAE decode")
                try:
                    RUNTIMES.clear()
                except Exception:
                    pass
                CONTEXT["decode_finalize_start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                CONTEXT.setdefault("finalize_guard_notes", []).append(
                    f"Wan pre-VAE decode guard failed: {type(e).__name__}: {e}"
                )
        _boundary_mark("wan_vae_decode_before_controller", "calling helper-side internal Wan VAE stream decode; no original fallback")
        started = time.perf_counter()
        try:
            CONTEXT["decode_controller_backend"] = "helper-side internal Wan VAE streaming decode; no original fallback"
            CONTEXT["decode_mode_used"] = "internal streaming only; original Wan decode disabled for VRAM Lab"
            CONTEXT["wan_original_decode_fallback_allowed"] = "NO"
            _stage_event(
                "wan_no_fallback_decode_start",
                "VRAM Lab decode must use internal streaming; original vae.decode fallback is disabled",
                torch_mod,
            )
            try:
                used_chunked, out = _maybe_chunked_wan_vae_decode(vae, orig_decode, tuple(args or ()), dict(kwargs or {}), torch_mod)
            except Exception as e:
                CONTEXT["wan_chunked_decode_used"] = f"FAILED: {type(e).__name__}: {e}"
                CONTEXT["wan_no_fallback_abort_reason"] = f"internal streaming raised {type(e).__name__}: {e}"
                CONTEXT.setdefault("decode_controller_notes", []).append(
                    f"helper-side internal Wan VAE streaming failed; original decode fallback disabled: {type(e).__name__}: {e}"
                )
                _stage_event(
                    "wan_no_fallback_decode_abort",
                    f"internal streaming failed; original decode fallback disabled: {type(e).__name__}: {e}",
                    torch_mod,
                )
                raise RuntimeError(
                    "VRAM Lab Wan VAE internal streaming decode failed and original vae.decode fallback is disabled. "
                    f"Reason: {type(e).__name__}: {e}"
                ) from e

            if not used_chunked:
                reason = str(CONTEXT.get("wan_chunked_decode_used", "internal streaming returned False without reason"))
                CONTEXT["wan_no_fallback_abort_reason"] = reason
                CONTEXT.setdefault("decode_controller_notes", []).append(
                    "helper-side internal Wan VAE streaming was not used; original decode fallback disabled: " + reason
                )
                _stage_event(
                    "wan_no_fallback_decode_abort",
                    "internal streaming rejected; original decode fallback disabled: " + reason,
                    torch_mod,
                )
                raise RuntimeError(
                    "VRAM Lab Wan VAE internal streaming decode was not available and original vae.decode fallback is disabled. "
                    f"Reason: {reason}"
                )

            _stage_event("wan_no_fallback_decode_ok", "internal streaming decode returned successfully", torch_mod)
        except BaseException:
            if guard is not None:
                try:
                    guard.stage("cuda_after_vae_decode_error", "Wan VAE decode raised")
                    guard.finish()
                except Exception:
                    pass
            raise
        CONTEXT["finalize_vae_decode_duration"] = f"{time.perf_counter() - started:.3f}s"
        _stage_event("wan_vae_decode_output_raw", _tensor_summary(out), torch_mod)
        _boundary_mark("wan_vae_decode_after_controller", "VRAM Lab decode controller returned")
        if guard is not None:
            try:
                guard.stage("cuda_after_vae_decode_raw", "after Wan VAE decode before output CPU transfer")
                out = guard.tensor_to_cpu(out, label="wan_vae_decode_output_to_cpu")
                CONTEXT["finalize_output_to_cpu"] = "YES: VAE decode output"
                guard.cleanup_cuda("after_vae_decode")
                guard.stage("cuda_after_vae_decode", "after Wan VAE decode output CPU transfer")
                guard.finish()
                CONTEXT["post_step_decode_guard_finished"] = "YES"
            except Exception as e:
                CONTEXT.setdefault("finalize_guard_notes", []).append(
                    f"Wan post-VAE decode guard failed: {type(e).__name__}: {e}"
                )
        _boundary_mark("wan_vae_decode_exit", "wrapped Wan VAE decode returning")
        return out

    try:
        wrapped_decode_returning_cpu._framevision_vram_vae_decode_wrapped = True  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        setattr(vae, "decode", wrapped_decode_returning_cpu)
        CONTEXT["finalize_vae_decode_guard"] = f"YES: wrapped {class_name}.vae.decode"
        _boundary_mark("wan_vae_decode_guard_installed", f"wrapped {class_name}.vae.decode")
    except Exception as e:
        CONTEXT["finalize_vae_decode_guard"] = f"FAILED: {type(e).__name__}: {e}"

def _write_report(status: str, error: str | None = None) -> None:
    lines: List[str] = []
    lines.append("==============================================================================")
    lines.append("FrameVision Wan 2.2 → VRAM Lab Integration")
    lines.append("==============================================================================")
    lines.append("Scope: Wan owns generation; VRAM Lab owns runtime hooks/residency.")
    lines.append(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Python: {sys.executable}")
    lines.append("")
    lines.append("Wan")
    lines.append("------------------------------------------------------------------------------")
    for key in (
        "vram_lab_mode", "wan_generate", "wan_root", "wan_task", "wan_size",
        "wan_frame_num", "wan_sample_steps", "wan_save_file", "generation_status", "output_path"
    ):
        lines.append(f"{key.replace('_', ' ')}: {CONTEXT.get(key, 'n/a')}")
    lines.append("")
    lines.append("VRAM Lab")
    lines.append("------------------------------------------------------------------------------")
    for key in (
        "gpu_name", "total_vram", "selected_policy", "vram_profile_note",
        "vram_aggressive_extra_gb", "vram_hot_block_budget", "vram_emergency_driver_free_floor",
        "wan_vram_profile", "wan_vram_profile_reserved_limit", "wan_vram_profile_note",
        "wan_flash_attention_toggle", "wan_flash_attention_arg_received", "wan_attention_backend", "wan_attention_backend_file",
        "wan_denoise_step_profiler_enabled", "wan_denoise_step_profiler_status", "wan_denoise_profile_limit",
        "wan_denoise_profiled_loop_count", "wan_denoise_profiled_step_count", "wan_denoise_total_step_time",
        "wan_denoise_avg_step_time", "wan_denoise_worst_peak_reserved_step", "wan_denoise_worst_min_driver_free_step",
        "wan_denoise_highest_allocated", "wan_denoise_highest_reserved", "wan_denoise_reserved_growth_steps",
        "wan_denoise_profile_decision", "wan_denoise_step_summary", "wan_denoise_step_profiler_failures",
        "wan_denoise_soft_guard_limit", "wan_denoise_soft_guard_full_ceiling", "wan_denoise_soft_guard_danger_start",
        "wan_denoise_soft_guard_seconds", "wan_denoise_soft_guard_corrections",
        "wan_denoise_soft_guard_aborts", "wan_denoise_soft_guard_abort_reason",
        "wan_denoise_soft_guard_recovered_aborts", "wan_denoise_soft_guard_last_recovered_abort",
        "wan_denoise_shared_memory_guard", "wan_denoise_shared_memory_guard_free_floor",
        "wan_denoise_shared_memory_guard_seconds", "wan_denoise_shared_memory_guard_last",
        "wan_denoise_shared_memory_guard_abort",
        "wan_denoise_deferred_runtime_cleanups", "wan_denoise_deferred_runtime_cleanup_last",
        "wan_denoise_sampling_active", "wan_denoise_protected_runtime_eviction",
        "wan_denoise_protected_runtime_eviction_skips", "wan_denoise_protected_runtime_eviction_last",
        "wan_denoise_protected_runtime_eviction_error",
        "wan_denoise_streaming_strategy", "wan_denoise_block_streaming", "wan_denoise_block_streaming_wrapped_blocks",
        "wan_denoise_block_streaming_prepare", "wan_denoise_block_streaming_calls",
        "wan_denoise_block_streaming_cuda_loads", "wan_denoise_block_streaming_cpu_offloads",
        "wan_denoise_block_streaming_last", "wan_denoise_block_streaming_error",
        "wan_denoise_internal_profiler", "wan_denoise_internal_profiler_target_blocks",
        "wan_denoise_internal_profiler_target_blocks_seen", "wan_denoise_internal_profiler_wrapped_modules",
        "wan_denoise_internal_profiler_max_calls", "wan_denoise_internal_profiler_calls",
        "wan_denoise_internal_profiler_records", "wan_denoise_internal_profiler_worst_reserved",
        "wan_denoise_internal_profiler_worst_allocated", "wan_denoise_internal_profiler_last",
        "wan_denoise_internal_profiler_error",
        "wan_denoise_ffn_chunking", "wan_denoise_ffn_chunking_wrapped_modules",
        "wan_denoise_ffn_chunking_chunk_size", "wan_denoise_ffn_chunking_dim",
        "wan_denoise_ffn_chunking_calls", "wan_denoise_ffn_chunking_chunks",
        "wan_denoise_ffn_chunking_fallbacks", "wan_denoise_ffn_chunking_prealloc_outputs",
        "wan_denoise_ffn_chunking_worst_reserved", "wan_denoise_ffn_chunking_worst_allocated",
        "wan_denoise_ffn_chunking_last", "wan_denoise_ffn_chunking_error",
        "wan_stage_suppressed_detail_events",
        "allocator_config_requested", "allocator_config_status", "wan_repo_touched", "hook_attach_status",
        "vram_forward_hooks_attached", "vram_hooked_component_names", "vram_hooked_block_count",
        "vram_pre_forward_calls", "vram_post_forward_calls", "vram_block_load_count",
        "vram_block_unload_count", "vram_forced_unload_count", "vram_cache_cleanup_count",
        "vram_emergency_cleanup_count", "vram_hot_block_trim_count",
        "vram_hook_block_pattern", "vram_blocks_currently_cuda", "vram_retained_cuda_refs",
        "vram_sample_hooked_block_names", "vram_active_block_name", "vram_largest_hooked_block",
        "vram_peak_cuda_during_hooked_execution", "vram_hook_failures",
        "cuda_before_run", "cuda_after_run",
        "finalize_guard_enabled", "decode_controller_enabled", "decode_controller_backend",
        "wan_vae_class_name", "wan_vae_decode_function_located", "wan_vae_decode_signature",
        "native_low_memory_decode_support_detected", "decode_controller_tile_size", "decode_mode_used",
        "wan_original_decode_fallback_allowed", "wan_no_fallback_abort_reason", "wan_chunked_decode_used",
        "wan_chunked_decode_chunks", "wan_internal_stream_latent_frames", "wan_internal_vae_decode_dtype",
        "wan_internal_vae_model_dtype", "wan_decoder_norm_fp16_cast_active", "wan_decoder_norm_fp16_cast_last_module", "wan_decoder_norm_fp16_cast_error", "wan_low_workspace_vae_decode",
        "wan_decoder_probe_enabled", "wan_decoder_probe_last_module", "wan_decoder_probe_failed_module",
        "wan_decoder_probe_failed_error", "wan_decoder_probe_abort_module", "wan_decoder_probe_abort_memory",
        "wan_decoder_probe_reserved_limit", "wan_decoder_probe_driver_free_floor", "wan_decoder_probe_restored",
        "wan_tiled_causalconv3d_active", "wan_tiled_causalconv3d_last_module", "wan_tiled_causalconv3d_last_shape",
        "wan_tiled_causalconv3d_tile", "wan_tiled_causalconv3d_tiles", "wan_tiled_causalconv3d_error",
        "wan_tiled_conv2d_active", "wan_tiled_conv2d_last_module", "wan_tiled_conv2d_last_shape",
        "wan_tiled_conv2d_tile", "wan_tiled_conv2d_tiles", "wan_tiled_conv2d_error",
        "wan_decoder_probe_tiled_module", "wan_decoder_probe_tiled_conv2d_module", "decode_controller_needed_internal_hook",
        "streaming_vae_decode_patch_installed", "streaming_vae_decode_patched_target",
        "streaming_vae_original_vae_class", "streaming_vae_internal_model_class",
        "streaming_vae_original_decode_source", "streaming_vae_decode_used",
        "streaming_vae_decode_fallback_used", "streaming_vae_decode_fallback_reason",
        "streaming_vae_decoded_chunks", "streaming_vae_chunk_temporal_lengths",
        "streaming_vae_total_temporal_length", "streaming_vae_chunks_moved_to_cpu",
        "streaming_vae_output_device", "streaming_vae_uses_inference_mode",
        "streaming_vae_peak_snapshot",
        "streaming_vae_peak_reserved", "streaming_vae_peak_reserved_stage", "streaming_vae_peak_reserved_chunk",
        "streaming_vae_peak_allocated", "streaming_vae_peak_allocated_stage", "streaming_vae_peak_allocated_chunk",
        "streaming_vae_min_driver_free", "streaming_vae_min_driver_free_stage", "streaming_vae_min_driver_free_chunk",
        "streaming_vae_decoder_chunk_times", "streaming_vae_unpatchify_chunk_times", "streaming_vae_cpu_move_chunk_times",
        "streaming_vae_decode_duration",
        "finalize_vae_decode_guard", "finalize_hooks_detached", "finalize_components_to_cpu",
        "finalize_output_to_cpu", "cuda_after_denoise_generation",
        "cuda_before_vae_decode", "cuda_after_vae_decode_raw", "cuda_after_vae_decode",
        "finalize_vae_decode_duration",
        "cuda_before_save_reencode", "cuda_after_save_reencode",
        "finalize_guard_end_memory", "finalize_save_reencode_duration",
        "finalize_guard_total_duration",
        "boundary_finder_enabled", "WanTI2V_generate_wrapper_installed", "WanTI2V_t2v_wrapper_installed",
        "WanTI2V_i2v_wrapper_installed", "WanT2V_generate_wrapper_installed", "WanI2V_generate_wrapper_installed",
        "wan_vae_decode_enter", "wan_vae_decode_before_orig",
        "wan_vae_decode_after_orig", "wan_vae_decode_exit",
        "WanTI2V.generate_duration", "WanTI2V.t2v_duration", "WanTI2V.i2v_duration",
        "WanT2V.generate_duration", "WanI2V.generate_duration",
        "failure_stage"
    ):
        lines.append(f"{key.replace('_', ' ')}: {CONTEXT.get(key, 'n/a')}")
    lines.append("")
    lines.append("Stage breakdown")
    lines.append("------------------------------------------------------------------------------")
    model_load = "n/a"
    for _k in ("WanTI2V_model_load_duration", "WanT2V_model_load_duration", "WanI2V_model_load_duration"):
        if CONTEXT.get(_k):
            model_load = str(CONTEXT.get(_k))
            break
    generate_duration = "n/a"
    for _k in ("WanTI2V.generate_duration", "WanTI2V.t2v_duration", "WanTI2V.i2v_duration", "WanT2V.generate_duration", "WanI2V.generate_duration"):
        if CONTEXT.get(_k):
            generate_duration = str(CONTEXT.get(_k))
            break
    lines.append(f"model load: {model_load}")
    lines.append("prompt/text/image encode: external-only; not separately measurable without an internal Wan hook")
    lines.append(f"visible denoise/sampling steps: included in Wan generate/t2v/i2v pre-decode window; total Wan method duration={generate_duration}")
    lines.append(f"Wan denoise step profiler: {CONTEXT.get('wan_denoise_step_profiler_enabled', 'NO')} / {CONTEXT.get('wan_denoise_step_profiler_status', 'n/a')}")
    lines.append(f"Wan denoise step summary: {CONTEXT.get('wan_denoise_step_summary', 'n/a')}")
    lines.append(f"Wan denoise worst peak reserved: {CONTEXT.get('wan_denoise_worst_peak_reserved_step', 'n/a')}")
    lines.append(f"Wan denoise worst driver-free: {CONTEXT.get('wan_denoise_worst_min_driver_free_step', 'n/a')}")
    lines.append(f"Wan denoise memory growth: {CONTEXT.get('wan_denoise_reserved_growth_steps', 'n/a')}")
    lines.append(f"Wan denoise recommendation: {CONTEXT.get('wan_denoise_profile_decision', 'n/a')}")
    lines.append(f"post-step decode/output creation start: {CONTEXT.get('decode_finalize_start_time', CONTEXT.get('cuda_before_vae_decode_time', 'n/a'))}")
    lines.append(f"post-step decode/output creation duration: {CONTEXT.get('finalize_vae_decode_duration', CONTEXT.get('decode_finalize_duration', 'n/a'))}")
    lines.append(f"CUDA before decode: {CONTEXT.get('cuda_before_vae_decode', 'n/a')}")
    lines.append(f"CUDA after decode raw: {CONTEXT.get('cuda_after_vae_decode_raw', 'n/a')}")
    lines.append(f"CUDA after decode cleanup: {CONTEXT.get('cuda_after_vae_decode', 'n/a')}")
    lines.append(f"driver free/total before decode: {CONTEXT.get('cuda_before_vae_decode_driver_free', 'n/a')} / {CONTEXT.get('cuda_before_vae_decode_driver_total', 'n/a')}")
    lines.append(f"driver free/total after decode: {CONTEXT.get('cuda_after_vae_decode_driver_free', 'n/a')} / {CONTEXT.get('cuda_after_vae_decode_driver_total', 'n/a')}")
    lines.append(f"VAE low-memory support detected externally: {CONTEXT.get('finalize_vae_low_memory_support', 'n/a')}")
    lines.append(f"decode controller enabled: {CONTEXT.get('decode_controller_enabled', 'n/a')}")
    lines.append(f"decode controller backend: {CONTEXT.get('decode_controller_backend', 'n/a')}")
    lines.append(f"Wan VAE class name: {CONTEXT.get('wan_vae_class_name', 'n/a')}")
    lines.append(f"Wan VAE decode function located: {CONTEXT.get('wan_vae_decode_function_located', 'n/a')}")
    lines.append(f"native low-memory decode support detected: {CONTEXT.get('native_low_memory_decode_support_detected', 'n/a')}")
    lines.append(f"decode mode used: {CONTEXT.get('decode_mode_used', 'n/a')}")
    lines.append(f"streaming VAE decode patch installed: {CONTEXT.get('streaming_vae_decode_patch_installed', 'n/a')}")
    lines.append(f"streaming VAE decode used: {CONTEXT.get('streaming_vae_decode_used', 'n/a')}")
    lines.append(f"streaming VAE decoded chunks: {CONTEXT.get('streaming_vae_decoded_chunks', 'n/a')}")
    lines.append(f"streaming VAE chunk temporal lengths: {CONTEXT.get('streaming_vae_chunk_temporal_lengths', 'n/a')}")
    lines.append(f"streaming VAE chunks moved to CPU: {CONTEXT.get('streaming_vae_chunks_moved_to_cpu', 'n/a')}")
    lines.append(f"streaming VAE peak reserved: {CONTEXT.get('streaming_vae_peak_reserved', 'n/a')}")
    lines.append(f"streaming VAE peak reserved stage: {CONTEXT.get('streaming_vae_peak_reserved_stage', 'n/a')}")
    lines.append(f"streaming VAE peak reserved chunk: {CONTEXT.get('streaming_vae_peak_reserved_chunk', 'n/a')}")
    lines.append(f"streaming VAE minimum driver free: {CONTEXT.get('streaming_vae_min_driver_free', 'n/a')}")
    lines.append(f"streaming VAE minimum driver free stage: {CONTEXT.get('streaming_vae_min_driver_free_stage', 'n/a')}")
    lines.append(f"streaming VAE minimum driver free chunk: {CONTEXT.get('streaming_vae_min_driver_free_chunk', 'n/a')}")
    lines.append(f"streaming VAE uses inference mode: {CONTEXT.get('streaming_vae_uses_inference_mode', 'n/a')}")
    lines.append(f"streaming VAE decoder chunk times: {CONTEXT.get('streaming_vae_decoder_chunk_times', 'n/a')}")
    lines.append(f"streaming VAE unpatchify chunk times: {CONTEXT.get('streaming_vae_unpatchify_chunk_times', 'n/a')}")
    lines.append(f"streaming VAE CPU move chunk times: {CONTEXT.get('streaming_vae_cpu_move_chunk_times', 'n/a')}")
    lines.append(f"shared-memory spill likely: {CONTEXT.get('finalize_shared_memory_spill_likely', 'UNKNOWN until finalize guard finishes')}")
    lines.append(f"save/re-encode duration: {CONTEXT.get('finalize_save_reencode_duration', 'n/a')}")
    lines.append(f"save itself fast or slow: {CONTEXT.get('finalize_save_fast_or_slow', 'n/a')}")
    lines.append(f"output moved to CPU: {CONTEXT.get('finalize_output_to_cpu', 'n/a')}")
    lines.append(f"final cleanup: {CONTEXT.get('finalize_guard_end_memory', 'n/a')}")
    if str(CONTEXT.get('decode_controller_backend', '')).lower() in ('native tiled', 'monkey-patched direct tiled'):
        lines.append(f"decode control path: {CONTEXT.get('decode_controller_backend')} via VRAM Lab controller; Wan repo untouched")
    elif CONTEXT.get('finalize_vae_decode_guard', '').startswith('YES'):
        lines.append("decode control path: external Wan instance VAE decode wrapper active; original Wan decode fallback disabled in VRAM Lab")
    else:
        lines.append("decode control path: not active; minimal next hook is Wan pipeline instance vae.decode")

    finalize_stages = CONTEXT.get("finalize_guard_stages", [])
    finalize_notes = CONTEXT.get("finalize_guard_notes", [])
    if finalize_stages or finalize_notes:
        lines.append("")
        lines.append("VRAM Lab finalize guard")
        lines.append("------------------------------------------------------------------------------")
        for note in list(finalize_stages or []):
            lines.append(str(note))
        if finalize_notes:
            lines.append("")
            lines.append("Finalize guard notes")
            for note in list(finalize_notes or []):
                lines.append(str(note))
    decode_notes = CONTEXT.get("decode_controller_notes", [])
    if decode_notes:
        lines.append("")
        lines.append("VRAM Lab decode controller notes")
        lines.append("------------------------------------------------------------------------------")
        for note in list(decode_notes or []):
            lines.append(str(note))
    stage_trace = CONTEXT.get("streaming_vae_stage_trace", [])
    if stage_trace:
        lines.append("")
        lines.append("Wan streaming VAE internal stage trace")
        lines.append("------------------------------------------------------------------------------")
        for note in list(stage_trace or []):
            lines.append(str(note))
    boundary_stages = CONTEXT.get("boundary_trace_stages", [])
    boundary_notes = CONTEXT.get("boundary_trace_notes", [])
    if boundary_stages or boundary_notes:
        lines.append("")
        lines.append("VRAM Lab boundary finder")
        lines.append("------------------------------------------------------------------------------")
        for note in list(boundary_stages or []):
            lines.append(str(note))
        if boundary_notes:
            lines.append("")
            lines.append("Boundary finder notes")
            for note in list(boundary_notes or []):
                lines.append(str(note))
    denoise_per_step = CONTEXT.get("wan_denoise_per_step", "")
    if denoise_per_step:
        lines.append("")
        lines.append("Wan denoise/sampling step logger")
        lines.append("------------------------------------------------------------------------------")
        lines.append(f"profiler: {CONTEXT.get('wan_denoise_step_profiler_enabled', 'NO')}")
        lines.append(f"profiled loops: {CONTEXT.get('wan_denoise_profiled_loop_count', '0')}")
        lines.append(f"profiled steps: {CONTEXT.get('wan_denoise_profiled_step_count', '0')}")
        lines.append(f"total step time: {CONTEXT.get('wan_denoise_total_step_time', 'n/a')}")
        lines.append(f"highest allocated: {CONTEXT.get('wan_denoise_highest_allocated', 'n/a')}")
        lines.append(f"highest reserved: {CONTEXT.get('wan_denoise_highest_reserved', 'n/a')}")
        lines.append(f"worst peak reserved: {CONTEXT.get('wan_denoise_worst_peak_reserved_step', 'n/a')}")
        lines.append(f"worst driver-free: {CONTEXT.get('wan_denoise_worst_min_driver_free_step', 'n/a')}")
        lines.append(f"growth check: {CONTEXT.get('wan_denoise_reserved_growth_steps', 'n/a')}")
        lines.append(f"soft guard: cleanup={CONTEXT.get('wan_denoise_soft_guard_limit', 'n/a')}, full_ceiling={CONTEXT.get('wan_denoise_soft_guard_full_ceiling', 'n/a')}, danger_start={CONTEXT.get('wan_denoise_soft_guard_danger_start', 'n/a')}, window={CONTEXT.get('wan_denoise_soft_guard_seconds', 'n/a')}, corrections={CONTEXT.get('wan_denoise_soft_guard_corrections', '0')}, aborts={CONTEXT.get('wan_denoise_soft_guard_aborts', '0')}")
        lines.append(f"shared-memory guard: {CONTEXT.get('wan_denoise_shared_memory_guard', 'n/a')}; floor={CONTEXT.get('wan_denoise_shared_memory_guard_free_floor', 'n/a')}; window={CONTEXT.get('wan_denoise_shared_memory_guard_seconds', 'n/a')}; last={CONTEXT.get('wan_denoise_shared_memory_guard_last', 'none')}")
        lines.append(f"soft guard abort reason: {CONTEXT.get('wan_denoise_soft_guard_abort_reason', 'none')}")
        lines.append(f"deferred runtime cleanups: {CONTEXT.get('wan_denoise_deferred_runtime_cleanups', '0')}; last={CONTEXT.get('wan_denoise_deferred_runtime_cleanup_last', 'none')}")
        lines.append(f"recommendation: {CONTEXT.get('wan_denoise_profile_decision', 'n/a')}")
        lines.append("")
        lines.append("recent per-step records:")
        for item in str(denoise_per_step).split(" | ")[-64:]:
            if item.strip():
                lines.append(item.strip())

    if CONTEXT.get("wan_denoise_internal_profiler"):
        lines.append("")
        lines.append("Wan denoise active-block internal profiler")
        lines.append("------------------------------------------------------------------------------")
        lines.append(f"profiler: {CONTEXT.get('wan_denoise_internal_profiler', 'NO')}")
        lines.append(f"target blocks: {CONTEXT.get('wan_denoise_internal_profiler_target_blocks', 'n/a')} seen={CONTEXT.get('wan_denoise_internal_profiler_target_blocks_seen', '0')}")
        lines.append(f"wrapped modules: {CONTEXT.get('wan_denoise_internal_profiler_wrapped_modules', '0')}")
        lines.append(f"calls/records: {CONTEXT.get('wan_denoise_internal_profiler_calls', '0')} / {CONTEXT.get('wan_denoise_internal_profiler_records', '0')}")
        lines.append(f"worst reserved: {CONTEXT.get('wan_denoise_internal_profiler_worst_reserved', 'n/a')}")
        lines.append(f"worst allocated: {CONTEXT.get('wan_denoise_internal_profiler_worst_allocated', 'n/a')}")
        lines.append(f"last: {CONTEXT.get('wan_denoise_internal_profiler_last', 'n/a')}")
        lines.append(f"error: {CONTEXT.get('wan_denoise_internal_profiler_error', 'n/a')}")
        lines.append("")
        lines.append("top submodules by peak reserved:")
        for line in _wan_internal_profiler_summary_lines(limit=18):
            lines.append(line)
        if WAN_DENOISE_INTERNAL_PROFILE_RECORDS:
            lines.append("")
            lines.append("recent internal profiler records:")
            for rec in WAN_DENOISE_INTERNAL_PROFILE_RECORDS[-24:]:
                try:
                    lines.append(
                        f"{rec.get('label')}: duration={float(rec.get('duration', 0.0)):.4f}s; "
                        f"peak_reserved={_fmt_bytes(rec.get('peak_reserved'))}; "
                        f"peak_allocated={_fmt_bytes(rec.get('peak_allocated'))}; "
                        f"start_reserved={_fmt_bytes(rec.get('start_reserved'))}; "
                        f"end_reserved={_fmt_bytes(rec.get('end_reserved'))}; "
                        f"Δreserved={_fmt_delta(rec.get('delta_reserved'))}; "
                        f"Δdriver={_fmt_delta(rec.get('delta_driver_used'))}"
                    )
                except Exception:
                    pass

    if CONTEXT.get("wan_denoise_ffn_chunking"):
        lines.append("")
        lines.append("Wan denoise FFN chunking")
        lines.append("------------------------------------------------------------------------------")
        lines.append(f"status: {CONTEXT.get('wan_denoise_ffn_chunking', 'n/a')}")
        lines.append(f"wrapped modules: {CONTEXT.get('wan_denoise_ffn_chunking_wrapped_modules', '0')}")
        lines.append(f"chunk size/dim: {CONTEXT.get('wan_denoise_ffn_chunking_chunk_size', 'n/a')} / {CONTEXT.get('wan_denoise_ffn_chunking_dim', 'n/a')}")
        lines.append(f"calls/chunks: {CONTEXT.get('wan_denoise_ffn_chunking_calls', '0')} / {CONTEXT.get('wan_denoise_ffn_chunking_chunks', '0')}")
        lines.append(f"fallbacks: {CONTEXT.get('wan_denoise_ffn_chunking_fallbacks', '0')}")
        lines.append(f"preallocated outputs: {CONTEXT.get('wan_denoise_ffn_chunking_prealloc_outputs', '0')}")
        lines.append(f"worst reserved: {CONTEXT.get('wan_denoise_ffn_chunking_worst_reserved', 'n/a')}")
        lines.append(f"worst allocated: {CONTEXT.get('wan_denoise_ffn_chunking_worst_allocated', 'n/a')}")
        lines.append(f"last: {CONTEXT.get('wan_denoise_ffn_chunking_last', 'n/a')}")
        lines.append(f"error: {CONTEXT.get('wan_denoise_ffn_chunking_error', 'n/a')}")

    # Internal stage logger details
    lines.append("")
    lines.append("Wan post-steps stage logger")
    lines.append("------------------------------------------------------------------------------")
    lines.append(f"stage logger enabled: {CONTEXT.get('wan_stage_logger_enabled', 'NO')}")
    lines.append(f"stage event count: {len(STAGE_EVENTS)}")
    if STAGE_EVENTS:
        candidates = []
        for ev in STAGE_EVENTS:
            key = str(ev.get("key", ""))
            if any(tag in key for tag in ("vae", "decode", "save", "generate", "t2v", "i2v", "runpy")):
                candidates.append(ev)
        biggest_driver = max(candidates or STAGE_EVENTS, key=lambda e: int(e.get("delta_driver_used", 0) or 0))
        biggest_reserved = max(candidates or STAGE_EVENTS, key=lambda e: int(e.get("delta_reserved", 0) or 0))
        lines.append(
            "largest driver-memory jump: "
            f"#{biggest_driver.get('idx')} {biggest_driver.get('key')} "
            f"delta={_fmt_delta(biggest_driver.get('delta_driver_used'))}; "
            f"cuda={biggest_driver.get('cuda')}"
        )
        lines.append(
            "largest torch-reserved jump: "
            f"#{biggest_reserved.get('idx')} {biggest_reserved.get('key')} "
            f"delta={_fmt_delta(biggest_reserved.get('delta_reserved'))}; "
            f"cuda={biggest_reserved.get('cuda')}"
        )
        lines.append("recent events:")
        for ev in STAGE_EVENTS[-80:]:
            dur = ev.get("duration", None)
            dur_txt = f"; duration={float(dur):.3f}s" if isinstance(dur, (int, float)) else ""
            peaks = ""
            if ev.get("peak_driver_used"):
                peaks = (
                    f"; peak_driver_used={_fmt_bytes(ev.get('peak_driver_used'))}; "
                    f"peak_reserved={_fmt_bytes(ev.get('peak_reserved'))}; "
                    f"min_driver_free={_fmt_bytes(ev.get('min_driver_free'))}"
                )
            lines.append(
                f"#{int(ev.get('idx', 0)):03d} {ev.get('time')} {ev.get('key')}: {ev.get('note')}"
                f"{dur_txt}; cuda={ev.get('cuda')}; "
                f"Δalloc={_fmt_delta(ev.get('delta_allocated'))}, "
                f"Δreserved={_fmt_delta(ev.get('delta_reserved'))}, "
                f"Δdriver_used={_fmt_delta(ev.get('delta_driver_used'))}{peaks}"
            )

    if error:
        lines.append("")
        lines.append("Error")
        lines.append("------------------------------------------------------------------------------")
        lines.append(error)

    pre = int(CONTEXT.get("vram_pre_forward_calls_int", 0) or 0)
    post = int(CONTEXT.get("vram_post_forward_calls_int", 0) or 0)
    completed = str(CONTEXT.get("generation_status", "")).lower() == "completed"
    decode_guard = str(CONTEXT.get("finalize_vae_decode_guard", "")).startswith("YES")
    streaming_used = str(CONTEXT.get("streaming_vae_decode_used", "")).upper().startswith("YES")
    spill_text = str(CONTEXT.get("finalize_shared_memory_spill_likely", "UNKNOWN")).upper()
    raw_driver_free = str(CONTEXT.get("cuda_after_vae_decode_raw", ""))
    raw_spill = ("driver_free=0 B" in raw_driver_free) or ("reserved=31" in raw_driver_free)
    try:
        total_bytes = int(str(CONTEXT.get("total_vram", "0")).split()[0].replace("GB", "").replace(",", ".")) * (1024 ** 3)
    except Exception:
        total_bytes = 0
    streaming_peak_reserved = int(CONTEXT.get("streaming_vae_peak_reserved_bytes", 0) or 0)
    streaming_min_free = int(CONTEXT.get("streaming_vae_min_driver_free_bytes", 10**18) or 10**18)
    streaming_spill = bool((streaming_min_free <= 0) or (total_bytes and streaming_peak_reserved > int(total_bytes * 0.98)))
    if completed and pre > 0 and post > 0 and decode_guard and streaming_used and spill_text.startswith("NO") and not raw_spill and not streaming_spill:
        decision = "PASS"
        next_step = "Step hooks and helper-side streaming VAE decode both ran without low driver-headroom symptoms. Compare timing and output quality against baseline."
    elif pre > 0 and post > 0 and decode_guard:
        decision = "WARN"
        next_step = "Step hooks still work and Wan VAE decode is now isolated. If decode still crawls, the report gives the exact internal decode boundary for the next minimal hook."
    elif pre > 0:
        decision = "WARN"
        next_step = "Wan reached live execution with VRAM Lab hooks active, but decode/finalize control did not fully engage. Focus only on the reported boundary/guard failure."
    else:
        decision = "FAIL"
        next_step = "Hooks did not fire. Fix hook attachment/discovery before changing decode/finalize behavior."
    lines.append("")
    lines.append(f"PASS/WARN/FAIL decision: {decision}")
    lines.append(f"Next recommended step: {next_step}")
    lines.append(f"WAN VRAM LAB INTEGRATION RESULT: {decision}")
    try:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            stage_lines = [
                "FrameVision Wan 2.2 VRAM Lab Stage Log",
                f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]
            for ev in STAGE_EVENTS:
                stage_lines.append(
                    f"#{int(ev.get('idx', 0)):03d} {ev.get('time')} {ev.get('key')} | {ev.get('note')} | "
                    f"cuda={ev.get('cuda')} | "
                    f"delta_alloc={_fmt_delta(ev.get('delta_allocated'))} | "
                    f"delta_reserved={_fmt_delta(ev.get('delta_reserved'))} | "
                    f"delta_driver_used={_fmt_delta(ev.get('delta_driver_used'))} | "
                    f"duration={ev.get('duration', 'n/a')} | "
                    f"peak_driver_used={_fmt_bytes(ev.get('peak_driver_used')) if ev.get('peak_driver_used') else 'n/a'} | "
                    f"peak_reserved={_fmt_bytes(ev.get('peak_reserved')) if ev.get('peak_reserved') else 'n/a'} | "
                    f"min_driver_free={_fmt_bytes(ev.get('min_driver_free')) if ev.get('min_driver_free') else 'n/a'}"
                )
            STAGE_LOG_PATH.write_text("\n".join(stage_lines) + "\n", encoding="utf-8")
        except Exception:
            pass
    except Exception:
        pass


def _extract_passthrough_value(args: List[str], flag: str, default: str = "n/a") -> str:
    try:
        if flag in args:
            idx = args.index(flag)
            if idx + 1 < len(args):
                return str(args[idx + 1])
    except Exception:
        pass
    return default


def _patch_wan_classes(mode: str, torch_mod: Any) -> None:
    if str(mode).lower() not in ("safe", "balanced", "aggressive"):
        return
    if str(VRAM_LAB_DIR) not in sys.path:
        sys.path.insert(0, str(VRAM_LAB_DIR))
    from vram_forward_hooks import attach_vram_hooks, apply_vram_lab_profile_defaults, AGGRESSIVE_EXTRA_GB  # type: ignore

    import wan  # type: ignore

    pattern = r"^(model|low_noise_model|high_noise_model)\.blocks\.\d+$"
    policy = apply_vram_lab_profile_defaults({
        "mode": str(mode).lower().strip(),
        "device": "cuda",
        "hook_name_regex": pattern,
        "max_blocks": 512,
    }, mode)
    CONTEXT["vram_profile_note"] = str(policy.get("profile_note", "n/a"))
    CONTEXT["vram_aggressive_extra_gb"] = f"{float(policy.get('aggressive_extra_gb', 0.0) or 0.0):.1f}"

    classes = []
    for name in ("WanTI2V", "WanT2V", "WanI2V"):
        cls = getattr(wan, name, None)
        if cls is not None:
            classes.append((name, cls))

    def make_init(class_name: str, cls: Any, orig_init: Any):
        def wrapped_init(self, *args, **kwargs):
            _boundary_mark("wan_model_load_start", f"constructing {class_name}")
            _model_load_started = time.perf_counter()
            orig_init(self, *args, **kwargs)
            CONTEXT[f"{class_name}_model_load_duration"] = f"{time.perf_counter() - _model_load_started:.3f}s"
            _boundary_mark("wan_model_load_end", f"constructed {class_name}; duration={CONTEXT.get(f'{class_name}_model_load_duration')}")
            component_map: Dict[str, Any] = {}
            if getattr(self, "model", None) is not None:
                component_map["model"] = getattr(self, "model")
            if getattr(self, "low_noise_model", None) is not None:
                component_map["low_noise_model"] = getattr(self, "low_noise_model")
            if getattr(self, "high_noise_model", None) is not None:
                component_map["high_noise_model"] = getattr(self, "high_noise_model")
            if component_map:
                runtime = attach_vram_hooks(component_map, policy=policy, torch_module=torch_mod)
                try:
                    setattr(self, "_framevision_vram_lab_runtime", runtime)
                except Exception:
                    pass
                RUNTIMES.append(runtime)
                _protect_runtime_block_eviction_during_wan_denoise(runtime)
                _install_wan_denoise_ffn_chunking(component_map, torch_mod)
                _install_wan_denoise_block_streaming(component_map, torch_mod)
                _install_wan_denoise_internal_block_profiler(component_map, torch_mod)
                CONTEXT["hook_attach_status"] = f"PASS: hooks attached after {class_name} construction"
                CONTEXT["selected_policy"] = f"{mode}; pattern={pattern}; {policy.get('profile_note', 'n/a')}"
                runtime.update_context(CONTEXT)
                # Install the post-denoise / VAE decode guard on the constructed
                # Wan pipeline instance. This keeps the Wan repo untouched while
                # moving cleanup earlier than save_video.
                try:
                    tracer = _make_boundary_tracer(torch_mod, "wan_boundary")
                    if tracer is not None:
                        # Wrap the real Wan instance methods only for timing. This
                        # finds whether the hidden post-step delay happens inside
                        # generate(), t2v(), i2v(), or VAE decode, without editing
                        # the Wan repo or changing return values.
                        for method_name in ("generate", "t2v", "i2v"):
                            if hasattr(self, method_name):
                                ok = tracer.wrap_bound_method(self, method_name, label=f"{class_name}.{method_name}")
                                CONTEXT[f"{class_name}_{method_name}_wrapper_installed"] = "YES" if ok else "NO"
                    _install_method_stage_loggers(self, class_name, torch_mod)
                    _install_instance_vae_decode_guard(self, class_name, torch_mod)
                except Exception as e:
                    CONTEXT["finalize_vae_decode_guard"] = f"FAILED: {type(e).__name__}: {e}"
            else:
                CONTEXT["hook_attach_status"] = f"FAIL: no model components found after {class_name} construction"
                try:
                    tracer = _make_boundary_tracer(torch_mod, "wan_boundary")
                    if tracer is not None:
                        # Wrap the real Wan instance methods only for timing. This
                        # finds whether the hidden post-step delay happens inside
                        # generate(), t2v(), i2v(), or VAE decode, without editing
                        # the Wan repo or changing return values.
                        for method_name in ("generate", "t2v", "i2v"):
                            if hasattr(self, method_name):
                                ok = tracer.wrap_bound_method(self, method_name, label=f"{class_name}.{method_name}")
                                CONTEXT[f"{class_name}_{method_name}_wrapper_installed"] = "YES" if ok else "NO"
                    _install_method_stage_loggers(self, class_name, torch_mod)
                    _install_instance_vae_decode_guard(self, class_name, torch_mod)
                except Exception as e:
                    CONTEXT["finalize_vae_decode_guard"] = f"FAILED: {type(e).__name__}: {e}"
        return wrapped_init

    patched = []
    for class_name, cls in classes:
        if getattr(cls, "_framevision_vram_lab_wrapped", False):
            continue
        orig = cls.__init__
        cls.__init__ = make_init(class_name, cls, orig)  # type: ignore[method-assign]
        try:
            cls._framevision_vram_lab_wrapped = True
        except Exception:
            pass
        patched.append(class_name)
    CONTEXT["hook_attach_status"] = f"waiting for Wan class construction; patched classes={', '.join(patched) if patched else 'none'}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Wan generate.py with FrameVision VRAM Lab hooks from outside the Wan repo.")
    parser.add_argument("--vram-lab", choices=["safe", "balanced", "aggressive"], required=True)
    parser.add_argument("--vram-profile", choices=["auto", "12", "16", "24"], default="auto", help="VRAM profile in GB, or auto; controls Wan VAE decode guard/tile sizing")
    parser.add_argument("--wan-generate", required=True)
    parser.add_argument("--wan-root", required=True)
    parser.add_argument("--disable-flash-attention", action="store_true", help="Force PyTorch SDPA fallback instead of FlashAttention for Wan attention")
    parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    ns = parser.parse_args()

    # Apply the FlashAttention toggle before probing/importing wan.modules.attention.
    # The normal Wan UI process environment is not enough for queued/VRAM-Lab jobs,
    # because they enter this helper directly.
    try:
        CONTEXT["wan_flash_attention_arg_received"] = "YES" if bool(getattr(ns, "disable_flash_attention", False)) else "NO"
    except Exception:
        CONTEXT["wan_flash_attention_arg_received"] = "unknown"
    if bool(getattr(ns, "disable_flash_attention", False)):
        os.environ["FV_WAN_DISABLE_FLASH_ATTENTION"] = "1"
        CONTEXT["wan_flash_attention_toggle"] = "OFF: UI requested SDPA fallback"
    else:
        if os.environ.get("FV_WAN_DISABLE_FLASH_ATTENTION", "").strip().lower() in ("1", "true", "yes", "on"):
            CONTEXT["wan_flash_attention_toggle"] = "OFF: disabled by environment"
        else:
            # Clear any stale inherited value if the UI/worker explicitly allowed Flash.
            os.environ.pop("FV_WAN_DISABLE_FLASH_ATTENTION", None)
            CONTEXT["wan_flash_attention_toggle"] = "ON: FlashAttention allowed"

    profile = _resolve_wan_vram_profile(str(getattr(ns, "vram_profile", "auto") or "auto").strip())
    CONTEXT["wan_vram_profile"] = profile
    CONTEXT["wan_vram_profile_reserved_limit"] = f"{_wan_profile_reserved_limit_gb():.1f} GB"
    CONTEXT["wan_vram_profile_note"] = "24GB=22.5GB guard; 16GB=14.5GB guard; 12GB=11.0GB guard; auto=<16GB -> 12, 16-22.9GB -> 16, >=23GB -> 24"

    passthrough = list(ns.passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    wan_generate = Path(ns.wan_generate).resolve()
    wan_root = Path(ns.wan_root).resolve()
    CONTEXT.update({
        "vram_lab_mode": ns.vram_lab,
        "wan_generate": str(wan_generate),
        "wan_root": str(wan_root),
        "wan_repo_touched": "NO: wrapper runs Wan repo as-is",
        "wan_task": _extract_passthrough_value(passthrough, "--task"),
        "wan_size": _extract_passthrough_value(passthrough, "--size"),
        "wan_frame_num": _extract_passthrough_value(passthrough, "--frame_num"),
        "wan_sample_steps": _extract_passthrough_value(passthrough, "--sample_steps"),
        "wan_sample_guide_scale": _extract_passthrough_value(passthrough, "--sample_guide_scale", _extract_passthrough_value(passthrough, "--guide_scale")),
        "wan_base_seed": _extract_passthrough_value(passthrough, "--base_seed", _extract_passthrough_value(passthrough, "--seed")),
        "wan_save_file": _extract_passthrough_value(passthrough, "--save_file"),
        "output_path": _extract_passthrough_value(passthrough, "--save_file"),
        "generation_status": "started",
        "allocator_config_requested": "expandable_segments:True",
    })

    # Must happen before torch import when possible.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        CONTEXT["allocator_config_status"] = "set before torch import: expandable_segments:True"
    else:
        CONTEXT["allocator_config_status"] = f"already set: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}"

    if str(wan_root) not in sys.path:
        sys.path.insert(0, str(wan_root))
    if str(APP_ROOT) not in sys.path:
        sys.path.insert(0, str(APP_ROOT))

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    err_text: str | None = None
    exit_code = 0
    try:
        os.chdir(str(wan_root))
        import torch  # type: ignore
        _make_boundary_tracer(torch, "wan_boundary")
        _boundary_mark("wan_main_after_torch_import", "torch imported; before Wan class patching")
        _log_wan_attention_backend(torch)
        CONTEXT["cuda_before_run"] = _cuda_snapshot(torch)
        try:
            if torch.cuda.is_available():
                CONTEXT["gpu_name"] = torch.cuda.get_device_name(0)
                try:
                    CONTEXT["total_vram"] = _fmt_bytes(torch.cuda.get_device_properties(0).total_memory)
                except Exception:
                    CONTEXT["total_vram"] = "n/a"
        except Exception:
            pass

        _patch_wan_classes(ns.vram_lab, torch)
        _boundary_mark("wan_after_class_patch", "Wan classes patched; before save_video guard")
        _install_wan_finalize_save_guard(torch)
        step_profiler = _install_wan_denoise_step_profiler(torch)
        _boundary_mark("wan_before_runpy", "entering Wan generate.py")
        sys.argv = [str(wan_generate)] + passthrough
        try:
            with _stage_watch("wan_runpy_total", "full Wan generate.py execution", torch):
                runpy.run_path(str(wan_generate), run_name="__main__")
        finally:
            try:
                if step_profiler is not None:
                    step_profiler.stop()
            except Exception:
                pass
        _boundary_mark("wan_after_runpy", "Wan generate.py returned")
        CONTEXT["generation_status"] = "completed"
        CONTEXT["failure_stage"] = "n/a"
        CONTEXT["cuda_after_run"] = _cuda_snapshot(torch)
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        exit_code = int(code or 0)
        if exit_code == 0:
            CONTEXT["generation_status"] = "completed"
            CONTEXT["failure_stage"] = "n/a"
        else:
            CONTEXT["generation_status"] = "failed"
            CONTEXT["failure_stage"] = "during Wan run"
            err_text = f"SystemExit: {e}"
    except BaseException as e:
        exit_code = 1
        CONTEXT["generation_status"] = "failed"
        CONTEXT["failure_stage"] = "during Wan run"
        err_text = "".join(traceback.format_exception(type(e), e, e.__traceback__))[-6000:]
        try:
            import torch  # type: ignore
            CONTEXT["cuda_after_run"] = _cuda_snapshot(torch)
        except Exception:
            pass
    finally:
        try:
            for rt in list(RUNTIMES):
                try:
                    rt.update_context(CONTEXT)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            _write_report(CONTEXT.get("generation_status", "failed"), err_text)
        finally:
            try:
                for rt in list(RUNTIMES):
                    try:
                        rt.detach_vram_hooks()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                os.chdir(str(old_cwd))
            except Exception:
                pass
            sys.argv = old_argv
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
