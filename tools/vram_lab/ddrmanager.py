# tools/vram_lab/ddrmanager.py
# FrameVision DDR RAM Manager backend
# Generic DDR / pagefile / commit profile manager for heavy AI jobs.
#
# This is intentionally model-agnostic. It does not know about LTX, Wan, Hunyuan,
# Qwen, GGUF, etc. It exposes one shared DDR policy that any helper/runner can
# ask for, then that helper translates the generic hints to its own runtime.
from __future__ import annotations

import ctypes
import gc
import json
import os
import platform
import shutil
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - fallback paths are kept for portable builds
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None  # type: ignore


_BYTES_GIB = 1024 ** 3
_BYTES_MIB = 1024 ** 2

# ddrmanager.py lives in: ROOT/tools/vram_lab/ddrmanager.py
ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "presets" / "setsave" / "ddr_manager.json"
LOG_PATH = ROOT / "logs" / "ddr_manager_latest.txt"

# Public profile names. Values are intentionally simple and app-wide.
PROFILE_OFF = "off"
PROFILE_AUTO = "auto"
PROFILE_24 = "24gb"
PROFILE_32 = "32gb"
PROFILE_64 = "64gb"
PROFILE_128 = "128gb"
_VALID_PROFILES = {PROFILE_OFF, PROFILE_AUTO, PROFILE_24, PROFILE_32, PROFILE_64, PROFILE_128}


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _now_stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _gb(value: int) -> float:
    return float(value or 0) / float(_BYTES_GIB)


def _fmt_gib(value: int) -> str:
    return f"{_gb(value):.2f} GB"


def _gib(value: float) -> int:
    return int(float(value) * _BYTES_GIB)


def _clamp(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def _profile_to_gb(profile: str) -> int:
    profile = normalize_profile(profile)
    if profile == PROFILE_24:
        return 24
    if profile == PROFILE_32:
        return 32
    if profile == PROFILE_64:
        return 64
    if profile == PROFILE_128:
        return 128
    return 0


def normalize_profile(profile: Any) -> str:
    """Normalize user/UI profile values to stable internal names."""
    text = str(profile if profile is not None else PROFILE_AUTO).strip().lower()
    text = text.replace(" ", "").replace("_", "").replace("-", "")
    if text in {"", "auto", "automatic", "default"}:
        return PROFILE_AUTO
    if text in {"0", "false", "disabled", "disable", "off", "none"}:
        return PROFILE_OFF
    if text in {"24", "24g", "24gb", "24gib", "failsafe"}:
        return PROFILE_24
    if text in {"32", "32g", "32gb", "32gib", "low", "lowram"}:
        return PROFILE_32
    if text in {"64", "64g", "64gb", "64gib", "normal"}:
        return PROFILE_64
    if text in {"128", "128g", "128gb", "128gib", "high"}:
        return PROFILE_128
    return PROFILE_AUTO


def profile_choices() -> Tuple[str, ...]:
    return (PROFILE_AUTO, PROFILE_24, PROFILE_32, PROFILE_64, PROFILE_128, PROFILE_OFF)


def _load_settings() -> Dict[str, Any]:
    try:
        if SETTINGS_PATH.is_file():
            with SETTINGS_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_settings(data: Dict[str, Any]) -> None:
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SETTINGS_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def is_enabled(default: bool = True) -> bool:
    data = _load_settings()
    profile = normalize_profile(data.get("profile", PROFILE_AUTO))
    if profile == PROFILE_OFF:
        return False
    value = data.get("enabled", default)
    if isinstance(value, bool):
        return value
    try:
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on", "enabled"}:
            return True
        if text in {"0", "false", "no", "off", "disabled"}:
            return False
    except Exception:
        pass
    return bool(default)


def ensure_settings_file(default_enabled: bool = True) -> None:
    """Create/upgrade the portable JSON settings file if it does not exist yet."""
    data = _load_settings()
    changed = False
    if not data:
        data = {
            "enabled": bool(default_enabled),
            "profile": PROFILE_AUTO,
            "module": "tools/vram_lab/ddrmanager.py",
            "note": "FrameVision DDR Manager. Portable JSON only. Select Auto / 24GB / 32GB / 64GB / 128GB / Off.",
        }
        changed = True
    if "profile" not in data:
        data["profile"] = PROFILE_AUTO
        changed = True
    if "enabled" not in data:
        data["enabled"] = bool(default_enabled)
        changed = True
    if "soft_manager" not in data:
        data["soft_manager"] = True
        changed = True
    if "hard_process_limit" not in data:
        # Keep off by default. A hard process limit can crash jobs when hit; the
        # normal manager path should use budgets/chunks/cleanup to keep jobs alive.
        data["hard_process_limit"] = False
        changed = True
    if changed:
        _save_settings(data)


def set_profile(profile: Any, enabled: Optional[bool] = None) -> Dict[str, Any]:
    """Update the portable DDR profile setting and return the saved settings."""
    ensure_settings_file(True)
    data = _load_settings()
    normalized = normalize_profile(profile)
    data["profile"] = normalized
    if enabled is not None:
        data["enabled"] = bool(enabled)
    elif normalized == PROFILE_OFF:
        data["enabled"] = False
    elif "enabled" not in data:
        data["enabled"] = True
    _save_settings(data)
    return data


def get_selected_profile(default: str = PROFILE_AUTO) -> str:
    ensure_settings_file(True)
    data = _load_settings()
    return normalize_profile(data.get("profile", default))


# -----------------------------------------------------------------------------
# System / RAM / pagefile detection
# -----------------------------------------------------------------------------

def recommended_reserve_bytes(total_ram_bytes: int, profile: Any = PROFILE_AUTO) -> int:
    """Return a sane physical RAM reserve for the selected profile.

    This reserve is the part the DDR Manager tries not to consume. The selected
    DDR profile is a behavior budget, not a statement about real installed RAM.
    """
    selected = normalize_profile(profile)
    if selected == PROFILE_24:
        return _gib(6)
    if selected == PROFILE_32:
        return _gib(8)
    if selected == PROFILE_64:
        return _gib(12)
    if selected == PROFILE_128:
        return _gib(18)

    total = max(0, _as_int(total_ram_bytes))
    if total <= 0:
        return _gib(6)

    total_gb = total / _BYTES_GIB
    reserve_gb = (total_gb * 0.25) + 2.0
    if total_gb <= 20:
        reserve_gb = min(max(reserve_gb, 5.0), 6.0)
    elif total_gb <= 40:
        reserve_gb = min(max(reserve_gb, 8.0), 10.0)
    elif total_gb <= 80:
        reserve_gb = min(max(reserve_gb, 14.0), 16.0)
    else:
        reserve_gb = min(max(reserve_gb, 20.0), 24.0)
    return int(reserve_gb * _BYTES_GIB)


def _memory_status_fallback() -> Dict[str, int]:
    # Best-effort fallback without psutil.
    if platform.system().lower() == "windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        try:
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total = int(stat.ullTotalPhys)
                avail = int(stat.ullAvailPhys)
                page_total = max(0, int(stat.ullTotalPageFile) - total)
                page_avail = max(0, int(stat.ullAvailPageFile) - avail)
                return {
                    "ram_total": total,
                    "ram_available": avail,
                    "ram_used": max(0, total - avail),
                    "pagefile_total": page_total,
                    "pagefile_available": page_avail,
                    "pagefile_used": max(0, page_total - page_avail),
                    "commit_limit": int(stat.ullTotalPageFile),
                    "commit_available": int(stat.ullAvailPageFile),
                    "commit_used": max(0, int(stat.ullTotalPageFile) - int(stat.ullAvailPageFile)),
                }
        except Exception:
            pass

    return {
        "ram_total": 0,
        "ram_available": 0,
        "ram_used": 0,
        "pagefile_total": 0,
        "pagefile_available": 0,
        "pagefile_used": 0,
        "commit_limit": 0,
        "commit_available": 0,
        "commit_used": 0,
    }


def _process_memory() -> Dict[str, int]:
    if psutil is not None:
        try:
            info = psutil.Process(os.getpid()).memory_info()
            return {"rss": int(getattr(info, "rss", 0)), "vms": int(getattr(info, "vms", 0))}
        except Exception:
            pass
    return {"rss": 0, "vms": 0}


def pagefile_status() -> Dict[str, Any]:
    """Return pagefile/swap and commit info."""
    fallback = _memory_status_fallback()

    total = fallback.get("pagefile_total", 0)
    used = fallback.get("pagefile_used", 0)
    available = fallback.get("pagefile_available", 0)

    if psutil is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sw = psutil.swap_memory()
            total = int(getattr(sw, "total", total))
            used = int(getattr(sw, "used", used))
            available = max(0, total - used)
        except Exception:
            pass

    warning_list = []
    if total <= 0:
        warning_list.append("Windows pagefile/swap appears disabled or could not be detected.")
    elif total < 16 * _BYTES_GIB:
        warning_list.append("Pagefile/swap looks small for very large model jobs.")

    return {
        "pagefile_total": int(total),
        "pagefile_used": int(used),
        "pagefile_available": int(available),
        "commit_limit": int(fallback.get("commit_limit", 0)),
        "commit_used": int(fallback.get("commit_used", 0)),
        "commit_available": int(fallback.get("commit_available", 0)),
        "warnings": warning_list,
    }


def _disk_status(path: Optional[str] = None) -> Dict[str, Any]:
    target = Path(path).resolve() if path else ROOT
    try:
        usage = shutil.disk_usage(str(target))
        return {
            "disk_path": str(target),
            "disk_total": int(usage.total),
            "disk_used": int(usage.used),
            "disk_free": int(usage.free),
        }
    except Exception:
        return {"disk_path": str(target), "disk_total": 0, "disk_used": 0, "disk_free": 0}


def _gpu_status() -> Dict[str, Any]:
    gpus = []
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                try:
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8", "ignore")
                except Exception:
                    name = f"GPU {index}"
                gpus.append({
                    "index": int(index),
                    "name": str(name),
                    "vram_total": int(mem.total),
                    "vram_used": int(mem.used),
                    "vram_available": int(max(0, mem.total - mem.used)),
                })
            pynvml.nvmlShutdown()
        except Exception:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    return {"gpus": gpus}


def _auto_profile_for_ram(total_ram_bytes: int) -> str:
    total_gb = _gb(_as_int(total_ram_bytes))
    if total_gb <= 0:
        return PROFILE_32
    if total_gb < 30:
        return PROFILE_24
    if total_gb < 56:
        return PROFILE_32
    if total_gb < 112:
        return PROFILE_64
    return PROFILE_128


def get_effective_profile(total_ram_bytes: Optional[int] = None) -> str:
    selected = get_selected_profile(PROFILE_AUTO)
    if selected == PROFILE_AUTO:
        if total_ram_bytes is None:
            total_ram_bytes = snapshot().get("ram_total", 0)
        return _auto_profile_for_ram(_as_int(total_ram_bytes))
    return selected


def snapshot() -> Dict[str, Any]:
    """Return one complete DDR/pagefile/process snapshot for callers and UI/HUD."""
    ensure_settings_file(True)

    mem = _memory_status_fallback()
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            mem["ram_total"] = int(vm.total)
            mem["ram_available"] = int(vm.available)
            mem["ram_used"] = int(vm.used)
        except Exception:
            pass

    ram_total = int(mem.get("ram_total", 0))
    selected = get_selected_profile(PROFILE_AUTO)
    effective = _auto_profile_for_ram(ram_total) if selected == PROFILE_AUTO else selected
    reserve = recommended_reserve_bytes(ram_total, effective)
    proc = _process_memory()
    page = pagefile_status()
    disk = _disk_status()
    gpu = _gpu_status()

    out: Dict[str, Any] = {
        "enabled": is_enabled(True),
        "selected_profile": selected,
        "effective_profile": effective,
        "profile_budget": _profile_to_gb(effective) * _BYTES_GIB,
        "timestamp": _now_stamp(),
        "platform": platform.platform(),
        "root": str(ROOT),
        "ram_total": ram_total,
        "ram_available": int(mem.get("ram_available", 0)),
        "ram_used": int(mem.get("ram_used", 0)),
        "ram_reserve": int(reserve),
        "process_rss": int(proc.get("rss", 0)),
        "process_vms": int(proc.get("vms", 0)),
    }
    out.update(page)
    out.update(disk)
    out.update(gpu)
    return out


# -----------------------------------------------------------------------------
# DDR profile policy
# -----------------------------------------------------------------------------

def profile_policy(profile: Any = None, total_ram_bytes: Optional[int] = None) -> Dict[str, Any]:
    """Return the generic DDR policy for a profile.

    This is the actual manager part. Models should ask for this and translate the
    generic hints into their own loader/offload/decode settings.
    """
    if profile is None:
        selected = get_selected_profile(PROFILE_AUTO)
    else:
        selected = normalize_profile(profile)

    if total_ram_bytes is None:
        # Avoid calling snapshot here to keep this function light and side-effect-free.
        mem = _memory_status_fallback()
        if psutil is not None:
            try:
                mem["ram_total"] = int(psutil.virtual_memory().total)
            except Exception:
                pass
        total_ram_bytes = int(mem.get("ram_total", 0))

    effective = _auto_profile_for_ram(total_ram_bytes) if selected == PROFILE_AUTO else selected
    enabled = effective != PROFILE_OFF and is_enabled(True)

    gb = _profile_to_gb(effective)
    # Defaults follow the simple WanGP-style idea: selected profile means selected
    # behavior budget. These are generic limits, not model-specific magic.
    table: Dict[str, Dict[str, Any]] = {
        PROFILE_24: {
            "profile_budget_gb": 24,
            "reserve_gb": 6,
            "cpu_stage_chunk_gb": 1.0,
            "max_single_cpu_load_gb": 3.0,
            "max_cached_cpu_gb": 2.0,
            "final_decode_chunk": 1,
            "preview_keep_limit": 0,
            "latent_keep_mode": "delete_after_run",
            "cleanup_level": "extreme",
            "prefer_streaming_load": True,
            "prefer_memory_map": True,
            "allow_pagefile_spill": True,
            "wait_when_commit_low": True,
            "pinned_memory": False,
            "async_transfers": False,
            "mmgp_budget_percent": "55%",
            "mmgp_star_budget_mb": 1000,
        },
        PROFILE_32: {
            "profile_budget_gb": 32,
            "reserve_gb": 8,
            "cpu_stage_chunk_gb": 2.0,
            "max_single_cpu_load_gb": 5.0,
            "max_cached_cpu_gb": 4.0,
            "final_decode_chunk": 1,
            "preview_keep_limit": 1,
            "latent_keep_mode": "minimal",
            "cleanup_level": "aggressive",
            "prefer_streaming_load": True,
            "prefer_memory_map": True,
            "allow_pagefile_spill": True,
            "wait_when_commit_low": True,
            "pinned_memory": False,
            "async_transfers": False,
            "mmgp_budget_percent": "70%",
            "mmgp_star_budget_mb": 1000,
        },
        PROFILE_64: {
            "profile_budget_gb": 64,
            "reserve_gb": 12,
            "cpu_stage_chunk_gb": 6.0,
            "max_single_cpu_load_gb": 10.0,
            "max_cached_cpu_gb": 10.0,
            "final_decode_chunk": 2,
            "preview_keep_limit": 3,
            "latent_keep_mode": "limited",
            "cleanup_level": "normal",
            "prefer_streaming_load": True,
            "prefer_memory_map": True,
            "allow_pagefile_spill": True,
            "wait_when_commit_low": True,
            "pinned_memory": True,
            "async_transfers": True,
            "mmgp_budget_percent": "85%",
            "mmgp_star_budget_mb": 3000,
        },
        PROFILE_128: {
            "profile_budget_gb": 128,
            "reserve_gb": 18,
            "cpu_stage_chunk_gb": 12.0,
            "max_single_cpu_load_gb": 20.0,
            "max_cached_cpu_gb": 24.0,
            "final_decode_chunk": 4,
            "preview_keep_limit": 5,
            "latent_keep_mode": "keep_limited",
            "cleanup_level": "light",
            "prefer_streaming_load": False,
            "prefer_memory_map": True,
            "allow_pagefile_spill": True,
            "wait_when_commit_low": False,
            "pinned_memory": True,
            "async_transfers": True,
            "mmgp_budget_percent": "90%",
            "mmgp_star_budget_mb": 5000,
        },
    }
    base = table.get(effective, table[PROFILE_32]).copy()
    if effective == PROFILE_OFF:
        base = {
            "profile_budget_gb": 0,
            "reserve_gb": 0,
            "cpu_stage_chunk_gb": 0,
            "max_single_cpu_load_gb": 0,
            "max_cached_cpu_gb": 0,
            "final_decode_chunk": 0,
            "preview_keep_limit": 999,
            "latent_keep_mode": "unchanged",
            "cleanup_level": "off",
            "prefer_streaming_load": False,
            "prefer_memory_map": False,
            "allow_pagefile_spill": True,
            "wait_when_commit_low": False,
            "pinned_memory": None,
            "async_transfers": None,
            "mmgp_budget_percent": "100%",
            "mmgp_star_budget_mb": 0,
        }

    base.update({
        "enabled": bool(enabled),
        "selected_profile": selected,
        "effective_profile": effective,
        "profile_budget": _gib(gb) if gb else 0,
        "ram_reserve": _gib(float(base.get("reserve_gb", 0))),
        "cpu_stage_chunk_bytes": _gib(float(base.get("cpu_stage_chunk_gb", 0))),
        "max_single_cpu_load_bytes": _gib(float(base.get("max_single_cpu_load_gb", 0))),
        "max_cached_cpu_bytes": _gib(float(base.get("max_cached_cpu_gb", 0))),
        "manager_mode": "budget_profile" if enabled else "off",
    })
    return base


def get_runtime_policy(
    label: str = "heavy AI job",
    estimated_peak_ram_bytes: int = 0,
    job_kind: str = "generic",
    write_log: bool = False,
) -> Dict[str, Any]:
    """Return snapshot + generic policy + route for a heavy runtime.

    This should be the preferred call for model-agnostic FrameVision helpers.
    It does not abort. It tells callers how to stay inside the selected DDR
    behavior profile.
    """
    snap = snapshot()
    policy = profile_policy(snap.get("selected_profile"), snap.get("ram_total"))
    estimated_peak = max(0, _as_int(estimated_peak_ram_bytes))

    enabled = bool(policy.get("enabled", True))
    budget = int(policy.get("profile_budget", 0))
    reserve = int(policy.get("ram_reserve", snap.get("ram_reserve", 0)))
    available = int(snap.get("ram_available", 0))
    used = int(snap.get("ram_used", 0))
    commit_available = int(snap.get("commit_available", 0))
    page_available = int(snap.get("pagefile_available", 0))

    warnings_list = list(snap.get("warnings", []) or [])
    actions = []
    route = "manager_off" if not enabled else "profile_ok"
    reason = "DDR Manager is disabled." if not enabled else "Using selected DDR profile budget."

    if enabled:
        # Treat the selected profile as a behavior budget. If the real machine has
        # more RAM, the budget still controls chunks/cache/retention. If it has
        # less RAM, the stricter real available amount wins.
        budget_headroom = max(0, budget - used) if budget else available
        usable_now = max(0, min(available, budget_headroom) - reserve)
        if available <= reserve:
            route = "cleanup_wait_continue"
            reason = "RAM is near reserve; cleanup/wait before next heavy stage."
            actions += ["cleanup", "wait", "continue"]
        elif estimated_peak and estimated_peak > usable_now:
            route = "survival_mode"
            reason = "Estimated peak exceeds current DDR budget headroom; use smaller chunks and safer cache policy."
            actions += ["reduce_chunks", "stream_load", "cleanup", "continue"]
        else:
            actions += ["continue"]

        if int(policy.get("cpu_stage_chunk_bytes", 0)) > 0:
            actions.append("cap_cpu_stage_chunks")
        if policy.get("cleanup_level") in {"aggressive", "extreme"}:
            actions.append("cleanup_between_stages")
        if policy.get("latent_keep_mode") in {"delete_after_run", "minimal"}:
            actions.append("limit_latent_preview_retention")
        if policy.get("wait_when_commit_low") and commit_available and estimated_peak and commit_available < estimated_peak:
            route = "cleanup_wait_continue"
            reason = "Windows commit headroom is low; cleanup/wait before continuing."
            warnings_list.append("Windows commit availability is lower than the estimated peak.")
            actions += ["wait_for_commit"]
        if policy.get("allow_pagefile_spill") and page_available and page_available < _gib(4):
            warnings_list.append("Pagefile headroom is low for survival mode.")

    out: Dict[str, Any] = {}
    out.update(snap)
    out.update({f"policy_{k}": v for k, v in policy.items()})
    out.update({
        "label": str(label or "heavy AI job"),
        "job_kind": str(job_kind or "generic"),
        "route": route,
        "reason": reason,
        "actions": sorted(set(actions)),
        "estimated_peak_ram": estimated_peak,
        "warnings": warnings_list,
    })
    if write_log:
        write_latest_log(out)
    return out


def apply_policy_to_kwargs(kwargs: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply generic DDR hints to a kwargs dict used by offload-style runners.

    This does not import MMGP and is safe for any helper. If a runner understands
    keys like budgets/pinnedMemory/asyncTransfers, it can use the returned dict.
    Runners that do not understand these keys can still read the explicit
    ddr_policy field.
    """
    if kwargs is None:
        kwargs = {}
    else:
        kwargs = dict(kwargs)
    if policy is None:
        policy = profile_policy()

    kwargs["ddr_policy"] = policy.copy()
    if not policy.get("enabled", True):
        return kwargs

    # MMGP/WanGP-style generic budgets: safe to ignore by callers that do not use them.
    budgets = kwargs.get("budgets")
    if not isinstance(budgets, dict):
        budgets = {}
    star_budget = int(policy.get("mmgp_star_budget_mb", 0) or 0)
    if star_budget > 0:
        budgets.setdefault("*", star_budget)
    kwargs["budgets"] = budgets

    pinned = policy.get("pinned_memory", None)
    if pinned is not None:
        kwargs.setdefault("pinnedMemory", bool(pinned))
    async_transfers = policy.get("async_transfers", None)
    if async_transfers is not None:
        kwargs.setdefault("asyncTransfers", bool(async_transfers))

    kwargs.setdefault("prefer_streaming_load", bool(policy.get("prefer_streaming_load", False)))
    kwargs.setdefault("prefer_memory_map", bool(policy.get("prefer_memory_map", False)))
    kwargs.setdefault("cpu_stage_chunk_bytes", int(policy.get("cpu_stage_chunk_bytes", 0)))
    kwargs.setdefault("max_single_cpu_load_bytes", int(policy.get("max_single_cpu_load_bytes", 0)))
    kwargs.setdefault("max_cached_cpu_bytes", int(policy.get("max_cached_cpu_bytes", 0)))
    kwargs.setdefault("final_decode_chunk", int(policy.get("final_decode_chunk", 0)))
    kwargs.setdefault("cleanup_level", str(policy.get("cleanup_level", "normal")))
    kwargs.setdefault("latent_keep_mode", str(policy.get("latent_keep_mode", "limited")))
    return kwargs


def cleanup_memory(level: Optional[str] = None) -> Dict[str, Any]:
    """Generic cleanup helper. Does not unload app-specific models by itself."""
    policy = profile_policy()
    level = str(level or policy.get("cleanup_level", "normal")).lower()
    before = snapshot()
    collected = 0
    try:
        collected = int(gc.collect())
    except Exception:
        collected = 0
    # Optional torch cleanup without requiring torch as dependency.
    torch_cache_flushed = False
    try:
        import torch  # type: ignore
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if level in {"aggressive", "extreme"}:
                torch.cuda.ipc_collect()
            torch_cache_flushed = True
    except Exception:
        pass
    if level == "extreme":
        try:
            time.sleep(0.05)
        except Exception:
            pass
    after = snapshot()
    result = {
        "timestamp": _now_stamp(),
        "level": level,
        "gc_collected": collected,
        "torch_cache_flushed": torch_cache_flushed,
        "ram_available_before": before.get("ram_available", 0),
        "ram_available_after": after.get("ram_available", 0),
        "commit_available_before": before.get("commit_available", 0),
        "commit_available_after": after.get("commit_available", 0),
    }
    write_latest_log(result)
    return result


def wait_for_headroom(
    needed_bytes: int = 0,
    timeout_sec: float = 20.0,
    poll_sec: float = 0.5,
    cleanup_first: bool = True,
) -> Dict[str, Any]:
    """Wait briefly for DDR/commit headroom instead of aborting a job."""
    needed = max(0, _as_int(needed_bytes))
    policy = profile_policy()
    if cleanup_first:
        cleanup_memory(policy.get("cleanup_level", "normal"))
    start = time.time()
    last = snapshot()
    ok = False
    while True:
        snap = snapshot()
        last = snap
        reserve = int(policy.get("ram_reserve", snap.get("ram_reserve", 0)))
        ram_ok = int(snap.get("ram_available", 0)) >= reserve + needed
        commit_avail = int(snap.get("commit_available", 0))
        commit_ok = True if commit_avail <= 0 or needed <= 0 else commit_avail >= needed
        if ram_ok and commit_ok:
            ok = True
            break
        if time.time() - start >= float(timeout_sec):
            break
        try:
            time.sleep(max(0.05, float(poll_sec)))
        except Exception:
            break
    result = {
        "ok": ok,
        "waited_sec": round(time.time() - start, 3),
        "needed_bytes": needed,
        "snapshot": last,
        "policy": policy,
    }
    write_latest_log(result)
    return result


# -----------------------------------------------------------------------------
# Backward-compatible preflight decisions
# -----------------------------------------------------------------------------

def preflight_load(
    label: str,
    file_size_bytes: int = 0,
    estimated_extra_ram_bytes: int = 0,
    allow_pagefile: bool = True,
    write_log: bool = True,
) -> Dict[str, Any]:
    """Return a clear survival decision for heavy file/model loading.

    Backward compatible with the old checker, but no longer defaults to an abort
    style route. It prefers survival actions: reduce chunks, stream/mmap, cleanup,
    wait, continue. Callers can still choose to abort in truly critical cases.
    """
    snap = snapshot()
    policy = profile_policy(snap.get("selected_profile"), snap.get("ram_total"))

    label = str(label or "heavy model load")
    file_size = max(0, _as_int(file_size_bytes))
    extra = max(0, _as_int(estimated_extra_ram_bytes))
    estimated_need = file_size + extra

    ram_available = int(snap.get("ram_available", 0))
    ram_total = int(snap.get("ram_total", 0))
    ram_reserve = int(policy.get("ram_reserve", snap.get("ram_reserve", recommended_reserve_bytes(ram_total))))
    projected_ram_available = ram_available - estimated_need

    page_total = int(snap.get("pagefile_total", 0))
    page_available = int(snap.get("pagefile_available", 0))
    commit_available = int(snap.get("commit_available", 0))
    disk_free = int(snap.get("disk_free", 0))

    warnings_list = list(snap.get("warnings", []) or [])
    enabled = bool(policy.get("enabled", True))

    allowed = True
    route = "profile_ok"
    reason = "Using selected DDR profile budget."
    actions = ["continue"]

    if not enabled:
        route = "manager_off"
        reason = "DDR RAM Manager is disabled in presets/setsave/ddr_manager.json."
    elif ram_total <= 0 or ram_available <= 0:
        route = "survival_mode"
        reason = "Could not reliably detect system RAM; use conservative route."
        actions = ["stream_load", "reduce_chunks", "cleanup", "continue"]
    elif ram_available <= ram_reserve:
        route = "cleanup_wait_continue"
        reason = "Physical RAM is close to the selected DDR reserve."
        actions = ["cleanup", "wait", "continue"]
    elif estimated_need <= 0:
        if ram_available < (ram_reserve + _gib(2)):
            route = "cleanup_wait_continue"
            reason = "No load size was provided and physical RAM is close to reserve."
            actions = ["cleanup", "wait", "continue"]
        else:
            reason = "No large load estimate provided; current DDR state looks safe."
    elif projected_ram_available < ram_reserve:
        # This used to be a deny. For the real manager it becomes survival mode.
        route = "survival_mode"
        reason = "Full CPU load would cross the DDR reserve; use streaming/chunked route."
        actions = ["stream_load", "memory_map", "reduce_chunks", "cleanup", "continue"]
    elif projected_ram_available < (ram_reserve + _gib(2)):
        route = "allow_with_pagefile" if allow_pagefile and page_total > 0 else "survival_mode"
        reason = "Close to DDR reserve; use safer chunks and cleanup between stages."
        actions = ["reduce_chunks", "cleanup", "continue"]
    else:
        reason = "Enough DDR headroom after estimated load."

    if allow_pagefile and estimated_need > 0:
        if page_total <= 0:
            warnings_list.append("Pagefile/swap is disabled or not detected; survival mode has less room to recover.")
        elif page_available < min(max(estimated_need // 4, _gib(2)), _gib(16)):
            warnings_list.append("Pagefile/swap available space is low for this estimated load.")

    if disk_free and disk_free < _gib(10):
        warnings_list.append("Installation drive has less than 10 GB free; temporary files/pagefile growth may be risky.")

    if allow_pagefile and commit_available and estimated_need and commit_available < estimated_need:
        route = "cleanup_wait_continue" if allowed else route
        reason = "Commit headroom is lower than the estimated load; cleanup/wait or use safer route."
        warnings_list.append("Windows commit availability is lower than the estimated load.")
        actions += ["wait_for_commit"]

    decision: Dict[str, Any] = {
        "allowed": bool(allowed),
        "route": route,
        "reason": reason,
        "actions": sorted(set(actions)),
        "label": label,
        "timestamp": snap.get("timestamp", _now_stamp()),
        "selected_profile": snap.get("selected_profile", PROFILE_AUTO),
        "effective_profile": snap.get("effective_profile", PROFILE_32),
        "profile_budget": int(policy.get("profile_budget", 0)),
        "ram_total": ram_total,
        "ram_available": ram_available,
        "ram_used": int(snap.get("ram_used", 0)),
        "ram_reserve": ram_reserve,
        "projected_ram_available": int(projected_ram_available),
        "pagefile_total": page_total,
        "pagefile_used": int(snap.get("pagefile_used", 0)),
        "pagefile_available": page_available,
        "commit_limit": int(snap.get("commit_limit", 0)),
        "commit_used": int(snap.get("commit_used", 0)),
        "commit_available": commit_available,
        "disk_free": disk_free,
        "process_rss": int(snap.get("process_rss", 0)),
        "process_vms": int(snap.get("process_vms", 0)),
        "file_size": file_size,
        "estimated_extra_ram": extra,
        "estimated_need": int(estimated_need),
        "allow_pagefile": bool(allow_pagefile),
        "warnings": warnings_list,
        "policy": policy,
    }

    if write_log:
        write_latest_log(decision)

    return decision


def format_decision(decision: Dict[str, Any]) -> str:
    label = str(decision.get("label", "DDR preflight"))
    allowed = bool(decision.get("allowed", True))
    route = str(decision.get("route", "warn"))
    reason = str(decision.get("reason", ""))
    status = "ALLOW" if allowed else "DENY"
    profile = str(decision.get("effective_profile", decision.get("selected_profile", "auto"))).upper()
    parts = [
        f"[{status}] {label}: DDR {profile} • {route} — {reason}",
        f"RAM avail {_fmt_gib(_as_int(decision.get('ram_available')))} / reserve {_fmt_gib(_as_int(decision.get('ram_reserve')))}",
    ]
    budget = _as_int(decision.get("profile_budget"))
    if budget:
        parts.append(f"profile budget {_fmt_gib(budget)}")
    file_size = _as_int(decision.get("file_size"))
    extra = _as_int(decision.get("estimated_extra_ram"))
    if file_size or extra:
        parts.append(f"load {_fmt_gib(file_size)} + extra {_fmt_gib(extra)}")
    page_total = _as_int(decision.get("pagefile_total"))
    page_avail = _as_int(decision.get("pagefile_available"))
    if page_total:
        parts.append(f"pagefile avail {_fmt_gib(page_avail)} / total {_fmt_gib(page_total)}")
    actions = decision.get("actions") or []
    if actions:
        parts.append("actions: " + ", ".join(str(a) for a in actions[:6]))
    warnings_found = decision.get("warnings") or []
    if warnings_found:
        parts.append("warning: " + " | ".join(str(w) for w in warnings_found[:3]))
    return " ; ".join(parts)


def write_latest_log(decision_or_snapshot: Dict[str, Any]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("w", encoding="utf-8") as f:
            if "route" in decision_or_snapshot:
                f.write(format_decision(decision_or_snapshot) + "\n\n")
            json.dump(decision_or_snapshot, f, indent=2)
            f.write("\n")
    except Exception:
        pass


def status_line() -> str:
    """One-line status for System Monitor / live HUD integration."""
    snap = snapshot()
    enabled = "ON" if snap.get("enabled", True) else "OFF"
    selected = str(snap.get("selected_profile", PROFILE_AUTO)).upper()
    effective = str(snap.get("effective_profile", PROFILE_32)).upper()
    profile_text = effective if selected == PROFILE_AUTO.upper() else selected
    warning = ""
    warning_list = snap.get("warnings") or []
    if warning_list:
        warning = f" • {warning_list[0]}"
    return (
        f"DDR Manager {enabled} • profile {profile_text} "
        f"• RAM {_fmt_gib(_as_int(snap.get('ram_available')))} free "
        f"/ {_fmt_gib(_as_int(snap.get('ram_total')))} total "
        f"• reserve {_fmt_gib(_as_int(snap.get('ram_reserve')))} "
        f"• pagefile {_fmt_gib(_as_int(snap.get('pagefile_available')))} free"
        f"{warning}"
    )


def quick_ok(label: str, estimated_total_bytes: int, allow_pagefile: bool = True) -> bool:
    """Compatibility wrapper. True means continue; callers should read route for survival hints."""
    return bool(preflight_load(label, estimated_total_bytes, 0, allow_pagefile, write_log=True).get("allowed", True))


def quick_policy_kwargs(kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Tiny convenience wrapper for helpers that want DDR policy injected."""
    return apply_policy_to_kwargs(kwargs, profile_policy())


if __name__ == "__main__":
    # Manual smoke test: python tools/vram_lab/ddrmanager.py
    ensure_settings_file(True)
    d = preflight_load("manual smoke test", file_size_bytes=0, estimated_extra_ram_bytes=0)
    print(format_decision(d))
    print(status_line())
    print(json.dumps(profile_policy(), indent=2))
