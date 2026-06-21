#!/usr/bin/env python3
from __future__ import annotations

"""Isolated HiDream runner for FrameVision Planner.

Planner uses this helper for HiDream text-to-image and HiDream reference-edit
jobs.  The goal is to keep HiDream in a clean external CLI process instead of
letting the embedded Planner/UI runtime affect the model process.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


_HIDREAM_MODEL_INFO: Dict[str, Dict[str, Any]] = {
    "dev_2604_bf16": {
        "label": "Dev 2604 BF16",
        "folder": "HiDream-O1-Image-Dev-2604-BF16",
        "variant": "dev",
        "steps": 30,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
    #    "timesteps": "dev",
    },
    "dev_2604_fp8": {
        "label": "Dev 2604 FP8",
        "folder": "HiDream-O1-Image-Dev-2604-FP8",
        "variant": "dev",
        "steps": 30,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
     #   "timesteps": "dev",
    },
    "dev_fp8": {
        "label": "Dev FP8",
        "folder": "HiDream-O1-Image-Dev-FP8",
        "variant": "dev",
        "steps": 30,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
      #  "timesteps": "dev",
    },
    "dev": {
        "label": "Dev BF16",
        "folder": "HiDream-O1-Image-Dev-BF16",
        "variant": "dev",
        "steps": 30,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
   #     "timesteps": "dev",
    },
    "base_fp8": {
        "label": "Base FP8",
        "folder": "HiDream-O1-Image-FP8",
        "variant": "full",
        "steps": 50,
        "guidance_scale": 5.0,
        "shift": 3.0,
        "scheduler_name": "flash",
        "timesteps": "none",
    },
    "base": {
        "label": "Base BF16",
        "folder": "HiDream-O1-Image-BF16",
        "variant": "full",
        "steps": 50,
        "guidance_scale": 5.0,
        "shift": 3.0,
        "scheduler_name": "flash",
        "timesteps": "none",
    },
}


def _safe_str(value: Any, default: str = "") -> str:
    try:
        text = str(value if value is not None else "")
    except Exception:
        return default
    text = text.strip()
    return text if text else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return list(value)
    if isinstance(value, (str, Path)):
        text = _safe_str(value)
        return [text] if text else []
    return []


def _project_root_from_this_file() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd().resolve()


def _root(root: Any = None) -> Path:
    if root:
        try:
            return Path(root).expanduser().resolve()
        except Exception:
            pass
    return _project_root_from_this_file()


def _hidream_root(root: Path) -> Path:
    return (root / "models" / "hidream_bf16").resolve()


def _hidream_model_dir(root: Path, model_key: str) -> Path:
    info = _HIDREAM_MODEL_INFO.get(_safe_str(model_key), _HIDREAM_MODEL_INFO["base"])
    return (_hidream_root(root) / _safe_str(info.get("folder"))).resolve()


def _hidream_model_installed(root: Path, model_key: str) -> bool:
    try:
        d = _hidream_model_dir(root, model_key)
        return bool(d.exists() and (d / "config.json").exists())
    except Exception:
        return False


def _pick_hidream_model_key(root: Path, preferred: str = "") -> str:
    preferred = _safe_str(preferred)
    if preferred and _hidream_model_installed(root, preferred):
        return preferred
    for key in ("dev_2604_bf16", "dev", "dev_2604_fp8", "dev_fp8", "base_fp8", "base"):
        if _hidream_model_installed(root, key):
            return key
    return ""


def _hidream_cli_path(root: Path) -> str:
    here = Path(__file__).resolve().parent
    candidates = [
        (here / "hidream_cli.py").resolve(),
        (root / "helpers" / "hidream_cli.py").resolve(),
        (root / "hidream_cli.py").resolve(),
        (root / "models" / "hidream_bf16" / "run_hidream.py").resolve(),
        (root / "models" / "hidream_bf16" / "run_hidream_bf16.py").resolve(),
    ]
    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return str(c)
        except Exception:
            continue
    return ""


def _hidream_python_path(root: Path) -> str:
    candidates = [
        (root / "environments" / ".hidream_dev" / "python.exe").resolve(),
        (root / "environments" / ".hidream_bf16" / "python.exe").resolve(),
        (root / ".hidream_dev" / "python.exe").resolve(),
        (root / ".hidream_bf16" / "python.exe").resolve(),
    ]
    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return str(c)
        except Exception:
            continue
    return sys.executable or "python"


def _hidream_defaults_for_key(model_key: str) -> Dict[str, Any]:
    info = dict(_HIDREAM_MODEL_INFO.get(_safe_str(model_key), _HIDREAM_MODEL_INFO["base"]))
    return {
        "steps": _safe_int(info.get("steps"), 50),
        "guidance_scale": _safe_float(info.get("guidance_scale"), 0.0),
        "shift": _safe_float(info.get("shift"), 1.0),
        "scheduler_name": _safe_str(info.get("scheduler_name"), "flash") or "flash",
        "timesteps": _safe_str(info.get("timesteps"), "none") or "none",
        "variant": _safe_str(info.get("variant"), "full") or "full",
        "label": _safe_str(info.get("label"), model_key),
    }


def _hidream_missing_message(root: Path) -> str:
    searched = [str(_hidream_model_dir(root, key)) for key in ("dev_2604_bf16", "dev", "dev_2604_fp8", "dev_fp8", "base_fp8", "base")]
    return "HiDream selected but no installed model was found. Expected one of:\n" + "\n".join(searched)


def _existing_unique_reference_paths(paths: Any, *, limit: int = 5) -> List[str]:
    out: List[str] = []
    for raw in _as_list(paths):
        path = _safe_str(raw).strip().strip('"').strip("'")
        if not path:
            continue
        try:
            if os.path.isfile(path) and path not in out:
                out.append(path)
        except Exception:
            continue
        if len(out) >= max(1, int(limit or 5)):
            break
    return out


def _collect_reference_candidates(t2i_job: Dict[str, Any], *, limit: int = 5) -> Tuple[List[str], str, List[str]]:
    """Collect and expand HiDream multi-reference paths from Planner metadata."""
    hard_limit = max(1, min(5, int(limit or 5)))
    requested: List[str] = []
    sources: List[Tuple[str, Any]] = []

    def _add_source(name: str, value: Any) -> None:
        sources.append((name, value))
        for item in _as_list(value):
            text = _safe_str(item)
            if text and text not in requested:
                requested.append(text)

    for key in (
        "actual_image_model_reference_paths_passed",
        "selected_reference_sheet_paths",
        "character_reference_sheet_paths",
        "refs_used",
        "ref_images",
        "refs",
        "available_reference_sheet_paths",
    ):
        if key in t2i_job:
            _add_source(key, t2i_job.get(key))

    href = t2i_job.get("hidream_reference") if isinstance(t2i_job.get("hidream_reference"), dict) else {}
    cref = t2i_job.get("character_reference") if isinstance(t2i_job.get("character_reference"), dict) else {}
    for prefix, ref in (("hidream_reference", href), ("character_reference", cref)):
        if not isinstance(ref, dict):
            continue
        for key in (
            "actual_image_model_reference_paths_passed",
            "selected_reference_sheet_paths",
            "character_reference_sheet_paths",
            "available_reference_sheet_paths",
            "ref_images",
            "refs",
        ):
            if key in ref:
                _add_source(f"{prefix}.{key}", ref.get(key))
        sheets = ref.get("character_reference_sheets") if isinstance(ref.get("character_reference_sheets"), dict) else {}
        for slot in ("char_01", "char_02", "char_03", "char_04", "char_05"):
            if slot in sheets:
                _add_source(f"{prefix}.character_reference_sheets.{slot}", sheets.get(slot))

    selected: List[str] = []
    source_used = "no_valid_reference_paths"
    for source, values in sources:
        paths = _existing_unique_reference_paths(values, limit=hard_limit)
        if paths:
            selected = paths[:]
            source_used = source
            break

    available: List[str] = []
    for source, values in sources:
        if "available" in source or "character_reference_sheets" in source or source in ("ref_images", "refs"):
            for path in _existing_unique_reference_paths(values, limit=hard_limit):
                if path not in available:
                    available.append(path)
                if len(available) >= hard_limit:
                    break
        if len(available) >= hard_limit:
            break

    # Important Music Clip Creator behavior: if review/shot metadata only selected
    # one ref but the job still has multiple valid available refs, expand again so
    # multi-subject HiDream calls do not silently collapse to text-only/solo behavior.
    try:
        min_refs = int(t2i_job.get("hidream_min_refs") or t2i_job.get("wanted_reference_count") or 0)
    except Exception:
        min_refs = 0
    if min_refs <= 0 and len(available) > 1:
        min_refs = len(available)
    if selected and min_refs > len(selected):
        expanded = list(selected)
        for path in available:
            if path not in expanded:
                expanded.append(path)
            if len(expanded) >= min_refs or len(expanded) >= hard_limit:
                break
        if len(expanded) > len(selected):
            selected = expanded[:hard_limit]
            source_used += "+expanded_from_available_reference_paths"

    if not selected and available:
        selected = available[:hard_limit]
        source_used = "available_reference_paths"

    return selected[:hard_limit], source_used, requested


def _hidream_cli_reference_arg(cli_path: Any) -> Tuple[str, str]:
    candidates = [
        "--ref_images",
        "--reference_images",
        "--reference_image",
        "--ref_image",
        "--subject_reference_images",
        "--subject_refs",
        "--input_images",
        "--input_image",
    ]
    text = ""
    try:
        text = Path(_safe_str(cli_path)).read_text(encoding="utf-8", errors="ignore")[:500000]
    except Exception:
        text = ""
    for arg in candidates:
        if arg in text:
            return arg, "detected_in_hidream_cli"
    return "--ref_images", "legacy_default_not_detected"


def _append_hidream_reference_args(cmd: List[str], cli_path: Any, reference_paths: Any) -> Dict[str, Any]:
    ref_paths = _existing_unique_reference_paths(reference_paths, limit=5)
    arg_name, arg_source = _hidream_cli_reference_arg(cli_path)
    if not ref_paths:
        return {
            "reference_paths_requested": [_safe_str(x) for x in _as_list(reference_paths) if _safe_str(x)],
            "reference_paths_passed": [],
            "reference_arg_name": arg_name,
            "reference_arg_source": arg_source,
            "reference_handoff_supported": False,
            "reference_handoff_reason": "no_valid_reference_paths_to_pass",
        }
    if arg_name.endswith("_image") or arg_name in {"--ref_image", "--input_image", "--reference_image"}:
        for rp in ref_paths:
            cmd += [arg_name, rp]
    else:
        cmd += [arg_name] + ref_paths
    return {
        "reference_paths_requested": [_safe_str(x) for x in _as_list(reference_paths) if _safe_str(x)],
        "reference_paths_passed": ref_paths,
        "reference_arg_name": arg_name,
        "reference_arg_source": arg_source,
        "reference_handoff_supported": True,
        "reference_handoff_reason": "direct_reference_paths_added_to_hidream_command",
    }


def _looks_like_blank_or_empty_image(path: str) -> Tuple[bool, str]:
    """Best-effort guard for the beige/gray empty-latent failure images."""
    try:
        from PIL import Image, ImageStat
        with Image.open(path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if w < 32 or h < 32:
                return True, f"too_small:{w}x{h}"
            small = img.resize((64, 64))
            stat = ImageStat.Stat(small)
            std = sum(float(x) for x in stat.stddev) / max(1, len(stat.stddev))
            extrema = small.getextrema()
            span = sum(float(hi - lo) for lo, hi in extrema) / max(1, len(extrema))
            if std < 2.0 and span < 10.0:
                return True, f"low_detail_std={std:.2f}_span={span:.2f}"
    except Exception:
        pass
    return False, ""


def _clean_subprocess_env(root: Path, ref_paths: List[str]) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # Reduce random user-site / inherited PySide side effects while keeping the
    # HiDream conda/venv environment intact.
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["FRAMEVISION_ROOT"] = str(root)
    env["FRAMEVISION_HIDREAM_ISOLATED_PLANNER_CLI"] = "1"
    if ref_paths:
        try:
            env["FRAMEVISION_HIDREAM_REFERENCE_IMAGES"] = json.dumps(ref_paths, ensure_ascii=False)
            env["FRAMEVISION_HIDREAM_REFERENCE_IMAGE_PATHS"] = os.pathsep.join(ref_paths)
        except Exception:
            pass
    return env


def run_hidream_planner_image(
    *,
    t2i_job: Dict[str, Any],
    images_dir: str,
    sid: str,
    root: Any = None,
    log_func: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    root_path = _root(root)
    model_key = _safe_str(t2i_job.get("hidream_model_key")) or _pick_hidream_model_key(root_path, "")
    model_key = _pick_hidream_model_key(root_path, model_key)
    if not model_key:
        raise RuntimeError(_hidream_missing_message(root_path))
    if not _hidream_model_installed(root_path, model_key):
        raise RuntimeError(f"HiDream model '{model_key}' is not installed at: {_hidream_model_dir(root_path, model_key)}")

    cli = _hidream_cli_path(root_path)
    if not cli or not os.path.isfile(cli):
        raise RuntimeError("HiDream selected but helpers/hidream_cli.py was not found.")
    py = _hidream_python_path(root_path)
    defaults = _hidream_defaults_for_key(model_key)

    out_path = _safe_str(t2i_job.get("out_file") or t2i_job.get("target"))
    if (not out_path) or ("{" in out_path) or ("}" in out_path):
        out_path = os.path.join(str(images_dir), f"{sid}.png")
    if not os.path.splitext(out_path)[1]:
        out_path += ".png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass

    width = _safe_int(t2i_job.get("width"), 1280)
    height = _safe_int(t2i_job.get("height"), 704)
    seed = _safe_int(t2i_job.get("seed"), 0)
    steps = _safe_int(t2i_job.get("steps"), int(defaults["steps"]))
    guidance = _safe_float(t2i_job.get("guidance_scale", t2i_job.get("cfg", t2i_job.get("cfg_scale", defaults["guidance_scale"]))), float(defaults["guidance_scale"]))
    shift = _safe_float(t2i_job.get("shift", defaults["shift"]), float(defaults["shift"]))
    scheduler_name = _safe_str(t2i_job.get("scheduler_name"), str(defaults["scheduler_name"])) or "flash"
    timesteps = _safe_str(t2i_job.get("timesteps"), str(defaults["timesteps"])) or "none"

    ref_paths, ref_source, ref_requested = _collect_reference_candidates(t2i_job, limit=5)
    if len(ref_paths) > 1:
        width, height = 1920, 1088

    cmd = [
        py,
        cli,
        "--model_key", model_key,
        "--prompt", _safe_str(t2i_job.get("prompt")),
        "--output_image", out_path,
        "--width", str(width),
        "--height", str(height),
        "--seed", str(seed),
        "--steps", str(steps),
        "--guidance_scale", str(guidance),
        "--shift", str(shift),
        "--scheduler_name", scheduler_name,
        "--timesteps", timesteps,
        "--noise_scale_start", str(_safe_float(t2i_job.get("noise_scale_start", 7.5), 7.5)),
        "--noise_scale_end", str(_safe_float(t2i_job.get("noise_scale_end", 7.5), 7.5)),
        "--noise_clip_std", str(_safe_float(t2i_job.get("noise_clip_std", 2.5), 2.5)),
        "--device_map", _safe_str(t2i_job.get("device_map"), "cuda") or "cuda",
        "--resolution_mode", "framevision",
    ]
    negative = _safe_str(t2i_job.get("negative_prompt") or t2i_job.get("negative") or t2i_job.get("neg_prompt"))
    if negative:
        cmd += ["--negative_prompt", negative]

    reference_handoff = _append_hidream_reference_args(cmd, cli, ref_paths)
    ref_paths_passed = list(reference_handoff.get("reference_paths_passed") or [])
    if ref_paths_passed and bool(t2i_job.get("keep_original_aspect") or t2i_job.get("hidream_keep_original_aspect")):
        cmd += ["--keep_original_aspect"]

    logs_dir = root_path / "logs" / "hidream_planner"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    log_path = logs_dir / f"{_safe_str(sid, 'shot')}_{int(time.time())}.log"

    if log_func:
        if ref_paths_passed:
            log_func(f"[IMG] {sid} [HiDream {defaults['label']} Isolated Reference Edit] refs={len(ref_paths_passed)}, source={ref_source}, {width}x{height}, steps={steps}, cfg={guidance}")
        else:
            log_func(f"[IMG] {sid} [HiDream {defaults['label']} Isolated CLI] {width}x{height}, steps={steps}, cfg={guidance}")
        log_func(f"[HiDream isolated] cli={os.path.basename(cli)} py={py}")
        if ref_requested and not ref_paths_passed:
            log_func(f"[HiDream isolated] reference paths requested but none were valid; requested={len(ref_requested)}")

    env = _clean_subprocess_env(root_path, ref_paths_passed)
    rc = 1
    try:
        with log_path.open("w", encoding="utf-8", errors="replace") as lf:
            lf.write("[FrameVision Planner HiDream isolated CLI]\n")
            lf.write(f"root={root_path}\npython={py}\ncli={cli}\noutput={out_path}\n")
            lf.write(f"model_key={model_key}\nrefs_passed={len(ref_paths_passed)}\nref_source={ref_source}\n")
            lf.write("command:\n" + " ".join([str(x) for x in cmd]) + "\n\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(root_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                env=env,
            )
            if proc.stdout:
                for raw in proc.stdout:
                    if stop_check and stop_check():
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        raise RuntimeError("Cancelled by user.")
                    line = raw.decode("utf-8", errors="replace").rstrip() if isinstance(raw, bytes) else str(raw).rstrip()
                    if line:
                        try:
                            lf.write(line + "\n")
                            lf.flush()
                        except Exception:
                            pass
                        if log_func:
                            log_func(line)
            rc = int(proc.wait() or 0)
            lf.write(f"\nexit_code={rc}\n")
    except Exception as exc:
        raise RuntimeError(f"HiDream isolated CLI run failed: {exc}")

    ok = bool(rc == 0 and os.path.exists(out_path) and os.path.getsize(out_path) >= 1024)
    blank, blank_reason = (False, "")
    if ok:
        blank, blank_reason = _looks_like_blank_or_empty_image(out_path)
        if blank:
            ok = False
            try:
                os.rename(out_path, out_path + ".empty_failed.png")
            except Exception:
                pass
    if not ok:
        reason = f"exit code {rc}"
        if blank_reason:
            reason = f"empty/low-detail output detected ({blank_reason})"
        raise RuntimeError(f"HiDream isolated CLI failed or returned no usable output image ({reason}). See log: {log_path}")

    return {
        "ok": True,
        "files": [out_path],
        "out_file": out_path,
        "rc": rc,
        "backend": "hidream",
        "model": f"HiDream {defaults['label']}",
        "hidream_model_key": model_key,
        "hidream_model_dir": str(_hidream_model_dir(root_path, model_key)),
        "hidream_cli": cli,
        "hidream_python": py,
        "hidream_isolated_cli": True,
        "hidream_log_path": str(log_path),
        "reference_handoff": reference_handoff,
        "reference_source": ref_source,
        "actual_image_model_reference_paths_passed": ref_paths_passed,
        "refs_used": ref_paths_passed,
    }
