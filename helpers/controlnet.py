from __future__ import annotations

"""ControlNet helper glue for FrameVision Z-Image Turbo.

This module is intentionally defensive:
- It probes helpers/zimage_cli.py --help to discover supported flags.
- It only appends flags that appear to exist.
- If probing fails, it does nothing (falls back to normal Z-Image).
"""

from pathlib import Path
import re
import subprocess
from typing import Iterable, Optional, Set, Tuple, Dict, Any

_FLAG_CACHE: Dict[Tuple[str, str], Set[str]] = {}
_HELP_CACHE: Dict[Tuple[str, str], str] = {}

def _probe_help(pyexe: str, cli_path: Path) -> str:
    key = (str(pyexe), str(cli_path))
    if key in _HELP_CACHE:
        return _HELP_CACHE[key]
    try:
        p = subprocess.run(
            [str(pyexe), str(cli_path), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=8,
            shell=False,
        )
        txt = (p.stdout or "") + "\n" + (p.stderr or "")
    except Exception:
        txt = ""
    _HELP_CACHE[key] = txt
    return txt

def get_supported_flags(pyexe: str, cli_path: Path) -> Set[str]:
    """Return a set of --flags supported by zimage_cli.py (best-effort)."""
    key = (str(pyexe), str(cli_path))
    if key in _FLAG_CACHE:
        return _FLAG_CACHE[key]
    help_text = _probe_help(pyexe, cli_path)
    flags = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", help_text or ""))
    _FLAG_CACHE[key] = flags
    return flags

def _flag_looks_boolean(flag: str, help_text: str) -> bool:
    """Heuristic: argparse prints '--flag ARG' when it expects a value."""
    try:
        for line in (help_text or "").splitlines():
            if flag not in line:
                continue
            # find the first occurrence position
            idx = line.find(flag)
            after = line[idx + len(flag):].strip()
            # If it immediately lists another option or description, treat as boolean.
            if not after:
                return True
            if after.startswith(","):
                return True
            # If it looks like an arg placeholder, it's not boolean.
            # Common placeholders: ARG, PATH, FILE, INT, FLOAT, STR, etc.
            if re.match(r"^[A-Z0-9_<\[]", after):
                return False
            # If next token is another flag, it's boolean
            if after.startswith("--"):
                return True
        return False
    except Exception:
        return False

def _pick_flag(supported: Set[str], candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in supported:
            return c
    return None

def _resolve_path(p: str) -> Optional[Path]:
    try:
        pp = Path(p)
        if not pp.is_absolute():
            root = Path(__file__).resolve().parents[1]  # helpers/ -> project root
            pp = (root / pp).resolve()
        if pp.exists():
            return pp
    except Exception:
        pass
    return None

def zimage_add_controlnet_args(args: list, job: dict, pyexe: str, cli_path: Path) -> list:
    """Append ControlNet-related args to an existing argv list (best-effort)."""
    try:
        enabled = bool(job.get("controlnet_enabled"))
        if not enabled:
            return args
        img_raw = str(job.get("controlnet_image") or "").strip()
        if not img_raw:
            return args

        img_path = _resolve_path(img_raw)
        if img_path is None:
            return args

        ctype = str(job.get("controlnet_type") or "Canny").strip().lower()
        strength = float(job.get("controlnet_strength") or 0.85)
        start = float(job.get("controlnet_start") or 0.0)
        end = float(job.get("controlnet_end") or 1.0)

        supported = get_supported_flags(pyexe, cli_path)
        help_text = _probe_help(pyexe, cli_path)

        # Candidate flag names (we choose the first matching one)
        img_flag = _pick_flag(supported, [
            "--control_image", "--controlnet_image", "--cn_image", "--control_img", "--guide_image", "--cond_image"
        ])
        type_flag = _pick_flag(supported, [
            "--control_type", "--controlnet_type", "--cn_type", "--control", "--control_mode", "--cond_type"
        ])
        strength_flag = _pick_flag(supported, [
            "--control_strength", "--controlnet_strength", "--cn_strength", "--control_scale", "--cond_scale"
        ])
        start_flag = _pick_flag(supported, [
            "--control_start", "--controlnet_start", "--cn_start", "--cond_start"
        ])
        end_flag = _pick_flag(supported, [
            "--control_end", "--controlnet_end", "--cn_end", "--cond_end"
        ])
        enable_flag = _pick_flag(supported, [
            "--controlnet", "--enable_controlnet", "--enable-controlnet", "--use_controlnet", "--cn"
        ])

        # Some CLIs enable ControlNet automatically when a guide image is provided,
        # so the image flag is the most important.
        if img_flag:
            args += [img_flag, str(img_path)]

        if type_flag:
            # Normalize a few common UI names
            norm = ctype
            if norm in ("hed", "lineart"):
                pass
            elif norm in ("canny", "edges", "edge"):
                norm = "canny"
            elif norm in ("pose", "openpose"):
                norm = "pose"
            elif norm in ("depth", "midas"):
                norm = "depth"
            args += [type_flag, norm]

        if strength_flag:
            args += [strength_flag, str(strength)]

        if start_flag:
            args += [start_flag, str(start)]

        if end_flag:
            args += [end_flag, str(end)]

        # Only add an explicit enable flag if it looks boolean (no ARG placeholder)
        if enable_flag and _flag_looks_boolean(enable_flag, help_text):
            args += [enable_flag]

    except Exception:
        # Best-effort only
        return args
    return args
