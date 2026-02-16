
"""
Planner (PySide6) - Prompt -> Finished Video pipeline UI 

Goal:
- Provide a clear, wire-ready UI and pipeline that is connected to:
  - Qwen3-VL (story + prompts + describe + song lyrics)
  - Whisper (transcript)
  - Qwen3 TTS (voice)
  - Ace step and Heartmula, for instrumental background music (ace) or full tracks (HeartMula)
  - Image generation (Z-Image, Qwen 2512, SDXL, etc.)
  - Video generation (Qwen 2.2 5B, HunyuanVideo, etc.)
  - Videoclip Creator preset runner (FrameVision internal tool)

This file focuses on:
- being user friendly : little work for users, a lot of work behind the scenes with hardcoded 'best settings' for most tools
- Options + UX with user friendl
- A staged pipeline runner (QThread) with logs + progress
- A job config dict that can be serialized and passed to later code
"""

from __future__ import annotations

import os
import shutil
import json
import time
import datetime
import uuid
import hashlib
import sys
import io
import contextlib
import traceback
import random
import re
import subprocess
import tempfile
import gc
import queue
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple


# -----------------------------
# Project root / path helpers
# -----------------------------
# Planner lives in <root>/helpers/planner.py
# outputs and model paths rooted at <root>, not inside /helpers.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]  # one folder up from helpers/

def _root() -> Path:
    return _PROJECT_ROOT

def _default_output_base() -> str:
    return str((_root() / "output" / "planner").resolve())

def _abspath_from_root(p: str) -> str:
    try:
        pp = Path(p)
        if not pp.is_absolute():
            pp = _root() / pp
        return str(pp.resolve())
    except Exception:
        return str((_root() / p).resolve())

def _qwen_model_dir() -> str:
    return str((_root() / "models" / "describe" / "default" / "qwen3vl2b").resolve())



def _qwen3tts_tokenizer_path() -> str:
    # Qwen3 TTS expects these model folders under <root>/models/
    return str((_root() / "models" / "Qwen3-TTS-Tokenizer-12Hz").resolve())


def _qwen3tts_model_path_for_mode(mode: str) -> str:
    m = (mode or "").strip().lower()
    root_dir = _root() / "models"
    if m == "custom":
        return str((root_dir / "Qwen3-TTS-12Hz-1.7B-CustomVoice").resolve())
    # clone / design use the Base model
    return str((root_dir / "Qwen3-TTS-12Hz-1.7B-Base").resolve())



# -----------------------------
# VRAM guard (best-effort)
# -----------------------------
def _vram_release(tag: str = "") -> None:
    """Best-effort release of GPU memory between heavy stages.

    This cannot force third-party CUDA contexts to unload, but it helps reduce
    fragmentation and frees PyTorch allocator caches when models ran in-process.
    """
    try:
        gc.collect()
    except Exception:
        pass

    # Optional unload hooks (only if present; never required).
    try:
        for mod_name, fn_names in (
            ("helpers.txt2img", ("unload_models", "unload", "cleanup", "free")),
            ("helpers.qwen2511", ("unload_models", "unload", "cleanup", "free")),
            ("helpers.qwen2512", ("unload_models", "unload", "cleanup", "free")),
            ("helpers.prompt", ("unload_models", "unload", "cleanup", "free", "release_model", "release")),
        ):
            try:
                mod = __import__(mod_name, fromlist=["*"])
            except Exception:
                continue
            for fn in fn_names:
                try:
                    f = getattr(mod, fn, None)
                    if callable(f):
                        f()
                        break
                except Exception:
                    continue
    except Exception:
        pass

    # Torch CUDA cache cleanup
    try:
        import torch  # type: ignore
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                # Frees IPC handles / shared blocks in some cases.
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass

# -----------------------------
# Probe & logs (debug only)
# -----------------------------
# When enabled, writes two logs under <root>/logs/:
# - probe.log: startup + anything "moving"
# - job.log: per-job pipeline actions
#
# This is intentionally simple (no rotation) and should stay OFF by default.

_PLANNER_SETTINGS_PATH = _root() / "presets" / "setsave" / "planner_settings.json"

# Chunk 9B1: Planner-only upscaling settings (per-project folder)
_PLANNER_UPSCALE_JSON_NAME = "planner_upscale.json"


def _load_planner_settings() -> Dict[str, Any]:
    try:
        if _PLANNER_SETTINGS_PATH.exists():
            with open(_PLANNER_SETTINGS_PATH, "r", encoding="utf-8") as f:
                obj = json.load(f)
                return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}

def _save_planner_settings(obj: Dict[str, Any]) -> None:
    try:
        _PLANNER_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PLANNER_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# info: Chunk 10 side quest — Own storyline prompt parser (Step 2: preview only)
# Splits the user-pasted storyline into a list of prompts.
# Deterministic rules (no "smart" NLP):
#   1) Marker mode: lines that start with [01] or (01) or [prompt 01] / (prompt 01)
#   2) List mode: bullets (-, *, •) or numbered lists (1., 2)
#   3) Paragraph mode: split on blank lines

def _parse_own_storyline_prompts(text: str) -> Tuple[List[Dict[str, Any]], str]:
    src = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = src.split("\n")

    marker_re = re.compile(r"^\s*[\[\(]\s*(?:prompt\s*)?(\d{1,3})\s*[\]\)]\s*[:\-–—]?\s*(.*)$", re.IGNORECASE)
    bullet_re = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+(.*)$")

    prompts: List[Dict[str, Any]] = []

    # --- Tier 1: marker mode ---
    found_marker = False
    cur_num: Optional[int] = None
    cur_lines: List[str] = []

    for ln in lines:
        mm = marker_re.match(ln)
        if mm:
            found_marker = True
            # flush previous
            if cur_lines:
                body = "\n".join(cur_lines).strip()
                if body:
                    prompts.append({"index": int(cur_num) if cur_num is not None else (len(prompts) + 1), "text": body})
            # start new
            try:
                cur_num = int(mm.group(1))
            except Exception:
                cur_num = None
            remainder = (mm.group(2) or "").strip()
            cur_lines = [remainder] if remainder else []
            continue

        if found_marker:
            # continuation line (can include blanks)
            cur_lines.append(ln)

    if found_marker:
        if cur_lines:
            body = "\n".join(cur_lines).strip()
            if body:
                prompts.append({"index": int(cur_num) if cur_num is not None else (len(prompts) + 1), "text": body})
        return prompts, "marker"

    # --- Tier 2: list mode ---
    found_list = False
    cur_lines = []
    cur_has_item = False

    def flush_list() -> None:
        nonlocal cur_lines
        body = "\n".join(cur_lines).strip()
        cur_lines = []
        if body:
            prompts.append({"index": len(prompts) + 1, "text": body})

    for ln in lines:
        bm = bullet_re.match(ln)
        if bm:
            found_list = True
            if cur_has_item:
                flush_list()
            cur_has_item = True
            cur_lines = [(bm.group(1) or "").strip()]
            continue
        if not found_list:
            continue
        # continuation lines: keep if indented and not empty; stop on blank line
        if ln.strip() == "":
            if cur_has_item:
                flush_list()
                cur_has_item = False
            continue
        if cur_has_item:
            cur_lines.append(ln.strip())

    if found_list:
        if cur_has_item:
            flush_list()
        return prompts, "list"

    # --- Tier 3: paragraph mode ---
    raw = src.strip()
    if not raw:
        return [], "paragraph"

    parts = re.split(r"\n\s*\n+", raw)
    for part in parts:
        s = (part or "").strip()
        if not s:
            continue
        # collapse internal newlines to spaces for prompt friendliness
        s = re.sub(r"\s*\n\s*", " ", s).strip()
        if s:
            prompts.append({"index": len(prompts) + 1, "text": s})

    return prompts, "paragraph"



# -----------------------------
# Default UI negatives (always reset on app start)
# -----------------------------
# These are meant to prevent common "collage / cloning / multi-panel" drift.
# Users can edit/remove them per-session; they will revert on next restart.

_DEFAULT_UI_NEGATIVES = (
        "duplicate person, cloned face, multiple faces, extra face, "
    "blurry, low quality, out of frame, jpeg artifacts, "
    "text, watermark, logo, subtitles, caption, label, "
    "deformed, disfigured, bad anatomy, bad hands, extra fingers, "
    "missing fingers, extra limbs, mutated hands"

                )

def _parse_negative_text(t: str) -> List[str]:
    raw = (t or "").replace(";", ",")
    parts: List[str] = []
    for chunk in re.split(r"[,\n\r]+", raw):
        s = (chunk or "").strip()
        if s:
            parts.append(s)
    # de-dupe while preserving order
    out: List[str] = []
    seen = set()
    for p in parts:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


class _PlannerLogManager:
    def __init__(self) -> None:
        self.enabled: bool = False
        self._probe_fp = None
        self._job_fp = None

    def _ts(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")

    def enable(self, on: bool) -> None:
        on = bool(on)
        if on == self.enabled:
            return
        self.enabled = on
        try:
            if not on:
                try:
                    if self._probe_fp:
                        self._probe_fp.flush()
                        self._probe_fp.close()
                except Exception:
                    pass
                try:
                    if self._job_fp:
                        self._job_fp.flush()
                        self._job_fp.close()
                except Exception:
                    pass
                self._probe_fp = None
                self._job_fp = None
                return

            logs_dir = _root() / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            self._probe_fp = open(str(logs_dir / "probe.log"), "a", encoding="utf-8", buffering=1)
            self._job_fp = open(str(logs_dir / "job.log"), "a", encoding="utf-8", buffering=1)
            self._probe_fp.write(f"\n--- probe logging enabled @ {self._ts()} ---\n")
            self._job_fp.write(f"\n--- job logging enabled @ {self._ts()} ---\n")
        except Exception:
            # If logging fails for any reason, disable silently.
            self.enabled = False
            self._probe_fp = None
            self._job_fp = None

    def log_probe(self, msg: str) -> None:
        if not self.enabled or not self._probe_fp:
            return
        try:
            self._probe_fp.write(f"[{self._ts()}] {msg}\n")
        except Exception:
            pass

    def log_job(self, job_id: str, msg: str) -> None:
        if not self.enabled or not self._job_fp:
            return
        try:
            jid = (job_id or "").strip() or "-"
            self._job_fp.write(f"[{self._ts()}] [{jid}] {msg}\n")
        except Exception:
            pass

# Load persisted toggle so logging can start immediately at app launch when desired.
_PLANNER_SETTINGS = _load_planner_settings()
_LOGGER = _PlannerLogManager()
try:
    _LOGGER.enable(bool(_PLANNER_SETTINGS.get("probe_logs", False)))
except Exception:
    pass

# -----------------------------
# Qwen3-VL text generation helper
# -----------------------------
# IMPORTANT:
# We intentionally DO NOT import helpers.prompt at module import time.
# Importing Transformers/PyTorch stacks can keep CUDA contexts and allocator caches
# resident for the entire Planner process. Instead we run Qwen3-VL in a short-lived
# subprocess. When the subprocess exits, VRAM is genuinely released.

# Ensure project root and helpers are importable even when this file is run standalone.
for _p in (str(_root()), str((_root() / "helpers"))):
    try:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    except Exception:
        pass

_HAVE_QWEN_TEXT = True
_QWEN_IMPORT_ERROR: Optional[Exception] = None
def _qwen_generate_text(*, model_path: Path, system_prompt: str, user_prompt: str,
                       temperature: float, max_new_tokens: int, cancel_check=None) -> str:
    """Legacy-compatible wrapper. Runs Qwen in a short-lived subprocess."""
    out, err, rc = _run_qwen_text_subprocess(
        model_path=str(model_path),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=float(temperature),
        max_new_tokens=int(max_new_tokens),
    )
    if int(rc) != 0 and not (out or "").strip():
        raise RuntimeError(f"Qwen subprocess failed (rc={rc}).\\n{err}")
    return (out or "")

def _run_qwen_text_subprocess(*,
                             model_path: str,
                             system_prompt: str,
                             user_prompt: str,
                             temperature: float,
                             max_new_tokens: int) -> Tuple[str, str, int]:
    """Run Qwen text generation in a short-lived subprocess and return (stdout, stderr, returncode)."""
    req = {
        "model_path": str(model_path),
        "system_prompt": system_prompt or "",
        "user_prompt": user_prompt or "",
        "temperature": float(temperature),
        "max_new_tokens": int(max_new_tokens),
    }
    tmp_dir = Path(tempfile.gettempdir()) / "framevision_planner_qwen"
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    req_path = tmp_dir / f"qwen_req_{uuid.uuid4().hex}.json"

    _safe_write_text(str(req_path), json.dumps(req, ensure_ascii=False, indent=2))

    # The runner reads JSON and prints ONLY the model output to stdout.
    runner = r'''
import json, sys, traceback
from pathlib import Path
try:
    req_path = Path(sys.argv[1])
    req = json.loads(req_path.read_text(encoding="utf-8"))
    model_path = Path(req["model_path"])
    # Import inside subprocess only
    try:
        from helpers.prompt import _generate_with_qwen_text as gen
    except Exception:
        from prompt import _generate_with_qwen_text as gen

    out = gen(
        model_path=model_path,
        system_prompt=req.get("system_prompt",""),
        user_prompt=req.get("user_prompt",""),
        temperature=float(req.get("temperature",0.3)),
        max_new_tokens=int(req.get("max_new_tokens",1024)),
        cancel_check=None,
    )
    sys.stdout.write((out or "").strip())
except Exception:
    traceback.print_exc()
    sys.exit(2)
'''
    cmd = [sys.executable, "-c", runner, str(req_path)]
    # Make sure subprocess can import project modules
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([str(_root()), str(_root() / "helpers"), env.get("PYTHONPATH","")])

    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        out = (p.stdout or "").strip()
        err = (p.stderr or "").strip()
        rc = int(p.returncode)
        return out, err, rc
    finally:
        try:
            req_path.unlink(missing_ok=True)  # type: ignore
        except Exception:
            pass



# -----------------------------
# Video generation presets 
# -----------------------------
# These represent "generation targets" (proxy). Final export resolution/quality is handled later (upscale/encode).
_HUNYUAN_PRESETS = {
    # HunyuanVideo 1.5 proxy targets (used by Planner automation)
    # Tier mapping is driven by UI 'Generation quality' (Low/Medium/High).
    # All tiers: 20 fps.
    "low": {
        "model": "hunyuan",
        "quality": "low",
        "res": "304p",
        "target_size": 420,   # aligns to Hunyuan 1.5 auto-bucket
        "fps": 20,
        "steps": 10,
        "min_sec": 3.0,
        "max_sec": 5.0,
        "max_frames": 121,
        "model_key": "480p_i2v_step_distilled",
        "attn_backend": "auto",
        "cpu_offload": True,
        "vae_tiling": True,
    },
    "medium": {
        "model": "hunyuan",
        "quality": "medium",
        "res": "368p",
        "target_size": 480,
        "fps": 15,
        "steps": 10,
        "min_sec": 2.5,
        "max_sec": 4.0,
        "max_frames": 97,
        "model_key": "480p_i2v_step_distilled",
        "attn_backend": "auto",
        "cpu_offload": True,
        "vae_tiling": True,
    },
    "high": {
        "model": "hunyuan",
        "quality": "high",
        "res": "432p",
        "target_size": 576,
        "fps": 20,
        "steps": 10,
        "min_sec": 2.0,
        "max_sec": 3.0,
        "max_frames": 77,
        "model_key": "480p_i2v_step_distilled",
        "attn_backend": "auto",
        "cpu_offload": True,
        "vae_tiling": True,
    },
}

_WAN22_PRESETS = {
    # Wan 2.2 presets target the official generate.py CLI (ti2v-5B).
    # NOTE: size values must match what your Wan2.2 generate.py expects (we use the same strings as helpers/wan22.py).
    "low":    {"model": "wan22", "quality": "low",    "res": "544p", "size_landscape": "1280*544", "size_portrait": "544*1280", "fps": 15, "steps": 20, "guidance": 4, "min_sec": 3.0, "max_sec": 5.0, "offload_model": True,
               "note": "Lighter; good default for long runs"},
    "medium": {"model": "wan22", "quality": "medium", "res": "704p", "size_landscape": "1280*704", "size_portrait": "704*1280", "fps": 15, "steps": 25, "guidance": 4, "min_sec": 2.5, "max_sec": 4.0, "offload_model": True,
               "note": "Balanced"},
    "high":   {"model": "wan22", "quality": "high",   "res": "704p", "size_landscape": "1280*704", "size_portrait": "704*1280", "fps": 24, "steps": 30, "guidance": 4, "min_sec": 2.0, "max_sec": 3.5, "offload_model": True,
               "note": "Best native motion, heavier"},
}


def _normalize_key(s: str) -> str:
    s0 = (s or "").strip().lower()
    # UI labels often include "(default)"; normalize to the base key
    if "(" in s0:
        s0 = s0.split("(", 1)[0].strip()
    # keep only first word for quality labels like "low", "medium", "high", "normal"
    if " " in s0:
        s0 = s0.split(" ", 1)[0].strip()
    return s0.replace(" ", "_")


def _seed_to_int(seed: str) -> int:
    """Deterministically convert any seed string into a stable 32-bit positive int."""
    try:
        s = (seed or "").encode("utf-8", errors="ignore")
        h = hashlib.sha256(s).digest()
        # Use 4 bytes; mask to signed-safe positive range
        return int.from_bytes(h[:4], "little", signed=False) & 0x7FFFFFFF
    except Exception:
        try:
            return abs(hash(seed)) & 0x7FFFFFFF
        except Exception:
            return 0

def _parse_vnum(name: str) -> Tuple[int, int, int]:
    """Parse versions like V3.0, v2, V10.1.2 from a filename."""
    n = name or ""
    m = re.search(r"[\-_\s]v(\d+)(?:\.(\d+))?(?:\.(\d+))?", n, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"\bV(\d+)(?:\.(\d+))?(?:\.(\d+))?\b", n, flags=re.IGNORECASE)
    if not m:
        return (0, 0, 0)
    a = int(m.group(1) or 0)
    b = int(m.group(2) or 0)
    c = int(m.group(3) or 0)
    return (a, b, c)


def _extract_version_tuple(name: str) -> Tuple[int, int, int]:
    """Compatibility alias: some parts of the planner expect this helper.
    Parses version tags like V3.0 / v2 / V10.1.2 from filenames."""
    return _parse_vnum(name)

def _find_newest_qwen2512_turbo_lora() -> Optional[str]:
    """Find newest Wuli Qwen-Image-2512 Turbo LoRA under <root>/models/loras/** by highest V number."""
    base = _root() / "models" / "loras"
    if not base.exists():
        return None
    target = "wuli-qwen-image-2512-turbo-lora"
    best = None
    best_v = (0, 0, 0)
    # prefer files over folders if both exist; but accept either (some loras are folders)
    for dirpath, dirnames, filenames in os.walk(str(base)):
        for name in list(filenames) + list(dirnames):
            if target not in name.lower():
                continue
            v = _parse_vnum(name)
            if v > best_v:
                best_v = v
                best = os.path.join(dirpath, name)
            elif v == best_v and best is not None:
                # tie-break: prefer '4steps', then bf16, then shorter path
                cur = name.lower()
                old = os.path.basename(best).lower()
                score = (("4steps" in cur), ("bf16" in cur), -len(os.path.join(dirpath, name)))
                score_old = (("4steps" in old), ("bf16" in old), -len(best))
                if score > score_old:
                    best = os.path.join(dirpath, name)
    return best




def _find_newest_qwen2511_multi_angle_lora() -> Optional[str]:
    """Find newest multi-angle LoRA for Qwen Edit 2511 under <root>/models/loras/** (best-effort)."""
    base = _root() / "models" / "loras"
    if not base.exists():
        return None

    wanted_any = ("multi-angle", "multiangle", "multi_angle", "96", "angles")
    preferred = ("2511", "qwen", "edit")

    best = None
    best_v = (0, 0, 0)
    best_score = -1
    best_mtime = 0.0

    for dirpath, _, filenames in os.walk(str(base)):
        for name in filenames:
            n = name.lower()
            if not (n.endswith(".safetensors") or n.endswith(".pt") or n.endswith(".bin")):
                continue
            if not any(w in n for w in wanted_any):
                continue

            v = _extract_version_tuple(name)
            score_pref = sum(1 for p in preferred if p in n)
            try:
                mtime = os.path.getmtime(os.path.join(dirpath, name))
            except Exception:
                mtime = 0.0

            if best is None:
                best = os.path.join(dirpath, name)
                best_v = v
                best_score = score_pref
                best_mtime = mtime
                continue

            if v > best_v:
                best = os.path.join(dirpath, name)
                best_v = v
                best_score = score_pref
                best_mtime = mtime
            elif v == best_v:
                if score_pref > best_score:
                    best = os.path.join(dirpath, name)
                    best_score = score_pref
                    best_mtime = mtime
                elif score_pref == best_score and mtime > best_mtime:
                    best = os.path.join(dirpath, name)
                    best_mtime = mtime

    return best



def _load_shots_list(shots_json_path: str) -> List[Dict[str, Any]]:
    """Read shots.json supporting both top-level list and legacy {"shots": [...]} wrapper."""
    try:
        obj = _safe_read_json(shots_json_path)
    except Exception:
        obj = None
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        v = obj.get("shots")
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
    return []


# -----------------------------
# Character Bible + LOCK blocks
# -----------------------------

def _plan_characters(plan_obj: Any) -> List[Dict[str, Any]]:
    """Extract a best-effort character list from plan.json."""
    out: List[Dict[str, Any]] = []
    if isinstance(plan_obj, dict):
        chars = plan_obj.get("characters")
        if isinstance(chars, list):
            for c in chars:
                if isinstance(c, dict):
                    name = str(c.get("name") or c.get("id") or "").strip()
                    role = str(c.get("role") or c.get("type") or "").strip()
                    if name:
                        out.append({"name": name, "role": role})
                elif isinstance(c, str) and c.strip():
                    out.append({"name": c.strip(), "role": ""})
    return out

def _normalize_no_item(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    low = s.lower().strip()
    for prefix in ("no ", "no:", "avoid ", "don't ", "do not "):
        if low.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    # strip trailing punctuation
    return s.strip().strip(".,;:!").strip()


# Aggressive visual description sanitizer to prevent metadata bleeding
_VISUAL_CONTAMINATION_PATTERNS = [
    # Technical labels that pollute images
    (r'\b(?:shot|camera|angle|frame|lighting|mood|scene|take|cut|fade)\s*[:;]\s*', '', re.IGNORECASE),
    (r'\b(?:wide|close|medium|establishing)\s*(?:shot|angle|view)\s*[:;]?\s*', '', re.IGNORECASE),
    (r'\b(?:dolly|zoom|pan|tilt|track|crane)\s*(?:in|out|left|right|up|down)?\s*[:;]?\s*', '', re.IGNORECASE),
    (r'\b(?:emo|drama|tension|energy|atmospheric)\s*[:;]\s*', '', re.IGNORECASE),
    # Cinematic terms that get literalized
    (r'\bshallow\s+depth\s+of\s+field\b', 'soft background', re.IGNORECASE),
    (r'\bdepth\s+of\s+field\b', '', re.IGNORECASE),
    (r'\b(?:rule\s+of\s+thirds|golden\s+ratio)\b', '', re.IGNORECASE),
    # JSON artifacts
    (r'["\']?visual_description["\']?\s*[:=]\s*["\']?', '', re.IGNORECASE),
    (r'["\']?stage_directions["\']?\s*[:=]\s*', '', re.IGNORECASE),
    # UI elements
    (r'\b(?:hud|ui|overlay|interface|holographic\s+display)\s*\w*', '', re.IGNORECASE),
]

def _sanitize_visual_description(text: str) -> str:
    """Remove any metadata contamination from visual description."""
    if not text:
        return ""

    # Strip code blocks if present
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`[^`]+`', '', text)

    # Remove label pollution
    for pattern, repl, flags in _VISUAL_CONTAMINATION_PATTERNS:
        text = re.sub(pattern, repl, text, flags=flags)

    # Remove standalone technical terms at start
    text = re.sub(r'^\s*(?:wide|close|medium|establishing|high|low)\s*[,.]?\s*', '', text, flags=re.IGNORECASE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Remove empty parentheses left by removals
    text = re.sub(r'\(\s*\)', '', text)
    text = re.sub(r'\[\s*\]', '', text)

    # Last resort: if still contaminated, extract sentences that don't contain bad words
    bad_words = ['shot', 'camera', 'lighting', 'cut to', 'fade in', 'fade out', 'angle:', 'lens:']
    if any(bad in text.lower() for bad in bad_words):
        sentences = re.split(r'[.!?]+', text)
        clean_sentences = []
        for sent in sentences:
            if sent.strip() and not any(bad in sent.lower() for bad in bad_words):
                clean_sentences.append(sent.strip())
        if clean_sentences:
            text = '. '.join(clean_sentences) + '.'

    return text.strip()

def _detect_character_taxonomy(desc: str) -> str:
    """Detect if character is human, animal, or creature/fantasy."""
    desc_lower = desc.lower()
    
    animal_terms = ['dog', 'cat', 'wolf', 'lion', 'bear', 'fox', 'rabbit', 'bird', 
                   'eagle', 'dragon', 'creature', 'monster', 'beast', 'animal',
                   'furry', 'paws', 'claws', 'fur', 'feathers', 'scales', 'tail',
                   'tiger', 'leopard', 'panda', 'raccoon', 'otter', 'deer']
    
    human_terms = ['person', 'man', 'woman', 'human', 'character', 'girl', 'boy',
                  'adult', 'child', 'wearing', 'outfit', 'clothing', 'suit', 'dress']
    
    animal_score = sum(1 for term in animal_terms if term in desc_lower)
    human_score = sum(1 for term in human_terms if term in desc_lower)
    
    if animal_score > human_score:
        return "animal"
    elif "hybrid" in desc_lower or "anthro" in desc_lower or "creature" in desc_lower:
        return "creature"
    return "human"

def _ensure_character_bible(manifest: Dict[str, Any], plan_obj: Any) -> List[Dict[str, Any]]:
    """Ensure manifest['project']['character_bible'] exists and is taxonomy-aware."""
    proj = manifest.setdefault("project", {})
    bible = proj.get("character_bible")
    if isinstance(bible, list):
        bible_list = [x for x in bible if isinstance(x, dict)]
    else:
        bible_list = []

    if not bible_list:
        # Create characters based on plan characters with taxonomy detection
        for c in _plan_characters(plan_obj):
            name = str(c.get("name") or "").strip() or "Character"
            role = str(c.get("role") or "").strip()
            desc = f"{role} {c.get('description', '')}"
            taxonomy = _detect_character_taxonomy(desc)
            
            bible_list.append({
                "name": name,
                "role": role,
                "taxonomy": taxonomy,
                # Human fields
                "face_traits": [],
                "hair": "",
                "outfit": "",
                # Animal/Creature fields
                "species": "",
                "anatomy": {},
                "coat": {},
                "facial": {},
                "limbs": {},
                "posture_movement": [],
                # Common
                "palette": [],
                "vibe": [],
                "do_not_change": [],
                "ref_images": [],  
            })

    # Normalize keys/types based on taxonomy
    norm: List[Dict[str, Any]] = []
    for c in bible_list:
        if not isinstance(c, dict):
            continue
        
        taxonomy = c.get("taxonomy", "human")
        
        if taxonomy == "animal":
            norm.append({
                "name": str(c.get("name") or "").strip() or "Character",
                "role": str(c.get("role") or "").strip(),
                "taxonomy": "animal",
                "species": str(c.get("species") or "").strip(),
                "anatomy": {
                    "body_type": str(c.get("anatomy", {}).get("body_type") or "").strip(),
                    "size_relative": str(c.get("anatomy", {}).get("size_relative") or "").strip(),
                    "distinctive_features": [str(x).strip() for x in (c.get("anatomy", {}).get("distinctive_features") or []) if str(x).strip()],
                },
                "coat": {
                    "type": str(c.get("coat", {}).get("type") or "fur").strip(),
                    "color_primary": str(c.get("coat", {}).get("color_primary") or "").strip(),
                    "color_secondary": str(c.get("coat", {}).get("color_secondary") or "").strip(),
                    "texture": str(c.get("coat", {}).get("texture") or "").strip(),
                    "patterns": [str(x).strip() for x in (c.get("coat", {}).get("patterns") or []) if str(x).strip()],
                },
                "facial": {
                    "snout_shape": str(c.get("facial", {}).get("snout_shape") or "").strip(),
                    "ears": str(c.get("facial", {}).get("ears") or "").strip(),
                    "eyes": str(c.get("facial", {}).get("eyes") or "").strip(),
                    "distinctive": [str(x).strip() for x in (c.get("facial", {}).get("distinctive") or []) if str(x).strip()],
                },
                "limbs": {
                    "paws_hooves": str(c.get("limbs", {}).get("paws_hooves") or "").strip(),
                    "tail": str(c.get("limbs", {}).get("tail") or "").strip(),
                    "wings": str(c.get("limbs", {}).get("wings") or "").strip(),
                },
                "posture_movement": [str(x).strip() for x in (c.get("posture_movement") or []) if str(x).strip()][:3],
                "palette": [str(x).strip() for x in (c.get("palette") or []) if str(x).strip()],
                "vibe": [str(x).strip() for x in (c.get("vibe") or []) if str(x).strip()],
                "do_not_change": [str(x).strip() for x in (c.get("do_not_change") or []) if str(x).strip()],
                "ref_images": list(c.get("ref_images") or []),
            })
        elif taxonomy == "creature":
            norm.append({
                "name": str(c.get("name") or "").strip() or "Character",
                "role": str(c.get("role") or "").strip(),
                "taxonomy": "creature",
                "base_anatomy": str(c.get("base_anatomy") or "").strip(),
                "skin_coat": {
                    "covering": str(c.get("skin_coat", {}).get("covering") or "").strip(),
                    "colors": [str(x).strip() for x in (c.get("skin_coat", {}).get("colors") or []) if str(x).strip()],
                    "texture": str(c.get("skin_coat", {}).get("texture") or "").strip(),
                },
                "distinctive_features": [str(x).strip() for x in (c.get("distinctive_features") or []) if str(x).strip()],
                "face_head": [str(x).strip() for x in (c.get("face_head") or []) if str(x).strip()],
                "palette": [str(x).strip() for x in (c.get("palette") or []) if str(x).strip()],
                "do_not_change": [str(x).strip() for x in (c.get("do_not_change") or []) if str(x).strip()],
                "ref_images": list(c.get("ref_images") or []),
            })
        else:  # human
            norm.append({
                "name": str(c.get("name") or "").strip() or "Character",
                "role": str(c.get("role") or "").strip(),
                "taxonomy": "human",
                "face_traits": [str(x).strip() for x in (c.get("face_traits") or []) if str(x).strip()],
                "hair": str(c.get("hair") or "").strip(),
                "outfit": str(c.get("outfit") or "").strip(),
                "palette": [str(x).strip() for x in (c.get("palette") or []) if str(x).strip()],
                "vibe": [str(x).strip() for x in (c.get("vibe") or []) if str(x).strip()],
                "do_not_change": [str(x).strip() for x in (c.get("do_not_change") or []) if str(x).strip()],
                "ref_images": list(c.get("ref_images") or []),
            })
    
    proj["character_bible"] = norm
    return norm
    proj = manifest.setdefault("project", {})
    bible = proj.get("character_bible")
    if isinstance(bible, list):
        bible_list = [x for x in bible if isinstance(x, dict)]
    else:
        bible_list = []

    if not bible_list:
        # Create characters based on plan characters (editable later).
        for c in _plan_characters(plan_obj):
            bible_list.append({
                "name": str(c.get("name") or "").strip() or "Character",
                "role": str(c.get("role") or "").strip(),
                "face_traits": [],
                "hair": "",
                "outfit": "",
                "palette": [],
                "vibe": [],
                "do_not_change": [],
                "ref_images": [],  
            })

    # Normalize keys/types
    norm: List[Dict[str, Any]] = []
    for c in bible_list:
        if not isinstance(c, dict):
            continue
        norm.append({
            "name": str(c.get("name") or "").strip() or "Character",
            "role": str(c.get("role") or "").strip(),
            "face_traits": [str(x).strip() for x in (c.get("face_traits") or []) if str(x).strip()],
            "hair": str(c.get("hair") or "").strip(),
            "outfit": str(c.get("outfit") or "").strip(),
            "palette": [str(x).strip() for x in (c.get("palette") or []) if str(x).strip()],
            "vibe": [str(x).strip() for x in (c.get("vibe") or []) if str(x).strip()],
            "do_not_change": [str(x).strip() for x in (c.get("do_not_change") or []) if str(x).strip()],
            "ref_images": list(c.get("ref_images") or []),
        })
    proj["character_bible"] = norm
    return norm

def _pick_relevant_characters(bible: List[Dict[str, Any]], shot_text: str) -> List[Dict[str, Any]]:
    """Pick characters whose *names* explicitly appear in shot_text.

    - If no names match, return [] (no fallback to bible[0]).
    - Matching is word-boundary-ish to avoid substring collisions.
    """
    if not bible:
        return []

    st = (shot_text or "").lower()

    # Safety: never match ultra-common tokens as "names"
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with", "from", "by", "as", "is", "it"
    }

    picked: List[Dict[str, Any]] = []
    for c in bible:
        nm = str(c.get("name") or "").strip()
        if not nm:
            continue
        nml = nm.lower()
        if nml in stop:
            continue

        # Word-boundary-ish match for full name (supports multi-word names)
        try:
            pat = r"(?<![A-Za-z0-9_])" + re.escape(nml) + r"(?![A-Za-z0-9_])"
            if re.search(pat, st):
                picked.append(c)
        except Exception:
            if nml in st.split():
                picked.append(c)

    # cap to avoid huge prompts
    return picked[:2]

# --- AUTO Character Bible v2 (ID binding + identity anchors) -----------------

def _shot_has_people_hint(text: str) -> bool:
    t = (text or "").lower()
    # Conservative heuristic: if these appear, we assume humans are present
    kws = [
        " man", " woman", " people", " person", " couple", " boy", " girl", " child", " kid",
        " waiter", " waitress", " detective", " bartender", " customer", " crowd",
        " they ", " he ", " she ", " him ", " her ",
    ]
    for k in kws:
        if k.strip() in ("he", "she", "him", "her", "they"):
            # word boundary-ish
            if re.search(rf"\b{k.strip()}\b", t):
                return True
        else:
            if k in (" " + t + " "):
                return True
    return False

def _auto_cb_v2_enabled(character_bible_enabled: bool, own_character_bible_enabled: bool, own_storyline_enabled: bool) -> bool:
    # Hard gate per spec
    return bool(character_bible_enabled) and (not bool(own_character_bible_enabled)) and (not bool(own_storyline_enabled))

def _auto_cb_v2_char_map(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    try:
        proj = manifest.get("project") if isinstance(manifest.get("project"), dict) else {}
        acb = proj.get("auto_character_bible_v2") if isinstance(proj.get("auto_character_bible_v2"), dict) else {}
        chars = acb.get("characters") if isinstance(acb.get("characters"), list) else []
    except Exception:
        chars = []
    out: Dict[str, Dict[str, Any]] = {}
    for c in chars:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if cid:
            out[cid] = c
    return out

def _auto_cb_v2_bindings(manifest: Dict[str, Any]) -> Dict[str, List[str]]:
    try:
        proj = manifest.get("project") if isinstance(manifest.get("project"), dict) else {}
        acb = proj.get("auto_character_bible_v2") if isinstance(proj.get("auto_character_bible_v2"), dict) else {}
        binds = acb.get("shot_bindings") if isinstance(acb.get("shot_bindings"), dict) else {}
    except Exception:
        binds = {}
    out: Dict[str, List[str]] = {}
    for sid, ids in (binds or {}).items():
        if not sid:
            continue
        if isinstance(ids, list):
            out[str(sid)] = [str(x).strip() for x in ids if str(x).strip()]
    return out

def _auto_cb_v2_append_anchors(prompt: str, present_ids: List[str], char_map: Dict[str, Dict[str, Any]], engine_hint: str = "") -> str:
    """
    AUTO Character Bible v2 (AUTO mode only), prompt balancer:
    - Avoid name brittleness by removing display names.
    - Keep scene/action intact by keeping identity anchors compact.
    - Reduce cloning risk on some engines (e.g. Z-Image) by avoiding repeated subject restatement.
    """
    try:
        import re
    except Exception:
        re = None  # type: ignore

    p = str(prompt or "").strip()
    if not p:
        return p

    # Helper: role label derived from role_tags when available.
    def _role_label(c: Dict[str, Any]) -> str:
        tags = c.get("role_tags")
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(t).strip().lower() for t in (tags or []) if str(t).strip()]
        if any(t in ("woman", "female", "girl", "she", "her") for t in tags):
            return "a woman"
        if any(t in ("man", "male", "boy", "he", "his") for t in tags):
            return "a man"
        # Generic fallback
        return "a person"

    # Helper: aggressively compress an identity anchor into a short, high-signal fragment.
    def _compress_anchor(anchor: str, max_parts: int = 4, max_chars: int = 120) -> str:
        a = str(anchor or "").strip()
        if not a:
            return ""
        # Split by commas; keep early items which tend to be the strongest identity cues.
        parts = [x.strip() for x in a.split(",") if x.strip()]
        cleaned = []
        for x in parts:
            xl = x.lower().strip()
            # Drop low-value/scene-breaking tokens.
            if "build" in xl:
                continue
            if xl in ("male", "female", "man", "woman"):
                continue
            if xl.startswith("no distinct") or xl.startswith("no visible") or xl.startswith("no marks"):
                continue
            cleaned.append(x.strip())
            if len(cleaned) >= max_parts:
                break
        out = ", ".join(cleaned) if cleaned else a
        # Hard cap by chars (trim at last comma if possible)
        if len(out) > max_chars:
            out = out[:max_chars].rstrip()
            if "," in out:
                out = out.rsplit(",", 1)[0].rstrip()
        return out.strip().strip(",")

    # 1) Resolve present characters into (cid, role, anchor, display_name)
    present_chars: List[Tuple[str, str, str, str]] = []
    for cid in present_ids or []:
        c = char_map.get(str(cid))
        if not isinstance(c, dict):
            continue
        ia = str(c.get("identity_anchor") or "").strip()
        if not ia:
            continue
        role = _role_label(c)
        dn = str(c.get("display_name") or "").strip()
        present_chars.append((str(cid), role, ia, dn))

    if not present_chars:
        return p

    # 2) Remove display names from the prompt (names are brittle and can create new "subjects").
    # Replace possessive forms first.
    try:
        for _cid, role, ia, dn in present_chars:
            if not dn:
                continue
            # e.g., "Jamie's" / "Jamies" / "Jamie"
            p = re.sub(rf"\b{re.escape(dn)}\b\s*'s\b", "the person's", p, flags=re.IGNORECASE)
            p = re.sub(rf"\b{re.escape(dn)}s\b", "the person's", p, flags=re.IGNORECASE)
            p = re.sub(rf"\b{re.escape(dn)}\b", "the person", p, flags=re.IGNORECASE)
    except Exception:
        pass

    # 3) Optional: very short inline identity for the common template "a man and a woman" (keep it tiny).
    # This preserves your desired "inline" feel without turning the entire prompt into face-descriptions.
    inline_done = False
    try:
        if len(present_chars) >= 2:
            # Pick best man/woman ordering when available.
            man = None
            woman = None
            for item in present_chars:
                if item[1] == "a man" and man is None:
                    man = item
                if item[1] == "a woman" and woman is None:
                    woman = item
            if man is None:
                man = present_chars[0]
            if woman is None:
                woman = present_chars[1] if len(present_chars) > 1 else present_chars[0]

            man_short = _compress_anchor(man[2], max_parts=2, max_chars=60)
            woman_short = _compress_anchor(woman[2], max_parts=2, max_chars=60)

            # Replace only the first occurrence to avoid duplicate restatement.
            p2 = re.sub(
                r"\ba\s+man\s+and\s+a\s+woman\b",
                f"a man, {man_short} and a woman, {woman_short}",
                p,
                count=1,
                flags=re.IGNORECASE,
            )
            if p2 != p:
                p = p2
                inline_done = True
    except Exception:
        inline_done = False

    # 4) Build a compact identity block appended once (keeps scene/action intact).
    # Use a stable "C1/C2" label and a compact anchor form.
    try:
        block_parts = []
        for _cid, role, ia, dn in present_chars:
            ia_compact = _compress_anchor(ia, max_parts=4, max_chars=140)
            if ia_compact:
                block_parts.append(f"{_cid}: {ia_compact}")
        if block_parts:
            # Append as a single sentence to avoid prompt-length/format issues.
            p = (p.rstrip().rstrip(".") + ". Identity anchors (keep consistent): " + "; ".join(block_parts) + ".").strip()
    except Exception:
        pass

    # 5) Anti-clone guard for engines that tend to duplicate subjects when the prompt is verbose.
    try:
        eh = str(engine_hint or "").lower()
        is_z = ("z-image" in eh) or ("zimage" in eh) or ("z_image" in eh)
        if is_z and len(present_chars) >= 2:
            p = (p + " Exactly two people: one man and one woman. No duplicates, no extra faces.").strip()
    except Exception:
        pass

    # Cleanup: collapse stray whitespace/newlines
    try:
        p = re.sub(r"[ \t]+", " ", p)
        p = re.sub(r"\s*\n\s*", " ", p).strip()
    except Exception:
        pass
    return p

def _auto_cb_v2_validate(payload: Any, shots: List[Dict[str, Any]]) -> Tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "Payload is not a JSON object."
    chars = payload.get("characters")
    if not isinstance(chars, list) or not chars:
        return False, "Missing or empty characters[]."
    ids = []
    for c in chars:
        if not isinstance(c, dict):
            return False, "characters[] must contain objects."
        cid = str(c.get("id") or "").strip()
        ia = str(c.get("identity_anchor") or "").strip()
        if not cid:
            return False, "A character is missing id."
        if not re.match(r"^C\d+$", cid):
            return False, f"Character id must look like C1/C2/... (got {cid})."
        if len(ia) < 20:
            return False, f"identity_anchor too short for {cid}."
        # Clothing ban (best-effort)
        if re.search(r"\balways wears\b|\bwears\b|\boutfit\b|\bdress\b|\bjeans\b|\bjacket\b", ia.lower()):
            return False, f"identity_anchor for {cid} appears to include clothing; must be clothing-free."
        ids.append(cid)
    if len(set(ids)) != len(ids):
        return False, "Duplicate character ids."
    # Validate shot bindings
    s_map = {str(s.get("id") or "").strip(): s for s in shots if isinstance(s, dict)}
    binds = payload.get("shots") or payload.get("shot_bindings")
    # We accept either: shots[] with id+present_characters OR shot_bindings{} mapping.
    if isinstance(binds, list):
        for sh in binds:
            if not isinstance(sh, dict):
                return False, "shots[] must contain objects."
            sid = str(sh.get("id") or "").strip()
            pcs = sh.get("present_characters")
            if not sid or sid not in s_map:
                return False, f"shots[] contains unknown or missing shot id: {sid}"
            if not isinstance(pcs, list):
                return False, f"present_characters missing for shot {sid}."
            pcs2 = [str(x).strip() for x in pcs if str(x).strip()]
            if _shot_has_people_hint(str(s_map[sid].get('visual_description') or s_map[sid].get('seed') or '')) and not pcs2:
                return False, f"Shot {sid} appears to include people but present_characters is empty."
            for cid in pcs2:
                if cid not in ids:
                    return False, f"Shot {sid} references unknown character id {cid}."
    elif isinstance(binds, dict):
        for sid, pcs in binds.items():
            sid2 = str(sid or "").strip()
            if not sid2:
                continue
            if sid2 not in s_map:
                return False, f"shot_bindings references unknown shot id {sid2}."
            pcs2 = [str(x).strip() for x in (pcs or []) if str(x).strip()] if isinstance(pcs, list) else []
            if _shot_has_people_hint(str(s_map[sid2].get('visual_description') or s_map[sid2].get('seed') or '')) and not pcs2:
                return False, f"Shot {sid2} appears to include people but present_characters is empty."
            for cid in pcs2:
                if cid not in ids:
                    return False, f"Shot {sid2} references unknown character id {cid}."
    else:
        return False, "Missing shots bindings (shots[] or shot_bindings{})."
    return True, ""

def _auto_cb_v2_normalize(payload: Dict[str, Any], shots: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize Qwen output into {characters:[], shot_bindings:{sid:[C..]}}"""
    out: Dict[str, Any] = {"characters": [], "shot_bindings": {}}
    chars = payload.get("characters") if isinstance(payload.get("characters"), list) else []
    out["characters"] = [c for c in chars if isinstance(c, dict)]
    # accept either shots[] or shot_bindings{}
    if isinstance(payload.get("shot_bindings"), dict):
        sb = payload.get("shot_bindings") or {}
        for sid, pcs in sb.items():
            if isinstance(pcs, list):
                out["shot_bindings"][str(sid)] = [str(x).strip() for x in pcs if str(x).strip()]
    elif isinstance(payload.get("shots"), list):
        for sh in payload.get("shots") or []:
            if not isinstance(sh, dict):
                continue
            sid = str(sh.get("id") or "").strip()
            pcs = sh.get("present_characters")
            if sid and isinstance(pcs, list):
                out["shot_bindings"][sid] = [str(x).strip() for x in pcs if str(x).strip()]
    # Ensure every shot id exists as key (empty list ok)
    for s in shots:
        if isinstance(s, dict):
            sid = str(s.get("id") or "").strip()
            if sid and sid not in out["shot_bindings"]:
                out["shot_bindings"][sid] = []
    return out



def _identity_lock_block(chars: List[Dict[str, Any]]) -> str:
    """Generate consistency lock appropriate for character taxonomy."""
    if not chars:
        return ""
    
    lines: List[str] = []
    lines.append("Identity consistency: Keep the character identity consistent with the reference details below. Single photo, one view.")

    has_human = any(str(c.get("taxonomy") or "human").strip().lower() == "human" for c in chars)

    for c in chars:
        name = (c.get("name") or "").strip() or "Character"
        taxonomy = c.get("taxonomy", "human")
        
        if taxonomy == "animal":
            species = (c.get("species") or "").strip() or "Animal"
            parts = [
                f"{name} ({species}).",
            ]
            
            anatomy = c.get("anatomy", {})
            if anatomy.get("body_type") and anatomy.get("size_relative"):
                parts.append(f"Body: {anatomy['body_type']}, {anatomy['size_relative']}.")
            
            coat = c.get("coat", {})
            if coat.get("color_primary"):
                color_info = coat['color_primary']
                if coat.get("patterns"):
                    color_info += f" with {', '.join(coat['patterns'][:2])}"
                parts.append(f"Coat: {color_info}.")
            
            if coat.get("texture") and coat.get("type"):
                parts.append(f"Texture: {coat['texture']} {coat['type']}.")
            
            facial = c.get("facial", {})
            if facial.get("distinctive"):
                parts.append(f"Markings: {', '.join(facial['distinctive'][:2])}.")
            if facial.get("eyes"):
                parts.append(f"Eyes: {facial['eyes']}.")
            
            limbs = c.get("limbs", {})
            if limbs.get("tail"):
                parts.append(f"Tail: {limbs['tail']}.")
            
            if c.get("posture_movement"):
                parts.append(f"Movement: {', '.join(c['posture_movement'][:2])}.")
                
            lines.append(" ".join(parts))
            
        elif taxonomy == "creature":
            parts = [
                f"{name} (creature).",
            ]
            if c.get("base_anatomy"):
                parts.append(f"Form: {c['base_anatomy']}.")
            
            skin = c.get("skin_coat", {})
            if skin.get("covering") and skin.get("colors"):
                parts.append(f"Covering: {skin['covering']} in {', '.join(skin['colors'][:2])}.")
            
            if c.get("distinctive_features"):
                parts.append(f"Features: {', '.join(c['distinctive_features'][:3])}.")
                
            lines.append(" ".join(parts))
            
        else:  # human
            role = (c.get("role") or "").strip()
            tag = f"{name}" + (f" ({role})" if role else "")
            face = ", ".join([x for x in (c.get("face_traits") or []) if x]) or "(face traits pending)"
            hair = (c.get("hair") or "").strip() or "(hair pending)"
            outfit = (c.get("outfit") or "").strip() or "(outfit pending)"
            vibe = ", ".join([x for x in (c.get("vibe") or []) if x]) or "(vibe pending)"
            
            lines.append(f"{tag}. Face: {face}. Hair: {hair}. Vibe: {vibe}.")
    

    if has_human:
        lines.append("Keep face shape, hairstyle, and outfit consistent within this single shot.")
    else:
        lines.append("Keep species anatomy and natural appearance consistent within this single shot. No human clothing, no human hands.")
    return " ".join([x.strip() for x in lines if x.strip()]).strip()

def _camera_language_block(camera: str, lighting: str, mood: str) -> str:
    cam = (camera or "medium shot").strip()
    light = (lighting or "cinematic lighting").strip()
    md = (mood or "neutral").strip()
    return (
        f"Framing: {cam}. Lighting: {light}. Mood: {md}. "
        "Cinematic composition; stable subject proportions; natural perspective; no sudden style shifts. "
        "Single photo, one view, natural perspective."
    ).strip()

def _drift_prevention_negatives(chars: List[Dict[str, Any]]) -> List[str]:
    out = [
        "low quality", "blurry", "out of frame", "jpeg artifacts",
        "watermark", "logo", "text", "words", "letters", "typography", "subtitle", "subtitles", "caption", "label",
        "deformed", "disfigured", "bad anatomy", "bad hands",
        "extra fingers", "missing fingers", "extra limbs", "mutated hands",
        "different person", "different face", "identity drift",
        "different hairstyle", "different outfit", "age change", "gender swap",
        "duplicate", "cloned",
        ]
    for c in chars:
        for item in (c.get("do_not_change") or []):
            tok = _normalize_no_item(str(item))
            if tok:
                out.append(tok)
    # de-dupe preserving order
    seen = set()
    dedup = []
    for x in out:
        k = x.lower().strip()
        if not k or k in seen:
            continue
        seen.add(k)
        dedup.append(x)
    return dedup


# -----------------------------
# Chunk 2: Drama curve + Shot language + Motifs
# -----------------------------

_DRAMA_PHASES = ("Hook", "Build", "Turn", "Resolve")

def _phase_for_index(i: int, n: int) -> str:
    """Label shot index into a simple drama curve."""
    try:
        i = int(i)
        n = int(n)
    except Exception:
        return "Build"
    if n <= 0:
        return "Build"
    # 0-2: Hook, middle: Build, n-2..n-1: Turn/Resolve
    if i <= 2:
        return "Hook"
    if i >= max(0, n - 1):
        return "Resolve"
    if i >= max(0, n - 2):
        return "Turn"
    # otherwise build
    return "Build"

def _normalize_camera_type(camera: str) -> str:
    c = (camera or "").lower()
    if any(k in c for k in ("establish", "wide establishing", "master shot")):
        return "establishing"
    if any(k in c for k in ("wide", "long shot")):
        return "wide"
    if any(k in c for k in ("close", "closeup", "close-up", "macro", "insert")):
        return "close_up"
    if any(k in c for k in ("over the shoulder", "over-the-shoulder", "ots")):
        return "ots"
    if any(k in c for k in ("pov", "point of view", "first person")):
        return "pov"
    if any(k in c for k in ("tracking", "dolly", "handheld", "follow", "push in", "push-in", "pull back", "pull-back")):
        return "moving"
    if any(k in c for k in ("aerial", "drone", "top down", "top-down", "bird")):
        return "aerial"
    # default bucket
    return "medium"

def _camera_phrase_for_type(t: str) -> str:
    t = (t or "").strip()
    if t == "establishing":
        return "establishing wide shot"
    if t == "wide":
        return "wide shot"
    if t == "close_up":
        return "meaningful close-up"
    if t == "ots":
        return "over-the-shoulder shot"
    if t == "pov":
        return "POV shot"
    if t == "moving":
        return "tracking shot"
    if t == "aerial":
        return "aerial shot"
    return "medium shot"

def _choose_alt_camera_type(prev_type: str, seed_text: str, phase: str) -> str:
    """Pick a different camera type deterministically (no repeats)."""
    # A small menu that creates variety without being too random.
    menu = ["medium", "wide", "close_up", "moving", "ots", "pov", "aerial", "establishing"]
    # Phase nudges: Hook likes bold frames; Turn likes close-ups; Resolve likes medium/wide.
    phase = (phase or "Build").strip()
    if phase == "Hook":
        menu = ["wide", "moving", "close_up", "aerial", "medium", "ots", "pov", "establishing"]
    elif phase == "Turn":
        menu = ["close_up", "ots", "pov", "moving", "medium", "wide", "aerial", "establishing"]
    elif phase == "Resolve":
        menu = ["medium", "wide", "moving", "close_up", "establishing", "ots", "pov", "aerial"]

    prev_type = (prev_type or "").strip()
    candidates = [x for x in menu if x != prev_type]
    if not candidates:
        candidates = ["medium"]
    r = random.Random(_sha1_text(seed_text or ""))
    return candidates[int(r.random() * len(candidates)) % len(candidates)]

def _looks_like_location_change(seed_txt: str, notes: str) -> bool:
    blob = f"{seed_txt}\n{notes}".lower()
    # Heuristic triggers
    keys = [
        "new location", "cut to", "arrive", "arrival", "travel", "enter", "exit",
        "outside", "inside", "interior", "exterior", "street", "room", "hallway",
        "beach", "city", "desert", "mountain", "studio", "club", "stage",
    ]
    return any(k in blob for k in keys)

def _enforce_shot_language(shots: List[Dict[str, Any]]) -> None:
    """Mutate shots in-place to follow camera/shot-language rules and add notes."""
    if not shots:
        return
    n = len(shots)
    prev_type = ""
    for idx, sh in enumerate(shots, start=1):
        seed_txt = str(sh.get("seed") or "")
        notes = str(sh.get("notes") or "")
        cam = str(sh.get("camera") or "medium shot")
        phase = str(sh.get("phase") or _phase_for_index(idx, n))

        sh.setdefault("shot_language_notes", [])
        sln = sh.get("shot_language_notes")
        if not isinstance(sln, list):
            sln = []
            sh["shot_language_notes"] = sln

        cur_type = _normalize_camera_type(cam)

        # Rule: no same type twice in a row
        if prev_type and cur_type == prev_type:
            new_type = _choose_alt_camera_type(prev_type, seed_txt, phase)
            new_cam = _camera_phrase_for_type(new_type)
            sh["camera"] = new_cam
            sln.append(f"Adjusted camera to avoid repeat: '{cam}' -> '{new_cam}'.")
            cur_type = new_type

        # Rule: establishing mostly early OR on location change
        if cur_type == "establishing" and idx > max(3, int(n * 0.25)) and not _looks_like_location_change(seed_txt, notes):
            # Swap to medium/wide to keep later shots intentional.
            new_type = _choose_alt_camera_type("establishing", seed_txt, phase)
            if new_type == "establishing":
                new_type = "medium"
            new_cam = _camera_phrase_for_type(new_type)
            sh["camera"] = new_cam
            sln.append(f"Downshifted late establishing shot to '{new_cam}' (no location change detected).")

        # Rule: close-ups must be meaningful
        cam2 = str(sh.get("camera") or "")
        if _normalize_camera_type(cam2) == "close_up":
            blob = f"{seed_txt}\n{notes}".lower()
            meaningful = any(k in blob for k in ("emotion", "reaction", "tear", "smile", "eyes", "hands", "detail", "reveal", "clue", "turn", "shock"))
            if not meaningful:
                sh["notes"] = (notes + " " if notes else "") + "Make the close-up meaningful (emotion or key detail reveal)."
                sln.append("Close-up flagged: added meaning cue to notes.")

        prev_type = _normalize_camera_type(str(sh.get("camera") or ""))

def _ensure_project_motifs(manifest: Dict[str, Any], plan_obj: Any, stable_key: str) -> Dict[str, Any]:
    """Create or load simple project motifs (no color palette feature).

    Stored under manifest['project'] as:
      - motifs: [..] (up to 2 strings)

    Returns dict: {'motifs': [...]}.
    """
    proj = manifest.setdefault("project", {})
    motifs = proj.get("motifs")
    if isinstance(motifs, list) and all(isinstance(x, str) for x in motifs) and any(str(x).strip() for x in motifs):
        clean = [str(x).strip() for x in motifs if str(x).strip()][:2]
        # Guardrail: avoid reflective/glass motifs unless the user explicitly asked for them.
        # (These motifs can cause unwanted "glass fence / reflections" drift in animal shots.)
        filtered = [m for m in clean if not any(k in m.lower() for k in ("reflect", "reflection", "glass", "mirror", "prop", "token", "charm", "amulet", "talisman", "trinket"))]
        if filtered:
            return {"motifs": filtered[:2]}

    # Stable pick from a small safe "visual motif" menu.
    # Avoid HUD/glitch, accent colors, or bokeh-like effects that cause blobs/overlays.
    motif_menu = [
        "a recurring texture motif (subtle film grain)",
        "a recurring composition motif (leading lines)",
        "a recurring lighting motif (soft rim light)",
    ]

    r = random.Random(_sha1_text(stable_key or "motifs"))
    pick1 = motif_menu[int(r.random() * len(motif_menu)) % len(motif_menu)]
    pick2 = motif_menu[int(r.random() * len(motif_menu)) % len(motif_menu)]
    if pick2 == pick1:
        pick2 = motif_menu[(motif_menu.index(pick1) + 2) % len(motif_menu)]

    motifs = [pick1, pick2][:2]
    proj["motifs"] = motifs
    # Do NOT store accent_color anymore (palette feature removed).
    proj.pop("accent_color", None)
    return {"motifs": motifs}


def _motif_block(motifs_info: Dict[str, Any]) -> str:
    motifs = motifs_info.get("motifs") if isinstance(motifs_info, dict) else None
    motifs = [str(x).strip() for x in (motifs or []) if str(x).strip()]
    if not motifs:
        return ""
    parts: List[str] = []
    for mm in motifs[:2]:
        parts.append(f"Cohesion detail: {mm}.")
    parts.append("Use motifs subtly; do not dominate the scene.")
    return " ".join([p.strip() for p in parts if p.strip()]).strip()

def _lint_shot_prompt(chars: List[Dict[str, Any]], seed_txt: str, notes: str, final_prompt: str) -> List[str]:
    """Return lint notes (WARN/FAIL strings) for this shot."""
    notes_out: List[str] = []
    shot_blob = f"{seed_txt}\n{notes}".lower()

    for c in chars:
        nm = (c.get("name") or "Character").strip()
        face = c.get("face_traits") or []
        hair = (c.get("hair") or "").strip()
        outfit = (c.get("outfit") or "").strip()
        if len(face) < 3:
            notes_out.append(f"WARN: {nm} face_traits has <3 items (lock weak).")

        # Simple conflict check: if do_not_change token is mentioned positively in the shot text.
        for item in (c.get("do_not_change") or []):
            tok = _normalize_no_item(str(item))
            if tok and tok.lower() in shot_blob:
                notes_out.append(f"WARN: Shot text mentions '{tok}' but it's in {nm}.do_not_change (possible conflict).")

    
    # Chunk 2 checks: drama/motifs block presence
    if "Narrative beat:" not in (final_prompt or ""):
        notes_out.append("WARN: Narrative beat missing from prompt.")
    return notes_out


def _cb_compact_identity_phrase(c: Dict[str, Any]) -> str:
    """Compact, positive-only identity phrase for prompt (avoid clone-inducing repetition)."""
    try:
        taxonomy = str(c.get("taxonomy") or "").strip().lower()
    except Exception:
        taxonomy = ""
    name = str(c.get("name") or "").strip()
    role = str(c.get("role") or "").strip()

    face = c.get("face_traits") or []
    if not isinstance(face, list):
        face = []
    face = [str(x).strip() for x in face if str(x).strip()]
    hair = str(c.get("hair") or "").strip()
    outfit = str(c.get("outfit") or "").strip()
    age = str(c.get("age") or "").strip()

    species = str(c.get("species") or "").strip()
    markings = str(c.get("markings") or "").strip()
    accessories = str(c.get("accessories") or "").strip()

    parts: List[str] = []

    if taxonomy == "animal":
        lead = species or role or name or "animal"
        parts.append(f"a {lead}".strip())
        if markings:
            parts.append(markings)
        if accessories:
            parts.append(accessories)
    elif taxonomy == "creature":
        lead = role or name or "creature"
        parts.append(f"a {lead}".strip())
        if face:
            parts.append(", ".join(face[:2]))
        if outfit:
            parts.append(outfit)
    else:
        if role:
            lead = role
        elif age:
            lead = age
        elif name and name.lower() not in ("character", "person", "man", "woman"):
            lead = name
        else:
            lead = "person"
        parts.append(f"a {lead}".strip())
        if face:
            parts.append(", ".join(face[:2]))
        if hair:
            parts.append(hair)
        if outfit:
            parts.append(outfit)

    phrase = " with ".join([p for p in parts if p])
    phrase = re.sub(r"[ ]{2,}", " ", phrase).strip()
    return phrase

def _people_policy_clause(chars: List[Dict[str, Any]], blob: str) -> str:
    """Positive-only clause that prevents accidental extra people/clones (works even when negatives are ignored)."""
    human_chars = [c for c in (chars or []) if str(c.get("taxonomy") or "").strip().lower() == "human"]
    if human_chars:
        descs = [_cb_compact_identity_phrase(c) for c in human_chars]
        descs = [d for d in descs if d]
        n = len(descs)
        if n == 1:
            return f"Scene contains exactly one person: {descs[0]}."
        return f"Scene contains exactly {n} people: " + "; ".join(descs) + "."
    try:
        has_people = _shot_has_people_hint(blob or "")
    except Exception:
        has_people = False
    if not has_people:
        return "Scene contains zero people."
    return ""



# -----------------------------
# Prompt hygiene (Chunk 3B replacement)
# - Remove / rewrite "hit words" that models literalize into unwanted objects (locks, chains, borders, collages, etc.)
# - Runs as a final pass on prompts before saving.
# -----------------------------
_HITWORD_REWRITES: List[Tuple[re.Pattern, str]] = [
    # Avoid literal padlocks/chains from "lock/locked"
    (re.compile(r"\bidentity\s+lock\b", re.IGNORECASE), "identity consistency"),
    (re.compile(r"\block(ed|ing)?\b", re.IGNORECASE), "consistent"),
    # Avoid watcher silhouettes / aquarium vibe (keep it natural; never inject "abstract shape")
    (re.compile(r"\bforeground\s+silhouette\b", re.IGNORECASE), "foreground figure"),
    (re.compile(r"\bsilhouette\b", re.IGNORECASE), "figure"),
    # "abstract shape" wording causes identity drift; rewrite to neutral phrasing
    (re.compile(r"\babstract\s+shape\b", re.IGNORECASE), "figure"),
    # Replace shadow-y phrasing that keeps recurring in prompts
    (re.compile(r"\bcasting\s+soft\s+shadows\b", re.IGNORECASE), "casting gentle shade"),
    (re.compile(r"\bsoft\s+shadows\b", re.IGNORECASE), "soft shade"),
    # Avoid repetitive "forest" wording (use a neutral synonym)
    (re.compile(r"\bforests\b", re.IGNORECASE), "woodlands"),
    (re.compile(r"\bforest\b", re.IGNORECASE), "woodland"),
    (re.compile(r"\bviewer\b|\bwatching\b|\bspectator\b|\bobserver\b", re.IGNORECASE), "unobstructed view"),
    (re.compile(r"\baquarium\b|\bbehind\s+glass\b|\bthrough\s+glass\b|\bwindow\s+reflection\b", re.IGNORECASE), "clean view"),
    # Borders / frames / overlays
    (re.compile(r"\bsubtle\s+stripes\b", re.IGNORECASE), "subtle film grain"),
    (re.compile(r"\bstripes\b|\bstriped\b", re.IGNORECASE), "subtle texture"),
    (re.compile(r"\bborder\b|\bframe\b|\bframing\b", re.IGNORECASE), "composition"),
    (re.compile(r"\bvignette\b", re.IGNORECASE), "soft lighting falloff"),
    # Halo / ring from rim-light wording
    (re.compile(r"\bneon\s+rim\s+light\b", re.IGNORECASE), "soft cinematic edge light"),
    (re.compile(r"\bhalo\b|\bangel\s+ring\b", re.IGNORECASE), "soft glow"),
]

_HITWORD_NEGATIVE_APPEND: List[str] = [
    # Prevent common artifacts (keep short; avoid "panel/collage" priming)
    "border", "vignette border", "mosaic",
    "padlock", "lock", "chain", "collar", "leash", "shackles",
    "spectator", "silhouette", "person watching", "behind glass", "aquarium",
    "halo", "ring above head",
]
def _sanitize_prompt_text(s: str) -> str:
    s = (s or "").replace("\r", "").strip()
    if not s:
        return ""
    out = s

    # Remove non-visual "beat labels"/bullets that some models literalize (e.g., Hook -> fishhook),
    # and strip list-prefix tokens that can bias the scene (e.g., "- a green alien ...").
    # This only affects the *leading* portion of each paragraph and avoids touching character bible headers.
    try:
        paragraphs = re.split(r"\n{2,}", out)
        cleaned = []
        for para in paragraphs:
            t = (para or "").strip()
            if not t:
                continue

            tl = t.lower()
            if tl.startswith("own character bible") or tl.startswith("main characters"):
                cleaned.append(t)
                continue

            # Strip common bullet/number prefixes
            t = re.sub(r"^\s*[-*•]+\s*", "", t)
            t = re.sub(r"^\s*\(?\d{1,3}\)?[\.)]\s*", "", t)

            # Strip "Narrative beat: Hook." / "Phase: Resolve." style prefixes
            t = re.sub(
                r"^\s*(?:narrative\s+beat|beat|phase)\s*:\s*(?:hook|build|turn|resolve|intro|outro)\b[\.:;\-]*\s*",
                "",
                t,
                flags=re.IGNORECASE,
            )

            # Strip leading "a/an/the <subject> Hook." prefix (common in seeded shot lists)
            t = re.sub(
                r"^\s*(?:a|an|the)\s+[^\.]{1,60}?\s+(?:hook|build|turn|resolve|intro|outro)\b[\.:;\-]*\s*",
                "",
                t,
                flags=re.IGNORECASE,
            )

            # Strip standalone beat/label words at the start
            t = re.sub(
                r"^\s*(?:hook|build|turn|resolve|intro|outro|scene|shot|take)\b[\.:;\-]*\s*",
                "",
                t,
                flags=re.IGNORECASE,
            )

            cleaned.append(t.strip())

        if cleaned:
            out = "\n\n".join(cleaned).strip()
    except Exception:
        out = out.strip()

    for rx, repl in _HITWORD_REWRITES:
        out = rx.sub(repl, out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _sanitize_negative_list(neg_csv: str) -> str:
    base = _parse_negative_text(neg_csv or "")
    base_l = {x.lower() for x in base}
    for tok in _HITWORD_NEGATIVE_APPEND:
        if tok and tok.lower() not in base_l:
            base.append(tok)
            base_l.add(tok.lower())
    return ", ".join([x for x in base if (x or "").strip()])


# Render-prompt hygiene helpers (prevents caption/watermark text when shot specs contain internal tokens)
_NONVISUAL_TOKEN_RX = re.compile(r"\b[a-z]{2,}_[a-z0-9_]{2,}\b", re.IGNORECASE)
_STANDALONE_MARKER_LINE_RX = re.compile(r"^\s*[a-z0-9_\-]{3,}\s*$", re.IGNORECASE)

def _ascii_only(s: str) -> str:
    try:
        return (s or "").encode("ascii", "ignore").decode("ascii")
    except Exception:
        return str(s or "")

def _strip_nonvisual_markers(s: str) -> str:
    """Remove internal marker tokens (underscores/ids) that image models often render as on-screen text."""
    s = (s or "").replace("\r", "")
    if not s.strip():
        return ""
    # Drop standalone marker lines like "exit_store" / "pp_snop_foo_2"
    lines = []
    for ln in s.split("\n"):
        t = ln.strip()
        if not t:
            continue
        if _STANDALONE_MARKER_LINE_RX.match(t) and ("_" in t or any(ch.isdigit() for ch in t)):
            continue
        lines.append(ln)
    out = "\n".join(lines)
    # Remove inline underscore tokens
    out = _NONVISUAL_TOKEN_RX.sub("", out)
    # Collapse repeated whitespace
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()



# -----------------------------
# Text / watermark guardrails
# -----------------------------
# Many txt2img engines will occasionally hallucinate "caption-like" text (especially if the prompt
# contains label-style formatting). By default, Planner should *avoid* on-screen text unless the
# user explicitly asks for it.

_TEXT_REQUEST_HINTS = [
    # English
    "add text", "with text", "caption", "subtitles", "subtitle", "logo", "watermark",
    "title card", "titlecard", "typography", "sign", "signage", "poster", "banner",
    "label", "name tag", "nametag", "words", "letters", "numbers", "qr", "barcode",
    "speech bubble", "comic text", "lower third", "on-screen text", "onscreen text",
    # Common non-English hints (very small list; still ASCII-safe)
    "hanzi", "kanji", "kana", "hangul",
]

_TEXT_REQUEST_NEGATIONS = [
    "no text", "without text", "no caption", "no captions", "no subtitles", "no watermark", "no logo",
    "no writing", "no signage", "no labels",
]

_TEXT_NEGATIVE_KEYWORDS = [
    "text", "word", "words", "letter", "letters", "number", "numbers",
    "watermark", "logo", "subtitle", "subtitles", "caption", "label",
    "typography", "sign", "signage", "writing", "calligraphy",
    "hanzi", "kanji", "kana", "hangul", "glyph", "characters", "symbol",
    "qr", "barcode", "ui", "hud", "overlay",
]

_ANTI_TEXT_NEGS = [
    "text", "letters", "words", "numbers",
    "watermark", "logo", "signature", "stamp", "emblem",
    "subtitles", "caption", "label", "title", "poster",
    "sign", "signage", "writing", "calligraphy", "typography",
    "Chinese characters", "Hanzi", "Kanji", "Kana", "Hangul",
    "QR code", "barcode",
    "HUD", "UI", "overlay",
    "symbols",
    "glyphs",
]


# -----------------------------
# Subject fidelity (animals)
# -----------------------------
# Some prompt enhancers/distillers may hallucinate humans (e.g., "two people walking") even when
# the original shot seed is clearly about animals. When we detect animal keywords in the ORIGINAL
# shot seed, we enforce a strict subject lock on prompt_compiled/prompt_used.

_ANIMAL_KEYWORDS = [
    # common pets
    "cat", "kitten", "dog", "puppy",
    # farm
    "horse", "pony", "cow", "bull", "calf", "sheep", "goat", "pig", "boar",
    "chicken", "hen", "rooster", "duck", "goose", "turkey",
    # small mammals
    "rabbit", "bunny", "hare", "hamster", "guinea pig", "mouse", "rat", "squirrel", "raccoon",
    "otter", "beaver",
    # wild mammals
    "deer", "elk", "moose", "bear", "wolf", "fox", "lion", "tiger", "leopard", "cheetah",
    "elephant", "giraffe", "zebra", "monkey", "ape", "gorilla", "chimp", "orangutan",
    "kangaroo", "koala", "panda",
    # birds
    "bird", "eagle", "hawk", "owl", "parrot", "pigeon", "crow", "raven", "swan", "penguin",
    # reptiles/amphibians
    "turtle", "lizard", "snake", "frog", "toad", "alligator", "crocodile",
    # sea life
    "fish", "dolphin", "whale", "shark", "seal",
]

# Words that should NOT appear in the main prompt body when an animal subject is detected.
# (They are allowed only inside the explicit hard constraint line we append.)
_HUMAN_WORDS = [
    "people", "person", "couple", "character", "characters",
    "man", "woman", "men", "women",
    "child", "children", "boy", "girl",
    "human", "humans",
]



# Heuristic: if the user's start prompt does not specify a subject type, default to humans.
# Used to steer planning/shots away from random animal protagonists (unless requested).
_OTHER_SUBJECT_HINTS = [
    "robot", "android", "cyborg", "alien", "extraterrestrial",
    "monster", "creature", "dragon", "dinosaur",
    "zombie", "vampire", "werewolf", "ghost",
    "demon", "angel", "mech", "golem",
]

def _infer_default_subject_taxonomy(prompt: str, extra_info: str = "") -> str:
    """Return 'human' (default), 'animal', or 'creature' based on the start prompt + extra info."""
    blob = f"{prompt or ''} {extra_info or ''}".strip()
    if not blob:
        return "human"
    # If the user explicitly mentions animals, default to animal.
    try:
        if _detect_animals_in_seed(blob):
            return "animal"
    except Exception:
        pass
    low = blob.lower()
    for w in _OTHER_SUBJECT_HINTS:
        try:
            if re.search(r"\b" + re.escape(w) + r"\b", low):
                return "creature"
        except Exception:
            continue
    return "human"

# Words to ban from prompt body for animal shots (glass/reflection framing).
_GLASS_WORDS = [
    "glass", "mirror", "mirrors", "reflect", "reflection", "reflections", "reflective",
    "window", "windows", "transparent", "transparency", "see-through",
    "barrier", "barriers", "glass fence", "glass fences", "fence", "fences",
]

# Symbol / grid / glyph style injections that can cause unwanted overlays or artifacts.
_SYMBOL_GRID_WORDS = [
    "glyph", "glyphs", "sigil", "sigils", "rune", "runes",
    "symbol", "symbols", "grid", "grids", "cipher", "occult",
]


# -----------------------------
# Subject Guard (generic)
# -----------------------------
# Generic guard to reduce subject drift from prompt enhancer/template garnish.
# - Default-gates humans, glass/reflection framing, and symbol/grid/glyph overlays unless the ORIGINAL seed mentions them.
# - Validates that at least one key keyword from the ORIGINAL seed survives into the compiled prompt.
# - If validation fails, forces a minimal fallback: SUBJECT line + short safe cinematic hint.

_SUBJECT_GUARD_STOPWORDS = {
    "the","a","an","and","or","but","with","without","in","on","at","to","from","of","for","into","over","under",
    "near","by","as","is","are","was","were","be","been","being",
    "this","that","these","those",
    "then","than","while","during",
    "it","its","their","his","her","my","your","our",
    "very","really","just","only","still",
}

_SUBJECT_GUARD_KEEP_HINTS = {
    # small, conservative list to improve detection on very short seeds
    "cat","dog","horse","cow","sheep","goat","pig","chicken","duck","bird",
    "farm","beach","street","alley","room","kitchen","bedroom",
    "robot","alien","spaceship","car","truck","train",
}

def _seed_mentions_word(seed0: str, word: str) -> bool:
    s = (seed0 or "").lower()
    w = (word or "").strip().lower()
    if not s or not w:
        return False
    if " " in w or "-" in w:
        return w in s
    return bool(re.search(r"\b" + re.escape(w) + r"\b", s))

def _subject_guard_keywords(seed_text: str) -> List[str]:
    seed0 = _strip_nonvisual_markers(_ascii_only(seed_text or "")).lower().strip()
    if not seed0:
        return []
    toks = re.findall(r"[a-z0-9]+", seed0)
    out: List[str] = []
    seen = set()
    for t in toks:
        if len(t) < 3:
            continue
        if t in _SUBJECT_GUARD_STOPWORDS:
            continue
        if t not in seen:
            out.append(t)
            seen.add(t)
    for h in sorted(_SUBJECT_GUARD_KEEP_HINTS):
        if h in seen:
            continue
        if _seed_mentions_word(seed0, h):
            out.append(h)
            seen.add(h)
    return out

def _subject_guard_strip(seed0: str, prompt_text: str) -> str:
    body = (prompt_text or "").strip()
    if not body:
        return ""

    # Remove existing SUBJECT prefix if present (we only add it on fallback)
    body = re.sub(r"^\s*SUBJECT\s*:\s*.*?\.\s*", "", body, flags=re.IGNORECASE | re.DOTALL).strip()

    explicit_humans = any(_seed_mentions_word(seed0, w) for w in _HUMAN_WORDS)
    animals_present = bool(_detect_animals_in_seed(seed0))
    # Default is humans when the seed is ambiguous; strip humans only for clearly animal seeds.
    allow_humans = bool(explicit_humans or (not animals_present))
    allow_glass = any(_seed_mentions_word(seed0, w) for w in _GLASS_WORDS) or any(_seed_mentions_word(seed0, w) for w in ("reflective","reflection","reflections","mirror","mirrors","glass","window","windows","transparent"))
    allow_symbols = any(_seed_mentions_word(seed0, w) for w in _SYMBOL_GRID_WORDS) or any(_seed_mentions_word(seed0, w) for w in ("geometric","geometry","grid","pattern","symbols","glyphs"))

    if not allow_humans:
        body = _strip_keyword_sentences(body, _HUMAN_WORDS)

    if not allow_glass:
        body = _strip_keyword_sentences(body, _GLASS_WORDS + ["reflective", "mirrorlike", "mirror-like", "polished metal", "chrome", "window reflection", "glass framing"])

    if not allow_symbols:
        body = _strip_keyword_sentences(body, _SYMBOL_GRID_WORDS + ["geometric", "geometry", "grid", "pattern woven", "woven into", "sigil", "rune", "cipher"])

    body = re.sub(r"\s{2,}", " ", body).strip()
    return body

def _apply_subject_guard(seed_text: str, prompt_text: str, negative_text: str) -> Tuple[str, str, Dict[str, Any]]:
    seed0 = _strip_nonvisual_markers(_ascii_only(seed_text or "")).strip()
    seed0_low = seed0.lower().strip()

    kws = _subject_guard_keywords(seed0)
    stripped = _subject_guard_strip(seed0_low, prompt_text)

    validated = True
    fallback_used = False
    hits: List[str] = []

    pcheck = (stripped or prompt_text or "").lower()
    if kws:
        for kw in kws:
            if kw and re.search(r"\b" + re.escape(kw) + r"\b", pcheck):
                hits.append(kw)
        validated = bool(hits)

    out_prompt = (stripped or prompt_text or "").strip()

    if kws and not validated:
        fallback_used = True
        out_prompt = (f"SUBJECT: {seed0}." + " " + "Cinematic photo, natural perspective, clean composition, stable subject proportions.").strip()

    dbg = {
        "validated": bool(validated),
        "fallback_used": bool(fallback_used),
        "seed_keywords": int(len(kws)),
        "keyword_hits": hits[:8],
    }
    return out_prompt.strip(), (negative_text or "").strip(), dbg

def _strip_keyword_sentences(text: str, keywords: List[str]) -> str:
    """Remove sentence-like chunks containing any of the given keywords (case-insensitive)."""
    body = (text or "").strip()
    if not body:
        return ""
    kws = [k.strip().lower() for k in (keywords or []) if str(k or "").strip()]
    if not kws:
        return body
    chunks = re.split(r"(?<=[.!?])\s+|\n+", body)
    out: List[str] = []
    for ch in chunks:
        t = (ch or "").strip()
        if not t:
            continue
        low = t.lower()
        if any(k in low for k in kws):
            continue
        out.append(t)
    res = " ".join(out).strip()
    res = re.sub(r"\s{2,}", " ", res).strip()
    return res

def _seed_has_cat(seed: str) -> bool:
    s = (seed or "").lower()
    return bool(re.search(r"\b(cat|kitten|cats|kittens)\b", s))

def _seed_has_dog(seed: str) -> bool:
    s = (seed or "").lower()
    return bool(re.search(r"\b(dog|puppy|dogs|puppies)\b", s))

_COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

def _parse_species_count(seed: str, species: str) -> Optional[int]:
    """Best-effort count extraction for a single species word (e.g. cat/dog)."""
    s = (seed or "").lower()
    if not s.strip():
        return None

    if species == "cat":
        toks = r"cats?|kittens?"
    elif species == "dog":
        toks = r"dogs?|pupp(?:y|ies)"
    else:
        toks = re.escape(species) + r"s?"

    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:\w+\s+){0,2}(%s)\b" % toks, s)
    if m:
        raw = m.group(1)
        if raw.isdigit():
            try:
                return int(raw)
            except Exception:
                return None
        return _COUNT_WORDS.get(raw, None)

    if re.search(r"\b(a|an)\s+(%s)\b" % toks, s):
        return 1

    return None

def _animal_mode_meta(seed_text: str) -> Dict[str, Any]:
    """Return animal-mode flags + expectations derived from the ORIGINAL shot seed."""
    seed0 = _strip_nonvisual_markers(seed_text or "")
    seed0 = _ascii_only(seed0).strip()
    animals = _detect_animals_in_seed(seed0)
    animal_mode = bool(animals)

    expects_catdog = False
    cat_n = None
    dog_n = None

    if animal_mode and _seed_has_cat(seed0) and _seed_has_dog(seed0):
        other = []
        for a in animals:
            al = str(a or "").lower()
            if al in ("cat", "kitten", "dog", "puppy"):
                continue
            other.append(al)
        if not other:
            expects_catdog = True
            cat_n = _parse_species_count(seed0, "cat") or 1
            dog_n = _parse_species_count(seed0, "dog") or 1

    return {
        "seed": seed0,
        "animal_mode": bool(animal_mode),
        "expects_catdog": bool(expects_catdog),
        "cat_n": cat_n,
        "dog_n": dog_n,
    }

def _build_animal_anchor(meta: Dict[str, Any]) -> str:
    seed0 = str(meta.get("seed") or "").strip()
    if not seed0:
        return ""

    if bool(meta.get("expects_catdog")):
        cat_n = int(meta.get("cat_n") or 1)
        dog_n = int(meta.get("dog_n") or 1)

        s = seed0.lower()
        action = "walking" if ("walk" in s) else "standing"
        loc = "on a farm" if ("farm" in s) else ""

        cat_word = "cat" if cat_n == 1 else "cats"
        dog_word = "dog" if dog_n == 1 else "dogs"
        cat_c = "one" if cat_n == 1 else str(cat_n)
        dog_c = "one" if dog_n == 1 else str(dog_n)
        a = f"SUBJECT: exactly {cat_c} {cat_word} and exactly {dog_c} {dog_word} {action}"
        if loc:
            a += f" {loc}"
        a += "."
        return a

    return f"SUBJECT: {seed0}."

def _validate_animal_prompt(meta: Dict[str, Any], prompt_text: str) -> bool:
    """Validation pass AFTER prompt_compiled is built (animal shots only)."""
    if not bool(meta.get("animal_mode")):
        return True

    raw = (prompt_text or "").strip()
    if not raw:
        return False

    # Remove hard-constraint sentences ("No ...") from the validation scan; validate descriptive body only.
    chunks = re.split(r"(?<=[.!?])\s+|\n+", raw)
    body_parts: List[str] = []
    for ch in chunks:
        t = (ch or "").strip()
        if not t:
            continue
        low = t.lower()
        if low.startswith("no "):
            continue
        body_parts.append(t)

    p = " ".join(body_parts).lower()

    if bool(meta.get("expects_catdog")):
        if not re.search(r"\bcat\b", p):
            return False
        if not re.search(r"\bdog\b", p):
            return False

        if re.search(r"\b(?:people|person|man|woman|men|women|couple|child|children)\b", p):
            return False

        if re.search(r"\b(?:glass|mirror|mirrors|reflection|reflections|reflect|reflective|window|windows)\b", p):
            return False

    return True

def _apply_subject_fidelity_v2(seed_text: str, prompt_text: str, negative_text: str) -> Tuple[str, str, Dict[str, Any]]:
    """Animal-shot subject fidelity + drift guards + post-build validation with fallback."""
    meta = _animal_mode_meta(seed_text or "")
    animal_mode = bool(meta.get("animal_mode"))

    debug = {
        "animal_mode": bool(animal_mode),
        "validated": True,
        "fallback_used": False,
    }

    if not animal_mode:
        return (prompt_text or "").strip(), (negative_text or "").strip(), debug

    seed0 = str(meta.get("seed") or "").strip()
    anchor = _build_animal_anchor(meta)

    body = _strip_human_language_from_prompt(prompt_text or "")
    body = _strip_keyword_sentences(body, _GLASS_WORDS + _SYMBOL_GRID_WORDS)

    constraints: List[str] = []
    constraints.append("No humans. No men, no women, no children.")
    constraints.append("No glass, no mirrors, no reflections, no windows, no transparent barriers, no glass fences.")
    if bool(meta.get("expects_catdog")):
        cat_n = int(meta.get("cat_n") or 1)
        dog_n = int(meta.get("dog_n") or 1)
        cat_word = "cat" if cat_n == 1 else "cats"
        dog_word = "dog" if dog_n == 1 else "dogs"
        cat_c = "one" if cat_n == 1 else str(cat_n)
        dog_c = "one" if dog_n == 1 else str(dog_n)
        constraints.append(f"No extra animals. Exactly {cat_c} {cat_word} and exactly {dog_c} {dog_word}.")

    parts: List[str] = []
    if anchor:
        parts.append(anchor)
    if body:
        parts.append(body)
    parts.extend(constraints)

    final_prompt = " ".join([p.strip() for p in parts if str(p or "").strip()]).strip()
    final_prompt = re.sub(r"\s{2,}", " ", final_prompt).strip()

    # Reinforce negatives
    neg = (negative_text or "").strip()
    neg_parts = _parse_negative_text(neg)
    have = {p.lower() for p in neg_parts}

    extra_negs = [
        "people", "person", "human", "humans", "man", "woman", "men", "women", "child", "children",
        "glass", "mirror", "mirrors", "reflection", "reflections", "reflective", "window", "windows", "transparent", "glass fence", "glass fences",
        "glyph", "sigil", "runes", "symbols", "grid",
    ]
    for x in extra_negs:
        xl = x.lower()
        if xl not in have:
            neg_parts.append(x)
            have.add(xl)

    final_negative = ", ".join([p.strip() for p in neg_parts if p.strip()]).strip()
    final_negative = _sanitize_negative_list(final_negative)

    ok = _validate_animal_prompt(meta, final_prompt)
    debug["validated"] = bool(ok)
    if not ok:
        debug["fallback_used"] = True
        minimal = "Cinematic lighting, clean composition, sharp focus, detailed, natural perspective."
        fb_parts = []
        if seed0:
            fb_parts.append(f"SUBJECT: {seed0}.")
        else:
            fb_parts.append("SUBJECT: animals.")
        fb_parts.append(minimal)
        fb_parts.extend(constraints)
        final_prompt = " ".join([p.strip() for p in fb_parts if str(p or "").strip()]).strip()
        final_prompt = re.sub(r"\s{2,}", " ", final_prompt).strip()

    return final_prompt, final_negative, debug
def _detect_animals_in_seed(seed_text: str) -> List[str]:
    """Return a de-duplicated list of animal keywords present in the seed_text."""
    s = (seed_text or "").lower()
    if not s.strip():
        return []
    found: List[str] = []
    for kw in _ANIMAL_KEYWORDS:
        if " " in kw:
            if kw in s:
                found.append(kw)
            continue
        rx = re.compile(r"\b" + re.escape(kw) + r"s?\b", re.IGNORECASE)
        if rx.search(s):
            found.append(kw)
    # de-dupe while preserving order
    out: List[str] = []
    seen = set()
    for x in found:
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out

def _strip_human_language_from_prompt(prompt_body: str) -> str:
    """Remove sentences that introduce humans/people language."""
    body = (prompt_body or "").strip()
    if not body:
        return ""

    # If the body already contains a SUBJECT prefix from a previous run, remove it (we re-add cleanly).
    body = re.sub(r"^\s*SUBJECT\s*:\s*.*?\.\s*", "", body, flags=re.IGNORECASE | re.DOTALL)

    # Sentence-ish splitting (keep it simple and robust for both newlines and punctuation).
    chunks = re.split(r"(?<=[.!?])\s+|\n+", body)
    kept: List[str] = []
    human_rx = re.compile(r"\b(?:" + "|".join([re.escape(w) for w in _HUMAN_WORDS]) + r")\b", re.IGNORECASE)
    for ch in chunks:
        t = (ch or "").strip()
        if not t:
            continue
        if human_rx.search(t):
            continue
        kept.append(t)

    out = " ".join(kept).strip()
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out

def _apply_subject_fidelity(seed_text: str, prompt_text: str, negative_text: str) -> Tuple[str, str]:
    """Backward-compatible wrapper for subject fidelity (animals)."""
    p, n, _dbg = _apply_subject_fidelity_v2(seed_text, prompt_text, negative_text)
    return p, n

_LABEL_STRIP_RX = re.compile(
    r"^\s*(narrative beat|direction|camera|lighting|mood|phase|style|notes|seed|shot|scene|take)\s*:\s*",
    re.IGNORECASE
)

def _prompt_requests_text(*chunks: str) -> bool:
    """
    Return True if the user prompt explicitly asks for text/signage/logos/captions.
    This is intentionally conservative: only opt-in when there are strong hints.
    """
    blob = " ".join([(c or "") for c in chunks]).strip().lower()
    if not blob:
        return False

    # If user explicitly negates text, treat as "no text" even if the word appears.
    for neg in _TEXT_REQUEST_NEGATIONS:
        if neg in blob:
            return False

    # Strong hints
    for kw in _TEXT_REQUEST_HINTS:
        if kw in blob:
            return True

    # Quoted strings often indicate intended on-screen text (e.g., "HELLO WORLD")
    if re.search(r"['\"][^'\"]{2,40}['\"]", blob):
        return True

    return False

def _filter_text_negatives_list(items: List[str]) -> List[str]:
    out: List[str] = []
    for it in (items or []):
        low = (it or "").strip().lower()
        if not low:
            continue
        if any(k in low for k in _TEXT_NEGATIVE_KEYWORDS):
            continue
        out.append(it)
    return out

def _adjust_text_negatives_csv(neg_csv: str, allow_text: bool) -> str:
    items = _parse_negative_text(neg_csv or "")
    if allow_text:
        items = _filter_text_negatives_list(items)
        return ", ".join(items).strip()
    # add strong anti-text tokens (de-duped)
    seen = {x.lower() for x in items}
    for tok in _ANTI_TEXT_NEGS:
        if tok.lower() not in seen:
            items.append(tok)
            seen.add(tok.lower())
    return ", ".join(items).strip()

def _proseify_prompt(s: str) -> str:
    """
    Convert label-heavy multi-line specs into natural prose to reduce accidental caption rendering.
    """
    s = (s or "").replace("\r", "")
    if not s.strip():
        return ""
    lines: List[str] = []
    for ln in s.split("\n"):
        t = (ln or "").strip()
        if not t:
            continue
        # Strip known label prefixes
        t = _LABEL_STRIP_RX.sub("", t).strip()
        if not t:
            continue
        lines.append(t)
    out = " ".join(lines)
    out = re.sub(r"[ \t]{2,}", " ", out)
    return out.strip()

def _finalize_render_prompt(render_prompt: str, spec_prompt: str, allow_text: bool) -> str:
    rp = (render_prompt or "").strip()
    if not rp:
        rp = (spec_prompt or "").strip()
    rp = _strip_nonvisual_markers(_ascii_only(rp))
    # If prompt still looks like a labeled spec, prose-ify it
    if "\n" in rp or ":" in rp:
        rp2 = _proseify_prompt(rp)
        if rp2:
            rp = rp2
    rp = _sanitize_prompt_text(rp)
    return rp.strip()


# Qwen3-VL render-prompt distillation is disabled for txt2img; prompts are compiled locally.
_ENABLE_RENDER_PROMPT_DISTILLER = False



def _assemble_shot_prompt(
    shot: Dict[str, Any],
    bible: List[Dict[str, Any]],
    extra_info: str,
    user_negatives: str = "",
    image_model: str = "",
    allow_text: bool = False,
    own_character_prompts: Optional[List[str]] = None,
    user_prompt: str = "",
) -> Tuple[str, str, List[str]]:
    """Build a *real* txt2img prompt (visual-only) from user inputs + shot metadata.

    Key rule: never emit guideline/producer prose into the positive prompt.
    All constraints and drift-prevention live in the NEGATIVE prompt list.
    """
    stage_raw = shot.get("stage_directions") or {}
    if not isinstance(stage_raw, dict):
        stage_raw = {}

    # Shot visual seed (already generated upstream)
    visual_raw = str(shot.get("visual_description") or "").strip()
    visual_raw = _sanitize_visual_description(visual_raw) if visual_raw else ""
    seed_txt = (visual_raw or str(shot.get("seed") or "").strip())
    seed_txt = _strip_nonvisual_markers(seed_txt)

    # Default subject is humans; only strip human language for clearly animal-only shots.
    try:
        _am = _animal_mode_meta(seed_txt)
        _is_animal_shot = bool(_am.get("animal_mode"))
    except Exception:
        _is_animal_shot = False


    # Producer notes / purpose are useful, but often include guidelines; sanitize aggressively.
    purpose = str(stage_raw.get("purpose") or "").strip()
    notes = (purpose or str(shot.get("notes") or "").strip())
    notes = _strip_nonvisual_markers(notes)
    if _is_animal_shot:
        notes = _strip_human_language_from_prompt(notes)

    # Camera / lighting / mood as lightweight tags (avoid spec formatting)
    camera = str(stage_raw.get("camera") or shot.get("camera") or "medium shot").strip()
    lighting = str(stage_raw.get("lighting") or shot.get("lighting") or "cinematic lighting").strip()
    mood = str(stage_raw.get("mood") or shot.get("mood") or "neutral").strip()

    # User start prompt & extra box (style/details)
    up = _strip_nonvisual_markers(str(user_prompt or "").strip())
    if _is_animal_shot:
        up = _strip_human_language_from_prompt(up)
    ex = _strip_nonvisual_markers(str(extra_info or "").strip())
    if _is_animal_shot:
        ex = _strip_human_language_from_prompt(ex)

    # Own Character Bible (manual 1–2): keep as plain prose only (no headers/bullets)
    own_prompts = [str(x or "").strip() for x in (own_character_prompts or []) if str(x or "").strip()]
    own_lock_prose = ""
    if own_prompts:
        cleaned: List[str] = []
        for p in own_prompts:
            tt = _strip_nonvisual_markers(p)
            tt = re.sub(r"[\r\n\t]+", " ", tt)
            tt = re.sub(r"[ ]{2,}", " ", tt).strip()
            if tt:
                cleaned.append(tt)
        own_lock_prose = " ; ".join(cleaned).strip()

    # Character Bible (auto): pick relevant characters and build a compact, positive-only identity clause.
    chars = _pick_relevant_characters(bible, f"{seed_txt}\n{notes}\n{up}\n{ex}")
    people_clause = _people_policy_clause(chars, f"{up}\n{seed_txt}\n{notes}\n{ex}")

    # Build positive prompt as a single render-friendly paragraph.
    # IMPORTANT: no "no text", "avoid", "do not", "rules", etc in positive.
    bits: List[str] = []

    # People/identity clause early to prevent accidental extra people/clones (works when negatives are ignored)
    if people_clause:
        bits.append(people_clause)

    # Framing up front tends to help
    for b in [camera, lighting, mood]:
        if b:
            bits.append(b)

    # Core intent: user prompt → shot visual seed → brief notes
    for b in [up, seed_txt, notes]:
        if b:
            bits.append(b)

    # Auto character/subject anchors: include compact identity phrases (plain prose, no IDs/labels).
    # This helps models that ignore negatives stay on the intended subjects.
    if chars:
        try:
            anchors = [_cb_compact_identity_phrase(c) for c in chars]
            anchors = [a for a in anchors if a]
            if anchors:
                bits.append("Main subjects: " + "; ".join(anchors[:2]) + ".")
        except Exception:
            pass

    # Add character prose (if provided) as plain description, not instruction
    if own_lock_prose:
        bits.append(own_lock_prose)

    if ex:
        bits.append(ex)

    # Add a small quality/style tail (positive-only)
    bits.append("cinematic, high detail, sharp focus, clean composition")

    final_prompt = ", ".join([b.strip().strip(",") for b in bits if b and b.strip()]).strip()
    final_prompt = _sanitize_prompt_text(final_prompt)

    # Negatives: merge user negatives + safe global drift prevention
    ui_negs = _parse_negative_text(user_negatives)
    if allow_text:
        ui_negs = _filter_text_negatives_list(ui_negs)

    model_l = (image_model or "").lower()
    is_qwen_img = ("qwen" in model_l) or ("2512" in model_l)

    base_negs = [
        "low quality", "blurry", "out of frame", "jpeg artifacts",
        "watermark", "logo", "subtitles", "caption", "label", "overlay", "HUD", "UI",
        "border", "frame", "poster", "title card", "contact sheet", "collage", "split screen",
        "duplicate", "cloned", "multiple copies", "twins", "group photo",
        "mirror", "reflection", "glossy reflections", "glass dome", "display case", "showroom",
        "deformed", "disfigured", "bad anatomy", "bad hands",
        "extra fingers", "missing fingers", "mutated hands",
    ]

    # Text suppression belongs in negatives only unless user explicitly requests text
    if not allow_text:
        base_negs.extend(["text", "letters", "words", "Chinese characters", "Hanzi", "Kanji", "Kana", "Hangul"])

    # If not Qwen, add character drift prevention too (lightweight)
    neg_list = base_negs
    if not is_qwen_img:
        try:
            # existing helper may include helpful drift blockers
            neg_list = (base_negs + _drift_prevention_negatives(_pick_relevant_characters(bible, f"{seed_txt}\n{notes}")))
        except Exception:
            neg_list = base_negs

    final_negative = ", ".join([x for x in (neg_list + ui_negs) if (x or "").strip()])
    final_negative = _adjust_text_negatives_csv(final_negative, allow_text)
    final_negative = _sanitize_negative_list(final_negative)

    # Keep lint warnings (doesn't affect prompt)
    chars = _pick_relevant_characters(bible, f"{seed_txt}\n{notes}")
    lint = _lint_shot_prompt(chars, seed_txt, notes, final_prompt)
    return final_prompt, final_negative, lint

def _video_model_key(label: str) -> str:
    l = (label or "").lower()
    if "wan" in l and "2.2" in l:
        return "wan22"
    if "hunyuan" in l:
        return "hunyuan"
    # default (also for Auto / Qwen entries for now)
    return "hunyuan"

def _resolve_generation_profile(video_model_label: str, gen_quality_label: str) -> Dict[str, Any]:
    mk = _video_model_key(video_model_label)
    gk = _normalize_key(gen_quality_label)
    if mk == "wan22":
        if gk not in _WAN22_PRESETS:
            gk = "low"
        return dict(_WAN22_PRESETS[gk])
    # hunyuan default
    if gk not in _HUNYUAN_PRESETS:
        gk = "medium"
    return dict(_HUNYUAN_PRESETS[gk])

def _stable_uniform(seed_text: str, lo: float, hi: float) -> float:
    # deterministic float in [lo, hi]
    r = random.Random(_sha1_text(seed_text))
    if hi <= lo:
        return float(lo)
    return lo + (hi - lo) * r.random()


from PySide6.QtCore import Qt, QObject, Signal, Slot, QThread, QSize, QTimer
from PySide6.QtGui import QFont, QAction, QPixmap, QIcon, QStandardItemModel, QStandardItem, QColor, QPainter
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QTextEdit,
    QPlainTextEdit,
    QLineEdit,
    QPushButton,
    QGroupBox,
    QCheckBox,
    QButtonGroup,
    QRadioButton,
    QSlider,
    QComboBox,
    QSpinBox,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QMessageBox,
    QSplitter,
    QToolButton,
    QMenu,
    QTabWidget,
    QTableView,
    QHeaderView,
    QAbstractItemView,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QSizePolicy,
    QScrollArea,
    QStackedWidget,
    QFrame,
    QGraphicsDropShadowEffect,
)



# Optional: video preview in Clip Review (falls back to external open if unavailable)
_HAVE_QT_MEDIA = False
try:
    from PySide6.QtCore import QUrl
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtMultimediaWidgets import QVideoWidget
    _HAVE_QT_MEDIA = True
except Exception:
    QUrl = None  # type: ignore
    QMediaPlayer = None  # type: ignore
    QAudioOutput = None  # type: ignore
    QVideoWidget = None  # type: ignore
    _HAVE_QT_MEDIA = False


# -----------------------------
# Data model
# -----------------------------

@dataclass
class PlannerJob:
    job_id: str
    created_at: float

    # Step 1 inputs
    prompt: str
    negatives: str
    storytelling: bool
    music_background: bool
    silent: bool
    storytelling_volume: int  # 0-100
    music_volume: int         # 0-100

    # Chunk 7A music source
    music_mode: str           # none | file | ace15
    music_preset: str         # legacy simple preset name (kept for job.json compatibility)

    # Ace Step 1.5 (Chunk 7A replacement)
    ace15_preset_id: str      # selected preset id/key from presetmanager.json
    ace15_lyrics_enabled: bool
    ace15_lyrics_text: str
    ace15_audio_format: str   # mp3 | wav


    # Chunk 6B narration
    narration_enabled: bool
    narration_mode: str        # builtin | clone
    narration_voice: str       # builtin only
    narration_sample_path: str # clone only
    narration_language: str    # e.g. auto

    approx_duration_sec: int  # 5 .. 240
    resolution_preset: str    # Format label (e.g. "Landscape (16:9)")
    quality_preset: str       # Legacy: kept for job.json compatibility (always "1×" now)
    extra_info: str

    # Optional "future" attachments
    attachments: Dict[str, List[str]]  # keys: json/images/videos/text/transcripts

    # Internal: derived encoding targets 
    encoding: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _quality_defaults(name: str) -> Dict[str, Any]:
    """
    quality mapping:
    - Low: smaller disk, lower bitrate / higher CRF
    - Medium: default-ish: CRF 18 OR ~3000k
    - High: close to max but not extreme: CRF 16 OR ~6000k
    """
    n = (name or "").strip().lower()
    if n == "low":
        return {"mode": "crf", "crf": 22, "bitrate_kbps": 1500, "note": "Small files, faster encode"}
    if n == "high":
        return {"mode": "crf", "crf": 16, "bitrate_kbps": 6000, "note": "Near-high quality, larger files"}
    # Medium default
    return {"mode": "crf", "crf": 18, "bitrate_kbps": 3000, "note": "Balanced (default)"}


def _duration_label(seconds: int) -> str:
    m = seconds // 60
    s = seconds % 60
    if m <= 0:
        return f"{seconds}s"
    return f"{m}m {s:02d}s"


# -----------------------------
# Chunk 9A — Settings belong in Settings (format + upscaling)
# -----------------------------

def _aspect_mode_key(label_or_key: str) -> str:
    s = (label_or_key or "").strip().lower()
    if "portrait" in s or "9:16" in s:
        return "portrait"
    if "1:1" in s or "square" in s:
        return "square"
    return "landscape"

def _aspect_mode_display(mode: str) -> str:
    m = _aspect_mode_key(mode)
    if m == "portrait":
        return "Portrait"
    if m == "square":
        return "1:1"
    return "Landscape"

def _upscale_factor_from_label(label: str) -> int:
    s = (label or "").strip().lower()
    if "4" in s:
        return 4
    if "2" in s:
        return 2
    return 1

def _square_size_from_base(base_w: int, base_h: int) -> int:
    try:
        mx = int(max(int(base_w), int(base_h)))
    except Exception:
        mx = 1024
    # Use a sane square size aligned to 64px
    return 1024 if mx >= 1200 else 768

def _apply_aspect_to_size(base_w: int, base_h: int, aspect_mode: str) -> Tuple[int, int]:
    m = _aspect_mode_key(aspect_mode)
    if m == "portrait":
        return int(base_h), int(base_w)
    if m == "square":
        s = _square_size_from_base(base_w, base_h)
        return int(s), int(s)
    return int(base_w), int(base_h)




def _sha1_text(t: str) -> str:
    return hashlib.sha1((t or "").encode("utf-8", errors="ignore")).hexdigest()


def _sha1_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """SHA1 of a file on disk (hex). Returns empty string on failure."""
    try:
        h = hashlib.sha1()
        with open(path, "rb") as f:
            while True:
                b = f.read(int(chunk_size))
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""



def _auto_title_from_prompt(prompt: str) -> str:
    """Derive a human-friendly title from the user's main prompt (first 6–10 words, cleaned)."""
    try:
        s = (prompt or "").replace("\n", " ").strip()
    except Exception:
        s = ""
    # Remove some noisy punctuation while keeping readable spacing
    s = re.sub(r"[\t\r]+", " ", s)
    s = re.sub(r"[\[\]{}()<>\"“”'`]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "Untitled job"

    words = s.split()
    if len(words) > 10:
        words = words[:10]
    # Ensure we prefer at least 6 words when available
    if len(words) < 6:
        title = " ".join(words)
    else:
        title = " ".join(words[:max(6, min(len(words), 10))])

    # Final cleanup: trim dangling punctuation
    title = title.strip(" -_.,;:!?—–")
    return title or "Untitled job"


def _slugify_title(title: str) -> str:
    """Lowercase, spaces→dash, remove non [a-z0-9-], collapse dashes, trim."""
    try:
        s = (title or "").strip().lower()
    except Exception:
        s = ""
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"\-+", "-", s)
    s = s.strip("-")
    return s


def _next_slug_counter(output_base: str, slug: str) -> int:
    """Scan only output_base for folders starting with f"{slug}_" and return next counter (1-based)."""
    try:
        base = str(output_base or "").strip()
        if not base:
            return 1
        if not os.path.isdir(base):
            return 1
        prefix = f"{slug}_"
        max_n = 0
        for name in os.listdir(base):
            if not name.startswith(prefix):
                continue
            full = os.path.join(base, name)
            if not os.path.isdir(full):
                continue
            suffix = name[len(prefix):]
            m = re.match(r"^(\d+)", suffix)
            if not m:
                continue
            try:
                n = int(m.group(1))
            except Exception:
                continue
            if n > max_n:
                max_n = n
        return max_n + 1
    except Exception:
        return 1


def _safe_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)




def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def _safe_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _file_ok(path: str, min_bytes: int = 2) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except Exception:
        return False

def _extract_first_json(raw: str) -> Optional[Any]:
    """Robustly extract the first JSON object/array from a model response."""
    if not raw:
        return None
    s = raw.strip()
    # Strip fenced blocks if present
    if "```" in s:
        # try to keep content inside the first fence
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1] if parts[1].strip() else parts[2]
            s = s.strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()

    # Find first { or [
    start = None
    for k,ch in enumerate(s):
        if ch in "{[":
            start = k
            break
    if start is None:
        return None

    def _balanced_extract(ss: str, st: int) -> Optional[str]:
        depth = 0
        in_str = False
        esc = False
        open_ch = ss[st]
        close_ch = "}" if open_ch == "{" else "]"
        for i in range(st, len(ss)):
            ch = ss[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        return ss[st:i+1]
        return None

    cand = _balanced_extract(s, start)
    if cand:
        try:
            return json.loads(cand)
        except Exception:
            pass

    # Fallback: try progressively from start
    for end in range(len(s), start+1, -1):
        chunk = s[start:end].strip()
        try:
            return json.loads(chunk)
        except Exception:
            continue
    return None


def _append_prompt_used(path: str, title: str, system_prompt: str, user_prompt: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write(f"{title}\n")
        f.write("-"*80 + "\n")
        f.write("SYSTEM:\n" + (system_prompt or "") + "\n\n")
        f.write("USER:\n" + (user_prompt or "") + "\n")
        f.write("="*80 + "\n\n")


def _qwen_json_call(step_name: str, system_prompt: str, user_prompt: str, raw_path: str, prompts_used_path: str,
                    error_path: str, temperature: float = 0.2, max_new_tokens: int = 2048) -> Tuple[Optional[Any], str]:
    """Call Qwen3-VL via a short-lived subprocess and force JSON-only. Returns (parsed_json_or_none, raw_text).

    Reliability rules:
    - Always writes raw_path (plan_raw/shots_raw) even if parsing fails.
    - Writes prompts_used_path with system/user prompts.
    - Writes error_path with traceback on failure.
    - Robust JSON extraction: first parseable {...} or [...] block.

    VRAM rule:
    - Qwen is executed out-of-process so it cannot pin VRAM in the Planner process.
    """
    _append_prompt_used(prompts_used_path, step_name, system_prompt, user_prompt)
    raw_text = ""
    try:
        if not _HAVE_QWEN_TEXT:
            raise RuntimeError(f"Qwen text generator not available: {_QWEN_IMPORT_ERROR!r}")
        model_path = Path(_qwen_model_dir())
        if not (model_path.exists() and any(model_path.iterdir())):
            raise RuntimeError(f"Qwen3-VL model folder not found or empty: {model_path}")

        out, err, rc = _run_qwen_text_subprocess(
            model_path=str(model_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
        )
        raw_text = (out or "").strip()

        # Save raw (optionally append stderr without affecting JSON extraction)
        save_blob = raw_text
        err = (err or "").strip()
        if err:
            save_blob = (save_blob + "\n\n[stderr]\n" + err).strip()
        _safe_write_text(raw_path, save_blob + ("\n" if save_blob else ""))

        if rc != 0 and not raw_text:
            raise RuntimeError(f"Qwen subprocess failed (rc={rc}).")

        parsed = _extract_first_json(raw_text)
        if parsed is None:
            raise RuntimeError("Model response did not contain parseable JSON.")

        # Best-effort VRAM cleanup (mostly a no-op now, but helps if other stages use torch).
        _vram_release("after qwen json (subprocess)")
        return parsed, raw_text

    except Exception:
        # Always write whatever we have + error traceback
        try:
            if raw_text:
                _safe_write_text(raw_path, raw_text + "\n")
        except Exception:
            pass
        _safe_write_text(error_path, traceback.format_exc())
        return None, raw_text



# -----------------------------
# Pipeline runner 
# -----------------------------

class PipelineSignals(QObject):
    log = Signal(str)
    stage = Signal(str)
    progress = Signal(int)  # 0-100
    finished = Signal(dict)  # result payload
    failed = Signal(str)

    request_image_review = Signal(object)  # payload: {output_dir, manifest_path, images_dir, ...}
    request_video_review = Signal(object)  # payload: {output_dir, manifest_path, clips_dir, shots_path, images_dir}

    image_regen_started = Signal(str)      # sid
    image_regen_done = Signal(str, str)    # sid, path
    image_regen_failed = Signal(str, str)  # sid, error

    clip_regen_started = Signal(str)       # sid
    clip_regen_done = Signal(str, str)     # sid, path
    clip_regen_failed = Signal(str, str)   # sid, error

    # Live preview hook: emit the path of a newly completed image/clip as soon as it exists.
    # UI decides whether to show it (Settings -> Preview on/off).
    asset_created = Signal(str)            # path



class PipelineWorker(QThread):
    """
    threaded runner: simulates the stages and emits logs.
    Replace stage bodies with real calls later.
    """

    def __init__(self, job: PlannerJob, out_dir: str):
        super().__init__()
        self.job = job
        self.out_dir = out_dir
        self.signals = PipelineSignals()
        self._stop_requested = False
        self._last_pct = 0  # last emitted progress percent (for skip logs)
        self._review_cmd_q: "queue.Queue[dict]" = queue.Queue()
        self._in_review_gate = False


    def request_stop(self) -> None:
        self._stop_requested = True



    def post_review_command(self, cmd: Dict[str, Any]) -> None:
        """Thread-safe: UI can push review commands while the worker is paused."""
        try:
            self._review_cmd_q.put(dict(cmd or {}))
        except Exception:
            pass

    def _image_review_gate(self, *, output_dir: str, manifest_path: str, images_dir: str) -> None:
        """Pause after images and optionally allow per-shot regeneration."""
        payload = {
            "job_id": str(self.job.job_id),
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "images_dir": str(images_dir),
        }
        self._in_review_gate = True
        # Drain any stale commands from prior gates
        try:
            while True:
                self._review_cmd_q.get_nowait()
        except Exception:
            pass

        try:
            self.signals.request_image_review.emit(payload)
        except Exception:
            # If UI can't handle the gate, just continue.
            self._in_review_gate = False
            return

        # Command loop: CONTINUE / CANCEL / REGEN
        while True:
            if self._stop_requested:
                raise RuntimeError("Cancelled by user.")
            try:
                cmd = self._review_cmd_q.get(timeout=0.25)
            except Exception:
                continue

            ctype = str((cmd or {}).get("type") or "").strip().lower()
            if ctype in ("continue", "resume", "ok"):
                self._in_review_gate = False
                return
            if ctype in ("cancel", "stop", "abort"):
                self._stop_requested = True
                raise RuntimeError("Cancelled by user.")
            if ctype in ("regen", "recreate"):
                sid = str((cmd or {}).get("sid") or "").strip()
                new_prompt = str((cmd or {}).get("prompt") or "").strip()
                if not sid or not new_prompt:
                    continue
                try:
                    self.signals.image_regen_started.emit(sid)
                except Exception:
                    pass
                try:
                    seed_override = None
                    try:
                        if (cmd or {}).get("seed") is not None:
                            seed_override = int((cmd or {}).get("seed"))
                    except Exception:
                        seed_override = None

                    out_path = self._regen_one_image(
                        sid=sid,
                        new_prompt=new_prompt,
                        seed_override=seed_override,
                        manifest_path=str(manifest_path),
                        images_dir=str(images_dir),
                    )
                    try:
                        self.signals.image_regen_done.emit(sid, str(out_path))
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        self.signals.image_regen_failed.emit(sid, str(e))
                    except Exception:
                        pass
                continue


    def _clip_review_gate(self, *, output_dir: str, manifest_path: str, clips_dir: str, shots_path: str, images_dir: str) -> None:
        """Chunk 8B: pause after clips and optionally allow per-shot clip regeneration."""
        # VRAM guard: free any in-process model VRAM before opening the clip reviewer
        _vram_release("before clip review")
        payload = {
            "job_id": str(self.job.job_id),
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "clips_dir": str(clips_dir),
            "shots_path": str(shots_path),
            "images_dir": str(images_dir),
        }
        self._in_review_gate = True
        # Drain any stale commands from prior gates
        try:
            while True:
                self._review_cmd_q.get_nowait()
        except Exception:
            pass

        try:
            self.signals.request_video_review.emit(payload)
        except Exception:
            # If UI can't handle the gate, just continue.
            self._in_review_gate = False
            return

        # Command loop: CONTINUE / CANCEL / CLIP_REGEN / MARK_REVIEWED
        while True:
            if self._stop_requested:
                raise RuntimeError("Cancelled by user.")
            try:
                cmd = self._review_cmd_q.get(timeout=0.25)
            except Exception:
                continue

            ctype = str((cmd or {}).get("type") or "").strip().lower()
            if ctype in ("continue", "resume", "ok"):
                self._in_review_gate = False
                return
            if ctype in ("cancel", "stop", "abort"):
                self._stop_requested = True
                raise RuntimeError("Cancelled by user.")
            if ctype in ("mark_reviewed", "clips_reviewed", "reviewed"):
                try:
                    self._mark_clips_reviewed(manifest_path=str(manifest_path))
                except Exception:
                    pass
                continue
            if ctype in ("clip_regen", "regen_clip", "recreate_clip", "regen_video", "recreate"):
                sid = str((cmd or {}).get("sid") or "").strip()
                if not sid:
                    continue
                try:
                    self.signals.clip_regen_started.emit(sid)
                except Exception:
                    pass
                try:
                    new_prompt = str((cmd or {}).get("prompt") or "").strip()

                    seed_override = None
                    try:
                        if (cmd or {}).get("seed") is not None:
                            seed_override = int((cmd or {}).get("seed"))
                    except Exception:
                        seed_override = None

                    out_path = self._regen_one_clip(
                        sid=sid,
                        new_prompt=(new_prompt if new_prompt else None),
                        seed_override=seed_override,
                        manifest_path=str(manifest_path),
                        clips_dir=str(clips_dir),
                        shots_path=str(shots_path),
                        images_dir=str(images_dir),
                    )
                    try:
                        self.signals.clip_regen_done.emit(sid, str(out_path))
                    except Exception:
                        pass
                except Exception as e:
                    try:
                        self.signals.clip_regen_failed.emit(sid, str(e))
                    except Exception:
                        pass
                continue

    def _mark_clips_reviewed(self, *, manifest_path: str) -> None:
        """Persist a debug flag that the user entered the clip review dialog."""
        if not manifest_path:
            return
        manifest = _safe_read_json(manifest_path) or {}
        if not isinstance(manifest, dict):
            return
        manifest.setdefault("review", {})["clips_reviewed"] = True
        manifest.setdefault("review", {})["ts_clips_reviewed"] = time.time()
        _safe_write_json(manifest_path, manifest)

    def _regen_one_clip(self, *, sid: str, new_prompt: Optional[str] = None, seed_override: Optional[int] = None, manifest_path: str, clips_dir: str, shots_path: str, images_dir: str) -> str:
        """Regenerate a single shot clip in the worker thread, using the currently selected/approved image."""
        sid = str(sid or "").strip()
        if not sid:
            raise ValueError("Missing shot id (sid).")

        manifest = _safe_read_json(manifest_path) or {}
        if not isinstance(manifest, dict):
            manifest = {}

        # Resolve current clip entry + target output file
        clips_list = []
        try:
            clips_list = (manifest.get("paths") or {}).get("clips") or []
        except Exception:
            clips_list = []
        if not isinstance(clips_list, list):
            clips_list = []

        clip_entry = None
        for it in clips_list:
            if isinstance(it, dict) and str(it.get("id") or "") == sid:
                clip_entry = it
                break

        out_file = str((clip_entry or {}).get("file") or "").strip()
        if not out_file and clips_dir and os.path.isdir(clips_dir):
            try:
                cands = sorted([str(p) for p in Path(clips_dir).glob(f"*_{sid}.mp4") if p.is_file()])
                if cands:
                    out_file = cands[0]
            except Exception:
                pass
        if not out_file:
            # Fallback naming
            out_file = os.path.join(clips_dir, f"{sid}.mp4")
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        # Backup current clip as a "take" (Option A)
        take_path = ""
        try:
            if os.path.exists(out_file):
                takes_dir = os.path.join(clips_dir, "_takes")
                os.makedirs(takes_dir, exist_ok=True)
                ext0 = os.path.splitext(out_file)[1] or ".mp4"
                base = f"{self.job.job_id}_{sid}_take"
                n = 1
                while True:
                    cand = os.path.join(takes_dir, f"{base}{n:02d}{ext0}")
                    if not os.path.exists(cand):
                        take_path = cand
                        break
                    n += 1
                try:
                    import shutil
                    shutil.move(out_file, take_path)
                except Exception:
                    import shutil
                    shutil.copy2(out_file, take_path)
                    try:
                        os.remove(out_file)
                    except Exception:
                        pass
        except Exception:
            pass

        # Resolve image path for this shot (use latest approved image)
        shot_map = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
        rec = shot_map.get(sid) if isinstance(shot_map, dict) else None
        if not isinstance(rec, dict):
            rec = {}

        img_path = str(rec.get("file") or "").strip()
        if not img_path:
            try:
                imgs = (manifest.get("paths") or {}).get("images") or []
            except Exception:
                imgs = []
            if isinstance(imgs, list):
                for it in imgs:
                    if isinstance(it, dict) and str(it.get("id") or "") == sid and it.get("file"):
                        img_path = str(it.get("file"))
                        break
        if not img_path and images_dir:
            guess = os.path.join(images_dir, f"{sid}.png")
            if os.path.isfile(guess):
                img_path = guess

        if not img_path or not os.path.isfile(img_path):
            raise RuntimeError(f"Missing start image for {sid}: {img_path}")

        # Prompt/negative: allow override from review UI; otherwise reuse existing compiled i2v prompt
        prompt_override = str(new_prompt or "").strip() if new_prompt is not None else ""
        negative = str(rec.get("i2v_negative") or "").strip()
        prompt = (prompt_override or str(rec.get("i2v_prompt") or "").strip())
        if not prompt:
            prompt = "Slow camera move, subtle parallax, keep subject stable. Match the image exactly."
        if prompt_override:
            # Persist prompt override into the shot record for debugging and future reruns
            try:
                oldp = str(rec.get("i2v_prompt") or "").strip()
                if oldp and oldp != prompt_override and "i2v_prompt_original" not in rec:
                    rec["i2v_prompt_original"] = oldp
                rec["i2v_prompt"] = prompt_override
                rec["ts_i2v_prompt_edit"] = time.time()
            except Exception:
                pass

        # Pull generation settings from the clip entry and/or saved video profile
        try:
            prof = (manifest.get("settings") or {}).get("video_profile") or {}
        except Exception:
            prof = {}
        if not isinstance(prof, dict):
            prof = {}

        def _i(v, default=0):
            try:
                return int(v)
            except Exception:
                return int(default)

        fps = _i((clip_entry or {}).get("fps"), _i(prof.get("fps"), 20))
        steps = _i((clip_entry or {}).get("steps"), _i(prof.get("steps"), 10))
        max_frames = _i(prof.get("max_frames"), 61)
        target_size = _i((clip_entry or {}).get("target_size"), _i(prof.get("target_size"), 0))
        model_key = str((clip_entry or {}).get("model_key") or prof.get("model_key") or (manifest.get("settings") or {}).get("video_model") or "480p_i2v_step_distilled")

        seed = (clip_entry or {}).get("seed", None)
        if seed is None:
            try:
                seed = int(rec.get("seed_int"))
            except Exception:
                seed = None
        # Optional: retry with a fresh seed from the review UI.
        if seed_override is not None:
            try:
                seed = int(seed_override)
            except Exception:
                seed = seed_override


        frames = _i((clip_entry or {}).get("frames"), 0)
        if frames <= 0:
            # Compute from shots.json duration and clamp to profile max_frames
            dur = 0.0
            try:
                shots = _load_shots_list(shots_path) if shots_path else []
                for sh in shots:
                    if isinstance(sh, dict) and str(sh.get("id") or "") == sid:
                        try:
                            dur = float(sh.get("duration_sec") or 0.0)
                        except Exception:
                            dur = 0.0
                        break
            except Exception:
                dur = 0.0
            if dur <= 0.0:
                dur = float(max_frames) / float(max(1, fps))
            frames = int(round(float(dur) * float(max(1, fps))))
            frames = max(16, min(int(max_frames), int(frames)))

        attn_backend = str(prof.get("attn_backend") or "auto")
        cpu_offload = bool(prof.get("cpu_offload", True))
        vae_tiling = bool(prof.get("vae_tiling", True))

        # Only HunyuanVideo is wired right now (Chunk 5). Keep behavior identical.
        if _video_model_key(self.job.encoding.get("video_model") or "") != "hunyuan":
            raise RuntimeError("Clip regeneration is only supported for HunyuanVideo 1.5 in the current build.")

        # Run the exact same CLI call as initial clip generation
        py = _root() / ".hunyuan15_env" / "Scripts" / "python.exe"
        if not py.exists():
            raise RuntimeError(
                f"HunyuanVideo 1.5 environment not found: {py}\n"
                "Install it first via Tools → HunyuanVideo 1.5 → Install/Update Cuda."
            )
        cli = _root() / "helpers" / "hunyuan15_cli.py"
        if not cli.exists():
            raise RuntimeError(f"Missing CLI: {cli}")

        args = [
            str(py),
            str(cli),
            "generate",
            "--model", str(model_key),
            "--prompt", str(prompt),
            "--negative", str(negative or ""),
            "--image", str(img_path),
            "--output", str(out_file),
            "--fps", str(int(fps)),
            "--frames", str(int(frames)),
            "--steps", str(int(steps)),
            "--bitrate-kbps", str(int(2000)),
            "--auto-aspect",
        ]
        if int(target_size) > 0:
            args += ["--target-size", str(int(target_size))]
        args += ["--attn", str(attn_backend or "auto")]
        if bool(cpu_offload):
            args += ["--offload"]
        if bool(vae_tiling):
            args += ["--tiling"]
        if seed is not None:
            args += ["--seed", str(int(seed))]

        import subprocess
        # VRAM guard: make sure no old CUDA context (e.g., Qwen3-VL) is eating VRAM before regen
        _vram_release("before clip regen")
        subprocess.run(args, cwd=str(_root()), check=True)

        if not os.path.isfile(out_file) or os.path.getsize(out_file) < 1024:
            raise RuntimeError(f"Clip output missing/too small for {sid}: {out_file}")

        # Persist into manifest (per-shot + clip list)
        try:
            # Update per-shot record
            rec["clip_file"] = str(out_file)
            rec["ts_clip_done"] = time.time()
            rec["clip_regen_count"] = int(rec.get("clip_regen_count") or 0) + 1
            if take_path:
                takes = rec.get("clip_takes")
                if not isinstance(takes, list):
                    takes = []
                takes.append(str(take_path))
                rec["clip_takes"] = takes
            shot_map[sid] = rec
            manifest["shots"] = shot_map

            # Update clip list entry (keep same filename for downstream assembly)
            if clip_entry is None:
                clip_entry = {"id": sid, "file": str(out_file)}
                clips_list.append(clip_entry)
            clip_entry["file"] = str(out_file)
            clip_entry["frames"] = int(frames)
            clip_entry["fps"] = int(fps)
            clip_entry["steps"] = int(steps)
            # Persist seed override (optional)
            try:
                if seed_override is not None:
                    if clip_entry.get("seed") is not None and "seed_original" not in clip_entry:
                        clip_entry["seed_original"] = int(clip_entry.get("seed") or 0)
                    clip_entry["ts_seed_override"] = time.time()
                    try:
                        rec_seed_old = rec.get("clip_seed")
                        if rec_seed_old is not None and "clip_seed_original" not in rec:
                            rec["clip_seed_original"] = int(rec_seed_old or 0)
                    except Exception:
                        pass
                    rec["clip_seed"] = int(seed or 0) if seed is not None else None
                    rec["ts_clip_seed_override"] = time.time()
            except Exception:
                pass

            clip_entry["seed"] = seed
            clip_entry["target_size"] = int(target_size)
            clip_entry["model_key"] = str(model_key)
            clip_entry["regen_count"] = int(clip_entry.get("regen_count") or 0) + 1
            if take_path:
                tk = clip_entry.get("takes")
                if not isinstance(tk, list):
                    tk = []
                tk.append(str(take_path))
                clip_entry["takes"] = tk

            manifest.setdefault("paths", {})["clips_dir"] = str(clips_dir)
            manifest["paths"]["clips"] = clips_list
            _safe_write_json(manifest_path, manifest)
        except Exception:
            pass

        return str(out_file)

    def _regen_one_image(self, *, sid: str, new_prompt: str, seed_override: Optional[int] = None, manifest_path: str, images_dir: str) -> str:
        """Regenerate a single shot image in the worker thread and persist prompt override into manifest."""
        sid = str(sid or "").strip()
        if not sid:
            raise ValueError("Missing shot id (sid).")

        # VRAM guard: ensure prompt/story models aren't holding VRAM when regenerating an image.
        _vram_release("before image regen")

        manifest = _safe_read_json(manifest_path) or {}
        shots_map = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
        rec = shots_map.get(sid) if isinstance(shots_map, dict) else None
        if not isinstance(rec, dict):
            rec = {}

        # Resolve target file
        target = str(rec.get("file") or "").strip()
        if not target:
            # Try manifest paths list
            try:
                imgs = (manifest.get("paths") or {}).get("images") or []
            except Exception:
                imgs = []
            if isinstance(imgs, list):
                for it in imgs:
                    if isinstance(it, dict) and str(it.get("id") or "") == sid and it.get("file"):
                        target = str(it.get("file"))
                        break
        if not target:
            target = os.path.join(images_dir, f"{sid}.png")

        # Backup current as a "take"
        try:
            if os.path.exists(target):
                takes_dir = os.path.join(images_dir, "_takes")
                os.makedirs(takes_dir, exist_ok=True)
                ext0 = os.path.splitext(target)[1] or ".png"
                base = f"{self.job.job_id}_{sid}_take"
                n = 1
                while True:
                    cand = os.path.join(takes_dir, f"{base}{n:02d}{ext0}")
                    if not os.path.exists(cand):
                        break
                    n += 1
                    if n > 99:
                        break
                try:
                    shutil.move(target, cand)
                except Exception:
                    try:
                        os.replace(target, cand)
                    except Exception:
                        pass
        except Exception:
            pass

        # Persist prompt override in manifest BEFORE rendering
        try:
            if "prompt_compiled_original" not in rec:
                rec["prompt_compiled_original"] = str(rec.get("prompt_compiled") or "").strip()
            rec["prompt_user_override"] = str(new_prompt).strip()
            rec["prompt_compiled"] = str(new_prompt).strip()
            rec["prompt_used"] = str(new_prompt).strip()
            rec["ts_prompt_override"] = time.time()
            # Seed override for regen (optional)
            try:
                if seed_override is not None:
                    if "seed_int_original" not in rec and rec.get("seed_int") is not None:
                        rec["seed_int_original"] = int(rec.get("seed_int") or 0)
                    if "seed_original" not in rec and rec.get("seed") is not None:
                        rec["seed_original"] = int(_seed_to_int(str(rec.get("seed"))) or 0)
                    rec["seed_int"] = int(seed_int or 0)
                    rec["seed"] = int(seed_int or 0)
                    rec["ts_seed_override"] = time.time()
            except Exception:
                pass
            shots_map[sid] = rec
            manifest["shots"] = shots_map
            _safe_write_json(manifest_path, manifest)
            # Keep paths.images seed in sync (best effort)
            try:
                if seed_override is not None:
                    paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
                    imgs = paths.get("images") if isinstance(paths.get("images"), list) else []
                    for it in imgs:
                        if isinstance(it, dict) and str(it.get("id") or "") == sid:
                            it["seed"] = int(seed_int or 0)
                            it["ts_seed_override"] = time.time()
                            break
                    paths["images"] = imgs
                    manifest["paths"] = paths
                    _safe_write_json(manifest_path, manifest)
            except Exception:
                pass
        except Exception:
            pass

        # Build a one-shot txt2img job dict similar to step_render_all_images
        t2i_settings_path = str((_root() / "presets" / "setsave" / "txt2img.json").resolve())
        base_job = _safe_read_json(t2i_settings_path) if os.path.exists(t2i_settings_path) else {}
        if not isinstance(base_job, dict):
            base_job = {}

        negative = str(rec.get("negative_compiled") or rec.get("negative_used") or "").strip()
        if not negative:
            negative = _ascii_only(str(self.job.negatives or "").strip())

        seed_val = rec.get("seed_int")
        if seed_val is None:
            try:
                seed_val = int(rec.get("seed") or 0)
            except Exception:
                seed_val = 0
        try:
            seed_int = int(seed_val or 0)
        except Exception:
            seed_int = 0

        # If requested, retry with a fresh random seed.
        if seed_override is not None:
            try:
                seed_int = int(seed_override)
            except Exception:
                seed_int = int(seed_int or 0)

        # Determine if this shot used qwen2511 ref-strategy
        use_qwen2511 = False
        try:
            if str(rec.get("ref_strategy") or "").strip() == "qwen2511_best":
                use_qwen2511 = True
        except Exception:
            use_qwen2511 = False

        if not use_qwen2511:
            try:
                at = self.job.attachments or {}
            except Exception:
                at = {}
            try:
                ref_files = (at.get("ref_images") or []) or (at.get("images") or [])
            except Exception:
                ref_files = []
            try:
                rs = str(self.job.encoding.get("ref_strategy") or "").strip()
            except Exception:
                rs = ""
            if rs == "qwen2511_best" and bool(ref_files):
                use_qwen2511 = True

        def _pick_zimage_gguf_for_quality() -> str:
            """Pick a Z-image diffusion GGUF from models/Z-Image-Turbo GGUF based on Generation quality.
            - Low:    pick the lowest quant available (lowest Q number)
            - Medium: pick Q5 if possible; otherwise the closest ABOVE 5 (smallest Q>=5); if none, pick the highest below 5
            - High:   pick the highest quant available (highest Q number)
            Notes:
            - Only diffusion GGUFs are eligible (must look like z_image_turbo / z-image-turbo / zimage turbo).
            - Unknown quant names are de-prioritized and used only as a last resort.
            """
            try:
                gq = str(self.job.encoding.get("gen_quality_preset") or "").lower().strip()
                mode = "medium"
                if "high" in gq:
                    mode = "high"
                elif "low" in gq:
                    mode = "low"
                elif "med" in gq:
                    mode = "medium"

                gguf_dir = (_root() / "models" / "Z-Image-Turbo GGUF").resolve()
                if not gguf_dir.exists():
                    return ""

                cands = []
                unknown = []
                for p in gguf_dir.glob("**/*.gguf"):
                    try:
                        if not p.is_file():
                            continue
                        name = p.name.lower()
                        if not ("z_image_turbo" in name or "z-image-turbo" in name or ("zimage" in name and "turbo" in name)):
                            continue
                        m = re.search(r"\bQ(\d+)\b", p.name, flags=re.IGNORECASE)
                        qn = int(m.group(1)) if m else -1
                        try:
                            sz = p.stat().st_size
                        except Exception:
                            sz = 0
                        if qn < 0:
                            unknown.append((sz, str(p)))
                        else:
                            cands.append((qn, sz, str(p)))
                    except Exception:
                        continue

                if not cands:
                    if not unknown:
                        return ""
                    unknown.sort(key=lambda t: (-t[0], t[1]))
                    return unknown[0][1]

                qs = sorted({q for q, _sz, _p in cands})

                if mode == "high":
                    pick_q = max(qs)
                elif mode == "medium":
                    above = [q for q in qs if q >= 5]
                    if above:
                        pick_q = min(above)
                    else:
                        below = [q for q in qs if q < 5]
                        pick_q = max(below) if below else max(qs)
                else:
                    pick_q = min(qs)

                best = [t for t in cands if t[0] == pick_q]
                best.sort(key=lambda t: (-t[1], t[2]))
                return best[0][2]
            except Exception:
                return ""



        image_model_sel = ""
        try:
            image_model_sel = str(self.job.encoding.get("image_model") or "").strip()
        except Exception:
            image_model_sel = ""

        # Own Character Bible — final injection at dispatch-time (protect against truncation/distillers)
        try:
            if (not bool(_alt_storymode)) and bool(_own_active) and str(_own_prose or "").strip():
                _p0 = str(new_prompt or "").strip()
                if _p0 and ("OWN CHARACTER BIBLE" not in _p0) and ("Main characters (keep consistent):" not in _p0):
                    new_prompt = (_p0 + " Main characters (keep consistent): " + str(_own_prose).strip() + ".").strip()
                else:
                    new_prompt = _p0
        except Exception:
            pass

        # Auto Character Bible v2: identity anchors are appended per-shot at dispatch-time in the main render loop.
        # (Do NOT inject here; regen should only respect Own Character Bible when enabled.)


        t2i_job = dict(base_job)
        t2i_job["prompt"] = str(new_prompt).strip()

        # Probe logger: capture regen prompt (after any final injection).
        try:
            if t2i_job.get("prompt"):
                _p1 = str(t2i_job.get("prompt") or "").replace("\n", " ").strip()
                _LOGGER.log_probe(f"t2i_prompt_regen {sid}: {_p1}")
        except Exception:
            pass

        t2i_job["negative_prompt"] = negative
        t2i_job["negative"] = negative
        t2i_job["neg_prompt"] = negative
        t2i_job["seed"] = seed_int
        t2i_job["batch"] = 1
        t2i_job["output"] = images_dir


        aspect_mode = str(self.job.encoding.get("aspect_mode") or "landscape")
        _eng = str(t2i_job.get("engine") or manifest.get("settings", {}).get("image_engine") or "").lower().strip()
        _sel = (image_model_sel or "").lower().strip()
        if _sel.startswith("auto"):
            pass
        elif "z-image" in _sel and "low" in _sel:
            _eng = "zimage_gguf"
        elif "z-image" in _sel:
            _eng = "zimage"
        elif "qwen" in _sel:
            _eng = "qwen2512"
        elif "sdxl" in _sel:
            _eng = "diffusers"
        if not _eng:
            _eng = "zimage_gguf"
        t2i_job["engine"] = _eng

        if "qwen" in _eng:
            bw, bh = 1344, 768
            ww, hh = _apply_aspect_to_size(bw, bh, aspect_mode)
            t2i_job["width"] = int(ww)
            t2i_job["height"] = int(hh)
            t2i_job["sampler"] = "Euler"
            t2i_job["steps"] = 9
            t2i_job["cfg_scale"] = 2.5
            t2i_job["cfg"] = 2.5
        elif "zimage" in _eng:
            bw, bh = 1344, 768
            ww, hh = _apply_aspect_to_size(bw, bh, aspect_mode)
            t2i_job["width"] = int(ww)
            t2i_job["height"] = int(hh)
            t2i_job["steps"] = 10
            t2i_job["cfg_scale"] = 0.0
            t2i_job["cfg"] = 0.0
            if _eng == "zimage_gguf":
                gguf_path = _pick_zimage_gguf_for_quality()
                if gguf_path:
                    t2i_job["lora_path"] = gguf_path
                else:
                    raise RuntimeError("Z-image Low VRAM selected but no diffusion .gguf was found in models/Z-Image-Turbo GGUF")
        else:
            try:
                w = int(t2i_job.get("width") or 0)
                h = int(t2i_job.get("height") or 0)
            except Exception:
                w = 0
                h = 0
            if w <= 0 or h <= 0:
                bw, bh = 1920, 1080
                ww, hh = _apply_aspect_to_size(bw, bh, aspect_mode)
                t2i_job["width"] = int(ww)
                t2i_job["height"] = int(hh)

        out_file = None
        res = None

        if use_qwen2511:
            try:
                from helpers import qwen2511 as _q2511  # type: ignore
            except Exception:
                import qwen2511 as _q2511  # type: ignore

            try:
                at = self.job.attachments or {}
            except Exception:
                at = {}
            try:
                ref_files = (at.get("ref_images") or []) or (at.get("images") or [])
            except Exception:
                ref_files = []
            refs = list(ref_files[:2])

            aspect_mode = str(self.job.encoding.get("aspect_mode") or "")
            _hq = bool(self.job.encoding.get("qwen2511_high_quality") or False)

            bw, bh = (1280, 720) if _hq else (1024, 576)

            qw, qh = _apply_aspect_to_size(bw, bh, aspect_mode)
            q_job = {
                "prompt": str(new_prompt).strip(),
                "negative_prompt": negative,
                "negative": negative,
                "seed": seed_int,
                "width": int(qw),
                "height": int(qh),
                "vae_device": "cpu",
                "vae_on_cpu": True,
                "vae_cpu": True,
                "use_vae_on_cpu": True,
                "refs": refs,
                "ref_images": refs,
                "batch": 1,
                "out_file": target,
            }

            try:
                if hasattr(_q2511, "build_sdcli_cmd") and hasattr(_q2511, "detect_sdcli_caps"):
                    sdcli_path = getattr(_q2511, "DEFAULT_SDCLI", "sd-cli.exe")
                    try:
                        sp = getattr(_q2511, "SETSAVE_PATH", "")
                        if sp and hasattr(_q2511, "_read_jsonish"):
                            _s = _q2511._read_jsonish(sp)
                            if isinstance(_s, dict) and _s.get("sdcli_path"):
                                sdcli_path = str(_s.get("sdcli_path"))
                    except Exception:
                        pass

                    caps = _q2511.detect_sdcli_caps(sdcli_path)
                    try:
                        d = _q2511.default_model_paths() if hasattr(_q2511, "default_model_paths") else {}
                    except Exception:
                        d = {}
                    unet_path = str(d.get("unet") or "")
                    llm_path = str(d.get("llm") or "")
                    mmproj_path = str(d.get("mmproj") or "")
                    vae_path = str(d.get("vae") or "")
                    # Qwen Edit 2511 UNet GGUF auto-pick (prefer Q5, then Q4, then any).
                    # Models are expected under: <root>/models/qwen2511gguf/unet/*.gguf
                    try:
                        _unet_dir = _root() / "models" / "qwen2511gguf" / "unet"
                        _ggufs = []
                        try:
                            if _unet_dir.exists():
                                _ggufs = [str(x) for x in _unet_dir.glob("*.gguf") if x.is_file()]
                        except Exception:
                            _ggufs = []
                        if _ggufs:
                            def _qwen2511_prio(_p: str):
                                _n = os.path.basename(_p).lower()
                                # Common names: ...-Q5_K_M.gguf / ...-Q4_K_M.gguf
                                if re.search(r"(?:^|[-_\s])q5(?:$|[-_\s])", _n) or ("q5" in _n):
                                    return (0, _n)
                                if re.search(r"(?:^|[-_\s])q4(?:$|[-_\s])", _n) or ("q4" in _n):
                                    return (1, _n)
                                return (2, _n)
                            _ggufs = sorted(_ggufs, key=_qwen2511_prio)
                            unet_path = str(_ggufs[0])
                        else:
                            # If helper defaults didn't provide a valid UNet either, hard fail with a clear popup.
                            if not (unet_path and os.path.exists(unet_path)):
                                QMessageBox.warning(
                                    self,
                                    "Qwen Edit 2511 missing model",
                                    "Qwen Edit 2511 cannot run because no UNet GGUF was found.\n\n"
                                    "Please download at least one model to:\n"
                                    "models/qwen2511gguf/unet/\n\n"
                                    "Recommended: qwen-image-edit-2511-Q5_K_M.gguf"
                                )
                                raise RuntimeError("Qwen2511: no UNet GGUF found in models/qwen2511gguf/unet")
                    except Exception:
                        # Let the caller decide whether to fall back; we only guarantee we don't crash here.
                        pass

                    tmp_blank = os.path.join(images_dir, f"_blank_{self.job.job_id}_{sid}_{int(time.time()*1000)}.png")

                    try:
                        if hasattr(_q2511, "_write_blank_png"):
                                            _q2511._write_blank_png(tmp_blank, int(q_job.get("width") or 1024), int(q_job.get("height") or 576))
                    except Exception:
                        tmp_blank = ""

                    cmd = _q2511.build_sdcli_cmd(
                        sdcli_path=sdcli_path,
                        caps=caps,
                        init_img=tmp_blank,
                        mask_path="",
                        ref_images=refs,
                        use_increase_ref_index=True,
                        disable_auto_resize_ref_images=False,
                        prompt=str(q_job.get("prompt") or ""),
                        negative=str(q_job.get("negative_prompt") or q_job.get("negative") or ""),
                        unet_path=unet_path,
                        llm_path=llm_path,
                        mmproj_path=mmproj_path,
                        vae_path=vae_path,
                        steps=int(q_job.get("steps") or 20),
                        cfg=float(q_job.get("cfg") or 4.0),
                        seed=int(q_job.get("seed") or 0),
                        width=int(q_job.get("width") or 1024),

                                        height=int(q_job.get("height") or 576),
                        strength=1.0,
                        sampling_method="euler",
                        shift=2.3,
                        out_file=str(q_job.get("out_file") or ""),
                        use_vae_tiling=False,
                        vae_tile_size="",
                        vae_tile_overlap=0.0,
                        use_offload=False,
                        use_mmap=False,
                        use_vae_on_cpu=True,
                        use_clip_on_cpu=False,
                        use_diffusion_fa=True,
                        lora_model_dir="",
                        lora_name=str(q_job.get("lora") or q_job.get("lora_path") or ""),
                        lora_strength=float(q_job.get("lora_strength") or q_job.get("lora_scale") or 1.0),
                    )

                    rc = 1
                    try:
                        if hasattr(_q2511, "_run_capture"):
                            rc, _out_text = _q2511._run_capture(cmd)
                        else:
                            rc = os.system(" ".join([str(x) for x in cmd]))
                    except Exception:
                        rc = 1

                    ok = (rc == 0 and os.path.exists(target))
                    res = {"ok": bool(ok), "files": ([target] if ok else []), "out_file": target, "rc": int(rc)}

                    try:
                        if tmp_blank and os.path.exists(tmp_blank):
                            os.remove(tmp_blank)
                    except Exception:
                        pass
                else:
                    raise RuntimeError("sd-cli builder not available")
            except Exception:
                if hasattr(_q2511, "generate_one_from_job"):
                    res = _q2511.generate_one_from_job(q_job, images_dir)
                elif hasattr(_q2511, "run_job"):
                    res = _q2511.run_job(q_job, images_dir)
                elif hasattr(_q2511, "run"):
                    res = _q2511.run(q_job, images_dir)
                else:
                    raise RuntimeError("qwen2511 entrypoint not found")

            out_file = target if os.path.exists(target) else None

        else:
            try:
                from helpers import txt2img as _t2i  # type: ignore
            except Exception:
                import txt2img as _t2i  # type: ignore

            if not hasattr(_t2i, "generate_one_from_job"):
                raise RuntimeError("txt2img generator entrypoint not found (generate_one_from_job)")

            _t2i_job = dict(t2i_job)

            # For SDXL/Diffusers we WANT deterministic filenames and we DO support filename_template/format.
            # (Some older backends crash on unknown keys, so we only strip these for non-diffusers engines.)
            try:
                _eng0 = str(_t2i_job.get("engine") or "").lower().strip()
            except Exception:
                _eng0 = ""
            if "diffusers" in _eng0:
                # Write directly into the Planner images_dir as {sid}.png
                _t2i_job["filename_template"] = f"{sid}.png"
                _t2i_job["format"] = "png"
            else:
                for _k in ("filename_template", "format"):
                    _t2i_job.pop(_k, None)

            # Some txt2img backends forward cancel_event into diffusers internals;
            # older builds of _gen_via_diffusers don't accept it. Planner already
            # supports cancellation via self._stop_requested, so strip it here.
            _t2i_job.pop("cancel_event", None)

            _max_strips = 64
            _stripped = []
            while True:
                if self._stop_requested:
                    raise RuntimeError("Cancelled by user.")
                try:
                    res = _t2i.generate_one_from_job(_t2i_job, images_dir)
                    break
                except TypeError as _te:
                    _msg = str(_te)
                    _m = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", _msg)
                    if _m:
                        _bad = _m.group(1)
                        if _bad in _t2i_job and len(_stripped) < _max_strips:
                            _t2i_job.pop(_bad, None)
                            _stripped.append(_bad)
                            try:
                                self.signals.log.emit(f"[IMG] backend stripped unsupported key: {_bad}")
                            except Exception:
                                pass
                            continue
                    raise

            try:
                if isinstance(res, dict) and res.get("files"):
                    out_file = res["files"][0]
            except Exception:
                out_file = None

            if not out_file or not os.path.exists(out_file):
                err = ""
                try:
                    if isinstance(res, dict):
                        err = str(res.get("error") or res.get("err") or res.get("message") or "")
                except Exception:
                    err = ""
                raise RuntimeError(err or "generator returned no output file")

            try:
                ext = os.path.splitext(out_file)[1] or ".png"
                target2 = os.path.join(images_dir, f"{sid}{ext}")
                if os.path.abspath(out_file) != os.path.abspath(target2):
                    try:
                        os.replace(out_file, target2)
                    except Exception:
                        try:
                            if os.path.exists(target2):
                                os.remove(target2)
                        except Exception:
                            pass
                        shutil.move(out_file, target2)
                target = target2
                out_file = target2
            except Exception:
                target = out_file

        if not (out_file and os.path.exists(out_file)):
            raise RuntimeError("Regeneration produced no output file.")

        try:
            shots_map = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
            rec2 = shots_map.get(sid) if isinstance(shots_map, dict) else None
            if not isinstance(rec2, dict):
                rec2 = {}
            rec2["file"] = str(out_file)
            rec2["status"] = "done"
            rec2["ts_done"] = time.time()
            shots_map[sid] = rec2
            manifest["shots"] = shots_map

            try:
                imgs = (manifest.get("paths") or {}).get("images") or []
            except Exception:
                imgs = []
            if isinstance(imgs, list):
                found = False
                for it in imgs:
                    if isinstance(it, dict) and str(it.get("id") or "") == sid:
                        it["file"] = str(out_file)
                        found = True
                if not found:
                    imgs.append({"id": sid, "file": str(out_file), "seed": seed_int})
                manifest.setdefault("paths", {})["images"] = imgs

            if manifest.get("paths", {}).get("first_image") and not os.path.exists(str(manifest["paths"]["first_image"])):
                manifest["paths"]["first_image"] = str(out_file)
            elif not manifest.get("paths", {}).get("first_image"):
                manifest.setdefault("paths", {})["first_image"] = str(out_file)

            _safe_write_json(manifest_path, manifest)
        except Exception:
            pass

        return str(out_file)

    def _tick(self, msg: str, pct: int, sleep_s: float = 0.25) -> None:
        if self._stop_requested:
            raise RuntimeError("Cancelled by user.")
        self.signals.log.emit(msg)
        try:
            _LOGGER.log_job(self.job.job_id, msg)
            _LOGGER.log_probe(f"job:{self.job.job_id} {msg}")
        except Exception:
            pass
        self._last_pct = int(pct)
        self.signals.progress.emit(pct)
        time.sleep(sleep_s)


    def _maybe_rename_final_cut_pretty(self, *, output_dir: str, manifest_path: str) -> None:
        """Rename final/final_cut.mp4 to a user-friendly name, while keeping a stable alias.

        Naming: <prefix(<=15)>_<YYYY-MM-DD>_<NN>.mp4
        Where prefix is derived from the job folder name.
        """
        try:
            job_dir_name = os.path.basename(os.path.normpath(str(output_dir or ""))) or "job"
            # prefix: first 15 chars, sanitize for Windows filenames
            prefix = job_dir_name.strip()[:15] or "job"
            prefix = re.sub(r"[\\/:*?\"<>|]", "_", prefix)
            prefix = re.sub(r"\s+", "_", prefix).strip("_ ") or "job"

            final_dir = os.path.join(str(output_dir), "final")
            stable_path = os.path.join(final_dir, "final_cut.mp4")
            if (not os.path.isdir(final_dir)) or (not os.path.exists(stable_path)):
                return
            try:
                if os.path.getsize(stable_path) <= 1024:
                    return
            except Exception:
                return

            date_str = datetime.datetime.now().strftime("%Y-%m-%d")
            n = 1
            new_path = ""
            while n < 1000:
                cand = os.path.join(final_dir, f"{prefix}_{date_str}_{n:02d}.mp4")
                if not os.path.exists(cand):
                    new_path = cand
                    break
                n += 1
            if not new_path:
                return

            # If it already looks renamed, do nothing
            try:
                base = os.path.basename(stable_path).lower()
                if base != "final_cut.mp4":
                    return
            except Exception:
                pass

            # Move stable -> pretty name (atomic rename where possible)
            try:
                os.replace(stable_path, new_path)
            except Exception:
                try:
                    shutil.move(stable_path, new_path)
                except Exception:
                    return

            # Keep a stable alias for downstream tools (best effort)
            try:
                shutil.copy2(new_path, stable_path)
                # Prefer the pretty file in UI scans: make alias slightly older
                try:
                    ts = os.path.getmtime(new_path)
                    os.utime(stable_path, (max(0.0, ts - 2.0), max(0.0, ts - 2.0)))
                except Exception:
                    pass
            except Exception:
                pass

            # Update manifest with a pointer (non-breaking)
            try:
                man = _safe_read_json(str(manifest_path)) or {}
                man.setdefault("paths", {})["final_named_video"] = str(new_path)
                _safe_write_json(str(manifest_path), man)
            except Exception:
                pass
        except Exception:
            return

    def run(self) -> None:
        try:
            os.makedirs(self.out_dir, exist_ok=True)

            # NOTE:
            # Plan + Shots are now real (Qwen3-VL JSON), so we drive progress from the
            # stepwise artifact pipeline below and avoid misleading "not wired" messages.

            self.signals.stage.emit("Starting")
            self.signals.log.emit(f"Job: {self.job.job_id}")

            # Alternative storymode (direct Qwen prompt list for txt2img)
            try:
                _alt_storymode = bool((self.job.encoding or {}).get("alternative_storymode"))
            except Exception:
                _alt_storymode = False
            if bool(_alt_storymode):
                try:
                    self.signals.log.emit("[RUN] Alternative storymode: direct Qwen prompt list")
                except Exception:
                    pass

            # info: Chunk 10 side quest — Own storyline mode (Step 3: pipeline binding)
            try:
                _own_storyline_enabled = bool((self.job.encoding or {}).get("own_storyline_enabled"))
            except Exception:
                _own_storyline_enabled = False

            _own_storyline_prompts: List[Dict[str, Any]] = []
            _own_storyline_parser_mode = ""
            _own_storyline_digest = ""

            if bool(_own_storyline_enabled):
                try:
                    _own_storyline_prompts = (self.job.encoding or {}).get("own_storyline_prompts") or []
                except Exception:
                    _own_storyline_prompts = []
                if not isinstance(_own_storyline_prompts, list):
                    _own_storyline_prompts = []
                if not _own_storyline_prompts:
                    try:
                        _txt = str((self.job.encoding or {}).get("own_storyline_text") or "")
                    except Exception:
                        _txt = ""
                    try:
                        _own_storyline_prompts, _own_storyline_parser_mode = _parse_own_storyline_prompts(_txt)
                    except Exception:
                        _own_storyline_prompts, _own_storyline_parser_mode = [], "paragraph"

                try:
                    _n0 = len(_own_storyline_prompts) if isinstance(_own_storyline_prompts, list) else 0
                except Exception:
                    _n0 = 0
                if int(_n0) <= 0:
                    raise RuntimeError(
                        "Own storyline is enabled but no prompts were detected. "
                        "Add markers like [01] [02] or (01) (02) so the planner can split your text into prompts."
                    )


                try:
                    _own_storyline_digest = _sha1_text(json.dumps(_own_storyline_prompts, sort_keys=True, ensure_ascii=False))
                except Exception:
                    _own_storyline_digest = ""

                try:
                    self.signals.log.emit(f"[RUN] Own storyline: using {_n0} prompts")
                except Exception:
                    pass

                # Own storyline supersedes Alternative storymode
                _alt_storymode = False


            try:
                _LOGGER.log_probe(f"Pipeline start: {self.job.job_id}")
                _LOGGER.log_job(self.job.job_id, "Pipeline start")
            except Exception:
                pass
            self.signals.log.emit(f"Project root: {_root()}")
            self.signals.log.emit(f"Output directory: {self.out_dir}")
            self.signals.log.emit("----")

            # Local log helper used by nested step functions
            def _log(msg: str) -> None:
                try:
                    self.signals.log.emit(str(msg))
                except Exception:
                    pass
                try:
                    _LOGGER.log_job(self.job.job_id, str(msg))
                    _LOGGER.log_probe(f"job:{self.job.job_id} {msg}")
                except Exception:
                    pass
            self.signals.progress.emit(0)
            # -----------------------------
            # Stepwise, idempotent artifact pipeline 
            # Each step writes its own output. Future wiring simply replaces a step body.
            # -----------------------------
            story_dir = os.path.join(self.out_dir, "story")
            prompts_dir = os.path.join(self.out_dir, "prompts")
            audio_dir = os.path.join(self.out_dir, "audio")
            images_dir = os.path.join(self.out_dir, "images")
            clips_dir = os.path.join(self.out_dir, "clips")
            final_dir = os.path.join(self.out_dir, "final")
            errors_dir = os.path.join(self.out_dir, "errors")

            for d in (story_dir, prompts_dir, audio_dir, images_dir, clips_dir, final_dir, errors_dir):
                os.makedirs(d, exist_ok=True)

            manifest_path = os.path.join(self.out_dir, "manifest.json")
            readme_path = os.path.join(self.out_dir, "README.txt")

            # Inputs fingerprint for this job
            input_fingerprint = _sha1_text(json.dumps({
                "prompt": self.job.prompt,
                "extra_info": self.job.extra_info,
"duration": self.job.approx_duration_sec,
                "resolution": self.job.resolution_preset,
                "quality": self.job.quality_preset,
                "storytelling": self.job.storytelling,
                "music_background": self.job.music_background,
                "silent": self.job.silent,
                "storytelling_volume": self.job.storytelling_volume,
                "music_volume": self.job.music_volume,
                "image_model": self.job.encoding.get("image_model"),
                "video_model": self.job.encoding.get("video_model"),
                "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                "generation_profile": self.job.encoding.get("generation_profile"),
                "image_model": self.job.encoding.get("image_model"),
                "video_model": self.job.encoding.get("video_model"),
                "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
            }, sort_keys=True))

            manifest = _safe_read_json(manifest_path) or {}
            manifest.setdefault("job_id", self.job.job_id)
            manifest["project_root"] = str(_root())
            manifest["output_dir"] = self.out_dir
            # Chunk 8C1: Human title + slug (folder-friendly) + job config pointer
            try:
                _t = str((self.job.encoding or {}).get("title") or _auto_title_from_prompt(self.job.prompt))
            except Exception:
                _t = _auto_title_from_prompt(self.job.prompt)
            try:
                _s = str((self.job.encoding or {}).get("slug") or _slugify_title(_t) or "job")
            except Exception:
                _s = _slugify_title(_t) or "job"

            manifest.setdefault("title", _t)
            manifest.setdefault("slug", _s)
            manifest.setdefault("job_config_file", "job.json")
            manifest["input_fingerprint"] = input_fingerprint
            manifest.setdefault("steps", {})
            manifest.setdefault("paths", {})
            manifest.setdefault("settings", {})
            manifest["settings"].update({
                "duration_sec": self.job.approx_duration_sec,
                "resolution_preset": self.job.resolution_preset,
                "quality_preset": self.job.quality_preset,
                "storytelling": self.job.storytelling,
                "music_background": self.job.music_background,
                "silent": self.job.silent,
                "storytelling_volume": self.job.storytelling_volume,
                "music_volume": self.job.music_volume,
            })
            manifest["paths"].update({
                "story_dir": story_dir,
                "prompts_dir": prompts_dir,
                "audio_dir": audio_dir,
                "images_dir": images_dir,
                "clips_dir": clips_dir,
                "final_dir": final_dir,
                "errors_dir": errors_dir,
            })

            # Chunk 8C1: Persist full job config for future Resume/Repair
            try:
                job_cfg_path = os.path.join(self.out_dir, "job.json")
                job_cfg = asdict(self.job)
                job_cfg.update({
                    "title": manifest.get("title"),
                    "slug": manifest.get("slug"),
                    "project_root": str(_root()),
                    "output_dir": self.out_dir,
                    "saved_at": time.time(),
                })
                _safe_write_json(job_cfg_path, job_cfg)
            except Exception as e:
                manifest.setdefault("errors", []).append({"stage": "job_config_write", "error": str(e)})
            # -----------------------------
            # Music attachment handling 
            # - Existing music (file): copy into audio/ and write Whisper
            # - New music (HeartMula or ace step): store  notes and mark pending
            # -----------------------------
            try:
                at = self.job.attachments or {}
            except Exception:
                at = {}

            music_files = []
            music_new = []
            try:
                music_files = at.get("music") or []
            except Exception:
                music_files = []
            try:
                music_new = at.get("music_new") or []
            except Exception:
                music_new = []

            if music_files:
                src_music = music_files[0]
                try:
                    ext = os.path.splitext(src_music)[1] or ".mp3"
                except Exception:
                    ext = ".mp3"
                dst_music = os.path.join(audio_dir, f"music{ext}")
                try:
                    import shutil as _shutil
                    _shutil.copy2(src_music, dst_music)
                except Exception as e:
                    manifest.setdefault("errors", []).append({"stage": "music_copy", "error": str(e), "src": src_music})
                manifest.setdefault("project", {}).setdefault("audio", {})
                manifest["project"]["audio"].update({
                    "source": "ready_made_file",
                    "src_path": src_music,
                    "copied_path": dst_music,
                })
                manifest.setdefault("paths", {})["music_file"] = dst_music

                # Whisper
                whisper_meta_path = os.path.join(audio_dir, "whisper_meta.json")
                whisper_segments_path = os.path.join(audio_dir, "whisper_segments.json")
                transcript_path_placeholder = os.path.join(audio_dir, "transcript.txt")
                _safe_write_json(whisper_meta_path, {"status": "pending", "source_audio": os.path.basename(dst_music), "word_count": 0})
                _safe_write_json(whisper_segments_path, [])
                if not _file_ok(transcript_path_placeholder, 2):
                    _safe_write_text(transcript_path_placeholder, "PENDING (transcription not generated yet)\n")
                manifest["paths"]["whisper_meta_json"] = whisper_meta_path
                manifest["paths"]["whisper_segments_json"] = whisper_segments_path
                manifest["paths"]["transcript_txt"] = transcript_path_placeholder
                try:
                    if _LOGGER.enabled:
                        _LOGGER.log_job(f"[music] ready-made copied: {dst_music}")
                except Exception:
                    pass

            elif bool(getattr(self.job, "music_background", False)) and str(getattr(self.job, "music_mode", "") or "") in ("ace", "ace15"):
                # Ace-Step background music (legacy "ace" or new "ace15"): reserve slot; generation runs after assembly (duration locked).
                mm = str(getattr(self.job, "music_mode", "") or "")
                manifest.setdefault("project", {}).setdefault("audio", {})
                manifest["project"]["audio"].update({
                    "source": ("ace_step_15" if mm == "ace15" else "ace_step"),
                    "status": "pending_generation",
                    # Keep legacy field for old jobs; new jobs store ace15_preset_id too.
                    "preset": str(getattr(self.job, "music_preset", "") or "Cinematic"),
                    "ace15_preset_id": str(getattr(self.job, "ace15_preset_id", "") or ""),
                    "ace15_lyrics_enabled": bool(getattr(self.job, "ace15_lyrics_enabled", False)),
                })
                notes_path = os.path.join(audio_dir, "music_pending.txt")
                if not _file_ok(notes_path, 2):
                    _safe_write_text(
                        notes_path,
                        "PENDING: Auto music background (Ace-Step) will be generated after visuals are assembled.\n"
                    )
                manifest.setdefault("paths", {})["music_pending_txt"] = notes_path
                try:
                    if _LOGGER.enabled:
                        _LOGGER.log_job("[music] Ace-Step selected (pending generation)")
                except Exception:
                    pass

            elif str(getattr(self.job, "music_mode", "") or "") == "heartmula":
                # HeartMula music with lyrics (Chunk 7B): reserve slot; generation runs after assembly (duration locked).
                manifest.setdefault("project", {}).setdefault("audio", {})
                manifest["project"]["audio"].update({
                    "source": "heartmula",
                    "status": "pending_generation",
                })
                notes_path = os.path.join(audio_dir, "heartmula_pending.txt")
                if not _file_ok(notes_path, 2):
                    _safe_write_text(
                        notes_path,
                        "PENDING: Music with lyrics (HeartMula) will be generated after visuals are assembled.\n"
                    )
                manifest.setdefault("paths", {})["heartmula_pending_txt"] = notes_path
                try:
                    if _LOGGER.enabled:
                        _LOGGER.log_job("[music] HeartMula selected (pending generation)")
                except Exception:
                    pass

            elif music_new:
                # HeartMula 
                manifest.setdefault("project", {}).setdefault("audio", {})
                manifest["project"]["audio"].update({
                    "source": "heartmula",
                    "status": "pending_generation",
                })
                notes_path = os.path.join(audio_dir, "heartmula_notes.txt")
                if not _file_ok(notes_path, 2):
                    _safe_write_text(
                        notes_path,
                        "HeartMula\n"
                        "- Lyrics music will be generated later in the pipeline once duration is known.\n"
                        "- This note records the chosen audio source in the manifest.\n"
                    )
                manifest.setdefault("paths", {})["heartmula_notes_txt"] = notes_path
                try:
                    if _LOGGER.enabled:
                        _LOGGER.log_job("[music] HeartMula selected")
                except Exception:
                    pass

            _safe_write_json(manifest_path, manifest)



            # -----------------------------
            # Chunk 4: Reference images handling (copy into project so runs are portable)
            # -----------------------------
            try:
                at = self.job.attachments or {}
            except Exception:
                at = {}

            try:
                ref_strategy = str(self.job.encoding.get("ref_strategy") or "").strip()
            except Exception:
                ref_strategy = ""

            try:
                ref_lora = str(self.job.encoding.get("ref_multi_angle_lora") or "").strip()
            except Exception:
                ref_lora = ""

            ref_files = []
            try:
                ref_files = (at.get("ref_images") or []) or (at.get("images") or [])
            except Exception:
                ref_files = []

            if ref_files:
                refs_dir = os.path.join(story_dir, "refs")
                os.makedirs(refs_dir, exist_ok=True)
                copied_paths = []
                copy_map = []
                import shutil as _shutil

                for idx, src in enumerate(ref_files, start=1):
                    ext = os.path.splitext(src)[1] or ".png"
                    dst = os.path.join(refs_dir, f"ref_{idx:02d}{ext}")
                    ok = False
                    try:
                        _shutil.copy2(src, dst)
                        ok = True
                    except Exception:
                        ok = False
                    if ok:
                        copied_paths.append(dst)
                    copy_map.append({"src": src, "dst": dst if ok else "", "ok": ok})

                manifest.setdefault("project", {}).setdefault("references", {})
                manifest["project"]["references"].update({
                    "strategy": ref_strategy or "unspecified",
                    "ref_count": len(ref_files),
                    "src_files": list(ref_files),
                    "copied_files": list(copied_paths),
                    "qwen2511_max_refs": 2,
                    "multi_angle_lora": ref_lora,
                    "copy_map": copy_map,
                })
                manifest.setdefault("paths", {})["ref_images_dir"] = refs_dir
                _safe_write_json(manifest_path, manifest)

                try:
                    if _LOGGER.enabled:
                        _LOGGER.log_job(f"[refs] strategy={ref_strategy or 'unspecified'} count={len(ref_files)}")
                except Exception:
                    pass

            def _set_step(name: str, status: str, note: str = "") -> None:
                cur = manifest["steps"].get(name) or {}
                cur.update({
                    "status": status,
                    "ts": time.time(),
                    "note": note,
                })
                manifest["steps"][name] = cur
                _safe_write_json(manifest_path, manifest)

            def _skip(name: str, why: str) -> None:
                self._tick(f"[SKIP] {name}: {why}", self._last_pct, 0.05)
                _set_step(name, "skipped", why)

            def _run(name: str, fn, pct: int) -> None:
                if self._stop_requested:
                    raise RuntimeError("Cancelled by user.")
                self.signals.stage.emit(name)
                self._tick(f"[RUN] {name}", pct, 0.05)
                _set_step(name, "running")
                try:
                    fn()
                    # Don't overwrite a step that marked itself as failed/stale.
                    cur = manifest["steps"].get(name) or {}
                    if cur.get("status") == "running":
                        _set_step(name, "done")
                except Exception as e:
                    err_path = os.path.join(errors_dir, f"{name.replace(' ', '_').lower()}.txt")
                    _safe_write_text(err_path, str(e))
                    _set_step(name, "failed", str(e))
                    raise
                finally:
                    # Upscale cleanup: always remove temporary frame unpack/repack folders
                    # (success, failure, or cancellation).
                    try:
                        lname = str(name).lower()
                        if "upscale" in lname:
                            tmp_w = os.path.join(final_dir, "_upscale_work")
                            if os.path.isdir(tmp_w):
                                shutil.rmtree(tmp_w, ignore_errors=True)
                        if "interpolate" in lname:
                            tmp_w = os.path.join(final_dir, "_interp_work")
                            if os.path.isdir(tmp_w):
                                shutil.rmtree(tmp_w, ignore_errors=True)
                    except Exception:
                        pass

            # Step 0: Whisper auto transcript (Chunk 3)
            # Tail progress helper: default to 99 until tail targets are computed later.
            _tail_pct_fn = lambda *a, **k: 99

            def _count_words(_s: str) -> int:
                try:
                    import re as _re
                    return len(_re.findall(r"[A-Za-z0-9']+", _s or ""))
                except Exception:
                    return 0

            def _file_fingerprint(_path: str) -> Dict[str, Any]:
                try:
                    st = os.stat(_path)
                    return {
                        "name": os.path.basename(_path),
                        "size": int(getattr(st, "st_size", 0) or 0),
                        "mtime": int(getattr(st, "st_mtime", 0) or 0),
                    }
                except Exception:
                    return {}

            def _find_whisper_model_dir() -> Optional[str]:
                r = str(_root())
                cands = [
                    os.path.join(r, "models", "whisper", "medium"),
                    os.path.join(r, "models", "whisper"),
                    os.path.join(r, "models", "faster_whisper", "medium"),
                    os.path.join(r, "models", "faster_whisper"),
                ]
                for c in cands:
                    try:
                        if os.path.isdir(c):
                            # Must contain at least one file (or subdir with files)
                            if any(os.scandir(c)):
                                return c
                    except Exception:
                        continue
                return None

            def step_whisper_auto() -> None:
                # NOTE: No UI options here. Whisper "just happens" when ready-made music is present.
                music_path = (manifest.get("paths", {}) or {}).get("music_file") or ""
                if not music_path or not os.path.isfile(music_path):
                    raise RuntimeError("Whisper: missing ready-made music file in audio/.")

                audio_fp = _file_fingerprint(music_path)
                manifest.setdefault("project", {}).setdefault("audio", {})
                manifest["project"]["audio"]["file_fingerprint"] = audio_fp

                # Import helper (kept in its own venv + subprocess to avoid DLL conflicts)
                try:
                    from helpers import whisper as _wh  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"Whisper: cannot import helpers/whisper.py: {e}")

                env_py = None
                try:
                    env_py = _wh._whisper_env_python()
                except Exception:
                    env_py = None
                if not env_py or not os.path.isfile(str(env_py)):
                    raise RuntimeError(
                        "Whisper environment not found. Expected: environments/.whisper. "
                        "Install Whisper via Optional Installs."
                    )

                runner = _wh._ensure_whisper_runner_file()

                model_dir = _find_whisper_model_dir()
                if not model_dir:
                    raise RuntimeError(
                        "Whisper model folder not found. Expected /models/whisper/ (preferred) "
                        "or /models/faster_whisper/medium/."
                    )

                try:
                    device = _wh._guess_device()
                except Exception:
                    device = "cpu"
                compute_type = "float16" if device == "cuda" else "int8"

                ffprobe_path = None
                try:
                    ffprobe_path = _wh._find_binary("ffprobe")
                except Exception:
                    ffprobe_path = None

                out_temp = os.path.join(str(_root()), "output", "_temp")
                os.makedirs(out_temp, exist_ok=True)

                payload = {
                    "root": str(_root()),
                    "media_path": str(music_path),
                    "model_dir": str(model_dir),
                    "device": device,
                    "compute_type": compute_type,
                    "language": "auto",
                    "task": "transcribe",
                    "ffprobe_path": str(ffprobe_path) if ffprobe_path else "",
                }

                payload_file = os.path.join(out_temp, f"_whisper_payload_{int(time.time()*1000)}.json")
                _safe_write_text(payload_file, json.dumps(payload, indent=2, ensure_ascii=False))

                cmd = [str(env_py), str(runner), str(payload_file)]
                self.signals.log.emit("[whisper] starting transcription (auto)")
                try:
                    _LOGGER.log_job(self.job.job_id, "[whisper] starting transcription (auto)")
                except Exception:
                    pass

                import subprocess

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(_root()),
                    text=True,
                    bufsize=1,
                )

                last_result = None
                base = max(0, int(self._last_pct))
                span = 6  # we occupy a small progress window before Plan
                if proc.stdout:
                    for line in proc.stdout:
                        if self._stop_requested:
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            raise RuntimeError("Cancelled by user.")
                        line = (line or "").strip()
                        if not line:
                            continue
                        # runner uses JSON-lines protocol
                        if line.startswith("{") and line.endswith("}"):
                            try:
                                msg = json.loads(line)
                                t = msg.get("type")
                                if t == "progress":
                                    v = int(msg.get("value", 0) or 0)
                                    pct = min(99, max(0, v))
                                    mapped = min(11, max(base, base + int((pct / 100.0) * span)))
                                    self._last_pct = mapped
                                    self.signals.progress.emit(mapped)
                                elif t == "result":
                                    last_result = msg.get("data") or {}
                                elif t == "error":
                                    raise RuntimeError(str(msg.get("message", "")))
                            except Exception:
                                # keep runner chatter in logs
                                self.signals.log.emit(f"[whisper] {line}")
                        else:
                            self.signals.log.emit(f"[whisper] {line}")

                rc = proc.wait()
                if rc != 0:
                    raise RuntimeError(f"Whisper runner failed (exit code {rc}).")
                if not last_result:
                    raise RuntimeError("Whisper runner returned no result.")

                seg_json_tmp = str(last_result.get("segments_json") or "")
                txt_tmp = str(last_result.get("text_path") or "")
                info = last_result.get("info") or {}

                if not seg_json_tmp or not os.path.isfile(seg_json_tmp):
                    raise RuntimeError(f"Whisper missing segments JSON: {seg_json_tmp}")
                if not txt_tmp or not os.path.isfile(txt_tmp):
                    raise RuntimeError(f"Whisper missing transcript text: {txt_tmp}")

                # Canonical outputs in the project folder
                whisper_meta_path = (manifest.get("paths", {}) or {}).get("whisper_meta_json") or os.path.join(audio_dir, "whisper_meta.json")
                whisper_segments_path = (manifest.get("paths", {}) or {}).get("whisper_segments_json") or os.path.join(audio_dir, "whisper_segments.json")
                transcript_path = (manifest.get("paths", {}) or {}).get("transcript_txt") or os.path.join(audio_dir, "transcript.txt")

                try:
                    import shutil as _shutil
                    _shutil.copy2(seg_json_tmp, whisper_segments_path)
                except Exception:
                    # fallback: read/write
                    _safe_write_text(whisper_segments_path, _safe_read_text(seg_json_tmp))

                try:
                    import shutil as _shutil
                    _shutil.copy2(txt_tmp, transcript_path)
                except Exception:
                    _safe_write_text(transcript_path, _safe_read_text(txt_tmp))

                transcript_text = (_safe_read_text(transcript_path) or "").strip()
                wc = _count_words(transcript_text)
                lyrics_mode = "lyrics" if wc >= 25 else "instrumental"

                # Persist meta + quick gate result
                meta = {
                    "status": "done",
                    "source_audio": os.path.basename(music_path),
                    "word_count": wc,
                    "lyrics_mode": lyrics_mode,
                    "audio_fingerprint": audio_fp,
                    "info": info,
                    "model_dir": str(model_dir),
                    "device": device,
                    "compute_type": compute_type,
                }
                _safe_write_json(whisper_meta_path, meta)

                manifest.setdefault("project", {}).setdefault("audio", {})
                manifest["project"]["audio"].update({
                    "whisper_status": "done",
                    "lyrics_mode": lyrics_mode,
                    "word_count": wc,
                })
                manifest.setdefault("paths", {})["whisper_meta_json"] = whisper_meta_path
                manifest["paths"]["whisper_segments_json"] = whisper_segments_path
                manifest["paths"]["transcript_txt"] = transcript_path
                _safe_write_json(manifest_path, manifest)

            # Decide whether Whisper should run (ready-made music only)
            _audio_src = ((manifest.get("project", {}) or {}).get("audio", {}) or {}).get("source") or ""
            _music_path = (manifest.get("paths", {}) or {}).get("music_file") or ""
            _whisper_meta = (manifest.get("paths", {}) or {}).get("whisper_meta_json") or os.path.join(audio_dir, "whisper_meta.json")
            _whisper_segments = (manifest.get("paths", {}) or {}).get("whisper_segments_json") or os.path.join(audio_dir, "whisper_segments.json")
            _transcript_txt = (manifest.get("paths", {}) or {}).get("transcript_txt") or os.path.join(audio_dir, "transcript.txt")

            if _audio_src != "ready_made_file" or not _music_path:
                _skip("Whisper (auto transcript)", "No ready-made music file selected.")
            else:
                cur_fp = _file_fingerprint(_music_path)
                meta = _safe_read_json(_whisper_meta) or {}
                already_done = (
                    meta.get("status") == "done"
                    and meta.get("audio_fingerprint") == cur_fp
                    and _file_ok(_whisper_segments, 2)
                    and _file_ok(_transcript_txt, 2)
                )
                if already_done:
                    _skip("Whisper (auto transcript)", "Up to date.")
                else:
                    _run("Whisper (auto transcript)", step_whisper_auto, 6)

            # Load Whisper results for downstream steps (word-count gate)
            _lyrics_mode = "instrumental"
            _transcript_text_full = ""
            _transcript_wc = 0
            _transcript_sha1 = ""
            try:
                if _file_ok(_transcript_txt, 2):
                    _transcript_text_full = (_safe_read_text(_transcript_txt) or "").strip()
                    _transcript_wc = _count_words(_transcript_text_full)
                    _lyrics_mode = "lyrics" if _transcript_wc >= 25 else "instrumental"
                    _transcript_sha1 = _sha1_text(_transcript_text_full) if _transcript_text_full else ""
            except Exception:
                pass
            _transcript_excerpt = _transcript_text_full[:6000] + ("..." if len(_transcript_text_full) > 6000 else "")


            # Step R: Reference guidance via Qwen3-VL Describe (Chunk 4A)
            refs_guidance_path = os.path.join(story_dir, "refs_guidance.txt")
            refs_describe_path = os.path.join(story_dir, "refs_describe.txt")

            # Determine whether to run the vision-describe step
            _ref_strategy = ""
            _copied_refs: List[str] = []
            try:
                _ref_strategy = str(self.job.encoding.get("ref_strategy") or "").strip()
            except Exception:
                _ref_strategy = ""
            try:
                _ref_info = (manifest.get("project", {}) or {}).get("references") or {}
                _copied_refs = list(_ref_info.get("copied_files") or [])
            except Exception:
                _copied_refs = []

            # Fingerprint: strategy + copied refs file stats (portable + stable enough)
            try:
                _ref_fp = _sha1_text(json.dumps({
                    "strategy": _ref_strategy,
                    "refs": [_file_fingerprint(p) for p in _copied_refs],
                }, sort_keys=True))
            except Exception:
                _ref_fp = ""

            def step_refs_qwen_describe() -> None:
                if not _copied_refs:
                    raise RuntimeError("No reference images copied into the project.")
                model_dir = Path(_qwen_model_dir())
                if not (model_dir.exists() and any(model_dir.iterdir())):
                    raise RuntimeError(f"Qwen3-VL model folder not found or empty: {model_dir}")

                # Lazy imports (keep planner startup fast)
                from PIL import Image  # type: ignore
                import torch  # type: ignore
                try:
                    from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore
                except Exception as e:
                    raise RuntimeError(f"transformers not available for Qwen3-VL Describe: {e}")

                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32

                processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
                model = AutoModelForVision2Seq.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, torch_dtype=dtype)
                try:
                    model.to(device)
                except Exception:
                    pass
                try:
                    model.eval()
                except Exception:
                    pass

                prompt = "Describe this image in detail."


                per_img_blocks: List[str] = []
                for i, p in enumerate(_copied_refs, start=1):
                    if not os.path.isfile(p):
                        continue
                    try:
                        img = Image.open(p).convert("RGB")
                    except Exception:
                        continue

                    # Qwen3-VL expects image placeholder tokens in the text input; build a proper chat prompt.
                    try:
                        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
                        chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = processor(text=[chat_text], images=[img], return_tensors="pt", padding=True)
                    except Exception:
                        # Fallback (may fail for some Qwen3-VL variants, but better than silently skipping)
                        inputs = processor(images=img, text=prompt, return_tensors="pt")
                    try:
                        for k, v in list(inputs.items()):
                            if hasattr(v, "to"):
                                inputs[k] = v.to(device)
                    except Exception:
                        pass

                    gen = model.generate(**inputs, max_new_tokens=520, do_sample=False, no_repeat_ngram_size=3)
                    # Decode only newly generated tokens (avoid echoing the prompt).
                    try:
                        input_ids = inputs.get("input_ids")
                        if input_ids is not None:
                            prompt_len = int(getattr(input_ids, "shape", [0, 0])[-1])
                            out_ids = gen[0][prompt_len:]
                            decoder = getattr(processor, "decode", None)
                            if decoder is None and hasattr(processor, "tokenizer"):
                                decoder = processor.tokenizer.decode
                            txt = decoder(out_ids, skip_special_tokens=True).strip() if decoder else processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
                        else:
                            txt = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
                    except Exception:
                        txt = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()

                    header = f"### REF_{i:02d}: {os.path.basename(p)}"
                    per_img_blocks.append(header + "\n" + txt)

                if not per_img_blocks:
                    raise RuntimeError("Reference describe produced no output (no readable images).")

                # Write per-image details
                _safe_write_text(refs_describe_path, "\n\n".join(per_img_blocks).strip() + "\n")

                # Merge into a single guidance doc (simple, deterministic)
                merged = (
                    "REFERENCE IMAGE GUIDANCE (Qwen3-VL Describe)\n"
                    "Use this as constraints when generating the plan, shots and prompts.\n\n"
                    + "\n\n".join(per_img_blocks).strip()
                    + "\n\n"
                    + "GLOBAL CONTINUITY RULES:\n"
                    + "- Keep the main subject(s) visually consistent across all shots (no random outfit swaps).\n"
                    + "- Reuse dominant colors/materials/props where appropriate.\n"
                    + "- Match lighting and setting style cues unless the story explicitly transitions.\n"
                )
                _safe_write_text(refs_guidance_path, merged.strip() + "\n")

                # Persist in manifest
                manifest.setdefault("paths", {})["refs_describe_txt"] = refs_describe_path
                manifest["paths"]["refs_guidance_txt"] = refs_guidance_path
                manifest.setdefault("project", {}).setdefault("references", {})
                manifest["project"]["references"]["guidance_file"] = refs_guidance_path
                # Store device info for debugging
                manifest["project"]["references"]["describe_device"] = device

                # Mark step record with fingerprint
                srec = manifest["steps"].get("Refs (Qwen3-VL Describe)") or {}
                srec.update({
                    "status": "done",
                    "fingerprint": _ref_fp,
                    "debug": {"refs": list(_copied_refs), "model_dir": str(model_dir), "device": device},
                    "ts": time.time(),
                })
                manifest["steps"]["Refs (Qwen3-VL Describe)"] = srec
                _safe_write_json(manifest_path, manifest)

            # Run / skip reference guidance step
            try:
                _refs_prev = (manifest.get("steps") or {}).get("Refs (Qwen3-VL Describe)") or {}
                _need_refs_step = (_ref_strategy == "qwen3vl_describe" and bool(_copied_refs))
                if not _need_refs_step:
                    _skip("Refs (Qwen3-VL Describe)", "No reference strategy A selected (or no refs).")
                else:
                    up_to_date = (
                        _file_ok(refs_guidance_path, 50)
                        and _file_ok(refs_describe_path, 50)
                        and _refs_prev.get("status") == "done"
                        and (_refs_prev.get("fingerprint") or "") == (_ref_fp or "")
                    )
                    if up_to_date:
                        _skip("Refs (Qwen3-VL Describe)", "refs_guidance.txt up-to-date (fingerprint match)")
                    else:
                        _run("Refs (Qwen3-VL Describe)", step_refs_qwen_describe, 9)
            except Exception:
                # If anything breaks here, fail soft: don't block the rest of the planner.
                try:
                    err_path = os.path.join(errors_dir, "refs_qwen_describe_error.txt")
                    _safe_write_text(err_path, traceback.format_exc())
                    manifest.setdefault("steps", {}).setdefault("Refs (Qwen3-VL Describe)", {})
                    manifest["steps"]["Refs (Qwen3-VL Describe)"].update({"status": "failed", "note": "Failed soft; see errors/refs_qwen_describe_error.txt", "ts": time.time()})
                    _safe_write_json(manifest_path, manifest)
                except Exception:
                    pass

            # Load guidance excerpt for downstream prompts
            _refs_guidance_excerpt = ""
            try:
                if _file_ok(refs_guidance_path, 10):
                    _refs_guidance_excerpt = (_safe_read_text(refs_guidance_path) or "").strip()
                    if len(_refs_guidance_excerpt) > 5000:
                        _refs_guidance_excerpt = _refs_guidance_excerpt[:5000] + "..."
            except Exception:
                _refs_guidance_excerpt = ""

            # Step A: Plan
            plan_path = os.path.join(story_dir, "plan.json")
            plan_raw_path = os.path.join(story_dir, "plan_raw.txt")
            qwen_prompts_used = os.path.join(story_dir, "qwen_prompts_used.txt")
            qwen_plan_err = os.path.join(errors_dir, "qwen_plan_error.txt")

            # Resolve generation profile (proxy targets)
            gen_profile = self.job.encoding.get("generation_profile")
            if not isinstance(gen_profile, dict):
                gen_profile = _resolve_generation_profile(
                    self.job.encoding.get("video_model") or "",
                    self.job.encoding.get("gen_quality_preset") or "",
                )
                self.job.encoding["generation_profile"] = gen_profile

            plan_fingerprint = _sha1_text(json.dumps({
                "prompt": self.job.prompt,
                "extra_info": self.job.extra_info,
                "duration_sec": self.job.approx_duration_sec,
                "audio": {
                    "storytelling": self.job.storytelling,
                    "music_background": self.job.music_background,
                    "silent": self.job.silent,
                    "storytelling_volume": self.job.storytelling_volume,
                    "music_volume": self.job.music_volume,
                },
                "models": {
                    "image_model": self.job.encoding.get("image_model"),
                    "video_model": self.job.encoding.get("video_model"),
                    "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                    "own_character_bible_enabled": self.job.encoding.get("own_character_bible_enabled"),
                    "own_character_1_prompt": self.job.encoding.get("own_character_1_prompt"),
                    "own_character_2_prompt": self.job.encoding.get("own_character_2_prompt"),
                },
                "generation_profile": gen_profile,
                "own_storyline_digest": _own_storyline_digest,
                "final_export": {
                    "resolution_preset": self.job.resolution_preset,
                    "quality_preset": self.job.quality_preset,
                    "encode": self.job.encoding,
                },
            }, sort_keys=True))


            # Own Character Bible (manual 1-2) — prepare injection early so ALL steps (plan, shots, prompts) can use it.
            try:
                _own_enabled = bool(self.job.encoding.get("own_character_bible_enabled"))
            except Exception:
                _own_enabled = False
            try:
                p1 = str(self.job.encoding.get("own_character_1_prompt") or "").strip()
                p2 = str(self.job.encoding.get("own_character_2_prompt") or "").strip()
                _own_prompts = [p for p in (p1, p2) if (p or "").strip()]
            except Exception:
                _own_prompts = []
            _own_active = bool(_own_enabled and _own_prompts)

            def _own_block_prose(prompts: List[str]) -> str:
                try:
                    ps = []
                    for p in prompts:
                        tt = _strip_nonvisual_markers(str(p or "").strip())
                        tt = re.sub(r"[\r\n\t]+", " ", tt)
                        tt = re.sub(r"[ ]{2,}", " ", tt).strip()
                        if tt:
                            ps.append(tt)
                    return " | ".join(ps)
                except Exception:
                    return ""

            _own_prose = _own_block_prose(_own_prompts) if _own_active else ""

            def step_plan() -> None:
                # Own storyline: skip Qwen plan and create a lightweight placeholder plan.
                if bool(_own_storyline_enabled):
                    plan = {
                        "title": f"Own storyline {self.job.job_id}",
                        "prompt": self.job.prompt,
                        "extra_info": self.job.extra_info,
                        "logline": self.job.prompt,
                        "setting": "",
                        "characters": [],
                        "tone": "",
                        "continuity_rules": [
                            "Use the user's prompts verbatim.",
                            "Keep recurring characters/props consistent if the user repeats them.",
                        ],
                        "beats": [],
                        "source": "own_storyline",
                    }
                    _safe_write_json(plan_path, plan)
                    manifest["paths"]["plan_json"] = plan_path
                    manifest["paths"]["plan_raw_txt"] = plan_raw_path
                    manifest["paths"]["qwen_prompts_used_txt"] = qwen_prompts_used
                    srec = manifest["steps"].get("Plan (story + constraints)") or {}
                    srec.update({
                        "status": "done",
                        "fingerprint": plan_fingerprint,
                        "note": "Own storyline enabled: plan step skipped (placeholder written).",
                        "ts": time.time(),
                    })
                    manifest["steps"]["Plan (story + constraints)"] = srec
                    _safe_write_json(manifest_path, manifest)
                    return

                # Determine audio mode for story
                audio_mode = ("silent" if self.job.silent else
                              ("both" if (self.job.storytelling and self.job.music_background) else
                               ("storytelling" if self.job.storytelling else
                                ("music" if self.job.music_background else "none"))))

                default_taxonomy = _infer_default_subject_taxonomy(self.job.prompt, self.job.extra_info)

                # Force JSON-only plan from Qwen with strict visual/stage separation
                sys_p = (
                    "You are a planning assistant for an offline video generator. "
                    "Return ONLY valid JSON. No markdown, no commentary, no code fences. "
                    "\n\n"
                    "CRITICAL RULES:\n"
                    "1. If PROMPT/EXTRA_INFO/OWN_CHARACTERS does not clearly specify animals/creatures/non-human subjects, assume the story is about HUMANS.\n"
                    "2. Separate STAGE DIRECTIONS (metadata) from VISUAL_DESCRIPTION (what appears on screen)\n"
                    "3. Characters must include taxonomy detection: set 'taxonomy' to 'human', 'animal', or 'creature'\n"
                    "4. For animals: use 'species' (e.g., 'Red Fox', 'Snowy Owl') not 'hair/outfit'\n"
                    "5. visual_description must NEVER contain: 'Camera:', 'Shot:', 'Lighting:', 'Cut to', 'Fade'\n"
                    "6. visual_description must NEVER contain technical cinematography terms\n"
                    "7. If you use character names, keep identity consistent in every beat/shot (name + taxonomy/species) and do not change species.\n"
                    "8. If OWN_CHARACTERS is provided, use ONLY those characters as recurring characters and ensure every beat/shot includes them (do not invent new protagonists)\n"
                    "\n"
                    "Example BAD visual_description (will break generation):\n"
                    "'Camera: static, Lighting: cool. Cut to the scene.'\n"

                )
                user_p = (
                    "Create a concise but useful story plan and constraints for this video project.\n"
                    "Requirements:\n"
                    "- Output MUST be JSON only.\n"
                    "- Include: title, logline, setting, characters (list), tone, continuity_rules (list), beats (list).\n"
                    "- Keep it short and actionable for prompt generation.\n\n"
                    f"PROMPT: {self.job.prompt}\n"
                    + (f"DEFAULT_SUBJECT_TAXONOMY: {default_taxonomy}\n")
                    + (f"EXTRA_INFO: {self.job.extra_info}\n" if (self.job.extra_info or '').strip() else "")
                    + (f"OWN_CHARACTER_BIBLE_ENABLED: true\nOWN_CHARACTERS:\n{_own_prose}\n" if bool(_own_active) else "")
                    + (f"REFERENCE_GUIDANCE:\n{_refs_guidance_excerpt}\n" if (_refs_guidance_excerpt or '').strip() else "")
                    + (f"LYRICS_MODE: {_lyrics_mode}\n" if self.job.music_background else "")
                    + (f"LYRICS_TRANSCRIPT:\n{_transcript_excerpt}\n" if (_lyrics_mode == "lyrics" and (_transcript_excerpt or "").strip()) else "")
                    + f"TARGET_DURATION_SEC: {self.job.approx_duration_sec}\n"
                    + f"AUDIO_MODE: {audio_mode}\n"
                    + f"VIDEO_MODEL: {self.job.encoding.get('video_model')}\n"
                    + f"GEN_QUALITY: {self.job.encoding.get('gen_quality_preset')}\n"
                    + f"GEN_PROFILE: {json.dumps(gen_profile, ensure_ascii=False)}\n"
                )

                parsed, raw = _qwen_json_call(
                    step_name="Plan (Qwen3-VL JSON)",
                    system_prompt=sys_p,
                    user_prompt=user_p,
                    raw_path=plan_raw_path,
                    prompts_used_path=qwen_prompts_used,
                    error_path=qwen_plan_err,
                    temperature=0.35,
                    max_new_tokens=2000,
                )

                if not isinstance(parsed, dict):
                    # Fallback placeholder, but mark failed and keep debug files
                    plan = {
                        "title": f"Planner plan {self.job.job_id}",
                        "prompt": self.job.prompt,
                        "extra_info": self.job.extra_info,
                        "logline": self.job.prompt,
                        "setting": "",
                        "characters": [],
                        "tone": "",
                        "continuity_rules": [
                            "Keep characters, outfits, and key props consistent across shots.",
                            "Keep the main location style consistent unless the shot says otherwise.",
                        ],
                        "beats": [
                            {"index": 1, "purpose": "setup", "moment": "Introduce the scene and main subject."},
                            {"index": 2, "purpose": "escalation", "moment": "Increase energy and visual novelty."},
                            {"index": 3, "purpose": "payoff", "moment": "Deliver a satisfying highlight moment."},
                        ],
                    }
                    _safe_write_json(plan_path, plan)
                    manifest["paths"]["plan_json"] = plan_path
                    # Record debug paths + fingerprint and mark as failed
                    manifest["paths"]["plan_raw_txt"] = plan_raw_path
                    manifest["paths"]["qwen_prompts_used_txt"] = qwen_prompts_used
                    manifest["paths"]["qwen_plan_error_txt"] = qwen_plan_err
                    srec = manifest["steps"].get("Plan (story + constraints)") or {}
                    srec.update({
                        "status": "failed",
                        "fingerprint": plan_fingerprint,
                        "debug": {
                            "raw": plan_raw_path,
                            "prompts_used": qwen_prompts_used,
                            "error": qwen_plan_err,
                        },
                        "note": "Qwen JSON failed; wrote placeholder plan.json",
                        "ts": time.time(),
                    })
                    manifest["steps"]["Plan (story + constraints)"] = srec
                    _safe_write_json(manifest_path, manifest)
                    return

                # Merge resolved settings into plan.
                parsed.setdefault("title", f"Planner plan {self.job.job_id}")
                parsed["prompt"] = self.job.prompt
                parsed["extra_info"] = self.job.extra_info
                parsed["duration_sec"] = int(self.job.approx_duration_sec)
                parsed["audio_mode"] = audio_mode
                parsed["generation"] = {
                    "video_model": self.job.encoding.get("video_model"),
                    "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                    "profile": gen_profile,
                }
                parsed["final_export"] = {
                    "resolution_preset": self.job.resolution_preset,
                    "quality_preset": self.job.quality_preset,
                    "encode": {
                        "mode": self.job.encoding.get("mode"),
                        "crf": self.job.encoding.get("crf"),
                        "bitrate_kbps": self.job.encoding.get("bitrate_kbps"),
                        "note": self.job.encoding.get("note"),
                    },
                }
                parsed["style_notes"] = {
                    "image_model": self.job.encoding.get("image_model"),
                    "video_model": self.job.encoding.get("video_model"),
                    "videoclip_preset": self.job.encoding.get("videoclip_preset"),
                }
                parsed["planner_plan_fingerprint"] = plan_fingerprint

                _safe_write_json(plan_path, parsed)
                manifest["paths"]["plan_json"] = plan_path
                manifest["paths"]["plan_raw_txt"] = plan_raw_path
                manifest["paths"]["qwen_prompts_used_txt"] = qwen_prompts_used
                # Clear prior error path if success
                if os.path.isfile(qwen_plan_err):
                    manifest["paths"]["qwen_plan_error_txt"] = qwen_plan_err

                srec = manifest["steps"].get("Plan (story + constraints)") or {}
                srec.update({
                    "status": "done",
                    "fingerprint": plan_fingerprint,
                    "debug": {
                        "raw": plan_raw_path,
                        "prompts_used": qwen_prompts_used,
                        "error": qwen_plan_err,
                    },
                    "note": "Generated with Qwen (JSON enforced).",
                    "ts": time.time(),
                })
                manifest["steps"]["Plan (story + constraints)"] = srec
                _safe_write_json(manifest_path, manifest)

            # Idempotency: skip only if plan exists AND fingerprint matches AND last status was done
            plan_prev = (manifest.get("steps") or {}).get("Plan (story + constraints)") or {}
            if _file_ok(plan_path, 10) and plan_prev.get("fingerprint") == plan_fingerprint and plan_prev.get("status") == "done":
                _skip("Plan (story + constraints)", "plan.json up-to-date (fingerprint match)")
                plan_changed = False
            else:
                _run("Plan (story + constraints)", step_plan, 12)
                plan_changed = True

# Step B: Shots
            shots_path = os.path.join(story_dir, "shots.json")
            shots_raw_path = os.path.join(story_dir, "shots_raw.txt")
            qwen_shots_err = os.path.join(errors_dir, "qwen_shots_error.txt")

            # Shots fingerprint depends on plan fingerprint + generation profile
            shots_fingerprint = _sha1_text(json.dumps({
                "plan_fingerprint": plan_fingerprint,
                "video_model": self.job.encoding.get("video_model"),
                "gen_quality_preset": self.job.encoding.get("gen_quality_preset"),
                "generation_profile": gen_profile,
                "own_storyline_digest": _own_storyline_digest,
                "own_character_bible_enabled": self.job.encoding.get("own_character_bible_enabled"),
                "own_character_1_prompt": self.job.encoding.get("own_character_1_prompt"),
                "own_character_2_prompt": self.job.encoding.get("own_character_2_prompt"),
            }, sort_keys=True))

            def step_shots() -> None:
                # Load plan for context (best effort)
                plan_obj = _safe_read_json(plan_path) or {}
                # Choose approximate number of shots based on desired duration
                # (avg duration chosen from generation preset; stable per job)
                avg_sec = float((gen_profile.get("min_sec", 2.5) + gen_profile.get("max_sec", 5.0)) / 2.0)
                min_sec = float(gen_profile.get("min_sec", 2.5))
                max_sec = float(gen_profile.get("max_sec", 5.0))
                target_total = float(max(1.0, float(getattr(self.job, "approx_duration_sec", 0) or 0.0)))

                                # Own storyline: build a minimal seeded shot list directly from the user's prompt blocks.
                if bool(_own_storyline_enabled):
                    # info: Chunk 10 side quest — Step 4 (time fit + strongest preset constraints)
                    # Own Storyline is the strongest input: for some presets we must reconcile prompt count vs duration.
                    # Only presets that are duration-ruled/sequential need this: Preset 1 (Hardcuts) and Preset 3 (Storyline Music videoclip).
                    try:
                        _vp = str((getattr(self.job, 'encoding', {}) or {}).get('videoclip_preset') or '').strip()
                    except Exception:
                        _vp = ""
                    _needs_time_fit = _vp in ("Storyline Preset (Hardcuts)", "Storyline Music videoclip")

                    # Normalize prompt list (keep order; strip newlines; ignore empties)
                    plist = _own_storyline_prompts if isinstance(_own_storyline_prompts, list) else []
                    clean_prompts: List[str] = []
                    for it in plist:
                        try:
                            txt = str((it or {}).get("text") or "").strip()
                        except Exception:
                            txt = ""
                        txt = txt.replace("\r", " ").replace("\n", " ")
                        txt = " ".join(txt.split()).strip()
                        if txt:
                            clean_prompts.append(txt)

                    # If Own Character Bible is enabled, force the characters into each prompt block
                    # so Own Storyline runs still keep character consistency.
                    try:
                        if bool(_own_active) and str(_own_prose or "").strip():
                            cp2: List[str] = []
                            pref = str(_own_prose or "").strip()
                            for p in clean_prompts:
                                pp = str(p or "").strip()
                                if not pp:
                                    continue
                                cp2.append(f"{pref}. {pp}")
                            if cp2:
                                clean_prompts = cp2
                    except Exception:
                        pass

                    if not clean_prompts:
                        raise RuntimeError("Own storyline is enabled but no usable prompts were found.")

                    # Duration constraints for videoclip generation (min/max per image clip)
                    try:
                        _min_s = float(min_sec)
                    except Exception:
                        _min_s = 2.5
                    try:
                        _max_s = float(max_sec)
                    except Exception:
                        _max_s = 5.0
                    try:
                        _avg_s = float(avg_sec)
                    except Exception:
                        _avg_s = (_min_s + _max_s) / 2.0

                    # Determine effective target duration (can be shortened if not enough images)
                    eff_target = float(target_total) if _needs_time_fit else float(target_total)
                    n_total = int(len(clean_prompts))
                    max_possible = float(n_total) * float(_max_s)

                    shortened = False
                    if _needs_time_fit and eff_target > max_possible + 1e-3:
                        # Not enough prompts to cover the requested duration at the preset max clip length.
                        # Prefer generating NEW prompts (so we create new images) rather than shortening the video
                        # or repeating the last image.
                        try:
                            import math as _math
                            need_total = int(_math.ceil(float(eff_target) / max(0.1, float(_max_s))))
                        except Exception:
                            need_total = int(float(eff_target) / max(0.1, float(_max_s)) + 0.999)

                        extra = max(0, int(need_total) - int(n_total))
                        if extra > 0:
                            # Best-effort: ask Qwen to continue the storyline with extra prompt blocks.
                            # If Qwen fails, fall back to deterministic "new angle" variants.
                            new_prompts: List[str] = []
                            try:
                                plan_obj2 = plan_obj if isinstance(plan_obj, dict) else {}
                                title2 = str(plan_obj2.get('title') or '').strip()
                                setting2 = str(plan_obj2.get('setting') or '').strip()
                                tone2 = str(plan_obj2.get('tone') or '').strip()
                                last2 = str(clean_prompts[-1] if clean_prompts else '').strip()
                                sys_p = (
                                    "You extend a user-provided storyline for an AI video planner. "
                                    "Return STRICT JSON only.\n\n"
                                    "You MUST output a JSON object: {\"prompts\": [..]} where each item is a single prompt string.\n"
                                    "Rules:\n"
                                    "- Keep continuity with the given storyline and last prompt.\n"
                                    "- Each prompt should describe ONE clear image moment.\n"
                                    "- Avoid repeating the exact same image; change angle, action, or detail.\n"
                                    "- No numbering, no extra keys, no commentary."
                                )
                                user_p = (
                                    f"Title: {title2}\n"
                                    f"Setting: {setting2}\n"
                                    f"Tone: {tone2}\n\n"
                                    "Existing storyline prompts (in order):\n"
                                    + "\n".join([f"- {p}" for p in clean_prompts[-min(12, len(clean_prompts)) :]])
                                    + "\n\n"
                                    f"Last prompt: {last2}\n\n"
                                    f"Create {extra} NEW follow-up prompts to extend the storyline. "
                                    "Return JSON: {\"prompts\": [..]}"
                                )
                                parsed2, _raw2 = _qwen_json_call(
                                    step_name="Own Storyline (extend prompts)",
                                    system_prompt=sys_p,
                                    user_prompt=user_p,
                                    raw_path=shots_raw_path,
                                    prompts_used_path=qwen_prompts_used,
                                    error_path=qwen_shots_err,
                                    temperature=0.4,
                                    max_new_tokens=1024,
                                )
                                if isinstance(parsed2, dict):
                                    arr = parsed2.get('prompts')
                                    if isinstance(arr, list):
                                        for x in arr:
                                            s = str(x or '').replace("\r", " ").replace("\n", " ")
                                            s = " ".join(s.split()).strip()
                                            if s:
                                                new_prompts.append(s)
                            except Exception:
                                new_prompts = []

                            if len(new_prompts) < extra:
                                # Deterministic fallback variants (still NEW images, not duplicates)
                                base = clean_prompts[-1] if clean_prompts else ""
                                for i in range(extra - len(new_prompts)):
                                    k2 = len(new_prompts) + i + 1
                                    new_prompts.append(
                                        f"New angle {k2}: {base}. Different camera angle, lighting, and one small new detail.".strip()
                                    )

                            # Extend prompt list
                            clean_prompts.extend(new_prompts[:extra])
                            n_total = int(len(clean_prompts))
                            max_possible = float(n_total) * float(_max_s)
                            eff_target = float(target_total)
                        else:
                            # No extra prompts needed; keep target.
                            eff_target = float(target_total)

                    # If we have too many prompts to fit even at minimum per-clip duration, deterministically skip middle prompts.
                    prompts_used = list(clean_prompts)
                    if _needs_time_fit:
                        try:
                            too_many = (float(n_total) * float(_min_s)) > float(eff_target) + 1e-6
                        except Exception:
                            too_many = False
                        if too_many:
                            # How many prompts can fit at minimum length?
                            try:
                                import math as _math
                                k = int(_math.floor(float(eff_target) / max(0.1, float(_min_s))))
                            except Exception:
                                k = int(float(eff_target) / max(0.1, float(_min_s)))
                            k = max(1, min(n_total, int(k)))

                            # Keep first + last when possible, skip only the middle.
                            if k <= 1 or n_total <= 1:
                                keep_idxs = [0]
                            elif k == 2:
                                keep_idxs = [0, n_total - 1]
                            else:
                                middle = list(range(1, n_total - 1))
                                need = max(0, int(k) - 2)
                                # Deterministic sample (seeded by job_id + storyline digest + preset)
                                seed_basis = f"{getattr(self.job, 'job_id', '')}|{_own_storyline_digest}|{_vp}"
                                try:
                                    _r = random.Random(int(_seed_to_int(seed_basis) or 0))
                                except Exception:
                                    _r = random.Random(0)
                                if need >= len(middle):
                                    chosen = middle
                                else:
                                    chosen = sorted(_r.sample(middle, need))
                                keep_idxs = [0] + chosen + [n_total - 1]

                            prompts_used = [clean_prompts[i] for i in keep_idxs if 0 <= i < n_total]
                            # Recompute max_possible after skipping (not strictly needed, but keeps metadata consistent)
                            try:
                                max_possible = float(len(prompts_used)) * float(_max_s)
                            except Exception:
                                pass

                    # Build shots list
                    out_shots: List[Dict[str, Any]] = []
                    n_used = int(len(prompts_used))

                    # Helper: fit durations to a target sum while respecting bounds (copied from main path).
                    def _fit_durations_total_own(_shots: List[Dict[str, Any]], _target: float, _min_v: float, _max_v: float) -> None:
                        if not _shots:
                            return
                        try:
                            _target = float(_target)
                        except Exception:
                            _target = 0.0
                        if _target <= 0.0:
                            return

                        durs: List[float] = []
                        for _sh in _shots:
                            try:
                                d = float(_sh.get("duration_sec") or 0.0)
                            except Exception:
                                d = 0.0
                            if d <= 0.0:
                                d = float(_min_v)
                            durs.append(d)

                        s = float(sum(durs))
                        if s <= 0.0:
                            durs = [float(_target) / float(len(_shots))] * int(len(_shots))
                            s = float(sum(durs))

                        # Scale first, then iteratively nudge to hit the target while respecting bounds.
                        scale = float(_target) / float(max(1e-6, s))
                        durs = [max(float(_min_v), min(float(_max_v), float(d) * scale)) for d in durs]

                        for _ in range(25):
                            cur = float(sum(durs))
                            diff = float(_target) - cur
                            if abs(diff) < 0.02:
                                break
                            if diff > 0:
                                idxs = [i for i, d in enumerate(durs) if d < float(_max_v) - 1e-6]
                            else:
                                idxs = [i for i, d in enumerate(durs) if d > float(_min_v) + 1e-6]
                            if not idxs:
                                break
                            step = diff / float(len(idxs))
                            for i in idxs:
                                durs[i] = max(float(_min_v), min(float(_max_v), float(durs[i]) + float(step)))

                        for _sh, d in zip(_shots, durs):
                            try:
                                _sh["duration_sec"] = round(float(d), 2)
                            except Exception:
                                _sh["duration_sec"] = round(float(_min_v), 2)

                    for i, txt in enumerate(prompts_used, start=1):
                        sid = f"S{i:02d}"
                        # Base duration: stable but within bounds
                        if _needs_time_fit and shortened:
                            dur = float(_max_s)
                        else:
                            try:
                                dur = float(_stable_uniform(str(txt), float(_min_s), float(_max_s)))
                            except Exception:
                                dur = float(max(float(_min_s), min(float(_max_s), float(_avg_s))))
                        dur = float(max(float(_min_s), min(float(_max_s), float(dur))))
                        out_shots.append({
                            "id": sid,
                            "index": int(i),
                            "phase": _phase_for_index(i, max(1, n_used)),
                            "visual_description": str(txt),
                            "seed": str(txt),
                            "seed_int": int(_seed_to_int(str(txt)) or 0),
                            "duration_sec": float(round(dur, 2)),
                            "source": "own_storyline",
                        })

                    # If not shortened, try to fit durations to the selected target duration (within bounds).
                    if _needs_time_fit and (not shortened):
                        _fit_durations_total_own(out_shots, float(eff_target), float(_min_s), float(_max_s))

                    # Debug raw file: keep a copy of the user prompts (for resume/repair)
                    try:
                        raw_lines = []
                        for sh in out_shots:
                            raw_lines.append(f"[{int(sh.get('index') or 0):02d}] {sh.get('visual_description')}")
                        _safe_write_text(shots_raw_path, "\n".join(raw_lines).strip() + "\n")
                    except Exception:
                        pass

                    _safe_write_json(shots_path, out_shots)
                    manifest["paths"]["shots_json"] = shots_path
                    manifest["paths"]["shots_raw_txt"] = shots_raw_path
                    manifest["paths"]["qwen_prompts_used_txt"] = qwen_prompts_used
                    manifest["settings"]["n_shots"] = len(out_shots)

                    # Persist effective duration (useful for debugging; downstream audio prefers probing final video anyway)
                    if _needs_time_fit:
                        try:
                            manifest["settings"]["own_storyline_effective_duration_sec"] = float(round(float(eff_target), 2))
                        except Exception:
                            pass
                        try:
                            manifest["settings"]["own_storyline_shortened_video"] = bool(shortened)
                        except Exception:
                            pass
                        try:
                            manifest["settings"]["own_storyline_skipped_prompts"] = int(max(0, n_total - len(out_shots)))
                        except Exception:
                            pass

                    srec = manifest["steps"].get("Shots (seeded shot list)") or {}
                    srec.update({
                        "status": "done",
                        "fingerprint": shots_fingerprint,
                        "note": "Own storyline enabled: shots created from user prompts (Qwen skipped).",
                        "ts": time.time(),
                    })
                    manifest["steps"]["Shots (seeded shot list)"] = srec
                    _safe_write_json(manifest_path, manifest)
                    return
                default_taxonomy = _infer_default_subject_taxonomy(self.job.prompt, self.job.extra_info)
                # Choose approximate number of shots based on desired duration
                # (avg duration chosen from generation preset; stable per job)
                avg_sec = float((gen_profile.get("min_sec", 2.5) + gen_profile.get("max_sec", 5.0)) / 2.0)
                min_sec = float(gen_profile.get("min_sec", 2.5))
                max_sec = float(gen_profile.get("max_sec", 5.0))
                target_total = float(max(1.0, float(getattr(self.job, "approx_duration_sec", 0) or 0.0)))

                # Pick a shot count that can realistically fit the target duration given min/max per-shot lengths.
                # (This is important for short test runs like 5–15 seconds.)
                try:
                    import math as _math
                    lo = max(1, int(_math.ceil(target_total / max(0.1, max_sec))))
                    hi = max(1, int(target_total // max(0.1, min_sec)))
                    if hi < lo:
                        hi = lo
                    n_guess = int(round(target_total / max(0.5, avg_sec)))
                    n_shots = max(lo, min(80, min(hi, max(1, n_guess))))
                except Exception:
                    n_shots = max(1, min(80, int(target_total / max(1.5, avg_sec))))

                sys_p = (
                    "You are a shot-list generator for an offline video pipeline. "
                    "Return ONLY valid JSON. No markdown, no commentary, no code fences. "
                    "The JSON must be a single object with key 'shots' as an array. "
                    "Each shot MUST include fields: id, stage_directions, visual_description, subjects. "
                    "CRITICAL: visual_description must be 2-3 sentences of pure visual content with NO technical cinematography terms. It MUST include surroundings: clearly state the setting/location and add 3-7 concrete environmental details (foreground/midground/background cues, materials or objects, time of day, and atmosphere). It MUST NOT contain: Camera:, Shot:, Lighting:, Cut to, Fade. The overall sequence should tell a coherent mini-adventure with a clear goal, obstacles, escalation, a twist, and a payoff near the end. Default to humans unless the plan/prompt explicitly indicates animals or creatures; do not introduce animal protagonists unless requested." + " If OWN_CHARACTERS is provided, every shot MUST include those characters in subjects and visual_description, and you must not introduce new named protagonists."
                )
                user_p = (
                    "Generate a concise seeded shot list for the plan below.\n"
                    "Rules:\n"
                    "- Output MUST be JSON only.\n"
                    "- Provide exactly 'shots' array with {id, stage_directions, visual_description, subjects}.\n"
                    "- stage_directions: {camera, lighting, mood, purpose}\n"
                    "- visual_description: 2-3 sentences of pure visual content, NO technical terms\n- visual_description MUST include setting + surroundings (location plus 3-7 concrete environmental details; include foreground/midground/background cues, time of day, weather/atmosphere, and a distinctive prop/landmark when relevant).\n"
                    "- Use ids like S01, S02, ... up to the requested count.\n"
                    "- Keep each shot to ONE clear action and ONE main subject (the subject can be one or two characters if needed).\n"
                    "- Keep continuity across shots.\n- Across the full shot list, create a clear adventure arc: early shots establish the mission/goal, mid shots introduce obstacles and escalation, late shots deliver a twist and a climax, and the final shots resolve with a satisfying payoff.\n\n"
                    f"REQUESTED_SHOTS: {n_shots}\n"
                    f"DEFAULT_SUBJECT_TAXONOMY: {default_taxonomy}\n"
                    f"TARGET_TOTAL_DURATION_SEC: {target_total:.1f}\n"
                    f"GEN_PROFILE: {json.dumps(gen_profile, ensure_ascii=False)}\n"
                    + (f"LYRICS_MODE: {_lyrics_mode}\n" if self.job.music_background else "")
                    + (f"LYRICS_TRANSCRIPT:\n{_transcript_excerpt}\n" if (_lyrics_mode == "lyrics" and (_transcript_excerpt or "").strip()) else "")
                    + (f"REFERENCE_GUIDANCE:\n{_refs_guidance_excerpt}\n" if (_refs_guidance_excerpt or '').strip() else "")
                    + (f"OWN_CHARACTER_BIBLE_ENABLED: true\nOWN_CHARACTERS:\n{_own_prose}\n\nRULES_FOR_OWN_CHARACTERS:\n- Use ONLY these characters as recurring characters.\n- Every shot MUST include them in subjects and visual_description.\n- Do not introduce new named protagonists.\n\n" if bool(_own_active) else "")
                    + "PLAN_JSON:\n"
                    + json.dumps(plan_obj, ensure_ascii=False)
                )

                parsed, raw = _qwen_json_call(
                    step_name="Shots (Qwen3-VL JSON)",
                    system_prompt=sys_p,
                    user_prompt=user_p,
                    raw_path=shots_raw_path,
                    prompts_used_path=qwen_prompts_used,
                    error_path=qwen_shots_err,
                    temperature=0.4,
                    max_new_tokens=2600,
                )

                shots_list: List[Dict[str, Any]] = []

                if isinstance(parsed, list):
                    shots_list = [x for x in parsed if isinstance(x, dict)]
                elif isinstance(parsed, dict):
                    v = parsed.get("shots")
                    if isinstance(v, list):
                        shots_list = [x for x in v if isinstance(x, dict)]
                    else:
                        # try treat dict itself as shot map (unlikely)
                        pass

                if not shots_list:
                    # Fallback placeholder list
                    shots_list = []
                    for i in range(1, n_shots + 1):
                        sid = f"S{i:02d}"
                        shots_list.append({
                            "id": sid,
                            "stage_directions": {"camera": "medium shot", "lighting": "cinematic lighting", "mood": "neutral", "purpose": "continuity"},
                            "visual_description": f"{self.job.prompt} — moment {i} of {n_shots}.",
                            "subjects": [],
                            "seed": f"{self.job.prompt} — moment {i} of {n_shots}",
                            "camera": "medium shot",
                            "mood": "neutral",
                            "lighting": "cinematic lighting",
                            "notes": "One clear action, one subject. Keep continuity with prior shots.",
                        })
                    # mark failed but continue
                    manifest["paths"]["shots_raw_txt"] = shots_raw_path
                    manifest["paths"]["qwen_shots_error_txt"] = qwen_shots_err

                    srec = manifest["steps"].get("Shots (seeded shot list)") or {}
                    srec.update({
                        "status": "failed",
                        "fingerprint": shots_fingerprint,
                        "debug": {
                            "raw": shots_raw_path,
                            "prompts_used": qwen_prompts_used,
                            "error": qwen_shots_err,
                        },
                        "note": "Qwen JSON failed; wrote placeholder shots.json",
                        "ts": time.time(),
                    })
                    manifest["steps"]["Shots (seeded shot list)"] = srec

                # Normalize + add generation intent fields
                out_shots: List[Dict[str, Any]] = []
                for idx, sh in enumerate(shots_list, start=1):
                    sid = sh.get("id") or f"S{idx:02d}"
                    stage_raw = sh.get("stage_directions") or {}
                    if not isinstance(stage_raw, dict):
                        stage_raw = {}
                    cam = stage_raw.get("camera") or sh.get("camera") or "medium shot"
                    mood = stage_raw.get("mood") or sh.get("mood") or "neutral"
                    light = stage_raw.get("lighting") or sh.get("lighting") or "cinematic lighting"
                    purpose = stage_raw.get("purpose") or sh.get("purpose") or ""
                    visual_raw = sh.get("visual_description") or sh.get("seed") or ""
                    # Sanitize visual description to prevent metadata bleeding
                    visual_clean = _sanitize_visual_description(str(visual_raw))
                    subjects = sh.get("subjects")
                    if not isinstance(subjects, list):
                        subjects = []
                    seed = str(sh.get("seed") or visual_clean or f"{self.job.prompt} — moment {idx}").strip()
                    notes = str(sh.get("notes") or purpose or "One clear action, one subject. Keep continuity.").strip()
                    stage = {"camera": str(cam), "lighting": str(light), "mood": str(mood), "purpose": str(purpose).strip()}

                    # deterministic duration per shot within preset range
                    dur = _stable_uniform(str(seed), float(gen_profile.get("min_sec", 2.5)), float(gen_profile.get("max_sec", 5.0)))
                    dur = round(dur, 2)

                    out_shots.append({
                        "id": str(sid),
                        "seed": str(seed),
                        "seed_int": _seed_to_int(str(seed)),
                        "camera": str(cam),
                        "mood": str(mood),
                        "lighting": str(light),
                        "notes": str(notes),
                        "stage_directions": stage,
                        "visual_description": str(visual_clean),
                        "subjects": subjects,
                        # placeholders for downstream compilers
                        "duration_sec": dur,
                        "gen_fps": int(gen_profile.get("fps", 20)),
                        "gen_res": str(gen_profile.get("res", "384p")),
                        "steps": int(gen_profile.get("steps", 9)),
                        "video_model_key": str(gen_profile.get("model", "hunyuan")),
                    })



                # Adjust per-shot durations so total matches the requested project length as closely as possible.
                # We keep variability, but force the sum toward target_total (within min/max constraints).
                def _fit_durations_total(_shots: List[Dict[str, Any]], _target: float, _min_s: float, _max_s: float) -> None:
                    if not _shots:
                        return
                    try:
                        _target = float(_target)
                    except Exception:
                        _target = 0.0
                    if _target <= 0.0:
                        return

                    durs: List[float] = []
                    for _sh in _shots:
                        try:
                            d = float(_sh.get("duration_sec") or 0.0)
                        except Exception:
                            d = 0.0
                        if d <= 0.0:
                            d = float(_min_s)
                        durs.append(d)

                    s = float(sum(durs))
                    if s <= 0.0:
                        durs = [float(_target) / float(len(_shots))] * int(len(_shots))
                        s = float(sum(durs))

                    # Scale first, then iteratively nudge to hit the target while respecting bounds.
                    scale = float(_target) / float(max(1e-6, s))
                    durs = [max(float(_min_s), min(float(_max_s), float(d) * scale)) for d in durs]

                    for _ in range(25):
                        cur = float(sum(durs))
                        diff = float(_target) - cur
                        if abs(diff) < 0.02:
                            break
                        if diff > 0:
                            idxs = [i for i, d in enumerate(durs) if d < float(_max_s) - 1e-6]
                        else:
                            idxs = [i for i, d in enumerate(durs) if d > float(_min_s) + 1e-6]
                        if not idxs:
                            break
                        step = diff / float(len(idxs))
                        for i in idxs:
                            durs[i] = max(float(_min_s), min(float(_max_s), float(durs[i]) + float(step)))

                    for _sh, d in zip(_shots, durs):
                        try:
                            _sh["duration_sec"] = round(float(d), 2)
                        except Exception:
                            _sh["duration_sec"] = round(float(_min_s), 2)

                _fit_durations_total(out_shots, target_total, min_sec, max_sec)

                # Chunk 2: Drama curve labels + shot language rules (camera variety, meaningful close-ups, late establishing)
                for j, _sh in enumerate(out_shots, start=1):
                    try:
                        _sh["phase"] = _phase_for_index(j, len(out_shots))
                    except Exception:
                        _sh["phase"] = "Build"
                try:
                    _enforce_shot_language(out_shots)
                except Exception:
                    pass
                _safe_write_json(shots_path, out_shots)
                manifest["paths"]["shots_json"] = shots_path
                manifest["paths"]["shots_raw_txt"] = shots_raw_path
                manifest["paths"]["qwen_prompts_used_txt"] = qwen_prompts_used
                # Only set n_shots if we have list
                manifest["settings"]["n_shots"] = len(out_shots)

                # If step not already marked failed above, mark done now
                srec = manifest["steps"].get("Shots (seeded shot list)") or {}
                if srec.get("status") != "failed":
                    srec.update({
                        "status": "done",
                        "fingerprint": shots_fingerprint,
                        "debug": {
                            "raw": shots_raw_path,
                            "prompts_used": qwen_prompts_used,
                            "error": qwen_shots_err,
                        },
                        "note": "Generated with Qwen (JSON enforced).",
                        "ts": time.time(),
                    })
                    manifest["steps"]["Shots (seeded shot list)"] = srec

                _safe_write_json(manifest_path, manifest)

            shots_prev = (manifest.get("steps") or {}).get("Shots (seeded shot list)") or {}

            # Shots depend on plan: if plan changed, always regenerate shots
            if not plan_changed and _file_ok(shots_path, 10) and shots_prev.get("fingerprint") == shots_fingerprint and shots_prev.get("status") == "done":
                _skip("Shots (seeded shot list)", "shots.json up-to-date (fingerprint match)")
            else:
                if plan_changed and _file_ok(shots_path, 10):
                    # Mark stale in manifest
                    srec = manifest["steps"].get("Shots (seeded shot list)") or {}
                    srec.update({"status": "stale", "note": "Plan changed; regenerating shots", "ts": time.time()})
                    manifest["steps"]["Shots (seeded shot list)"] = srec
                    _safe_write_json(manifest_path, manifest)
                _run("Shots (seeded shot list)", step_shots, 28)


            # Chunk 4: Qwen Edit 2511 reference-image mode flag (used to avoid Character Bible injection).
            try:
                _at = self.job.attachments or {}
            except Exception:
                _at = {}
            try:
                _ref_files = (_at.get("ref_images") or []) or (_at.get("images") or [])
            except Exception:
                _ref_files = []
            try:
                _ref_strategy = str(self.job.encoding.get("ref_strategy") or "").strip()
            except Exception:
                _ref_strategy = ""
            _planner_use_qwen2511_refs = bool(_ref_strategy == "qwen2511_best" and bool(_ref_files))

            # Character Bible gating:
            # - Auto Character Bible uses the main toggle and is NOT allowed in Own Storyline.
            # - Own Character Bible (manual 1–2) is allowed even in Own Storyline.
            # - "Use Character Bible" for downstream prompt injection means: auto OR manual.
            try:
                _auto_cb_toggle = bool(self.job.encoding.get("character_bible_enabled", True))
            except Exception:
                _auto_cb_toggle = True

            _planner_auto_character_bible_enabled = bool(_auto_cb_toggle) and (not bool(_own_storyline_enabled)) and (not bool(_own_active))
            _planner_use_character_bible = bool(_planner_auto_character_bible_enabled or bool(_own_active))


# Step B2: Character Bible (locks for consistency)
            character_bible_path = os.path.join(story_dir, "character_bible.json")
            character_bible_raw_path = os.path.join(story_dir, "character_bible_raw.txt")
            qwen_bible_err = os.path.join(errors_dir, "qwen_bible_error.txt")

            def step_character_bible() -> None:
                plan_obj = _safe_read_json(plan_path) if os.path.exists(plan_path) else {}

                # If already present on disk, prefer it (editable by user).
                bible_list: List[Dict[str, Any]] = []
                if os.path.exists(character_bible_path):
                    try:
                        obj = _safe_read_json(character_bible_path)
                        if isinstance(obj, dict) and isinstance(obj.get("characters"), list):
                            bible_list = [x for x in obj.get("characters") if isinstance(x, dict)]
                        elif isinstance(obj, list):
                            bible_list = [x for x in obj if isinstance(x, dict)]
                    except Exception:
                        bible_list = []

                # If missing, try to generate a first pass via Qwen (best-effort).
                if not bible_list:
                    try:
                        # Generate taxonomy-aware character bible entries (one per character)
                        characters = plan_obj.get("characters", [])
                        bible_list = []
                        for c in characters:
                            if not isinstance(c, dict):
                                continue

                            name = c.get("name", "Character")
                            role = c.get("role", "")
                            desc = f"{role} {c.get('description', '')}"
                            setting = plan_obj.get("setting", "")

                            # Detect taxonomy
                            taxonomy = _detect_character_taxonomy(desc)

                            # CHOOSE SCHEMA BASED ON TAXONOMY
                            if taxonomy == "animal":
                                schema = """{
                  "name": "character name",
                  "taxonomy": "animal",
                  "species": "specific breed/species (e.g., Red Fox, Snowy Owl, Leopard gecko)",
                  "anatomy": {
                    "body_type": "slender / muscular / stocky / compact",
                    "size_relative": "small / medium / large compared to human",
                    "distinctive_features": ["specific visible traits"]
                  },
                  "coat": {
                    "type": "fur / feathers / scales / skin",
                    "color_primary": "main color (be specific: cream, charcoal, auburn)",
                    "color_secondary": "markings color",
                    "texture": "fluffy / sleek / coarse / silky / scruffy",
                    "patterns": ["tabby stripes", "white chest patch", "black mask", "sock markings"]
                  },
                  "facial": {
                    "snout_shape": "long / short / flat / pointed",
                    "ears": "pointed upright / floppy / tufted / large",
                    "eyes": "color and shape (almond, round, etc)",
                    "distinctive": ["whisker length", "freckles", "beauty marks"]
                  },
                  "limbs": {
                    "paws_hooves": "description of feet",
                    "tail": "plumed / short / curled / bushy",
                    "wings": "only if applicable"
                  },
                  "posture_movement": ["graceful", "jerky", "predatory", "playful", "sluggish"],
                  "do_not_change": ["no collar", "no clothing", "natural appearance only"],
                  "palette": ["2-4 color keywords"],
                  "vibe": ["2-3 personality adjectives"]
                }"""
                                rules = """CRITICAL RULES FOR ANIMALS:
                - NO clothing, outfits, accessories, or human items unless explicitly requested
                - Describe NATURAL anatomy (fur, feathers, scales) NOT human features
                - Species must be specific (not just "bird" but "Snowy Owl")
                - Focus on markings, fur texture, and body proportions"""
                            elif taxonomy == "creature":
                                schema = """{
                  "name": "character name",
                  "taxonomy": "creature",
                  "base_anatomy": "quadruped / humanoid / avian / serpentine / hybrid",
                  "skin_coat": {
                    "covering": "scales / fur / feathers / chitin / stone / energy",
                    "colors": ["primary", "secondary", "glow colors"],
                    "texture": "smooth / rough / armored / crystalline"
                  },
                  "distinctive_features": ["horns", "wings", "tail type", "extra limbs", "glowing veins"],
                  "face_head": ["specific monster/animal hybrid features"],
                  "do_not_change": ["immutable traits"],
                  "palette": ["color keywords"],
                  "vibe": ["descriptive adjectives"]
                }"""
                                rules = "Describe creature anatomy. May include minimal clothing if humanoid, but focus on natural features."
                            else:  # human
                                schema = """{
                  "name": "character name",
                  "taxonomy": "human",
                  "face_traits": ["6-10 stable facial characteristics (face only)"],
                  "hair": "hair details ONLY if clearly implied by the prompt; otherwise empty string",
                  "outfit": "",
                  "palette": ["2-4 colors (optional)"],
                  "vibe": ["2-3 visual adjectives (optional)"],
                  "do_not_change": ["do not invent outfits; keep clothing from the shot prompt", "no extra people", "no makeup unless explicitly requested"]
                }"""
                                rules = "Human character. Focus on FACE identity anchors (no outfits). Hair only if clearly implied. Never add makeup unless explicitly requested."

                            # Build system prompt with correct schema
                            sys_p = f"""You are a Character Bible generator. Create a locked identity reference.

                Output STRICT JSON following this schema exactly:
                {schema}

                {rules}

                IMPORTANT:
                - Be SPECIFIC: "Brown" is bad. "Chocolate brown with caramel undercoat" is good.
                - Use only VISUAL traits that appear on screen. No personality, no backstory.
                - The "do_not_change" field must include things to avoid (e.g., "no clothing" for animals)."""

                            user_p = f"""Create character bible for: {name}
                Context: {desc}
                Setting: {setting}
                Taxonomy detected: {taxonomy}

                Return only the JSON object."""

                            parsed, _raw = _qwen_json_call(
                                step_name=f"Character Bible ({name})",
                                system_prompt=sys_p,
                                user_prompt=user_p,
                                raw_path=character_bible_raw_path,
                                prompts_used_path=qwen_prompts_used,
                                error_path=qwen_bible_err,
                                temperature=0.5,
                                max_new_tokens=2048,
                            )

                            if isinstance(parsed, dict):
                                parsed["taxonomy"] = taxonomy  # Ensure taxonomy is preserved
                                bible_list.append(parsed)
                    except Exception:
                        bible_list = []

                # Always ensure we have at least placeholders based on plan.
                manifest.setdefault("project", {})
                manifest["project"]["character_bible"] = bible_list  # temporarily, normalized below
                bible_list = _ensure_character_bible(manifest, plan_obj)

                # Persist to disk (editable)
                try:
                    _safe_write_json(character_bible_path, {"characters": bible_list})
                except Exception:
                    pass

                manifest.setdefault("paths", {})["character_bible_json"] = character_bible_path
                # Step record
                srec = manifest["steps"].get("Character Bible") or {}
                srec.update({"status": "done", "note": "Stored in manifest.project.character_bible", "ts": time.time()})
                manifest["steps"]["Character Bible"] = srec
                _safe_write_json(manifest_path, manifest)


            if _planner_use_qwen2511_refs:
                # Qwen 2511 reference-image runs must rely on storyline + shot prompts only.
                # A Character Bible injected on top can invent identity details and cause drift.
                _skip("Character Bible", "Skipped for Qwen2511 ref strategy (use storyline prompts only)")
                try:
                    manifest.setdefault("project", {})["character_bible"] = []
                    srec = manifest["steps"].get("Character Bible") or {}
                    srec.update({"status": "skipped", "note": "Skipped for Qwen2511 ref strategy", "ts": time.time()})
                    manifest["steps"]["Character Bible"] = srec
                    _safe_write_json(manifest_path, manifest)
                except Exception:
                    pass

            elif bool(_own_active):
                # Own Character Bible is allowed even in Own Storyline.
                def step_own_character_bible() -> None:
                    try:
                        bible_list: List[Dict[str, Any]] = []
                        for i, p in enumerate(list(_own_prompts or [])[:2]):
                            pp = str(p or "").strip()
                            if not pp:
                                continue
                            bible_list.append({
                                "name": f"Character {i+1}",
                                "role": "",
                                "taxonomy": "human",
                                # Put the user's description into face_traits so it reliably shows up in prompt blocks.
                                "face_traits": [pp],
                                "hair": "",
                                "outfit": "",
                                "palette": [],
                                "vibe": [],
                                "do_not_change": [],
                                "ref_images": [],
                            })
                        manifest.setdefault("project", {})["character_bible"] = bible_list
                        # Normalize + persist
                        plan_obj = _safe_read_json(plan_path) if os.path.exists(plan_path) else {}
                        norm = _ensure_character_bible(manifest, plan_obj)
                        _safe_write_json(character_bible_path, {"characters": norm})
                        manifest.setdefault("paths", {})["character_bible_json"] = character_bible_path
                        srec = manifest["steps"].get("Character Bible") or {}
                        srec.update({"status": "done", "note": "Own character bible (manual)", "ts": time.time()})
                        manifest["steps"]["Character Bible"] = srec
                        _safe_write_json(manifest_path, manifest)
                    except Exception:
                        raise

                # Always (re)write manual bible when enabled so the user sees it applied immediately.
                _run("Character Bible", step_own_character_bible, 20)

            elif (not bool(_planner_auto_character_bible_enabled)):
                # Explain why Character Bible is skipped (auto bible off vs empty own bible)
                try:
                    _own_enabled = bool(self.job.encoding.get("own_character_bible_enabled", False))
                    _own_p1 = str(self.job.encoding.get("own_character_1_prompt", "") or "").strip()
                    _own_p2 = str(self.job.encoding.get("own_character_2_prompt", "") or "").strip()
                    _own_has_any = bool(_own_p1 or _own_p2)
                except Exception:
                    _own_enabled = False
                    _own_has_any = False
                try:
                    _char_toggle_raw = bool(self.job.encoding.get("character_bible_enabled", False))
                except Exception:
                    _char_toggle_raw = False

                if _own_enabled and (not _own_has_any) and (not _char_toggle_raw):
                    _skip("Character Bible", "Skipped (Both OFF — own character bible is enabled but empty)")
                elif bool(_own_storyline_enabled) and (not _char_toggle_raw):
                    _skip("Character Bible", "Skipped (Own storyline forces auto bible OFF)")
                else:
                    _skip("Character Bible", "Skipped (Character bible toggle is OFF)")
                try:
                    manifest.setdefault("project", {})["character_bible"] = []
                    srec = manifest["steps"].get("Character Bible") or {}
                    srec.update({"status": "skipped", "note": "Skipped by user toggle", "ts": time.time()})
                    manifest["steps"]["Character Bible"] = srec
                    _safe_write_json(manifest_path, manifest)
                except Exception:
                    pass
            else:
                if _file_ok(character_bible_path, 10):
                    _skip("Character Bible", "character_bible.json already exists")
                    # also ensure loaded into manifest (best-effort)
                    try:
                        plan_obj = _safe_read_json(plan_path) if os.path.exists(plan_path) else {}
                        manifest.setdefault("project", {})
                        if not isinstance(manifest["project"].get("character_bible"), list):
                            step_character_bible()
                        else:
                            _ensure_character_bible(manifest, plan_obj)
                            _safe_write_json(manifest_path, manifest)
                    except Exception:
                        pass
                else:
                    _run("Character Bible", step_character_bible, 40)



            # Step C: Image prompts
            image_prompts_path = os.path.join(prompts_dir, "image_prompts.txt")
            def step_image_prompts() -> None:
                shots = _load_shots_list(shots_path)
                plan_obj = _safe_read_json(plan_path) if os.path.exists(plan_path) else {}
                bible = [] if (_planner_use_qwen2511_refs or (not _planner_use_character_bible)) else _ensure_character_bible(manifest, plan_obj)
                wants_text = _prompt_requests_text(self.job.prompt, self.job.extra_info)

                # Own storyline: use user's prompt blocks verbatim as txt2img prompts.
                if bool(_own_storyline_enabled):
                    if not shots:
                        # Safety: if shots are missing, synthesize them from the prompt list.
                        plist = _own_storyline_prompts if isinstance(_own_storyline_prompts, list) else []
                        if not plist:
                            raise RuntimeError("Own storyline is enabled but shots.json is empty and no prompts were found.")
                        shots = []
                        for i, it in enumerate(plist, start=1):
                            try:
                                txt = str((it or {}).get("text") or "").strip()
                            except Exception:
                                txt = ""
                            txt = txt.replace("\r", " ").replace("\n", " ")
                            txt = " ".join(txt.split()).strip()
                            if not txt:
                                continue
                            shots.append({
                                "id": f"S{i:02d}",
                                "index": int(i),
                                "phase": _phase_for_index(i, max(1, len(plist))),
                                "visual_description": txt,
                                "seed": txt,
                                "seed_int": int(_seed_to_int(txt) or 0),
                                "duration_sec": float(0.0),
                                "source": "own_storyline",
                            })

                    base_negative = _ascii_only(str(self.job.negatives or ""))
                    base_negative = _adjust_text_negatives_csv(base_negative, bool(wants_text))
                    base_negative = _sanitize_negative_list(base_negative)

                    out = []
                    shot_map = manifest.setdefault("shots", {})

                    for i, sh in enumerate(shots, start=1):
                        sid = str(sh.get("id") or f"S{i:02d}")
                        pr = str(sh.get("visual_description") or "").strip()
                        pr = pr.replace("\r", " ").replace("\n", " ")
                        pr = " ".join(pr.split()).strip()
                        # If Own Character Bible is enabled, inject it directly into the prompt text
                        # so Own Storyline stays compatible with manual character locking.
                        try:
                            if bool(_own_active) and str(_own_prose or "").strip() and pr:
                                pr = f"{str(_own_prose).strip()}. {pr}".strip()
                        except Exception:
                            pass
                        if not pr:
                            # fallback: try direct prompt list by index
                            try:
                                pr = str((_own_storyline_prompts[i - 1] or {}).get("text") or "").strip()
                                pr = pr.replace("\r", " ").replace("\n", " ")
                                pr = " ".join(pr.split()).strip()
                            except Exception:
                                pr = ""
                        try:
                            if bool(_own_active) and str(_own_prose or "").strip() and pr and (str(_own_prose).strip() not in pr):
                                pr = f"{str(_own_prose).strip()}. {pr}".strip()
                        except Exception:
                            pass
                        if not pr:
                            continue

                        try:
                            _LOGGER.log_probe(f"t2i_prompt_compiled {sid}: {pr}")
                        except Exception:
                            pass

                        rec = shot_map.get(sid) if isinstance(shot_map.get(sid), dict) else {}
                        rec.update({
                            "id": sid,
                            "seed": str(sh.get("seed") or pr).strip(),
                            "seed_int": int(sh.get("seed_int") or _seed_to_int(pr) or 0),
                            "prompt_spec": pr,
                            "negative_spec": base_negative,
                            "prompt_compiled": pr,
                            "negative_compiled": base_negative,
                            "prompt_used": pr,
                            "negative_used": base_negative,
                            "lint": [],
                            "subject_guard": "",
                            "subject_fidelity": "",
                            "ts_compiled": time.time(),
                            "source": "own_storyline",
                        })
                        shot_map[sid] = rec

                        block = [f"--- {sid} ---", "PROMPT:", pr, "", "NEGATIVE:", base_negative, "", "LINT: (none)"]
                        out.append("\n".join(block).strip())

                    if not out:
                        raise RuntimeError("Own storyline is enabled but no prompts were usable (empty after cleanup).")

                    _safe_write_text(image_prompts_path, "\n\n".join(out).strip() + "\n")
                    manifest["paths"]["image_prompts_txt"] = image_prompts_path
                    _safe_write_json(manifest_path, manifest)
                    return

                # Alternative storymode: direct Qwen prompt list for txt2img (bypass shot/distill pipeline)
                if bool(_alt_storymode):
                    if not shots:
                        raise RuntimeError("shots.json is empty; cannot build prompt list")

                    def _one_line(s: str) -> str:
                        try:
                            s = str(s or "")
                        except Exception:
                            s = ""
                        s = s.replace("\r", " ").replace("\n", " ")
                        s = " ".join(s.split())
                        return s.strip()

                    # Build a strict, verbatim prefix for every prompt (NO SUBJECT/CONTEXT labels).
                    # The planner must not rewrite, expand, or repeat the subject; Qwen writes final txt2img prompts.
                    prefix = ""
                    character_notes: List[str] = []
                    try:
                        own_on = bool((self.job.encoding or {}).get("own_character_bible_enabled"))
                    except Exception:
                        own_on = False
                    try:
                        char1 = _one_line(str((self.job.encoding or {}).get("own_character_1_prompt") or ""))
                        char2 = _one_line(str((self.job.encoding or {}).get("own_character_2_prompt") or ""))
                    except Exception:
                        char1, char2 = "", ""

                    user_subject = _one_line(str(self.job.prompt or ""))

                    # If Own Character Bible is active, use Character 1/2 as the verbatim prefix (only once).
                    if bool(own_on) and bool(char1):
                        prefix = char1
                        if bool(char2):
                            prefix = prefix + ", " + char2
                        character_notes = [char1] + ([char2] if char2 else [])
                    else:
                        # Otherwise, treat the user's prompt as the subject/prefix.
                        prefix = user_subject

                    prefix = _one_line(prefix).rstrip(".").strip()
                    if not prefix:
                        raise RuntimeError("Planner prompt is empty; cannot build txt2img prompts")

                    # Keep variable name for downstream code; seed = prefix (verbatim).
                    seed = prefix
                    # Ask Qwen for a numbered list of prompts
                    N = int(len(shots))
                    list_log = os.path.join(prompts_dir, "qwen_prompt_list_alt_storymode.txt")

                                        # Ask Qwen for a direct txt2img prompt list (simple, one prompt per line).
                    notes_block = ""
                    try:
                        if character_notes:
                            notes_block = "Character notes (for consistency only; do NOT copy these verbatim into prompts):\n" + "\n".join([f"- {x}" for x in character_notes]) + "\n"
                    except Exception:
                        notes_block = ""

                    sys_p = (
                        "You are an expert at writing high-quality text-to-image prompts. "
                        "Return exactly N prompts. One prompt per line. No numbering. No commentary."
                    )
                    user_p = (
                        f"Subject (must be used verbatim as the prefix of every prompt): {seed}\n"
                        f"Number of prompts: {N}\n\n"
                        + notes_block +
                        "RULES:\n"
                        "- Return exactly N lines, one prompt per line (no bullets, no numbering).\n"
                        "- EVERY prompt must start with the exact subject prefix above (verbatim).\n"
                        "- After the required prefix, do NOT restate the subject/characters again in the same line.\n"
                        "- Do NOT use labels like SUBJECT: or CONTEXT:.\n"
                        "- Do NOT introduce extra main characters; no clones/duplicates/fusion.\n"
                        "- Avoid split-screen/collage/panels. Avoid recurring props unless the user asked.\n"
                        "- Spend tokens on vivid scene / location / action / lighting / composition variation.\n"
                    )
                    raw = _qwen_text_call(
                        step_name="Prompt list (Alternative storymode)",
                        system_prompt=sys_p,
                        user_prompt=user_p,
                        log_path=list_log,
                        temperature=0.35,
                        max_new_tokens=int(max(800, N * 140)),
                    )

                    def _parse_prompt_lines(blob: str, n: int, prefix_text: str) -> List[str]:
                        def _clean_line(s: str) -> str:
                            s = str(s or "")
                            s = s.replace("\r", " ").replace("\n", " ")
                            s = " ".join(s.split()).strip()
                            # Strip common numbering/bullets
                            s = re.sub(r"^\s*(?:\d+\s*[\.)]|[-*•])\s*", "", s).strip()
                            # Remove planner labels if Qwen echoes them
                            s = re.sub(r"(?i)^\s*(subject|context)\s*:\s*", "", s).strip()
                            return s
                    
                        prefix_text = _one_line(prefix_text).rstrip(".").strip()
                        outp: List[str] = []
                        if not blob:
                            return outp
                    
                        # 1) Try line-by-line (preferred: one prompt per line)
                        lines = [ln for ln in str(blob).splitlines() if str(ln).strip()]
                        for ln in lines:
                            s = _clean_line(ln)
                            if not s:
                                continue
                    
                            # Remove any accidental appended "Main characters..." section
                            if "main characters" in s.lower():
                                s = s.split("Main characters")[0].strip()
                                s = s.strip(" |,;")
                    
                            # Enforce exact prefix at the start (verbatim)
                            if prefix_text:
                                if not s.startswith(prefix_text):
                                    if s.lower().startswith(prefix_text.lower()):
                                        s = prefix_text + s[len(prefix_text):]
                                    else:
                                        s = (prefix_text + " " + s).strip()
                    
                                # Ensure the prefix does NOT appear again later in the same line
                                rest = s[len(prefix_text):]
                                try:
                                    rest = re.sub(re.escape(prefix_text), "", rest, flags=re.IGNORECASE)
                                except Exception:
                                    pass
                                s = (prefix_text + rest).strip()
                                s = " ".join(s.split())
                    
                            if s:
                                outp.append(s)
                            if len(outp) >= int(n):
                                break
                    
                        return outp
                    prompts = _parse_prompt_lines(raw, N, seed)
                    if len(prompts) != N and _HAVE_QWEN_TEXT and (_qwen_generate_text is not None):
                        try:
                            user_p2 = (
                                f"You must return EXACTLY {N} lines, one prompt per line. "
                                f"Each line must start with this exact prefix verbatim: {seed} "
                                "No numbering, no bullets, no commentary."
                            )
                            raw2 = _qwen_text_call(
                                step_name="Prompt list retry (Alternative storymode)",
                                system_prompt=sys_p,
                                user_prompt=user_p2,
                                log_path=os.path.join(prompts_dir, "qwen_prompt_list_alt_storymode_retry.txt"),
                                temperature=0.2,
                                max_new_tokens=int(max(800, N * 140)),
                            )
                            prompts = _parse_prompt_lines(raw2, N, seed)
                        except Exception:
                            pass

                    base_negative = _ascii_only(str(self.job.negatives or ""))
                    base_negative = _adjust_text_negatives_csv(base_negative, bool(wants_text))
                    base_negative = _sanitize_negative_list(base_negative)

                    out = []
                    shot_map = manifest.setdefault("shots", {})

                    for i, sh in enumerate(shots, start=1):
                        sid = str(sh.get("id") or f"S{i:02d}")
                        seed0 = (str(sh.get("visual_description") or "").strip() or str(sh.get("seed") or "").strip())
                        pr = prompts[i - 1] if (i - 1) < len(prompts) else seed
                        pr = str(pr or "").replace("\n", " ").replace("\r", " ").strip()
                        pr = " ".join(pr.split())
                        if not pr:
                            pr = str(seed or "").strip()
                        # Final guard: enforce exact prefix once (no repeats)
                        if seed:
                            if not pr.startswith(seed):
                                if pr.lower().startswith(seed.lower()):
                                    pr = seed + pr[len(seed):]
                                else:
                                    pr = (seed + " " + pr).strip()
                            rest = pr[len(seed):]
                            try:
                                rest = re.sub(re.escape(seed), "", rest, flags=re.IGNORECASE)
                            except Exception:
                                pass
                            pr = (seed + rest).strip()
                            pr = " ".join(pr.split())

                        try:
                            if pr:
                                _LOGGER.log_probe(f"t2i_prompt_compiled {sid}: {pr}")
                        except Exception:
                            pass

                        rec = shot_map.get(sid) if isinstance(shot_map.get(sid), dict) else {}
                        rec.update({
                            "id": sid,
                            "seed": seed0,
                            "seed_int": int(sh.get("seed_int") or _seed_to_int(seed0) or 0),
                            "prompt_spec": pr,
                            "negative_spec": base_negative,
                            "prompt_compiled": pr,
                            "negative_compiled": base_negative,
                            "prompt_used": pr,
                            "negative_used": base_negative,
                            "lint": [],
                            "subject_guard": "",
                            "subject_fidelity": "",
                            "ts_compiled": time.time(),
                            "alt_storymode_seed": seed,
                        })
                        shot_map[sid] = rec

                        block = [f"--- {sid} ---", "PROMPT:", pr, "", "NEGATIVE:", base_negative, "", "LINT: (none)"]
                        out.append("\n".join(block).strip())

                    _safe_write_text(image_prompts_path, "\n\n".join(out).strip() + "\n")
                    manifest["paths"]["image_prompts_txt"] = image_prompts_path
                    _safe_write_json(manifest_path, manifest)
                    return

                out = []
                shot_map = manifest.setdefault("shots", {})

                for sh in shots:
                    sid = str(sh.get("id") or "S??")

                    # Per-shot resume/skip (especially important for engines that don't self-skip, like Qwen Edit 2511)
                    try:
                        _shot_map_existing = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
                        _rec_existing = _shot_map_existing.get(sid) if isinstance(_shot_map_existing.get(sid), dict) else {}
                        _existing_file = str(_rec_existing.get("file") or "").strip()
                        _existing_ok = (
                            str(_rec_existing.get("status") or "") == "done"
                            and _existing_file
                            and os.path.exists(_existing_file)
                            and os.path.getsize(_existing_file) >= 1024
                        )
                        if _existing_ok:
                            image_records.append({"id": sid, "file": _existing_file, "seed": _rec_existing.get("seed")})
                            try:
                                # Keep manifest useful on partial resumes
                                manifest.setdefault("paths", {})["images_dir"] = images_dir
                                manifest["paths"]["images"] = image_records
                                if _existing_file and not manifest["paths"].get("first_image"):
                                    manifest["paths"]["first_image"] = _existing_file
                                _safe_write_json(manifest_path, manifest)
                            except Exception:
                                pass
                            try:
                                self.signals.log.emit(f"[skip] {sid} → {os.path.basename(_existing_file)} (already done)")
                            except Exception:
                                pass
                            try:
                                self.signals.asset_created.emit(_existing_file)
                            except Exception:
                                pass
                            continue
                    except Exception:
                        pass

                    # Fallback skip: if deterministic per-shot filename exists in images_dir, reuse it.
                    try:
                        _found = None
                        for _ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                            _cand = os.path.join(images_dir, f"{sid}{_ext}")
                            if os.path.exists(_cand) and os.path.getsize(_cand) >= 1024:
                                _found = _cand
                                break
                        if _found:
                            image_records.append({"id": sid, "file": _found, "seed": None})
                            try:
                                manifest.setdefault("paths", {})["images_dir"] = images_dir
                                manifest["paths"]["images"] = image_records
                                if _found and not manifest["paths"].get("first_image"):
                                    manifest["paths"]["first_image"] = _found
                                _safe_write_json(manifest_path, manifest)
                            except Exception:
                                pass
                            try:
                                self.signals.log.emit(f"[skip] {sid} → {os.path.basename(_found)} (already exists)")
                            except Exception:
                                pass
                            try:
                                self.signals.asset_created.emit(_found)
                            except Exception:
                                pass
                            continue
                    except Exception:
                        pass

                    seed0 = (str(sh.get("visual_description") or "").strip() or str(sh.get("seed") or "").strip())
                    dist_fp = ""
                    rec0 = shot_map.get(sid) if isinstance(shot_map.get(sid), dict) else {}
                    spec_prompt, spec_negative, lint = _assemble_shot_prompt(
                        sh, bible, self.job.extra_info, self.job.negatives,
                        image_model=str(self.job.encoding.get('image_model') or ''),
                        allow_text=bool(wants_text),
                                           own_character_prompts=_own_prompts if bool(_own_active) else None,
                        user_prompt=self.job.prompt,
                    )

                    # Distill a clean render prompt for txt2img (keeps story/spec rich, but prevents captions/watermarks)
                    render_prompt = spec_prompt
                    render_negative = spec_negative

                    if _ENABLE_RENDER_PROMPT_DISTILLER and _HAVE_QWEN_TEXT and (_qwen_generate_text is not None) and (not bool(_own_active)):
                        try:
                            # Fingerprint to avoid re-distilling if already done for this exact spec/model combo
                            model_key = str(self.job.encoding.get('image_model') or '')
                            dist_fp = _sha1_text(json.dumps({"spec": spec_prompt, "neg": spec_negative, "model": model_key}, sort_keys=True))
                            prev_fp = str(rec0.get("render_fingerprint") or "")
                            prev_rp = str(rec0.get("render_prompt") or "")
                            prev_rn = str(rec0.get("render_negative") or "")

                            if prev_fp and prev_fp == dist_fp and prev_rp.strip():
                                render_prompt = prev_rp
                                render_negative = prev_rn or spec_negative
                            else:
                                raw_path = os.path.join(prompts_dir, f"render_prompt_{sid}_raw.txt")
                                err_path = os.path.join(errors_dir, f"render_prompt_{sid}_error.txt")

                                # Subject Guard: default-gate garnish/humans unless explicitly present in the ORIGINAL seed/spec.
                                guard_note = (
                                    " If the shot seed/spec clearly contains an animal subject, do NOT introduce humans/people/couples/men/women/children. "
                                    " If the shot seed/spec does not explicitly mention people/humans, do NOT introduce any people. "
                                    "Do NOT introduce glass/mirrors/reflections/windows/transparent barriers or reflective framing unless explicitly present in the shot seed/spec. "
                                    "Do NOT introduce geometric symbols/grids/glyph overlays unless explicitly present in the shot seed/spec. "
                                )

                                sys_p = (
                                    "You are a render-prompt compiler for image generation. "
                                    "Output ONLY valid JSON with keys: prompt, negative. "
                                    "No markdown, no commentary. "
                                    "PROMPT RULES: English only, ASCII only, 1-2 short paragraphs of natural prose, "
                                    "NO labels/headers (no 'Direction:' etc), NO IDs/tags/underscores, NO filenames. "
                                    "Do NOT include any on-screen text/caption/watermark instructions. "
                                    "Preserve the intended subjects/entities and do not swap them. Each prompt must be standalone; do not reference previous/next images or sequences. Do not repeat 'the person' multiple times; if referring again, say 'the same person'." + guard_note
                                )
                                user_p = (
                                    "Convert the following SHOT SPEC into a clean render prompt for a txt2img model.\n"
                                    "Return JSON only.\n\n"
                                    "SHOT_SPEC:\n" + spec_prompt + "\n\n"
                                    "BASE_NEGATIVE:\n" + spec_negative + "\n"
                                )

                                parsed, _raw = _qwen_json_call(
                                    step_name=f"Render prompt ({sid}) (Qwen3-VL JSON)",
                                    system_prompt=sys_p,
                                    user_prompt=user_p,
                                    raw_path=raw_path,
                                    prompts_used_path=qwen_prompts_used,
                                    error_path=err_path,
                                    temperature=0.2,
                                    max_new_tokens=800,
                                )
                                if isinstance(parsed, dict):
                                    rp = str(parsed.get("prompt") or "").strip()
                                    rn = str(parsed.get("negative") or "").strip()
                                    if rp:
                                        render_prompt = rp
                                    if rn:
                                        render_negative = rn
                        except Exception:
                            # Best-effort: keep spec prompt
                            pass

                    # Final safety: prose-ify prompt and guard against accidental caption rendering
                    render_prompt = _finalize_render_prompt(render_prompt, spec_prompt, bool(wants_text))
                    render_negative = _ascii_only(render_negative)
                    render_negative = _adjust_text_negatives_csv(render_negative, bool(wants_text))
                    render_negative = _sanitize_negative_list(render_negative)

                    # Subject fidelity: if seed contains animals, force prompt to stay animal-only
                    render_prompt, render_negative, _sg = _apply_subject_guard(seed0, render_prompt, render_negative)

                    # Persist render cache AFTER all safety passes (so cached prompt never drifts to humans)
                    try:
                        if dist_fp and isinstance(rec0, dict):
                            rec0["render_fingerprint"] = dist_fp
                            rec0["render_prompt"] = str(render_prompt or "").strip()
                            rec0["render_negative"] = str(render_negative or "").strip()
                            shot_map[sid] = rec0
                    except Exception:
                        pass

                    # Use distilled render prompt for txt2img (keep SPEC_PROMPT separately in logs/manifest)
                    prompt = str(render_prompt or spec_prompt or "").strip()
                    negative = str(render_negative or spec_negative or "").strip()

                    # Probe logger: capture every txt2img prompt that the planner compiles.
                    try:
                        if prompt:
                            _p1 = str(prompt).replace("\n", " ").strip()
                            _LOGGER.log_probe(f"t2i_prompt_compiled {sid}: {_p1}")
                    except Exception:
                        pass


                    # Store lint + compiled prompt/negative into per-shot manifest records
                    rec = shot_map.get(sid) if isinstance(shot_map.get(sid), dict) else {}
                    rec.update({
                        "id": sid,
                        "seed": seed0,
                        "seed_int": int(sh.get("seed_int") or _seed_to_int(seed0) or 0),
                        "camera": str(sh.get("camera") or ""),
                        "lighting": str(sh.get("lighting") or ""),
                        "mood": str(sh.get("mood") or ""),
                        "phase": str(sh.get("phase") or ""),
                        "prompt_spec": spec_prompt,
                        "negative_spec": spec_negative,
                        "prompt_compiled": render_prompt,
                        "negative_compiled": render_negative,
                        "subject_guard": _sg,
                        "subject_fidelity": _sg,
                        "lint": lint,
                        "ts_compiled": time.time(),
                    })
                    shot_map[sid] = rec

                    block = [f"--- {sid} ---", "SPEC_PROMPT:", spec_prompt, "", "RENDER_PROMPT:", render_prompt, "", "NEGATIVE:", render_negative, ""]
                    if lint:
                        block.append("LINT:")
                        block.extend([f"- {x}" for x in lint])
                    else:
                        block.append("LINT: (none)")
                    out.append("\n".join(block).strip())

                _safe_write_text(image_prompts_path, "\n\n".join(out).strip() + "\n")
                manifest["paths"]["image_prompts_txt"] = image_prompts_path
                _safe_write_json(manifest_path, manifest)

            if _file_ok(image_prompts_path, 10):
                _skip("Image prompts (from shots)", "image_prompts.txt already exists")
            else:
                _run("Image prompts (from shots)", step_image_prompts, 44)

            
                        # Step C2: Render REAL images for all shots using Txt2Img engine selection
            images_dir = os.path.join(self.out_dir, "images")
            os.makedirs(images_dir, exist_ok=True)

            def step_render_all_images() -> None:
                shots = _load_shots_list(shots_path)
                if not shots:
                    raise RuntimeError("shots.json is empty; cannot render images")

                # VRAM guard: Qwen/other in-process models can sit on several GB of VRAM
                # until the first generation completes. Purge BEFORE starting images so
                # the first heavy job doesn't begin with avoidable VRAM already occupied.
                _vram_release("before images")

                # Load user's last Txt2Img settings if available (respects their engine/model selection)
                t2i_settings_path = str((_root() / "presets" / "setsave" / "txt2img.json").resolve())
                base_job = _safe_read_json(t2i_settings_path) if os.path.exists(t2i_settings_path) else {}
                if not isinstance(base_job, dict):
                    base_job = {}

                # Planner-selected image model (overrides base txt2img settings when set)
                image_model_sel = ""
                try:
                    image_model_sel = str(self.job.encoding.get("image_model") or "").strip()
                except Exception:
                    image_model_sel = ""

                def _pick_zimage_gguf_for_quality() -> str:
                    """Pick a Z-image diffusion GGUF from models/Z-Image-Turbo GGUF based on Generation quality.
                    - Low:    pick the lowest quant available (lowest Q number)
                    - Medium: pick Q5 if possible; otherwise the closest ABOVE 5 (smallest Q>=5); if none, pick the highest below 5
                    - High:   pick the highest quant available (highest Q number)
                    IMPORTANT: ignore non-diffusion GGUFs (e.g. LLMs) so we don't pass the wrong file.
                    """
                    try:
                        gq = str(self.job.encoding.get("gen_quality_preset") or "").lower().strip()
                        mode = "medium"
                        if "high" in gq:
                            mode = "high"
                        elif "low" in gq:
                            mode = "low"
                        elif "med" in gq:
                            mode = "medium"

                        gguf_dir = (_root() / "models" / "Z-Image-Turbo GGUF").resolve()
                        if not gguf_dir.exists():
                            return ""

                        cands = []
                        unknown = []

                        for p in gguf_dir.glob("**/*.gguf"):
                            try:
                                if not p.is_file():
                                    continue
                                lp = str(p).lower()
                                if any(seg in lp for seg in ("\\bin\\", "/bin/")):
                                    continue
                                name = p.name.lower()
                                if any(x in name for x in ("instruct", "qwen", "llm", "text", "encoder")):
                                    if "z_image_turbo" not in name and "z-image-turbo" not in name and "zimage" not in name:
                                        continue
                                if not ("z_image_turbo" in name or "z-image-turbo" in name or ("zimage" in name and "turbo" in name)):
                                    continue

                                m = re.search(r"\bQ(\d+)\b", p.name, flags=re.IGNORECASE)
                                qn = int(m.group(1)) if m else -1
                                try:
                                    sz = p.stat().st_size
                                except Exception:
                                    sz = 0
                                if qn < 0:
                                    unknown.append((sz, str(p)))
                                else:
                                    cands.append((qn, sz, str(p)))
                            except Exception:
                                continue

                        if not cands and not unknown:
                            for p in gguf_dir.glob("**/*.gguf"):
                                try:
                                    if not p.is_file():
                                        continue
                                    name = p.name.lower()
                                    if any(x in name for x in ("instruct", "qwen", "llm")):
                                        continue
                                    m = re.search(r"\bQ(\d+)\b", p.name, flags=re.IGNORECASE)
                                    qn = int(m.group(1)) if m else -1
                                    try:
                                        sz = p.stat().st_size
                                    except Exception:
                                        sz = 0
                                    if qn < 0:
                                        unknown.append((sz, str(p)))
                                    else:
                                        cands.append((qn, sz, str(p)))
                                except Exception:
                                    continue

                        if not cands:
                            if not unknown:
                                return ""
                            unknown.sort(key=lambda t: (-t[0], t[1]))
                            return unknown[0][1]

                        qs = sorted({q for q, _sz, _p in cands})
                        if mode == "high":
                            pick_q = max(qs)
                        elif mode == "medium":
                            above = [q for q in qs if q >= 5]
                            if above:
                                pick_q = min(above)
                            else:
                                below = [q for q in qs if q < 5]
                                pick_q = max(below) if below else max(qs)
                        else:
                            pick_q = min(qs)

                        best = [t for t in cands if t[0] == pick_q]
                        best.sort(key=lambda t: (-t[1], t[2]))
                        return best[0][2]
                    except Exception:
                        return ""


# Character bible (locks)

# Character bible (locks)
                plan_obj = _safe_read_json(plan_path) if os.path.exists(plan_path) else {}
                bible = [] if (not _planner_use_character_bible) else _ensure_character_bible(manifest, plan_obj)

                # AUTO Character Bible v2 (hard gate): generate stable IDs + identity anchors + per-shot bindings.
                _auto_cb_v2 = _auto_cb_v2_enabled(
                    bool(_planner_auto_character_bible_enabled),
                    bool(_own_active),
                    bool(_own_storyline_enabled),
                )
                if _auto_cb_v2 and _HAVE_QWEN_TEXT and (_qwen_generate_text is not None):
                    try:
                        proj = manifest.setdefault("project", {})
                        acb = proj.get("auto_character_bible_v2")
                        if not isinstance(acb, dict):
                            acb = {}
                        # Fingerprint: shots content + plan fingerprint (best effort)
                        try:
                            _shots_sig = [{"id": str(s.get("id") or ""), "t": str(s.get("visual_description") or s.get("seed") or "")[:200]} for s in shots if isinstance(s, dict)]
                        except Exception:
                            _shots_sig = []
                        _acb_fp = _sha1_text(json.dumps({
                            "plan": str((plan_obj or {}).get("planner_plan_fingerprint") or ""),
                            "shots": _shots_sig,
                        }, sort_keys=True))
                        if str(acb.get("fingerprint") or "") != _acb_fp:
                            # Build strict Qwen prompt
                            raw_path = os.path.join(prompts_dir, "auto_character_bible_v2_raw.txt")
                            err_path = os.path.join(errors_dir, "auto_character_bible_v2_error.txt")
                            sys_p = (
                                "You output ONLY valid JSON. No markdown. No extra keys unless asked. "
                                "TASK: Create a stable AUTO Character Bible with ID binding and identity anchors for image generation. "
                                "RULES: Character IDs must be C1, C2, C3... (stable within this run). "
                                "Each character must include: id, optional display_name, optional role_tags, required identity_anchor. "
                                "identity_anchor MUST describe only stable visual identity (face shape/cheekbones/nose, eye color/shape, hair color+hairstyle, skin tone, age range+build, 1-2 distinctive marks). "
                                "identity_anchor MUST NOT include clothing or 'always wears' or outfits. "
                                "Also return per-shot bindings: For each shot, specify present_characters as a list of character IDs. "
                                "Do NOT rely on names appearing in the shot text."
                            )
                            # Shots input (keep small)
                            shot_lines = []
                            for s in shots:
                                if not isinstance(s, dict):
                                    continue
                                sid = str(s.get("id") or "").strip()
                                st = str(s.get("visual_description") or s.get("seed") or "").strip()
                                st = " ".join(st.replace("\n", " ").replace("\r", " ").split())
                                if sid:
                                    shot_lines.append({"id": sid, "text": st})
                            user_p_base = (
                                "STORYLINE (context):\n" + str(self.job.prompt or "").strip() + "\n\n"
                                "SHOTS (id + text):\n" + json.dumps(shot_lines, ensure_ascii=True) + "\n\n"
                                "Return JSON with keys: characters, shots.\n"
                                "shots must be an array of objects: {id, present_characters:[...]}.\n"
                            )
                            last_err = ""
                            parsed_ok = None
                            for attempt in range(1, 4):
                                user_p = user_p_base
                                if last_err:
                                    user_p += "\nVALIDATION ERRORS TO FIX:\n" + last_err + "\nReturn corrected JSON only."
                                parsed, _raw = _qwen_json_call(
                                    step_name=f"Auto Character Bible v2 (attempt {attempt})",
                                    system_prompt=sys_p,
                                    user_prompt=user_p,
                                    raw_path=raw_path,
                                    prompts_used_path=qwen_prompts_used,
                                    error_path=err_path,
                                    temperature=0.2,
                                    max_new_tokens=1400,
                                )
                                ok, why = _auto_cb_v2_validate(parsed, shots)
                                if ok:
                                    parsed_ok = parsed
                                    break
                                last_err = why or "Invalid JSON structure."
                            if not parsed_ok:
                                raise RuntimeError("Auto Character Bible v2: Qwen output failed validation: " + (last_err or "unknown"))
                            normed = _auto_cb_v2_normalize(parsed_ok, shots)
                            acb = {"fingerprint": _acb_fp, "characters": normed.get("characters") or [], "shot_bindings": normed.get("shot_bindings") or {}}
                            proj["auto_character_bible_v2"] = acb
                            manifest["project"] = proj
                            _safe_write_json(manifest_path, manifest)
                        else:
                            proj["auto_character_bible_v2"] = acb
                            manifest["project"] = proj
                    except Exception as _e_auto:
                        # Best-effort: if v2 fails, do not break the pipeline; fall back to legacy name-matching.
                        try:
                            self.signals.log.emit(f"[AUTO CB v2] disabled (error): {_e_auto}")
                        except Exception:
                            pass


                wants_text = _prompt_requests_text(self.job.prompt, self.job.extra_info)


                # Chunk 4: reference strategy flags (used by image rendering)
                try:
                    at = self.job.attachments or {}
                except Exception:
                    at = {}

                try:
                    ref_files = (at.get("ref_images") or []) or (at.get("images") or [])
                except Exception:
                    ref_files = []

                try:
                    ref_strategy = str(self.job.encoding.get("ref_strategy") or "").strip()
                except Exception:
                    ref_strategy = ""

                use_qwen2511 = (ref_strategy == "qwen2511_best" and bool(ref_files))
                used_ref_files = list(ref_files[:2]) if use_qwen2511 else list(ref_files)

                try:
                    multi_lora = str(self.job.encoding.get("ref_multi_angle_lora") or "").strip()
                except Exception:
                    multi_lora = ""

                if use_qwen2511:
                    self.signals.log.emit(f"[REF] Using Qwen Edit 2511 for images (refs: {len(used_ref_files)}).")
                    if multi_lora:
                        self.signals.log.emit("[REF] Multi-angle LoRA detected; planner will apply it on a cadence.")
                if use_qwen2511:
                    bible = []  # Qwen2511 must use storyline/shot prompts only (no Character Bible lock).
                image_records: List[Dict[str, Any]] = []
                total = len(shots)

                def _img_step_pct(idx: int, total_n: int) -> int:
                    """Map per-shot progress during the Images step into a stable header percent range."""
                    base = 52
                    end = 69
                    try:
                        t = int(total_n)
                        i = int(idx)
                    except Exception:
                        return base
                    if t <= 1:
                        return end
                    span = max(0, int(end - base))
                    try:
                        v = int(base + int((i - 1) * (span / float(max(1, t - 1)))))
                    except Exception:
                        v = base
                    if v < base:
                        v = base
                    if v > end:
                        v = end
                    return int(v)

                for i, sh in enumerate(shots, start=1):
                    sid = str(sh.get("id") or f"S{i:02d}")
                    # Partial-resume safeguard: if an image for this shot already exists, keep it.
                    # Users can manually delete specific shot images (e.g. S03/S07) to force regeneration of only those.
                    _existing_img = ""
                    try:
                        _rec0 = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                        if isinstance(_rec0, dict):
                            _p0 = str(_rec0.get("file") or _rec0.get("img_file") or "").strip()
                            if _p0 and os.path.isfile(_p0) and os.path.getsize(_p0) >= 1024:
                                _existing_img = _p0
                        if not _existing_img:
                            for _ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                                _cand = os.path.join(images_dir, f"{sid}{_ext}")
                                if os.path.isfile(_cand) and os.path.getsize(_cand) >= 1024:
                                    _existing_img = _cand
                                    break
                    except Exception:
                        _existing_img = ""

                    if _existing_img:
                        try:
                            rec0 = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                            if not isinstance(rec0, dict):
                                rec0 = {}
                            rec0["file"] = _existing_img
                            if "ts_done" not in rec0:
                                rec0["ts_done"] = time.time()
                            shot_map[sid] = rec0
                        except Exception:
                            pass

                        image_records.append({"id": sid, "file": _existing_img, "seed": None})

                        # Update manifest incrementally so the reviewer can open immediately
                        try:
                            manifest.setdefault("paths", {})["images_dir"] = images_dir
                            manifest["paths"]["images"] = image_records
                            if _existing_img and not manifest["paths"].get("first_image"):
                                manifest["paths"]["first_image"] = _existing_img
                            _safe_write_json(manifest_path, manifest)
                        except Exception:
                            pass

                        self.signals.stage.emit(f"Images — {sid} (skip {i}/{total})")
                        self.signals.log.emit(f"[IMG] {sid}: existing image found; skipping regeneration")
                        try:
                            self.signals.progress.emit(_img_step_pct(i, total))
                        except Exception:
                            pass

                        # Live preview: show as soon as we know this shot's image path.
                        try:
                            if _existing_img and os.path.exists(_existing_img):
                                self.signals.asset_created.emit(_existing_img)
                        except Exception:
                            pass
                        continue

                    # Live header updates (per-shot) while rendering images.
                    self.signals.stage.emit(f"Images — {sid} ({i}/{total})")
                    try:
                        self.signals.progress.emit(_img_step_pct(i, total))
                    except Exception:
                        pass

                    seed_txt = (str(sh.get("visual_description") or "").strip() or str(sh.get("seed", "") or "").strip())
                    camera = str(sh.get("camera", "medium shot") or "medium shot")
                    lighting = str(sh.get("lighting", "cinematic lighting") or "cinematic lighting")
                    mood = str(sh.get("mood", "neutral") or "neutral")

                    # Alternative storymode: use the direct Qwen prompt list compiled in step_image_prompts

                    __use_alt = bool(_alt_storymode)

                    if __use_alt:

                        try:

                            _rec_alt = shot_map.get(sid) if isinstance(shot_map, dict) else {}

                            if not isinstance(_rec_alt, dict):

                                _rec_alt = {}

                            prompt = str(_rec_alt.get("prompt_user_override") or _rec_alt.get("prompt_compiled") or _rec_alt.get("prompt_used") or _rec_alt.get("prompt_spec") or "").strip()

                            negative = str(_rec_alt.get("negative_compiled") or _rec_alt.get("negative_used") or self.job.negatives or "").strip()

                        except Exception:

                            prompt = ""

                            negative = str(self.job.negatives or "").strip()

                        # Ensure single-line prompts to avoid multi-prompt parsing by engines

                        prompt = str(prompt or "").replace("\n", " ").replace("\r", " ").replace("\\n", " ").strip()

                        prompt = " ".join(prompt.split())

                        negative = _ascii_only(str(negative or ""))

                        negative = _adjust_text_negatives_csv(negative, bool(wants_text))

                        negative = _sanitize_negative_list(negative)

                        lint = []

                        _sg = ""

                        # Probe logger: capture every txt2img prompt before dispatch.

                        try:

                            if prompt:

                                _p1 = str(prompt).replace("\n", " ").strip()

                                _LOGGER.log_probe(f"t2i_prompt_dispatch {sid}: {_p1}")

                        except Exception:

                            pass

                        # Store prompt/negative into manifest records (keep downstream review/queue behavior unchanged)

                        try:

                            shot_map2 = manifest.setdefault("shots", {})

                            rec = shot_map2.get(sid) if isinstance(shot_map2.get(sid), dict) else {}

                            rec.update({

                                "id": sid,

                                "seed": seed_txt,

                                "seed_int": int(sh.get("seed_int") or _seed_to_int(seed_txt) or 0),

                                "camera": camera,

                                "lighting": lighting,

                                "mood": mood,

                                "phase": str(sh.get("phase") or ""),

                                "prompt_used": prompt,

                                "prompt_compiled": prompt,

                                "negative_compiled": negative,

                                "subject_guard": _sg,

                                "subject_fidelity": _sg,

                                "negative_used": negative,

                                "lint": lint,

                                "ts_prompt": time.time(),

                            })

                            shot_map2[sid] = rec

                        except Exception:

                            pass

                    else:

                        

                        

                                            # Build a locked prompt/negative per shot (Character Bible + camera language)

                                            spec_prompt, spec_negative, lint = _assemble_shot_prompt(
                        sh, bible, self.job.extra_info, self.job.negatives,

                                                image_model=str(self.job.encoding.get('image_model') or ''),

                                                allow_text=bool(wants_text),

                                                                   own_character_prompts=_own_prompts if bool(_own_active) else None,

                                                user_prompt=self.job.prompt,

                                            )

                        

                                            # Distill a clean render prompt for txt2img (keeps story/spec rich, but prevents captions/watermarks)

                                            render_prompt = spec_prompt

                                            render_negative = spec_negative

                        

                                            if _ENABLE_RENDER_PROMPT_DISTILLER and _HAVE_QWEN_TEXT and (_qwen_generate_text is not None) and (not bool(_own_active)):

                                                try:

                                                    # Fingerprint to avoid re-distilling if already done for this exact spec/model combo

                                                    model_key = str(self.job.encoding.get('image_model') or '')

                                                    dist_fp = _sha1_text(json.dumps({"spec": spec_prompt, "neg": spec_negative, "model": model_key}, sort_keys=True))

                                                    prev_fp = str(rec0.get("render_fingerprint") or "")

                                                    prev_rp = str(rec0.get("render_prompt") or "")

                                                    prev_rn = str(rec0.get("render_negative") or "")

                        

                                                    if prev_fp and prev_fp == dist_fp and prev_rp.strip():

                                                        render_prompt = prev_rp

                                                        render_negative = prev_rn or spec_negative

                                                    else:

                                                        raw_path = os.path.join(prompts_dir, f"render_prompt_{sid}_raw.txt")

                                                        err_path = os.path.join(errors_dir, f"render_prompt_{sid}_error.txt")

                        

                                                        # Subject Guard: default-gate garnish/humans unless explicitly present in the ORIGINAL seed/spec.

                                                        guard_note = (

                                                            " If the shot seed/spec clearly contains an animal subject, do NOT introduce humans/people/couples/men/women/children. "

                                                            " If the shot seed/spec does not explicitly mention people/humans, do NOT introduce any people. "

                                                            "Do NOT introduce glass/mirrors/reflections/windows/transparent barriers or reflective framing unless explicitly present in the shot seed/spec. "

                                                            "Do NOT introduce geometric symbols/grids/glyph overlays unless explicitly present in the shot seed/spec. "

                                                        )

                        

                                                        sys_p = (

                                                            "You are a render-prompt compiler for image generation. "

                                                            "Output ONLY valid JSON with keys: prompt, negative. "

                                                            "No markdown, no commentary. "

                                                            "PROMPT RULES: English only, ASCII only, 1-2 short paragraphs of natural prose, "

                                                            "NO labels/headers (no 'Direction:' etc), NO IDs/tags/underscores, NO filenames. "

                                                            "Do NOT include any on-screen text/caption/watermark instructions. "

                                                            "Preserve the intended subjects/entities and do not swap them. Each prompt must be standalone; do not reference previous/next images or sequences. Do not repeat 'the person' multiple times; if referring again, say 'the same person'." + guard_note

                                                        )

                                                        user_p = (

                                                            "Convert the following SHOT SPEC into a clean render prompt for a txt2img model.\n"

                                                            "Return JSON only.\n\n"

                                                            "SHOT_SPEC:\n" + spec_prompt + "\n\n"

                                                            "BASE_NEGATIVE:\n" + spec_negative + "\n"

                                                        )

                        

                                                        parsed, _raw = _qwen_json_call(

                                                            step_name=f"Render prompt ({sid}) (Qwen3-VL JSON)",

                                                            system_prompt=sys_p,

                                                            user_prompt=user_p,

                                                            raw_path=raw_path,

                                                            prompts_used_path=qwen_prompts_used,

                                                            error_path=err_path,

                                                            temperature=0.2,

                                                            max_new_tokens=800,

                                                        )

                                                        if isinstance(parsed, dict):

                                                            rp = str(parsed.get("prompt") or "").strip()

                                                            rn = str(parsed.get("negative") or "").strip()

                                                            if rp:

                                                                render_prompt = rp

                                                            if rn:

                                                                render_negative = rn

                                                except Exception:

                                                    # Best-effort: keep spec prompt

                                                    pass

                        

                                            # Final safety: prose-ify prompt and guard against accidental caption rendering

                                            render_prompt = _finalize_render_prompt(render_prompt, spec_prompt, bool(wants_text))

                                            render_negative = _ascii_only(render_negative)

                                            render_negative = _adjust_text_negatives_csv(render_negative, bool(wants_text))

                                            render_negative = _sanitize_negative_list(render_negative)

                        

                                            # Use final render prompt vars for downstream calls

                                            render_prompt, render_negative, _sg = _apply_subject_guard(seed_txt, render_prompt, render_negative)

                        

                                            prompt = render_prompt

                                            negative = render_negative

                        

                                            # Probe logger: capture every txt2img prompt before dispatch.

                                            try:

                                                if prompt:

                                                    _p1 = str(prompt).replace("\n", " ").strip()

                                                    _LOGGER.log_probe(f"t2i_prompt_dispatch {sid}: {_p1}")

                                            except Exception:

                                                pass

                        

                                            # Lint results stored in manifest; optionally echoed to job.log when Probe & logs is ON.

                                            try:

                                                shot_map = manifest.setdefault("shots", {})

                                                rec = shot_map.get(sid) if isinstance(shot_map.get(sid), dict) else {}

                                                rec.update({

                                                    "id": sid,

                                                    "seed": seed_txt,

                                                    "seed_int": int(sh.get("seed_int") or _seed_to_int(seed_txt) or 0),

                                                    "camera": camera,

                                                    "lighting": lighting,

                                                    "mood": mood,

                                                    "phase": str(sh.get("phase") or ""),

                                                    "prompt_used": prompt,

                                                    "prompt_compiled": prompt,

                                                    "negative_compiled": negative,

                                                    "subject_guard": _sg,

                                                "subject_fidelity": _sg,

                                                    "negative_used": negative,

                                                    "lint": lint,

                                                    "ts_prompt": time.time(),

                                                })

                                                shot_map[sid] = rec

                                                if getattr(_LOGGER, "enabled", False) and lint:

                                                    for ln in lint:

                                                        _LOGGER.log_job(self.job.job_id, f"[LINT] {sid}: {ln}")

                                            except Exception:

                                                pass

                        

                                            

                    t2i_job = dict(base_job)

                    # Force "one image" and override the prompt/seed
                    # Own Character Bible — final injection for this prompt
                    try:
                        if bool(_own_active) and str(_own_prose or "").strip():
                            _p0 = str(prompt or "").strip()
                            if _p0 and ("OWN CHARACTER BIBLE" not in _p0) and ("Main characters (keep consistent):" not in _p0):
                                prompt = (_p0 + " Main characters (keep consistent): " + str(_own_prose).strip() + ".").strip()
                            else:
                                prompt = _p0
                    except Exception:
                        pass
                    
                    # Auto Character Bible v2 (hard gate): append identity anchors for the characters present in this shot.
                    try:
                        if _auto_cb_v2_enabled(bool(_planner_auto_character_bible_enabled), bool(_own_active), bool(_own_storyline_enabled)):
                            _cm = _auto_cb_v2_char_map(manifest)
                            _sb = _auto_cb_v2_bindings(manifest)
                            _present = []
                            try:
                                _present = sh.get("present_characters") if isinstance(sh.get("present_characters"), list) else []
                            except Exception:
                                _present = []
                            if not _present:
                                _present = _sb.get(str(sid)) or []
                            _present = [str(x).strip() for x in (_present or []) if str(x).strip()]
                            if _present and _cm:
                                prompt = _auto_cb_v2_append_anchors(str(prompt or ""), _present, _cm, str(base_job.get("engine") or base_job.get("model") or base_job.get("model_id") or base_job.get("pipeline") or ""))
                    except Exception:
                        pass

                    # Auto Character Bible v2: for clone-prone engines (Z-Image), strengthen negative prompt too.
                    try:
                        _eh = str(base_job.get("engine") or base_job.get("model") or base_job.get("model_id") or base_job.get("pipeline") or "").lower()
                        if ("z-image" in _eh) or ("zimage" in _eh) or ("z_image" in _eh):
                            _neg_add = "extra people, duplicate person, clone, twin, multiple faces, extra face, extra head"
                            if isinstance(negative, str) and negative.strip():
                                if _neg_add.lower() not in negative.lower():
                                    negative = (negative.strip().rstrip(",") + ", " + _neg_add).strip()
                            elif isinstance(negative, str):
                                negative = _neg_add
                    except Exception:
                        pass


                    t2i_job["prompt"] = prompt
                    t2i_job["negative_prompt"] = negative
                    t2i_job["negative"] = negative
                    t2i_job["neg_prompt"] = negative
                    t2i_job["seed"] = int(sh.get("seed_int") or _seed_to_int(seed_txt) or 0)
                    t2i_job["batch"] = 1

                    # Persist the FINAL prompt we actually send (so Review shows the same text)
                    try:
                        shot_map2 = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else None
                        if isinstance(shot_map2, dict):
                            rec2 = shot_map2.get(sid) if isinstance(shot_map2.get(sid), dict) else None
                            if isinstance(rec2, dict):
                                rec2["prompt_used"] = str(prompt or "").strip()
                                rec2["prompt_compiled"] = str(prompt or "").strip()
                                shot_map2[sid] = rec2
                    except Exception:
                        pass


                    # Planner defaults: allow Planner image model override; otherwise keep user's last txt2img engine.
                    _eng = str(t2i_job.get("engine") or "").lower().strip()
                    _sel = (image_model_sel or "").lower().strip()

                    if _sel.startswith("auto"):
                        # Auto: keep user's last engine if available, otherwise default to Z-image GGUF
                        pass
                    elif "z-image" in _sel and "low" in _sel:
                        _eng = "zimage_gguf"
                    elif "z-image" in _sel:
                        _eng = "zimage"
                    elif "qwen" in _sel:
                        _eng = "qwen2512"
                    # SDXL via Diffusers
                    elif "sdxl" in _sel:
                        _eng = "diffusers"
                    # (GMT Image is planned; not wired here yet)

                    if not _eng:
                        _eng = "zimage_gguf"

                    # Ensure downstream sees the engine we decided on
                    t2i_job["engine"] = _eng

                    aspect_mode = str(self.job.encoding.get("aspect_mode") or "")

                    if use_qwen2511:
                        # Qwen Edit 2511 ref-image mode: hardcode a lower res to reduce VRAM use.
                        # Do NOT apply Qwen 2512 locked defaults here.
                        _hq = bool(self.job.encoding.get('qwen2511_high_quality') or False)

                        bw, bh = (1280, 720) if _hq else (1024, 576)
                        ww, hh = _apply_aspect_to_size(bw, bh, aspect_mode)
                        t2i_job["width"] = int(ww)
                        t2i_job["height"] = int(hh)
                    elif "qwen" in _eng:
                        # Locked-in for Qwen 2512 (Turbo LoRA friendly): Euler, 9 steps, CFG 2.5, 1344x768
                        bw, bh = 1344, 768
                        ww, hh = _apply_aspect_to_size(bw, bh, aspect_mode)
                        t2i_job["width"] = int(ww)
                        t2i_job["height"] = int(hh)
                        t2i_job["sampler"] = "Euler"
                        t2i_job["steps"] = 9
                        t2i_job["cfg_scale"] = 2.5
                        t2i_job["cfg"] = 2.5
                    elif "zimage" in _eng:
                        # Locked-in for Z-image: 10 steps, CFG 0, 1344x768 (text-to-image defaults)
                        bw, bh = 1344, 768
                        ww, hh = _apply_aspect_to_size(bw, bh, aspect_mode)
                        t2i_job["width"] = int(ww)
                        t2i_job["height"] = int(hh)
                        t2i_job["steps"] = 10
                        t2i_job["cfg_scale"] = 0.0
                        t2i_job["cfg"] = 0.0
                        if _eng == "zimage_gguf":
                            # For GGUF, pick the lowest-quant diffusion GGUF available
                            gguf_path = _pick_zimage_gguf_for_quality()
                            if gguf_path:
                                # txt2img's zimage gguf backend repurposes lora_path as diffusion gguf override
                                t2i_job["lora_path"] = gguf_path
                            else:
                                raise RuntimeError("Z-image Low Vram selected but no diffusion .gguf was found in models/Z-Image-Turbo GGUF")
                    
                    elif "diffusers" in _eng:
                        # SDXL via Diffusers:
                        # - Prefer Juggernaut in /models/sdxl/ when present
                        # - Otherwise keep the user's selected model_path
                        t2i_job["steps"] = 30
                        t2i_job["cfg_scale"] = 5.0
                        t2i_job["cfg"] = 5.0
                        try:
                            mp = str(t2i_job.get("model_path") or "").strip()
                        except Exception:
                            mp = ""
                        try:
                            sdxl_dir = (_root() / "models" / "sdxl").resolve()
                            if not sdxl_dir.exists():
                                sdxl_dir = (_root() / "models" / "SDXL").resolve()
                            jug = ""
                            if sdxl_dir.exists():
                                for f in sorted(sdxl_dir.glob("*.safetensors")):
                                    if "juggernaut" in f.name.lower():
                                        jug = str(f.resolve())
                                        break
                            if jug:
                                mp = jug
                        except Exception:
                            pass
                        if mp:
                            t2i_job["model_path"] = mp
                                                # Force SDXL resolution (planner hardcode; ignore UI/global size)
                        t2i_job["width"] = 1344
                        t2i_job["height"] = 768

                    else:
                        # Other engines: keep UI size when present, otherwise fall back to 1080p
                        try:
                            w = int(t2i_job.get("width") or 0)
                            h = int(t2i_job.get("height") or 0)
                        except Exception:
                            w = 0
                            h = 0
                        if w <= 0 or h <= 0:
                            bw, bh = 1920, 1080
                            ww, hh = _apply_aspect_to_size(bw, bh, aspect_mode)
                            t2i_job["width"] = int(ww)
                            t2i_job["height"] = int(hh)


                    # Ensure engine key exists for downstream dispatcher
                    t2i_job.setdefault("engine", "qwen2512")

                    # Output folder for the generator (txt2img uses 'output' key)
                    t2i_job["output"] = images_dir


                    # Safety: only use ultra-low steps (4–8) when Turbo LoRA is actually present.
                    try:
                        engine = str(t2i_job.get("engine") or "").lower()
                        steps_val = t2i_job.get("steps", t2i_job.get("num_steps", None))
                        cfg_val = t2i_job.get("cfg", t2i_job.get("guidance", None))
                        steps = int(steps_val) if steps_val is not None else None
                        cfg = float(cfg_val) if cfg_val is not None else None

                        if "qwen" in engine:
                            has_lora = bool(t2i_job.get("lora_path") or t2i_job.get("loras") or t2i_job.get("lora"))
                            turbo_path = None
                            turbo_used = False

                            if (steps is not None and steps <= 8) and not has_lora:
                                turbo_path = _find_newest_qwen2512_turbo_lora()
                                if turbo_path:
                                    t2i_job["lora_path"] = turbo_path
                                    t2i_job.setdefault("lora_scale", 1.0)
                                    t2i_job["steps"] = int(steps if steps is not None else 5)
                                    turbo_used = True
                                    if cfg is not None and cfg > 3.0:
                                        t2i_job["cfg"] = 2.0
                                        t2i_job["cfg_scale"] = 2.0
                                else:
                                    t2i_job["steps"] = 20

                            # record into manifest for traceability (best-effort)
                            manifest.setdefault("settings", {})
                            if turbo_path:
                                manifest["settings"]["qwen2512_turbo_lora_path"] = turbo_path
                                manifest["settings"]["qwen2512_turbo_lora_used"] = bool(turbo_used)
                            manifest["settings"]["image_sampler"] = str(t2i_job.get("sampler") or "")
                            manifest["settings"]["image_steps"] = int(t2i_job.get("steps") or 0)
                            try:
                                manifest["settings"]["image_cfg"] = float(t2i_job.get("cfg") or t2i_job.get("cfg_scale") or 0)
                            except Exception:
                                pass
                            manifest["settings"]["image_width"] = int(t2i_job.get("width") or 0)
                            manifest["settings"]["image_height"] = int(t2i_job.get("height") or 0)
                    except Exception:
                        pass

                    # Chunk 4: choose renderer based on reference strategy
                    res = None

                    if use_qwen2511:
                        # Call into helpers/qwen2511.py (best-effort, with fallbacks)
                        try:
                            from helpers import qwen2511 as _q2511  # type: ignore
                        except Exception:
                            import qwen2511 as _q2511  # type: ignore

                        # Default test/output resolution for this chunk: 1024x576 (16:9)
                        aspect_mode = str(self.job.encoding.get("aspect_mode") or "")
                        _hq = bool(self.job.encoding.get("qwen2511_high_quality") or False)

                        bw, bh = (1280, 720) if _hq else (1024, 576)

                        qw, qh = _apply_aspect_to_size(bw, bh, aspect_mode)
                        q_job = {
                            "prompt": prompt,
                            "negative_prompt": negative,
                            "negative": negative,
                            "seed": int(sh.get("seed_int") or _seed_to_int(seed_txt) or 0),
                            "width": int(qw),
                            "height": int(qh),
                            "vae_device": "cpu",
                            "vae_on_cpu": True,
                            "vae_cpu": True,
                            "use_vae_on_cpu": True,
                                                        "refs": list(used_ref_files[:2]),
                            "ref_images": list(used_ref_files[:2]),
                            "batch": 1,
                            "out_file": os.path.join(images_dir, f"{self.job.job_id}_{sid}.png"),
                        }

                        # Multi-angle cadence: every 5 images, 3 become multi-angle variants.
                        idx0 = int(i - 1) % 5
                        do_multi = (idx0 in (0, 2, 4))
                        if do_multi and multi_lora:
                            q_job["lora_path"] = multi_lora
                            q_job["lora"] = multi_lora
                            q_job["multi_angle"] = True
                            q_job["prompt"] = (q_job["prompt"] or "") + "\\n\\nMulti-angle variant: different camera angle of the same subject."

                        # Update manifest per-shot record with ref info
                        try:
                            shot_map = manifest.setdefault("shots", {})
                            rec = shot_map.get(sid) if isinstance(shot_map.get(sid), dict) else {}
                            rec["ref_strategy"] = "qwen2511_best"
                            rec["refs_used"] = list(used_ref_files[:2])
                            rec["multi_angle"] = bool(do_multi and bool(multi_lora))
                            shot_map[sid] = rec
                        except Exception:
                            pass

                        self.signals.log.emit(f"[IMG] {sid} ({i}/{total}) [qwen2511]")
                        # Planner-side execution: prefer sd-cli command build so VAE-on-CPU is honored.
                        try:
                            if hasattr(_q2511, "build_sdcli_cmd") and hasattr(_q2511, "detect_sdcli_caps"):
                                # sd-cli path (prefer saved settings if available)
                                sdcli_path = getattr(_q2511, "DEFAULT_SDCLI", "sd-cli.exe")
                                try:
                                    sp = getattr(_q2511, "SETSAVE_PATH", "")
                                    if sp and hasattr(_q2511, "_read_jsonish"):
                                        _s = _q2511._read_jsonish(sp)
                                        if isinstance(_s, dict) and _s.get("sdcli_path"):
                                            sdcli_path = str(_s.get("sdcli_path"))
                                except Exception:
                                    pass
                        
                                caps = _q2511.detect_sdcli_caps(sdcli_path)
                        
                                # Resolve model paths (fallback to helper defaults)
                                try:
                                    d = _q2511.default_model_paths() if hasattr(_q2511, "default_model_paths") else {}
                                except Exception:
                                    d = {}
                                unet_path = str(d.get("unet") or "")
                                llm_path = str(d.get("llm") or "")
                                mmproj_path = str(d.get("mmproj") or "")
                                vae_path = str(d.get("vae") or "")
                                # Prefer Q5, then Q4, then any other UNet GGUF in models/qwen2511gguf/unet
                                try:
                                    _unet_dir = _root() / "models" / "qwen2511gguf" / "unet"
                                    _cand = [str(x) for x in _unet_dir.glob("*.gguf") if x.is_file()] if _unet_dir.exists() else []
                                    if _cand:
                                        def _q2511_prio(p: str):
                                            n = os.path.basename(p).lower()
                                            mm = re.search(r"(?:^|[-_\s])q(\d+)(?:$|[-_\s])", n)
                                            q = int(mm.group(1)) if mm else None
                                            if q == 5:
                                                return (0, n)
                                            if q == 4:
                                                return (1, n)
                                            return (2, n)
                                        _cand = sorted(_cand, key=_q2511_prio)
                                        unet_path = str(_cand[0])
                                except Exception:
                                    pass

                                try:
                                    if unet_path:
                                        self.signals.log.emit(f"[qwen2511] UNet selected: {os.path.basename(unet_path)}")
                                except Exception:
                                    pass

                        
                                refs = list(q_job.get("ref_images") or q_job.get("refs") or [])[:2]
                        
                                # Create blank canvas as image 1 (keeps ref numbering stable)
                                tmp_blank = os.path.join(images_dir, f"_blank_{self.job.job_id}_{sid}_{int(time.time()*1000)}.png")
                                try:
                                    if hasattr(_q2511, "_write_blank_png"):
                                        _q2511._write_blank_png(tmp_blank, int(q_job.get("width") or 1024), int(q_job.get("height") or 576))
                                except Exception:
                                    tmp_blank = ""
                        
                                _sm = str(q_job.get("sampling_method") or q_job.get("sampler") or "euler").strip().lower()
                                if _sm != "euler":
                                    _sm = "euler"
                        
                                # IMPORTANT: enforce VAE on CPU for VRAM headroom.
                                _use_vae_cpu = True
                        
                                cmd = _q2511.build_sdcli_cmd(
                                    sdcli_path=sdcli_path,
                                    caps=caps,
                                    init_img=tmp_blank,
                                    mask_path=str(q_job.get("mask_path") or ""),
                                    ref_images=refs,
                                    use_increase_ref_index=True,
                                    disable_auto_resize_ref_images=False,
                                    prompt=str(q_job.get("prompt") or ""),
                                    negative=str(q_job.get("negative_prompt") or q_job.get("negative") or ""),
                                    unet_path=unet_path,
                                    llm_path=llm_path,
                                    mmproj_path=mmproj_path,
                                    vae_path=vae_path,
                                    steps=int(q_job.get("steps") or 20),
                                    cfg=float(q_job.get("cfg") or 4.0),
                                    seed=int(q_job.get("seed") or 0),
                                    width=int(q_job.get("width") or 1024),
                                    height=int(q_job.get("height") or 576),
                                    strength=float(q_job.get("strength") or 1.0),
                                    sampling_method=_sm,
                                    shift=float(q_job.get("shift") or q_job.get("flow") or 2.3),
                                    out_file=str(q_job.get("out_file") or ""),
                                    use_vae_tiling=False,
                                    vae_tile_size="",
                                    vae_tile_overlap=0.0,
                                    use_offload=False,
                                    use_mmap=False,
                                    use_vae_on_cpu=_use_vae_cpu,
                                    use_clip_on_cpu=False,
                                    use_diffusion_fa=True,
                                    lora_model_dir=str(q_job.get("lora_model_dir") or ""),
                                    lora_name=str(q_job.get("lora_name") or q_job.get("lora") or q_job.get("lora_path") or ""),
                                    lora_strength=float(q_job.get("lora_strength") or q_job.get("lora_scale") or 1.0),
                                )
                        
                                rc = 1
                                try:
                                    if hasattr(_q2511, "_run_capture"):
                                        rc, _out_text = _q2511._run_capture(cmd)
                                    else:
                                        rc = os.system(" ".join([str(x) for x in cmd]))
                                except Exception:
                                    rc = 1
                        
                                out_file = str(q_job.get("out_file") or "")
                                ok = (rc == 0 and out_file and os.path.exists(out_file))
                                res = {"ok": bool(ok), "files": ([out_file] if ok else []), "out_file": out_file, "rc": int(rc)}
                        
                                try:
                                    if tmp_blank and os.path.exists(tmp_blank):
                                        os.remove(tmp_blank)
                                except Exception:
                                    pass
                            else:
                                raise RuntimeError("build_sdcli_cmd/detect_sdcli_caps not found")
                        except Exception as _e:
                            # If the user is missing the UNet GGUF, we already showed a popup; do not fall back.
                            try:
                                if "Qwen2511: no UNet GGUF found" in str(_e):
                                    raise
                            except Exception:
                                pass
                            # Fallback to helper entrypoints
                            q_job["use_vae_on_cpu"] = True
                            q_job["vae_on_cpu"] = True
                            q_job["vae_cpu"] = True
                            q_job["vae_device"] = "cpu"
                            if hasattr(_q2511, "generate_one_from_job"):
                                res = _q2511.generate_one_from_job(q_job, images_dir)
                            elif hasattr(_q2511, "run_job"):
                                res = _q2511.run_job(q_job, images_dir)
                            elif hasattr(_q2511, "run"):
                                res = _q2511.run(q_job, images_dir)
                            else:
                                raise RuntimeError("qwen2511 entrypoint not found (expected build_sdcli_cmd or generate_one_from_job/run_job/run)")

                    else:
                        # Call into helpers/txt2img.py
                        try:
                            from helpers import txt2img as _t2i  # type: ignore
                        except Exception:
                            import txt2img as _t2i  # type: ignore

                        self.signals.log.emit(f"[IMG] {sid} ({i}/{total})")
                        if hasattr(_t2i, "generate_one_from_job"):
                            # Safety: some txt2img backends pass job keys as **kwargs into engine-specific functions.
                            # Older Z-image implementations can crash on any unexpected key.
                            # Strategy:
                            # 1) strip a couple of known planner-only keys
                            # 2) if we still hit a TypeError for an unexpected keyword, remove that key and retry
                            _t2i_job = dict(t2i_job)
                            for _k in ("filename_template", "format"):
                                _t2i_job.pop(_k, None)

                            # Retry loop that progressively strips unsupported keyword args.
                            # This keeps the planner compatible with older txt2img backends without forcing tight allow-lists.
                            _max_strips = 64
                            _stripped = []
                            while True:
                                try:
                                    res = _t2i.generate_one_from_job(_t2i_job, images_dir)
                                    break
                                except TypeError as _te:
                                    _msg = str(_te)
                                    _m = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", _msg)
                                    if _m:
                                        _bad = _m.group(1)
                                        if _bad in _t2i_job and len(_stripped) < _max_strips:
                                            _t2i_job.pop(_bad, None)
                                            _stripped.append(_bad)
                                            try:
                                                self.signals.log.emit(f"[IMG] backend stripped unsupported key: {_bad}")
                                            except Exception:
                                                pass
                                            continue
                                    raise
                        elif hasattr(_t2i, "generate_qwen_images"):
                            res = _t2i.generate_qwen_images(t2i_job)
                        else:
                            raise RuntimeError("txt2img generator entrypoint not found")

                    out_file = None
                    try:
                        if isinstance(res, dict) and res.get("files"):
                            out_file = res["files"][0]
                    except Exception:
                        out_file = None

                    # Fallbacks: some backends may save a file but not return a 'files' list.
                    try:
                        if (not out_file) and isinstance(_t2i_job, dict):
                            _ft = str(_t2i_job.get("filename_template") or "").strip()
                            if _ft:
                                _cand = os.path.join(images_dir, _ft)
                                if os.path.isfile(_cand) and os.path.getsize(_cand) >= 1024:
                                    out_file = _cand
                    except Exception:
                        pass
                    try:
                        if not out_file:
                            # Pick newest image in images_dir (created/modified very recently)
                            _recent = []
                            _now = time.time()
                            for _ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                                for _p in Path(images_dir).glob(_ext):
                                    try:
                                        mt = _p.stat().st_mtime
                                        if (_now - mt) <= 180.0:  # last 3 minutes
                                            _recent.append((mt, str(_p)))
                                    except Exception:
                                        continue
                            if _recent:
                                _recent.sort(key=lambda t: t[0], reverse=True)
                                out_file = _recent[0][1]
                    except Exception:
                        pass

                    # Update per-shot manifest record with output + status
                    # Normalize image filename to a deterministic per-shot name (no seed-based filenames).
                    # This keeps downstream steps stable and allows reliable skip/resume behavior.
                    try:
                        if out_file and os.path.exists(out_file):
                            ext = os.path.splitext(out_file)[1] or ".png"
                            # Prefer .jpg if the backend wrote jpg but didn't include an extension somehow
                            target = os.path.join(images_dir, f"{sid}{ext}")
                            if os.path.abspath(out_file) != os.path.abspath(target):
                                # If a prior deterministic file exists, replace it (reruns should overwrite shot outputs).
                                try:
                                    os.replace(out_file, target)
                                except Exception:
                                    # Cross-device or locked file fallback
                                    try:
                                        if os.path.exists(target):
                                            os.remove(target)
                                    except Exception:
                                        pass
                                    shutil.move(out_file, target)
                                out_file = target
                    except Exception:
                        pass

                    try:
                        shot_map = manifest.setdefault("shots", {})
                        rec = shot_map.get(sid) if isinstance(shot_map.get(sid), dict) else {}
                        rec["file"] = out_file
                        rec["status"] = "done" if (out_file and os.path.exists(out_file)) else "failed"
                        if not (out_file and os.path.exists(out_file)):
                            # Graceful fallback: write a placeholder image so downstream steps don't crash with
                            # 'Missing start image'. This also makes failures visible in the output folder.
                            try:
                                placeholder = os.path.join(images_dir, f"{sid}.png")
                                # If a placeholder already exists, keep it (reruns may intentionally skip).
                                if not os.path.exists(placeholder):
                                    try:
                                        from PIL import Image as _PIL_Image, ImageDraw as _PIL_Draw, ImageFont as _PIL_Font  # type: ignore
                                        _w = int(t2i_job.get("width") or 1024)
                                        _h = int(t2i_job.get("height") or 576)
                                        _w = max(256, min(_w, 2048))
                                        _h = max(256, min(_h, 2048))
                                        im = _PIL_Image.new("RGB", (_w, _h), (24, 24, 24))
                                        dr = _PIL_Draw.Draw(im)
                                        msg = f"FAILED {sid}\nengine={t2i_job.get('engine')}"
                                        try:
                                            if isinstance(res, dict):
                                                _e = res.get("error") or res.get("err") or res.get("message")
                                                if _e:
                                                    msg += "\n" + str(_e)[:200]
                                        except Exception:
                                            pass
                                        # red X
                                        dr.line((0, 0, _w, _h), fill=(200, 60, 60), width=8)
                                        dr.line((0, _h, _w, 0), fill=(200, 60, 60), width=8)
                                        # text
                                        try:
                                            font = _PIL_Font.load_default()
                                        except Exception:
                                            font = None
                                        dr.multiline_text((20, 20), msg, fill=(235, 235, 235), font=font, spacing=6)
                                        im.save(placeholder)
                                    except Exception:
                                        # Last resort: write a 1x1 PNG to reserve the path
                                        _png_1x1 = (
                                            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
                                            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00\x18\xdd\x8d\xb5\x00\x00\x00\x00IEND\xaeB`\x82"
                                        )
                                        with open(placeholder, "wb") as _fp:
                                            _fp.write(_png_1x1)
                                out_file = placeholder if os.path.exists(placeholder) else out_file
                                rec["file"] = out_file
                                rec["status"] = "failed_placeholder"
                                try:
                                    self.signals.log.emit(f"[IMG] {sid} produced no output — wrote placeholder")
                                except Exception:
                                    pass
                            except Exception:
                                # If even placeholder fails, keep the original behavior (will surface as missing later).
                                pass

                        rec["ts_done"] = time.time()
                        shot_map[sid] = rec
                    except Exception:
                        pass

                    image_records.append({"id": sid, "file": out_file, "seed": t2i_job.get("seed")})

                    # Live preview: emit immediately after each shot output is written.
                    try:
                        if out_file and os.path.exists(out_file):
                            self.signals.asset_created.emit(out_file)
                    except Exception:
                        pass

                    # Update manifest incrementally so a partial run is still useful
                    try:
                        manifest.setdefault("paths", {})["images_dir"] = images_dir
                        manifest["paths"]["images"] = image_records
                        if out_file and not manifest["paths"].get("first_image"):
                            manifest["paths"]["first_image"] = out_file
                        manifest.setdefault("settings", {})["image_engine"] = str(res.get("backend") or t2i_job.get("engine") or "") if isinstance(res, dict) else str(t2i_job.get("engine") or "")
                        manifest["settings"]["image_model"] = str(res.get("model") or "") if isinstance(res, dict) else str(t2i_job.get("model") or "")
                        _safe_write_json(manifest_path, manifest)
                    except Exception:
                        pass

                # Final write
                manifest.setdefault("paths", {})["images_dir"] = images_dir
                manifest["paths"]["images"] = image_records
                _safe_write_json(manifest_path, manifest)

                # VRAM guard: release any in-process image model memory before later video/music stages
                _vram_release("after images")

            # Skip if images already exist for all shots
            try:
                _shots_n = len(_load_shots_list(shots_path))
                _existing = []
                for _ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                    _existing.extend(list(Path(images_dir).glob(_ext)))
                if _shots_n > 0 and len(_existing) >= _shots_n:
                    _skip("Images (all shots)", f"{len(_existing)}/{_shots_n} images already exist")
                else:
                    _run("Images (all shots)", step_render_all_images, 52)
            except Exception:
                _run("Images (all shots)", step_render_all_images, 52)



            # Review gate (Chunk 8A): optionally pause after images for interactive inspection/regeneration
            if bool((self.job.encoding or {}).get("allow_edit_while_running")):
                self._image_review_gate(
                    output_dir=self.out_dir,
                    manifest_path=manifest_path,
                    images_dir=images_dir,
                )

            # Step D: Transcript (optional)
            transcript_path = os.path.join(audio_dir, "transcript.txt")
            def step_transcript() -> None:
                if self.job.storytelling and not self.job.silent:
                    _safe_write_text(
                        transcript_path,
                        "TRANSCRIPT (will be generated by Whisper/Qwen later)\n\n"
                        f"Prompt: {self.job.prompt}\n"
                        + (f"Extra: {self.job.extra_info.strip()}\n" if self.job.extra_info.strip() else "")
                    )
                else:
                    _safe_write_text(transcript_path, "SKIPPED (storytelling off or silent)\n")
                manifest["paths"]["transcript_txt"] = transcript_path
                _safe_write_json(manifest_path, manifest)

            if _file_ok(transcript_path, 2):
                _skip("Transcript (optional)", "transcript.txt already exists")
            else:
                _run("Transcript (optional)", step_transcript, 55)

                        # Step E: I2V prompts (compiled per shot)
            i2v_prompts_path = os.path.join(prompts_dir, "i2v_prompts.txt")
            i2v_prompts_json = os.path.join(prompts_dir, "i2v_prompts.json")

            def step_i2v_prompts() -> None:
                shots = _load_shots_list(shots_path)
                shot_map = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
                out_txt: List[str] = []
                out_list: List[Dict[str, Any]] = []

                for sh in shots:
                    sid = str(sh.get("id") or "S??")
                    seed0 = (str(sh.get("visual_description") or "").strip() or str(sh.get("seed") or "").strip())
                    dist_fp = ""
                    rec0 = shot_map.get(sid) if isinstance(shot_map.get(sid), dict) else {}
                    rec = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                    compiled = str((rec.get("prompt_compiled") if isinstance(rec, dict) else "") or "").strip()
                    neg = str((rec.get("negative_compiled") if isinstance(rec, dict) else "") or "").strip()

                    # I2V: reuse compiled prompt + add a tight motion directive.
                    motion = (
                        "Motion: slow camera move, subtle parallax, gentle atmospheric movement. "
                        "Keep the subject stable and match the start image exactly. "
                        "No new objects, no extra characters, no collage/mosaic."
                    )
                    i2v_prompt = (compiled + "\\n\\n" + motion).strip() if compiled else motion
                    i2v_negative = neg

                    # Store on per-shot record for downstream engines
                    if isinstance(rec, dict):
                        rec["i2v_prompt"] = i2v_prompt
                        rec["i2v_negative"] = i2v_negative
                        rec["ts_i2v"] = time.time()
                        shot_map[sid] = rec

                    out_txt.append(f"{sid}: {i2v_prompt}")
                    out_list.append({"id": sid, "prompt": i2v_prompt, "negative": i2v_negative})

                _safe_write_text(i2v_prompts_path, "\n".join(out_txt).strip() + "\n")
                _safe_write_json(i2v_prompts_json, {"shots": out_list})

                manifest["paths"]["i2v_prompts_txt"] = i2v_prompts_path
                manifest["paths"]["i2v_prompts_json"] = i2v_prompts_json
                manifest["shots"] = shot_map
                _safe_write_json(manifest_path, manifest)

            if _file_ok(i2v_prompts_path, 10) and _file_ok(i2v_prompts_json, 10):
                _skip("I2V prompts (from shots)", "i2v prompts already exist")
            else:
                _run("I2V prompts (from shots)", step_i2v_prompts, 70)

            # Step F: Video clips (HunyuanVideo 1.5)
            # Creates stable per-shot MP4s for later assembly (Chunk 6).
            clips_manifest_path = os.path.join(clips_dir, "clips_manifest.json")

            def _hunyuan15_env_python() -> Path:
                # MUST use python.exe from the tool's own /.hunyuan15_env/
                return _root() / ".hunyuan15_env" / "Scripts" / "python.exe"

            def _run_hunyuan15_clip(*, prompt: str, negative: str, image_path: str, out_path: str,
                                    fps: int, frames: int, steps: int, seed: Optional[int],
                                    target_size: int, model_key: str, bitrate_kbps: int = 2000,
                                    attn_backend: str = "auto", cpu_offload: bool = True, vae_tiling: bool = True) -> None:
                py = _hunyuan15_env_python()
                if not py.exists():
                    raise RuntimeError(
                        f"HunyuanVideo 1.5 environment not found: {py}\n"
                        "Install it first via Tools → HunyuanVideo 1.5 → Install/Update Cuda."
                    )

                cli = _root() / "helpers" / "hunyuan15_cli.py"
                if not cli.exists():
                    raise RuntimeError(f"Missing CLI: {cli}")

                args = [
                    str(py),
                    str(cli),
                    "generate",
                    "--model", str(model_key),
                    "--prompt", str(prompt),
                    "--negative", str(negative or ""),
                    "--image", str(image_path),
                    "--output", str(out_path),
                    "--fps", str(int(fps)),
                    "--frames", str(int(frames)),
                    "--steps", str(int(steps)),
                    "--bitrate-kbps", str(int(bitrate_kbps)),
                    "--auto-aspect",
                ]
                if int(target_size) > 0:
                    args += ["--target-size", str(int(target_size))]
                # Required advanced settings (match UI defaults)
                args += ["--attn", str(attn_backend or "auto")]
                if bool(cpu_offload):
                    args += ["--offload"]
                if bool(vae_tiling):
                    args += ["--tiling"]
                if seed is not None:
                    args += ["--seed", str(int(seed))]
                # Run
                subprocess.run(args, cwd=str(_root()), check=True)

            def step_video_clips_hunyuan15() -> None:
                shots = _load_shots_list(shots_path)
                if not shots:
                    raise RuntimeError("shots.json is empty; cannot generate video clips")
                shots = _load_shots_list(shots_path)
                if not shots:
                    raise RuntimeError("shots.json is empty; cannot generate video clips")
                # VRAM guard: ensure prior in-process models release GPU memory before Hunyuan starts
                _vram_release("before hunyuan15")

                # Map shot id -> image path (from manifest images list if available)
                id_to_img: Dict[str, str] = {}
                try:
                    imgs = manifest.get("paths", {}).get("images")
                    if isinstance(imgs, list):
                        for it in imgs:
                            if isinstance(it, dict) and it.get("id") and it.get("file"):
                                id_to_img[str(it["id"])] = str(it["file"])
                except Exception:
                    pass

                shot_map = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
                prof = dict(gen_profile) if isinstance(gen_profile, dict) else {}
                fps = int(prof.get("fps") or 20)
                steps = int(prof.get("steps") or 10)
                max_frames = int(prof.get("max_frames") or 61)
                target_size = int(prof.get("target_size") or 0)
                model_key = str(prof.get("model_key") or "480p_i2v_step_distilled")

                # Clip fingerprint (stable skip when unchanged)
                shots_blob = json.dumps(shots, sort_keys=True)
                imgs_fp = _sha1_text(json.dumps(sorted(list(id_to_img.items())), sort_keys=True))
                clip_fingerprint = _sha1_text(json.dumps({
                    "shots": shots_blob,
                    "images_fp": imgs_fp,
                    "profile": prof,
                    "i2v_prompts_json": _sha1_file(i2v_prompts_json) if os.path.exists(i2v_prompts_json) else "",
                }, sort_keys=True))

                prior = _safe_read_json(clips_manifest_path) if os.path.exists(clips_manifest_path) else {}
                if isinstance(prior, dict) and prior.get("fingerprint") == clip_fingerprint:
                    # Verify files exist
                    ok = True
                    lst = prior.get("clips") if isinstance(prior.get("clips"), list) else []
                    for it in lst:
                        if not isinstance(it, dict):
                            ok = False
                            break
                        p = it.get("file")
                        if not p or not os.path.isfile(str(p)):
                            ok = False
                            break
                    if ok and lst:
                        manifest.setdefault("paths", {})["clips_dir"] = clips_dir
                        manifest["paths"]["clips"] = lst
                        manifest.setdefault("settings", {})["video_engine"] = "hunyuan15"
                        manifest["settings"]["video_model"] = model_key
                        manifest["settings"]["video_profile"] = prof
                        
                        # Ensure per-shot clip fields exist (for review/debugging)
                        try:
                            sm = shot_map if isinstance(shot_map, dict) else {}
                            if isinstance(lst, list) and isinstance(sm, dict):
                                for _it in lst:
                                    if isinstance(_it, dict) and _it.get('id') and _it.get('file'):
                                        _sid2 = str(_it.get('id') or '').strip()
                                        if not _sid2:
                                            continue
                                        _rec2 = sm.get(_sid2) if isinstance(sm.get(_sid2), dict) else {}
                                        if not isinstance(_rec2, dict):
                                            _rec2 = {}
                                        _rec2['clip_file'] = str(_it.get('file'))
                                        if 'clip_regen_count' not in _rec2:
                                            try:
                                                _rec2['clip_regen_count'] = int(_it.get('regen_count') or 0)
                                            except Exception:
                                                _rec2['clip_regen_count'] = 0
                                        if 'clip_takes' not in _rec2:
                                            tk = _it.get('takes')
                                            _rec2['clip_takes'] = list(tk) if isinstance(tk, list) else []
                                        sm[_sid2] = _rec2
                                manifest['shots'] = sm
                        except Exception:
                            pass

                        _safe_write_json(manifest_path, manifest)
                        return  # skip

                clips_out: List[Dict[str, Any]] = []
                logs_dir = os.path.join(self.out_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)

                # Partial-resume safeguard: keep already-rendered clips and only regenerate missing/stale ones.
                # Build an index of existing clips from prior manifest and current job manifest.
                existing_by_id: Dict[str, Dict[str, Any]] = {}
                try:
                    if isinstance(prior, dict):
                        _lst2 = prior.get("clips") if isinstance(prior.get("clips"), list) else []
                        for _it in _lst2:
                            if isinstance(_it, dict) and _it.get("id") and _it.get("file"):
                                _p = str(_it.get("file"))
                                if _p and os.path.isfile(_p) and os.path.getsize(_p) >= 1024:
                                    existing_by_id[str(_it.get("id"))] = dict(_it)
                except Exception:
                    pass
                try:
                    if isinstance(shot_map, dict):
                        for _sid0, _rec0 in shot_map.items():
                            if not _sid0:
                                continue
                            if isinstance(_rec0, dict) and _rec0.get("clip_file"):
                                _p0 = str(_rec0.get("clip_file"))
                                if _p0 and os.path.isfile(_p0) and os.path.getsize(_p0) >= 1024:
                                    if str(_sid0) not in existing_by_id:
                                        existing_by_id[str(_sid0)] = {"id": str(_sid0), "file": _p0, "intent_fp": str(_rec0.get("clip_intent_fp") or "")}
                except Exception:
                    pass

                # Input mtimes used for a conservative "stale" heuristic when fingerprints are missing.
                try:
                    prompts_mtime = float(os.path.getmtime(i2v_prompts_json)) if os.path.isfile(i2v_prompts_json) else 0.0
                except Exception:
                    prompts_mtime = 0.0
                try:
                    shots_mtime = float(os.path.getmtime(shots_path)) if os.path.isfile(shots_path) else 0.0
                except Exception:
                    shots_mtime = 0.0

                for i, sh in enumerate(shots, start=1):
                    if self._stop_requested:
                        raise RuntimeError("Cancelled by user.")

                    sid = str(sh.get("id") or f"S{i:02d}")
                    rec = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                    prompt = str((rec.get("i2v_prompt") if isinstance(rec, dict) else "") or "").strip()
                    negative = str((rec.get("i2v_negative") if isinstance(rec, dict) else "") or "").strip()
                    if not prompt:
                        # Fallback: basic motion-only prompt
                        prompt = "Slow camera move, subtle parallax, keep subject stable. Match the image exactly."

                    # Own Character Bible — final injection for video prompt (i2v)
                    try:
                        if bool(_own_active) and str(_own_prose or "").strip():
                            _p0 = str(prompt or "").strip()
                            if _p0 and ("OWN CHARACTER BIBLE" not in _p0) and ("Main characters (keep consistent):" not in _p0):
                                prompt = (_p0 + " Main characters (keep consistent): " + str(_own_prose).strip() + ".").strip()
                            else:
                                prompt = _p0
                    except Exception:
                        pass
                    img_path = id_to_img.get(sid, "")
                    if not img_path:
                        # Try a simple filename guess
                        guess = os.path.join(images_dir, f"{sid}.png")
                        if os.path.isfile(guess):
                            img_path = guess
                    if not img_path or not os.path.isfile(img_path):
                        raise RuntimeError(f"Missing start image for {sid}. Expected in {images_dir} or manifest paths.")

                    # Duration -> frames (clamped to tier max)
                    try:
                        dur = float(sh.get("duration_sec") or 0.0)
                    except Exception:
                        dur = 0.0
                    if dur <= 0.0:
                        # Default to tier max
                        dur = float(max_frames) / float(max(1, fps))
                    frames = int(round(dur * float(max(1, fps))))
                    frames = max(16, min(int(max_frames), int(frames)))

                    # Stable seed per shot
                    seed = None
                    try:
                        seed = int(rec.get("seed_int")) if isinstance(rec, dict) else None
                    except Exception:
                        seed = None

                    out_file = os.path.join(clips_dir, f"shot_{i:03d}_{sid}.mp4")
                    log_file = os.path.join(logs_dir, f"hunyuan15_{i:03d}_{sid}.log")

                    had_existing_before = bool(os.path.isfile(out_file) and os.path.getsize(out_file) >= 1024)

                    # Per-shot intent fingerprint (used to safely skip already-rendered clips).
                    try:
                        _img_sig = ""
                        if img_path and os.path.isfile(img_path):
                            try:
                                _img_sig = _sha1_file(img_path) or ""
                            except Exception:
                                _img_sig = ""
                        intent_fp = _sha1_text(json.dumps({
                            "id": sid,
                            "prompt": prompt,
                            "negative": negative,
                            "image_sig": _img_sig,
                            "fps": int(fps),
                            "frames": int(frames),
                            "steps": int(steps),
                            "seed": seed,
                            "target_size": int(target_size),
                            "model_key": str(model_key),
                            "profile": prof,
                        }, sort_keys=True))
                    except Exception:
                        intent_fp = ""

                    # Skip regeneration if an acceptable existing clip is present (supports manual deletes of specific shots).
                    try:
                        if os.path.isfile(out_file) and os.path.getsize(out_file) >= 1024:
                            _ex = existing_by_id.get(sid) if isinstance(existing_by_id, dict) else None
                            _ex_fp = str(_ex.get("intent_fp") or "") if isinstance(_ex, dict) else ""
                            _rec_fp = str(rec.get("clip_intent_fp") or "") if isinstance(rec, dict) else ""
                            try:
                                _clip_mtime = float(os.path.getmtime(out_file))
                            except Exception:
                                _clip_mtime = 0.0
                            try:
                                _img_mtime = float(os.path.getmtime(img_path)) if img_path and os.path.isfile(img_path) else 0.0
                            except Exception:
                                _img_mtime = 0.0

                            _up_to_date = True
                            # Existing clip present; keep it (manual delete forces regeneration)

                            if _up_to_date:
                                self.signals.stage.emit(f"Clips (Hunyuan) — {sid} (skip {i}/{len(shots)})")
                                self.signals.log.emit(f"[hunyuan15] {sid}: existing clip found; skipping regeneration")
                                self.signals.progress.emit(min(99, 72 + int((i - 1) * (25 / max(1, len(shots))))))

                                clips_out.append({
                                    "id": sid,
                                    "file": out_file,
                                    "frames": int(frames),
                                    "fps": int(fps),
                                    "steps": int(steps),
                                    "seed": seed,
                                    "target_size": int(target_size),
                                    "model_key": str(model_key),
                                    "intent_fp": intent_fp,
                                })

                                # Persist per-shot clip metadata (used for review/debugging)
                                try:
                                    rec2 = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                                    if not isinstance(rec2, dict):
                                        rec2 = {}
                                    rec2["clip_file"] = out_file
                                    rec2["clip_intent_fp"] = intent_fp
                                    if "clip_regen_count" not in rec2:
                                        rec2["clip_regen_count"] = 0
                                    if "clip_takes" not in rec2:
                                        rec2["clip_takes"] = []
                                    # Don't bump ts_clip_done here; keep original if present.
                                    shot_map[sid] = rec2
                                except Exception:
                                    pass

                                # Update manifest incrementally (so the reviewer can open immediately)
                                manifest.setdefault("paths", {})["clips_dir"] = clips_dir
                                manifest["paths"]["clips"] = clips_out
                                manifest.setdefault("settings", {})["video_engine"] = "hunyuan15"
                                manifest["settings"]["video_model"] = model_key
                                manifest["settings"]["video_profile"] = prof
                                _safe_write_json(manifest_path, manifest)

                                # Live preview: show clip as soon as it's available.
                                try:
                                    if out_file and os.path.isfile(out_file):
                                        self.signals.asset_created.emit(out_file)
                                except Exception:
                                    pass
                                continue
                    except Exception:
                        pass

                    self.signals.stage.emit(f"Clips (Hunyuan) — {sid} ({i}/{len(shots)})")
                    self.signals.log.emit(f"[hunyuan15] {sid}: {frames} frames @ {fps} fps, target_size={target_size}, steps={steps}")
                    self.signals.progress.emit(min(99, 72 + int((i - 1) * (25 / max(1, len(shots))))))

                    # Capture stdout/stderr
                    try:
                        with open(log_file, "w", encoding="utf-8", errors="ignore") as lf:
                            lf.write(f"[planner] {sid}\n")
                            lf.write(f"[planner] out: {out_file}\n")
                            lf.write(f"[planner] img: {img_path}\n")
                            lf.write(f"[planner] frames: {frames} fps: {fps} steps: {steps} seed: {seed}\n")
                            lf.write("\n")
                            lf.flush()
                            # Run clip
                            _run_hunyuan15_clip(
                                prompt=prompt,
                                negative=negative,
                                image_path=img_path,
                                out_path=out_file,
                                fps=fps,
                                frames=frames,
                                steps=steps,
                                seed=seed,
                                target_size=target_size,
                                model_key=model_key,
                                attn_backend=str(prof.get("attn_backend") or "auto"),
                                cpu_offload=bool(prof.get("cpu_offload", True)),
                                vae_tiling=bool(prof.get("vae_tiling", True)),
                            )
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"HunyuanVideo failed for {sid}. See log: {log_file}\n{e}") from e

                    if not os.path.isfile(out_file) or os.path.getsize(out_file) < 1024:
                        raise RuntimeError(f"Clip output missing/too small for {sid}: {out_file}")
                    clips_out.append({
                        "id": sid,
                        "file": out_file,
                        "frames": int(frames),
                        "fps": int(fps),
                        "steps": int(steps),
                        "seed": seed,
                        "target_size": int(target_size),
                        "model_key": str(model_key),
                        "intent_fp": intent_fp,
                    })

                    # Persist per-shot clip metadata (used for review/debugging)
                    try:
                        rec2 = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                        if not isinstance(rec2, dict):
                            rec2 = {}
                        rec2["clip_file"] = out_file
                        rec2["clip_intent_fp"] = intent_fp
                        # Regen count: bump only if we overwrote an existing clip for this shot.
                        try:
                            cur = int(rec2.get("clip_regen_count") or 0)
                        except Exception:
                            cur = 0
                        rec2["clip_regen_count"] = (cur + 1) if bool(had_existing_before) else cur
                        if "clip_takes" not in rec2:
                            rec2["clip_takes"] = []
                        rec2["ts_clip_done"] = time.time()
                        shot_map[sid] = rec2
                    except Exception:
                        pass

                    # Update manifest incrementally
                    manifest.setdefault("paths", {})["clips_dir"] = clips_dir
                    manifest["paths"]["clips"] = clips_out
                    manifest.setdefault("settings", {})["video_engine"] = "hunyuan15"
                    manifest["settings"]["video_model"] = model_key
                    manifest["settings"]["video_profile"] = prof
                    _safe_write_json(manifest_path, manifest)

                    # Live preview: show clip immediately after it is written.
                    try:
                        if out_file and os.path.isfile(out_file):
                            self.signals.asset_created.emit(out_file)
                    except Exception:
                        pass

                # Final clips manifest (for resume/skip)
                _safe_write_json(clips_manifest_path, {"fingerprint": clip_fingerprint, "clips": clips_out, "profile": prof})
                manifest.setdefault("paths", {})["clips_dir"] = clips_dir
                manifest["paths"]["clips"] = clips_out
                _safe_write_json(manifest_path, manifest)


            def step_video_clips_wan22() -> None:
                shots = _load_shots_list(shots_path)
                shot_map = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
                prof = _resolve_generation_profile(self.job.encoding.get("video_model") or "", self.job.encoding.get("gen_quality_preset") or "")

                # Select size string based on aspect (WAN uses explicit WxH strings)
                aspect = str(self.job.encoding.get("aspect") or "")
                size_land = str(prof.get("size_landscape") or "1280*704")
                size_port = str(prof.get("size_portrait") or "704*1280")
                if "portrait" in aspect.lower():
                    size_str = size_port
                else:
                    size_str = size_land

                fps = int(prof.get("fps") or 20)
                steps = int(prof.get("steps") or 25)
                guidance = float(prof.get("guidance") or 4.0)
                offload_model = bool(prof.get("offload_model", True))

                # Fingerprint for resume/skip
                clip_fingerprint = _sha1_text(json.dumps({
                    "engine": "wan22",
                    "profile": prof,
                    "shots_sha1": _sha1_file(shots_path) if os.path.exists(shots_path) else "",
                    "i2v_sha1": _sha1_file(i2v_prompts_json) if os.path.exists(i2v_prompts_json) else "",
                    "images_dir_fp": _file_fingerprint(images_dir),
                }, sort_keys=True))

                # If clips_manifest exists and matches fingerprint, we can skip
                try:
                    if os.path.exists(clips_manifest_path):
                        cm = _safe_read_json(clips_manifest_path)
                        if isinstance(cm, dict) and str(cm.get("fingerprint") or "") == clip_fingerprint:
                            # Ensure clips list is also present in main manifest
                            clips_out = cm.get("clips") if isinstance(cm.get("clips"), list) else []
                            manifest.setdefault("paths", {})["clips_dir"] = clips_dir
                            manifest["paths"]["clips"] = clips_out
                            manifest.setdefault("settings", {})["video_engine"] = "wan22"
                            manifest["settings"]["video_profile"] = prof
                            _safe_write_json(manifest_path, manifest)
                            self.signals.log.emit("[wan22] clips already match fingerprint; skipping render")
                            return
                except Exception:
                    pass

                clips_out: List[Dict[str, Any]] = []

                # Reuse existing clips if present & intent matches
                existing_intents: Dict[str, str] = {}
                try:
                    cm = _safe_read_json(clips_manifest_path) if os.path.exists(clips_manifest_path) else {}
                    if isinstance(cm, dict):
                        for it in (cm.get("clips") or []):
                            if isinstance(it, dict) and it.get("id") and it.get("intent_fp"):
                                existing_intents[str(it["id"])] = str(it["intent_fp"])
                                clips_out.append(it)
                except Exception:
                    clips_out = []

                for i, sh in enumerate(shots, start=1):
                    if not isinstance(sh, dict):
                        continue
                    sid = str(sh.get("id") or "").strip()
                    if not sid:
                        continue

                    # Input image
                    img_path = ""
                    try:
                        # Prefer manifest shot image file if present
                        rec = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                        if isinstance(rec, dict) and rec.get("file"):
                            img_path = str(rec.get("file"))
                    except Exception:
                        img_path = ""

                    if not img_path:
                        # fallback: <images_dir>/<sid>.png
                        cand = os.path.join(images_dir, f"{sid}.png")
                        if os.path.exists(cand):
                            img_path = cand

                    if not img_path or not os.path.exists(img_path):
                        raise RuntimeError(f"[wan22] Missing start image for shot {sid}: {img_path}")

                    # Prompts
                    rec = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                    if not isinstance(rec, dict):
                        rec = {}
                    prompt = str(rec.get("i2v_prompt") or "").strip()
                    negative = str(rec.get("i2v_negative") or "").strip()

                    # Duration → frames
                    try:
                        dsec = float(sh.get("duration_sec") or 0.0)
                    except Exception:
                        dsec = 0.0
                    if dsec <= 0.0:
                        dsec = _stable_uniform(f"{self.job.job_id}:{sid}:dur", float(prof.get("min_sec") or 2.5), float(prof.get("max_sec") or 4.0))
                    frames = max(12, int(round(dsec * fps)))

                    # Seed (stable)
                    seed_txt = str(sh.get("seed") or sid)
                    seed = _seed_to_int(seed_txt)

                    out_file = os.path.join(clips_dir, f"{sid}.mp4")
                    log_file = os.path.join(clips_dir, f"{sid}.wan22.log.txt")

                    intent_fp = _sha1_text(json.dumps({
                        "sid": sid,
                        "img_fp": _file_fingerprint(img_path),
                        "prompt": prompt,
                        "negative": negative,
                        "fps": fps,
                        "frames": frames,
                        "steps": steps,
                        "guidance": guidance,
                        "seed": seed,
                        "size": size_str,
                        "offload_model": offload_model,
                    }, sort_keys=True))

                    # Skip if output exists and intent matches previous
                    if os.path.isfile(out_file) and os.path.getsize(out_file) > 1024:
                        prev = existing_intents.get(sid) or str(rec.get("clip_intent_fp") or "")
                        if prev and prev == intent_fp:
                            self.signals.log.emit(f"[wan22] {sid}: reusing existing clip (intent match)")
                            clips_out.append({
                                "id": sid,
                                "file": out_file,
                                "frames": int(frames),
                                "fps": int(fps),
                                "steps": int(steps),
                                "seed": seed,
                                "size": str(size_str),
                                "guidance": float(guidance),
                                "intent_fp": intent_fp,
                            })
                            # Live preview: show clip immediately when reused.
                            try:
                                if out_file and os.path.isfile(out_file):
                                    self.signals.asset_created.emit(out_file)
                            except Exception:
                                pass
                            continue

                    self.signals.stage.emit(f"Clips (WAN 2.2) — {sid} ({i}/{len(shots)})")
                    self.signals.log.emit(f"[wan22] {sid}: {frames} frames @ {fps} fps, size={size_str}, steps={steps}, guidance={guidance}")
                    self.signals.progress.emit(min(99, 72 + int((i - 1) * (25 / max(1, len(shots))))))

                    _run_wan22_clip(
                        prompt=prompt,
                        negative=negative,
                        image_path=img_path,
                        out_path=out_file,
                        fps=fps,
                        frames=frames,
                        steps=steps,
                        seed=seed,
                        size_str=size_str,
                        guidance=guidance,
                        offload_model=offload_model,
                        t5_cpu=bool(prof.get("t5_cpu", False)),
                        log_path=log_file,
                    )

                    if not os.path.isfile(out_file) or os.path.getsize(out_file) < 1024:
                        raise RuntimeError(f"WAN 2.2 clip output missing/too small for {sid}: {out_file}")

                    clips_out.append({
                        "id": sid,
                        "file": out_file,
                        "frames": int(frames),
                        "fps": int(fps),
                        "steps": int(steps),
                        "seed": seed,
                        "size": str(size_str),
                        "guidance": float(guidance),
                        "intent_fp": intent_fp,
                    })

                    # Persist to shot record
                    try:
                        rec2 = shot_map.get(sid) if isinstance(shot_map, dict) else {}
                        if not isinstance(rec2, dict):
                            rec2 = {}
                        rec2["clip_file"] = out_file
                        rec2["clip_intent_fp"] = intent_fp
                        rec2["ts_clip_done"] = time.time()
                        shot_map[sid] = rec2
                    except Exception:
                        pass

                    manifest.setdefault("paths", {})["clips_dir"] = clips_dir
                    manifest["paths"]["clips"] = clips_out
                    manifest.setdefault("settings", {})["video_engine"] = "wan22"
                    manifest["settings"]["video_profile"] = prof
                    _safe_write_json(manifest_path, manifest)

                    # Live preview: show clip immediately after it is written.
                    try:
                        if out_file and os.path.isfile(out_file):
                            self.signals.asset_created.emit(out_file)
                    except Exception:
                        pass

                # Final write
                _safe_write_json(clips_manifest_path, {"fingerprint": clip_fingerprint, "clips": clips_out, "profile": prof})
                manifest.setdefault("paths", {})["clips_dir"] = clips_dir
                manifest["paths"]["clips"] = clips_out
                manifest["shots"] = shot_map
                manifest.setdefault("settings", {})["video_engine"] = "wan22"
                manifest["settings"]["video_profile"] = prof
                _safe_write_json(manifest_path, manifest)

            # Run or skip based on selected video model
            mk = _video_model_key(self.job.encoding.get("video_model") or "")
            if mk == "wan22":
                _run("Video clips (WAN 2.2)", step_video_clips_wan22, 95)
            elif mk == "hunyuan":
                _run("Video clips (HunyuanVideo 1.5)", step_video_clips_hunyuan15, 95)
            else:
                _skip("Video clips", "Video model not wired yet")

            # Review gate (Chunk 8A placeholder): optionally pause after clips for review (video editing not implemented yet)
            if bool((self.job.encoding or {}).get("allow_edit_while_running")):
                self._clip_review_gate(
                    output_dir=self.out_dir,
                    manifest_path=manifest_path,
                    clips_dir=clips_dir,
                    shots_path=shots_path,
                    images_dir=images_dir,
                )

            # Step G: Assemble final video (Chunk 6) — timeline → final_cut
            final_video = os.path.join(final_dir, f"{self.job.job_id}_final.mp4")
            timeline_json = os.path.join(final_dir, "timeline.json")
            assembly_log = os.path.join(final_dir, "assembly_log.txt")

            # Assembly fingerprint: if clips + shot durations + encode settings didn't change, allow skip.
            _clip_fingerprint = ""
            try:
                _cm = _safe_read_json(clips_manifest_path) if os.path.exists(clips_manifest_path) else {}
                if isinstance(_cm, dict):
                    _clip_fingerprint = str(_cm.get("fingerprint") or "")
            except Exception:
                _clip_fingerprint = ""

            try:
                _clips_list = (manifest.get("paths") or {}).get("clips") or []
            except Exception:
                _clips_list = []

            try:
                _clips_stats = []
                if isinstance(_clips_list, list):
                    for _it in _clips_list:
                        if isinstance(_it, dict) and _it.get("file"):
                            _clips_stats.append(_file_fingerprint(str(_it.get("file"))))
            except Exception:
                _clips_stats = []

            _assembly_fingerprint = _sha1_text(json.dumps({
                "clips_fingerprint": _clip_fingerprint,
                "clips_stats": _clips_stats,
                "shots_sha1": _sha1_file(shots_path) if os.path.exists(shots_path) else "",
                "encode": {
                    "mode": self.job.encoding.get("mode"),
                    "crf": self.job.encoding.get("crf"),
                    "bitrate_kbps": self.job.encoding.get("bitrate_kbps"),
                },
            }, sort_keys=True))

            _assemble_step_name = "Assemble final video (timeline → final_cut)"

            # Progress window for the Assemble step is computed later based on which post-steps are enabled.
            _assemble_progress_win = {"start": 95, "end": 99}

            def step_final() -> None:
                # NOTE: This step does NOT add audio yet (Chunk 6B). It creates a stable visual "final cut".
                clips = (manifest.get("paths") or {}).get("clips")
                if not isinstance(clips, list) or not clips:
                    raise RuntimeError("No clips found. Run 'Video clips' step first (Chunk 5).")

                shots = _load_shots_list(shots_path)
                shot_durs: Dict[str, float] = {}
                for sh in shots:
                    if not isinstance(sh, dict):
                        continue
                    sid = str(sh.get("id") or "").strip()
                    if not sid:
                        continue
                    try:
                        d = float(sh.get("duration_sec") or 0.0)
                    except Exception:
                        d = 0.0
                    shot_durs[sid] = max(0.0, d)

                def _tool(exe_name: str) -> str:
                    # Prefer tool-bundled ffmpeg/ffprobe under <root>/presets/bin/
                    b = _root() / "presets" / "bin"
                    cands = []
                    if os.name == "nt":
                        cands += [b / f"{exe_name}.exe", b / f"{exe_name}.bat", b / exe_name]
                    else:
                        cands += [b / exe_name]
                    for p in cands:
                        try:
                            if p.exists():
                                return str(p)
                        except Exception:
                            pass
                    # Fallback to PATH
                    return exe_name

                ffprobe = _tool("ffprobe")
                ffmpeg = _tool("ffmpeg")

                def _run_cmd(args: List[str]) -> None:
                    if _LOGGER.enabled:
                        _LOGGER.log_job(self.job.job_id, "[ffmpeg] " + " ".join([str(a) for a in args]))
                    subprocess.run(args, cwd=str(_root()), check=True)

                def _probe_json(media_path: str) -> Dict[str, Any]:
                    args = [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(media_path)]
                    cp = subprocess.run(args, cwd=str(_root()), capture_output=True, text=True)
                    if cp.returncode != 0:
                        raise RuntimeError(f"ffprobe failed for: {media_path}\n{cp.stderr.strip()}")
                    try:
                        obj = json.loads(cp.stdout or "")
                        return obj if isinstance(obj, dict) else {}
                    except Exception:
                        return {}

                def _parse_rate(s: str) -> float:
                    # "20/1" -> 20.0
                    s = (s or "").strip()
                    if not s:
                        return 0.0
                    if "/" in s:
                        a, b = s.split("/", 1)
                        try:
                            aa = float(a.strip() or 0.0)
                            bb = float(b.strip() or 1.0)
                            if bb == 0:
                                return 0.0
                            return aa / bb
                        except Exception:
                            return 0.0
                    try:
                        return float(s)
                    except Exception:
                        return 0.0

                def _even(n: int) -> int:
                    try:
                        n = int(n)
                    except Exception:
                        n = 0
                    return max(2, int(n) // 2 * 2)

                # Target format = first clip
                first_path = str((clips[0] or {}).get("file") or "")
                if not first_path or not os.path.isfile(first_path):
                    raise RuntimeError(f"First clip missing: {first_path}")

                first_info = _probe_json(first_path)
                vstream = None
                for st in (first_info.get("streams") or []):
                    if isinstance(st, dict) and st.get("codec_type") == "video":
                        vstream = st
                        break
                if not isinstance(vstream, dict):
                    raise RuntimeError(f"No video stream found in first clip: {first_path}")

                tw = _even(int(vstream.get("width") or 0))
                th = _even(int(vstream.get("height") or 0))
                fps = _parse_rate(str(vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or ""))
                if fps <= 0.0:
                    fps = 20.0
                target_fps = int(round(fps))
                target_fps = max(10, min(60, target_fps))

                # Encode settings for assembly
                mode = str(self.job.encoding.get("mode") or "crf").strip().lower()
                crf = int(self.job.encoding.get("crf") or 18)
                bitrate_kbps = int(self.job.encoding.get("bitrate_kbps") or 3000)

                # Normalize each clip to match first clip (fps/res) with cover-crop.
                norm_dir = os.path.join(final_dir, "_normalized")
                os.makedirs(norm_dir, exist_ok=True)

                log_lines: List[str] = []
                log_lines.append("Planner Chunk 6 — Assembly log")
                log_lines.append(f"Target: {tw}x{th} @ {target_fps}fps")
                log_lines.append(f"Encode: mode={mode} crf={crf} bitrate_kbps={bitrate_kbps}")
                log_lines.append("")

                timeline: List[Dict[str, Any]] = []
                concat_list_path = os.path.join(norm_dir, "concat_list.txt")
                concat_lines: List[str] = []

                cur_t = 0.0
                for idx, it in enumerate(clips, start=1):
                    if self._stop_requested:
                        raise RuntimeError("Cancelled by user.")

                    if not isinstance(it, dict):
                        continue
                    sid = str(it.get("id") or f"S{idx:02d}")
                    src = str(it.get("file") or "")
                    if not src or not os.path.isfile(src):
                        raise RuntimeError(f"Missing clip for {sid}: {src}")

                    # Decide desired duration
                    desired = float(shot_durs.get(sid, 0.0) or 0.0)
                    if desired <= 0.0:
                        # best-effort: use frames/fps from clip manifest, else probe duration
                        try:
                            frames = int(it.get("frames") or 0)
                            cfps = float(it.get("fps") or 0) or float(target_fps)
                            if frames > 0 and cfps > 0:
                                desired = float(frames) / float(cfps)
                        except Exception:
                            desired = 0.0
                    if desired <= 0.0:
                        # probe format duration
                        info = _probe_json(src)
                        try:
                            desired = float((info.get("format") or {}).get("duration") or 0.0)
                        except Exception:
                            desired = 0.0
                    desired = max(0.1, float(desired))

                    out_norm = os.path.join(norm_dir, f"clip_{idx:03d}_{sid}.mp4")

                    # If exists and newer than source, keep it (simple resume)
                    try:
                        if os.path.isfile(out_norm) and os.path.getmtime(out_norm) >= os.path.getmtime(src) and os.path.getsize(out_norm) > 1024:
                            log_lines.append(f"[skip] {sid} → {os.path.basename(out_norm)} (already normalized)")
                        else:
                            vf = f"scale={tw}:{th}:force_original_aspect_ratio=increase,crop={tw}:{th},fps={target_fps}"
                            args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                                    "-t", f"{desired:.3f}",
                                    "-vf", vf,
                                    "-an",
                                    "-pix_fmt", "yuv420p",
                                    "-movflags", "+faststart",
                                    "-preset", "veryfast"]
                            if mode == "bitrate":
                                args += ["-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k", "-bufsize", f"{bitrate_kbps*2}k"]
                            else:
                                args += ["-crf", str(crf)]
                            args += ["-c:v", "libx264", str(out_norm)]
                            _run_cmd(args)
                            log_lines.append(f"[norm] {sid}: {os.path.basename(src)} → {os.path.basename(out_norm)} (t={desired:.3f}s)")
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"ffmpeg normalize failed for {sid}: {src}\n{e}") from e

                    if not os.path.isfile(out_norm) or os.path.getsize(out_norm) < 1024:
                        raise RuntimeError(f"Normalized clip missing/too small for {sid}: {out_norm}")

                    # Concat list requires absolute paths; quote to allow spaces.
                    _safe_path = str(out_norm).replace("'", "\\'")
                    concat_lines.append("file '{}'".format(_safe_path))

                    timeline.append({
                        "index": int(idx),
                        "shot_id": sid,
                        "source_clip": src,
                        "normalized_clip": out_norm,
                        "start_sec": round(cur_t, 3),
                        "duration_sec": round(desired, 3),
                        "end_sec": round(cur_t + desired, 3),
                    })
                    cur_t += desired

                    # Map per-clip progress into the Assemble step window so optional post-steps
                    # (narration/music/upscale/interpolate) still have room before 100%.
                    try:
                        p0 = int((_assemble_progress_win or {}).get('start', 95))
                        p1 = int((_assemble_progress_win or {}).get('end', 99))
                    except Exception:
                        p0, p1 = 95, 99
                    p0 = max(0, min(99, p0))
                    p1 = max(p0, min(99, p1))
                    nclips = max(1, int(len(clips) if isinstance(clips, list) else 1))
                    if nclips <= 1:
                        mapped = p1
                    else:
                        span = max(1, (p1 - p0))
                        mapped = p0 + int(round(((idx - 1) / max(1, (nclips - 1))) * span))
                    self.signals.progress.emit(int(max(p0, min(p1, mapped))))

                # Write timeline + concat list
                _safe_write_json(timeline_json, {
                    "target": {"width": int(tw), "height": int(th), "fps": int(target_fps)},
                    "timeline": timeline,
                    "audio": {
                        "music_file": str((manifest.get("paths") or {}).get("music_file") or ""),
                        "note": "Audio muxing not done yet"
                    },
                })
                _safe_write_text(concat_list_path, "\n".join(concat_lines).strip() + "\n")

                # Final concat -> final_video (re-encode for safety)
                try:
                    args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                            "-f", "concat", "-safe", "0",
                            "-i", str(concat_list_path),
                            "-an",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            "-preset", "veryfast"]
                    if mode == "bitrate":
                        args += ["-b:v", f"{bitrate_kbps}k", "-maxrate", f"{bitrate_kbps}k", "-bufsize", f"{bitrate_kbps*2}k"]
                    else:
                        args += ["-crf", str(crf)]
                    args += ["-c:v", "libx264", str(final_video)]
                    _run_cmd(args)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"ffmpeg concat failed. See assembly_log: {assembly_log}\n{e}") from e

                if not os.path.isfile(final_video) or os.path.getsize(final_video) < 1024:
                    raise RuntimeError(f"Final video missing/too small: {final_video}")

                # Music : if enabled but no file, reserve a note for later.
                try:
                    if self.job.music_background and not self.job.silent and not str((manifest.get('paths') or {}).get('music_file') or ''):
                        p = os.path.join(audio_dir, "music_pending.txt")
                        if not _file_ok(p, 2):
                            _safe_write_text(p, "PENDING: Auto music bed (Ace Step) to be generated later.\n")
                        manifest.setdefault("paths", {})["music_pending_txt"] = p
                except Exception:
                    pass

                # Write log
                log_lines.append("")
                log_lines.append(f"Timeline: {timeline_json}")
                log_lines.append(f"Final: {final_video}")
                _safe_write_text(assembly_log, "\n".join(log_lines).strip() + "\n")

                # Update manifest
                manifest["final_video"] = final_video
                manifest.setdefault("paths", {})["final_video"] = final_video
                manifest["paths"]["timeline_json"] = timeline_json
                manifest["paths"]["assembly_log_txt"] = assembly_log
                manifest.setdefault("project", {}).setdefault("assembly", {})
                manifest["project"]["assembly"].update({
                    "target_width": int(tw),
                    "target_height": int(th),
                    "target_fps": int(target_fps),
                    "normalized_dir": norm_dir,
                    "clip_count": int(len(timeline)),
                    "duration_sec": round(cur_t, 3),
                })
                # Store fingerprint for idempotency
                srec = manifest["steps"].get(_assemble_step_name) or {}
                srec.update({
                    "fingerprint": _assembly_fingerprint,
                    "debug": {"timeline": timeline_json, "assembly_log": assembly_log},
                    "note": "Hard-cut concat (no audio yet).",
                    "ts": time.time(),
                })
                manifest["steps"][_assemble_step_name] = srec
                _safe_write_json(manifest_path, manifest)

            # Step H: Narration + optional user music (Chunk 6B)
            # - Generate narration script (Qwen3 text/VL JSON path)
            # - Generate narration audio (Qwen3 TTS)
            # - If user provided a background music file: loop/trim to match duration and mix under narration
            # NOTE: "Auto music creation" 
            final_cut_mp4 = os.path.join(final_dir, "final_cut.mp4")
            narration_txt = os.path.join(audio_dir, "narration.txt")
            narration_json = os.path.join(audio_dir, "narration.json")
            narration_wav = os.path.join(audio_dir, "narration.wav")
            tts_log = os.path.join(audio_dir, "tts_log.txt")
            mix_log = os.path.join(final_dir, "mix_log.txt")

            def _tool2(exe_name: str) -> str:
                b = _root() / "presets" / "bin"
                cands = []
                if os.name == "nt":
                    cands += [b / f"{exe_name}.exe", b / f"{exe_name}.bat", b / exe_name]
                else:
                    cands += [b / exe_name]
                for p2 in cands:
                    try:
                        if p2.exists():
                            return str(p2)
                    except Exception:
                        pass
                return exe_name

            ffprobe2 = _tool2("ffprobe")
            ffmpeg2 = _tool2("ffmpeg")

            def _audio_file_ok_lenient(audio_path: str, min_bytes: int = 256) -> bool:
                """Lenient audio validation for short test runs.

                Planner pipelines often test with very short music targets (e.g. 5–20s).
                Some music backends can produce perfectly valid audio files that are
                smaller than our historical "1024 bytes" guard.

                Accept if:
                - file exists
                - has at least a small number of bytes
                - and, when ffprobe can parse it, contains an audio stream.

                If ffprobe can't parse (or isn't present), we still allow the pipeline
                to continue so downstream ffmpeg will provide the actionable error.
                """
                try:
                    if (not audio_path) or (not os.path.isfile(audio_path)):
                        return False
                    sz = int(os.path.getsize(audio_path) or 0)
                    if sz >= 1024:
                        return True
                    if sz < int(min_bytes):
                        return False
                except Exception:
                    return False

                try:
                    args = [ffprobe2, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(audio_path)]
                    cp = subprocess.run(args, cwd=str(_root()), capture_output=True, text=True)
                    if cp.returncode == 0:
                        obj = json.loads(cp.stdout or "{}")
                        for st in (obj.get("streams") or []):
                            try:
                                if str((st or {}).get("codec_type") or "").lower() == "audio":
                                    return True
                            except Exception:
                                continue
                except Exception:
                    pass

                return True

            def _probe_duration_sec(video_path: str) -> float:
                # Primary: JSON show_format
                args = [ffprobe2, "-v", "error", "-print_format", "json", "-show_format", str(video_path)]
                cp = subprocess.run(args, cwd=str(_root()), capture_output=True, text=True)
                if cp.returncode == 0:
                    try:
                        obj = json.loads(cp.stdout or "{}")
                        dur = float(((obj.get("format") or {}).get("duration") or 0.0))
                        if dur > 0.0:
                            return float(dur)
                    except Exception:
                        pass

                # Fallback 1: plain duration value (format=duration)
                try:
                    args2 = [ffprobe2, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
                    cp2 = subprocess.run(args2, cwd=str(_root()), capture_output=True, text=True)
                    if cp2.returncode == 0:
                        s = (cp2.stdout or "").strip().splitlines()[-1].strip() if (cp2.stdout or "").strip() else ""
                        dur2 = float(s) if s else 0.0
                        if dur2 > 0.0:
                            return float(dur2)
                except Exception:
                    pass

                # Fallback 2: stream duration (v:0)
                try:
                    args3 = [ffprobe2, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
                    cp3 = subprocess.run(args3, cwd=str(_root()), capture_output=True, text=True)
                    if cp3.returncode == 0:
                        s = (cp3.stdout or "").strip().splitlines()[-1].strip() if (cp3.stdout or "").strip() else ""
                        dur3 = float(s) if s else 0.0
                        if dur3 > 0.0:
                            return float(dur3)
                except Exception:
                    pass

                return 0.0

            # Compute duration for looping/trimming music and padding narration.
            # IMPORTANT:
            # - We prefer the *actual assembled video* duration when it is longer than the stored manifest value.
            # - This prevents a "30s lock" when the plan/manifest duration is shorter than the real visual duration
            #   (e.g. user-selected longer clips, or jobs resumed with a new final video).
            try:
                duration_sec = float(((manifest.get("project") or {}).get("assembly") or {}).get("duration_sec") or 0.0)
            except Exception:
                duration_sec = 0.0

            # Probe the assembled visual regardless (best-effort). If probe is longer, trust the probe.
            dur_probe = 0.0
            try:
                dur_probe = float(_probe_duration_sec(final_video) or 0.0)
            except Exception:
                dur_probe = 0.0

            if dur_probe > 0.01:
                if duration_sec <= 0.01:
                    duration_sec = dur_probe
                elif dur_probe > (duration_sec + 1.0):
                    # If the real video is longer than the stored duration, follow the real video.
                    duration_sec = dur_probe

            duration_sec = max(0.0, float(duration_sec))

            # Chunk 7A: Background music (Ace-Step) automation
            _ace_step_name = "Music (Ace background) "

            _ACE_STYLE_MAP = {
                "Cinematic": "epic trailer music, hybrid orchestra, big percussion, braams, 80 BPM, dramatic, tension, heroic, rises and hits, mostly instrumental choir accents",
                "Ambient": "ambient, evolving pads, soft drones, sparse piano, no drums, 60 BPM or slower, dreamy, spacious, cinematic, instrumental soundscape",
                "Dance": "drum and bass, clean mix, breakbeat drums, rolling sub bass, reese bass, sharp snares, 172 BPM, high energy, futuristic, club, 4x4 beat",
            }

            def _detect_ace_env_dir2() -> Path:
                # Preferred: <root>/presets/extra_env/.ace_env
                # Legacy:    <root>/.ace_env
                root = _root()
                pref = root / "presets" / "extra_env" / ".ace_env"
                legacy = root / ".ace_env"

                def _py_in(d: Path) -> Path:
                    if os.name == "nt":
                        return d / "Scripts" / "python.exe"
                    return d / "bin" / "python"

                # Pick first location that actually contains python.
                for d in (pref, legacy):
                    try:
                        if _py_in(d).exists():
                            return d
                    except Exception:
                        pass

                # Otherwise, pick the first existing folder (so error messages point somewhere useful).
                for d in (pref, legacy):
                    try:
                        if d.exists():
                            return d
                    except Exception:
                        pass

                # Final fallback: preferred path (matches Optional Installs layout).
                return pref

            def _ace_python2(ace_env_dir: Path) -> Path:
                if sys.platform.startswith("win"):
                    return ace_env_dir / "Scripts" / "python.exe"
                return ace_env_dir / "bin" / "python"

            def _ace_runner2(ace_env_dir: Path) -> Path:
                # Prefer runner inside selected env. Fallbacks cover older layouts.
                root = _root()
                cand = [
                    ace_env_dir / "ACE-Step" / "framevision_ace_runner.py",
                    root / "presets" / "extra_env" / ".ace_env" / "ACE-Step" / "framevision_ace_runner.py",
                    root / ".ace_env" / "ACE-Step" / "framevision_ace_runner.py",
                ]
                for p in cand:
                    try:
                        if p.exists():
                            return p
                    except Exception:
                        pass
                return cand[0]

            def _compute_ace_fingerprint(preset: str, style_text: str, seed: int, duration_s: float, runner: Path) -> str:
                try:
                    runner_mtime = int(runner.stat().st_mtime) if runner.exists() else 0
                except Exception:
                    runner_mtime = 0
                return _sha1_text(json.dumps({
                    "mode": "ace",
                    "preset": str(preset),
                    "style_text": str(style_text),
                    "seed": int(seed),
                    "duration_sec": round(float(duration_s), 3),
                    "runner_mtime": int(runner_mtime),
                    "ace_env": str(_detect_ace_env_dir2()),
                }, sort_keys=True))

            def step_music_ace_7a() -> None:
                # Only run if background music is enabled AND the user selected Ace.
                if (not bool(getattr(self.job, "music_background", False))) or (str(getattr(self.job, "music_mode", "") or "") != "ace"):
                    raise RuntimeError("internal: step called but music mode is not ace")

                audio_out = os.path.join(audio_dir, "music_ace.wav")
                ace_log = os.path.join(audio_dir, "ace_log.txt")
                ace_meta = os.path.join(audio_dir, "ace_meta.json")

                preset = str(getattr(self.job, "music_preset", "") or "Cinematic").strip() or "Cinematic"
                if preset not in _ACE_STYLE_MAP:
                    preset = "Cinematic"
                style_text = _ACE_STYLE_MAP[preset]
                dur_s = float(duration_sec or 0.0)

                # Seed: reuse if preset + duration + style match, otherwise generate a new one (reproducible resume)
                manifest.setdefault("music", {})
                ace_state = {}
                try:
                    ace_state = (manifest.get("music") or {}).get("ace") or {}
                except Exception:
                    ace_state = {}
                prev_preset = str(ace_state.get("preset") or "")
                prev_style = str(ace_state.get("style_text") or "")
                try:
                    prev_dur = float(ace_state.get("duration_sec") or 0.0)
                except Exception:
                    prev_dur = 0.0

                seed = None
                try:
                    seed = int(ace_state.get("seed")) if str(ace_state.get("seed", "")).strip() != "" else None
                except Exception:
                    seed = None

                if seed is None or prev_preset != preset or prev_style != style_text or abs(prev_dur - dur_s) > 0.05:
                    seed = random.randint(0, 2**31 - 1)

                # Determine env + runner
                ace_env_dir = _detect_ace_env_dir2()
                ace_py = _ace_python2(ace_env_dir)
                runner = _ace_runner2(ace_env_dir)
                if not ace_py.exists():
                    raise RuntimeError(
                        f"ACE env python not found at: {ace_py}\n"
                        "Install it first via Optional Installs → Ace Step Music."
                    )
                if not runner.exists():
                    raise RuntimeError(
                        f"ACE runner script not found at: {runner}\n"
                        "Make sure ACE-Step is installed correctly."
                    )

                fingerprint = _compute_ace_fingerprint(preset, style_text, int(seed), dur_s, runner)

                # Idempotent skip if meta + output match
                if _file_ok(audio_out, 1024) and _file_ok(ace_meta, 10):
                    try:
                        prev = json.loads(_safe_read_text(ace_meta) or "{}")
                        if str(prev.get("fingerprint") or "") == fingerprint:
                            # Ensure manifest wiring
                            manifest.setdefault("paths", {})["music_file"] = audio_out
                            manifest.setdefault("music", {})
                            manifest["music"].setdefault("ace", {})
                            manifest["music"]["ace"].update({
                                "mode": "ace",
                                "preset": preset,
                                "seed": int(seed),
                                "duration_sec": round(float(dur_s), 3),
                                "style_text": style_text,
                                "fingerprint": fingerprint,
                                "output_file": audio_out,
                            })
                            _safe_write_json(manifest_path, manifest)
                            return
                    except Exception:
                        pass

                # Build ACE job JSON (minimal, fully automated)
                # Try to locate checkpoints: prefer config override, else default inside ACE env.
                ckpt_dir = None
                try:
                    cfg_path = _root() / "presets" / "setsave" / "ace.json"
                    cfg = json.loads(_safe_read_text(cfg_path) or "{}") if cfg_path.exists() else {}
                except Exception:
                    cfg = {}
                try:
                    rel = str(cfg.get("checkpoint_path") or "").strip()
                    if rel:
                        p = (_root() / rel).resolve()
                        if p.exists():
                            ckpt_dir = str(p)
                except Exception:
                    ckpt_dir = None
                if not ckpt_dir:
                    ckpt_cand = (ace_env_dir / "ACE-Step" / "checkpoints").resolve()
                    ckpt_dir = str(ckpt_cand)

                # Defaults match Ace UI runner expectations
                ace_job = {
                    "checkpoint_path": ckpt_dir,
                    "dtype": "bfloat16",
                    "torch_compile": False,
                    "cpu_offload": bool(cfg.get("cpu_offload", False)),
                    "overlapped_decode": bool(cfg.get("overlapped_decode", False)),
                    "device_id": int(cfg.get("device_id", 0) or 0),
                    "prompt": style_text,
                    "negative_prompt": "",
                    "lyrics": "[instrumental]",
                    "audio_duration": float(dur_s),
                    "infer_step": int(cfg.get("infer_step", 60) or 60),
                    "guidance_scale": float(cfg.get("guidance_scale", 15.0) or 15.0),
                    "scheduler_type": str(cfg.get("scheduler_type", "euler") or "euler"),
                    "cfg_type": str(cfg.get("cfg_type", "apg") or "apg"),
                    "manual_seeds": str(int(seed)),
                    "omega_scale": float(cfg.get("omega_scale", 10.0) or 10.0),
                    "guidance_interval": float(cfg.get("guidance_interval", 0.5) or 0.5),
                    "guidance_interval_decay": float(cfg.get("guidance_interval_decay", 0.0) or 0.0),
                    "min_guidance_scale": float(cfg.get("min_guidance_scale", 3.0) or 3.0),
                    "use_erg_tag": bool(cfg.get("use_erg_tag", True)),
                    "use_erg_lyric": bool(cfg.get("use_erg_lyric", False)),
                    "use_erg_diffusion": bool(cfg.get("use_erg_diffusion", True)),
                    "oss_steps": ", ".join(str(x) for x in (cfg.get("oss_steps") or [])) if isinstance(cfg.get("oss_steps"), (list, tuple)) else str(cfg.get("oss_steps") or ""),
                    "guidance_scale_text": float(cfg.get("guidance_scale_text", 5.0) or 5.0),
                    "guidance_scale_lyric": float(cfg.get("guidance_scale_lyric", 1.5) or 1.5),
                    "audio2audio_enable": False,
                    "ref_audio_strength": 0.0,
                    "ref_audio_input": None,
                    "output_path": str(audio_out),
                }

                # Persist minimal metadata (seed + settings) for reproducibility
                manifest.setdefault("music", {})
                manifest["music"].setdefault("ace", {})
                manifest["music"]["ace"].update({
                    "mode": "ace",
                    "preset": preset,
                    "seed": int(seed),
                    "duration_sec": round(float(dur_s), 3),
                    "style_text": style_text,
                    "fingerprint": fingerprint,
                    "output_file": audio_out,
                })
                _safe_write_json(manifest_path, manifest)

                # Write job to temp and run in ACE env
                tmp_dir = Path(tempfile.gettempdir())
                job_path = tmp_dir / f"planner_ace_job_{os.getpid()}_{int(time.time())}.json"
                _safe_write_json(str(job_path), ace_job)

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(int(ace_job.get("device_id", 0)))

                cmd = [str(ace_py), str(runner), str(job_path)]
                cp = subprocess.run(cmd, cwd=str(_root()), capture_output=True, text=True, env=env)

                # Write log (always overwrite)
                try:
                    log_txt = []
                    log_txt.append("CMD: " + " ".join(cmd))
                    log_txt.append("")
                    log_txt.append("--- STDOUT ---")
                    log_txt.append(cp.stdout or "")
                    log_txt.append("")
                    log_txt.append("--- STDERR ---")
                    log_txt.append(cp.stderr or "")
                    _safe_write_text(ace_log, "\n".join(log_txt).strip() + "\n")
                except Exception:
                    pass

                if cp.returncode != 0:
                    tail = (cp.stderr or "")[-2000:] if (cp.stderr or "") else "Unknown error"
                    raise RuntimeError(f"ACE-Step runner failed (code {cp.returncode}):\n{tail}")

                if not _audio_file_ok_lenient(audio_out):
                    raise RuntimeError(f"ACE output missing/too small: {audio_out}")

                # Save meta file
                meta_obj = {
                    "mode": "ace",
                    "preset": preset,
                    "seed": int(seed),
                    "duration_sec": round(float(dur_s), 3),
                    "style_text": style_text,
                    "output_file": audio_out,
                    "fingerprint": fingerprint,
                    "ts": time.time(),
                }
                _safe_write_json(ace_meta, meta_obj)

                # Wire into manifest for Chunk 6B mixer
                manifest.setdefault("paths", {})["music_file"] = audio_out
                _safe_write_json(manifest_path, manifest)

            
            # --- Ace Step 1.5 (new) wiring (Chunk 7A) ---
            def _ace15_detect_env_python() -> Path:
                root = _root()
                cand = [
                    # Preferred (new pack)
                    root / "environments" / ".ace_15",
                    # Legacy optional installs locations
                    root / "presets" / "extra_env" / ".ace_env",
                    root / ".ace_env",
                ]
                for d in cand:
                    try:
                        if d.exists():
                            if os.name == "nt":
                                py = d / "Scripts" / "python.exe"
                            else:
                                py = d / "bin" / "python"
                            if py.exists():
                                return py
                    except Exception:
                        pass
                # Fall back to the first candidate even if missing (used for error message)
                d0 = cand[0]
                return (d0 / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))

            def _ace15_detect_project_root() -> Path:
                root = _root()
                cand = [
                    # Preferred (new pack)
                    root / "models" / "ace_step_15" / "repo" / "ACE-Step-1.5",
                    # Some installs keep repo at root/.ace_env/
                    root / ".ace_env" / "ACE-Step-1.5",
                    root / "models" / "ace_step_15" / "ACE-Step-1.5",
                ]
                for p in cand:
                    try:
                        if p.exists():
                            return p
                    except Exception:
                        pass
                return cand[0]

            def _ace15_cli_py(project_root: Path) -> Path:
                return project_root / "cli.py"

            def _ace15_load_preset_payload(preset_id: str) -> dict:
                """Resolve 'Genre::Subgenre' into the preset payload dict (from presetmanager.json)."""
                root = _root()
                p = root / "presets" / "setsave" / "ace15presets" / "presetmanager.json"
                obj = {}
                try:
                    obj = json.loads(_safe_read_text(str(p)) or "{}") if p.exists() else {}
                except Exception:
                    obj = {}
                if not isinstance(obj, dict):
                    return {}
                genres = obj.get("genres") or {}
                if not isinstance(genres, dict) or not genres:
                    return {}

                # Pick default if missing/unknown
                pid = (preset_id or "").strip()
                if "::" in pid:
                    gk, sk = pid.split("::", 1)
                elif " / " in pid:
                    gk, sk = pid.split(" / ", 1)
                else:
                    gk, sk = "", ""

                def _first_payload() -> dict:
                    for g in sorted(genres.keys(), key=lambda s: str(s).lower()):
                        gd = genres.get(g) or {}
                        subs = (gd.get("subgenres") or {}) if isinstance(gd, dict) else {}
                        if not isinstance(subs, dict) or not subs:
                            continue
                        for s in sorted(subs.keys(), key=lambda x: str(x).lower()):
                            pd = subs.get(s) or {}
                            if isinstance(pd, dict) and pd:
                                return pd
                    return {}

                if gk and sk:
                    gd = genres.get(gk) or {}
                    subs = (gd.get("subgenres") or {}) if isinstance(gd, dict) else {}
                    pd = subs.get(sk) or {}
                    if isinstance(pd, dict) and pd:
                        return pd

                return _first_payload()

            def _ace15_toml_escape_str(s: str) -> str:
                s = (s or "").replace("\r\n", "\n").replace("\r", "\n")
                s = s.replace("\\", "\\\\").replace('"', '\\"')
                s = s.replace("\n", "\\n")
                return s

            def _ace15_toml_dumps_flat(d: dict) -> str:
                lines = []
                for k, v in d.items():
                    if v is None:
                        continue
                    if isinstance(v, bool):
                        val = "true" if v else "false"
                    elif isinstance(v, int):
                        val = str(v)
                    elif isinstance(v, float):
                        val = ("%.6f" % v).rstrip("0").rstrip(".")
                        if val == "":
                            val = "0"
                    else:
                        val = f'"{_ace15_toml_escape_str(str(v))}"'
                    lines.append(f"{k} = {val}")
                return "\n".join(lines) + "\n"

            def _compute_ace15_fingerprint(preset_key: str, preset_payload: dict, lyrics: str, seed: int, duration_s: float, cli_py: Path) -> str:
                try:
                    cli_mtime = int(cli_py.stat().st_mtime) if cli_py.exists() else 0
                except Exception:
                    cli_mtime = 0
                # Keep it stable (preset dict can be large)
                try:
                    pd_hash = _sha1_text(json.dumps(preset_payload or {}, sort_keys=True))
                except Exception:
                    pd_hash = ""
                return _sha1_text(json.dumps({
                    "mode": "ace15",
                    "preset_key": str(preset_key),
                    "preset_hash": str(pd_hash),
                    "lyrics": str(lyrics),
                    "seed": int(seed),
                    "duration_sec": round(float(duration_s), 3),
                    "cli_mtime": int(cli_mtime),
                }, sort_keys=True))

            def step_music_ace15_7a() -> None:
                nonlocal music_file
                # Only run if background music is enabled AND the user selected Ace Step 1.5.
                if (not bool(getattr(self.job, "music_background", False))) or (str(getattr(self.job, "music_mode", "") or "") != "ace15"):
                    raise RuntimeError("internal: step called but music mode is not ace15")

                # Duration: prefer the computed pipeline duration if available; otherwise fall back to the user’s requested approx duration.
                try:
                    dur_s = float(duration_sec) if duration_sec is not None else 0.0
                except Exception:
                    dur_s = 0.0
                if dur_s <= 0.1:
                    try:
                        dur_s = float(getattr(self.job, "approx_duration_sec", 0) or 0.0)
                    except Exception:
                        dur_s = 0.0
                # When Videoclip Creator is the assembly step, we must follow the intended project duration (not the placeholder pre-assembly output).
                try:
                    if bool(_use_videoclip_creator):
                        dur_s = float(getattr(self.job, "approx_duration_sec", dur_s) or dur_s)
                except Exception:
                    pass
                # Final safety clamp (ACE expects a sensible positive duration in seconds)
                if dur_s <= 0.1:
                    dur_s = 15.0
                dur_s = float(max(5.0, min(240.0, dur_s)))

                preset_key = str(getattr(self.job, "ace15_preset_id", "") or "").strip()
                payload = _ace15_load_preset_payload(preset_key)
                if not payload:
                    raise RuntimeError("Ace Step 1.5: presetmanager.json missing or empty. Create presets first in the Ace Step 1.5 tool.")

                envpy = _ace15_detect_env_python()
                proj = _ace15_detect_project_root()
                clipy = _ace15_cli_py(proj)
                if not envpy.exists():
                    raise RuntimeError(
                        f"Ace Step 1.5 env python not found at: {envpy}\n"
                        "Install it first via Optional Installs → Ace Step 1.5 Music."
                    )
                if not clipy.exists():
                    raise RuntimeError(
                        f"Ace Step 1.5 cli.py not found at: {clipy}\n"
                        "Make sure ACE-Step-1.5 repo is installed correctly."
                    )

                out_tmp = os.path.join(audio_dir, "ace15_out")
                Path(out_tmp).mkdir(parents=True, exist_ok=True)

                ace_fmt = str(getattr(self.job, "ace15_audio_format", "wav") or "wav").lower().strip()
                if ace_fmt not in ("mp3","wav"):
                    ace_fmt = "wav"

                audio_out = os.path.join(audio_dir, f"music_ace15.{ace_fmt}")
                music_file = str(audio_out)
                ace_log = os.path.join(audio_dir, "ace15_log.txt")
                ace_meta = os.path.join(audio_dir, "ace15_meta.json")

                # Lyrics wiring
                lyrics_enabled = bool(getattr(self.job, "ace15_lyrics_enabled", False))
                lyrics_text = str(getattr(self.job, "ace15_lyrics_text", "") or "").strip()

                # Seed handling (resume friendly)
                manifest.setdefault("music", {})
                ace_state = {}
                try:
                    ace_state = (manifest.get("music") or {}).get("ace15") or {}
                except Exception:
                    ace_state = {}

                seed = None
                try:
                    seed = int(ace_state.get("seed")) if str(ace_state.get("seed", "")).strip() != "" else None
                except Exception:
                    seed = None
                if seed is None:
                    seed = random.randint(0, 2**31 - 1)

                fingerprint = _compute_ace15_fingerprint(preset_key, payload, lyrics_text if lyrics_enabled else "", int(seed), dur_s, clipy)

                # Idempotent skip
                if _file_ok(audio_out, 1024) and _file_ok(ace_meta, 10):
                    try:
                        prev = json.loads(_safe_read_text(ace_meta) or "{}")
                        if str(prev.get("fingerprint") or "") == fingerprint:
                            manifest.setdefault("paths", {})["music_file"] = audio_out
                            manifest.setdefault("music", {})
                            manifest["music"].setdefault("ace15", {})
                            manifest["music"]["ace15"].update({
                                "mode": "ace15",
                                "preset_key": preset_key,
                                "seed": int(seed),
                                "duration_sec": round(float(dur_s), 3),
                                "fingerprint": fingerprint,
                                "output_file": audio_out,
                            })
                            _safe_write_json(manifest_path, manifest)
                            return
                    except Exception:
                        pass

                # Build config.toml for ACE-Step-1.5 cli.py
                cfg = {}

                # Required / core
                cfg["task_type"] = "text2music"
                cfg["save_dir"] = str(out_tmp)
                cfg["output_dir"] = str(out_tmp)
                cfg["audio_format"] = str(getattr(self.job, "ace15_audio_format", "wav") or "wav").lower()
                cfg["duration"] = float(dur_s)
                cfg["batch_size"] = 1
                cfg["seed"] = int(seed)

                # Preset payload mapping (best-effort, keys vary by build)
                # Caption / prompt
                caption = str(payload.get("caption") or "").strip()
                if caption:
                    cfg["caption"] = caption

                # Musical controls
                try:
                    bpm = int(payload.get("bpm") or 0)
                except Exception:
                    bpm = 0
                if bpm > 0:
                    cfg["bpm"] = bpm
                # timesignature/keyscale (allow auto)
                ts = str(payload.get("timesignature") or payload.get("time_sig") or "auto").strip() or "auto"
                ks = str(payload.get("keyscale") or payload.get("key_scale") or "auto").strip() or "auto"
                if ts and ts.lower() != "auto":
                    cfg["timesignature"] = ts
                if ks and ks.lower() != "auto":
                    cfg["keyscale"] = ks

                # Backend and LM toggles
                backend = str(payload.get("backend") or "").strip()
                if backend:
                    cfg["backend"] = backend

                thinking = bool(payload.get("thinking", False))
                cfg["thinking"] = bool(thinking)

                # Negative prompt for LM guidance
                neg = str(payload.get("lm_negative_prompt") or payload.get("negatives") or "").strip()
                if neg:
                    cfg["lm_negative_prompt"] = neg

                # COT toggles
                for k in ("use_cot_caption", "use_cot_language", "use_cot_metas"):
                    if k in payload:
                        cfg[k] = bool(payload.get(k))

                # Model selections
                main_sel = str(payload.get("main_model_path") or payload.get("main_model") or payload.get("dit_model") or "").strip()
                if main_sel:
                    cfg["main_model_path"] = main_sel
                    # Important: some ACE builds route DiT selection through config_path (SFT fix)
                    cfg["config_path"] = main_sel
                    # Compatibility
                    cfg["main_model"] = main_sel
                    cfg["dit_model"] = main_sel

                lm_sel = str(payload.get("lm_model_path") or payload.get("lm_model") or "").strip()
                if lm_sel:
                    cfg["lm_model_path"] = lm_sel
                    cfg["lm_model"] = lm_sel

                # Optional generation controls
                try:
                    gs = float(payload.get("guidance_scale") or payload.get("guidance") or 0.0)
                except Exception:
                    gs = 0.0
                if gs and gs > 0.0:
                    cfg["guidance_scale"] = float(gs)

                im = str(payload.get("infer_method") or payload.get("infer_method") or "").strip()
                if im and im.lower() != "auto":
                    cfg["infer_method"] = im

                try:
                    steps = int(payload.get("inference_steps") or payload.get("steps") or 0)
                except Exception:
                    steps = 0
                if steps and steps > 0:
                    cfg["inference_steps"] = int(steps)

                # Lyrics / instrumental
                if not lyrics_enabled:
                    cfg["instrumental"] = True
                    cfg["lyrics"] = "[Instrumental]"
                    cfg["use_cot_lyrics"] = False
                else:
                    cfg["instrumental"] = False
                    # If user left lyrics empty, let ACE try auto-lyrics (requires thinking in many builds).
                    if lyrics_text:
                        cfg["lyrics"] = lyrics_text
                        cfg["use_cot_lyrics"] = False
                    else:
                        cfg["lyrics"] = None
                        cfg["use_cot_lyrics"] = True if bool(thinking) else False

                # Write config file into temp out dir (so it gets preserved with outputs)
                cfg_path = os.path.join(out_tmp, f"planner_ace15_{int(time.time())}.toml")
                _safe_write_text(cfg_path, _ace15_toml_dumps_flat(cfg))

                env_run = os.environ.copy()
                env_run["PYTHONIOENCODING"] = "utf-8"
                env_run["PYTHONUTF8"] = "1"
                # Force UTF-8 mode to avoid Windows UnicodeEncodeError in ACE-Step console output
                cmd = [str(envpy), "-X", "utf8", str(clipy), "-c", str(cfg_path)]
                _ace_t0 = time.time()

                # Run ACE-Step 1.5 with stdin so we can auto-continue when it waits for Enter.
                out_lines = []
                rc = 0
                try:
                    proc = subprocess.Popen(
                        cmd,
                        cwd=str(proj),
                        env=env_run,
                        stdout=subprocess.PIPE,
                        stdin=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        bufsize=1,
                    )
                    assert proc.stdout is not None
                    assert proc.stdin is not None
                    for line in proc.stdout:
                        if self._stop_requested:
                            try:
                                proc.terminate()
                            except Exception:
                                pass
                            try:
                                proc.kill()
                            except Exception:
                                pass
                            raise RuntimeError("Cancelled by user.")
                        out_lines.append((line or "").rstrip("\n"))

                        # Auto-continue for ACE-Step interactive draft prompt (it writes instruction.txt and waits for Enter).
                        if "Press Enter when ready to continue." in line:
                            try:
                                proc.stdin.write("\n")
                                proc.stdin.flush()
                                out_lines.append("NOTE: Auto-pressed Enter to continue.")
                            except Exception:
                                pass
                    rc = proc.wait()
                finally:
                    # Move repo instruction.txt into the temp output folder so it doesn't block future runs.
                    try:
                        instr = Path(proj) / "instruction.txt"
                        if instr.exists() and instr.is_file():
                            stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                            dst = Path(out_tmp) / f"instruction_used_{stamp}.txt"
                            try:
                                shutil.move(str(instr), str(dst))
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Write log (always overwrite)
                try:
                    log_txt = []
                    log_txt.append("CMD: " + " ".join(cmd))
                    log_txt.append(f"CWD: {proj}")
                    log_txt.append("")
                    log_txt.append("--- OUTPUT ---")
                    log_txt.extend(out_lines)
                    _safe_write_text(ace_log, "\n".join(log_txt).strip() + "\n")
                except Exception:
                    pass

                if rc != 0:
                    tail = ("\n".join(out_lines))[-2000:] if out_lines else "Unknown error"
                    raise RuntimeError(f"Ace Step 1.5 failed (code {rc}):\n{tail}")

                # Pick newest audio file from out_tmp and normalize name for the pipeline
                newest = ""
                try:
                    auds = list_audio_files(Path(out_tmp))
                    if auds:
                        # Prefer the requested output format if present
                        try:
                            want_ext = f".{ace_fmt}"
                        except Exception:
                            want_ext = ".wav"
                        best = None
                        for p in auds:
                            try:
                                if str(p).lower().endswith(want_ext):
                                    best = p
                                    break
                            except Exception:
                                continue
                        newest = str(best or auds[0])
                    else:
                        newest = ""
                except Exception:
                    newest = ""

                # Fallback: ACE-Step sometimes ignores output_dir and saves into its default outputs folder.
                if not newest or not os.path.isfile(newest):
                    try:
                        want_exts = {".wav", ".mp3"}
                        want_ext = f".{ace_fmt}".lower().strip()
                        if want_ext not in want_exts:
                            want_ext = ".wav"
                        roots = [
                            Path(out_tmp),
                            Path(proj) / "outputs",
                            Path(proj) / "output",
                            Path(proj),
                        ]
                        cand = []
                        for r in roots:
                            try:
                                if not r.exists():
                                    continue
                                for p in r.rglob("*"):
                                    try:
                                        if not p.is_file():
                                            continue
                                        ext = p.suffix.lower()
                                        if ext not in want_exts:
                                            continue
                                        # only consider files created/modified after the run started
                                        if p.stat().st_mtime < (_ace_t0 - 5.0):
                                            continue
                                        cand.append(p)
                                    except Exception:
                                        continue
                            except Exception:
                                continue
                        # prefer requested format, then newest mtime
                        if cand:
                            cand.sort(key=lambda p: (0 if p.suffix.lower() == want_ext else 1, -p.stat().st_mtime))
                            newest = str(cand[0])
                    except Exception:
                        pass

                if not newest or not os.path.isfile(newest):
                    raise RuntimeError(f"Ace Step 1.5 produced no audio in: {out_tmp}")

                try:
                    shutil.copy2(newest, audio_out)
                except Exception:
                    # fallback: move/rename
                    try:
                        shutil.move(newest, audio_out)
                    except Exception:
                        pass

                if not _audio_file_ok_lenient(audio_out):
                    raise RuntimeError(f"Ace Step 1.5 output missing/too small: {audio_out}")

                meta_obj = {
                    "mode": "ace15",
                    "preset_key": preset_key,
                    "seed": int(seed),
                    "duration_sec": round(float(dur_s), 3),
                    "lyrics_enabled": bool(lyrics_enabled),
                    "fingerprint": fingerprint,
                    "output_file": audio_out,
                    "ts": time.time(),
                }
                _safe_write_json(ace_meta, meta_obj)

                # Wire into manifest for Chunk 6B mixer
                manifest.setdefault("paths", {})["music_file"] = audio_out
                manifest.setdefault("music", {})
                manifest["music"].setdefault("ace15", {})
                manifest["music"]["ace15"].update({
                    "mode": "ace15",
                    "preset_key": preset_key,
                    "seed": int(seed),
                    "duration_sec": round(float(dur_s), 3),
                    "fingerprint": fingerprint,
                    "output_file": audio_out,
                })
                _safe_write_json(manifest_path, manifest)

# Run ace generation as its own step (after assembly duration is locked, before audio mux)
            try:
                if bool(getattr(self.job, "music_background", False)) and str(getattr(self.job, "music_mode", "") or "") == "ace":
                    _run(_ace_step_name, step_music_ace_7a, _tail_pct_fn(_ace_step_name))
                if bool(getattr(self.job, "music_background", False)) and str(getattr(self.job, "music_mode", "") or "") == "ace15":
                    _ace15_step_name = "Music (Ace Step 1.5)"
                    _run(_ace15_step_name, step_music_ace15_7a, _tail_pct_fn(_ace15_step_name))
                    # If Videoclip Creator assembly was chosen, it likely ran earlier using a silent bed (because music didn't exist yet).
                    # Now that Ace15 music exists, re-run Videoclip Creator so it can segment + pick clips across the full music duration.
                    try:
                        if bool(_use_videoclip_creator):
                            preset_path2 = os.path.join(str(_root()), 'presets', 'setsave', 'plannerclip.json')
                            music_now = str((manifest.get('paths') or {}).get('music_file') or '')
                            if music_now and os.path.exists(music_now):
                                fp2 = _compute_mclip_fingerprint(music_now, preset_path2)
                                prev2 = (manifest.get('steps') or {}).get(_mclip_step_name) or {}
                                if (prev2.get('fingerprint') != fp2):
                                    _log('[INFO] Re-running Videoclip Creator now that Ace Step 1.5 music is available...')
                                    _run(_mclip_step_name, step_videoclip_creator_10a, _tail_pct_fn(_mclip_step_name))
                                    # Update step record
                                    try:
                                        srec3 = (manifest.get('steps') or {}).get(_mclip_step_name) or {}
                                        srec3['fingerprint'] = fp2
                                        srec3.setdefault('debug', {}).update({'timeline_json': timeline_json, 'rerun_after_ace15': True})
                                        srec3['ts'] = time.time()
                                        manifest.setdefault('steps', {})[_mclip_step_name] = srec3
                                        _safe_write_json(manifest_path, manifest)
                                    except Exception:
                                        pass
                    except Exception:
                        # Never fail the whole job because of this rerun; the user still has the first assembly output.
                        pass
            except Exception:
                raise


            # Chunk 7B: Music with lyrics (HeartMula) automation (after assembly duration is locked, before audio mux)
            _heartmula_step_name = "Music with lyrics (HeartMula)"
            _HEARTMULA_STYLE_TEXT = "energetic, electronic, synthesizer, drum machine"

            def _detect_mula_env_dir2() -> Path:
                # Optional installs commonly live either at root/.mula_env or presets/extra_env/.mula_env
                root = _root()
                cand = [
                    root / ".mula_env",
                    root / "presets" / "extra_env" / ".mula_env",
                ]
                for d in cand:
                    try:
                        if d.exists():
                            return d
                    except Exception:
                        pass
                return cand[0]

            def _mula_python2(env_dir: Path) -> Path:
                if os.name == "nt":
                    return env_dir / "Scripts" / "python.exe"
                return env_dir / "bin" / "python"

            def _clean_lyrics_text(s: str) -> str:
                t = (s or "").strip()
                # Strip common wrappers
                t = t.replace("```", "").strip()
                if t.lower().startswith("lyrics:"):
                    t = t.split(":", 1)[-1].strip()
                # Remove leading/trailing quotes
                if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
                    t = t[1:-1].strip()
                # Collapse excessive blank lines
                lines = [ln.rstrip() for ln in t.splitlines()]
                out = []
                blank = 0
                for ln in lines:
                    if not ln.strip():
                        blank += 1
                        if blank <= 1:
                            out.append("")
                    else:
                        blank = 0
                        out.append(ln)
                return "\n".join(out).strip()

            def _filter_lyrics_lines(t: str) -> list[str]:
                """Remove non-lyric meta lines (labels, stage directions, markdown-ish wrappers)."""
                import re as _re
                banned_starts = (
                    "verse", "chorus", "bridge", "hook", "outro", "intro", "refrain", "pre-chorus", "pre chorus",
                    "final chorus",
                )
                banned_contains = (
                    "fade out", "instrumental", "synth pulse", "drum fill", "guitar solo",
                    "spoken", "narration",
                )
                keep: list[str] = []
                for ln in (t or "").splitlines():
                    s = (ln or "").strip()
                    if not s:
                        continue
                    low = s.lower().strip()
                    # Standalone labels and label-like lines
                    if low in banned_starts:
                        continue
                    if _re.match(r"^\(?\s*(" + "|".join([_re.escape(x) for x in banned_starts]) + r")\b", low):
                        continue
                    if _re.match(r"^(" + "|".join([_re.escape(x) for x in banned_starts]) + r")\s*[:\-]?", low):
                        # e.g. "Chorus:" / "Verse 2" / "Outro -"
                        # Only skip if it looks like a section header (very short).
                        if len(low) <= 28:
                            continue
                    # Bracketed meta lines
                    if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
                        if len(s) <= 80:
                            continue
                    # Stage directions / non-lyrics
                    if any(k in low for k in banned_contains):
                        continue
                    keep.append(s)
                return keep

            def _finalize_lyrics_for_heartmula(raw: str, dur_target: float) -> tuple[str, int]:
                """Returns (lyrics_text, line_count) with meta stripped and basic length control."""
                t = _clean_lyrics_text(raw)
                lines = _filter_lyrics_lines(t)
                if not lines:
                    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
                # Duration-aware max line caps (keeps results from bloating)
                if dur_target <= 35.0:
                    max_lines = 10
                elif dur_target < 60.0:
                    max_lines = 18
                else:
                    max_lines = 48
                lines = lines[:max_lines]
                out = "\n".join(lines).strip()
                return out, len([ln for ln in out.splitlines() if ln.strip()])

            def step_music_heartmula_7b() -> None:

                env_dir = _detect_mula_env_dir2()
                py = _mula_python2(env_dir)
                if not py.exists():
                    raise RuntimeError(
                        f"HeartMula environment not found. Expected python at: {py} (install HeartMula via Optional Installs)."
                    )

                                                # Determine visual duration as close to generation time as possible.
                # IMPORTANT: prefer probing the actual assembled video file. Stored durations can be stale.
                dur_video = 0.0
                try:
                    dur_video = float(_probe_duration_sec(final_video) or 0.0)
                except Exception:
                    dur_video = 0.0
                if dur_video <= 0.01:
                    try:
                        dur_video = float(duration_sec or 0.0)
                    except Exception:
                        dur_video = 0.0
                if dur_video <= 0.01:
                    try:
                        dur_video = float(getattr(self.job, "approx_duration_sec", 0) or 0.0)
                    except Exception:
                        dur_video = 0.0
                dur_video = max(0.0, float(dur_video))
                # HeartMula hard max 4 minutes
                dur_clamped = max(1.0, min(240.0, dur_video))
                # HeartMula performs poorly on very short targets; generate at least 30s and trim in mux.
                dur_gen = max(30.0, float(dur_clamped))
                dur_gen = min(240.0, float(dur_gen))

                # Storyline source: use plan.json as the finalized storyline proxy
                plan_obj = {}
                storyline_txt = ""
                storyline_hash = ""
                try:
                    if _file_ok(plan_path, 2):
                        storyline_txt = _safe_read_text(plan_path).strip()
                        storyline_hash = _sha1_text(storyline_txt)
                        try:
                            plan_obj = json.loads(storyline_txt)
                        except Exception:
                            plan_obj = {}
                except Exception:
                    plan_obj = {}

                # info: Chunk 10 side quest — HeartMula lyrics: prefer Own Storyline text/prompts when enabled
                try:
                    _os_enabled_for_music = bool((self.job.encoding or {}).get("own_storyline_enabled"))
                except Exception:
                    _os_enabled_for_music = False
                own_story_brief = ""
                if bool(_os_enabled_for_music):
                    try:
                        _os_text = str((self.job.encoding or {}).get("own_storyline_text") or "").strip()
                    except Exception:
                        _os_text = ""
                    try:
                        _os_prompts = (self.job.encoding or {}).get("own_storyline_prompts") or []
                    except Exception:
                        _os_prompts = []
                    lines_os = []
                    try:
                        for p in list(_os_prompts)[:24]:
                            if isinstance(p, dict):
                                idx = str(p.get("index") or "").strip()
                                txt = str(p.get("text") or "").strip()
                            else:
                                idx = ""
                                txt = str(p).strip()
                            if not txt:
                                continue
                            if idx:
                                lines_os.append(f"- S{int(idx):02d}: {txt}")
                            else:
                                lines_os.append(f"- {txt}")
                    except Exception:
                        lines_os = []
                    if lines_os:
                        own_story_brief = "OWN STORYLINE (user-provided prompts):\n" + "\n".join(lines_os)
                    elif _os_text:
                        # Keep it short enough for lyric prompting
                        own_story_brief = _os_text
                    # Override storyline hash source so HeartMula can detect changes even if plan.json is skipped
                    try:
                        _hash_src = own_story_brief.strip() if own_story_brief.strip() else _os_text.strip()
                        if _hash_src:
                            storyline_hash = _sha1_text(_hash_src)
                    except Exception:
                        pass

                title = str(plan_obj.get("title", "") or "").strip()
                logline = str(plan_obj.get("logline", "") or "").strip()
                setting = str(plan_obj.get("setting", "") or "").strip()
                tone = str(plan_obj.get("tone", "") or "").strip()
                beats = plan_obj.get("beats") or []
                beats_lines = []
                try:
                    for b in list(beats)[:12]:
                        beats_lines.append(f"- {str(b).strip()}")
                except Exception:
                    beats_lines = []

                about = "\n".join(
                    [x for x in [
                        f"TITLE: {title}" if title else "",
                        f"LOGLINE: {logline}" if logline else "",
                        f"SETTING: {setting}" if setting else "",
                        f"TONE: {tone}" if tone else "",
                        "KEY BEATS:\n" + "\n".join(beats_lines) if beats_lines else "",
                    ] if x]
                ).strip()

                # info: Chunk 10 side quest — HeartMula lyrics: use Own Storyline brief when available
                try:
                    if (own_story_brief or "").strip():
                        about = (own_story_brief or "").strip()
                except Exception:
                    pass

                # Lyrics: reuse existing lyrics.txt if present; otherwise generate.                # Lyrics: reuse existing lyrics.txt if present; otherwise generate (lyrics only, no labels).
                lyrics_path = os.path.join(audio_dir, "lyrics.txt")
                qwen_log_path = os.path.join(audio_dir, "qwen_lyrics_log.txt")
                qwen_log_1 = os.path.join(audio_dir, "qwen_lyrics_log_attempt1.txt")
                qwen_log_2 = os.path.join(audio_dir, "qwen_lyrics_log_attempt2.txt")

                # info: Chunk 10 side quest — HeartMula lyrics: reuse only if storyline matches
                meta_probe_path = os.path.join(audio_dir, "heartmula_meta.json")

                min_lines = 5
                lyrics_txt = ""
                line_count = 0
                used_raw = ""
                used_sys = ""
                used_usr = ""

                # info: Chunk 10 side quest — HeartMula lyrics: regenerate if storyline changed
                _stored_story_hash = ""
                try:
                    if _file_ok(meta_probe_path, 2):
                        _m = _safe_read_json(meta_probe_path) or {}
                        _stored_story_hash = str(((_m.get("fingerprint") or {}).get("storyline_hash")) or "").strip()
                except Exception:
                    _stored_story_hash = ""
                _reuse_lyrics_ok = False
                try:
                    if storyline_hash and _stored_story_hash and storyline_hash == _stored_story_hash:
                        _reuse_lyrics_ok = True
                except Exception:
                    _reuse_lyrics_ok = False

                if _file_ok(lyrics_path, 2) and _reuse_lyrics_ok:
                    used_raw = _safe_read_text(lyrics_path).strip()
                    lyrics_txt, line_count = _finalize_lyrics_for_heartmula(used_raw, float(dur_gen))
                    # Normalize on disk to avoid accidentally keeping meta lines from older runs
                    try:
                        _safe_write_text(lyrics_path, lyrics_txt.strip() + "\n")
                    except Exception:
                        pass
                else:
                    # Duration-aware constraints (HeartMula: minimum effective target is 30s)
                    if dur_gen <= 35.0:
                        rule = "Output 8 to 12 short lyric lines (no section labels)."
                    elif dur_gen < 60.0:
                        rule = "Output 12 to 18 lyric lines with a clear repeating hook (no section labels)."
                    elif abs(dur_gen - 60.0) < 0.01:
                        rule = "Output a full 60-second structure with a repeating hook, but WITHOUT labels like Verse/Chorus."
                    else:
                        rule = "Output a full song structure sized for the duration, WITHOUT labels like Verse/Chorus."

                    used_sys = (
                        "You are an expert at creating song lyrics with rhymes. Output ONLY the lyrics lines.\n"
                        "Hard rules:\n"
                        "- No explanations, no headings, no markdown.\n"
                        "- Do NOT write section labels (no 'Verse', 'Chorus', 'Bridge', 'Outro', etc).\n"
                        "- Do NOT use parentheses or brackets.\n"
                        "- Do NOT include stage directions (e.g., 'fade out', 'soft synth pulse').\n"
                        "- One lyric line per line. No blank lines."
                    )
                    used_usr = (
                        "Write singable lyrics for an energetic electronic song.\n"
                        "Theme must match the storyline below and include self-discovery.\n\n"
                        f"{about}\n\n"
                        f"Duration target: {dur_gen:.1f} seconds. {rule}\n"
                        "Return ONLY the lyrics text."
                    )

                    raw1 = _qwen_text_call(_heartmula_step_name, used_sys, used_usr, qwen_log_1, temperature=0.4, max_new_tokens=900)
                    used_raw = raw1
                    lyrics_txt, line_count = _finalize_lyrics_for_heartmula(raw1, float(dur_gen))

                    # Retry once if model still returns meta/too few lines
                    if line_count < min_lines:
                        used_usr2 = (
                            used_usr
                            + "\n\nIMPORTANT: Output at least 5 lyric lines. No labels, no brackets, no parentheses."
                        )
                        raw2 = _qwen_text_call(_heartmula_step_name, used_sys, used_usr2, qwen_log_2, temperature=0.4, max_new_tokens=900)
                        used_raw = raw2
                        lyrics_txt, line_count = _finalize_lyrics_for_heartmula(raw2, float(dur_gen))

                    _safe_write_text(lyrics_path, lyrics_txt.strip() + "\n")

                    # Merge logs into the main debug file + include cleaned lyrics
                    try:
                        parts = []
                        if _file_ok(qwen_log_1, 2):
                            parts.append("=== ATTEMPT 1 ===\n" + _safe_read_text(qwen_log_1).strip())
                        if _file_ok(qwen_log_2, 2):
                            parts.append("=== ATTEMPT 2 ===\n" + _safe_read_text(qwen_log_2).strip())
                        parts.append("=== CLEANED LYRICS USED ===\n" + (lyrics_txt or "").strip())
                        _safe_write_text(qwen_log_path, "\n\n".join(parts).strip() + "\n")
                    except Exception:
                        pass

                lyrics_hash = _sha1_text(lyrics_txt)

                out_path = os.path.join(audio_dir, "music_heartmula.mp3")
                meta_path = os.path.join(audio_dir, "heartmula_meta.json")
                log_path = os.path.join(audio_dir, "heartmula_log.txt")
                payload_path = os.path.join(audio_dir, "heartmula_payload.json")

                # Fingerprint to skip regeneration
                runner_path = _root() / "helpers" / "heartmula_7b_runner.py"
                runner_mtime = 0
                try:
                    runner_mtime = int(runner_path.stat().st_mtime)
                except Exception:
                    runner_mtime = 0

                fingerprint = {
                    "mode": "heartmula",
                    "model_version": "3B",
                    "duration_sec": float(dur_gen),
                    "video_duration_sec": float(dur_clamped),
                    "seed": "none",
                    "style_text": _HEARTMULA_STYLE_TEXT,
                    "qwen_model_dir": str(_qwen_model_dir()),
                    "storyline_hash": storyline_hash,
                    "lyrics_hash": lyrics_hash,
                    "runner_mtime": runner_mtime,
                }

                if _file_ok(out_path, 1024) and _file_ok(meta_path, 2):
                    try:
                        meta = _safe_read_json(meta_path) or {}
                        if meta.get("fingerprint") == fingerprint:
                            # Wire manifest and return
                            manifest.setdefault("paths", {})["music_file"] = out_path
                            manifest.setdefault("music", {}).setdefault("heartmula", {})
                            manifest["music"]["heartmula"].update({
                                "mode": "heartmula",
                                "duration_sec": float(dur_gen),
                                "video_duration_sec": float(dur_clamped),
                                "seed": "none",
                                "style_text": _HEARTMULA_STYLE_TEXT,
                                "lyrics_path": lyrics_path,
                                "lyrics_hash": lyrics_hash,
                                "fingerprint": fingerprint,
                                "output_path": out_path,
                            })
                            return
                    except Exception:
                        pass

                # Run HeartMula via venv python + runner script
                payload = {
                    "root": str(_root()),
                    "audio_dir": audio_dir,
                    "output_path": out_path,
                    "duration_sec": float(dur_gen),
                    "style_text": _HEARTMULA_STYLE_TEXT,
                    "lyrics_path": lyrics_path,
                    "lyrics_text": lyrics_txt,
                    "seed": None,
                }
                _safe_write_json(payload_path, payload)

                # Ensure runner script exists
                if not runner_path.exists():
                    raise RuntimeError(f"HeartMula runner script missing: {runner_path}")


                # VRAM guard: ensure prior in-process models release GPU memory before HeartMula starts
                _vram_release("before heartmula")

                cmd = [str(py), str(runner_path), "--payload", str(payload_path)]
                try:
                    with open(log_path, "w", encoding="utf-8", errors="ignore") as lf:
                        lf.write("CMD: " + " ".join(cmd) + "\n\n")
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                        last_json = None
                        assert proc.stdout is not None
                        for line in proc.stdout:
                            if self._stop_requested:
                                try:
                                    proc.terminate()
                                except Exception:
                                    pass
                                try:
                                    proc.kill()
                                except Exception:
                                    pass
                                raise RuntimeError("Cancelled by user.")
                            lf.write(line)
                            s = (line or "").strip()
                            if s.startswith("{") and s.endswith("}"):
                                try:
                                    last_json = json.loads(s)
                                except Exception:
                                    pass
                        rc = proc.wait()
                        if rc != 0:
                            raise RuntimeError(f"HeartMula runner failed with code {rc}. See: {log_path}")
                        if isinstance(last_json, dict) and last_json.get("out_path"):
                            out_path2 = str(last_json.get("out_path") or "").strip()
                            if out_path2:
                                out_path = out_path2
                except Exception:
                    # Keep log file, bubble up
                    raise

                if not _file_ok(out_path, 1024):
                    raise RuntimeError(f"HeartMula output not created: {out_path}")

                # Save meta + wire manifest
                _safe_write_json(meta_path, {"fingerprint": fingerprint, "out_path": out_path, "payload": payload})
                manifest.setdefault("paths", {})["music_file"] = out_path
                manifest.setdefault("music", {}).setdefault("heartmula", {})
                manifest["music"]["heartmula"].update({
                    "mode": "heartmula",
                    "duration_sec": float(dur_gen),
                                "video_duration_sec": float(dur_clamped),
                    "seed": "none",
                    "style_text": _HEARTMULA_STYLE_TEXT,
                    "lyrics_path": lyrics_path,
                    "lyrics_hash": lyrics_hash,
                    "fingerprint": fingerprint,
                    "output_path": out_path,
                    "log_path": log_path,
                })

            try:
                if str(getattr(self.job, "music_mode", "") or "") == "heartmula":
                    # HeartMula should run even if Background music is off.
                    # Also force "silent" off and disable narration by default (music videoclip).
                    try:
                        self.job.silent = False
                    except Exception:
                        pass
                    try:
                        self.job.narration_enabled = False
                    except Exception:
                        pass
                    _run(_heartmula_step_name, step_music_heartmula_7b, _tail_pct_fn(_heartmula_step_name))
            except Exception:
                raise

            # Collect music path if available (already copied into job/audio earlier)
            music_file = ""
            try:
                music_file = str((manifest.get("paths") or {}).get("music_file") or "").strip()
            except Exception:
                music_file = ""
            if music_file and (not os.path.exists(music_file)):
                music_file = ""

            # Persist narration + music settings in manifest in a resume-friendly way (always update)
            manifest.setdefault("narration", {})
            manifest.setdefault("music", {})
            try:
                manifest["narration"].update({
                    "enabled": bool(self.job.narration_enabled),
                    "mode": str(self.job.narration_mode or "builtin"),
                    "voice": str(self.job.narration_voice or "ryan"),
                    "sample_path": str(self.job.narration_sample_path or ""),
                    "language": str(self.job.narration_language or "auto"),
                })
            except Exception:
                pass
            try:
                manifest["music"].update({
                    "enabled": bool(self.job.music_background and bool(music_file)),
                    "file_path": str(music_file or ""),
                    "volume": int(self.job.music_volume),
                })
            except Exception:
                pass
            try:
                manifest["storytelling_volume"] = int(self.job.storytelling_volume)
            except Exception:
                pass
            _safe_write_json(manifest_path, manifest)

            _audio_step_name = "Narration + optional music "

            def _compute_audio_fingerprint() -> str:
                return _sha1_text(json.dumps({
                    "final_video": _file_fingerprint(final_video),
                    "duration_sec": duration_sec,
                    "narration": manifest.get("narration") or {},
                    "music": manifest.get("music") or {},
                    "storytelling_volume": manifest.get("storytelling_volume"),
                }, sort_keys=True))

            def step_audio_mix() -> None:
                nonlocal music_file
                # Safety: if clone is selected but sample missing, block with a clear error
                if bool(self.job.narration_enabled) and str(self.job.narration_mode or "") == "clone":
                    sp = str(self.job.narration_sample_path or "").strip()
                    if not sp:
                        raise RuntimeError("Narration is set to 'add your own…' but no voice sample file was provided.")
                    # Copy sample into job/audio for resume-friendly stability
                    try:
                        ext = os.path.splitext(sp)[1] or ".wav"
                    except Exception:
                        ext = ".wav"
                    stable_sp = os.path.join(audio_dir, f"voice_sample{ext}")
                    try:
                        import shutil as _shutil
                        _shutil.copy2(sp, stable_sp)
                        sp = stable_sp
                    except Exception:
                        pass
                    try:
                        manifest.setdefault("narration", {})["sample_path"] = str(sp)
                        self.job.narration_sample_path = str(sp)
                        _safe_write_json(manifest_path, manifest)
                    except Exception:
                        pass

                    # Clone mode: ALWAYS supply ref_text by transcribing the voice sample via Whisper.
                    # Resume-friendly: reuse existing transcript unless the sample changed.
                    try:
                        _transcript_rel = "audio/voice_sample_transcript.txt"
                        _transcript_path = os.path.join(audio_dir, "voice_sample_transcript.txt")
                        _sample_fp = _file_fingerprint(str(sp))
                        _narr = (manifest.get("narration") or {}) if isinstance(manifest, dict) else {}
                        _prev_fp = (_narr.get("voice_sample_fingerprint") or {}) if isinstance(_narr, dict) else {}
                        _reuse = _file_ok(_transcript_path, 2) and ((not _prev_fp) or (_prev_fp == _sample_fp))
                        _generated = False
                        if _reuse:
                            if not (_safe_read_text(_transcript_path) or "").strip():
                                _reuse = False
                        if not _reuse:
                            _generated = True
                            from helpers import whisper as _wh  # type: ignore
                            env_py = _wh._whisper_env_python()
                            if (not env_py) or (not os.path.isfile(str(env_py))):
                                raise RuntimeError("Whisper environment not found. Install Whisper via Optional Installs.")
                            runner = _wh._ensure_whisper_runner_file()
                            model_dir = _find_whisper_model_dir()
                            if not model_dir:
                                raise RuntimeError("Whisper model folder not found. Expected /models/whisper/ (preferred) or /models/faster_whisper/medium/.")
                            try:
                                device = _wh._guess_device()
                            except Exception:
                                device = "cpu"
                            compute_type = "float16" if device == "cuda" else "int8"
                            try:
                                ffprobe_path = str(_wh._find_binary("ffprobe") or "")
                            except Exception:
                                ffprobe_path = ""
                            out_temp = os.path.join(str(_root()), "output", "_temp")
                            os.makedirs(out_temp, exist_ok=True)
                            whisper_payload = {
                                "root": str(_root()),
                                "media_path": str(sp),
                                "model_dir": str(model_dir),
                                "device": device,
                                "compute_type": compute_type,
                                "language": "auto",
                                "task": "transcribe",
                                "ffprobe_path": ffprobe_path,
                            }
                            payload_file = os.path.join(out_temp, f"_whisper_payload_voice_sample_{int(time.time()*1000)}.json")
                            _safe_write_text(payload_file, json.dumps(whisper_payload, indent=2, ensure_ascii=False))
                            cmd_whisper = [str(env_py), str(runner), str(payload_file)]
                            try:
                                with open(tts_log, "a", encoding="utf-8", errors="replace") as lf:
                                    lf.write("[clone_ref_text] mode=clone\n")
                                    lf.write("[clone_ref_text] sample_path=" + str(sp) + "\n")
                                    lf.write("[clone_ref_text] transcript=" + ("generated" if _generated else "reused") + "\n")
                                    lf.write("[clone_ref_text] whisper_cmd: " + " ".join(cmd_whisper) + "\n")
                            except Exception:
                                pass
                            cpw = subprocess.run(cmd_whisper, cwd=str(_root()), capture_output=True, text=True)
                            out_all = (cpw.stdout or "") + ((("\n" + cpw.stderr) if cpw.stderr else ""))
                            if cpw.returncode != 0:
                                try:
                                    with open(tts_log, "a", encoding="utf-8", errors="replace") as lf:
                                        lf.write("[clone_ref_text] whisper_failed exit=" + str(cpw.returncode) + "\n")
                                        if out_all.strip():
                                            lf.write(out_all + "\n")
                                except Exception:
                                    pass
                                raise RuntimeError("Whisper transcription subprocess failed.")
                            last_result = None
                            for _line in (out_all or "").splitlines():
                                s = (_line or "").strip()
                                if not (s.startswith("{") and s.endswith("}")):
                                    continue
                                try:
                                    msg = json.loads(s)
                                    if isinstance(msg, dict) and msg.get("type") == "result":
                                        last_result = msg.get("data") or {}
                                except Exception:
                                    continue
                            if not isinstance(last_result, dict) or not last_result:
                                try:
                                    with open(tts_log, "a", encoding="utf-8", errors="replace") as lf:
                                        lf.write("[clone_ref_text] whisper_failed: no result payload\n")
                                        if out_all.strip():
                                            lf.write(out_all + "\n")
                                except Exception:
                                    pass
                                raise RuntimeError("Whisper transcription returned no result.")
                            txt_tmp = str(last_result.get("text_path") or "").strip()
                            if (not txt_tmp) or (not os.path.isfile(txt_tmp)):
                                raise RuntimeError(f"Whisper missing transcript text: {txt_tmp}")
                            try:
                                import shutil as _shutil
                                _shutil.copy2(txt_tmp, _transcript_path)
                            except Exception:
                                _safe_write_text(_transcript_path, _safe_read_text(txt_tmp))
                        _ref_text = (_safe_read_text(_transcript_path) or "").strip() if os.path.exists(_transcript_path) else ""
                        if not _ref_text:
                            raise RuntimeError("Whisper produced empty transcript for voice sample.")
                        manifest.setdefault("narration", {})
                        manifest["narration"]["ref_text"] = _ref_text
                        manifest["narration"]["voice_sample_transcript_file"] = _transcript_rel
                        manifest["narration"]["voice_sample_fingerprint"] = _sample_fp
                        manifest["narration"]["voice_sample_transcript_fingerprint"] = _file_fingerprint(_transcript_path)
                        _safe_write_json(manifest_path, manifest)
                        try:
                            with open(tts_log, "a", encoding="utf-8", errors="replace") as lf:
                                preview = _ref_text[:200].replace("\r", " ").replace("\n", " ")
                                lf.write("[clone_ref_text] ref_text_preview=" + preview + "\n")
                        except Exception:
                            pass
                    except Exception as _e:
                        try:
                            with open(tts_log, "a", encoding="utf-8", errors="replace") as lf:
                                lf.write("\n[clone_ref_text] ERROR: " + str(_e) + "\n")
                                lf.write(traceback.format_exc() + "\n")
                        except Exception:
                            pass
                        raise RuntimeError("Whisper transcription failed for voice sample; see audio/tts_log.txt")

                # 1) Narration script + TTS (optional)
                narr_wav_ready = False
                narr_text = ""
                if bool(self.job.narration_enabled) and (not bool(self.job.silent)):
                    # Generate narration script using Qwen JSON path
                    plan_obj = _safe_read_json(plan_path) if os.path.exists(plan_path) else {}
                    shots_obj = _safe_read_json(shots_path) if os.path.exists(shots_path) else {}
                    sys_p = (
                        "You are a helpful video narrator. Produce a short spoken narration script for the video. "
                        "Return strict JSON only."
                    )
                    user_p = (
                        "Create a narration script that matches the plan and shots.\n"
                        "Constraints:\n"
                        "- Keep it concise and natural spoken language.\n"
                        "- No bullet points.\n"
                        "- Avoid mentioning camera settings.\n"
                        "- Target a spoken duration close to TARGET_SPOKEN_DURATION_SEC (±15%).\n"
                        "- Keep it short: do not exceed MAX_WORDS words.\n"
                        "- Output JSON with keys: narration_text.\n\n"
                        f"LANGUAGE: {str(self.job.narration_language or 'auto')}\n\n"
                        f"TARGET_SPOKEN_DURATION_SEC: {duration_sec:.1f}\n"
                        f"MAX_WORDS: {max(10, int(round(duration_sec * 2.2)))}\n\n"
                        f"PROMPT: {self.job.prompt}\n\n"
                        f"EXTRA: {self.job.extra_info}\n\n"
                        f"PLAN_JSON: {json.dumps(plan_obj, ensure_ascii=False)}\n\n"
                        f"SHOTS_JSON: {json.dumps(shots_obj, ensure_ascii=False)}\n"
                    )
                    raw_path = os.path.join(prompts_dir, "narration_raw.txt")
                    used_path = os.path.join(prompts_dir, "narration_prompts_used.txt")
                    err_path = os.path.join(prompts_dir, "narration_error.txt")
                    narr_json_obj, raw_text = _qwen_json_call(
                        "Narration script",
                        sys_p,
                        user_p,
                        raw_path,
                        used_path,
                        err_path,
                        temperature=0.2,
                        max_new_tokens=1024,
                    )
                    narration_text = ""
                    try:
                        if isinstance(narr_json_obj, dict):
                            narration_text = str(narr_json_obj.get("narration_text") or "").strip()
                    except Exception:
                        narration_text = ""
                    if not narration_text:
                        # fallback: use raw text
                        narration_text = (raw_text or "").strip()
                    if not narration_text:
                        raise RuntimeError("Narration script generation returned empty text.")
                    _safe_write_text(narration_txt, narration_text)
                    _safe_write_json(narration_json, {"narration_text": narration_text, "language": str(self.job.narration_language or "auto")})
                    narr_text = narration_text

                    # Generate narration audio via Qwen3 TTS worker (helpers/qwentts_ui.py)
                    # Built-in voices use mode=custom + speaker token; clone uses mode=clone + ref audio
                    root_dir = _root()
                    helpers_qwentts = root_dir / "helpers" / "qwentts_ui.py"
                    if not helpers_qwentts.exists():
                        raise RuntimeError("Missing helpers/qwentts_ui.py (required for narration TTS).")

                    if os.name == "nt":
                        env_py = root_dir / "environments" / ".qwen3tts" / "Scripts" / "python.exe"
                    else:
                        env_py = root_dir / "environments" / ".qwen3tts" / "bin" / "python"
                    if not env_py.exists():
                        raise RuntimeError("Qwen3 TTS environment not found. Install Qwen3 TTS via Optional Installs.")

                    mode = "custom" if str(self.job.narration_mode or "builtin") != "clone" else "clone"
                    # Clone mode uses Whisper transcript as ref_text (stored in manifest).
                    ref_text = ""
                    if mode == "clone":
                        try:
                            ref_text = str((manifest.get("narration") or {}).get("ref_text") or "").strip()
                        except Exception:
                            ref_text = ""
                        if not ref_text:
                            try:
                                _tp = os.path.join(audio_dir, "voice_sample_transcript.txt")
                                if os.path.exists(_tp):
                                    ref_text = (_safe_read_text(_tp) or "").strip()
                            except Exception:
                                ref_text = ""
                    payload = {
                        "mode": mode,
                        "payload": {
                            "model_path": str(_qwen3tts_model_path_for_mode(mode)),
                            "tokenizer_path": str(_qwen3tts_tokenizer_path()),
                            "text": narr_text,
                            "language": str(self.job.narration_language or "auto"),
                            "speaker": str(self.job.narration_voice or "ryan"),
                            "ref_audio_path": str(self.job.narration_sample_path or ""),
                            "common": {
                                "output_name": f"narration_{self.job.job_id}",
                                "add_timestamp": False,
                                "output_dir": str(audio_dir),
                                "output_format": "wav",
                            }
                        }
                    }

                    if mode == "clone":
                        payload["payload"]["ref_text"] = ref_text
                        # Fallback supported by Qwen3 TTS clone if transcription is unavailable.
                        payload["payload"]["x_vector_only_mode"] = (False if ref_text else True)

                    cmd = [str(env_py), "-u", str(helpers_qwentts.resolve()), "--worker", "--task", "generate"]
                    with open(tts_log, "a", encoding="utf-8", errors="replace") as lf:
                        lf.write("[cmd] " + " ".join(cmd) + "\n")
                        cp = subprocess.run(cmd, cwd=str(root_dir), input=json.dumps(payload), text=True, capture_output=True)
                        lf.write(cp.stdout or "")
                        if cp.stderr:
                            lf.write("\n[stderr]\n" + cp.stderr + "\n")
                    if cp.returncode != 0:
                        raise RuntimeError(f"TTS failed (exit={cp.returncode}). See: {tts_log}")
                                        # qwentts_ui returns JSON to stdout; extract out_path if present
                    out_path = ""
                    try:
                        import re as _re
                        stdout_text = (cp.stdout or "")
                        # Look for the last __RESULT__{...} block in stdout (worker prints it on success).
                        m_all = _re.findall(r"__RESULT__\s*(\{.*\})", stdout_text)
                        if m_all:
                            obj = json.loads(m_all[-1])
                            if obj.get("ok") and obj.get("out_path"):
                                out_path = str(obj.get("out_path"))
                        if not out_path:
                            # Fallback: try to locate any JSON with an out_path key.
                            m2 = _re.findall(r"(\{[^\n\r]*\})", stdout_text)
                            for raw in reversed(m2):
                                try:
                                    obj = json.loads(raw)
                                    if isinstance(obj, dict) and obj.get("out_path"):
                                        out_path = str(obj.get("out_path"))
                                        break
                                except Exception:
                                    continue
                    except Exception:
                        out_path = ""
                    def _force_local_wav(src_path: str, dst_wav: str) -> bool:
                        """Ensure narration lands in this job's audio/ as narration.wav."""
                        try:
                            if not src_path or (not os.path.exists(src_path)):
                                return False
                            os.makedirs(os.path.dirname(dst_wav), exist_ok=True)
                            # If already a wav, try a straight copy first
                            try:
                                import shutil as _shutil
                                _shutil.copyfile(src_path, dst_wav)
                                return os.path.exists(dst_wav)
                            except Exception:
                                pass
                            # Fallback: transcode via ffmpeg (handles mp3/ogg/wav etc.)
                            try:
                                args = [
                                    ffmpeg2, "-y",
                                    "-i", str(src_path),
                                    "-vn",
                                    "-acodec", "pcm_s16le",
                                    "-ar", "44100",
                                    "-ac", "2",
                                    str(dst_wav),
                                ]
                                cp2 = subprocess.run(args, cwd=str(root_dir), capture_output=True, text=True)
                                if cp2.returncode == 0 and os.path.exists(dst_wav):
                                    return True
                                # log ffmpeg failure detail into tts_log
                                try:
                                    with open(tts_log, "a", encoding="utf-8", errors="replace") as lf:
                                        lf.write("\n[ffmpeg] " + " ".join(args) + "\n")
                                        if cp2.stdout:
                                            lf.write(cp2.stdout + "\n")
                                        if cp2.stderr:
                                            lf.write("[ffmpeg_stderr]\n" + cp2.stderr + "\n")
                                except Exception:
                                    pass
                            except Exception:
                                pass
                            return False
                        except Exception:
                            return False

                    # Resolve the produced narration file and copy/transcode it into job/audio/narration.wav
                    searched = []
                    src_candidates = []

                    if out_path and os.path.exists(out_path):
                        src_candidates.append(out_path)

                    # Common output folder used by qwentts_ui (and other helpers)
                    try:
                        out_dirs = [
                            root_dir / "output" / "audio" / "qwen3tts",
                            root_dir / "output" / "qwen3tts",
                        ]
                        for out_dir in out_dirs:
                            if not out_dir.exists():
                                continue
                        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                                cands = sorted(
                                    [p for p in out_dir.glob(f"narration*{ext}")],
                                    key=lambda p: p.stat().st_mtime,
                                    reverse=True
                                )
                                for p in cands:
                                    src_candidates.append(str(p))
                    except Exception:
                        pass

                    # Last resort: search job audio dir itself in case backend wrote there with a different name/ext
                    try:
                        for ext in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                            cands = sorted(
                                [p for p in Path(audio_dir).glob(f"narration*{ext}")],
                                key=lambda p: p.stat().st_mtime,
                                reverse=True
                            )
                            for p in cands:
                                src_candidates.append(str(p))
                    except Exception:
                        pass

                    # Deduplicate candidates while preserving order
                    _seen = set()
                    src_candidates2 = []
                    for p in src_candidates:
                        if not p:
                            continue
                        if p in _seen:
                            continue
                        _seen.add(p)
                        src_candidates2.append(p)
                    src_candidates = src_candidates2

                    for p in src_candidates:
                        searched.append(p)
                        if _force_local_wav(p, narration_wav):
                            break

                    if not os.path.exists(narration_wav):
                        raise RuntimeError(
                            "Narration WAV was not produced in the job folder (missing audio/narration.wav). "
                            "TTS did run, but output could not be copied/transcoded. "
                            f"Searched: {searched}. See: {tts_log}"
                        )

                    narr_wav_ready = True
                    manifest.setdefault("paths", {})["narration_wav"] = narration_wav
                    manifest.setdefault("paths", {})["narration_txt"] = narration_txt
                    _safe_write_json(manifest_path, manifest)

                # 2) Mix/mux audio into final_cut.mp4
                # Decide audio sources:
                # Decide audio sources:
                # HeartMula (and any explicit music mode) must include music even if the Background music toggle is off.
                force_music = str(getattr(self.job, "music_mode", "") or "").strip() in ("ace", "ace15", "heartmula", "file")
                have_music = bool((not bool(getattr(self.job, "silent", False))) and bool(music_file) and os.path.exists(music_file) and (bool(getattr(self.job, "music_background", False)) or force_music))
                have_narr = bool(narr_wav_ready and os.path.exists(narration_wav))

                if not have_music and not have_narr:
                    # Nothing to do; keep visual-only final
                    try:
                        import shutil as _shutil
                        _shutil.copy2(final_video, final_cut_mp4)
                    except Exception:
                        pass
                    manifest.setdefault("paths", {})["final_video"] = final_cut_mp4
                    _safe_write_json(manifest_path, manifest)
                    return

                story_gain = max(0.0, min(2.0, float(int(self.job.storytelling_volume)) / 100.0))
                music_gain = max(0.0, min(2.0, float(int(self.job.music_volume)) / 100.0))
                # Prefer probing the actual visual file for duration at mux time (more reliable than stored duration).
                dur = 0.0
                try:
                    dur = float(_probe_duration_sec(final_video) or 0.0)
                except Exception:
                    dur = 0.0
                if dur <= 0.01:
                    dur = max(0.0, float(duration_sec))
                if dur <= 0.01:
                    dur = 999999.0

                                # Guard: ensure the visual base final video exists before we try to mux narration/music.
                # Avoid a race where the assembly output isn't on disk yet.
                _t0 = time.time()
                _last_sz = -1
                while (time.time() - _t0) < 60.0:
                    try:
                        if _file_ok(str(final_video), 1024):
                            _sz = int(os.path.getsize(str(final_video)))
                            if _sz == _last_sz:
                                break  # size stable
                            _last_sz = _sz
                        else:
                            _last_sz = -1
                    except Exception:
                        _last_sz = -1
                    time.sleep(0.5)
                if not _file_ok(str(final_video), 1024):
                    raise RuntimeError(f"Base final video not ready for audio mux: {final_video}")
                                # Decide whether we need to loop music. If HeartMula produced a short track for a longer video,
                # regenerate music to match the current visual duration (avoids looping the intro for minutes).
                music_dur = 0.0
                need_loop = False
                if have_music and dur < 999999.0:
                    try:
                        music_dur = float(_probe_duration_sec(music_file) or 0.0)
                    except Exception:
                        music_dur = 0.0
                    need_loop = (music_dur <= 0.01) or ((music_dur + 1.0) < float(dur))

                    if need_loop and str(getattr(self.job, "music_mode", "") or "").strip() == "heartmula":
                        try:
                            step_music_heartmula_7b()
                            # Reload path from manifest (HeartMula writes it there)
                            _mf2 = str((manifest.get("paths") or {}).get("music_file") or "").strip()
                            if _mf2 and os.path.exists(_mf2):
                                music_file = _mf2
                                try:
                                    music_dur = float(_probe_duration_sec(music_file) or 0.0)
                                except Exception:
                                    music_dur = 0.0
                                need_loop = (music_dur <= 0.01) or ((music_dur + 1.0) < float(dur))
                        except Exception:
                            # If regeneration fails, fall back to looping.
                            need_loop = True
# Build ffmpeg command
                cmd = [ffmpeg2, "-y", "-i", str(final_video)]
                filter_parts = []
                map_audio = ""
                if have_narr:
                    cmd += ["-i", str(narration_wav)]
                    # pad & trim to duration to keep timeline consistent
                    filter_parts.append(f"[1:a]volume={story_gain},apad,atrim=0:{dur}[narr]")
                if have_music:
                    cmd += (["-stream_loop", "-1", "-i", str(music_file)] if need_loop else ["-i", str(music_file)])
                    # music is looped; trim to duration
                    # input index depends on whether narration exists
                    mi = 2 if have_narr else 1
                    filter_parts.append(f"[{mi}:a]volume={music_gain},atrim=0:{dur}[mus]")

                if have_narr and have_music:
                    filter_parts.append("[narr][mus]amix=inputs=2:duration=first:dropout_transition=2[aout]")
                    map_audio = "[aout]"
                elif have_narr:
                    map_audio = "[narr]"
                else:
                    map_audio = "[mus]"

                cmd += ["-filter_complex", ";".join(filter_parts)]
                cmd += ["-map", "0:v:0", "-map", map_audio, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(final_cut_mp4)]

                with open(mix_log, "w", encoding="utf-8", errors="replace") as lf:
                    lf.write("[cmd] " + " ".join([str(x) for x in cmd]) + "\n")
                    cp = subprocess.run(cmd, cwd=str(_root()), capture_output=True, text=True)
                    lf.write(cp.stdout or "")
                    if cp.stderr:
                        lf.write("\n[stderr]\n" + cp.stderr + "\n")
                if cp.returncode != 0 or (not os.path.exists(final_cut_mp4)):
                    raise RuntimeError(f"ffmpeg mix failed (exit={cp.returncode}). See: {mix_log}")

                # Persist outputs for resume / Chunk 7
                manifest.setdefault("paths", {})["final_video"] = final_cut_mp4
                try:
                    tl = _safe_read_json(timeline_json) if os.path.exists(timeline_json) else {}
                except Exception:
                    tl = {}
                if not isinstance(tl, dict):
                    tl = {}
                tl["audio"] = {
                    "narration": manifest.get("narration") or {},
                    "music": manifest.get("music") or {},
                    "storytelling_volume": manifest.get("storytelling_volume"),
                }
                _safe_write_json(timeline_json, tl)
                _safe_write_json(manifest_path, manifest)

            # Chunk 10A (preset 2): Optional Videoclip Creator assembly (uses auto_music_sync.py).
            _use_videoclip_creator = False
            try:
                _vp = str((getattr(self.job, 'encoding', {}) or {}).get('videoclip_preset') or '').strip()
                # index 1/2 => Videoclip presets => use videoclip creator
                _use_videoclip_creator = (_vp.startswith('Videoclip Preset') or (_vp == 'Storyline Music videoclip'))
            except Exception:
                _use_videoclip_creator = False
            
            _mclip_step_name = "Videoclip Creator assembly "
            
            def _fingerprint_clips_dir(_d: str) -> Dict[str, Any]:
                try:
                    items = []
                    if _d and os.path.isdir(_d):
                        for fn in sorted(os.listdir(_d)):
                            if not fn.lower().endswith(".mp4"):
                                continue
                            p = os.path.join(_d, fn)
                            try:
                                st = os.stat(p)
                                items.append({'name': fn, 'size': int(st.st_size), 'mtime': float(st.st_mtime)})
                            except Exception:
                                continue
                    return {'count': len(items), 'items': items[:200]}  # cap to avoid huge manifests
                except Exception:
                    return {'count': 0, 'items': []}

            def _compute_mclip_fingerprint(audio_path: str, preset_path: str) -> str:
                try:
                    return _sha1_text(json.dumps({
                        'audio': _file_fingerprint(audio_path),
                        'preset': _file_fingerprint(preset_path),
                        'vp': str(_vp),
                        'clips_dir': _fingerprint_clips_dir(str(clips_dir)),
                        'job_id': str(getattr(self.job, 'job_id', '') or ''),
                    }, sort_keys=True))
                except Exception:
                    return _sha1_text(str(time.time()))
            
            def step_videoclip_creator_10a() -> None:
                nonlocal final_cut_mp4, final_video, music_file
                # Safety checks
                if not os.path.isdir(str(clips_dir)):
                    raise RuntimeError(f"Clips folder not found: {clips_dir}")
                
                # Choose audio for videoclip creator
                force_music = str(getattr(self.job, 'music_mode', '') or '').strip() in ('ace', 'ace15', 'heartmula', 'file')
                have_music = bool((not bool(getattr(self.job, 'silent', False))) and bool(music_file) and os.path.exists(music_file) and (bool(getattr(self.job, 'music_background', False)) or force_music))
                have_narr = bool((not bool(getattr(self.job, 'silent', False))) and bool(getattr(self.job, 'narration_enabled', False)) and _file_ok(narration_wav, 512))
                audio_for_creator = ''
                mix_audio_path = os.path.join(audio_dir, '_planner_videoclip_audio.m4a')
                mix_log2 = os.path.join(audio_dir, '_planner_videoclip_audio_mix.log')
                if have_music and (not have_narr):
                    audio_for_creator = str(music_file)
                elif have_narr and (not have_music):
                    audio_for_creator = str(narration_wav)
                elif have_music and have_narr:
                    # Mix narration over music into a single audio file (duration follows music).
                    try:
                        music_dur = float(_probe_duration_sec(music_file) or 0.0)
                    except Exception:
                        music_dur = 0.0
                    if music_dur <= 0.01:
                        # Fallback: just use music if duration cannot be probed.
                        audio_for_creator = str(music_file)
                    else:
                        story_gain = max(0.0, min(2.0, float(int(self.job.storytelling_volume)) / 100.0))
                        music_gain = max(0.0, min(2.0, float(int(self.job.music_volume)) / 100.0))
                        cmd = [
                            ffmpeg2, '-y',
                            '-i', str(music_file),
                            '-i', str(narration_wav),
                            '-filter_complex', f"[0:a]volume={music_gain}[m];[1:a]volume={story_gain},apad,atrim=0:{music_dur}[n];[m][n]amix=inputs=2:duration=first:dropout_transition=2[a]",
                            '-map', '[a]',
                            '-c:a', 'aac', '-b:a', '192k',
                            str(mix_audio_path),
                        ]
                        with open(mix_log2, 'w', encoding='utf-8', errors='replace') as lf:
                            lf.write('[cmd] ' + ' '.join([str(x) for x in cmd]) + '\n')
                            cp = subprocess.run(cmd, cwd=str(_root()), capture_output=True, text=True)
                            lf.write(cp.stdout or '')
                            if cp.stderr:
                                lf.write('\n[stderr]\n' + cp.stderr + '\n')
                        if cp.returncode == 0 and _file_ok(mix_audio_path, 1024):
                            audio_for_creator = str(mix_audio_path)
                        else:
                            # Fallback to music only if mix failed
                            audio_for_creator = str(music_file)
                else:
                    # No music/narration available. Videoclip Creator still needs an audio track to segment.
                    # Generate a silent audio bed matching the total clip duration (best-effort) so preset 2/3 can run.
                    try:
                        total_dur = 0.0
                        # Sum durations of the generated clips (mp4) in clips_dir.
                        for fn in sorted(os.listdir(str(clips_dir))):
                            if not fn.lower().endswith(".mp4"):
                                continue
                            p = os.path.join(str(clips_dir), fn)
                            try:
                                d = float(_probe_duration_sec(p) or 0.0)
                            except Exception:
                                d = 0.0
                            if d > 0.01:
                                total_dur += d
                        if total_dur <= 0.01:
                            total_dur = 10.0
                        # Clamp to a sane range to avoid accidental giant silent beds.
                        total_dur = max(1.0, min(total_dur, 60.0 * 60.0))
                        cmd = [
                            ffmpeg2, "-y",
                            "-f", "lavfi",
                            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                            "-t", str(total_dur),
                            "-c:a", "aac",
                            "-b:a", "192k",
                            str(mix_audio_path),
                        ]
                        with open(mix_log2, "w", encoding="utf-8", errors="replace") as lf:
                            lf.write("[cmd] " + " ".join([str(x) for x in cmd]) + "\n")
                            cp = subprocess.run(cmd, cwd=str(_root()), capture_output=True, text=True)
                            lf.write(cp.stdout or "")
                            if cp.stderr:
                                lf.write("\n[stderr]\n" + cp.stderr + "\n")
                        if cp.returncode == 0 and _file_ok(mix_audio_path, 1024):
                            audio_for_creator = str(mix_audio_path)
                            try:
                                _log(f"[INFO] No music/narration provided; generated silent audio bed ({total_dur:.2f}s) for Videoclip Creator.")
                            except Exception:
                                pass
                        else:
                            raise RuntimeError("Failed to generate silent audio for Videoclip Creator. See: " + str(mix_log2))
                    except Exception as e:
                        raise RuntimeError("Videoclip Creator preset selected but no audio is available, and silent-bed generation failed.") from e

                # Load preset settings from root presets/setsave/plannerclip.json
                preset_path = os.path.join(str(_root()), 'presets', 'setsave', 'plannerclip.json')
                if not os.path.isfile(preset_path):
                    raise RuntimeError(f"plannerclip.json not found: {preset_path}")
                preset_obj = _safe_read_json(preset_path) if os.path.isfile(preset_path) else {}
                if not isinstance(preset_obj, dict):
                    preset_obj = {}
                presets_list = preset_obj.get('presets') or []
                if not isinstance(presets_list, list) or (not presets_list):
                    raise RuntimeError('plannerclip.json has no presets.')
                # Pick preferred preset id if present, else first preset
                _preferred_ids = ('plannerclip_default', 'get_it_done_fast')
                chosen = None
                for pid in _preferred_ids:
                    for p in presets_list:
                        if isinstance(p, dict) and str(p.get('id') or '') == pid:
                            chosen = p
                            break
                    if chosen is not None:
                        break
                if chosen is None:
                    chosen = presets_list[0] if isinstance(presets_list[0], dict) else {}
                preset_id = str(chosen.get('id') or '')
                settings = chosen.get('settings') or {}
                if not isinstance(settings, dict):
                    settings = {}
                # Enforce planner defaults for Videoclip presets: keep source + keep source letterbox + order mode
                settings.setdefault('combo_res', 0)
                settings.setdefault('combo_fit', 0)
                # Preset 2: Transitions => shuffle; Preset 3: Storyline Music videoclip => sequential
                if "storyline" in str(_vp).strip().lower():
                    settings['combo_clip_order'] = 1
                else:
                    settings['combo_clip_order'] = 2
                
                # Analyze audio + build segments
                try:
                    from helpers import auto_music_sync as _ams  # type: ignore
                except Exception:
                    import auto_music_sync as _ams  # type: ignore

                # Some older auto_music_sync builds validate payload fields using
                # unquoted global name constants (e.g. ffmpeg_path) and may forget
                # to define one of them, causing NameError. Some builds also treat these
                # names as callables (e.g. ffmpeg_path()), so we define keys that behave
                # as both strings and no-arg callables.
                try:
                    class _AMSKey(str):
                        def __call__(self):
                            return str(self)
                    for _k in ("analysis", "segments", "audio_path", "output_dir", "ffmpeg_path", "ffprobe_path"):
                        _cur = getattr(_ams, _k, None)
                        if _cur is None:
                            setattr(_ams, _k, _AMSKey(_k))
                        elif isinstance(_cur, str) and (not callable(_cur)):
                            setattr(_ams, _k, _AMSKey(_cur))
                except Exception:
                    pass
                sens = int(settings.get('slider_sens', 10) or 10)
                try:
                    cfg = _ams.MusicAnalysisConfig(sensitivity=sens)
                except Exception:
                    cfg = None
                analysis = _ams.analyze_music(str(audio_for_creator), ffmpeg2, cfg) if cfg is not None else _ams.analyze_music(str(audio_for_creator), ffmpeg2, _ams.MusicAnalysisConfig())
                # NOTE: Older auto_music_sync versions may not expose scan_sources().
                # Provide a robust fallback scanner to keep Chunk 10A working across versions.
                if hasattr(_ams, "scan_sources"):
                    sources = _ams.scan_sources(str(clips_dir), ffprobe2)
                else:
                    def _scan_sources_fallback(_folder: str, _ffprobe: str):
                        exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
                        out = []
                        try:
                            from pathlib import Path as _Path
                        except Exception:
                            _Path = None
                        for _r, _ds, _fs in os.walk(_folder):
                            _fs_sorted = sorted(_fs)
                            for _fn in _fs_sorted:
                                try:
                                    _p = (_Path(_r) / _fn) if _Path is not None else os.path.join(_r, _fn)
                                    _suf = str(_p).lower()
                                    if not any(_suf.endswith(e) for e in exts):
                                        continue
                                    _pp = str(_p)
                                    _dur = float(_probe_duration_sec(_pp) or 0.0)
                                    _w = 0; _h = 0; _fps = 0.0
                                    try:
                                        args = [_ffprobe, "-v", "error", "-print_format", "json", "-select_streams", "v:0", "-show_streams", _pp]
                                        cp = subprocess.run(args, cwd=str(_root()), capture_output=True, text=True)
                                        if cp.returncode == 0:
                                            obj = json.loads(cp.stdout or "{}")
                                            st = (obj.get("streams") or [{}])[0] or {}
                                            _w = int(st.get("width") or 0)
                                            _h = int(st.get("height") or 0)
                                            rr = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/1"
                                            try:
                                                n, d = rr.split("/")
                                                d2 = float(d)
                                                _fps = (float(n) / d2) if d2 != 0 else float(n)
                                            except Exception:
                                                _fps = 0.0
                                    except Exception:
                                        pass
                                    out.append({
                                        "path": _pp,
                                        "filepath": _pp,
                                        "duration_sec": _dur,
                                        "duration": _dur,
                                        "width": _w,
                                        "height": _h,
                                        "fps": _fps,
                                    })
                                except Exception:
                                    continue
                        return out
                    # Try a few likely legacy function names before using the built-in fallback.
                    if hasattr(_ams, "scan_clips"):
                        sources = _ams.scan_clips(str(clips_dir), ffprobe2)
                    elif hasattr(_ams, "scan_folder"):
                        sources = _ams.scan_folder(str(clips_dir), ffprobe2)
                    elif hasattr(_ams, "list_sources"):
                        sources = _ams.list_sources(str(clips_dir), ffprobe2)
                    else:
                        sources = _scan_sources_fallback(str(clips_dir), ffprobe2)

                # Normalize sources to support both dict-style and attribute-style access.
                # Some auto_music_sync versions expect objects with .duration/.path attributes,
                # while others return dicts from scan_sources(). We wrap dicts into a proxy that
                # supports both access patterns.
                class _SourceProxy(dict):
                    __slots__ = ()
                    def __getattr__(self, k):
                        try:
                            return self[k]
                        except KeyError:
                            raise AttributeError(k)
                    def __setattr__(self, k, v):
                        self[k] = v
                def _normalize_sources(_srcs):
                    out = []
                    for s in (_srcs or []):
                        if isinstance(s, dict) and not isinstance(s, _SourceProxy):
                            sp = _SourceProxy(s)
                            # Ensure common fields exist
                            if 'duration' not in sp:
                                if 'duration_sec' in sp:
                                    sp['duration'] = sp.get('duration_sec')
                            if 'duration_sec' not in sp:
                                if 'duration' in sp:
                                    sp['duration_sec'] = sp.get('duration')
                            if 'path' not in sp:
                                sp['path'] = sp.get('filepath') or sp.get('file') or sp.get('src') or ''
                            out.append(sp)
                        else:
                            # best-effort: add a .duration attribute if missing
                            try:
                                if (not hasattr(s, 'duration')) and hasattr(s, 'duration_sec'):
                                    try:
                                        setattr(s, 'duration', getattr(s, 'duration_sec'))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                            out.append(s)
                    return out
                sources = _normalize_sources(sources)

                if not sources:
                    raise RuntimeError('No source clips found for videoclip creator.')
                
                nofx = bool(settings.get('check_nofx', False))
                fx_idx = int(settings.get('combo_fx', 0) or 0)
                if nofx:
                    fx_level = 'none'
                elif fx_idx == 0:
                    fx_level = 'minimal'
                elif fx_idx == 1:
                    fx_level = 'moderate'
                else:
                    fx_level = 'high'
                
                micro_mode = 0
                if bool(settings.get('check_micro_chorus', False)):
                    micro_mode = 1
                elif bool(settings.get('check_micro_all', False)):
                    micro_mode = 2
                elif bool(settings.get('check_micro_verses', False)):
                    micro_mode = 3
                beats_per = max(1, int(settings.get('spin_beats_per_seg', 8) or 8))
                transition_mode = int(settings.get('combo_transitions', 1) or 1)
                if nofx:
                    transition_mode = 1
                clip_order_mode = int(settings.get('combo_clip_order', 2) or 2)
                force_full_length = bool(settings.get('check_full_length', False))
                seed_enabled = bool(settings.get('check_use_seed', False))
                seed_value = int(settings.get('spin_seed', 0) or 0)
                transition_random = bool(settings.get('check_trans_random', False)) and (not nofx)
                transition_modes_enabled = [transition_mode] if transition_random else []
                
                slow_enabled = bool(settings.get('check_slow_enable', False)) and (not nofx)
                slow_sections = []
                if slow_enabled:
                    if bool(settings.get('check_slow_intro', False)): slow_sections.append('intro')
                    if bool(settings.get('check_slow_break', False)): slow_sections.append('break')
                    if bool(settings.get('check_slow_chorus', False)): slow_sections.append('chorus')
                    if bool(settings.get('check_slow_drop', False)): slow_sections.append('drop')
                    if bool(settings.get('check_slow_outro', False)): slow_sections.append('outro')
                slow_factor = max(0.10, min(1.0, float(int(settings.get('slider_slow_factor', 50) or 50)) / 100.0)) if slow_enabled else 1.0
                slow_random = bool(settings.get('check_slow_random', False)) if slow_enabled else False
                
                # Cine (most are optional; pass through defaults if missing)
                cine_enable = bool(settings.get('check_cine_enable', False)) and (not nofx)
                cine_freeze = bool(settings.get('check_cine_freeze', False))
                cine_stutter = bool(settings.get('check_cine_stutter', False))
                cine_reverse = bool(settings.get('check_cine_reverse', False))
                cine_speedup_forward = bool(settings.get('check_cine_speedup_forward', False))
                cine_speedup_backward = bool(settings.get('check_cine_speedup_backward', False))
                cine_speed_ramp = bool(settings.get('check_cine_speed_ramp', False))
                cine_speedup_forward_factor = float(settings.get('spin_cine_speedup_forward', 2.0) or 2.0)
                cine_speedup_backward_factor = float(settings.get('spin_cine_speedup_backward', 2.0) or 2.0)
                cine_freeze_len = float(int(settings.get('slider_cine_freeze_len', 40) or 40)) / 100.0
                cine_freeze_zoom = float(int(settings.get('slider_cine_freeze_zoom', 20) or 20)) / 100.0
                cine_tear_v = bool(settings.get('check_cine_tear_v', False))
                cine_tear_h = bool(settings.get('check_cine_tear_h', False))
                cine_tear_v_strength = float(int(settings.get('slider_cine_tear_v_strength', 40) or 40)) / 100.0
                cine_tear_h_strength = float(int(settings.get('slider_cine_tear_h_strength', 40) or 40)) / 100.0
                cine_color_cycle = bool(settings.get('check_cine_color_cycle', False))
                cine_color_cycle_speed_ms = int(settings.get('slider_cine_color_cycle_speed', 250) or 250)
                cine_stutter_repeats = int(settings.get('spin_cine_stutter_repeats', 3) or 3)
                cine_reverse_len = float(int(settings.get('slider_cine_reverse_len', 40) or 40)) / 100.0
                cine_ramp_in = float(int(settings.get('slider_cine_ramp_in', 15) or 15)) / 100.0
                cine_ramp_out = float(int(settings.get('slider_cine_ramp_out', 15) or 15)) / 100.0
                cine_boomerang = bool(settings.get('check_cine_boomerang', False))
                cine_boomerang_bounces = int(settings.get('slider_cine_boomerang_bounces', 2) or 2)
                cine_dimension = bool(settings.get('check_cine_dimension', False))
                cine_pan916 = bool(settings.get('check_cine_pan916', False))
                cine_pan916_speed_ms = int(settings.get('slider_cine_pan916_speed', 150) or 150)
                cine_pan916_parts = int(settings.get('slider_cine_pan916_parts', 4) or 4)
                cine_pan916_transparent = bool(settings.get('check_cine_pan916_transparent', False))
                cine_pan916_random = bool(settings.get('check_cine_pan916_random', False))
                cine_mosaic = bool(settings.get('check_cine_mosaic', False))
                cine_mosaic_screens = int(settings.get('slider_cine_mosaic_screens', 9) or 9)
                cine_mosaic_random = bool(settings.get('check_cine_mosaic_random', False))
                cine_flip = bool(settings.get('check_cine_flip', False))
                cine_rotate = bool(settings.get('check_cine_rotate', False))
                cine_rotate_max_degrees = float(settings.get('slider_cine_rotate_degrees', 90) or 90)
                cine_multiply = bool(settings.get('check_cine_multiply', False))
                cine_multiply_screens = int(settings.get('slider_cine_multiply_screens', 6) or 6)
                cine_multiply_random = bool(settings.get('check_cine_multiply_random', False))
                cine_dolly = bool(settings.get('check_cine_dolly', False))
                cine_dolly_strength = float(int(settings.get('slider_cine_dolly_strength', 70) or 70)) / 100.0
                cine_kenburns = bool(settings.get('check_cine_kenburns', False))
                cine_kenburns_strength = float(int(settings.get('slider_cine_kenburns_strength', 55) or 55)) / 100.0
                cine_motion_dir = int(settings.get('combo_cine_motion_dir', 0) or 0)
                
                # Impact FX
                impact_enable = bool(settings.get('check_impact_enable', False)) and (not nofx)
                impact_flash = bool(settings.get('check_impact_flash', False))
                impact_shock = bool(settings.get('check_impact_shock', False))
                impact_echo_trail = bool(settings.get('check_impact_echo', False))
                impact_confetti = bool(settings.get('check_impact_confetti', False))
                impact_color_cycle = bool(settings.get('check_impact_colorcycle', False))
                impact_zoom = bool(settings.get('check_impact_zoom', False))
                impact_shake = bool(settings.get('check_impact_shake', False))
                impact_fog = bool(settings.get('check_impact_fog', False))
                impact_fire_gold = bool(settings.get('check_impact_fire_gold', False))
                impact_fire_multi = bool(settings.get('check_impact_fire_multi', False))
                impact_random = bool(settings.get('check_impact_random', False))
                impact_flash_strength = float(int(settings.get('slider_impact_flash', 80) or 80)) / 100.0
                impact_flash_speed_ms = int(settings.get('slider_impact_flash_speed', 250) or 250)
                impact_shock_strength = float(int(settings.get('slider_impact_shock', 75) or 75)) / 100.0
                impact_echo_trail_strength = float(int(settings.get('slider_impact_echo', 80) or 80)) / 100.0
                impact_confetti_density = float(int(settings.get('slider_impact_confetti', 70) or 70)) / 100.0
                impact_color_cycle_speed = float(int(settings.get('slider_impact_colorcycle', 75) or 75)) / 100.0
                impact_zoom_amount = float(int(settings.get('slider_impact_zoom', 25) or 25)) / 100.0
                impact_shake_strength = float(int(settings.get('slider_impact_shake', 70) or 70)) / 100.0
                impact_fog_density = float(int(settings.get('slider_impact_fog', 0) or 0)) / 100.0
                impact_fire_gold_intensity = float(int(settings.get('slider_impact_fire_gold', 0) or 0)) / 100.0
                impact_fire_multi_intensity = float(int(settings.get('slider_impact_fire_multi', 0) or 0)) / 100.0
                
                audio_duration = float(_probe_duration_sec(str(audio_for_creator)) or 0.0)
                segments = _ams.build_timeline(
                    analysis,
                    sources,
                    fx_level=fx_level,
                    microclip_mode=micro_mode,
                    beats_per_segment=beats_per,
                    transition_mode=transition_mode,
                    clip_order_mode=clip_order_mode,
                    force_full_length=force_full_length,
                    seed_enabled=seed_enabled,
                    seed_value=seed_value,
                    transition_random=transition_random,
                    transition_modes_enabled=transition_modes_enabled,
                    intro_transitions_only=True,
                    slow_motion_enabled=slow_enabled,
                    slow_motion_factor=slow_factor,
                    slow_motion_sections=slow_sections,
                    slow_motion_random=slow_random,
                    cine_enable=cine_enable,
                    cine_freeze=cine_freeze,
                    cine_stutter=cine_stutter,
                    cine_reverse=cine_reverse,
                    cine_speedup_forward=cine_speedup_forward,
                    cine_speedup_forward_factor=cine_speedup_forward_factor,
                    cine_speedup_backward=cine_speedup_backward,
                    cine_speedup_backward_factor=cine_speedup_backward_factor,
                    cine_speed_ramp=cine_speed_ramp,
                    cine_freeze_len=cine_freeze_len,
                    cine_freeze_zoom=cine_freeze_zoom,
                    cine_tear_v=cine_tear_v,
                    cine_tear_v_strength=cine_tear_v_strength,
                    cine_tear_h=cine_tear_h,
                    cine_tear_h_strength=cine_tear_h_strength,
                    cine_color_cycle=cine_color_cycle,
                    cine_color_cycle_speed_ms=cine_color_cycle_speed_ms,
                    cine_stutter_repeats=cine_stutter_repeats,
                    cine_reverse_len=cine_reverse_len,
                    cine_ramp_in=cine_ramp_in,
                    cine_ramp_out=cine_ramp_out,
                    cine_boomerang=cine_boomerang,
                    cine_boomerang_bounces=cine_boomerang_bounces,
                    cine_dimension=cine_dimension,
                    cine_pan916=cine_pan916,
                    cine_pan916_speed_ms=cine_pan916_speed_ms,
                    cine_pan916_parts=cine_pan916_parts,
                    cine_pan916_transparent=cine_pan916_transparent,
                    cine_pan916_random=cine_pan916_random,
                    cine_mosaic=cine_mosaic,
                    cine_mosaic_screens=cine_mosaic_screens,
                    cine_mosaic_random=cine_mosaic_random,
                    cine_flip=cine_flip,
                    cine_rotate=cine_rotate,
                    cine_rotate_max_degrees=cine_rotate_max_degrees,
                    cine_multiply=cine_multiply,
                    cine_multiply_screens=cine_multiply_screens,
                    cine_multiply_random=cine_multiply_random,
                    cine_dolly=cine_dolly,
                    cine_dolly_strength=cine_dolly_strength,
                    cine_kenburns=cine_kenburns,
                    cine_kenburns_strength=cine_kenburns_strength,
                    cine_motion_dir=cine_motion_dir,
                    audio_duration=audio_duration,
                    impact_enable=impact_enable,
                    impact_flash=impact_flash,
                    impact_shock=impact_shock,
                    impact_echo_trail=impact_echo_trail,
                    impact_confetti=impact_confetti,
                    impact_color_cycle=impact_color_cycle,
                    impact_zoom=impact_zoom,
                    impact_shake=impact_shake,
                    impact_fog=impact_fog,
                    impact_fire_gold=impact_fire_gold,
                    impact_fire_multi=impact_fire_multi,
                    impact_random=impact_random,
                    impact_flash_strength=impact_flash_strength,
                    impact_flash_speed_ms=impact_flash_speed_ms,
                    impact_shock_strength=impact_shock_strength,
                    impact_echo_trail_strength=impact_echo_trail_strength,
                    impact_confetti_density=impact_confetti_density,
                    impact_color_cycle_speed=impact_color_cycle_speed,
                    impact_zoom_amount=impact_zoom_amount,
                    impact_shake_strength=impact_shake_strength,
                    impact_fog_density=impact_fog_density,
                    impact_fire_gold_intensity=impact_fire_gold_intensity,
                    impact_fire_multi_intensity=impact_fire_multi_intensity,
                    image_sources=[],
                    section_overrides=None,
                    image_segment_interval=0,
                )
                if not segments:
                    raise RuntimeError('Videoclip Creator timeline is empty.')
                
                # Target resolution + fit mode (we enforce keep-source defaults)
                target_res = None
                fit_mode = int(settings.get('combo_fit', 0) or 0)
                
                # Visual overlay
                use_visual_overlay = bool(settings.get('check_visual_overlay', False)) and (not nofx)
                visual_overlay_opacity = float(int(settings.get('slider_visual_opacity', 40) or 40)) / 100.0
                
                # Visual strategy
                visual_strategy = 0
                if bool(settings.get('check_visual_strategy_segment', False)):
                    visual_strategy = 1
                elif bool(settings.get('check_visual_strategy_section', False)):
                    visual_strategy = 2
                
                intro_fade = bool(settings.get('check_intro_fade', True)) and (not nofx)
                outro_fade = bool(settings.get('check_outro_fade', True)) and (not nofx)
                
                out_name = os.path.basename(str(final_video))
                
                payload = {
                    'analysis': getattr(_ams, '_analysis_to_dict')(analysis),
                    'segments': getattr(_ams, '_segments_to_list')(segments),
                    'audio_path': str(audio_for_creator),
                    'output_dir': str(final_dir),
                    'ffmpeg_path': str(ffmpeg2),
                    'ffprobe_path': str(ffprobe2),
                    'target_resolution': target_res,
                    'fit_mode': fit_mode,
                    'transition_mode': transition_mode,
                    'intro_fade': intro_fade,
                    'outro_fade': outro_fade,
                    'use_visual_overlay': use_visual_overlay,
                    'visual_overlay_opacity': visual_overlay_opacity,
                    'visual_strategy': visual_strategy,
                    'out_name_override': out_name,
                }
                
                # Persist payload for debug/resume
                payload_path = os.path.join(final_dir, '_planner_videoclip_payload.json')
                try:
                    _safe_write_json(payload_path, payload)
                except Exception:
                    pass
                
                # Run render synchronously using tool's headless queue entry.
                # Some builds of auto_music_sync expect a JSON *path* rather than a dict payload.
                _log('[RUN] Music Videoclip Creator ')
                try:
                    _ams.run_queue_payload(payload_path)
                except Exception as e:
                    raise RuntimeError('Videoclip Creator failed.') from e

                # Some Videoclip Creator builds may ignore out_name_override or output_dir.
                # If the expected output isn't present, try to locate the newest mp4 produced in/near the output_dir
                # and adopt it as the produced output.
                produced_mp4 = str(final_video)
                if not _file_ok(produced_mp4, 1024):
                    try:
                        cand_dirs = [str(final_dir), str(_root() / "output" / "planner"), str(_root() / "output")]
                        newest = None
                        newest_mtime = 0.0
                        for d in cand_dirs:
                            if not d or (not os.path.isdir(d)):
                                continue
                            for r, _ds, fs in os.walk(d):
                                for fn in fs:
                                    if not fn.lower().endswith(".mp4"):
                                        continue
                                    p = os.path.join(r, fn)
                                    try:
                                        if not _file_ok(p, 1024):
                                            continue
                                        mt = os.path.getmtime(p)
                                        if mt > newest_mtime:
                                            newest_mtime = mt
                                            newest = p
                                    except Exception:
                                        continue
                        if newest and _file_ok(newest, 1024):
                            try:
                                os.makedirs(str(final_dir), exist_ok=True)
                                shutil.copy2(newest, produced_mp4)
                            except Exception:
                                produced_mp4 = newest
                    except Exception:
                        pass

                if not _file_ok(str(produced_mp4), 1024):
                    raise RuntimeError(f'Videoclip Creator output not created: {final_video}')
                
                # Normalize output: Videoclip Creator typically writes a job-specific name (e.g. <jobid>_final.mp4).
                # Planner also maintains a stable "final_cut.mp4" for UI + downstream tools.
                produced_mp4 = str(final_video)
                stable_final_cut = os.path.join(str(final_dir), "final_cut.mp4")
                try:
                    if os.path.abspath(produced_mp4) != os.path.abspath(stable_final_cut):
                        # Copy (not move) so the original remains for debugging.
                        shutil.copy2(produced_mp4, stable_final_cut)
                except Exception:
                    # Best-effort: if copy fails, keep using the produced file.
                    stable_final_cut = produced_mp4

                final_cut_mp4 = str(stable_final_cut)
                manifest.setdefault('paths', {})['final_video'] = str(final_cut_mp4)
                manifest.setdefault('paths', {})['final_cut_path'] = str(final_cut_mp4)
                try:
                    _log(f"[OK] Videoclip Creator output: {produced_mp4}")
                    if os.path.abspath(produced_mp4) != os.path.abspath(final_cut_mp4):
                        _log(f"[OK] Copied to stable final cut: {final_cut_mp4}")
                except Exception:
                    pass
                try:
                    tl = _safe_read_json(timeline_json) if os.path.exists(timeline_json) else {}
                except Exception:
                    tl = {}
                if not isinstance(tl, dict):
                    tl = {}
                tl['videoclip_creator'] = {
                    'enabled': True,
                    'preset_id': preset_id,
                    'preset_file': preset_path,
                    'audio_path': str(audio_for_creator),
                    'clips_dir': str(clips_dir),
                    'payload_path': payload_path,
                }
                _safe_write_json(timeline_json, tl)
                _safe_write_json(manifest_path, manifest)
                

            # -----------------------------
            # Tail progress planning (more realistic % after first final cut)
            #
            # Problem:
            # - The UI could reach 100% immediately after the first visual final cut is created,
            #   even though optional post-steps may still run (narration, music generation/mix,
            #   upscaling, interpolation, etc.).
            #
            # Fix:
            # - Build a small "tail plan" of remaining steps and reserve 95–99 for them.
            # - Each tail step gets its own % target; 100 is only emitted at the very end.
            # -----------------------------
            def _peek_planner_upscale_settings(job_dir2: str) -> dict:
                try:
                    p2 = os.path.join(str(job_dir2), _PLANNER_UPSCALE_JSON_NAME)
                    if not p2 or (not os.path.isfile(p2)):
                        return {}
                    with open(p2, "r", encoding="utf-8", errors="ignore") as f:
                        obj2 = json.load(f)
                    return obj2 if isinstance(obj2, dict) else {}
                except Exception:
                    return {}

            def _build_tail_targets(step_names: List[str], start_pct: int = 95, end_pct: int = 99) -> Tuple[Dict[str, int], Dict[str, Tuple[int, int]]]:
                names = [s for s in (step_names or []) if isinstance(s, str) and s.strip()]
                if not names:
                    return {}, {}
                start_pct = int(max(0, min(99, start_pct)))
                end_pct = int(max(start_pct, min(99, end_pct)))
                n = len(names)
                targets: Dict[str, int] = {}
                windows: Dict[str, Tuple[int, int]] = {}
                for i, nm in enumerate(names):
                    if n == 1:
                        t = end_pct
                    else:
                        t = start_pct + int(round((i / max(1, (n - 1))) * (end_pct - start_pct)))
                    targets[nm] = int(max(start_pct, min(end_pct, t)))
                for i, nm in enumerate(names):
                    a = targets.get(nm, start_pct)
                    if i + 1 < n:
                        b = targets.get(names[i + 1], end_pct)
                    else:
                        b = end_pct
                    windows[nm] = (int(a), int(max(a, b)))
                return targets, windows

            _pre_up = _peek_planner_upscale_settings(str(self.out_dir or ""))
            _pre_want_up = bool((_pre_up or {}).get("enabled", False))
            _pre_have_engine = bool(str((_pre_up or {}).get("engine_key") or "").strip())
            _pre_want_interp = bool((_pre_up or {}).get("interpolate_60fps_fast", False))

            _tail_steps: List[str] = []
            if bool(_use_videoclip_creator):
                _tail_steps.append(str(_mclip_step_name))
            else:
                _tail_steps.append(str(_assemble_step_name))
                try:
                    if bool(getattr(self.job, "music_background", False)) and str(getattr(self.job, "music_mode", "") or "") == "ace":
                        _tail_steps.append(str(_ace_step_name))
                except Exception:
                    pass
                try:
                    if str(getattr(self.job, "music_mode", "") or "") == "heartmula":
                        _tail_steps.append(str(_heartmula_step_name))
                except Exception:
                    pass
                _tail_steps.append(str(_audio_step_name))

            if bool(_pre_want_up and _pre_have_engine):
                _tail_steps.append("Upscale final cut ")
            if bool(_pre_want_interp):
                _tail_steps.append("Interpolate to 60fps ")

            _tail_targets, _tail_windows = _build_tail_targets(_tail_steps, start_pct=95, end_pct=99)

            def _tail_pct(step_name: str, fallback: int = 99) -> int:
                try:
                    return int(_tail_targets.get(str(step_name), int(fallback)))
                except Exception:
                    return int(fallback)

            try:
                if str(_assemble_step_name) in _tail_windows:
                    a, b = _tail_windows.get(str(_assemble_step_name), (95, 99))
                    _assemble_progress_win["start"] = int(a)
                    _assemble_progress_win["end"] = int(b)
            except Exception:
                pass

            _tail_pct_fn = _tail_pct
            # Run assemble/audio depending on preset selection
            did_videoclip = False
            if _use_videoclip_creator:
                # Even if no music/narration is available, we can still run Videoclip Creator by generating a silent audio bed.
                # Do NOT disable videoclip creator here; let step_videoclip_creator_10a handle the silent-audio fallback.
                try:
                    pass
                except Exception:
                    pass
                if _use_videoclip_creator:
                    preset_path2 = os.path.join(str(_root()), 'presets', 'setsave', 'plannerclip.json')
                    fp = _compute_mclip_fingerprint(str((manifest.get('paths') or {}).get('music_file') or narration_wav), preset_path2)
                    prev = (manifest.get('steps') or {}).get(_mclip_step_name) or {}
                    if _file_ok(final_video, 1024) and prev.get('fingerprint') == fp and prev.get('status') == 'done':
                        _skip(_mclip_step_name, 'videoclip creator output up-to-date (fingerprint match)')
                    else:
                        _run(_mclip_step_name, step_videoclip_creator_10a, _tail_pct_fn(_mclip_step_name))
                    # Mark step record
                    try:
                        srec2 = (manifest.get('steps') or {}).get(_mclip_step_name) or {}
                        srec2['fingerprint'] = fp
                        srec2.setdefault('debug', {}).update({'timeline_json': timeline_json})
                        srec2['ts'] = time.time()
                        manifest.setdefault('steps', {})[_mclip_step_name] = srec2
                        _safe_write_json(manifest_path, manifest)
                    except Exception:
                        pass
                
                # Verify/ensure stable output exists before skipping hardcuts.
                stable_final_cut = os.path.join(str(final_dir), "final_cut.mp4")
                try:
                    if (not _file_ok(stable_final_cut, 1024)) and _file_ok(final_video, 1024):
                        shutil.copy2(final_video, stable_final_cut)
                except Exception:
                    pass
                did_videoclip = _file_ok(stable_final_cut, 1024) or _file_ok(final_video, 1024)

                if did_videoclip:
                    # Skip the standard assemble + audio mux steps because videoclip creator already outputs a final mp4 with audio.
                    _log('[OK] Videoclip Creator finished; skipping hardcuts assembly + audio mux.')
                else:
                    _log('[WARN] Videoclip Creator was selected but produced no output; falling back to hardcuts assembly + audio mux.')
            if not did_videoclip:
                assemble_prev = (manifest.get("steps") or {}).get(_assemble_step_name) or {}
                if _file_ok(final_video, 1024) and _file_ok(timeline_json, 10) and assemble_prev.get("fingerprint") == _assembly_fingerprint and assemble_prev.get("status") == "done":
                    _skip(_assemble_step_name, "final cut up-to-date (fingerprint match)")
                else:
                    _run(_assemble_step_name, step_final, _tail_pct_fn(_assemble_step_name))



                # Skip / run audio step using fingerprint
                _audio_fingerprint = _compute_audio_fingerprint()
                audio_prev = (manifest.get("steps") or {}).get(_audio_step_name) or {}

                # If narration is enabled, do not skip the audio step unless the job-local narration.wav exists.
                _need_narr = bool(getattr(self.job, "narration_enabled", False)) and (not bool(getattr(self.job, "silent", False)))
                if _need_narr and (not _file_ok(narration_wav, 512)):
                    _run(_audio_step_name, step_audio_mix, _tail_pct_fn(_audio_step_name))
                elif _file_ok(final_cut_mp4, 1024) and audio_prev.get("fingerprint") == _audio_fingerprint and audio_prev.get("status") == "done":
                    _skip(_audio_step_name, "audio mix up-to-date (fingerprint match)")
                else:
                    _run(_audio_step_name, step_audio_mix, _tail_pct_fn(_audio_step_name))
                # Ensure audio step fingerprint is recorded for idempotency
                srec = (manifest.get("steps") or {}).get(_audio_step_name) or {}
                try:
                    srec["fingerprint"] = _audio_fingerprint
                    srec.setdefault("debug", {}).update({
                        "tts_log": tts_log,
                        "mix_log": mix_log,
                        "narration_txt": narration_txt,
                        "narration_wav": narration_wav,
                        "final_cut": final_cut_mp4,
                    })
                    srec.setdefault("note", "Narration + optional music mix.")
                    srec["ts"] = time.time()
                except Exception:
                    pass
                manifest.setdefault("steps", {})[_audio_step_name] = srec
                _safe_write_json(manifest_path, manifest)


            # Step I: Upscale final cut (Chunk 9B2) — post-processing only (no interpolation)
            # Conditions:
            # - Planner upscaling toggle enabled (planner_upscale.json -> enabled)
            # - Engine + model selected (Settings → Upscaling)
            _upscale_step_name = "Upscale final cut "

            def _read_planner_upscale_settings(job_dir2: str) -> dict:
                try:
                    p2 = os.path.join(str(job_dir2), _PLANNER_UPSCALE_JSON_NAME)
                    if not p2 or (not os.path.isfile(p2)):
                        return {}
                    with open(p2, "r", encoding="utf-8", errors="ignore") as f:
                        obj2 = json.load(f)
                    return obj2 if isinstance(obj2, dict) else {}
                except Exception:
                    return {}

            def _infer_model_scale(engine_label: str, model_text: str) -> int:
                s = (model_text or "").lower().replace(" ", "")
                # common patterns: x2 / 2x / _x4 / -x4-
                for pat in (r"x2", r"2x"):
                    if pat in s:
                        return 2
                for pat in (r"x4", r"4x"):
                    if pat in s:
                        return 4
                # Some engines imply 4× defaults; keep conservative and assume 2× when unknown
                # (we'll still resize to target factor afterward).
                lab = (engine_label or "").lower()
                if "4x" in lab or "x4" in lab:
                    return 4
                return 2

            def _probe_video_meta(video_path: str) -> dict:
                try:
                    args = [ffprobe2, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(video_path)]
                    cp = subprocess.run(args, cwd=str(_root()), capture_output=True, text=True)
                    if cp.returncode != 0:
                        return {}
                    obj = json.loads(cp.stdout or "{}")
                    return obj if isinstance(obj, dict) else {}
                except Exception:
                    return {}

            def _even2(n: int) -> int:
                try:
                    n = int(n)
                except Exception:
                    n = 0
                return max(2, int(n) // 2 * 2)

            def _src_video_bitrate_kbps(meta: dict) -> int:
                # Prefer container bit_rate; fallback: 0 (unknown)
                try:
                    br = int(float(((meta.get("format") or {}).get("bit_rate") or 0.0)))
                    if br <= 0:
                        return 0
                    return max(1, int(round(br / 1000.0)))
                except Exception:
                    return 0

            def _choose_target_bitrate_kbps(src_kbps: int, default_kbps: int = 3500) -> int:
                # "use source when it is already close to 3500"
                try:
                    src_kbps = int(src_kbps or 0)
                except Exception:
                    src_kbps = 0
                if src_kbps > 0:
                    if abs(src_kbps - int(default_kbps)) <= 500:
                        return int(src_kbps)
                return int(default_kbps)

            def _try_run_upsc_module(job_dict: dict) -> bool:
                # Best-effort: reuse helpers/upsc.py if it exposes a run API.
                try:
                    from helpers import upsc as _upsc  # type: ignore
                except Exception:
                    try:
                        import upsc as _upsc  # type: ignore
                    except Exception:
                        _upsc = None  # type: ignore

                if _upsc is None:
                    return False

                # Candidate function names used across versions
                candidates = [
                    "run_job",
                    "run_upscale_job",
                    "execute_job",
                    "process_job",
                    "run",
                    "upscale_video",
                    "run_video_upscale",
                    "run_video",
                    "upscale",
                ]

                for name in candidates:
                    fn = getattr(_upsc, name, None)
                    if not callable(fn):
                        continue

                    # Try a few common call signatures.
                    for mode in ("job", "kwargs", "args"):
                        try:
                            if mode == "job":
                                res = fn(job_dict)
                            elif mode == "kwargs":
                                res = fn(**job_dict)
                            else:
                                res = fn(
                                    job_dict.get("input_path") or job_dict.get("in_path"),
                                    job_dict.get("output_path") or job_dict.get("out_path"),
                                    job_dict.get("engine_label") or job_dict.get("engine"),
                                    job_dict.get("model_text") or job_dict.get("model"),
                                    job_dict.get("scale") or job_dict.get("factor") or 1,
                                )
                            # Interpret result
                            if isinstance(res, bool):
                                return bool(res)
                            if isinstance(res, int):
                                return int(res) == 0
                            if isinstance(res, dict):
                                if bool(res.get("ok")):
                                    return True
                                if "returncode" in res:
                                    return int(res.get("returncode") or 1) == 0
                            # Non-standard truthy result
                            if res is None:
                                # Some runners return None on success; treat as success if output exists.
                                outp = str(job_dict.get("output_path") or "")
                                if outp and os.path.exists(outp) and os.path.getsize(outp) > 1024:
                                    return True
                        except TypeError:
                            continue
                        except Exception:
                            # Don't fail the whole pipeline just because one candidate signature didn't match.
                            continue
                return False

            def _run_ncnn_folder_engine(engine_exe: str, engine_label: str, model_text: str, in_dir: str, out_dir2: str, scale: int, log_path: str) -> None:
                # Fallback runner for the common NCNN Vulkan CLIs (Real-ESRGAN / RealSR / SRMD family).
                # This is used only if helpers/upsc.py doesn't expose a usable run API.
                exe = str(engine_exe or "")
                if not exe:
                    raise RuntimeError("Upscale engine exe is empty.")
                if not os.path.exists(exe):
                    raise RuntimeError(f"Upscale engine exe not found: {exe}")

                args = [exe]

                # Most supported tools follow: -i <in> -o <out> -n <model> -s <scale> -f <ext>
                args += ["-i", str(in_dir), "-o", str(out_dir2)]

                # Some engines may not require a model flag; keep it when provided.
                if model_text and model_text.strip() and model_text.strip() != "(default)":
                    args += ["-n", str(model_text).strip()]

                if scale and int(scale) > 1:
                    args += ["-s", str(int(scale))]

                # Prefer PNG outputs for reliable ffmpeg ingest.
                args += ["-f", "png"]

                cp = subprocess.run(args, cwd=str(_root()), capture_output=True, text=True)
                try:
                    _safe_write_text(
                        log_path,
                        "[cmd] " + " ".join([str(x) for x in args]) + "\n\n"
                        + "--- STDOUT ---\n" + (cp.stdout or "") + "\n\n"
                        + "--- STDERR ---\n" + (cp.stderr or "") + "\n"
                    )
                except Exception:
                    pass
                if cp.returncode != 0:
                    tail = (cp.stderr or "")[-2000:] if (cp.stderr or "") else "Unknown error"
                    raise RuntimeError(f"Upscale engine failed (exit={cp.returncode}).\n{tail}")

            def step_upscale_final_cut_9b2() -> None:
                # Always retain raw final_cut path
                manifest.setdefault("paths", {})["final_cut_path"] = final_cut_mp4

                job_dir2 = str(self.out_dir or "")
                settings = _read_planner_upscale_settings(job_dir2)
                if not bool(settings.get("enabled", False)):
                    raise RuntimeError("internal: called but planner upscaling is disabled")

                engine_key = str(settings.get("engine_key") or "").strip().lower()

                # Input must be the final_cut only (single MP4)
                if not _file_ok(final_cut_mp4, 1024):
                    raise RuntimeError(f"Final Cut MP4 missing/too small: {final_cut_mp4}")

                # Output goes to /final as final_cut_upscaled.mp4
                out_up = os.path.join(final_dir, "final_cut_upscaled.mp4")
                os.makedirs(final_dir, exist_ok=True)
                log_up = os.path.join(final_dir, "upscale_log.txt")

                # Real-ESRGAN x4plus (normal speed)
                if engine_key == "realesrgan4x":
                    rootp = str(_root())
                    mdir = os.path.join(rootp, "models", "realesrgan")
                    exe = os.path.join(mdir, "realesrgan-ncnn-vulkan.exe")
                    model_base = os.path.join(mdir, "realesrgan-x4plus")

                    if not (os.path.isfile(model_base + ".bin") and os.path.isfile(model_base + ".param")):
                        raise RuntimeError(
                            "RealESRGAN 4x model is missing.\n\n"
                            "Expected:\n"
                            f"  {model_base}.bin\n"
                            f"  {model_base}.param"
                        )
                    if not os.path.isfile(exe):
                        raise RuntimeError(f"Real-ESRGAN engine exe not found: {exe}")

                    # Probe fps for frame-accurate re-encode
                    src_meta = _probe_video_meta(final_cut_mp4)
                    fps = 20.0
                    try:
                        for st in (src_meta.get("streams") or []):
                            if isinstance(st, dict) and st.get("codec_type") == "video":
                                r = st.get("avg_frame_rate") or st.get("r_frame_rate") or ""
                                rr = str(r or "")
                                if rr and "/" in rr:
                                    a, b = rr.split("/", 1)
                                    fps = float(a) / float(b) if float(b) != 0.0 else float(a)
                                elif rr:
                                    fps = float(rr)
                                break
                    except Exception:
                        fps = 20.0
                    if fps <= 0.1:
                        fps = 20.0

                    # Temp dirs
                    frames_in = os.path.join(final_dir, "_up_re4x_in")
                    frames_out = os.path.join(final_dir, "_up_re4x_out")
                    try:
                        if os.path.isdir(frames_in):
                            shutil.rmtree(frames_in, ignore_errors=True)
                        if os.path.isdir(frames_out):
                            shutil.rmtree(frames_out, ignore_errors=True)
                    except Exception:
                        pass
                    os.makedirs(frames_in, exist_ok=True)
                    os.makedirs(frames_out, exist_ok=True)

                    # Extract frames
                    in_pat = os.path.join(frames_in, "%08d.png")
                    cmd_extract = [
                        ffmpeg2, "-hide_banner", "-y",
                        "-i", str(final_cut_mp4),
                        "-vsync", "0",
                        "-start_number", "1",
                        str(in_pat),
                    ]
                    self._tick("[upscale] RealESRGAN 4x: extracting frames…", self._last_pct, 0.01)
                    cp0 = subprocess.run(cmd_extract, cwd=str(_root()), capture_output=True, text=True)
                    try:
                        _safe_write_text(
                            log_up,
                            "[extract_cmd] " + " ".join([str(x) for x in cmd_extract]) + "\n\n"
                            + (cp0.stderr or "") + "\n"
                        )
                    except Exception:
                        pass
                    if cp0.returncode != 0:
                        tail = (cp0.stderr or cp0.stdout or "")[-2000:]
                        raise RuntimeError(f"ffmpeg frame extract failed (exit={cp0.returncode}).\n{tail}")

                    # Run Real-ESRGAN on folder
                    model_name = "realesrgan-x4plus"
                    cmd_up = [
                        exe,
                        "-i", str(frames_in),
                        "-o", str(frames_out),
                        "-n", model_name,
                        "-s", "4",
                        "-m", str(mdir),
                        "-f", "png",
                    ]
                    self._tick("[upscale] RealESRGAN 4x: upscaling frames…", self._last_pct, 0.01)
                    cp1 = subprocess.run(cmd_up, cwd=str(_root()), capture_output=True, text=True)
                    try:
                        _safe_write_text(
                            log_up,
                            (_safe_read_text(log_up) or "")
                            + "\n\n[upscale_cmd] " + " ".join([str(x) for x in cmd_up]) + "\n\n"
                            + "--- STDOUT ---\n" + (cp1.stdout or "") + "\n\n"
                            + "--- STDERR ---\n" + (cp1.stderr or "") + "\n"
                        )
                    except Exception:
                        pass
                    if cp1.returncode != 0:
                        tail = (cp1.stderr or cp1.stdout or "")[-4000:]
                        raise RuntimeError(f"RealESRGAN upscaler failed (exit={cp1.returncode}).\n{tail}")

                    # Re-encode to MP4 (copy audio when possible)
                    out_pat = os.path.join(frames_out, "%08d.png")
                    tmp_out = os.path.join(final_dir, "final_cut_upscaled__tmp.mp4")
                    cmd_enc = [
                        ffmpeg2, "-hide_banner", "-y",
                        "-framerate", str(float(fps)),
                        "-start_number", "1",
                        "-i", str(out_pat),
                        "-i", str(final_cut_mp4),
                        "-map", "0:v:0",
                        "-map", "1:a?",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "18",
                        "-pix_fmt", "yuv420p",
                        "-c:a", "copy",
                        "-movflags", "+faststart",
                        "-shortest",
                        str(tmp_out),
                    ]
                    self._tick("[upscale] RealESRGAN 4x: encoding mp4…", self._last_pct, 0.01)
                    cp2 = subprocess.run(cmd_enc, cwd=str(_root()), capture_output=True, text=True)
                    if cp2.returncode != 0 or (not _file_ok(tmp_out, 1024)):
                        # fallback: AAC audio
                        cmd_enc2 = list(cmd_enc)
                        try:
                            ai = cmd_enc2.index("copy")
                            cmd_enc2[ai] = "aac"
                            cmd_enc2.insert(ai + 1, "-b:a")
                            cmd_enc2.insert(ai + 2, "192k")
                        except Exception:
                            pass
                        cp3 = subprocess.run(cmd_enc2, cwd=str(_root()), capture_output=True, text=True)
                        try:
                            _safe_write_text(
                                log_up,
                                (_safe_read_text(log_up) or "")
                                + "\n\n[encode_cmd] " + " ".join([str(x) for x in cmd_enc]) + "\n"
                                + (cp2.stderr or "") + "\n\n"
                                + "[encode_fallback_cmd] " + " ".join([str(x) for x in cmd_enc2]) + "\n"
                                + (cp3.stderr or "") + "\n"
                            )
                        except Exception:
                            pass
                        if cp3.returncode != 0 or (not _file_ok(tmp_out, 1024)):
                            tail = (cp3.stderr or cp2.stderr or "")[-4000:]
                            raise RuntimeError(f"ffmpeg encode failed (exit={cp3.returncode}).\n{tail}")
                    else:
                        try:
                            _safe_write_text(
                                log_up,
                                (_safe_read_text(log_up) or "")
                                + "\n\n[encode_cmd] " + " ".join([str(x) for x in cmd_enc]) + "\n"
                                + (cp2.stderr or "") + "\n"
                            )
                        except Exception:
                            pass

                    # Move into place
                    try:
                        if os.path.exists(out_up):
                            os.remove(out_up)
                    except Exception:
                        pass
                    try:
                        os.replace(tmp_out, out_up)
                    except Exception:
                        try:
                            import shutil
                            shutil.copy2(tmp_out, out_up)
                        except Exception:
                            pass
                        pass

                    # Record manifest paths
                    manifest.setdefault("paths", {})["final_upscaled_path"] = out_up
                    manifest.setdefault("paths", {})["current_final_path"] = out_up
                    srec = (manifest.get("steps") or {}).get(_upscale_step_name) or {}
                    try:
                        srec["status"] = "done"
                        srec["note"] = "RealESRGAN x4plus: final_cut → final_cut_upscaled.mp4"
                        srec["ts"] = time.time()
                    except Exception:
                        pass
                    manifest.setdefault("steps", {})[_upscale_step_name] = srec
                    _safe_write_json(manifest_path, manifest)
                    
                    # Cleanup temp work folders
                    try:
                        if os.path.isdir(frames_in):
                            shutil.rmtree(frames_in, ignore_errors=True)
                        if os.path.isdir(frames_out):
                            shutil.rmtree(frames_out, ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        if os.path.exists(tmp_out):
                            os.remove(tmp_out)
                    except Exception:
                        pass
                    return

                # SeedVR2 runner (HQ / slow)
                elif engine_key == "seedvr2":
                    # Resolve settings with safe defaults
                    resolution = int(settings.get("seedvr2_resolution") or 1080)
                    temporal_overlap = 1 if int(settings.get("seedvr2_temporal_overlap") or 0) else 0
                    batch_size = int(settings.get("seedvr2_batch_size") or 1)
                    chunk_size = int(settings.get("seedvr2_chunk_size") or 20)
                    color_corr = str(settings.get("seedvr2_color_correction") or "lab").strip() or "lab"
                    attn_mode = str(settings.get("seedvr2_attention_mode") or "sdpa").strip() or "sdpa"
                    dit_model = str(settings.get("seedvr2_dit_model") or settings.get("model_text") or "seedvr2_ema_3b-Q4_K_M.gguf").strip()

                    seed_py = os.path.join(str(_root()), "environments", ".seedvr2", "Scripts", "python.exe")
                    seed_cli = os.path.join(str(_root()), "presets", "extra_env", "seedvr2_src", "ComfyUI-SeedVR2_VideoUpscaler", "inference_cli.py")
                    model_dir = os.path.join(str(_root()), "models", "SEEDVR2")

                    if not os.path.exists(seed_py):
                        raise RuntimeError(f"SeedVR2 python not found: {seed_py}")
                    if not os.path.exists(seed_cli):
                        raise RuntimeError(f"SeedVR2 inference_cli.py not found: {seed_cli}")
                    if not os.path.isdir(model_dir):
                        raise RuntimeError(f"SeedVR2 model_dir not found: {model_dir}")

                    # Make ffmpeg discoverable (presets/bin)
                    env = dict(os.environ)
                    try:
                        env["PATH"] = str(os.path.join(str(_root()), "presets", "bin")) + os.pathsep + str(env.get("PATH") or "")
                    except Exception:
                        pass

                    # Windows console encoding can be cp1252; SeedVR2 prints emoji in logs.
                    # Force UTF-8 so the child process doesn't crash with UnicodeEncodeError.
                    try:
                        env["PYTHONUTF8"] = "1"
                        env["PYTHONIOENCODING"] = "utf-8"
                    except Exception:
                        pass

                    cmd = [
                        seed_py,
                        "-X", "utf8",
                        seed_cli,
                        str(final_cut_mp4),
                        "--output", str(out_up),
                        "--output_format", "mp4",
                        "--video_backend", "ffmpeg",
                        "--model_dir", str(model_dir),
                        "--dit_model", str(dit_model),
                        "--resolution", str(int(resolution)),
                        "--batch_size", str(int(batch_size)),
                        "--chunk_size", str(int(chunk_size)),
                        "--temporal_overlap", str(int(temporal_overlap)),
                        "--color_correction", str(color_corr),
                        "--attention_mode", str(attn_mode),
                    ]

                    self._tick("[upscale] SeedVR2 upscaling…", self._last_pct, 0.02)
                    cp = subprocess.run(
                        cmd,
                        cwd=str(_root()),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        env=env,
                    )
                    try:
                        _safe_write_text(
                            log_up,
                            "[cmd] " + " ".join([str(x) for x in cmd]) + "\n\n"
                            + "--- STDOUT ---\n" + (cp.stdout or "") + "\n\n"
                            + "--- STDERR ---\n" + (cp.stderr or "") + "\n"
                        )
                    except Exception:
                        pass
                    if cp.returncode != 0:
                        tail = (cp.stderr or cp.stdout or "")[-4000:]
                        raise RuntimeError(f"SeedVR2 failed (exit={cp.returncode}).\n{tail}")

                    if not _file_ok(out_up, 1024):
                        raise RuntimeError("SeedVR2 produced no output mp4 (missing/too small). See upscale_log.txt.")

                    # Ensure audio is preserved (SeedVR2 sometimes outputs video-only depending on backend).
                    try:
                        src_meta = _probe_video_meta(final_cut_mp4)
                        out_meta = _probe_video_meta(out_up)
                        src_has_audio = any((isinstance(st, dict) and st.get("codec_type") == "audio") for st in (src_meta.get("streams") or []))
                        out_has_audio = any((isinstance(st, dict) and st.get("codec_type") == "audio") for st in (out_meta.get("streams") or []))
                        if src_has_audio and (not out_has_audio):
                            tmp_mux = os.path.join(final_dir, "final_cut_upscaled__mux.mp4")
                            mux_cmd = [
                                ffmpeg2, "-hide_banner", "-y",
                                "-i", out_up,
                                "-i", final_cut_mp4,
                                "-map", "0:v:0",
                                "-map", "1:a?",
                                "-c", "copy",
                                "-shortest",
                                "-movflags", "+faststart",
                                tmp_mux
                            ]
                            cp2 = subprocess.run(mux_cmd, cwd=str(_root()), capture_output=True, text=True, env=env)
                            try:
                                _safe_write_text(log_up, (_safe_read_text(log_up) or "") + "\n\n[audio_remux] " + " ".join(mux_cmd) + "\n" + (cp2.stderr or ""))
                            except Exception:
                                pass
                            if cp2.returncode == 0 and _file_ok(tmp_mux, 1024):
                                try:
                                    os.replace(tmp_mux, out_up)
                                except Exception:
                                    shutil.copy2(tmp_mux, out_up)
                                    try:
                                        os.remove(tmp_mux)
                                    except Exception:
                                        pass
                    except Exception:
                        # Non-fatal: keep video-only output if remux fails
                        pass

                    # Record manifest paths
                    manifest.setdefault("paths", {})["final_upscaled_path"] = out_up
                    manifest.setdefault("paths", {})["current_final_path"] = out_up
                    srec = (manifest.get("steps") or {}).get(_upscale_step_name) or {}
                    try:
                        srec["status"] = "done"
                        srec["note"] = "SeedVR2: final_cut → final_cut_upscaled.mp4"
                        srec["ts"] = time.time()
                    except Exception:
                        pass

                    manifest.setdefault("steps", {})[_upscale_step_name] = srec
                    _safe_write_json(manifest_path, manifest)
                    return

                # Anime upscaler (Real-ESRGAN x4plus-anime)
                elif engine_key == "anime":
                    rootp = str(_root())
                    mdir = os.path.join(rootp, "models", "realesrgan")
                    exe = os.path.join(mdir, "realesrgan-ncnn-vulkan.exe")
                    model_base = os.path.join(mdir, "realesrgan-x4plus-anime")
                    # Require model files
                    if not (os.path.isfile(model_base + ".bin") and os.path.isfile(model_base + ".param")):
                        raise RuntimeError(
                            "Anime upscaler model is missing.\n\n"
                            "Expected:\n"
                            f"  {model_base}.bin\n"
                            f"  {model_base}.param"
                        )
                    if not os.path.isfile(exe):
                        raise RuntimeError(f"Real-ESRGAN engine exe not found: {exe}")

                    # Probe fps + audio presence
                    src_meta = _probe_video_meta(final_cut_mp4)
                    fps = 20.0
                    try:
                        for st in (src_meta.get("streams") or []):
                            if isinstance(st, dict) and st.get("codec_type") == "video":
                                r = st.get("avg_frame_rate") or st.get("r_frame_rate") or ""
                                rr = str(r or "")
                                if rr and "/" in rr:
                                    a, b = rr.split("/", 1)
                                    fps = float(a) / float(b) if float(b) != 0.0 else float(a)
                                elif rr:
                                    fps = float(rr)
                                break
                    except Exception:
                        fps = 20.0
                    if fps <= 0.1:
                        fps = 20.0

                    # Temp dirs
                    frames_in = os.path.join(final_dir, "_up_anime_in")
                    frames_out = os.path.join(final_dir, "_up_anime_out")
                    try:
                        if os.path.isdir(frames_in):
                            shutil.rmtree(frames_in, ignore_errors=True)
                        if os.path.isdir(frames_out):
                            shutil.rmtree(frames_out, ignore_errors=True)
                    except Exception:
                        pass
                    os.makedirs(frames_in, exist_ok=True)
                    os.makedirs(frames_out, exist_ok=True)

                    # Extract frames
                    in_pat = os.path.join(frames_in, "%08d.png")
                    cmd_extract = [
                        ffmpeg2, "-hide_banner", "-y",
                        "-i", str(final_cut_mp4),
                        "-vsync", "0",
                        "-start_number", "1",
                        str(in_pat),
                    ]
                    self._tick("[upscale] Anime: extracting frames…", self._last_pct, 0.01)
                    cp0 = subprocess.run(cmd_extract, cwd=str(_root()), capture_output=True, text=True)
                    try:
                        _safe_write_text(
                            log_up,
                            "[extract_cmd] " + " ".join([str(x) for x in cmd_extract]) + "\n\n"
                            + (cp0.stderr or "") + "\n"
                        )
                    except Exception:
                        pass
                    if cp0.returncode != 0:
                        tail = (cp0.stderr or cp0.stdout or "")[-2000:]
                        raise RuntimeError(f"ffmpeg frame extract failed (exit={cp0.returncode}).\n{tail}")

                    # Run Real-ESRGAN on folder
                    model_name = "realesrgan-x4plus-anime"
                    cmd_up = [
                        exe,
                        "-i", str(frames_in),
                        "-o", str(frames_out),
                        "-n", model_name,
                        "-s", "4",
                        "-m", str(mdir),
                        "-f", "png",
                    ]
                    self._tick("[upscale] Anime: upscaling frames…", self._last_pct, 0.01)
                    cp1 = subprocess.run(cmd_up, cwd=str(_root()), capture_output=True, text=True)
                    try:
                        _safe_write_text(
                            log_up,
                            (_safe_read_text(log_up) or "")
                            + "\n\n[upscale_cmd] " + " ".join([str(x) for x in cmd_up]) + "\n\n"
                            + "--- STDOUT ---\n" + (cp1.stdout or "") + "\n\n"
                            + "--- STDERR ---\n" + (cp1.stderr or "") + "\n"
                        )
                    except Exception:
                        pass
                    if cp1.returncode != 0:
                        tail = (cp1.stderr or cp1.stdout or "")[-4000:]
                        raise RuntimeError(f"Anime upscaler failed (exit={cp1.returncode}).\n{tail}")

                    # Re-encode to MP4 (copy audio when possible)
                    out_pat = os.path.join(frames_out, "%08d.png")
                    tmp_out = os.path.join(final_dir, "final_cut_upscaled__tmp.mp4")
                    cmd_enc = [
                        ffmpeg2, "-hide_banner", "-y",
                        "-framerate", str(float(fps)),
                        "-start_number", "1",
                        "-i", str(out_pat),
                        "-i", str(final_cut_mp4),
                        "-map", "0:v:0",
                        "-map", "1:a?",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "18",
                        "-pix_fmt", "yuv420p",
                        "-c:a", "copy",
                        "-movflags", "+faststart",
                        "-shortest",
                        str(tmp_out),
                    ]
                    self._tick("[upscale] Anime: encoding mp4…", self._last_pct, 0.01)
                    cp2 = subprocess.run(cmd_enc, cwd=str(_root()), capture_output=True, text=True)
                    if cp2.returncode != 0 or (not _file_ok(tmp_out, 1024)):
                        # fallback: AAC audio
                        cmd_enc2 = list(cmd_enc)
                        try:
                            ai = cmd_enc2.index("copy")
                            cmd_enc2[ai] = "aac"
                            cmd_enc2.insert(ai + 1, "-b:a")
                            cmd_enc2.insert(ai + 2, "192k")
                        except Exception:
                            pass
                        cp3 = subprocess.run(cmd_enc2, cwd=str(_root()), capture_output=True, text=True)
                        try:
                            _safe_write_text(
                                log_up,
                                (_safe_read_text(log_up) or "")
                                + "\n\n[encode_cmd] " + " ".join([str(x) for x in cmd_enc]) + "\n"
                                + (cp2.stderr or "") + "\n\n"
                                + "[encode_fallback_cmd] " + " ".join([str(x) for x in cmd_enc2]) + "\n"
                                + (cp3.stderr or "") + "\n"
                            )
                        except Exception:
                            pass
                        if cp3.returncode != 0 or (not _file_ok(tmp_out, 1024)):
                            tail = (cp3.stderr or cp2.stderr or "")[-4000:]
                            raise RuntimeError(f"ffmpeg encode failed (exit={cp3.returncode}).\n{tail}")
                    else:
                        try:
                            _safe_write_text(
                                log_up,
                                (_safe_read_text(log_up) or "")
                                + "\n\n[encode_cmd] " + " ".join([str(x) for x in cmd_enc]) + "\n"
                                + (cp2.stderr or "") + "\n"
                            )
                        except Exception:
                            pass

                    # Move into place
                    try:
                        if os.path.exists(out_up):
                            os.remove(out_up)
                    except Exception:
                        pass
                    try:
                        os.replace(tmp_out, out_up)
                    except Exception:
                        # fallback copy
                        try:
                            import shutil
                            shutil.copy2(tmp_out, out_up)
                        except Exception:
                            pass
                        pass
                    # Record manifest paths
                    manifest.setdefault("paths", {})["final_upscaled_path"] = out_up
                    manifest.setdefault("paths", {})["current_final_path"] = out_up
                    srec = (manifest.get("steps") or {}).get(_upscale_step_name) or {}
                    try:
                        srec["status"] = "done"
                        srec["note"] = "RealESRGAN x4plus-anime: final_cut → final_cut_upscaled.mp4"
                        srec["ts"] = time.time()
                    except Exception:
                        pass
                    manifest.setdefault("steps", {})[_upscale_step_name] = srec
                    _safe_write_json(manifest_path, manifest)
                    
                    # Cleanup temp work folders
                    try:
                        if os.path.isdir(frames_in):
                            shutil.rmtree(frames_in, ignore_errors=True)
                        if os.path.isdir(frames_out):
                            shutil.rmtree(frames_out, ignore_errors=True)
                    except Exception:
                        pass
                    try:
                        if os.path.exists(tmp_out):
                            os.remove(tmp_out)
                    except Exception:
                        pass
                    return


                # Unknown engine key (safety)
                raise RuntimeError(f"Unknown upscaler engine_key: {engine_key}")

            # info: End polish — fade out audio before optional upscaling (applies to final_cut.mp4)
            _fade_step_name = "Fade out audio (last 3s)"
            def _step_fade_out_audio() -> None:
                # Apply an audio fade-out to avoid abrupt music endings (HeartMula/AceStep/user audio).
                # Writes a temp file then replaces final_cut.mp4 atomically.
                nonlocal final_cut_mp4
                if not _file_ok(final_cut_mp4, 1024):
                    raise RuntimeError(f"Final cut missing/too small: {final_cut_mp4}")

                # Probe duration and ensure there is an audio stream
                meta = _probe_video_meta(final_cut_mp4)
                has_audio = False
                try:
                    for st in (meta.get("streams") or []):
                        if isinstance(st, dict) and st.get("codec_type") == "audio":
                            has_audio = True
                            break
                except Exception:
                    has_audio = False
                if not has_audio:
                    # No audio to fade; nothing to do.
                    return

                dur = 0.0
                try:
                    dur = float(_probe_duration_sec(final_cut_mp4) or 0.0)
                except Exception:
                    dur = 0.0
                if dur <= 0.05:
                    return

                fade_d = 3.0
                if dur < fade_d:
                    fade_d = max(0.0, dur)
                if fade_d <= 0.05:
                    return
                fade_st = max(0.0, dur - fade_d)

                tmp_out = os.path.join(final_dir, "final_cut__fade_tmp.mp4")
                try:
                    if os.path.exists(tmp_out):
                        os.remove(tmp_out)
                except Exception:
                    pass

                af = f"afade=t=out:st={fade_st:.3f}:d={fade_d:.3f}"
                cmd = [
                    str(ffmpeg2), "-y",
                    "-i", str(final_cut_mp4),
                    "-map", "0:v:0",
                    "-map", "0:a:0?",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-af", af,
                    "-movflags", "+faststart",
                    "-shortest",
                    str(tmp_out),
                ]
                cp = subprocess.run(cmd, cwd=str(_root()), capture_output=True, text=True)
                if cp.returncode != 0 or (not _file_ok(tmp_out, 1024)):
                    tail = (cp.stderr or "")[-2000:] if (cp.stderr or "") else "Unknown error"
                    raise RuntimeError(f"Audio fade failed (exit={cp.returncode}).\n{tail}")

                # Replace final_cut.mp4 atomically
                try:
                    os.replace(tmp_out, final_cut_mp4)
                except Exception:
                    # Fallback: copy then delete
                    shutil.copy2(tmp_out, final_cut_mp4)
                    try:
                        os.remove(tmp_out)
                    except Exception:
                        pass

            # Run fade-out step only when needed (idempotent via manifest post_sha1).
            try:
                _cur_sha1 = _sha1_file(final_cut_mp4) if os.path.exists(final_cut_mp4) else ""
                _fade_prev = (manifest.get("steps") or {}).get(_fade_step_name) or {}
                if _fade_prev.get("status") == "done" and str(_fade_prev.get("post_sha1") or "") == str(_cur_sha1 or "") and _file_ok(final_cut_mp4, 1024):
                    _skip(_fade_step_name, "already applied")
                else:
                    # Determine whether there is audio; if none, skip without failing.
                    _meta_f = _probe_video_meta(final_cut_mp4) if _file_ok(final_cut_mp4, 1024) else {}
                    _has_a = False
                    try:
                        for _st in (_meta_f.get("streams") or []):
                            if isinstance(_st, dict) and _st.get("codec_type") == "audio":
                                _has_a = True
                                break
                    except Exception:
                        _has_a = False

                    if not _has_a:
                        _skip(_fade_step_name, "no audio stream")
                    else:
                        _pre_sha1 = _cur_sha1
                        _run(_fade_step_name, _step_fade_out_audio, _tail_pct_fn(_fade_step_name))
                        # record post-sha1 for idempotency
                        try:
                            _post_sha1 = _sha1_file(final_cut_mp4) if os.path.exists(final_cut_mp4) else ""
                            srec = (manifest.get("steps") or {}).get(_fade_step_name) or {}
                            srec["fingerprint"] = _sha1_text(json.dumps({
                                "pre_sha1": str(_pre_sha1 or ""),
                                "fade_sec": 3.0,
                                "ffmpeg": str(ffmpeg2),
                            }, sort_keys=True))
                            srec["post_sha1"] = str(_post_sha1 or "")
                            srec["ts"] = time.time()
                            manifest.setdefault("steps", {})[_fade_step_name] = srec
                            _safe_write_json(manifest_path, manifest)
                        except Exception:
                            pass
            except Exception:
                # Never fail the whole run because of fade-out; it's a polish step.
                try:
                    _skip(_fade_step_name, "failed (non-fatal)")
                except Exception:
                    pass

# Decide whether to run Chunk 9B2 upscaling
            try:
                # Always retain raw final_cut path in manifest, even if we skip.
                manifest.setdefault("paths", {})["final_cut_path"] = final_cut_mp4
                _safe_write_json(manifest_path, manifest)
            except Exception:
                pass
            _up_settings = _read_planner_upscale_settings(str(self.out_dir or ""))
            _want_up = bool(_up_settings.get("enabled", False))
            _have_engine = bool(str(_up_settings.get("engine_key") or "").strip())

            _model_scale = 0
            try:
                # Only meaningful for legacy NCNN scale-based upscalers; SeedVR2 uses absolute resolution.
                if _have_engine and str(_up_settings.get("engine_key") or "").strip().lower() not in ("seedvr2",):
                    _model_scale = int(_infer_model_scale(str(_up_settings.get("engine_label") or ""), str(_up_settings.get("model_text") or "")))
            except Exception:
                _model_scale = 0

            # Fingerprint for idempotency
            _up_fp = _sha1_text(json.dumps({
                "input_sha1": _sha1_file(final_cut_mp4) if os.path.exists(final_cut_mp4) else "",
                "enabled": bool(_want_up),
                "engine_key": str(_up_settings.get("engine_key") or ""),
                # legacy fields kept for compatibility / debugging
                "engine_label": str(_up_settings.get("engine_label") or ""),
                "engine_exe": str(_up_settings.get("engine_exe") or ""),
                "model_text": str(_up_settings.get("model_text") or ""),
                # SeedVR2 knobs
                "seedvr2_resolution": int(_up_settings.get("seedvr2_resolution") or 0),
                "seedvr2_temporal_overlap": int(_up_settings.get("seedvr2_temporal_overlap") or 0),
                "seedvr2_batch_size": int(_up_settings.get("seedvr2_batch_size") or 0),
                "seedvr2_chunk_size": int(_up_settings.get("seedvr2_chunk_size") or 0),
                "seedvr2_color_correction": str(_up_settings.get("seedvr2_color_correction") or ""),
                "seedvr2_attention_mode": str(_up_settings.get("seedvr2_attention_mode") or ""),
                "seedvr2_dit_model": str(_up_settings.get("seedvr2_dit_model") or ""),
                # legacy scale info
                "model_scale": int(_model_scale),
                "ffmpeg": str(ffmpeg2),
                "ffprobe": str(ffprobe2),
            }, sort_keys=True))

            up_prev = (manifest.get("steps") or {}).get(_upscale_step_name) or {}
            out_up_path = os.path.join(final_dir, "final_cut_upscaled.mp4")

            if _want_up and _have_engine:
                if _file_ok(out_up_path, 1024) and up_prev.get("fingerprint") == _up_fp and up_prev.get("status") == "done":
                    _skip(_upscale_step_name, "upscaled output up-to-date (fingerprint match)")
                else:
                    _run(_upscale_step_name, step_upscale_final_cut_9b2, _tail_pct_fn(_upscale_step_name))
                    # store fingerprint after successful run
                    try:
                        srec = (manifest.get("steps") or {}).get(_upscale_step_name) or {}
                        srec["fingerprint"] = _up_fp
                        srec["ts"] = time.time()
                        manifest.setdefault("steps", {})[_upscale_step_name] = srec
                        _safe_write_json(manifest_path, manifest)
                    except Exception:
                        pass
            else:
                # Skip if any precondition is not met
                why = []
                if not _want_up:
                    why.append("upscaling toggle is off")
                if not _have_engine:
                    why.append("engine/model not selected")
                _skip(_upscale_step_name, ", ".join(why) if why else "conditions not met")



            

            # Step J: Interpolate final output to 60fps (Chunk 9C) — post-processing only
            # Conditions:
            # - Planner interpolation toggle enabled (planner_upscale.json -> interpolate_60fps_fast)
            # - Input selection:
            #     * If an upscaled final exists (from 9B2): interpolate that
            #     * Else: interpolate the raw final cut video
            # Fixed settings (no UI):
            # - Assume source FPS is 20
            # - Interpolate x3 -> output 60fps
            # - Use interp.py "Fast" profile: FFmpeg minterpolate (MCI), preset fast
            _interp_step_name = "Interpolate to 60fps "
            def _step_interpolate_60fps() -> None:
                # Input selection
                src = out_up_path if _file_ok(out_up_path, 1024) else final_cut_mp4
                if not _file_ok(src, 1024):
                    raise RuntimeError("No valid source video for interpolation.")

                # Output naming (working filenames only)
                if os.path.basename(src).lower().startswith("final_cut_upscaled"):
                    out_interp = os.path.join(final_dir, "final_cut_upscaled_60fps.mp4")
                else:
                    out_interp = os.path.join(final_dir, "final_cut_60fps.mp4")

                log_path = os.path.join(final_dir, "interp_60fps_log.txt")

                # Fast interpolation path (matches interp.py "Fast" profile): FFmpeg minterpolate (MCI)
                # Fixed settings:
                # - assume src fps is 20
                # - target fps is 60 (x3)
                vf = "minterpolate=fps=60:mi_mode=blend"

                cmd = [
                    ffmpeg2, "-hide_banner", "-y",
                    "-i", src,
                    "-map", "0:v:0", "-map", "0:a?",
                    "-vf", vf,
                    "-c:v", "libx264", "-preset", "veryfast", "-threads", "0",
                    "-b:v", "3500k", "-maxrate", "3500k", "-bufsize", "7000k",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    "-shortest",
                    out_interp
                ]

                with open(log_path, "a", encoding="utf-8", errors="ignore") as lf:
                    lf.write(f"\n[ffmpeg] m make the videoplay more smooth (interpolate to 60fps\n")
                    try:
                        lf.write("\n[cmd] " + " ".join([str(x) for x in cmd]) + "\n")
                    except Exception:
                        pass

                try:
                    with open(log_path, "a", encoding="utf-8", errors="ignore") as lf:
                        subprocess.run(cmd, check=True, stdout=lf, stderr=subprocess.STDOUT)
                except Exception:
                    # Fallback: re-encode audio to AAC (video policy unchanged)
                    cmd2 = list(cmd)
                    try:
                        ai = cmd2.index("copy")
                        cmd2[ai] = "aac"
                        cmd2.insert(ai + 1, "-b:a")
                        cmd2.insert(ai + 2, "192k")
                    except Exception:
                        cmd2 = [
                            ffmpeg2, "-hide_banner", "-y",
                            "-i", src,
                            "-map", "0:v:0", "-map", "0:a?",
                            "-vf", vf,
                            "-c:v", "libx264", "-preset", "veryfast", "-threads", "0",
                            "-b:v", "3500k", "-maxrate", "3500k", "-bufsize", "7000k",
                            "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k",
                            "-movflags", "+faststart",
                            "-shortest",
                            out_interp
                        ]
                    with open(log_path, "a", encoding="utf-8", errors="ignore") as lf:
                        lf.write("\n[ffmpeg] encode 60fps (audio aac fallback)\n")
                        try:
                            lf.write("\n[cmd] " + " ".join([str(x) for x in cmd2]) + "\n")
                        except Exception:
                            pass
                        subprocess.run(cmd2, check=True, stdout=lf, stderr=subprocess.STDOUT)

                if not _file_ok(out_interp, 1024):
                    raise RuntimeError("Interpolation output file was not created correctly.")

                # Update manifest pointers
                manifest.setdefault("paths", {})["final_interpolated_path"] = out_interp
                manifest["paths"]["final_video"] = out_interp
                manifest["final_video"] = out_interp
                _safe_write_json(manifest_path, manifest)

            # Apply interpolation if enabled
            _up_settings2 = _read_planner_upscale_settings(str(self.out_dir or ""))
            _want_interp = bool((_up_settings2 or {}).get("interpolate_60fps_fast", False))

            if _want_interp:
                # Determine current source for fingerprinting
                _interp_src = out_up_path if _file_ok(out_up_path, 1024) else final_cut_mp4
                _interp_out = os.path.join(final_dir, "final_cut_upscaled_60fps.mp4") if os.path.basename(_interp_src).lower().startswith("final_cut_upscaled") else os.path.join(final_dir, "final_cut_60fps.mp4")

                _interp_meta = {
                    "v": 2,
                    "src": os.path.basename(_interp_src),
                    "src_sha1": _sha1_file(_interp_src),
                    "engine": "ffmpeg-minterpolate",
                    "vf": "minterpolate=fps=60:mi_mode=blend",
                    "preset": "veryfast",
                    "fps_in": 20,
                    "fps_out": 60,
                    "vb_kbps": 3500,
                }
                _interp_fp = _sha1_text(json.dumps(_interp_meta, sort_keys=True))

                srec = (manifest.get("steps") or {}).get(_interp_step_name) or {}
                prev_fp = str(srec.get("fingerprint") or "")
                if _file_ok(_interp_out, 1024) and (prev_fp == _interp_fp) and (str(srec.get("status") or "") == "done"):
                    # Ensure pointers reflect the interpolated output
                    try:
                        manifest.setdefault("paths", {})["final_interpolated_path"] = _interp_out
                        manifest["paths"]["final_video"] = _interp_out
                        manifest["final_video"] = _interp_out
                        _safe_write_json(manifest_path, manifest)
                    except Exception:
                        pass
                    _skip(_interp_step_name, "already up-to-date")
                else:
                    _run(_interp_step_name, _step_interpolate_60fps, _tail_pct_fn(_interp_step_name))
                    try:
                        srec = manifest.get("steps", {}).get(_interp_step_name, {}) or {}
                        srec["fingerprint"] = _interp_fp
                        srec["ts"] = time.time()
                        srec["settings"] = _interp_meta
                        manifest.setdefault("steps", {})[_interp_step_name] = srec
                        _safe_write_json(manifest_path, manifest)
                    except Exception:
                        pass
            else:
                _skip(_interp_step_name, "toggle is off")

# README (always refresh, tiny)
            _safe_write_text(
                readme_path,
                "This job folder contains artifacts produced by the Planner pipeline:\n"
                "- story/plan.json (storyline + structure)\n"
                "- story/shots.json (shot list)\n"
                "- prompts/image_prompts.txt\n"
                "- prompts/i2v_prompts.txt\n"
                "- images/ (generated still images per shot)\n"
                "- clips/ (generated video clips per shot)\n"
                "- audio/ (music, narration, transcripts when enabled)\n"
                "- final/<jobid>_final.mp4 (assembled video)\n"
                "- manifest.json (paths + step statuses)\n"
                "Note: Upscaling and 60fps interpolation are optional post-steps (when enabled).\n"
            )

            try:
                self._maybe_rename_final_cut_pretty(output_dir=str(self.out_dir), manifest_path=str(manifest_path))
            except Exception:
                pass

            self._tick(f"Wrote/updated manifest: {manifest_path}", 100, 0.05)
            self._tick(f"Final video : {final_video}", 100, 0.10)

            result = {
                "job_id": self.job.job_id,
                "output_dir": self.out_dir,
                "final_video": (manifest.get("paths") or {}).get("final_video") or final_video,
                "job_config": asdict(self.job),
            }
            self.signals.finished.emit(result)
        except Exception as e:
            self.signals.failed.emit(str(e))


# -----------------------------
# UI
# -----------------------------


class _AssetsDotsDelegate(QStyledItemDelegate):
    """Paints small colored dots representing available assets in a job folder."""

    ROLE_MASK = int(Qt.UserRole) + 1

    # Bitmask
    HAS_SOUND = 1
    HAS_IMAGES = 2
    HAS_CLIPS = 4
    HAS_FINAL = 8
    HAS_UPSCALED = 16
    HAS_INTERPOLATED = 32

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        # Paint normal background/selection using the base delegate first (safe across styles),
        # then draw our dots on top.
        painter.save()
        try:
            opt = QStyleOptionViewItem(option)
        except Exception:
            opt = option
        try:
            self.initStyleOption(opt, index)
        except Exception:
            pass

        try:
            QStyledItemDelegate.paint(self, painter, opt, index)
        except Exception:
            if getattr(opt, "state", 0) & getattr(QStyleOptionViewItem, "State_Selected", 0):
                painter.fillRect(opt.rect, opt.palette.highlight())

        # Force clip to the full cell (some styles set tight clip regions during paint)
        try:
            painter.setClipping(True)
            painter.setClipRect(opt.rect)
        except Exception:
            pass

        try:
            mask = int(index.data(self.ROLE_MASK) or 0)
        except Exception:
            mask = 0

        # dot order: sound, images, clips, final, upscaled, interpolated
        dots = []
        if mask & self.HAS_SOUND:
            dots.append(QColor(220, 60, 60))   # red
        if mask & self.HAS_IMAGES:
            dots.append(QColor(245, 200, 40))  # yellow
        if mask & self.HAS_CLIPS:
            dots.append(QColor(50, 120, 220))  # blue
        if mask & self.HAS_FINAL:
            dots.append(QColor(60, 180, 90))   # green
        if mask & self.HAS_UPSCALED:
            dots.append(QColor(255, 140, 0))   # orange
        if mask & self.HAS_INTERPOLATED:
            dots.append(QColor(170, 80, 200))  # violet

        if not dots:
            painter.restore()
            return

        r = opt.rect
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Scale radius with row height so it's visible on different DPI / row sizes
        try:
            radius = max(4, min(7, int(r.height() * 0.20)))
        except Exception:
            radius = 5
        gap = radius + 2

        total_w = len(dots) * (radius * 2) + (len(dots) - 1) * gap
        x0 = r.x() + max(0, (r.width() - total_w) // 2)
        cy = r.y() + r.height() // 2

        painter.setPen(Qt.NoPen)
        for i, col in enumerate(dots):
            x = x0 + i * (radius * 2 + gap)
            painter.setBrush(col)
            painter.drawEllipse(x, cy - radius, radius * 2, radius * 2)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index) -> QSize:
        """Ensure the dots column never collapses to 0 width."""
        try:
            h = int(getattr(option, 'rect', QRect()).height() or 24)
        except Exception:
            h = 24
        # Enough space for 6 dots + gaps
        return QSize(122, max(18, h))




class _FinalVideoPreviewDialog(QDialog):
    """Small popup preview for a final rendered video."""

    def __init__(self, parent: QWidget, video_path: str):
        super().__init__(parent)
        self.setWindowTitle("Final result preview")
        self.setModal(True)
        try:
            self.resize(920, 560)
        except Exception:
            pass

        self._video_path = str(video_path or "")
        self._dragging = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        # Top bar: filename + close button
        top = QHBoxLayout()
        top.setSpacing(8)
        lab = QLabel(os.path.basename(self._video_path) or "final")
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(lab, 1)

        btn_close = QToolButton()
        try:
            btn_close.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        except Exception:
            btn_close.setText("X")
        btn_close.clicked.connect(self.close)
        top.addWidget(btn_close)
        lay.addLayout(top)

        self.video_widget = QVideoWidget()
        lay.addWidget(self.video_widget, 1)

        # Controls
        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.btn_play = QPushButton("Play")
        controls.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        controls.addWidget(self.slider, 1)

        lay.addLayout(controls)

        # Player
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        try:
            self._audio.setVolume(1.0)
        except Exception:
            pass

        try:
            self._player.setAudioOutput(self._audio)
        except Exception:
            pass
        try:
            self._player.setVideoOutput(self.video_widget)
        except Exception:
            pass

        try:
            if QUrl is not None:
                self._player.setSource(QUrl.fromLocalFile(self._video_path))
        except Exception:
            pass

        # Wiring
        self.btn_play.clicked.connect(self._toggle_play)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.slider.sliderMoved.connect(self._on_slider_moved)

        try:
            self._player.durationChanged.connect(self._on_duration)
            self._player.positionChanged.connect(self._on_position)
            self._player.playbackStateChanged.connect(self._on_state)
        except Exception:
            pass

    @Slot()
    def _toggle_play(self) -> None:
        try:
            st = self._player.playbackState()
            if st == QMediaPlayer.PlayingState:
                self._player.pause()
            else:
                self._player.play()
        except Exception:
            pass

    @Slot()
    def _on_state(self, _state=None) -> None:
        try:
            st = self._player.playbackState()
            self.btn_play.setText("Pause" if st == QMediaPlayer.PlayingState else "Play")
        except Exception:
            pass

    @Slot()
    def _on_duration(self, dur: int) -> None:
        try:
            self.slider.setRange(0, max(0, int(dur)))
        except Exception:
            pass

    @Slot()
    def _on_position(self, pos: int) -> None:
        if self._dragging:
            return
        try:
            self.slider.setValue(int(pos))
        except Exception:
            pass

    @Slot()
    def _on_slider_pressed(self) -> None:
        self._dragging = True

    @Slot()
    def _on_slider_released(self) -> None:
        self._dragging = False
        try:
            self._player.setPosition(int(self.slider.value()))
        except Exception:
            pass

    @Slot()
    def _on_slider_moved(self, v: int) -> None:
        try:
            self._player.setPosition(int(v))
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        try:
            self._player.stop()
        except Exception:
            pass
        super().closeEvent(event)


def _is_video_path(p: str) -> bool:
    try:
        ext = os.path.splitext(str(p or "").lower())[1]
    except Exception:
        ext = ""
    return ext in (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif")


class _HeaderPreviewDialog(QDialog):
    """Popup preview for a header thumbnail (image or looping video)."""

    def __init__(self, parent: QWidget, media_path: str):
        super().__init__(parent)
        self.setWindowTitle("Preview")
        self.setModal(True)
        try:
            self.resize(980, 620)
        except Exception:
            pass

        self._path = str(media_path or "")
        self._is_video = _is_video_path(self._path)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(8)

        top = QHBoxLayout()
        top.setSpacing(8)
        lab = QLabel(os.path.basename(self._path) or "preview")
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        top.addWidget(lab, 1)

        btn_close = QToolButton()
        try:
            btn_close.setIcon(self.style().standardIcon(QStyle.SP_TitleBarCloseButton))
        except Exception:
            btn_close.setText("Close")
        btn_close.clicked.connect(self.close)
        top.addWidget(btn_close)
        lay.addLayout(top)

        self._player = None
        self._audio = None
        self._video_widget = None

        if self._is_video and QMediaPlayer is not None and QVideoWidget is not None and QUrl is not None:
            self._video_widget = QVideoWidget()
            lay.addWidget(self._video_widget, 1)

            try:
                self._player = QMediaPlayer(self)
                self._audio = QAudioOutput(self)
                try:
                    self._audio.setVolume(0.0)  # silent loop
                except Exception:
                    pass
                try:
                    self._player.setAudioOutput(self._audio)
                except Exception:
                    pass
                try:
                    self._player.setVideoOutput(self._video_widget)
                except Exception:
                    pass
                try:
                    self._player.setSource(QUrl.fromLocalFile(self._path))
                except Exception:
                    pass

                try:
                    self._player.mediaStatusChanged.connect(self._on_media_status)
                except Exception:
                    pass

                try:
                    self._player.play()
                except Exception:
                    pass
            except Exception:
                ph = QLabel("Cannot preview video on this build.")
                ph.setAlignment(Qt.AlignCenter)
                lay.addWidget(ph, 1)
        else:
            img = QLabel()
            img.setAlignment(Qt.AlignCenter)
            img.setMinimumHeight(240)
            pm = QPixmap(self._path)
            if pm.isNull():
                img.setText("Cannot preview this file.")
            else:
                img.setPixmap(pm.scaled(1280, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            lay.addWidget(img, 1)

    @Slot()
    def _on_media_status(self, st=None) -> None:
        try:
            if self._player is None:
                return
            if st == QMediaPlayer.EndOfMedia:
                self._player.setPosition(0)
                self._player.play()
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        try:
            if self._player is not None:
                self._player.stop()
        except Exception:
            pass
        super().closeEvent(event)


class _ClickableThumbLabel(QLabel):
    """Simple clickable QLabel used for tiny header thumbnails."""
    clicked = Signal()

    def mousePressEvent(self, ev):  # type: ignore[override]
        try:
            self.clicked.emit()
        except Exception:
            pass
        try:
            super().mousePressEvent(ev)
        except Exception:
            pass



class PlannerPane(QWidget):
    """
    A wire-ready pane for "Prompt -> Finished Video".
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        # Queue mode (Settings → Use FrameVision Queue)
        # Default ON to preserve current behavior.
        try:
            _s = _load_planner_settings()
            self._use_framevision_queue = bool(_s.get("use_framevision_queue", True))
        except Exception:
            self._use_framevision_queue = True

        # Planner state
        self._music_file_path = ""
        self._worker: Optional[PipelineWorker] = None
        self._last_result: Optional[dict] = None
        # Chunk 9B1: current project folder for Planner-only upscaling settings
        self._active_project_dir: str = ""
        self._image_review_dialog: Optional[ImageReviewDialog] = None
        self._clip_review_dialog: Optional[ClipReviewDialog] = None

        # Multi-job queue (sequential runs inside Planner)
        # When a job is already running, new Generate clicks will enqueue jobs here.
        self._pending_jobs: List[dict] = []

        # Chunk 4: reference images strategy
        self._ref_strategy: str = ""  # qwen3vl_describe | qwen2511_best | qwenvl_reuse
        self._ref_multi_angle_lora: str = ""  # auto-detected path when available
        self._ref_qwen2511_high_quality: bool = False  # Chunk 4: Qwen 2511 HQ toggle (1280x720)

        # UI lock: when Qwen Edit 2511 is used for reference images, it becomes the image-creation engine.
        # In that mode, the Image model dropdown is disabled and shows a hint instead of allowing conflicting choices.
        self._img_model_lock_active: bool = False
        self._img_model_prev_index: int = -1

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # --- Sticky top banner (copied from Tools tab "fancy banner") ---
        # Sits above the scrollable content so it never moves.
        try:
            self.planner_banner = QLabel("The Planner : Idea -> Click -> Result")
            self.planner_banner.setObjectName("plannerTopBanner")
            self.planner_banner.setAlignment(Qt.AlignCenter)
            self.planner_banner.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.planner_banner.setFixedHeight(48)
            self.planner_banner.setStyleSheet(
                "#plannerTopBanner {"
                " font-size: 15px;"
                " font-weight: 600;"
                " padding: 8px 17px;"
                " border-radius: 12px;"
                " margin: 0 0 6px 0;"
                " color: rgba(232, 246, 255, 245);"
                " background: qlineargradient("
                "   x1:0, y1:0, x2:1, y2:0,"
                "   stop:0 #061327,"
                "   stop:0.5 #0b2a67,"
                "   stop:1 #04101e"
                " );"
                " letter-spacing: 0.5px;"
                "}"
            )
            root.addWidget(self.planner_banner)
            root.addSpacing(4)
        except Exception:
            pass

        header = QWidget()
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)


        # --- Footer style (moved header to bottom) ---
        try:
            header.setObjectName("plannerFooter")
            header.setMinimumHeight(78)
            header.setStyleSheet("""
                QWidget#plannerFooter {
                    background: rgba(8, 12, 10, 220);
	                    /* Remove the visible outline ring (it was only showing on the right side due to overlapping children). */
	                    border: none;
                    border-radius: 18px;
                    padding: 10px 12px;
                }
                QWidget#plannerFooter QLabel {
                    color: rgba(255, 255, 255, 230);
                }
                QWidget#plannerFooter QPushButton {
                    padding: 8px 16px;
                    border-radius: 14px;
                    font-weight: 600;
                }
            """)
            eff = QGraphicsDropShadowEffect(header)
            eff.setBlurRadius(28)
            eff.setOffset(0, 10)
            eff.setColor(QColor(0, 0, 0, 170))
            header.setGraphicsEffect(eff)
        except Exception:
            pass


        
        # Title + tiny status line
        title_box = QWidget()
        title_box_lay = QVBoxLayout(title_box)
        title_box_lay.setContentsMargins(0, 0, 0, 0)
        title_box_lay.setSpacing(0)

        title = QLabel("")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        f.setUnderline(True)
        title.setFont(f)

        # Elapsed runtime (updates on log/progress events; not a live timer)
        self.lbl_header_elapsed = QLabel("")
        try:
            ef = QFont()
            ef.setPointSize(9)
            self.lbl_header_elapsed.setFont(ef)
        except Exception:
            pass
        try:
            self.lbl_header_elapsed.setStyleSheet("opacity: 0.75; padding-left: 10px;")
        except Exception:
            pass

        self.lbl_header_status = QLabel("Idle")
        try:
            sf = QFont()
            sf.setPointSize(9)
            self.lbl_header_status.setFont(sf)
        except Exception:
            pass
        try:
            self.lbl_header_status.setStyleSheet("opacity: 0.75;")
        except Exception:
            pass

        title_box_lay.addWidget(title)
        title_box_lay.addWidget(self.lbl_header_elapsed)
        title_box_lay.addWidget(self.lbl_header_status)

        header_row.addWidget(title_box)

        # Preview strip (optional, toggled in Settings)
        self._preview_items = []  # list of dicts: {path, kind}
        self._header_preview_dialog = None
        # Cache for video thumbnails (first-frame PNGs) so we don't re-run ffmpeg repeatedly.
        # key: video_path -> (mtime, thumb_png_path)
        self._video_thumb_cache: Dict[str, Tuple[float, str]] = {}
        self._video_thumb_dir = str((_root() / "presets" / "setsave" / "planner_preview_thumbs").resolve())
        try:
            os.makedirs(self._video_thumb_dir, exist_ok=True)
        except Exception:
            pass
        self._preview_strip = QWidget()
        _ps = QHBoxLayout(self._preview_strip)
        _ps.setContentsMargins(0, 0, 0, 0)
        _ps.setSpacing(6)

        self._preview_thumbs = []
        for _i in range(5):
            t = _ClickableThumbLabel()
            t.setFixedSize(72, 40)
            t.setScaledContents(True)
            try:
                t.setFrameShape(QFrame.StyledPanel)
                t.setFrameShadow(QFrame.Plain)
            except Exception:
                pass
            try:
                t.setCursor(Qt.PointingHandCursor)
            except Exception:
                pass
            t.setToolTip("Click to preview")
            # capture index safely
            try:
                t.clicked.connect(lambda _=None, idx=_i: self._on_preview_thumb_clicked(idx))
            except Exception:
                pass
            _ps.addWidget(t)
            self._preview_thumbs.append(t)

        self._preview_strip.setVisible(False)
        header_row.addWidget(self._preview_strip)

        header_row.addStretch(1)

        # Generate / Cancel live in the header so they are always visible (even when switching tabs).

        self.btn_generate = QPushButton("Generate")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        def _on_generate_clicked():
            # Local handler to avoid Qt binding issues with missing method attributes.
            try:
                job = self._build_job()
            except Exception as e:
                QMessageBox.warning(self, "Cannot start", str(e))
                return
            try:
                _LOGGER.log_probe(f"Generate clicked: {job.job_id}")
                _LOGGER.log_job(job.job_id, "Generate clicked")
            except Exception:
                pass
            output_base = (self.out_dir_edit.text() or "").strip() or self._default_out_dir()
            output_base = _abspath_from_root(output_base)
            # Human-friendly folder naming: <slug>_<NNN>
            title = _auto_title_from_prompt(job.prompt)
            slug = _slugify_title(title) or "job"
            run_n = _next_slug_counter(output_base, slug)
            out_dir = os.path.join(output_base, f"{slug}_{run_n:03d}")

            # If a job is already running, queue this job for later.
            try:
                if self._is_planner_busy():
                    self._enqueue_job(job, out_dir, title=title, slug=slug)
                    return
            except Exception:
                pass


            # Start immediately
            self._run_job(job, out_dir, title=title, slug=slug)



        def _on_cancel_clicked():
            # Local handler to avoid Qt binding issues with missing method attributes.
            try:
                # Prefer the class method if present (keeps existing cancel logic).
                self._cancel_pipeline()
                return
            except AttributeError:
                pass
            except Exception:
                # If the method exists but errors, fall back to a minimal safe cancel.
                pass

            # Minimal fallback: request worker interruption/stop if running.
            try:
                if self._worker and self._worker.isRunning():
                    try:
                        self._worker.requestInterruption()
                    except Exception:
                        pass
                    try:
                        self._worker.stop()
                    except Exception:
                        pass
            except Exception:
                pass
        self.btn_generate.clicked.connect(_on_generate_clicked)
        self.btn_cancel.clicked.connect(_on_cancel_clicked)
        header_row.addWidget(self.btn_generate)
        header_row.addWidget(self.btn_cancel)

        # Header status state
        self._header_stage = ""
        self._header_pct = 0
        self._header_mode = "idle"


        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        root.addWidget(self.tabs, 1)

        # Footer (sticky controls)
        root.addWidget(header)

        # -----------------
        # Generate tab
        # -----------------
        generate_tab = QWidget()
        generate_layout = QVBoxLayout(generate_tab)
        generate_layout.setContentsMargins(0, 0, 0, 0)
        generate_layout.setSpacing(10)

        gen_left = QWidget()
        gen_left_layout = QVBoxLayout(gen_left)
        gen_left_layout.setContentsMargins(0, 0, 0, 0)
        gen_left_layout.setSpacing(10)

        gen_left_layout.addWidget(self._build_step1_group())
        gen_left_layout.addWidget(self._build_optional_group())
        gen_left_layout.addStretch(1)

        gen_scroll = QScrollArea()
        gen_scroll.setWidgetResizable(True)
        gen_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        gen_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        gen_scroll.setFrameShape(QFrame.NoFrame)
        gen_scroll.setWidget(gen_left)

        generate_layout.addWidget(gen_scroll, 1)
        self.tabs.addTab(generate_tab, "Generate")

        
        # -----------------
        # Settings tab
        # -----------------
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(8)

        # Logs toggle (collapsed by default)
        logs_toggle_row = QHBoxLayout()
        logs_toggle_row.setContentsMargins(0, 0, 0, 0)
        logs_toggle_row.setSpacing(8)

        self.btn_toggle_logs = QToolButton()
        self.btn_toggle_logs.setText("Show logs")
        self.btn_toggle_logs.setCheckable(True)
        self.btn_toggle_logs.setChecked(False)
        try:
            self.btn_toggle_logs.setToolButtonStyle(Qt.ToolButtonTextOnly)
        except Exception:
            pass

        logs_toggle_row.addWidget(self.btn_toggle_logs)
        logs_toggle_row.addStretch(1)
        settings_layout.addLayout(logs_toggle_row)

        self._settings_splitter = QSplitter(Qt.Vertical)

        # Top: Run settings (scrollable if needed)
        run_wrap = QWidget()
        run_wrap_lay = QVBoxLayout(run_wrap)
        run_wrap_lay.setContentsMargins(0, 0, 0, 0)
        run_wrap_lay.setSpacing(10)
        run_wrap_lay.addWidget(self._build_run_group())
        run_wrap_lay.addStretch(1)

        run_scroll = QScrollArea()
        run_scroll.setWidgetResizable(True)
        run_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        run_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        run_scroll.setFrameShape(QFrame.NoFrame)
        run_scroll.setWidget(run_wrap)

        self._settings_splitter.addWidget(run_scroll)

        # Bottom: Logs (toggle/collapsible)
        self._logs_group = self._build_log_group()
        self._settings_splitter.addWidget(self._logs_group)

        try:
            self._settings_splitter.setCollapsible(0, False)
            self._settings_splitter.setCollapsible(1, True)
        except Exception:
            pass

        settings_layout.addWidget(self._settings_splitter, 1)

        # default: collapse logs
        try:
            self._logs_group.setVisible(False)
            self._settings_splitter.setSizes([1, 0])
        except Exception:
            pass

        self.btn_toggle_logs.toggled.connect(self._on_toggle_logs_visibility)

        self.tabs.addTab(settings_tab, "Settings")

# -----------------
        # Results tab
        # -----------------
        self._results_loaded = False
        results_tab = self._build_results_tab()
        self.tabs.addTab(results_tab, "Results / History")
        self.tabs.currentChanged.connect(self._on_tabs_changed)


        self._sync_toggle_visibility()
        self._sync_silent_logic()
        try:
            self._apply_auto_music_volume_default()
        except Exception:
            pass
        try:
            if hasattr(self, "chk_preview"):
                self._apply_preview_visibility(bool(self.chk_preview.isChecked()))
        except Exception:
            pass

        try:
            _LOGGER.log_probe("UI ready")
        except Exception:
            pass

    # -------------------------
    # Step 1 UI
    # -------------------------

    def _build_step1_group(self) -> QGroupBox:
        box = QGroupBox("What do you want to make?")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)        # Prompt
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Example: aliens go out to a nightclub and have fun")
        self.prompt_edit.setFixedHeight(70)

        self.prompt_edit.textChanged.connect(self._on_prompt_changed)

        lay.addWidget(QLabel("Prompt"))
        lay.addWidget(self.prompt_edit)

        # Negatives (auto defaults, resets on restart)
        self.negatives_edit = QTextEdit()
        self.negatives_edit.setToolTip("These are auto-injected negatives used to reduce common drift (cloning, collage, extra parts).\n"
                                       "You can edit/remove them for this session. They will reset to defaults on next restart.")
        self.negatives_edit.setFixedHeight(44)  # ~2 lines
        try:
            self.negatives_edit.setPlainText(_DEFAULT_UI_NEGATIVES)
        except Exception:
            self.negatives_edit.setPlainText("")
        lay.addWidget(QLabel("Negatives"))
        lay.addWidget(self.negatives_edit)

        # Extra info (concise)
        self.extra_info = QTextEdit()
        self.extra_info.setPlaceholderText(
            "Optional: style, colors, background info, mood (happy/sad), cinematic notes, etc.\n"
            "Example: neon cyberpunk club, playful mood, vibrant colors, fast cuts."
        )
        self.extra_info.setFixedHeight(44)  # ~2 lines
        lay.addWidget(QLabel("Extra info (free text)"))
        lay.addWidget(self.extra_info)

        # Toggles row
        toggles = QHBoxLayout()
        toggles.setSpacing(12)

        self.chk_story = QCheckBox("Narration")
        self.chk_music = QCheckBox("Music background")
        self.chk_silent = QCheckBox("Silent")

        self.chk_story.toggled.connect(self._sync_toggle_visibility)
        self.chk_story.toggled.connect(lambda _=None: self._apply_auto_music_volume_default())
        self.chk_music.toggled.connect(self._sync_toggle_visibility)
        self.chk_silent.toggled.connect(self._sync_silent_logic)

        toggles.addWidget(self.chk_story)
        toggles.addWidget(self.chk_music)
        toggles.addWidget(self.chk_silent)
        toggles.addStretch(1)

        lay.addLayout(toggles)


        # Volume controls
        self.story_vol_row = QWidget()
        svl = QHBoxLayout(self.story_vol_row)
        svl.setContentsMargins(0, 0, 0, 0)
        svl.setSpacing(10)
        svl.addWidget(QLabel("Story telling volume"))
        self.sld_story_vol = QSlider(Qt.Horizontal)
        self.sld_story_vol.setRange(0, 100)
        self.sld_story_vol.setValue(100)  # default 100%
        self.lbl_story_vol = QLabel("100%")
        self.lbl_story_vol.setMinimumWidth(48)
        self.sld_story_vol.valueChanged.connect(lambda v: self.lbl_story_vol.setText(f"{v}%"))
        svl.addWidget(self.sld_story_vol, 1)
        svl.addWidget(self.lbl_story_vol)

        self.music_vol_row = QWidget()
        mvl = QHBoxLayout(self.music_vol_row)
        mvl.setContentsMargins(0, 0, 0, 0)
        mvl.setSpacing(10)
        mvl.addWidget(QLabel("Music background volume"))
        self.sld_music_vol = QSlider(Qt.Horizontal)
        self.sld_music_vol.setRange(0, 100)
        self.sld_music_vol.setValue(25)  # default 25%
        self.lbl_music_vol = QLabel("25%")
        self.lbl_music_vol.setMinimumWidth(48)
        self.sld_music_vol.valueChanged.connect(lambda v: self.lbl_music_vol.setText(f"{v}%"))

        # Auto defaults: narration ON -> 25%, narration OFF -> 100% (resets on restart)
        try:
            self._music_volume_user_override = False
        except Exception:
            pass
        try:
            self.sld_music_vol.sliderPressed.connect(self._mark_music_volume_overridden)
            self.sld_music_vol.sliderMoved.connect(lambda _v=None: self._mark_music_volume_overridden())
            self.sld_music_vol.actionTriggered.connect(lambda _a=None: self._mark_music_volume_overridden())
        except Exception:
            pass
        mvl.addWidget(self.sld_music_vol, 1)
        mvl.addWidget(self.lbl_music_vol)

        # Chunk 6B: Narration controls (simple UI)
        self.narration_row = QWidget()
        nrl = QHBoxLayout(self.narration_row)
        nrl.setContentsMargins(0, 0, 0, 0)
        nrl.setSpacing(10)
        nrl.addWidget(QLabel("Voice"))
        self.cmb_narr_voice = QComboBox()
        # Built-in tokens (must match exact names)
        self.cmb_narr_voice.addItems([
    "aiden", "dylan", "eric", "ono_anna", "ryan", "serena", "sohee", "uncle_fu", "vivian",
    "add your own…",
        ])
        self.cmb_narr_voice.setCurrentText("ryan")
        self.cmb_narr_voice.currentTextChanged.connect(self._sync_silent_logic)
        nrl.addWidget(self.cmb_narr_voice, 1)

        nrl.addWidget(QLabel("Language"))
        self.cmb_narr_lang = QComboBox()
        # Must include exact token: auto
        self.cmb_narr_lang.addItems(["auto", "English", "Dutch", "German", "French", "Spanish", "Italian"])
        self.cmb_narr_lang.setCurrentText("auto")
        self.cmb_narr_lang.currentTextChanged.connect(self._sync_silent_logic)
        nrl.addWidget(self.cmb_narr_lang, 1)

        self.voice_sample_row = QWidget()
        vsr = QHBoxLayout(self.voice_sample_row)
        vsr.setContentsMargins(0, 0, 0, 0)
        vsr.setSpacing(10)
        vsr.addWidget(QLabel("Voice sample"))
        self.voice_sample_path_edit = QLineEdit()
        self.voice_sample_path_edit.setPlaceholderText("Select a voice sample file (required for clone)")
        self.voice_sample_path_edit.textChanged.connect(self._sync_silent_logic)
        vsr.addWidget(self.voice_sample_path_edit, 1)
        self.btn_voice_sample = QPushButton("Browse")
        self.btn_voice_sample.clicked.connect(self._browse_voice_sample)
        vsr.addWidget(self.btn_voice_sample)

        lay.addWidget(self.story_vol_row)
        lay.addWidget(self.narration_row)
        lay.addWidget(self.voice_sample_row)
        lay.addWidget(self.music_vol_row)

        # Duration slider (5s - 4 min)
        duration_row = QWidget()
        dr = QHBoxLayout(duration_row)
        dr.setContentsMargins(0, 0, 0, 0)
        dr.setSpacing(10)
        dr.addWidget(QLabel("Approx duration"))

        self.sld_duration = QSlider(Qt.Horizontal)
        self.sld_duration.setRange(5, 240)
        # Keep label in sync immediately on startup.
        # (Previously it defaulted to 30s until the user moved the slider.)
        self.lbl_duration = QLabel("")
        self.lbl_duration.setMinimumWidth(70)
        self.sld_duration.valueChanged.connect(lambda v: self.lbl_duration.setText(_duration_label(int(v))))
        self.sld_duration.setValue(15)
        self.lbl_duration.setText(_duration_label(int(self.sld_duration.value())))

        dr.addWidget(self.sld_duration, 1)
        dr.addWidget(self.lbl_duration)
        lay.addWidget(duration_row)


        try:
            self._sync_settings_summary()
        except Exception:
            pass

        # End of user work note
        end_note = QLabel("When you press Generate, the app will run the full pipeline (story → images → video clips → assembly).")
        end_note.setWordWrap(True)
        end_note.setStyleSheet("opacity: 0.9;")
        lay.addWidget(end_note)

        return box

    # -------------------------
    # Optional + Run + Logs (restored methods)
    # -------------------------

    def _build_optional_group(self) -> QGroupBox:
        box = QGroupBox()  # title removed
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        # -------------------------
        # Chunk 4: Reference images (collapsed by default)
        # -------------------------
        ref_head = QHBoxLayout()
        ref_head.setSpacing(8)

        self.btn_ref_toggle = QToolButton()
        self.btn_ref_toggle.setText("Add images (reference)")
        self.btn_ref_toggle.setCheckable(True)
        self.btn_ref_toggle.setChecked(False)
        self.btn_ref_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_ref_toggle.setArrowType(Qt.RightArrow)
        self.btn_ref_toggle.toggled.connect(self._toggle_ref_block)

        ref_head.addWidget(self.btn_ref_toggle)
        ref_head.addStretch(1)
        lay.addLayout(ref_head)

        self.ref_block = QWidget()
        ref_lay = QVBoxLayout(self.ref_block)
        ref_lay.setContentsMargins(18, 0, 0, 0)
        ref_lay.setSpacing(6)

        ref_row = QHBoxLayout()
        ref_row.setSpacing(8)
        self.btn_ref_add = QPushButton("Add…")
        self.btn_ref_remove = QPushButton("Remove selected")
        self.btn_ref_clear = QPushButton("Clear")

        self.btn_ref_add.clicked.connect(self._on_add_ref_images_clicked)
        self.btn_ref_remove.clicked.connect(self._remove_selected_ref_images)
        self.btn_ref_clear.clicked.connect(self._clear_ref_images)

        ref_row.addWidget(self.btn_ref_add)
        ref_row.addWidget(self.btn_ref_remove)
        ref_row.addWidget(self.btn_ref_clear)
        ref_row.addStretch(1)
        ref_lay.addLayout(ref_row)

        self.ref_list = QListWidget()
        self.ref_list.setMinimumHeight(85)
        self.ref_list.setToolTip("Reference images used to keep identity/objects consistent across the story.")
        ref_lay.addWidget(self.ref_list)

        self.lbl_ref_strategy = QLabel("Strategy: (none)")
        self.lbl_ref_strategy.setToolTip("Chosen reference strategy (stored in job.encoding).")
        ref_lay.addWidget(self.lbl_ref_strategy)

        self.ref_block.setVisible(False)
        lay.addWidget(self.ref_block)

        # -------------------------
        # Add or create music (Chunk 7A)
        # -------------------------
        music_head = QHBoxLayout()
        music_head.setSpacing(6)

        self.btn_music_toggle = QToolButton()
        self.btn_music_toggle.setText("Add or create music")
        self.btn_music_toggle.setCheckable(True)
        self.btn_music_toggle.setChecked(False)
        self.btn_music_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_music_toggle.setArrowType(Qt.RightArrow)
        self.btn_music_toggle.toggled.connect(self._toggle_music_block)

        music_head.addWidget(self.btn_music_toggle)
        music_head.addStretch(1)
        lay.addLayout(music_head)

        self.music_block = QWidget()
        music_lay = QVBoxLayout(self.music_block)
        music_lay.setContentsMargins(18, 0, 0, 0)
        music_lay.setSpacing(6)

        # Generate with Ace Step 1.5 (toggle-like radio). When off, no Ace settings are shown.
        row_ace15 = QHBoxLayout()
        row_ace15.setSpacing(8)
        self.rad_music_ace15 = QRadioButton("Generate with Ace Step 1.5")
        row_ace15.addWidget(self.rad_music_ace15)
        row_ace15.addStretch(1)
        music_lay.addLayout(row_ace15)

        # Ace Step 1.5 settings (hidden unless Ace Step 1.5 is selected)
        self.ace15_block = QWidget()
        ace15_lay = QVBoxLayout(self.ace15_block)
        ace15_lay.setContentsMargins(24, 0, 0, 0)
        ace15_lay.setSpacing(6)

        # Preset selection (loaded from presets/setsave/ace15presets/presetmanager.json)
        row_preset = QHBoxLayout()
        row_preset.setSpacing(8)
        row_preset.addWidget(QLabel("Preset:"))
        self.cmb_ace15_preset = QComboBox()
        self.cmb_ace15_preset.setToolTip("Select a genre preset. Duration is always the project duration.")
        try:
            self._load_ace15_preset_combo(self.cmb_ace15_preset)
        except Exception:
            try:
                self.cmb_ace15_preset.addItem("Default", "default")
            except Exception:
                pass
        row_preset.addWidget(self.cmb_ace15_preset, 1)
        ace15_lay.addLayout(row_preset)

        # Output format
        row_fmt = QHBoxLayout()
        row_fmt.setSpacing(8)
        row_fmt.addWidget(QLabel("Output:"))
        self.cmb_ace15_format = QComboBox()
        self.cmb_ace15_format.setToolTip("Audio output format. WAV is lossless and preferred for quality.")
        self.cmb_ace15_format.addItem("WAV (lossless)", "wav")
        self.cmb_ace15_format.addItem("MP3 (smaller)", "mp3")
        row_fmt.addWidget(self.cmb_ace15_format, 1)
        ace15_lay.addLayout(row_fmt)

        # Lyrics toggle + box
        row_lyrics = QHBoxLayout()
        row_lyrics.setSpacing(8)
        self.chk_ace15_lyrics = QCheckBox("Lyrics")
        self.chk_ace15_lyrics.setToolTip("When enabled, you can provide lyrics. If empty, the planner will auto-create lyrics from the storyline (later step).")
        row_lyrics.addWidget(self.chk_ace15_lyrics)
        row_lyrics.addStretch(1)
        ace15_lay.addLayout(row_lyrics)

        self.txt_ace15_lyrics = QPlainTextEdit()
        self.txt_ace15_lyrics.setPlaceholderText("When no lyrics are added the system will create lyrics based on the storyline")
        self.txt_ace15_lyrics.setMinimumHeight(80)
        ace15_lay.addWidget(self.txt_ace15_lyrics)

        music_lay.addWidget(self.ace15_block)

        # Use your own (file)
        row_file = QHBoxLayout()
        row_file.setSpacing(8)
        self.rad_music_file = QRadioButton("Use your own (file)")
        self.rad_music_file.setChecked(True)

        self.btn_music_add_file = QPushButton("Add…")
        self.btn_music_add_file.setToolTip("Attach an existing music file (will be copied into this run's audio/ folder).")
        self.btn_music_add_file.clicked.connect(self._choose_music_file)

        self.btn_music_clear_file = QPushButton("Clear")
        self.btn_music_clear_file.setToolTip("Remove the selected music file.")
        self.btn_music_clear_file.clicked.connect(self._clear_music_file)

        self.lbl_music_file = QLabel("No file selected")
        self.lbl_music_file.setWordWrap(True)
        try:
            self.lbl_music_file.setStyleSheet("opacity: 0.75;")
        except Exception:
            pass

        row_file.addWidget(self.rad_music_file)
        row_file.addWidget(self.btn_music_add_file)
        row_file.addWidget(self.btn_music_clear_file)
        row_file.addWidget(self.lbl_music_file, 1)
        music_lay.addLayout(row_file)

        note = QLabel("Tip: Select Ace Step 1.5 to generate music, or attach your own file. Whisper checks imported files only.")
        note.setWordWrap(True)
        note.setStyleSheet("opacity: 0.75;")
        music_lay.addWidget(note)
# Default state: hidden until expanded; disabled unless Music background is checked
        self.music_block.setVisible(False)
        lay.addWidget(self.music_block)

        try:
            self.rad_music_ace15.toggled.connect(self._sync_music_source_ui)
            self.rad_music_file.toggled.connect(self._sync_music_source_ui)
            self.chk_ace15_lyrics.toggled.connect(self._sync_music_source_ui)
            self._sync_music_source_ui()
        except Exception:
            pass



        return box

    # -------------------------
    # Run + logs UI
    # -------------------------

    def _build_run_group(self) -> QGroupBox:
        box = QGroupBox("Run")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        # Queue mode (default ON): use FrameVision internal queue lock behavior.
        q_row = QWidget()
        ql = QHBoxLayout(q_row)
        ql.setContentsMargins(0, 0, 0, 0)
        ql.setSpacing(10)

        self.chk_use_framevision_queue = QCheckBox("Use FrameVision Queue")
        try:
            f = self.chk_use_framevision_queue.font()
            f.setBold(True)
            self.chk_use_framevision_queue.setFont(f)
        except Exception:
            pass

        try:
            self.chk_use_framevision_queue.setChecked(bool(getattr(self, "_use_framevision_queue", True)))
        except Exception:
            self.chk_use_framevision_queue.setChecked(True)

        self.lbl_queue_mode_note = QLabel("")
        try:
            self.lbl_queue_mode_note.setWordWrap(True)
            self.lbl_queue_mode_note.setStyleSheet("opacity: 0.8;")
        except Exception:
            pass

        ql.addWidget(self.chk_use_framevision_queue)
        ql.addWidget(self.lbl_queue_mode_note, 1)
        lay.addWidget(q_row)

        try:
            self.chk_use_framevision_queue.toggled.connect(self._on_toggle_use_framevision_queue)
        except Exception:
            pass
        try:
            self._sync_use_framevision_queue_label()
        except Exception:
            pass

        # Top toggles (moved to top of Settings tab)
        self.chk_allow_edit_while_running = QCheckBox("Pause for review (check/edit images and videos after creation")
        self.chk_allow_edit_while_running.setToolTip(
            "When enabled, the pipeline pauses after images (and later after clips) so you can inspect outputs, "
            "edit per-shot prompts, regenerate individual images, then continue."
        )
        self.chk_allow_edit_while_running.setChecked(False)
        try:
            s = _load_planner_settings()
            self.chk_allow_edit_while_running.setChecked(bool(s.get("allow_edit_while_running", False)))
        except Exception:
            pass
        try:
            f = self.chk_allow_edit_while_running.font()
            f.setBold(True)
            self.chk_allow_edit_while_running.setFont(f)
        except Exception:
            pass
        self.chk_allow_edit_while_running.toggled.connect(self._on_toggle_allow_edit_while_running)
        lay.addWidget(self.chk_allow_edit_while_running)

        # Preview toggle (header thumbnails)
        self.chk_preview = QCheckBox("Preview last results")
        self.chk_preview.setToolTip("When enabled, show up to 5 thumbnails in the header for the last created images/clips.\nClick a thumbnail to open a bigger preview (videos loop until closed).")
        self.chk_preview.setChecked(False)
        try:
            s = _load_planner_settings()
            self.chk_preview.setChecked(bool(s.get("preview_enabled", False)))
        except Exception:
            pass
        try:
            f = self.chk_preview.font()
            f.setBold(True)
            self.chk_preview.setFont(f)
        except Exception:
            pass
        self.chk_preview.toggled.connect(self._on_toggle_preview)
        lay.addWidget(self.chk_preview)

        # Character Bible toggle (default ON). Turn OFF when using animals/non-human characters.
        self.chk_character_bible = QCheckBox("Enable Character bible")
        self.chk_character_bible.setToolTip(
            "When enabled, the planner creates a Character Bible and injects identity-lock details for shots that mention character names.\n"
            "Turn this OFF when your shots use animals or other non-human characters, to prevent the model from drifting into humans."
        )
        self.chk_character_bible.setChecked(True)
        try:
            s = _load_planner_settings()
            self.chk_character_bible.setChecked(bool(s.get("character_bible_enabled", True)))
        except Exception:
            pass
        try:
            f = self.chk_character_bible.font()
            f.setBold(True)
            self.chk_character_bible.setFont(f)
        except Exception:
            pass
        self.chk_character_bible.toggled.connect(self._on_toggle_character_bible)
        lay.addWidget(self.chk_character_bible)

        # Own Character Bible (manual 1–2) — overrides auto Character Bible when enabled
        self.chk_own_character_bible = QCheckBox("Own character bible")
        self.chk_own_character_bible.setToolTip(
            "When enabled, you can define up to TWO manual character prompts.\n"
            "These characters will be injected into EVERY image and video prompt for consistency.\n"
            "This overrides the auto Character Bible (only one can be on at a time)."
        )
        self.chk_own_character_bible.setChecked(False)
        try:
            s = _load_planner_settings()
            self.chk_own_character_bible.setChecked(bool(s.get("own_character_bible_enabled", False)))
        except Exception:
            pass
        try:
            f = self.chk_own_character_bible.font()
            f.setBold(True)
            self.chk_own_character_bible.setFont(f)
        except Exception:
            pass
        self.chk_own_character_bible.toggled.connect(self._on_toggle_own_character_bible)
        lay.addWidget(self.chk_own_character_bible)

        # Alternative storymode — direct Qwen prompt list (bypasses shot/distill pipeline for txt2img prompts)
        self.chk_alternative_storymode = QCheckBox("Alternative storymode")
        self.chk_alternative_storymode.setToolTip(
            "Use Qwen to generate a direct prompt list from SUBJECT+CONTEXT. Bypasses the shot/distill pipeline in an attempt to reduce prompt artifacts and improve scene variation."
        )
        self.chk_alternative_storymode.setChecked(False)
        try:
            s = _load_planner_settings()
            self.chk_alternative_storymode.setChecked(bool(s.get("alternative_storymode", False)))
        except Exception:
            pass
        try:
            f = self.chk_alternative_storymode.font()
            f.setBold(True)
            self.chk_alternative_storymode.setFont(f)
        except Exception:
            pass
        self.chk_alternative_storymode.toggled.connect(self._on_toggle_alternative_storymode)
        lay.addWidget(self.chk_alternative_storymode)

        # info: Chunk 10 side quest — Own storyline toggle + textbox (Step 1: UI only)
        self.chk_own_storyline = QCheckBox("Own storymode (paste your own prompts)")
        self.chk_own_storyline.setToolTip(
            """When enabled, you can paste your own storyline/prompt list.
Start each new prompt with a marker like [01] or (01).
(In later steps, this will bypass the planner logic and go straight to image creation.)"""
        )
        self.chk_own_storyline.setChecked(False)
        try:
            s = _load_planner_settings()
            self.chk_own_storyline.setChecked(bool(s.get("own_storyline_enabled", False)))
        except Exception:
            pass
        try:
            f = self.chk_own_storyline.font()
            f.setBold(True)
            self.chk_own_storyline.setFont(f)
        except Exception:
            pass
        self.chk_own_storyline.toggled.connect(self._on_toggle_own_storyline)
        lay.addWidget(self.chk_own_storyline)

        self.own_storyline_block = QGroupBox("")
        osl = QVBoxLayout(self.own_storyline_block)
        osl.setSpacing(6)

        self.own_storyline_edit = QTextEdit()
        self.own_storyline_edit.setPlaceholderText(
            """Paste your storyline here. Start each new prompt with a marker, for example:
[01] A woman walks through neon rain
[02] Close-up of her silver jacket

You can also use: (01) (02)
If the planner sees a marker like [02] or (02), it becomes the next image prompt."""
        )
        self.own_storyline_edit.setFixedHeight(140)

        try:
            s = _load_planner_settings()
            self.own_storyline_edit.setPlainText(str(s.get("own_storyline_text", "") or ""))
        except Exception:
            pass
        try:
            self.own_storyline_edit.textChanged.connect(self._on_own_storyline_text_changed)
        except Exception:
            pass

        osl.addWidget(self.own_storyline_edit)

        # info: Chunk 10 side quest — Own storyline prompt preview (Step 2: counter only)
        self.lbl_own_storyline_count = QLabel('Detected prompts: 0')
        try:
            self.lbl_own_storyline_count.setWordWrap(True)
        except Exception:
            pass

        self.lbl_own_storyline_warn = QLabel('')
        try:
            self.lbl_own_storyline_warn.setWordWrap(True)
            self.lbl_own_storyline_warn.setVisible(False)
        except Exception:
            pass

        osl.addWidget(self.lbl_own_storyline_count)
        osl.addWidget(self.lbl_own_storyline_warn)

        self.own_storyline_block.setVisible(bool(self.chk_own_storyline.isChecked()))
        try:
            self._apply_own_storyline_lock_state(bool(self.chk_own_storyline.isChecked()))
        except Exception:
            pass
        try:
            self._update_own_storyline_prompt_preview()
        except Exception:
            pass
        lay.addWidget(self.own_storyline_block)

        # Manual character prompt boxes (shown only when Own Character Bible is enabled)
        self.own_character_bible_block = QGroupBox("")
        own_lay = QVBoxLayout(self.own_character_bible_block)
        own_lay.setSpacing(6)

        lbl1 = QLabel("Character 1 prompt")
        self.own_char_1 = QTextEdit()
        self.own_char_1.setPlaceholderText(
            "Describe the character for consistency (appearance, outfit, vibe, signature details).\n"
            "Example: 'Young woman with short black bob, silver bomber jacket, neon eyeliner, always wears the same jacket.'"
        )
        self.own_char_1.setFixedHeight(84)

        lbl2 = QLabel("Character 2 prompt (optional)")
        self.own_char_2 = QTextEdit()
        self.own_char_2.setPlaceholderText(
            "Optional second character (kept consistent across all prompts).\n"
            "Leave empty if you only want one main character."
        )
        self.own_char_2.setFixedHeight(84)

        try:
            s = _load_planner_settings()
            self.own_char_1.setPlainText(str(s.get("own_character_1_prompt", "") or ""))
            self.own_char_2.setPlainText(str(s.get("own_character_2_prompt", "") or ""))
        except Exception:
            pass

        try:
            self.own_char_1.textChanged.connect(self._on_own_character_prompt_changed)
            self.own_char_2.textChanged.connect(self._on_own_character_prompt_changed)
        except Exception:
            pass

        own_lay.addWidget(lbl1)
        own_lay.addWidget(self.own_char_1)
        own_lay.addWidget(lbl2)
        own_lay.addWidget(self.own_char_2)

        self.own_character_bible_block.setVisible(bool(self.chk_own_character_bible.isChecked()))
        lay.addWidget(self.own_character_bible_block)





        # Chunk 9A: moved settings (Format) into Settings tab
        fmt = QGridLayout()
        fmt.setHorizontalSpacing(10)
        fmt.setVerticalSpacing(6)

        fmt.addWidget(QLabel("Format"), 0, 0)
        self.cmb_aspect = QComboBox()
        self.cmb_aspect.addItems(["Landscape (16:9)", "Portrait (9:16)", "1:1"])
        self.cmb_aspect.setCurrentIndex(0)
        fmt.addWidget(self.cmb_aspect, 0, 1)
#        self.lbl_bitrate_info = QLabel("Video bitrate: 3500k (fixed)")
#        self.lbl_bitrate_info.setStyleSheet("opacity: 0.8;")
#        fmt.addWidget(self.lbl_bitrate_info, 1, 0, 1, 2)

        lay.addLayout(fmt)

        try:
            self.cmb_aspect.currentIndexChanged.connect(lambda _=None: self._sync_settings_summary())
        except Exception:
            pass


        # Model selection
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(QLabel("Image model"), 0, 0)
        self.cmb_image_model = QComboBox()
        self.cmb_image_model.addItems([
            "Auto",
            "Qwen Image 2512",
            "SDXL (Lowest vram, fast, low quality)",
            "Z-image Turbo (FP16 High VRAM)",
            "Z-image Turbo (GGUF Low VRAM)",
            "More (Maybe later)",
        ])
        # Default text-to-image engine: Z-image Turbo (GGUF Low VRAM)
        self.cmb_image_model.setCurrentIndex(4)
        grid.addWidget(self.cmb_image_model, 0, 1)

        try:
            self._sync_image_model_lock_for_qwen2511_refs()
        except Exception:
            pass

        grid.addWidget(QLabel("Video model"), 1, 0)
        self.cmb_video_model = QComboBox()
        self.cmb_video_model.addItems([
            "Auto (Hunyuan) ",
            "WAN 2.2 (slow, 720p)",
            "HunyuanVideo 1.5",
        ])
        self.cmb_video_model.setCurrentIndex(0)
        grid.addWidget(self.cmb_video_model, 1, 1)

        grid.addWidget(QLabel("Generation quality"), 2, 0)
        self.cmb_gen_quality = QComboBox()
        # populated dynamically based on video model selection
        grid.addWidget(self.cmb_gen_quality, 2, 1)

        def _refresh_gen_quality() -> None:
            vm = (self.cmb_video_model.currentText() or "").lower()
            cur = (self.cmb_gen_quality.currentText() or "").strip()
            self.cmb_gen_quality.blockSignals(True)
            self.cmb_gen_quality.clear()
            if "wan" in vm and "2.2" in vm:
                self.cmb_gen_quality.addItems(["Low (default)", "Medium", "High"])
                if cur.lower().startswith("high"):
                    self.cmb_gen_quality.setCurrentIndex(2)
                elif cur.lower().startswith("medium"):
                    self.cmb_gen_quality.setCurrentIndex(1)
                else:
                    self.cmb_gen_quality.setCurrentIndex(0)
            else:
                # hunyuan + fallback
                # Default Generation quality: Low
                self.cmb_gen_quality.addItems(["Low (default)", "Medium", "High"])
                if cur.lower().startswith("low"):
                    self.cmb_gen_quality.setCurrentIndex(0)
                elif cur.lower().startswith("high"):
                    self.cmb_gen_quality.setCurrentIndex(2)
                else:
                    self.cmb_gen_quality.setCurrentIndex(0)
            self.cmb_gen_quality.blockSignals(False)

        self.cmb_video_model.currentIndexChanged.connect(lambda _=None: _refresh_gen_quality())
        _refresh_gen_quality()

        grid.addWidget(QLabel("Videoclip Creator preset"), 3, 0)
        self.cmb_videoclip_preset = QComboBox()
        self.cmb_videoclip_preset.addItems([
            "Storyline Preset (Hardcuts)",
            "Videoclip Preset (Transitions)",
            "Storyline Music videoclip",
            "Other",
        ])

        # Restore persisted preset choice (Chunk 10A)
        try:
            s = _load_planner_settings()
            choice = str(s.get("videoclip_creator_preset", "") or "")
            # Backwards-compat: older placeholder labels
            if choice == "Storyline Preset (Hardcuts / placeholder)":
                choice = "Storyline Preset (Hardcuts)"
            elif choice == "Videoclip Preset (Transitions / placeholder)":
                choice = "Videoclip Preset (Transitions)"
            elif choice == "Storyline Music videoclip / placeholder":
                choice = "Storyline Music videoclip"
            elif choice == "Other (placeholder)":
                choice = "Other"
            if choice:
                i = self.cmb_videoclip_preset.findText(choice)
                if i >= 0:
                    self.cmb_videoclip_preset.setCurrentIndex(i)
                else:
                    self.cmb_videoclip_preset.setCurrentIndex(0)
            else:
                self.cmb_videoclip_preset.setCurrentIndex(0)
        except Exception:
            self.cmb_videoclip_preset.setCurrentIndex(0)

        try:
            self.cmb_videoclip_preset.currentIndexChanged.connect(self._on_videoclip_creator_preset_changed)
        except Exception:
            pass

        grid.addWidget(self.cmb_videoclip_preset, 3, 1)

        lay.addLayout(grid)

        # Output folder
        out_row = QHBoxLayout()
        out_row.setSpacing(8)
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("Output folder (optional). Defaults to ./output/planner/")
        self.btn_browse_out = QPushButton("Browse…")
        self.btn_browse_out.clicked.connect(self._browse_out_dir)

        out_row.addWidget(self.out_dir_edit, 1)
        out_row.addWidget(self.btn_browse_out)
        lay.addLayout(out_row)

        # Chunk 9B1: Planner-only Upscaling (no execution yet)
        lay.addWidget(self._build_upscaling_group())

        # Chunk 9C: Optional interpolation post-step (60fps)
        lay.addWidget(self._build_interpolation_group())


        # Buttons row (Generate/Cancel live in the header so they're always visible)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_export_job = QPushButton("Export job JSON")
        self.btn_open_output = QPushButton("Open output folder")

        self.btn_open_output.setEnabled(False)

        self.btn_export_job.clicked.connect(self._export_job_json)
        self.btn_open_output.clicked.connect(self._open_output_folder)

        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_export_job)
        btn_row.addWidget(self.btn_open_output)

        lay.addLayout(btn_row)

        # Progress
        self.lbl_stage = QLabel("Stage: —")
        self.lbl_stage.setStyleSheet("opacity: 0.9;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)

        lay.addWidget(self.lbl_stage)
        lay.addWidget(self.progress)

        # Debug toggle: Probe & logs (writes to ./logs/). Keep OFF unless debugging.
        self.chk_probe_logs = QCheckBox("Probe & logs")
        self.chk_probe_logs.setToolTip("Debug only. Writes continuous logs to /logs/.\nKeep this OFF during normal use to avoid endless logging that can slow down the app.")
        try:
            self.chk_probe_logs.setChecked(bool(_LOGGER.enabled))
        except Exception:
            self.chk_probe_logs.setChecked(False)
        self.chk_probe_logs.toggled.connect(self._on_toggle_probe_logs)
        lay.addWidget(self.chk_probe_logs)

        # NOTE: dry-run hint removed (was a temporary dev note)

        return box

    # -------------------------
    # Chunk 9B1: Planner-only Upscaling settings (UI + per-project JSON)
    # -------------------------

    def _upscale_json_path(self, project_dir: str) -> str:
        try:
            return os.path.join(str(project_dir), _PLANNER_UPSCALE_JSON_NAME)
        except Exception:
            return str(project_dir or "")

    def _read_upscale_json(self, project_dir: str) -> dict:
        p = self._upscale_json_path(project_dir)
        try:
            if not p or (not os.path.isfile(p)):
                return {}
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _write_upscale_json(self, project_dir: str, obj: dict) -> None:
        p = self._upscale_json_path(project_dir)
        try:
            if not project_dir:
                return
            os.makedirs(str(project_dir), exist_ok=True)
        except Exception:
            pass
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj if isinstance(obj, dict) else {}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _planner_detect_engines(self) -> List[Tuple[str, str]]:
        """Reuse helpers/upsc.py engine detection. Returns [(label, exe)]."""
        try:
            from helpers import upsc as _upsc  # type: ignore
        except Exception:
            try:
                import upsc as _upsc  # type: ignore
            except Exception:
                _upsc = None  # type: ignore

        if _upsc is None:
            return []

        try:
            eng = _upsc.detect_engines()  # type: ignore[attr-defined]
            if isinstance(eng, list):
                out: List[Tuple[str, str]] = []
                for it in eng:
                    try:
                        lab, exe = it
                        out.append((str(lab), str(exe)))
                    except Exception:
                        pass
                return out
        except Exception:
            pass
        return []

    def _planner_models_for_engine(self, engine_label: str) -> Tuple[List[str], Optional[dict]]:
        """Reuse helpers/upsc.py model listing where available.

        Returns (model_text_list, optional_model_data_template).
        """
        lab = (engine_label or "").strip().lower()

        try:
            from helpers import upsc as _upsc  # type: ignore
        except Exception:
            try:
                import upsc as _upsc  # type: ignore
            except Exception:
                _upsc = None  # type: ignore

        if _upsc is None:
            return (["(default)"], None)

        try:
            if "ultrasharp" in lab and hasattr(_upsc, "scan_ultrasharp_models"):
                return (list(_upsc.scan_ultrasharp_models()), None)  # type: ignore
            if "srmd (ncnn via realesrgan)" in lab and hasattr(_upsc, "scan_srmd_realesrgan_models"):
                return (list(_upsc.scan_srmd_realesrgan_models()), None)  # type: ignore
            if lab.startswith("srmd (ncnn)") and hasattr(_upsc, "scan_srmd_models"):
                return (list(_upsc.scan_srmd_models()), None)  # type: ignore
            if "waifu2x" in lab and hasattr(_upsc, "scan_waifu2x_models"):
                return (list(_upsc.scan_waifu2x_models()), None)  # type: ignore
            if "realsr (ncnn)" in lab and hasattr(_upsc, "scan_realsr_ncnn_models"):
                return (list(_upsc.scan_realsr_ncnn_models()), None)  # type: ignore
            if lab.startswith("real-esrgan") and hasattr(_upsc, "scan_realsr_models"):
                return (list(_upsc.scan_realsr_models()), None)  # type: ignore
        except Exception:
            pass

        # Engines without model catalogs in helpers/upsc.py (SwinIR/LapSRN/Upscayl/GFPGAN) are represented as default-only.
        return (["(default)"], None)

    def _build_upscaling_group(self) -> QGroupBox:
        box = QGroupBox("Upscaling")
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        # OFF view: toggle + info
        row = QHBoxLayout()
        row.setSpacing(8)
        self.chk_planner_upscale = QCheckBox("Enable upscaling (Planner)")
        self.lbl_planner_upscale_info = QLabel("Off — no upscaling will be applied.")
        try:
            self.lbl_planner_upscale_info.setStyleSheet("opacity: 0.8;")
        except Exception:
            pass

        row.addWidget(self.chk_planner_upscale)
        row.addWidget(self.lbl_planner_upscale_info, 1)
        lay.addLayout(row)

        # ON view: select exactly one engine (3 choices)
        self._upscale_controls = QWidget()
        v = QVBoxLayout(self._upscale_controls)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        # Engine radio row
        eng_row = QHBoxLayout()
        eng_row.setSpacing(10)
        eng_row.addWidget(QLabel("Engine"))

        self._upscale_engine_group = QButtonGroup(self)
        self._upscale_engine_group.setExclusive(True)

        self.rb_upscale_seedvr2 = QRadioButton("SeedVR2 (HQ / slow)")
        self.rb_upscale_realesrgan4x = QRadioButton("RealESRGAN 4x (normal speed)")
        self.rb_upscale_anime = QRadioButton("Anime (Fast, anime only)")

        self._upscale_engine_group.addButton(self.rb_upscale_seedvr2)
        self._upscale_engine_group.addButton(self.rb_upscale_realesrgan4x)
        self._upscale_engine_group.addButton(self.rb_upscale_anime)

        # Default selection
        self.rb_upscale_seedvr2.setChecked(True)

        eng_row.addWidget(self.rb_upscale_seedvr2)
        eng_row.addWidget(self.rb_upscale_realesrgan4x)
        eng_row.addWidget(self.rb_upscale_anime)
        eng_row.addStretch(1)
        v.addLayout(eng_row)

        # Stacked per-engine settings (only selected is visible)
        self._upscale_stack = QStackedWidget()
        v.addWidget(self._upscale_stack)

        # --- SeedVR2 settings ---
        seed_page = QWidget()
        sg = QGridLayout(seed_page)
        sg.setContentsMargins(0, 0, 0, 0)
        sg.setHorizontalSpacing(10)
        sg.setVerticalSpacing(6)

        sg.addWidget(QLabel("Resolution"), 0, 0)
        self.cmb_seedvr2_resolution = QComboBox()
        self.cmb_seedvr2_resolution.addItem("1080p (8–12 GB VRAM typical)", 1080)
        self.cmb_seedvr2_resolution.addItem("1440p (10–16 GB VRAM typical)", 1440)
        # default 1080p (1920 width)
        try:
            self.cmb_seedvr2_resolution.setCurrentIndex(0)
        except Exception:
            pass
        sg.addWidget(self.cmb_seedvr2_resolution, 0, 1)

        # Rough estimate hint (SeedVR2 is very slow; this is a static heuristic).
        self.lbl_seedvr2_estimate = QLabel("")
        self.lbl_seedvr2_estimate.setWordWrap(True)
        self.lbl_seedvr2_estimate.setStyleSheet("opacity: 0.75;")
        sg.addWidget(self.lbl_seedvr2_estimate, 1, 0, 1, 2)

        self.chk_seedvr2_temporal = QCheckBox("Temporal consistency (recommended for video)")
        self.chk_seedvr2_temporal.setToolTip("Improves frame-to-frame consistency (less shimmer) but uses more memory and is slower.")
        self.chk_seedvr2_temporal.setChecked(True)
        sg.addWidget(self.chk_seedvr2_temporal, 2, 0, 1, 2)

        # Advanced toggle (collapsed by default)
        self.btn_seedvr2_adv = QToolButton()
        self.btn_seedvr2_adv.setText("Advanced")
        self.btn_seedvr2_adv.setCheckable(True)
        self.btn_seedvr2_adv.setChecked(False)
        self.btn_seedvr2_adv.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_seedvr2_adv.setArrowType(Qt.RightArrow)
        sg.addWidget(self.btn_seedvr2_adv, 3, 0, 1, 2)

        self.seedvr2_adv_widget = QWidget()
        adv = QGridLayout(self.seedvr2_adv_widget)
        adv.setContentsMargins(16, 0, 0, 0)
        adv.setHorizontalSpacing(10)
        adv.setVerticalSpacing(6)

        adv.addWidget(QLabel("Batch size"), 0, 0)
        self.spin_seedvr2_batch = QSpinBox()
        self.spin_seedvr2_batch.setRange(1, 8)
        self.spin_seedvr2_batch.setValue(1)
        adv.addWidget(self.spin_seedvr2_batch, 0, 1)

        adv.addWidget(QLabel("Chunk size (frames)"), 1, 0)
        self.spin_seedvr2_chunk = QSpinBox()
        self.spin_seedvr2_chunk.setRange(1, 200)
        self.spin_seedvr2_chunk.setValue(20)
        adv.addWidget(self.spin_seedvr2_chunk, 1, 1)

        adv.addWidget(QLabel("Color correction"), 2, 0)
        self.cmb_seedvr2_color = QComboBox()
        for cc in ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"]:
            self.cmb_seedvr2_color.addItem(cc)
        self.cmb_seedvr2_color.setCurrentText("lab")
        adv.addWidget(self.cmb_seedvr2_color, 2, 1)

        adv.addWidget(QLabel("Attention"), 3, 0)
        self.cmb_seedvr2_attention = QComboBox()
        for am in ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"]:
            self.cmb_seedvr2_attention.addItem(am)
        self.cmb_seedvr2_attention.setCurrentText("sdpa")
        adv.addWidget(self.cmb_seedvr2_attention, 3, 1)

        adv.addWidget(QLabel("DiT model"), 4, 0)
        self.cmb_seedvr2_dit = QComboBox()
        # Placeholder for later auto-selection / catalogs.
        self.cmb_seedvr2_dit.addItem("seedvr2_ema_3b-Q4_K_M.gguf")
        adv.addWidget(self.cmb_seedvr2_dit, 4, 1)

        sg.addWidget(self.seedvr2_adv_widget, 4, 0, 1, 2)
        self.seedvr2_adv_widget.setVisible(False)

        self._upscale_stack.addWidget(seed_page)

        # --- RealESRGAN 4x (wired) ---
        re_page = QWidget()
        rl = QVBoxLayout(re_page)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        title = QLabel("RealESRGAN 4x (realesrgan-x4plus)")
        title.setStyleSheet("font-weight: 600;")
        title.setWordWrap(True)

        desc = QLabel("General-purpose 4× upscaler for real footage.\nModel: models/realesrgan/realesrgan-x4plus (.bin/.param)")
        desc.setWordWrap(True)
        desc.setStyleSheet("opacity: 0.8;")

        speed = QLabel("Speed: about 2–4 frames / second (depends on GPU and input resolution).")
        speed.setWordWrap(True)
        speed.setStyleSheet("opacity: 0.85;")

        rl.addWidget(title)
        rl.addWidget(desc)
        rl.addWidget(speed)
        rl.addStretch(1)
        self._upscale_stack.addWidget(re_page)

        # --- Anime upscaler (wired) ---
        an_page = QWidget()
        al = QVBoxLayout(an_page)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(6)

        title = QLabel("Anime upscaler (realesrgan-x4plus-anime)")
        title.setStyleSheet("font-weight: 600;")
        title.setWordWrap(True)

        desc = QLabel("Fast upscaler for anime / illustrated footage.\nModel: models/realesrgan/realesrgan-x4plus-anime (.bin/.param)\nWarning: using this on real footage will create anime-like results.")
        desc.setWordWrap(True)
        desc.setStyleSheet("opacity: 0.8;")

        speed = QLabel("Speed: about 10 frames / second (depends on GPU and input resolution).")
        speed.setWordWrap(True)
        speed.setStyleSheet("opacity: 0.85;")

        al.addWidget(title)
        al.addWidget(desc)
        al.addWidget(speed)
        al.addStretch(1)
        self._upscale_stack.addWidget(an_page)


        lay.addWidget(self._upscale_controls)

        # default collapsed
        self._set_upscale_controls_visible(False)

        # Wiring
        try:
            self.chk_planner_upscale.toggled.connect(self._on_planner_upscale_toggled)
        except Exception:
            pass

        def _apply_engine_stack() -> None:
            try:
                if self.rb_upscale_seedvr2.isChecked():
                    self._upscale_stack.setCurrentIndex(0)
                elif self.rb_upscale_realesrgan4x.isChecked():
                    self._upscale_stack.setCurrentIndex(1)
                else:
                    self._upscale_stack.setCurrentIndex(2)
            except Exception:
                pass

            # Update SeedVR2 estimate label when relevant
            try:
                _update_seedvr2_estimate()
            except Exception:
                pass
            try:
                proj = str(getattr(self, "_active_project_dir", "") or "")
                if proj:
                    self._persist_upscale_settings_for_project(proj, force_write=True)
            except Exception:
                pass
            try:
                self._sync_settings_summary()
            except Exception:
                pass

        try:
            self.rb_upscale_seedvr2.toggled.connect(lambda _on: _apply_engine_stack())
            self.rb_upscale_realesrgan4x.toggled.connect(lambda _on: _apply_engine_stack())
            self.rb_upscale_anime.toggled.connect(lambda _on: _apply_engine_stack())
        except Exception:
            pass

        def _toggle_adv(on: bool) -> None:
            try:
                self.seedvr2_adv_widget.setVisible(bool(on))
                self.btn_seedvr2_adv.setArrowType(Qt.DownArrow if bool(on) else Qt.RightArrow)
            except Exception:
                pass

        try:
            self.btn_seedvr2_adv.toggled.connect(_toggle_adv)
        except Exception:
            pass

        # Persist on any SeedVR2 setting change
        def _persist_seedvr2() -> None:
            try:
                proj = str(getattr(self, "_active_project_dir", "") or "")
                if proj:
                    self._persist_upscale_settings_for_project(proj, force_write=True)
            except Exception:
                pass
            try:
                _update_seedvr2_estimate()
            except Exception:
                pass
            try:
                self._sync_settings_summary()
            except Exception:
                pass

        def _update_seedvr2_estimate() -> None:
            """Static heuristic estimate shown when SeedVR2 is selected.

            SeedVR2 runtime varies wildly with GPU, model, clip length and settings.
            This label is just a reality-check to avoid surprise multi-hour runs.
            """
            try:
                if not hasattr(self, "lbl_seedvr2_estimate"):
                    return
                if not (hasattr(self, "rb_upscale_seedvr2") and self.rb_upscale_seedvr2.isChecked()):
                    # The SeedVR2 page is hidden when another engine is selected.
                    return
            except Exception:
                return

            try:
                res = int(self.cmb_seedvr2_resolution.currentData() or 1080)
            except Exception:
                res = 1080

            # Based on user-reported speeds:
            # - 1080p: ~37 min for 15 sec (~2.5 min per 1 sec video)
            # - 1440p: ~30 min for 15 sec (~2.0 min per 1 sec video)
            per_sec = 120.0 if res >= 1440 else 150.0
            min_per_sec = per_sec / 60.0
            min_per_min = per_sec * 60.0 / 60.0  # equals per_sec
            hrs_per_min = min_per_min / 60.0

            txt = (
                f"Estimate (very rough): ~{min_per_sec:.1f} min render per 1s video "
                f"(~{int(per_sec)}× realtime). 1 min ≈ {int(min_per_min)} min (~{hrs_per_min:.1f} h). "
                f"Varies by GPU/settings." 
            )
            try:
                self.lbl_seedvr2_estimate.setText(txt)
            except Exception:
                pass

        try:
            self.cmb_seedvr2_resolution.currentIndexChanged.connect(lambda _i: _persist_seedvr2())
            self.chk_seedvr2_temporal.toggled.connect(lambda _on: _persist_seedvr2())
            self.spin_seedvr2_batch.valueChanged.connect(lambda _v: _persist_seedvr2())
            self.spin_seedvr2_chunk.valueChanged.connect(lambda _v: _persist_seedvr2())
            self.cmb_seedvr2_color.currentIndexChanged.connect(lambda _i: _persist_seedvr2())
            self.cmb_seedvr2_attention.currentIndexChanged.connect(lambda _i: _persist_seedvr2())
            self.cmb_seedvr2_dit.currentIndexChanged.connect(lambda _i: _persist_seedvr2())
        except Exception:
            pass

        # Apply initial stack
        _apply_engine_stack()

        return box

    def _build_interpolation_group(self) -> QGroupBox:
        box = QGroupBox("Interpolation")
        lay = QVBoxLayout(box)
        lay.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.chk_planner_interp60 = QCheckBox("Make video playback more smooth (Interpolate to 60fps)")
        self.lbl_planner_interp60_hint = QLabel("Interpolation done with Rife.")
        try:
            self.lbl_planner_interp60_hint.setStyleSheet("opacity: 0.8;")
        except Exception:
            pass

        row.addWidget(self.chk_planner_interp60)
        row.addWidget(self.lbl_planner_interp60_hint, 1)
        lay.addLayout(row)

        try:
            self.chk_planner_interp60.toggled.connect(self._on_planner_interp60_toggled)
        except Exception:
            pass

        return box

    def _set_upscale_controls_visible(self, on: bool) -> None:
        try:
            self._upscale_controls.setVisible(bool(on))
        except Exception:
            pass
        try:
            if hasattr(self, "lbl_planner_upscale_info"):
                if bool(on):
                    self.lbl_planner_upscale_info.setText("On — settings are saved per project. ")
                else:
                    self.lbl_planner_upscale_info.setText("Off — no upscaling will be applied. ")
        except Exception:
            pass

    def _refresh_upscale_engines(self) -> None:
        try:
            engines = self._planner_detect_engines()
        except Exception:
            engines = []
        self._planner_upscale_engines = engines

        try:
            self.cmb_planner_upscale_engine.blockSignals(True)
            self.cmb_planner_upscale_engine.clear()
            for lab, exe in engines:
                self.cmb_planner_upscale_engine.addItem(str(lab), str(exe))
            self.cmb_planner_upscale_engine.blockSignals(False)
        except Exception:
            try:
                self.cmb_planner_upscale_engine.blockSignals(False)
            except Exception:
                pass

        # Populate models for current engine (or default)
        try:
            self._refresh_upscale_models_for_current_engine()
        except Exception:
            pass

    def _refresh_upscale_models_for_current_engine(self) -> None:
        try:
            lab = str(self.cmb_planner_upscale_engine.currentText() or "")
        except Exception:
            lab = ""
        if not lab:
            try:
                self.cmb_planner_upscale_model.clear()
                self.cmb_planner_upscale_model.addItem("(default)")
            except Exception:
                pass
            return

        if lab in self._planner_upscale_models_cache:
            models = self._planner_upscale_models_cache.get(lab) or ["(default)"]
        else:
            models, _md = self._planner_models_for_engine(lab)
            if not models:
                models = ["(default)"]
            self._planner_upscale_models_cache[lab] = list(models)

        cur = ""
        try:
            cur = str(self.cmb_planner_upscale_model.currentText() or "")
        except Exception:
            cur = ""

        try:
            self.cmb_planner_upscale_model.blockSignals(True)
            self.cmb_planner_upscale_model.clear()
            for m in models:
                self.cmb_planner_upscale_model.addItem(str(m))
            # keep previous if possible
            if cur and cur in models:
                self.cmb_planner_upscale_model.setCurrentText(cur)
            self.cmb_planner_upscale_model.blockSignals(False)
        except Exception:
            try:
                self.cmb_planner_upscale_model.blockSignals(False)
            except Exception:
                pass

    def _current_upscale_payload(self) -> dict:
        enabled = bool(getattr(self, "chk_planner_upscale", None) and self.chk_planner_upscale.isChecked())

        # Planner supports exactly 3 choices (Step 1): SeedVR2, RealESRGAN 4x, Anime.
        engine_key = "seedvr2"
        try:
            if hasattr(self, "rb_upscale_realesrgan4x") and self.rb_upscale_realesrgan4x.isChecked():
                engine_key = "realesrgan4x"
            elif hasattr(self, "rb_upscale_anime") and self.rb_upscale_anime.isChecked():
                engine_key = "anime"
            else:
                engine_key = "seedvr2"
        except Exception:
            engine_key = "seedvr2"

        # Backward-compatible labels (pipeline still reads engine_label/model_text today).
        if engine_key == "seedvr2":
            engine_label = "SeedVR2"
        elif engine_key == "realesrgan4x":
            engine_label = "RealESRGAN 4x"
        else:
            engine_label = "Anime"

        payload: Dict[str, Any] = {
            "enabled": bool(enabled),
            "engine_key": str(engine_key),
            "engine_label": str(engine_label),
            # engine_exe is intentionally blank in the simplified 3-choice UI (kept for compatibility).
            "engine_exe": "",
            "model_text": "",
        }

        # SeedVR2 detailed settings (only meaningful when selected).
        if engine_key == "seedvr2":
            try:
                payload["seedvr2_resolution"] = int(self.cmb_seedvr2_resolution.currentData() or 1080)
            except Exception:
                payload["seedvr2_resolution"] = 1080
            try:
                payload["seedvr2_temporal_overlap"] = 1 if bool(self.chk_seedvr2_temporal.isChecked()) else 0
            except Exception:
                payload["seedvr2_temporal_overlap"] = 1
            try:
                payload["seedvr2_batch_size"] = int(self.spin_seedvr2_batch.value())
            except Exception:
                payload["seedvr2_batch_size"] = 1
            try:
                payload["seedvr2_chunk_size"] = int(self.spin_seedvr2_chunk.value())
            except Exception:
                payload["seedvr2_chunk_size"] = 20
            try:
                payload["seedvr2_color_correction"] = str(self.cmb_seedvr2_color.currentText() or "lab")
            except Exception:
                payload["seedvr2_color_correction"] = "lab"
            try:
                payload["seedvr2_attention_mode"] = str(self.cmb_seedvr2_attention.currentText() or "sdpa")
            except Exception:
                payload["seedvr2_attention_mode"] = "sdpa"
            try:
                payload["seedvr2_dit_model"] = str(self.cmb_seedvr2_dit.currentText() or "seedvr2_ema_3b-Q4_K_M.gguf")
            except Exception:
                payload["seedvr2_dit_model"] = "seedvr2_ema_3b-Q4_K_M.gguf"

            # Keep model_text aligned with what we actually run later.
            payload["model_text"] = str(payload.get("seedvr2_dit_model") or "seedvr2_ema_3b-Q4_K_M.gguf")
        # Anime upscaler (Real-ESRGAN anime model; fixed model name)
        if engine_key == "anime":
            payload["model_text"] = "realesrgan-x4plus-anime"
            payload["scale"] = 4

        # RealESRGAN 4x placeholder (kept for later wiring)
        if engine_key == "realesrgan4x":
            payload["model_text"] = "realesrgan-x4plus"
            payload["scale"] = 4


        return payload

    def _persist_upscale_settings_for_project(self, project_dir: str, force_write: bool = False) -> None:
        """Write planner_upscale.json into the given project folder.

        This JSON acts as the Planner post-settings store (Chunk 9B1+):
          - Upscaling settings (engine/model + enabled)
          - Interpolation post-step toggle (Chunk 9C)

        Behavior:
          - If BOTH toggles are OFF: do nothing unless the file already exists or force_write=True.
          - If ANY toggle is ON: ensure the file exists and store the current selection/state.
        """
        project_dir = str(project_dir or "")
        if not project_dir:
            return

        # Gather UI state
        payload_up = self._current_upscale_payload()
        enabled_up = bool(payload_up.get("enabled"))

        interp_on = False
        try:
            interp_on = bool(getattr(self, "chk_planner_interp60", None) and self.chk_planner_interp60.isChecked())
        except Exception:
            interp_on = False

        p = self._upscale_json_path(project_dir)
        exists = bool(p and os.path.isfile(p))

        if (not enabled_up) and (not interp_on) and (not force_write) and (not exists):
            return

        # Merge with existing so toggling one feature doesn't wipe the other.
        cur = {}
        try:
            cur = self._read_upscale_json(project_dir) or {}
        except Exception:
            cur = {}
        if not isinstance(cur, dict):
            cur = {}

        # Always persist both toggles.
        cur["enabled"] = bool(enabled_up)
        cur["interpolate_60fps_fast"] = bool(interp_on)

        # Only persist engine/model/settings when upscaling is enabled (preserve prior selections otherwise).
        if enabled_up:
            try:
                # Keep the old keys so existing pipeline code stays compatible.
                cur["engine_key"] = str(payload_up.get("engine_key") or "seedvr2")
                cur["engine_label"] = str(payload_up.get("engine_label") or "SeedVR2")
                cur["engine_exe"] = str(payload_up.get("engine_exe") or "")
                cur["model_text"] = str(payload_up.get("model_text") or "")

                # SeedVR2 detailed settings (safe to store even if not used yet).
                for k in [
                    "seedvr2_resolution",
                    "seedvr2_temporal_overlap",
                    "seedvr2_batch_size",
                    "seedvr2_chunk_size",
                    "seedvr2_color_correction",
                    "seedvr2_attention_mode",
                    "seedvr2_dit_model",
                ]:
                    if k in payload_up:
                        cur[k] = payload_up.get(k)
            except Exception:
                pass

        self._write_upscale_json(project_dir, cur)

    def _load_upscale_settings_from_project(self, project_dir: str) -> None:
        project_dir = str(project_dir or "")
        if not project_dir:
            return
        obj = self._read_upscale_json(project_dir)
        if not isinstance(obj, dict) or not obj:
            # If no file, keep UI as-is but collapse controls
            try:
                if hasattr(self, "chk_planner_upscale"):
                    self.chk_planner_upscale.blockSignals(True)
                    self.chk_planner_upscale.setChecked(False)
                    self.chk_planner_upscale.blockSignals(False)
            except Exception:
                pass
            try:
                if hasattr(self, "chk_planner_interp60"):
                    self.chk_planner_interp60.blockSignals(True)
                    self.chk_planner_interp60.setChecked(False)
                    self.chk_planner_interp60.blockSignals(False)
            except Exception:
                pass
            try:
                self._set_upscale_controls_visible(False)
            except Exception:
                pass
            return

        enabled = bool(obj.get("enabled", False))
        interp_on = bool(obj.get("interpolate_60fps_fast", False))

        # Update interpolation toggle (independent of upscaling)
        try:
            if hasattr(self, "chk_planner_interp60"):
                self.chk_planner_interp60.blockSignals(True)
                self.chk_planner_interp60.setChecked(bool(interp_on))
                self.chk_planner_interp60.blockSignals(False)
        except Exception:
            pass

        # Update upscaling toggle + controls
        try:
            if hasattr(self, "chk_planner_upscale"):
                self.chk_planner_upscale.blockSignals(True)
                self.chk_planner_upscale.setChecked(bool(enabled))
                self.chk_planner_upscale.blockSignals(False)
        except Exception:
            pass

        self._set_upscale_controls_visible(bool(enabled))

        if not enabled:
            return

        # Restore selected engine (3-choice UI)
        try:
            engine_key = str(obj.get("engine_key") or "")
        except Exception:
            engine_key = ""
        if not engine_key:
            # Backward-compat: infer from old engine_label
            try:
                lab = str(obj.get("engine_label") or "")
            except Exception:
                lab = ""
            lab_l = (lab or "").strip().lower()
            if "anime" in lab_l:
                engine_key = "anime"
            elif "realesr" in lab_l:
                engine_key = "realesrgan4x"
            else:
                engine_key = "seedvr2"

        try:
            # Set radio buttons without triggering extra work
            if engine_key == "realesrgan4x":
                self.rb_upscale_realesrgan4x.setChecked(True)
            elif engine_key == "anime":
                self.rb_upscale_anime.setChecked(True)
            else:
                self.rb_upscale_seedvr2.setChecked(True)
        except Exception:
            pass

        # Apply stack visibility
        try:
            if engine_key == "realesrgan4x":
                self._upscale_stack.setCurrentIndex(1)
            elif engine_key == "anime":
                self._upscale_stack.setCurrentIndex(2)
            else:
                self._upscale_stack.setCurrentIndex(0)
        except Exception:
            pass

        # Restore SeedVR2 fields (if present)
        if engine_key == "seedvr2":
            try:
                want_res = int(obj.get("seedvr2_resolution", 1080))
            except Exception:
                want_res = 1080
            try:
                for i in range(self.cmb_seedvr2_resolution.count()):
                    if int(self.cmb_seedvr2_resolution.itemData(i) or 0) == int(want_res):
                        self.cmb_seedvr2_resolution.setCurrentIndex(i)
                        break
            except Exception:
                pass

            try:
                self.chk_seedvr2_temporal.setChecked(bool(int(obj.get("seedvr2_temporal_overlap", 1)) != 0))
            except Exception:
                pass

            try:
                self.spin_seedvr2_batch.setValue(int(obj.get("seedvr2_batch_size", 1)))
            except Exception:
                pass
            try:
                self.spin_seedvr2_chunk.setValue(int(obj.get("seedvr2_chunk_size", 20)))
            except Exception:
                pass
            try:
                self.cmb_seedvr2_color.setCurrentText(str(obj.get("seedvr2_color_correction", "lab")))
            except Exception:
                pass
            try:
                self.cmb_seedvr2_attention.setCurrentText(str(obj.get("seedvr2_attention_mode", "sdpa")))
            except Exception:
                pass
            try:
                self.cmb_seedvr2_dit.setCurrentText(str(obj.get("seedvr2_dit_model", "seedvr2_ema_3b-Q4_K_M.gguf")))
            except Exception:
                pass

    @Slot(bool)
    def _on_planner_upscale_toggled(self, on: bool) -> None:
        self._set_upscale_controls_visible(bool(on))
        try:
            proj = str(getattr(self, "_active_project_dir", "") or "")
            if proj:
                self._persist_upscale_settings_for_project(proj, force_write=True)
        except Exception:
            pass
        try:
            self._sync_settings_summary()
        except Exception:
            pass


    @Slot(bool)
    def _on_planner_interp60_toggled(self, on: bool) -> None:
        try:
            proj = str(getattr(self, "_active_project_dir", "") or "")
            if proj:
                self._persist_upscale_settings_for_project(proj, force_write=True)
        except Exception:
            pass


    @Slot(int)
    def _on_planner_upscale_engine_changed(self, _idx: int = 0) -> None:
        try:
            self._refresh_upscale_models_for_current_engine()
        except Exception:
            pass
        try:
            proj = str(getattr(self, "_active_project_dir", "") or "")
            if proj and bool(getattr(self, "chk_planner_upscale", None) and self.chk_planner_upscale.isChecked()):
                self._persist_upscale_settings_for_project(proj, force_write=True)
        except Exception:
            pass
        try:
            self._sync_settings_summary()
        except Exception:
            pass


    @Slot(int)
    def _on_planner_upscale_model_changed(self, _idx: int = 0) -> None:
        try:
            proj = str(getattr(self, "_active_project_dir", "") or "")
            if proj and bool(getattr(self, "chk_planner_upscale", None) and self.chk_planner_upscale.isChecked()):
                self._persist_upscale_settings_for_project(proj, force_write=True)
        except Exception:
            pass
        try:
            self._sync_settings_summary()
        except Exception:
            pass


    def _build_log_group(self) -> QGroupBox:
        box = QGroupBox("Logs")
        lay = QVBoxLayout(box)
        lay.setSpacing(8)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Pipeline logs will appear here…")
        self.log.setMinimumHeight(260)

        lay.addWidget(self.log)

        # Log actions
        row = QHBoxLayout()
        row.setSpacing(8)

        self.btn_clear_log = QPushButton("Clear log")
        self.btn_copy_log = QPushButton("Copy log")
        self.btn_clear_log.clicked.connect(self.log.clear)
        self.btn_copy_log.clicked.connect(self._copy_log)

        row.addWidget(self.btn_clear_log)
        row.addWidget(self.btn_copy_log)
        row.addStretch(1)

        lay.addLayout(row)
        return box

    
    # -------------------------
    # Results / History
    # -------------------------

    @Slot(int)
    def _on_tabs_changed(self, idx: int) -> None:
        try:
            if not hasattr(self, "tabs"):
                return
            if self.tabs.tabText(idx).lower().startswith("results"):
                if not bool(getattr(self, "_results_loaded", False)):
                    self._refresh_results()
                    self._results_loaded = True
        except Exception:
            pass

    def _get_output_base(self) -> str:
        try:
            output_base = (self.out_dir_edit.text() or "").strip() or self._default_out_dir()
        except Exception:
            output_base = self._default_out_dir()
        return _abspath_from_root(output_base)

    def _build_results_tab(self) -> QWidget:
        w = QWidget()
        root = QVBoxLayout(w)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Top row: output base + refresh
        top = QHBoxLayout()
        top.setSpacing(10)
        top.addWidget(QLabel("Output base:"))
        self.results_out_base = QLineEdit()
        self.results_out_base.setReadOnly(True)
        self.results_out_base.setText(self._get_output_base())
        top.addWidget(self.results_out_base, 1)

        self.btn_results_refresh = QPushButton("Refresh")
        self.btn_results_refresh.clicked.connect(self._refresh_results)
        top.addWidget(self.btn_results_refresh)

        root.addLayout(top)

        # Table
        self.results_view = QTableView()
        self.results_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_view.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_view.setAlternatingRowColors(True)

        self.results_model = QStandardItemModel(0, 4, self.results_view)
        self.results_model.setHorizontalHeaderLabels(["Title", "Assets", "Status", "Updated"])
        self.results_view.setModel(self.results_model)

        # Dots delegate in the assets column
        try:
            self.results_view.setItemDelegateForColumn(1, _AssetsDotsDelegate(self.results_view))
        except Exception:
            pass

        hdr = self.results_view.horizontalHeader()
        try:
            hdr.setStretchLastSection(False)
            hdr.setSectionResizeMode(0, QHeaderView.Stretch)     # title
            hdr.setSectionResizeMode(1, QHeaderView.Fixed)  # dots
            hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # status
            hdr.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # updated
            try:
                # Space for up to 6 asset dots (sound/images/clips/final + upscaled + interpolated)
                self.results_view.setColumnWidth(1, 122)
            except Exception:
                pass
        except Exception:
            pass

        try:
            self.results_view.verticalHeader().setVisible(False)
        except Exception:
            pass

        self.results_view.selectionModel().selectionChanged.connect(self._on_results_selection_changed)

        root.addWidget(self.results_view, 1)

        # Actions row
        actions = QHBoxLayout()
        actions.setSpacing(10)

        self.btn_res_open_folder = QPushButton("Open Folder")
        self.btn_res_open_manifest = QPushButton("Open Manifest")
        self.btn_res_open_errors = QPushButton("Open Errors")
        self.btn_res_resume = QPushButton("Resume")
        self.btn_res_remove = QPushButton("Remove Folder")
        self.btn_res_play_final = QPushButton("Play final result")

        self.btn_res_open_folder.clicked.connect(self._res_open_folder)
        self.btn_res_open_manifest.clicked.connect(self._res_open_manifest)
        self.btn_res_open_errors.clicked.connect(self._res_open_errors)
        self.btn_res_resume.clicked.connect(self._res_resume)
        self.btn_res_remove.clicked.connect(self._res_remove_folder)
        self.btn_res_play_final.clicked.connect(self._res_play_final)

        # Only show when a job with a final video is selected
        try:
            self.btn_res_play_final.setVisible(False)
        except Exception:
            pass

        actions.addWidget(self.btn_res_open_folder)
        actions.addWidget(self.btn_res_open_manifest)
        actions.addWidget(self.btn_res_open_errors)
        actions.addWidget(self.btn_res_resume)
        actions.addWidget(self.btn_res_remove)
        actions.addWidget(self.btn_res_play_final)
        actions.addStretch(1)

        root.addLayout(actions)

        self.lbl_results_info = QLabel("Select one to open or resume.")
        root.addWidget(self.lbl_results_info)

        self._set_results_actions_enabled(False)
        return w

    def _set_results_actions_enabled(self, enabled: bool) -> None:
        for b in (getattr(self, "btn_res_open_folder", None),
                  getattr(self, "btn_res_open_manifest", None),
                  getattr(self, "btn_res_open_errors", None),
                  getattr(self, "btn_res_resume", None),
                  getattr(self, "btn_res_remove", None),
                  getattr(self, "btn_res_play_final", None)):
            try:
                if b is not None:
                    b.setEnabled(bool(enabled))
            except Exception:
                pass

    @Slot()
    def _on_results_selection_changed(self) -> None:
        try:
            idxs = self.results_view.selectionModel().selectedRows()
            self._set_results_actions_enabled(bool(idxs))
        except Exception:
            self._set_results_actions_enabled(False)

        # Play button is only shown when this job has a final video in /final
        try:
            self._update_play_final_visibility()
        except Exception:
            pass

        # Chunk 9B1: load per-project upscaling settings when selecting a job
        try:
            p = self._get_selected_result_paths()
            job_dir = str(p.get("job_dir") or "")
            if job_dir:
                self._active_project_dir = job_dir
                self._load_upscale_settings_from_project(job_dir)
        except Exception:
            pass

    def _get_selected_result_paths(self) -> dict:
        out = {"job_dir": "", "manifest": "", "errors_dir": "", "job_json": "", "final_video": ""}
        try:
            idxs = self.results_view.selectionModel().selectedRows()
            if not idxs:
                return out
            row = idxs[0].row()
            job_dir = self.results_model.item(row, 0).data(Qt.UserRole) or ""
            job_dir = str(job_dir)
            if not job_dir:
                return out
            out["job_dir"] = job_dir
            out["manifest"] = os.path.join(job_dir, "manifest.json")
            try:
                fv = self.results_model.item(row, 0).data(int(Qt.UserRole) + 1) or ""
                out["final_video"] = str(fv or "")
            except Exception:
                out["final_video"] = ""
            out["errors_dir"] = os.path.join(job_dir, "errors")
            out["job_json"] = os.path.join(job_dir, "job.json")
        except Exception:
            pass
        return out

    def _open_path_native(self, p: str) -> None:
        p = str(p or "")
        if not p:
            return
        try:
            if os.name == "nt":
                os.startfile(p)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as e:
            QMessageBox.warning(self, "Open failed", str(e))


    def _get_selected_final_video_path(self) -> str:
        try:
            return str(self._get_selected_result_paths().get("final_video") or "")
        except Exception:
            return ""

    def _update_play_final_visibility(self) -> None:
        btn = getattr(self, "btn_res_play_final", None)
        if btn is None:
            return
        fv = self._get_selected_final_video_path()
        ok = bool(fv) and os.path.exists(fv)
        try:
            btn.setVisible(bool(ok))
        except Exception:
            pass
        try:
            btn.setEnabled(bool(ok))
        except Exception:
            pass

    @Slot()
    def _res_open_folder(self) -> None:
        p = self._get_selected_result_paths().get("job_dir", "")
        if p:
            self._open_path_native(p)

    @Slot()
    def _res_open_manifest(self) -> None:
        p = self._get_selected_result_paths().get("manifest", "")
        if p and os.path.exists(p):
            self._open_path_native(p)
        else:
            QMessageBox.information(self, "Not found", "manifest.json not found for this job.")

    @Slot()
    def _res_open_errors(self) -> None:
        p = self._get_selected_result_paths().get("errors_dir", "")
        if p and os.path.isdir(p):
            self._open_path_native(p)
        else:
            QMessageBox.information(self, "No errors", "No errors folder exists for this job.")

    @Slot()
    def _res_resume(self) -> None:
        paths = self._get_selected_result_paths()
        job_dir = paths.get("job_dir", "")
        job_json = paths.get("job_json", "")
        if not job_dir:
            return
        if not os.path.exists(job_json):
            QMessageBox.warning(self, "Cannot resume", "job.json not found in this job folder.")
            return

        try:
            data = _safe_read_json(job_json) or {}
            # Keep only PlannerJob fields
            allowed = set(getattr(PlannerJob, "__dataclass_fields__", {}).keys())
            cfg = {k: v for k, v in data.items() if k in allowed}
            job = PlannerJob(**cfg)
        except Exception as e:
            QMessageBox.warning(self, "Cannot resume", f"Failed to read job.json:\n{e}")
            return

        self._run_job(job, job_dir, resume_note=f"Resuming in-place: {os.path.basename(job_dir)}")

    
    @Slot()
    def _res_remove_folder(self) -> None:
        paths = self._get_selected_result_paths()
        job_dir = str(paths.get("job_dir") or "")
        if not job_dir:
            return

        try:
            if getattr(self, "_worker", None) is not None and self._worker.isRunning():
                QMessageBox.information(self, "Busy", "A pipeline job is currently running. Stop it before deleting folders.")
                return
        except Exception:
            pass

        name = os.path.basename(job_dir.rstrip("/\\"))
        try:
            ans = QMessageBox.question(
                self,
                "Remove folder",
                f"Delete this project folder?\n\n{name}\n\nThis will remove the entire job folder from disk.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if ans != QMessageBox.Yes:
                return
        except Exception:
            # If question dialog fails, be conservative and do nothing
            return

        try:
            shutil.rmtree(job_dir)
        except Exception as e:
            QMessageBox.warning(self, "Delete failed", str(e))
            return

        try:
            self._refresh_results()
        except Exception:
            pass

    @Slot()
    def _res_play_final(self) -> None:
        fv = self._get_selected_final_video_path()
        if not fv or (not os.path.exists(fv)):
            QMessageBox.information(self, "Not found", "No final video was found in this job's /final folder.")
            return

        # If Qt Multimedia isn't available, fallback to opening in OS player
        if not _HAVE_QT_MEDIA:
            self._open_path_native(fv)
            return

        try:
            dlg = _FinalVideoPreviewDialog(self, fv)
            dlg.exec()
        except Exception as e:
            QMessageBox.warning(self, "Preview failed", str(e))
    def _start_pipeline_with(self, job: PlannerJob, out_dir: str, resume_note: str = "") -> None:
        if self._worker and self._worker.isRunning():
            return

        try:
            _LOGGER.log_probe(f"Resume/Start pipeline: {job.job_id}")
            _LOGGER.log_job(job.job_id, "Resume/Start pipeline")
        except Exception:
            pass

                # Chunk 9B1: restore Planner-only upscaling settings from this project folder
        try:
            self._active_project_dir = str(out_dir or "")
            self._load_upscale_settings_from_project(out_dir)
        except Exception:
            pass

# Reset UI
        self.progress.setValue(0)
        self.lbl_stage.setText("Stage: Starting")
        try:
            self._header_stage = "Starting"
            self._header_pct = 0
        except Exception:
            pass
        self._set_header_status("Running: Starting", mode="running")
        self.log.clear()
        self._last_result = None
        self.btn_open_output.setEnabled(False)

        self._set_running(True)

        self._worker = PipelineWorker(job, out_dir)
        self._worker.signals.log.connect(self._append_log)
        self._worker.signals.stage.connect(self._on_worker_stage)
        self._worker.signals.progress.connect(self._on_worker_progress)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.failed.connect(self._on_failed)
        self._worker.signals.request_image_review.connect(self._on_request_image_review)
        self._worker.signals.request_video_review.connect(self._on_request_video_review)
        self._worker.signals.image_regen_started.connect(self._on_image_regen_started)
        self._worker.signals.image_regen_done.connect(self._on_image_regen_done)
        self._worker.signals.image_regen_failed.connect(self._on_image_regen_failed)
        self._worker.signals.clip_regen_started.connect(self._on_clip_regen_started)
        self._worker.signals.clip_regen_done.connect(self._on_clip_regen_done)
        self._worker.signals.clip_regen_failed.connect(self._on_clip_regen_failed)
        self._worker.signals.asset_created.connect(self._on_asset_created)

        if resume_note:
            self._append_log(resume_note)
        self._append_log("Starting pipeline…")
        self._worker.start()

    @Slot()
    def _refresh_results(self) -> None:
        try:
            base = self._get_output_base()
            self.results_out_base.setText(base)
        except Exception:
            base = self._get_output_base()

        rows = self._scan_results(base)
        self.results_model.setRowCount(0)

        for r in rows:
            title = str(r.get("title") or "")
            status = str(r.get("status") or "")
            updated = str(r.get("updated") or "")

            it_title = QStandardItem(title)
            it_assets = QStandardItem("")  # painted by delegate
            it_status = QStandardItem(status)
            it_updated = QStandardItem(updated)

            # Store job_dir on the title item for actions
            it_title.setData(str(r.get("job_dir") or ""), Qt.UserRole)

            # Store final video path (for Play button)
            it_title.setData(str(r.get("final_video") or ""), int(Qt.UserRole) + 1)

            # Dots mask + tooltip in assets column
            mask = int(r.get("assets_mask") or 0)
            it_assets.setData(mask, _AssetsDotsDelegate.ROLE_MASK)
            it_assets.setToolTip(str(r.get("assets_tooltip") or ""))

            self.results_model.appendRow([it_title, it_assets, it_status, it_updated])

        try:
            self.lbl_results_info.setText(f"Found {len(rows)} job(s). Select one to open or resume.")
        except Exception:
            pass

        self._set_results_actions_enabled(False)
        try:
            self._update_play_final_visibility()
        except Exception:
            pass

    def _scan_results(self, output_base: str) -> List[dict]:
        base = str(output_base or "").strip()
        if not base or (not os.path.isdir(base)):
            return []

        out: List[dict] = []
        try:
            subdirs = [p for p in Path(base).iterdir() if p.is_dir()]
        except Exception:
            return []

        # newest first
        subdirs.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

        for d in subdirs:
            mpath = d / "manifest.json"
            if not mpath.exists():
                continue

            manifest = _safe_read_json(str(mpath)) or {}
            title = str(manifest.get("title") or d.name)
            status = self._derive_job_status(str(d), manifest)
            updated_ts = self._derive_job_updated_ts(str(d), manifest)
            updated = updated_ts

            # counts (lightweight)
            images_dir = d / "images"
            clips_dir = d / "clips"
            audio_dir = d / "audio"
            final_dir = d / "final"

            img_n = 0
            clip_n = 0
            aud_n = 0

            try:
                if images_dir.is_dir():
                    img_n = sum(1 for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
            except Exception:
                img_n = 0

            try:
                if clips_dir.is_dir():
                    clip_n = sum(1 for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".mkv", ".webm"))
            except Exception:
                clip_n = 0

            try:
                if audio_dir.is_dir():
                    aud_n = sum(1 for p in audio_dir.iterdir() if p.is_file() and p.suffix.lower() in (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"))
            except Exception:
                aud_n = 0

            # final video: any video file inside /final (name doesn't matter)
            final_video = ""
            final_ok = False
            try:
                vext = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
                if final_dir.is_dir():
                    vids = [p for p in final_dir.iterdir() if p.is_file() and p.suffix.lower() in vext]
                    # Prefer a renamed/real output over the stable alias final_cut.mp4
                    try:
                        vids_non_stable = [p for p in vids if p.name.lower() != "final_cut.mp4"]
                        vids_stable = [p for p in vids if p.name.lower() == "final_cut.mp4"]
                    except Exception:
                        vids_non_stable = vids
                        vids_stable = []
                    vids_non_stable.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
                    vids_stable.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

                    for vp in (vids_non_stable + vids_stable):
                        try:
                            if vp.stat().st_size > 1024:
                                final_video = str(vp)
                                break
                        except Exception:
                            continue

                # fallback to manifest path if the /final folder isn't present or empty
                if not final_video:
                    fp = (manifest.get("paths") or {}).get("final_video") or ""
                    fp = str(fp or "")
                    if fp and os.path.exists(fp) and os.path.getsize(fp) > 1024:
                        final_video = fp

                final_ok = bool(final_video)
            except Exception:
                final_video = ""
                final_ok = False

            mask = 0
            if aud_n > 0:
                mask |= _AssetsDotsDelegate.HAS_SOUND
            if img_n > 0:
                mask |= _AssetsDotsDelegate.HAS_IMAGES
            if clip_n > 0:
                mask |= _AssetsDotsDelegate.HAS_CLIPS
            if final_ok:
                mask |= _AssetsDotsDelegate.HAS_FINAL

            # Optional post-steps (planner settings): upscaling + 60fps interpolation
            # Prefer manifest path pointers, fallback to step status.
            upscaled_ok = False
            interpolated_ok = False
            try:
                p = (manifest.get("paths") or {}) if isinstance(manifest.get("paths"), dict) else (manifest.get("paths") or {})
            except Exception:
                p = {}
            try:
                up_p = str((p or {}).get("final_upscaled_path") or "")
                if up_p and os.path.exists(up_p) and os.path.getsize(up_p) > 1024:
                    upscaled_ok = True
            except Exception:
                upscaled_ok = False
            try:
                ip_p = str((p or {}).get("final_interpolated_path") or "")
                if ip_p and os.path.exists(ip_p) and os.path.getsize(ip_p) > 1024:
                    interpolated_ok = True
            except Exception:
                interpolated_ok = False

            if (not upscaled_ok) or (not interpolated_ok):
                try:
                    steps = manifest.get("steps") or {}
                except Exception:
                    steps = {}
                if isinstance(steps, dict):
                    for k, rec in steps.items():
                        try:
                            kk = str(k or "").strip().lower()
                            st = str((rec or {}).get("status") or "").strip().lower() if isinstance(rec, dict) else ""
                            if st != "done":
                                continue
                            if (not upscaled_ok) and ("upscale" in kk):
                                upscaled_ok = True
                            if (not interpolated_ok) and ("interpolate" in kk and "60" in kk):
                                interpolated_ok = True
                        except Exception:
                            continue

            if upscaled_ok:
                mask |= _AssetsDotsDelegate.HAS_UPSCALED
            if interpolated_ok:
                mask |= _AssetsDotsDelegate.HAS_INTERPOLATED

            tip_lines = []
            tip_lines.append(f"Images: {img_n}")
            tip_lines.append(f"Clips: {clip_n}")
            tip_lines.append(f"Sound files: {aud_n}")
            tip_lines.append(f"Final cut: {'Yes' if final_ok else 'No'}")
            tip_lines.append(f"Upscaled: {'Yes' if upscaled_ok else 'No'}")
            tip_lines.append(f"Interpolated (60fps): {'Yes' if interpolated_ok else 'No'}")
            assets_tip = "\n".join(tip_lines)

            out.append({
                "job_dir": str(d),
                "title": title,
                "status": status,
                "updated": updated,
                "assets_mask": mask,
                "assets_tooltip": assets_tip,
                "final_video": final_video,
            })

        return out

    def _derive_job_updated_ts(self, job_dir: str, manifest: dict) -> str:
        # Prefer explicit timestamps in manifest, fallback to folder mtime
        ts = None
        try:
            ts = (manifest.get("last_updated") or manifest.get("updated_at") or None)
        except Exception:
            ts = None

        if ts is None:
            try:
                ts = os.path.getmtime(job_dir)
            except Exception:
                ts = 0

        try:
            dt = datetime.datetime.fromtimestamp(float(ts))
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""

    def _derive_job_status(self, job_dir: str, manifest: dict) -> str:
        # Failed if any step failed
        try:
            steps = manifest.get("steps") or {}
            if isinstance(steps, dict):
                for _, rec in steps.items():
                    if isinstance(rec, dict) and str(rec.get("status") or "").lower() == "failed":
                        return "Failed"
        except Exception:
            pass

        # Done if final cut exists
        try:
            final_cut = os.path.join(job_dir, "final", "final_cut.mp4")
            if os.path.exists(final_cut) and os.path.getsize(final_cut) > 1024:
                return "Done"
        except Exception:
            pass

        # Partial otherwise
        return "Partial"


# -------------------------
    # Behavior / wiring
    # -------------------------

    def _sync_quality_hint(self) -> None:
        # Legacy (pre-Chunk 9A) quality preset hint. Kept for compatibility.
        try:
            if not hasattr(self, "cmb_quality") or not hasattr(self, "lbl_quality_hint"):
                return
            q = self.cmb_quality.currentText()
            d = _quality_defaults(q)
            self.lbl_quality_hint.setText(
                f"{q} preset: {d.get('note')} — CRF {d.get('crf')} (target ~{d.get('bitrate_kbps')} kbps)."
            )
        except Exception:
            pass

    def _sync_settings_summary(self) -> None:
        # Chunk 9A: reflect Settings → Format / Upscaling in the Generate tab summary.
        try:
            aspect_label = ""
            if hasattr(self, "cmb_aspect"):
                aspect_label = str(self.cmb_aspect.currentText() or "")
            disp = _aspect_mode_display(aspect_label or "landscape")
            if hasattr(self, "lbl_summary_format"):
                self.lbl_summary_format.setText(disp)
        except Exception:
            pass
        try:
            txt = "Off"
            on = False
            try:
                if hasattr(self, "chk_planner_upscale"):
                    on = bool(self.chk_planner_upscale.isChecked())
            except Exception:
                on = False

            if on:
                eng = ""
                mod = ""
                try:
                    eng = str(self.cmb_planner_upscale_engine.currentText() or "").strip() if hasattr(self, "cmb_planner_upscale_engine") else ""
                except Exception:
                    eng = ""
                try:
                    mod = str(self.cmb_planner_upscale_model.currentText() or "").strip() if hasattr(self, "cmb_planner_upscale_model") else ""
                except Exception:
                    mod = ""
                if eng and mod and mod != "(default)":
                    txt = "On"
                else:
                    txt = "On (select model)"

            if hasattr(self, "lbl_summary_upscale"):
                self.lbl_summary_upscale.setText(txt)
        except Exception:
            pass


    # -------------------------
    # Auto background music volume
    # -------------------------
    def _mark_music_volume_overridden(self) -> None:
        """User touched the music volume slider; stop auto-defaulting until restart."""
        try:
            self._music_volume_user_override = True
        except Exception:
            pass

    def _apply_auto_music_volume_default(self) -> None:
        """Default music volume depends on Narration toggle:
        - Narration ON  -> 25%
        - Narration OFF -> 100%
        Only applies if user hasn't manually overridden the slider this session.
        """
        try:
            if bool(getattr(self, "_music_volume_user_override", False)):
                return
        except Exception:
            pass

        try:
            narr_on = bool(self.chk_story.isChecked())
        except Exception:
            narr_on = True

        default_v = 25 if narr_on else 100

        try:
            # Avoid treating programmatic updates as user overrides.
            self.sld_music_vol.blockSignals(True)
            self.sld_music_vol.setValue(int(default_v))
            self.sld_music_vol.blockSignals(False)
        except Exception:
            try:
                self.sld_music_vol.setValue(int(default_v))
            except Exception:
                pass

        try:
            self.lbl_music_vol.setText(f"{int(default_v)}%")
        except Exception:
            pass


    def _sync_toggle_visibility(self) -> None:
        self.story_vol_row.setVisible(self.chk_story.isChecked())
        self.music_vol_row.setVisible(self.chk_music.isChecked())
        self._sync_narration_ui()

    def _sync_narration_ui(self) -> None:
        """Chunk 6B: keep narration UI minimal and consistent."""
        storytelling = bool(self.chk_story.isChecked())
        # Show voice controls only when narration is enabled
        try:
            self.narration_row.setVisible(storytelling)
        except Exception:
            pass
    
        voice = ""
        try:
            voice = str(self.cmb_narr_voice.currentText() or "").strip()
        except Exception:
            voice = ""
    
        is_clone = (voice == "add your own…")
        try:
            self.voice_sample_row.setVisible(storytelling and is_clone)
        except Exception:
            pass
    
    def _browse_voice_sample(self) -> None:
        try:
            fn, _ = QFileDialog.getOpenFileName(
                self,
                "Select voice sample",
                "",
                "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg);;All files (*.*)"
            )
        except Exception:
            fn = ""
        if fn:
            try:
                self.voice_sample_path_edit.setText(fn)
            except Exception:
                pass
        self._sync_silent_logic()

    def _sync_silent_logic(self) -> None:
        silent = self.chk_silent.isChecked()
        self._sync_narration_ui()

        # Chunk 6B: narration settings (UI validation only; hard validation happens in _build_job)
        try:
            storytelling = bool(self.chk_story.isChecked())
        except Exception:
            storytelling = False

        narration_voice = "ryan"
        narration_mode = "builtin"
        narration_sample_path = ""
        narration_language = "auto"
        try:
            narration_language = str(self.cmb_narr_lang.currentText() or "auto").strip() or "auto"
        except Exception:
            narration_language = "auto"

        try:
            v = str(self.cmb_narr_voice.currentText() or "").strip()
        except Exception:
            v = ""
        if v == "add your own…":
            narration_mode = "clone"
            narration_voice = ""
            try:
                narration_sample_path = str(self.voice_sample_path_edit.text() or "").strip()
            except Exception:
                narration_sample_path = ""
        else:
            narration_mode = "builtin"
            narration_voice = v or "ryan"
            narration_sample_path = ""

        # UI-only block reason (don't throw during __init__)
        ui_err = ""
        if storytelling and (not silent) and narration_mode == "clone" and not narration_sample_path:
            ui_err = "Narration is set to 'add your own…' but no voice sample file was provided."

        try:
            self._ui_block_error = ui_err
        except Exception:
            pass

        if silent:
            # If silent: it doesn't make sense to keep audio toggles on.
            self.chk_story.setChecked(False)
            self.chk_music.setChecked(False)
            self.chk_story.setEnabled(False)
            self.chk_music.setEnabled(False)
            try:
                self._ui_block_error = ""
            except Exception:
                pass
        else:
            self.chk_story.setEnabled(True)
            self.chk_music.setEnabled(True)

        # Reflect updated validation state (prompt required + any UI-only blocks).
        self._refresh_header_actions()
        self._sync_toggle_visibility()

    def _on_toggle_probe_logs(self, on: bool) -> None:
        """Enable/disable debug logging to ./logs/. Keep OFF unless debugging."""
        try:
            _LOGGER.enable(bool(on))
        except Exception:
            pass
        try:
            s = _load_planner_settings()
            s["probe_logs"] = bool(on)
            _save_planner_settings(s)
        except Exception:
            pass
        try:
            _LOGGER.log_probe(f"Probe & logs toggled {'ON' if on else 'OFF'}")
        except Exception:
            pass
    def _on_toggle_character_bible(self, on: bool) -> None:
        """Persist Character Bible toggle. When OFF, the pipeline skips bible generation and lock injection."""
        # Mutual exclusivity: enabling Character Bible disables Own Character Bible.
        try:
            if bool(on) and hasattr(self, "chk_own_character_bible") and self.chk_own_character_bible.isChecked():
                self.chk_own_character_bible.blockSignals(True)
                self.chk_own_character_bible.setChecked(False)
                self.chk_own_character_bible.blockSignals(False)
                try:
                    if hasattr(self, "own_character_bible_block"):
                        self.own_character_bible_block.setVisible(False)
                except Exception:
                    pass
                try:
                    s = _load_planner_settings()
                    s["own_character_bible_enabled"] = False
                    _save_planner_settings(s)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            s = _load_planner_settings()
            s["character_bible_enabled"] = bool(on)
            _save_planner_settings(s)
        except Exception:
            pass
        try:
            _LOGGER.log_probe(f"Character bible toggled {'ON' if on else 'OFF'}")
        except Exception:
            pass

    def _on_toggle_own_character_bible(self, on: bool) -> None:
        """Persist Own Character Bible toggle and enforce mutual exclusivity with auto Character Bible."""
        try:
            if hasattr(self, "own_character_bible_block"):
                self.own_character_bible_block.setVisible(bool(on))
        except Exception:
            pass

        # Mutual exclusivity: enabling Own Character Bible disables Character Bible.
        try:
            if bool(on) and hasattr(self, "chk_character_bible") and self.chk_character_bible.isChecked():
                self.chk_character_bible.blockSignals(True)
                self.chk_character_bible.setChecked(False)
                self.chk_character_bible.blockSignals(False)
                try:
                    s = _load_planner_settings()
                    s["character_bible_enabled"] = False
                    _save_planner_settings(s)
                except Exception:
                    pass
        except Exception:
            pass

        try:
            s = _load_planner_settings()
            s["own_character_bible_enabled"] = bool(on)
            _save_planner_settings(s)
        except Exception:
            pass
        try:
            _LOGGER.log_probe(f"Own character bible toggled {'ON' if on else 'OFF'}")
        except Exception:
            pass

    

    def _on_toggle_alternative_storymode(self, on: bool) -> None:
        """Persist Alternative storymode toggle (direct Qwen prompt list)."""
        try:
            s = _load_planner_settings()
            s["alternative_storymode"] = bool(on)
            _save_planner_settings(s)
        except Exception:
            pass
        try:
            _LOGGER.log_probe(f"Alternative storymode toggled {'ON' if on else 'OFF'}")
        except Exception:
            pass
    # info: Chunk 10 side quest — Own storyline is strongest (UI lock)
    def _apply_own_storyline_lock_state(self, on: bool) -> None:
        """When Own storyline is ON, lock conflicting automation toggles.

        Rules:
        - Alternative storymode: forced OFF + disabled.
        - Auto Character Bible: forced OFF + disabled.
        - Own Character Bible: forced OFF each time Own storyline turns ON, but stays ENABLED
          so the user can opt back in manually if they really want it.
        """

        # Always-disable targets when Own storyline is ON
        hard_disable = [
            "chk_alternative_storymode",
            "chk_character_bible",
        ]

        if bool(on):
            # Force OFF + disable (greyed out)
            for attr in hard_disable:
                try:
                    if hasattr(self, attr):
                        w = getattr(self, attr)
                        try:
                            w.setChecked(False)
                        except Exception:
                            pass
                        try:
                            w.setEnabled(False)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Own Character Bible: force OFF, but keep enabled
            try:
                if hasattr(self, "chk_own_character_bible"):
                    w = getattr(self, "chk_own_character_bible")
                    try:
                        w.setChecked(False)
                    except Exception:
                        pass
                    try:
                        w.setEnabled(True)
                    except Exception:
                        pass
            except Exception:
                pass
        else:
            # Re-enable controls (leave them OFF; user can opt back in manually)
            for attr in hard_disable:
                try:
                    if hasattr(self, attr):
                        w = getattr(self, attr)
                        try:
                            w.setEnabled(True)
                        except Exception:
                            pass
                except Exception:
                    pass

            # Own Character Bible: ensure it's enabled when leaving Own storyline
            try:
                if hasattr(self, "chk_own_character_bible"):
                    getattr(self, "chk_own_character_bible").setEnabled(True)
            except Exception:
                pass


    # info: Chunk 10 side quest — Own storyline settings handlers (Step 1: UI only)
    def _on_toggle_own_storyline(self, on: bool) -> None:
        """Persist Own storyline toggle and show/hide the textbox."""
        try:
            if hasattr(self, "own_storyline_block"):
                self.own_storyline_block.setVisible(bool(on))
        except Exception:
            pass


        try:
            self._apply_own_storyline_lock_state(bool(on))
        except Exception:
            pass

        try:
            self._update_own_storyline_prompt_preview()
        except Exception:
            pass
        try:
            s = _load_planner_settings()
            s["own_storyline_enabled"] = bool(on)
            _save_planner_settings(s)
        except Exception:
            pass

    def _on_own_storyline_text_changed(self) -> None:
        """Persist the raw own storyline text (best-effort)."""
        try:
            txt = (self.own_storyline_edit.toPlainText() or "") if hasattr(self, "own_storyline_edit") else ""
        except Exception:
            txt = ""
        try:
            s = _load_planner_settings()
            s["own_storyline_text"] = str(txt)
            _save_planner_settings(s)
        except Exception:
            pass
        try:
            self._update_own_storyline_prompt_preview()
        except Exception:
            pass


    # info: Chunk 10 side quest — Own storyline preview updater (Step 2: preview only)
    def _update_own_storyline_prompt_preview(self) -> None:
        """Parse the own storyline textbox and update the live prompt counter (preview only)."""
        try:
            txt = (self.own_storyline_edit.toPlainText() or "") if hasattr(self, "own_storyline_edit") else ""
        except Exception:
            txt = ""

        try:
            prompts, mode = _parse_own_storyline_prompts(txt)
        except Exception:
            prompts, mode = [], "paragraph"

        # cache for later chunks (do not persist yet)
        try:
            self._own_storyline_parsed_prompts = prompts
            self._own_storyline_parser_mode = mode
        except Exception:
            pass

        n = 0
        try:
            n = len(prompts)
        except Exception:
            n = 0

        # update counter label
        try:
            if hasattr(self, "lbl_own_storyline_count") and self.lbl_own_storyline_count is not None:
                if n <= 0:
                    self.lbl_own_storyline_count.setText("Detected prompts: 0")
                else:
                    self.lbl_own_storyline_count.setText(f"Detected prompts: {n} → {n} images will be generated")
        except Exception:
            pass

        # show a gentle warning only for the most ambiguous fallback
        try:
            if hasattr(self, "lbl_own_storyline_warn") and self.lbl_own_storyline_warn is not None:
                if n > 0 and str(mode) == "paragraph":
                    self.lbl_own_storyline_warn.setText("No markers found — using paragraphs.")
                    self.lbl_own_storyline_warn.setVisible(True)
                else:
                    self.lbl_own_storyline_warn.setVisible(False)
        except Exception:
            pass

    def _on_own_character_prompt_changed(self) -> None:
        """Persist manual character prompts (best-effort, UI thread)."""
        try:
            c1 = (self.own_char_1.toPlainText() or "").strip() if hasattr(self, "own_char_1") else ""
            c2 = (self.own_char_2.toPlainText() or "").strip() if hasattr(self, "own_char_2") else ""
        except Exception:
            c1, c2 = "", ""
        try:
            s = _load_planner_settings()
            s["own_character_1_prompt"] = str(c1)
            s["own_character_2_prompt"] = str(c2)
            _save_planner_settings(s)
        except Exception:
            pass

    def _sync_use_framevision_queue_label(self) -> None:
        try:
            on = bool(self.chk_use_framevision_queue.isChecked())
        except Exception:
            on = bool(getattr(self, "_use_framevision_queue", True))
        try:
            if on:
                self.lbl_queue_mode_note.setText("Some 'Cancel' features are disabled when using the FrameVision queue.")
            else:
                self.lbl_queue_mode_note.setText("Using internal Planner queue, all 'Cancel' features are enabled.")
        except Exception:
            pass

    def _on_toggle_use_framevision_queue(self, on: bool) -> None:
        """Persist Planner queue mode (FrameVision queue lock vs standalone Planner queue)."""
        try:
            self._use_framevision_queue = bool(on)
        except Exception:
            self._use_framevision_queue = True
        try:
            s = _load_planner_settings()
            s["use_framevision_queue"] = bool(on)
            _save_planner_settings(s)
        except Exception:
            pass
        try:
            self._sync_use_framevision_queue_label()
        except Exception:
            pass

        # If switching OFF, stop any external FV lock polling and clear recovered state.
        if not bool(on):
            try:
                self._set_external_running(None)
            except Exception:
                pass
            try:
                t = getattr(self, "_external_poll_timer", None)
                if t is not None:
                    t.stop()
            except Exception:
                pass
            try:
                self._persist_running = None
                self._save_planner_queue_state()
            except Exception:
                pass

    def _on_toggle_allow_edit_while_running(self, on: bool) -> None:
        """Chunk 8A: Persist the interactive review/edit gate toggle."""
        try:
            s = _load_planner_settings()
            s["allow_edit_while_running"] = bool(on)
            _save_planner_settings(s)
        except Exception:
            pass

    def _on_toggle_preview(self, on: bool) -> None:
        """Persist the header preview strip toggle."""
        try:
            s = _load_planner_settings()
            s["preview_enabled"] = bool(on)
            _save_planner_settings(s)
        except Exception:
            pass
        try:
            self._apply_preview_visibility(bool(on))
        except Exception:
            pass

        try:
            _LOGGER.log_probe(f"Allow edit while running toggled {'ON' if on else 'OFF'}")
        except Exception:
            pass

    
    def _on_videoclip_creator_preset_changed(self, _=None) -> None:
        """Chunk 10A: Persist Videoclip Creator preset dropdown selection."""
        try:
            try:
                choice = self.cmb_videoclip_preset.currentText()
            except Exception:
                choice = ""
            s = _load_planner_settings()
            s["videoclip_creator_preset"] = str(choice or "")
            _save_planner_settings(s)
        except Exception:
            pass

    def _browse_out_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d:
            self.out_dir_edit.setText(d)

    def _copy_log(self) -> None:
        QApplication.clipboard().setText(self.log.toPlainText())



    # -------------------------
    # Optional — Reference images (Chunk 4)
    # -------------------------

    def _toggle_ref_block(self, checked: bool) -> None:
        try:
            self.ref_block.setVisible(bool(checked))
            self.btn_ref_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        except Exception:
            pass

    
    
    def _load_ace15_preset_combo(self, combo: "QComboBox") -> None:
        """Load Ace Step 1.5 genre presets from presetmanager.json.

        File shape (expected): {"version": 1, "genres": {"Genre": {"subgenres": {"Sub": {payload}}}}}

        We show items as "Genre / Subgenre" and store the internal key as "Genre::Subgenre".
        """
        try:
            combo.clear()
        except Exception:
            pass

        root = str(_root())
        path = os.path.join(root, "presets", "setsave", "ace15presets", "presetmanager.json")

        try:
            obj = _safe_read_json(path) if os.path.isfile(path) else None
        except Exception:
            obj = None

        items: list[tuple[str, str]] = []  # (label, key)

        # Preferred shape: genres/subgenres dict
        if isinstance(obj, dict) and isinstance(obj.get("genres"), dict):
            genres = obj.get("genres") or {}
            for g in sorted(genres.keys(), key=lambda s: str(s).lower()):
                gd = genres.get(g) or {}
                subs = (gd.get("subgenres") or {}) if isinstance(gd, dict) else {}
                if not isinstance(subs, dict):
                    continue
                for s in sorted(subs.keys(), key=lambda x: str(x).lower()):
                    key = f"{str(g)}::{str(s)}"
                    label = f"{str(g)} / {str(s)}"
                    items.append((label, key))

        # Fallback: list root / {"presets":[...]}
        if not items:
            presets = []
            try:
                if isinstance(obj, dict):
                    presets = obj.get("presets") or obj.get("items") or []
                elif isinstance(obj, list):
                    presets = obj
            except Exception:
                presets = []

            def _pick_id(o: dict) -> str:
                for k in ("id", "key", "name", "title", "label"):
                    v = o.get(k) if isinstance(o, dict) else None
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                return ""

            def _pick_name(o: dict) -> str:
                for k in ("title", "name", "label"):
                    v = o.get(k) if isinstance(o, dict) else None
                    if isinstance(v, str) and v.strip():
                        return v.strip()
                _id = _pick_id(o)
                return _id or "Preset"

            if isinstance(presets, list):
                for it in presets:
                    if not isinstance(it, dict):
                        continue
                    pid = _pick_id(it) or _pick_name(it)
                    pname = _pick_name(it)
                    items.append((pname, pid))

        if not items:
            try:
                combo.addItem("Default", "default")
            except Exception:
                pass
            return

        added = 0
        for label, key in items:
            try:
                combo.addItem(str(label), str(key))
                added += 1
            except Exception:
                pass

        if added <= 0:
            try:
                combo.addItem("Default", "default")
            except Exception:
                pass


    def _toggle_music_block(self, checked: bool) -> None:
        try:
            self.music_block.setVisible(bool(checked))
            self.btn_music_toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        except Exception:
            pass

    def _sync_music_source_ui(self) -> None:
        """Enable/disable widgets depending on selected music source.

        - Ace Step 1.5: shows preset + lyrics controls
        - Use your own (file): enables file picker
        """
        try:
            is_ace15 = bool(getattr(self, "rad_music_ace15", None) and self.rad_music_ace15.isChecked())
        except Exception:
            is_ace15 = False
        try:
            is_file = bool(getattr(self, "rad_music_file", None) and self.rad_music_file.isChecked())
        except Exception:
            is_file = False

        # Ace settings block
        try:
            self.ace15_block.setVisible(is_ace15)
        except Exception:
            pass

        # Lyrics box only visible when lyrics toggle is enabled (and Ace is selected)
        try:
            show_lyrics = bool(is_ace15 and getattr(self, "chk_ace15_lyrics", None) and self.chk_ace15_lyrics.isChecked())
        except Exception:
            show_lyrics = False
        try:
            self.txt_ace15_lyrics.setVisible(show_lyrics)
        except Exception:
            pass

        # File picker only relevant for "Use your own"
        try:
            self.btn_music_add_file.setEnabled(is_file)
        except Exception:
            pass
        try:
            has_file = bool((getattr(self, "_music_file_path", "") or "").strip())
        except Exception:
            has_file = False
        try:
            self.btn_music_clear_file.setEnabled(is_file and has_file)
        except Exception:
            pass

        # If user is selecting any music source, make sure the Music toggle is on and silent is off.
        if is_ace15 or is_file:
            try:
                if self.chk_silent.isChecked():
                    self.chk_silent.setChecked(False)
                self.chk_music.setChecked(True)
            except Exception:
                pass

    def _sync_image_model_lock_for_qwen2511_refs(self) -> None:
        "Disable Image model selection when Qwen Edit 2511 is actively driving image creation via reference images."
        try:
            has_refs = bool(getattr(self, "ref_list", None) and int(self.ref_list.count()) > 0)
        except Exception:
            has_refs = False
        try:
            strat = str(getattr(self, "_ref_strategy", "") or "").strip()
        except Exception:
            strat = ""

        lock = bool(has_refs and strat == "qwen2511_best")

        # The combo lives in Settings; guard for early calls during init.
        cmb = getattr(self, "cmb_image_model", None)

        hint = "Using Qwen Edit for reference image creation"
        hint_key = "__QWEN2511_REF_LOCK__"

        if lock:
            try:
                if not cmb:
                    self._img_model_lock_active = True
                    return
            except Exception:
                return

            if not bool(getattr(self, "_img_model_lock_active", False)):
                try:
                    self._img_model_prev_index = int(cmb.currentIndex())
                except Exception:
                    self._img_model_prev_index = -1

            # Ensure a visible, stable hint item exists (and is selected).
            try:
                idx_hint = -1
                for i in range(int(cmb.count())):
                    try:
                        if str(cmb.itemData(i) or "") == hint_key:
                            idx_hint = i
                            break
                    except Exception:
                        continue
                if idx_hint < 0:
                    cmb.insertItem(0, hint)
                    cmb.setItemData(0, hint_key)
                    idx_hint = 0
                cmb.setCurrentIndex(idx_hint)
            except Exception:
                pass

            try:
                cmb.setEnabled(False)
            except Exception:
                pass
            try:
                cmb.setToolTip(hint)
            except Exception:
                pass

            self._img_model_lock_active = True
            return

        # unlock
        if not bool(getattr(self, "_img_model_lock_active", False)):
            return

        try:
            if cmb:
                idx_hint = -1
                for i in range(int(cmb.count())):
                    try:
                        if str(cmb.itemData(i) or "") == hint_key:
                            idx_hint = i
                            break
                    except Exception:
                        continue
                if idx_hint >= 0:
                    cmb.removeItem(idx_hint)

                cmb.setEnabled(True)
                cmb.setToolTip("")

                prev = int(getattr(self, "_img_model_prev_index", -1))
                if prev >= 0 and prev < int(cmb.count()):
                    cmb.setCurrentIndex(prev)
        except Exception:
            pass

        self._img_model_lock_active = False



    def _on_add_ref_images_clicked(self) -> None:
        # Expand block when adding
        try:
            if hasattr(self, "btn_ref_toggle") and not self.btn_ref_toggle.isChecked():
                self.btn_ref_toggle.setChecked(True)
        except Exception:
            pass

        caption = "Select reference image(s)"
        filters = "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        files, _ = QFileDialog.getOpenFileNames(self, caption, "", filters)
        if not files:
            return

        had_any = False
        try:
            had_any = (self.ref_list.count() > 0)
        except Exception:
            had_any = False

        for p in files:
            item = QListWidgetItem(f"ref_image: {p}")
            item.setData(Qt.UserRole, {"kind": "ref_images", "path": p})
            self.ref_list.addItem(item)

        # Prompt for strategy once, when the first ref is added
        try:
            if (not had_any) and (self.ref_list.count() > 0) and not (getattr(self, "_ref_strategy", "") or "").strip():
                self._prompt_ref_strategy_choice()
        except Exception:
            pass

        try:
            self._sync_image_model_lock_for_qwen2511_refs()
        except Exception:
            pass

    def _remove_selected_ref_images(self) -> None:
        try:
            for item in self.ref_list.selectedItems():
                row = self.ref_list.row(item)
                self.ref_list.takeItem(row)
        except Exception:
            return

        try:
            self._sync_image_model_lock_for_qwen2511_refs()
        except Exception:
            pass

    def _clear_ref_images(self) -> None:
        try:
            self.ref_list.clear()
        except Exception:
            pass
        self._ref_strategy = ""
        self._ref_multi_angle_lora = ""
        self._ref_qwen2511_high_quality = False
        try:
            self.lbl_ref_strategy.setText("Strategy: (none)")
        except Exception:
            pass

        try:
            self._sync_image_model_lock_for_qwen2511_refs()
        except Exception:
            pass

    def _prompt_ref_strategy_choice(self) -> None:
        """Ask the user how reference images should be used (Chunk 4)."""
        try:
            n = int(self.ref_list.count())
        except Exception:
            n = 0

        msg = QMessageBox(self)
        msg.setWindowTitle('Reference images')
        msg.setIcon(QMessageBox.Question)
        msg.setText(
            f"{n} reference image(s) added. Choose how to use them:\n\n"
            'A) Use image(s) as reference → Qwen3-VL Describe generates story prompt guidance (fast)\n'
            'B) Best attempt consistency → Qwen Edit 2511 (max 2 refs, slower; 16GB+ VRAM recommended)\n'
            'Cancel → lightweight reuse (Qwen-VL look & reuse obvious objects)\n'
        )

        btn_a = msg.addButton('A) Qwen3-VL Describe', QMessageBox.AcceptRole)
        btn_b = msg.addButton('B) Qwen Edit 2511', QMessageBox.AcceptRole)
        btn_c = msg.addButton('Cancel (lightweight reuse)', QMessageBox.RejectRole)
        msg.setDefaultButton(btn_c)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_a:
            self._ref_strategy = 'qwen3vl_describe'
            self._ref_multi_angle_lora = ''
            self._ref_qwen2511_high_quality = False

        elif clicked == btn_b:
            self._ref_strategy = 'qwen2511_best'

            # Extra option: HQ mode (1280x720)
            try:
                self._ref_qwen2511_high_quality = bool(self._prompt_qwen2511_quality_choice())
            except Exception:
                self._ref_qwen2511_high_quality = False

            # Enforce max 2 refs for the Qwen 2511 path (keep list intact; only first 2 used)
            try:
                if self.ref_list.count() > 2:
                    QMessageBox.information(
                        self,
                        'Qwen 2511 refs',
                        'Qwen Edit 2511 uses max 2 reference images.\n'
                        'Extra references will be kept in the list but only the first 2 will be used.',
                    )
            except Exception:
                pass

            # Auto-detect multi-angle LoRA (if present) and inform user.
            try:
                lora = _find_newest_qwen2511_multi_angle_lora()
                if lora:
                    self._ref_multi_angle_lora = lora
                    QMessageBox.information(
                        self,
                        'Multi-angle LoRA',
                        'Multi-angle LoRA detected and will be auto-enabled for some shots:\n\n' + str(lora),
                    )
                else:
                    self._ref_multi_angle_lora = ''
                    QMessageBox.information(
                        self,
                        'Multi-angle LoRA',
                        'No multi-angle LoRA detected under models/loras/.\n'
                        'The planner will still run Qwen Edit 2511 without it.',
                    )
            except Exception:
                self._ref_multi_angle_lora = ''

        else:
            self._ref_strategy = 'qwenvl_reuse'
            self._ref_multi_angle_lora = ''
            self._ref_qwen2511_high_quality = False

        try:
            self.lbl_ref_strategy.setText(f"Strategy: {self._ref_strategy or '(none)'}")
        except Exception:
            pass

        try:
            self._sync_image_model_lock_for_qwen2511_refs()
        except Exception:
            pass

    def _prompt_qwen2511_quality_choice(self) -> bool:
        """Extra option shown after choosing Qwen Edit 2511 (Chunk 4).

        When enabled, Qwen 2511 renders at 1280x720 (slower; up to ~5 minutes per image).
        """
        try:
            msg = QMessageBox(self)
            msg.setWindowTitle("Qwen Edit 2511")
            msg.setIcon(QMessageBox.Question)
            msg.setText(
                "Quality setting for Qwen Edit 2511:\n\n"
                "Normal: 1024×568 (faster)\n"
                "High Quality: 1280×720 (up to 5 minutes per image)\n"
            )
            btn_normal = msg.addButton("Normal (faster)", QMessageBox.AcceptRole)
            btn_hq = msg.addButton("Use High Quality (slower)", QMessageBox.AcceptRole)
            msg.setDefaultButton(btn_normal)
            msg.exec()
            clicked = msg.clickedButton()
            return bool(clicked == btn_hq)
        except Exception:
            return False

    # -------------------------
    # Optional attachments
    # -------------------------
    def _choose_music_file(self) -> None:
        """Select a ready-made music file for the run."""
        try:
            fn, _ = QFileDialog.getOpenFileName(
                self,
                "Select music file",
                "",
                "Audio files (*.wav *.mp3 *.m4a *.flac *.ogg *.aac);All files (*.*)"
            )
        except Exception:
            fn = ""
        if not fn:
            return
        try:
            self._music_file_path = str(fn)
        except Exception:
            pass
        try:
            if hasattr(self, "lbl_music_file"):
                self.lbl_music_file.setText(os.path.basename(fn) if fn else "No file selected")
                self.lbl_music_file.setToolTip(fn)
        except Exception:
            pass
        try:
            if hasattr(self, "rad_music_file"):
                self.rad_music_file.setChecked(True)
        except Exception:
            pass
        self._sync_music_source_ui()

    def _clear_music_file(self) -> None:
        try:
            self._music_file_path = ""
        except Exception:
            pass
        try:
            if hasattr(self, "lbl_music_file"):
                self.lbl_music_file.setText("No file selected")
                self.lbl_music_file.setToolTip("")
        except Exception:
            pass
        self._sync_music_source_ui()



    def _collect_attachments(self) -> Dict[str, List[str]]:
        # Chunk 4: reference images are stored separately, but keep backward-compat with older "images" key.
        out = {"json": [], "ref_images": [], "images": [], "videos": [], "music": [], "music_new": [], "text": [], "transcripts": []}

        # Reference images list (if present)
        try:
            if hasattr(self, "ref_list"):
                for i in range(self.ref_list.count()):
                    item = self.ref_list.item(i)
                    data = item.data(Qt.UserRole) or {}
                    p = data.get("path", "")
                    if p:
                        out["ref_images"].append(p)
        except Exception:
            pass
        # Ready-made music file (if present)
        try:
            p = str(getattr(self, "_music_file_path", "") or "").strip()
            if p:
                out["music"].append(p)
        except Exception:
            pass

        # Backward compatibility: if older "images" were used, treat them as ref_images when ref_images is empty.
        if not out["ref_images"] and out["images"]:
            out["ref_images"] = list(out["images"])

        return out

    def _build_job(self) -> PlannerJob:
        prompt = (self.prompt_edit.toPlainText() or "").strip()
        own_story = False
        try:
            own_story = bool(getattr(self, "chk_own_storyline", None) and self.chk_own_storyline.isChecked())
        except Exception:
            own_story = False
        if not prompt and not own_story:
            raise ValueError("Prompt is empty.")
        # Own storymode does not require an extra prompt; keep a small placeholder for titles/folders.
        if not prompt and own_story:
            prompt = "Own storyline"

        negatives = (self.negatives_edit.toPlainText() or "").strip()

        # Collect attachments once (used for music file detection + ref images)
        attachments = self._collect_attachments()

        storytelling = self.chk_story.isChecked()

        # Chunk 7: determine music source.
        # - Ace Step 1.5: generates music (optionally with lyrics) using the preset manager.
        # - Use your own (file): attaches an existing file.
        try:
            sel_ace15 = bool(hasattr(self, "rad_music_ace15") and self.rad_music_ace15.isChecked())
        except Exception:
            sel_ace15 = False
        try:
            sel_file = bool(hasattr(self, "rad_music_file") and self.rad_music_file.isChecked())
        except Exception:
            sel_file = True

        # Music checkbox still acts as the master toggle for 'Use your own', but explicit modes override it.
        music_toggle = self.chk_music.isChecked()
        has_attached_music = bool((attachments or {}).get("music"))

        # If Ace Step 1.5 is selected, music is always requested regardless of the Music checkbox.
        # If file is selected, require the Music checkbox OR an attached file.
        music_requested = bool(sel_ace15 or (sel_file and (music_toggle or has_attached_music)))

        music_mode = "none"
        music_preset = ""  # legacy
        ace15_preset_id = ""
        ace15_lyrics_enabled = False
        ace15_lyrics_text = ""
        ace15_audio_format = "wav"

        if music_requested:
            if sel_ace15:
                music_mode = "ace15"
                try:
                    ace15_preset_id = str(self.cmb_ace15_preset.currentData() or "").strip()
                    if not ace15_preset_id:
                        ace15_preset_id = str(self.cmb_ace15_preset.currentText() or "").strip()
                except Exception:
                    ace15_preset_id = ""
                # Keep legacy field filled with a human readable name for older logs/UIs
                try:
                    music_preset = str(self.cmb_ace15_preset.currentText() or "").strip()
                except Exception:
                    music_preset = ""

                try:
                    ace15_lyrics_enabled = bool(self.chk_ace15_lyrics.isChecked())
                except Exception:
                    ace15_lyrics_enabled = False
                try:
                    ace15_lyrics_text = (self.txt_ace15_lyrics.toPlainText() or "").strip()
                except Exception:
                    ace15_lyrics_text = ""

                try:
                    ace15_audio_format = str(self.cmb_ace15_format.currentData() or "").strip().lower()
                    if ace15_audio_format not in ("mp3","wav"):
                        ace15_audio_format = str(self.cmb_ace15_format.currentText() or "").strip().lower()
                    if ace15_audio_format not in ("mp3","wav"):
                        ace15_audio_format = "wav"
                except Exception:
                    ace15_audio_format = "wav"
            else:
                music_mode = "file"

        # Base silent flag
        silent = self.chk_silent.isChecked()

        # Chunk 6B: narration is only gated by Storytelling + not silent.
        narration_enabled = bool(storytelling and (not silent))
# Chunk 6B: narration settings (validated here to block run cleanly)
        narration_voice = "ryan"
        narration_mode = "builtin"
        narration_sample_path = ""
        narration_language = "auto"

        try:
            narration_language = str(self.cmb_narr_lang.currentText() or "auto").strip() or "auto"
        except Exception:
            narration_language = "auto"

        try:
            v = str(self.cmb_narr_voice.currentText() or "").strip()
        except Exception:
            v = ""
        if v == "add your own…":
            narration_mode = "clone"
            narration_voice = ""
            try:
                narration_sample_path = str(self.voice_sample_path_edit.text() or "").strip()
            except Exception:
                narration_sample_path = ""
        else:
            narration_mode = "builtin"
            narration_voice = v or "ryan"
            narration_sample_path = ""

        if bool(narration_enabled) and narration_mode == "clone" and not narration_sample_path:
            raise ValueError("Narration is set to 'add your own…' but no voice sample file was provided.")
        # Chunk 9A: format (stored in job config)
        aspect_label = "Landscape (16:9)"
        try:
            if hasattr(self, "cmb_aspect"):
                aspect_label = str(self.cmb_aspect.currentText() or aspect_label)
        except Exception:
            aspect_label = "Landscape (16:9)"
        aspect_mode = _aspect_mode_key(aspect_label)
        # Upscaling is controlled only by Settings → Upscaling (toggle + engine/model).
        # The pipeline will upscale only when that toggle is ON and a model is selected.
        upscale_factor = 1

        # Fixed bitrate policy (no UI)
        video_bitrate_kbps = 3500

        # Keep CRF as a fallback, but default to bitrate mode
        enc = {"mode": "bitrate", "crf": 18, "bitrate_kbps": int(video_bitrate_kbps), "note": "Fixed bitrate policy"}
        enc.update({
            "aspect_mode": str(aspect_mode),
            "quality_upscale_factor": int(upscale_factor),
            "video_bitrate_kbps": int(video_bitrate_kbps),

            # Legacy keys (kept so older code/tools can still read something sensible)
            "resolution_preset": str(aspect_label),
            "quality_preset": f"{int(upscale_factor)}×",

            "image_model": self.cmb_image_model.currentText(),
            "video_model": self.cmb_video_model.currentText(),
            "videoclip_preset": self.cmb_videoclip_preset.currentText(),
            "gen_quality_preset": (self.cmb_gen_quality.currentText() if hasattr(self, "cmb_gen_quality") else ""),
            "allow_edit_while_running": bool(getattr(self, "chk_allow_edit_while_running", None) and self.chk_allow_edit_while_running.isChecked()),
            "character_bible_enabled": bool(getattr(self, "chk_character_bible", None) and self.chk_character_bible.isChecked()),
            "own_character_bible_enabled": bool(getattr(self, "chk_own_character_bible", None) and self.chk_own_character_bible.isChecked()),
            "alternative_storymode": bool(getattr(self, "chk_alternative_storymode", None) and self.chk_alternative_storymode.isChecked()),
            "own_storyline_enabled": bool(getattr(self, "chk_own_storyline", None) and self.chk_own_storyline.isChecked()),
        })

        # info: Chunk 10 side quest — Own storyline payload (Step 3: pipeline binding)
        try:
            _own_storyline_on = bool(enc.get("own_storyline_enabled"))
        except Exception:
            _own_storyline_on = False

        _own_storyline_text = ""
        _own_storyline_prompts = []
        _own_storyline_mode = "paragraph"

        if bool(_own_storyline_on):
            try:
                _own_storyline_text = (self.own_storyline_edit.toPlainText() or "") if hasattr(self, "own_storyline_edit") else ""
            except Exception:
                _own_storyline_text = ""
            try:
                _own_storyline_prompts = getattr(self, "_own_storyline_parsed_prompts", None)
                _own_storyline_mode = getattr(self, "_own_storyline_parser_mode", "paragraph")
            except Exception:
                _own_storyline_prompts = None
                _own_storyline_mode = "paragraph"
            if (not isinstance(_own_storyline_prompts, list)) or (not _own_storyline_prompts):
                try:
                    _own_storyline_prompts, _own_storyline_mode = _parse_own_storyline_prompts(_own_storyline_text)
                except Exception:
                    _own_storyline_prompts, _own_storyline_mode = [], "paragraph"

        enc["own_storyline_text"] = str(_own_storyline_text or "")
        enc["own_storyline_prompts"] = _own_storyline_prompts if isinstance(_own_storyline_prompts, list) else []
        enc["own_storyline_parser_mode"] = str(_own_storyline_mode or "paragraph")

        if bool(_own_storyline_on):
            # Own storyline bypasses planner-side prompt creation features (kept as metadata for later chunks).
            enc["character_bible_enabled"] = False
            enc["own_character_bible_enabled"] = False
            enc["alternative_storymode"] = False

        # Own Character Bible prompts (manual 1–2)
        try:
            c1 = (self.own_char_1.toPlainText() or "").strip() if hasattr(self, "own_char_1") else ""
            c2 = (self.own_char_2.toPlainText() or "").strip() if hasattr(self, "own_char_2") else ""
        except Exception:
            c1, c2 = "", ""
        enc["own_character_1_prompt"] = str(c1)
        enc["own_character_2_prompt"] = str(c2)

        # If Own Character Bible is enabled and at least one prompt exists, force-disable auto Character Bible for this run.
        try:
            if bool(enc.get("own_character_bible_enabled")) and bool((c1 or c2).strip()):
                enc["character_bible_enabled"] = False
        except Exception:
            pass

        # Resolve generation profile (proxy targets) now so fingerprints stay stable
        try:
            enc["generation_profile"] = _resolve_generation_profile(enc.get("video_model", ""), enc.get("gen_quality_preset", ""))
        except Exception:
            enc["generation_profile"] = {}


        # Chunk 9B1: Planner-only upscaling settings (saved per project folder; no execution)
        try:
            enc["planner_upscale"] = self._current_upscale_payload()
        except Exception:
            enc["planner_upscale"] = {"enabled": False}

        # Chunk 4: reference strategy (stored in encoding for pipeline behavior)
        try:
            ref_strategy = (getattr(self, "_ref_strategy", "") or "").strip()
        except Exception:
            ref_strategy = ""
        try:
            ref_lora = (getattr(self, "_ref_multi_angle_lora", "") or "").strip()
        except Exception:
            ref_lora = ""

        try:
            has_refs = bool((attachments or {}).get("ref_images"))
        except Exception:
            has_refs = False

        if has_refs and not ref_strategy:
            # Default path when user cancels the popup: lightweight reuse.
            ref_strategy = "qwenvl_reuse"

        if ref_strategy:
            enc["ref_strategy"] = ref_strategy
        if ref_lora:
            enc["ref_multi_angle_lora"] = ref_lora

        try:
            enc["qwen2511_high_quality"] = bool(getattr(self, "_ref_qwen2511_high_quality", False))
        except Exception:
            enc["qwen2511_high_quality"] = False

        job = PlannerJob(
            job_id=str(uuid.uuid4())[:8],
            created_at=time.time(),
            prompt=prompt,
            negatives=negatives,
            storytelling=storytelling,
            music_background=bool(music_requested),
            silent=silent,
            storytelling_volume=int(self.sld_story_vol.value()),
            music_volume=int(self.sld_music_vol.value()),
            music_mode=str(music_mode),
            music_preset=str(music_preset),
            ace15_preset_id=str(ace15_preset_id),
            ace15_lyrics_enabled=bool(ace15_lyrics_enabled),
            ace15_lyrics_text=str(ace15_lyrics_text),
            ace15_audio_format=str(ace15_audio_format),
            narration_enabled=bool(narration_enabled),
            narration_mode=str(narration_mode),
            narration_voice=str(narration_voice),
            narration_sample_path=str(narration_sample_path),
            narration_language=str(narration_language),
            approx_duration_sec=int(self.sld_duration.value()),
            resolution_preset=str(aspect_label),
            quality_preset=f"{int(upscale_factor)}×",
            extra_info=(self.extra_info.toPlainText() or "").strip(),
            attachments=attachments,
            encoding=enc,
        )
        return job

        # Persistent Planner queue: survive restarts and guard against double-runs while FV queue is locked.
        try:
            self._init_persistent_queue()
        except Exception:
            pass

    def _default_out_dir(self) -> str:
        # Default output path at project root
        return _default_output_base()

    def _append_log(self, line: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {line}")

        # Update elapsed runtime label opportunistically (no live timer)
        try:
            self._update_header_elapsed()
        except Exception:
            pass

    def _update_header_elapsed(self) -> None:
        """Update the small elapsed-runtime label next to the 'Planner' title.
        This is intentionally not a live clock; it updates when logs/progress events happen.
        """
        try:
            lbl = getattr(self, "lbl_header_elapsed", None)
            if lbl is None:
                return
            start = getattr(self, "_job_started_ts", None)
            mode = str(getattr(self, "_header_mode", "") or "")
            if (mode != "running") or (start is None):
                if lbl.text():
                    lbl.setText("")
                return
            try:
                elapsed = max(0, int(time.time() - float(start)))
            except Exception:
                elapsed = 0
            h = elapsed // 3600
            m = (elapsed % 3600) // 60
            s = elapsed % 60
            if h > 0:
                t = f"{h:d}:{m:02d}:{s:02d}"
            else:
                t = f"{m:02d}:{s:02d}"
            lbl.setText(f"⏱ {t}")
        except Exception:
            pass


    def _set_header_status(self, text: str, mode: str = "") -> None:
        try:
            prev = str(getattr(self, "_header_mode", "") or "")
            if mode:
                self._header_mode = str(mode)
            cur = str(getattr(self, "_header_mode", "") or "")
            # Track start time when transitioning into running mode
            if (cur == "running") and (prev != "running"):
                self._job_started_ts = time.time()
            # Clear when leaving running mode
            if (cur != "running") and (prev == "running"):
                self._job_started_ts = None
        except Exception:
            pass
        try:
            if hasattr(self, "lbl_header_status"):
                self.lbl_header_status.setText(str(text))
        except Exception:
            pass

        try:
            self._update_header_elapsed()
        except Exception:
            pass

    def _update_header_running_text(self) -> None:
        try:
            st = str(getattr(self, "_header_stage", "") or "").strip() or "Running"
        except Exception:
            st = "Running"

        # UI logger suffix: show when Own Character Bible is active.
        # This helps confirm the manual character pipeline is being used.
        try:
            enc = None
            try:
                if getattr(self, "_worker", None) is not None:
                    enc = getattr(getattr(self._worker, "job", None), "encoding", None)
            except Exception:
                enc = None

            if not isinstance(enc, dict):
                enc = {}
                try:
                    if hasattr(self, "chk_own_character_bible") and hasattr(self, "own_char_1") and hasattr(self, "own_char_2"):
                        enc["own_character_bible_enabled"] = bool(self.chk_own_character_bible.isChecked())
                        enc["own_character_1_prompt"] = str(self.own_char_1.toPlainText() or "")
                        enc["own_character_2_prompt"] = str(self.own_char_2.toPlainText() or "")
                except Exception:
                    pass

            own_on = bool(enc.get("own_character_bible_enabled"))
            c1 = str(enc.get("own_character_1_prompt") or "")
            c2 = str(enc.get("own_character_2_prompt") or "")
            own_has = bool((c1 or c2).strip())

            if own_on and own_has:
                st = f"{st} — Own character bible - ON"
            elif own_on and (not own_has):
                st = f"{st} — Own character bible - ON (empty)"
        except Exception:
            pass
        try:
            pct = int(getattr(self, "_header_pct", 0) or 0)
        except Exception:
            pct = 0
        if pct > 0:
            qn = 0
            try:
                qn = self._queue_count()
            except Exception:
                qn = 0
            _qtxt = f" — queued: {qn}" if qn > 0 else ""
            self._set_header_status(f"Running: {st} ({pct}%)" + _qtxt, mode="running")
        else:
            qn = 0
            try:
                qn = self._queue_count()
            except Exception:
                qn = 0
            _qtxt = f" — queued: {qn}" if qn > 0 else ""
            self._set_header_status(f"Running: {st}" + _qtxt, mode="running")

    @Slot()
    def _on_prompt_changed(self) -> None:
        # Any edit implies the next run is "Idle" (unless we're currently running).
        try:
            running = bool(getattr(self, "_running", False))
        except Exception:
            running = False
        if not running:
            try:
                if str(getattr(self, "_header_mode", "")) != "running":
                    qn = 0
                    try:
                        qn = self._queue_count()
                    except Exception:
                        qn = 0
                    _qtxt = f" — queued: {qn}" if qn > 0 else ""
                    self._set_header_status("Idle" + _qtxt, mode="idle")
            except Exception:
                pass
        self._refresh_header_actions()

    def _refresh_header_actions(self) -> None:
        # Context-aware header buttons
        try:
            running = bool(getattr(self, "_running", False))
        except Exception:
            running = False

        ui_block = ""
        try:
            ui_block = str(getattr(self, "_ui_block_error", "") or "").strip()
        except Exception:
            ui_block = ""

        prompt_ok = False
        prompt_text_ok = False
        own_story_enabled = False
        try:
            prompt_text_ok = bool((self.prompt_edit.toPlainText() or "").strip())
        except Exception:
            prompt_text_ok = False

        # Own storymode does NOT require the original prompt (it's optional).
        try:
            own_story_enabled = bool(getattr(self, "chk_own_storyline", None) and self.chk_own_storyline.isChecked())
        except Exception:
            own_story_enabled = False

        prompt_ok = bool(prompt_text_ok or own_story_enabled)

        # Allow queuing new jobs while one is running.
        can_generate = prompt_ok and (not bool(ui_block))
        try:
            self.btn_generate.setEnabled(bool(can_generate))
        except Exception:
            pass

        tip = ""
        if ui_block:
            tip = ui_block
        elif (not prompt_ok):
            tip = "Enter a prompt, or enable Own storymode."
        try:
            self.btn_generate.setToolTip(tip)
        except Exception:
            pass

        try:
            self.btn_cancel.setEnabled(bool(running))
        except Exception:
            pass

        # These are safe even if widgets don't exist yet during early init.
        try:
            self.btn_export_job.setEnabled(not bool(running))
        except Exception:
            pass
        try:
            self.btn_open_output.setEnabled((not bool(running)) and bool(self._last_result))
        except Exception:
            pass

    def _on_worker_stage(self, s: str) -> None:
        try:
            self._header_stage = str(s)
        except Exception:
            pass
        try:
            self.lbl_stage.setText(f"Stage: {s}")
        except Exception:
            pass
        self._update_header_running_text()

    def _on_worker_progress(self, pct: int) -> None:
        try:
            self._header_pct = int(pct)
        except Exception:
            pass
        try:
            self.progress.setValue(int(pct))
        except Exception:
            pass
        self._update_header_running_text()

    @Slot(bool)
    def _on_toggle_logs_visibility(self, on: bool) -> None:
        try:
            if hasattr(self, "btn_toggle_logs"):
                self.btn_toggle_logs.setText("Hide logs" if bool(on) else "Show logs")
        except Exception:
            pass
        try:
            if hasattr(self, "_logs_group"):
                self._logs_group.setVisible(bool(on))
        except Exception:
            pass
        try:
            if hasattr(self, "_settings_splitter"):
                if bool(on):
                    self._settings_splitter.setSizes([3, 2])
                else:
                    self._settings_splitter.setSizes([1, 0])
        except Exception:
            pass

    def _set_running(self, running: bool) -> None:
        # Keep a simple running flag for UI validation (Chunk 6B clone voice needs a sample file)
        try:
            self._running = bool(running)
        except Exception:
            pass

        # Buttons that depend only on running state
        try:
            self.btn_export_job.setEnabled(not bool(running))
        except Exception:
            pass
        try:
            self.btn_open_output.setEnabled((not bool(running)) and bool(self._last_result))
        except Exception:
            pass

        # Generate/Cancel depend on running + validation (prompt required + any UI-only blocks)
        self._refresh_header_actions()

        try:
            self.progress.setEnabled(True)
        except Exception:
            pass

    @Slot()
    def _start_pipeline(self) -> None:
        # Allow queuing multiple jobs; only one runs at a time.
        try:
            if self._worker and self._worker.isRunning():
                job = self._build_job()
                output_base = (self.out_dir_edit.text() or "").strip() or self._default_out_dir()
                output_base = _abspath_from_root(output_base)
                title = _auto_title_from_prompt(job.prompt)
                slug = _slugify_title(title) or "job"
                run_n = _next_slug_counter(output_base, slug)
                out_dir = os.path.join(output_base, f"{slug}_{run_n:03d}")
                self._enqueue_job(job, out_dir, title=title, slug=slug)
                return
        except Exception:
            pass

        try:
            job = self._build_job()
        except Exception as e:
            QMessageBox.warning(self, "Cannot start", str(e))
            return

        try:
            _LOGGER.log_probe(f"Generate clicked: {job.job_id}")
            _LOGGER.log_job(job.job_id, "Generate clicked")
        except Exception:
            pass

        output_base = (self.out_dir_edit.text() or "").strip() or self._default_out_dir()
        output_base = _abspath_from_root(output_base)

        # Chunk 8C1: Human-friendly folder naming: <slug>_<NNN>
        title = _auto_title_from_prompt(job.prompt)
        slug = _slugify_title(title) or "job"
        run_n = _next_slug_counter(output_base, slug)
        out_dir = os.path.join(output_base, f"{slug}_{run_n:03d}")

        # Start immediately
        self._run_job(job, out_dir, title=title, slug=slug)


    # -------------------------
    # Header preview strip
    # -------------------------

    @Slot(str)
    def _on_asset_created(self, path: str) -> None:
        """Live preview handler: show each newly created image/clip immediately."""
        try:
            self._preview_add(str(path or ""))
        except Exception:
            pass

    def _apply_preview_visibility(self, on: bool) -> None:
        # Always clear the UI state when the toggle changes.
        # Otherwise Qt can keep old pixmaps alive (looks like "stuck" thumbs after restart).
        try:
            self._preview_strip.setVisible(bool(on))
        except Exception:
            pass

        if not on:
            # Hard reset when turning off.
            try:
                self._preview_items = []
            except Exception:
                pass
            try:
                self._refresh_preview_strip()
            except Exception:
                pass
            try:
                if self._header_preview_dialog is not None:
                    self._header_preview_dialog.close()
            except Exception:
                pass
            return

        # Turning ON: clear any stale pixmaps, then (optionally) restore from last run.
        try:
            self._refresh_preview_strip()
        except Exception:
            pass
        try:
            self._preview_restore_from_settings()
        except Exception:
            pass

    def _preview_reset_for_new_job(self, out_dir: str = "") -> None:
        """Clear the header thumbnails at the moment a new job starts.

        This prevents showing stale previews from a previous job while the new pipeline is still warming up.
        """
        try:
            self._preview_items = []
        except Exception:
            pass
        try:
            self._refresh_preview_strip()
        except Exception:
            pass
        # Persist the directory so we can restore thumbnails after restart.
        try:
            if out_dir:
                s = _load_planner_settings()
                s["preview_last_output_dir"] = str(out_dir)
                _save_planner_settings(s)
        except Exception:
            pass

    def _preview_restore_from_settings(self) -> None:
        """Best-effort restore of header thumbnails after restart (when Preview is ON)."""
        # Only restore if the preview toggle is ON.
        try:
            if not (getattr(self, "chk_preview", None) and self.chk_preview.isChecked()):
                return
        except Exception:
            return

        try:
            s = _load_planner_settings()
            last_dir = str(s.get("preview_last_output_dir", "") or "").strip()
        except Exception:
            last_dir = ""

        if not last_dir or not os.path.isdir(last_dir):
            return

        # Ingest common subfolders first (most relevant), then fall back to the root.
        try:
            for sub in ("final", "clips", "images", "audio"):
                d = os.path.join(last_dir, sub)
                if os.path.isdir(d):
                    self._preview_ingest_dir(d)
            # Also scan the run root (some tools write directly into the job folder).
            self._preview_ingest_dir(last_dir)
        except Exception:
            pass

    def _preview_add(self, path: str) -> None:
        if not getattr(self, "chk_preview", None) or not self.chk_preview.isChecked():
            return
        p = str(path or "").strip()
        if not p or not os.path.exists(p):
            return

        kind = "video" if _is_video_path(p) else "image"
        # de-dupe
        try:
            self._preview_items = [it for it in (self._preview_items or []) if str(it.get("path")) != p]
        except Exception:
            self._preview_items = []
        try:
            self._preview_items.append({"path": p, "kind": kind})
        except Exception:
            self._preview_items = [{"path": p, "kind": kind}]

        # keep last 5
        try:
            if len(self._preview_items) > 5:
                self._preview_items = self._preview_items[-5:]
        except Exception:
            pass

        self._refresh_preview_strip()

    def _preview_ingest_dir(self, folder: str) -> None:
        if not getattr(self, "chk_preview", None) or not self.chk_preview.isChecked():
            return
        d = str(folder or "").strip()
        if not d or not os.path.isdir(d):
            return

        try:
            exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif")
            files = []
            for name in os.listdir(d):
                p = os.path.join(d, name)
                if os.path.isfile(p) and os.path.splitext(name.lower())[1] in exts:
                    try:
                        files.append((os.path.getmtime(p), p))
                    except Exception:
                        files.append((0.0, p))
            files.sort(key=lambda t: t[0])
            for _, fp in files[-3:]:
                self._preview_add(fp)
        except Exception:
            pass

    def _refresh_preview_strip(self) -> None:
        try:
            items = list(self._preview_items or [])
        except Exception:
            items = []

        for i, lab in enumerate(getattr(self, "_preview_thumbs", []) or []):
            try:
                lab.clear()
                lab.setVisible(False)
            except Exception:
                pass

            if i >= len(items):
                continue

            it = items[i] if isinstance(items[i], dict) else {}
            p = str(it.get("path") or "")
            k = str(it.get("kind") or "")
            if not p:
                continue

            try:
                if k == "video":
                    # Prefer a real first-frame thumbnail (looks much better than a generic icon).
                    pm = QPixmap()
                    try:
                        thumb = self._ensure_video_firstframe_thumb(p)
                        if thumb and os.path.exists(thumb):
                            pm = QPixmap(thumb)
                    except Exception:
                        pm = QPixmap()

                    if not pm.isNull():
                        lab.setPixmap(pm.scaled(lab.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        ico = self.style().standardIcon(QStyle.SP_MediaPlay)
                        lab.setPixmap(ico.pixmap(64, 36))
                else:
                    pm = QPixmap(p)
                    if not pm.isNull():
                        lab.setPixmap(pm.scaled(lab.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    else:
                        lab.setText("—")
                lab.setVisible(True)
                lab.setToolTip(p)
            except Exception:
                try:
                    lab.setVisible(True)
                except Exception:
                    pass

    def _ensure_video_firstframe_thumb(self, video_path: str) -> str:
        """Create (or reuse) a cached first-frame PNG thumbnail for a video.

        Returns the thumbnail PNG path, or "" on failure.
        """
        vp = str(video_path or "").strip()
        if not vp or not os.path.exists(vp):
            return ""

        # Cache key + mtime
        try:
            mt = float(os.path.getmtime(vp))
        except Exception:
            mt = 0.0

        try:
            cached = (self._video_thumb_cache or {}).get(vp)
            if cached and isinstance(cached, tuple) and len(cached) == 2:
                c_mt, c_path = cached
                if float(c_mt) == mt and c_path and os.path.exists(c_path):
                    return str(c_path)
        except Exception:
            pass

        # Stable output filename
        try:
            h = hashlib.sha1(vp.encode("utf-8", errors="ignore")).hexdigest()[:16]
        except Exception:
            h = "thumb"

        out_dir = str(getattr(self, "_video_thumb_dir", "") or "")
        if not out_dir:
            out_dir = str((_root() / "presets" / "setsave" / "planner_preview_thumbs").resolve())
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass

        out_png = os.path.join(out_dir, f"{h}.png")

        # Build ffmpeg command (grab a frame a tiny bit into the clip to avoid black decode frames)
        try:
            ffmpeg = self._ffmpeg_tool("ffmpeg")
        except Exception:
            ffmpeg = "ffmpeg"

        # Target is ~2x the label size for crisp downscale
        tw, th = 144, 80
        vf = f"scale={tw}:{th}:force_original_aspect_ratio=decrease,pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2"
        args = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            "0.10",
            "-i",
            vp,
            "-frames:v",
            "1",
            "-vf",
            vf,
            "-an",
            out_png,
        ]

        try:
            cp = subprocess.run(args, capture_output=True, text=True)
            if cp.returncode != 0 or not os.path.exists(out_png):
                # Fallback: try exact first frame (no seek)
                args2 = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    vp,
                    "-frames:v",
                    "1",
                    "-vf",
                    vf,
                    "-an",
                    out_png,
                ]
                cp2 = subprocess.run(args2, capture_output=True, text=True)
                if cp2.returncode != 0 or not os.path.exists(out_png):
                    return ""
        except Exception:
            return ""

        # Store cache
        try:
            self._video_thumb_cache[vp] = (mt, out_png)
        except Exception:
            pass
        return out_png

    def _on_preview_thumb_clicked(self, idx: int) -> None:
        if not getattr(self, "chk_preview", None) or not self.chk_preview.isChecked():
            return
        try:
            items = list(self._preview_items or [])
        except Exception:
            items = []
        if idx < 0 or idx >= len(items):
            return
        p = str((items[idx] or {}).get("path") or "")
        if not p:
            return

        try:
            if self._header_preview_dialog is not None:
                self._header_preview_dialog.close()
        except Exception:
            pass

        try:
            dlg = _HeaderPreviewDialog(self, p)
            self._header_preview_dialog = dlg
            dlg.exec()
        except Exception:
            pass
        self._header_preview_dialog = None


    @Slot(object)
    def _on_request_image_review(self, payload: object) -> None:
        # Gate after images: ask user whether to inspect now.
        if not (self._worker and self._worker.isRunning()):
            return
        p = payload if isinstance(payload, dict) else {}
        try:
            self._preview_ingest_dir(str(p.get("clips_dir") or ""))
        except Exception:
            pass
        try:
            self._preview_ingest_dir(str(p.get("images_dir") or ""))
        except Exception:
            pass
        try:
            resp = QMessageBox.question(
                self,
                "Inspect images now?",
                "Inspect images now?\n\nYes = open the Image Review window.\nNo = continue the pipeline.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
        except Exception:
            resp = QMessageBox.Yes

        if resp != QMessageBox.Yes:
            try:
                self._worker.post_review_command({"type": "continue"})
            except Exception:
                pass
            return

        dlg = ImageReviewDialog(self, worker=self._worker, payload=p)
        self._image_review_dialog = dlg
        dlg.exec()

        # If the dialog was closed without an explicit terminal action, continue.
        try:
            if self._worker and self._worker.isRunning() and not getattr(dlg, "_sent_terminal_cmd", False):
                self._worker.post_review_command({"type": "continue"})
        except Exception:
            pass
        self._image_review_dialog = None

    

    @Slot(object)
    def _on_request_video_review(self, payload: object) -> None:
        # Gate after clips (Chunk 8B): ask user whether to inspect clips now.
        if not (self._worker and self._worker.isRunning()):
            return
        p = payload if isinstance(payload, dict) else {}
        try:
            resp = QMessageBox.question(
                self,
                "Inspect clips now?",
                "Inspect clips now?\n\nYes = open the Clip Review window.\nNo = continue the pipeline.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
        except Exception:
            resp = QMessageBox.Yes


        if resp != QMessageBox.Yes:
            try:
                self._worker.post_review_command({"type": "continue"})
            except Exception:
                pass
            return

        # Mark reviewed for debugging/persistence
        try:
            self._worker.post_review_command({"type": "mark_reviewed"})
        except Exception:
            pass

        dlg = ClipReviewDialog(self, worker=self._worker, payload=p)
        self._clip_review_dialog = dlg
        dlg.exec()

        # If the dialog was closed without an explicit terminal action, continue.
        try:
            if self._worker and self._worker.isRunning() and not getattr(dlg, "_sent_terminal_cmd", False):
                self._worker.post_review_command({"type": "continue"})
        except Exception:
            pass
        self._clip_review_dialog = None

    @Slot(str)
    def _on_image_regen_started(self, sid: str) -> None:
        if self._image_review_dialog:
            try:
                self._image_review_dialog.on_regen_started(str(sid))
            except Exception:
                pass

    @Slot(str, str)
    def _on_image_regen_done(self, sid: str, path: str) -> None:
        if self._image_review_dialog:
            try:
                self._image_review_dialog.on_regen_done(str(sid), str(path))
            except Exception:
                pass

        try:
            self._preview_add(str(path))
        except Exception:
            pass

    @Slot(str, str)
    def _on_image_regen_failed(self, sid: str, err: str) -> None:
        if self._image_review_dialog:
            try:
                self._image_review_dialog.on_regen_failed(str(sid), str(err))
            except Exception:
                pass

    @Slot(str)
    def _on_clip_regen_started(self, sid: str) -> None:
        if self._clip_review_dialog:
            try:
                self._clip_review_dialog.on_regen_started(str(sid))
            except Exception:
                pass

    @Slot(str, str)
    def _on_clip_regen_done(self, sid: str, path: str) -> None:
        if self._clip_review_dialog:
            try:
                self._clip_review_dialog.on_regen_done(str(sid), str(path))
            except Exception:
                pass

        try:
            self._preview_add(str(path))
        except Exception:
            pass

    @Slot(str, str)
    def _on_clip_regen_failed(self, sid: str, err: str) -> None:
        if self._clip_review_dialog:
            try:
                self._clip_review_dialog.on_regen_failed(str(sid), str(err))
            except Exception:
                pass




    # -----------------------------
    # Persistent queue (survive restarts)
    # -----------------------------
    def _planner_state_path(self) -> str:
        try:
            base = _root() / "output" / "planner"
            base.mkdir(parents=True, exist_ok=True)
            return str((base / "_planner_queue_state.json").resolve())
        except Exception:
            return str((_root() / "output" / "planner" / "_planner_queue_state.json").resolve())

    def _plannerjob_from_dict(self, d: dict) -> Optional["PlannerJob"]:
        try:
            if not isinstance(d, dict):
                return None
            # Filter keys to dataclass fields (forward-compatible).
            fields = getattr(PlannerJob, "__dataclass_fields__", {}) or {}
            clean = {}
            for k in fields.keys():
                if k in d:
                    clean[k] = d.get(k)
            # Required fields safety
            if "job_id" not in clean:
                clean["job_id"] = str(uuid.uuid4())
            if "created_at" not in clean:
                clean["created_at"] = float(time.time())
            return PlannerJob(**clean)
        except Exception:
            return None

    def _serialize_job(self, job: Optional["PlannerJob"]) -> dict:
        try:
            if job is None:
                return {}
            return asdict(job)
        except Exception:
            try:
                return dict(job)  # type: ignore
            except Exception:
                return {}

    def _load_planner_queue_state(self) -> dict:
        path = self._planner_state_path()
        try:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _save_planner_queue_state(self) -> None:
        path = self._planner_state_path()
        try:
            state = {
                "version": 1,
                "saved_at": float(time.time()),
                "running": getattr(self, "_persist_running", None),
                "pending": [],
            }
            pending = getattr(self, "_pending_jobs", []) or []
            for item in pending:
                try:
                    j = item.get("job")
                    state["pending"].append({
                        "job": self._serialize_job(j),
                        "out_dir": str(item.get("out_dir") or ""),
                        "title": str(item.get("title") or ""),
                        "slug": str(item.get("slug") or ""),
                    })
                except Exception:
                    continue
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            try:
                os.replace(tmp, path)
            except Exception:
                shutil.copyfile(tmp, path)
        except Exception:
            pass

    def _scan_for_external_running(self) -> Optional[dict]:
        """Best-effort recovery if state file is missing/corrupt.
        Detect an FV 'queue lock' job by looking for a Planner project folder that has
        _planner_lock_job.txt but no _planner_done.flag yet.
        """
        # Standalone mode: do not attempt to recover/guard against FrameVision queue locks.
        try:
            if not bool(getattr(self, "_use_framevision_queue", True)):
                return None
        except Exception:
            pass
        # Prefer syncing against the FrameVision queue folders using the lock file metadata.
        try:
            best = self._external_best_active()
            if best:
                return best
        except Exception:
            pass

        try:
            base = _root() / "output" / "planner"
            if not base.exists():
                return None
            best = None
            best_m = 0.0
            for p in base.iterdir():
                try:
                    if not p.is_dir():
                        continue
                    lock_txt = p / "_planner_lock_job.txt"
                    done_flag = p / "_planner_done.flag"
                    if lock_txt.exists() and (not done_flag.exists()):
                        mt = float(p.stat().st_mtime)
                        if mt > best_m:
                            best_m = mt
                            best = {
                                "out_dir": str(p.resolve()),
                                "done_flag": str(done_flag.resolve()),
                                "active_flag": str((p / "_planner_active.flag").resolve()),
                            }
                except Exception:
                    continue
            return best
        except Exception:
            return None


    def _fv_guess_queue_dirs(self, pending_dir: str) -> dict:
        """Given the FrameVision 'pending' folder, try to infer sibling dirs for running/finished."""
        out = {}
        try:
            p = pathlib.Path(str(pending_dir)).resolve()
            out["pending"] = str(p)
            base = p.parent
            # Common sibling names used across versions.
            for name in ("running", "run", "active"):
                d = base / name
                if d.exists():
                    out["running"] = str(d)
                    break
            for name in ("finished", "done", "complete", "completed", "finished_jobs"):
                d = base / name
                if d.exists():
                    out["finished"] = str(d)
                    break
            # Fallbacks even if they don't exist yet.
            out.setdefault("running", str(base / "running"))
            out.setdefault("finished", str(base / "finished"))
        except Exception:
            pass
        return out

    def _fv_job_marker_exists(self, folder: str, job_id: str) -> bool:
        """Best-effort check whether a given job_id is represented in a queue folder."""
        try:
            if not folder or not job_id:
                return False
            d = pathlib.Path(folder)
            if not d.exists():
                return False
            # Fast-path common file/folder patterns.
            candidates = [
                d / f"{job_id}.json",
                d / f"{job_id}.txt",
                d / f"{job_id}",
            ]
            for c in candidates:
                if c.exists():
                    return True
            # Controlled glob fallback (avoid expensive recursive scans).
            for pat in (f"{job_id}.*", f"*{job_id}*.json", f"*{job_id}*"):
                for _ in d.glob(pat):
                    return True
            return False
        except Exception:
            return False

    def _parse_planner_lock_file(self, lock_path: str) -> dict:
        out = {}
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                for line in f.read().splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        out[str(k).strip()] = str(v).strip()
        except Exception:
            pass
        return out

    def _scan_for_external_queue_items(self) -> list:
        """Scan Planner output folders for jobs that were enqueued into the FrameVision queue.
        Returns a list of dicts with status: pending/running/finished/unknown.
        """
        items = []
        # Standalone mode: do not attempt to recover/guard against FrameVision queue locks.
        try:
            if not bool(getattr(self, "_use_framevision_queue", True)):
                return items
        except Exception:
            pass
        try:
            base = _root() / "output" / "planner"
            if not base.exists():
                return items
            for p in base.iterdir():
                try:
                    if not p.is_dir():
                        continue
                    lock_txt = p / "_planner_lock_job.txt"
                    done_flag = p / "_planner_done.flag"
                    if not lock_txt.exists():
                        continue
                    info = self._parse_planner_lock_file(str(lock_txt))
                    job_id = info.get("job_id", "")
                    pending_dir = info.get("pending_dir", "")
                    df = info.get("done_flag", "") or str(done_flag)
                    if df and os.path.exists(df):
                        status = "finished"
                    elif done_flag.exists():
                        status = "finished"
                    else:
                        qdirs = self._fv_guess_queue_dirs(pending_dir)
                        if self._fv_job_marker_exists(qdirs.get("running", ""), job_id):
                            status = "running"
                        elif self._fv_job_marker_exists(qdirs.get("pending", ""), job_id):
                            status = "pending"
                        elif self._fv_job_marker_exists(qdirs.get("finished", ""), job_id):
                            status = "finished"
                        else:
                            status = "unknown"
                    items.append({
                        "out_dir": str(p.resolve()),
                        "job_id": job_id,
                        "pending_dir": pending_dir,
                        "done_flag": df,
                        "status": status,
                        "mtime": float(p.stat().st_mtime) if p.exists() else 0.0,
                    })
                except Exception:
                    continue
        except Exception:
            pass
        return items

    def _external_queue_count(self) -> int:
        """Count external Planner-lock jobs that are still pending/running/unknown in the FV queue."""
        try:
            items = self._scan_for_external_queue_items() or []
            n = 0
            for it in items:
                st = str(it.get("status") or "")
                if st in ("pending", "running", "unknown"):
                    n += 1
            return int(n)
        except Exception:
            return 0

    def _external_best_active(self) -> Optional[dict]:
        """Pick the most recent external active item to display as a recovered run state."""
        try:
            items = self._scan_for_external_queue_items() or []
            best = None
            best_m = 0.0
            for it in items:
                st = str(it.get("status") or "")
                if st in ("pending", "running", "unknown"):
                    mt = float(it.get("mtime") or 0.0)
                    if mt > best_m:
                        best_m = mt
                        out_dir = str(it.get("out_dir") or "")
                        best = {
                            "out_dir": out_dir,
                            "done_flag": str(it.get("done_flag") or ""),
                            "active_flag": str((pathlib.Path(out_dir) / "_planner_active.flag").resolve()),
                        }
            return best
        except Exception:
            return None

    def _is_planner_busy(self) -> bool:
        # Busy if the in-process worker is running OR we recovered an external FV-locked run.
        try:
            if self._worker and self._worker.isRunning():
                return True
        except Exception:
            pass
        # Standalone mode: ignore any recovered external FV lock.
        try:
            if bool(getattr(self, "_use_framevision_queue", True)):
                info = getattr(self, "_external_running", None)
                if isinstance(info, dict) and info.get("done_flag"):
                    return not os.path.exists(str(info.get("done_flag")))
        except Exception:
            pass
        return False

    def _set_external_running(self, info: Optional[dict]) -> None:
        try:
            self._external_running = info if isinstance(info, dict) else None
        except Exception:
            self._external_running = None

        # Reflect in header (non-scary, but clear).
        try:
            if self._external_running:
                qn = self._queue_count()
                self._set_header_status(f"Running (recovered) — queued: {qn}", mode="running")
            else:
                # Keep current header if an in-process run is active; otherwise show Idle.
                try:
                    if not (self._worker and self._worker.isRunning()):
                        qn = 0
                        try:
                            qn = self._queue_count()
                        except Exception:
                            qn = 0
                        _qtxt = f" — queued: {qn}" if qn > 0 else ""
                        self._set_header_status("Idle" + _qtxt, mode="idle")
                except Exception:
                    qn = 0
                    try:
                        qn = self._queue_count()
                    except Exception:
                        qn = 0
                    _qtxt = f" — queued: {qn}" if qn > 0 else ""
                    self._set_header_status("Idle" + _qtxt, mode="idle")
        except Exception:
            pass

    def _poll_external_running(self) -> None:
        try:
            info = getattr(self, "_external_running", None)
            if not isinstance(info, dict):
                return
            done_flag = str(info.get("done_flag") or "").strip()
            if done_flag and os.path.exists(done_flag):
                # External run finished; clear and continue with our queued jobs.
                self._append_log("----")
                self._append_log("[QUEUE] Recovered run finished (done flag detected).")
                self._set_external_running(None)
                try:
                    self._persist_running = None
                except Exception:
                    pass
                try:
                    self._save_planner_queue_state()
                except Exception:
                    pass
                self._maybe_start_next_queued()
        except Exception:
            pass

    def _init_persistent_queue(self) -> None:
        # Load saved queue + running guard.
        state = self._load_planner_queue_state()
        pending_loaded = []
        running = None

        try:
            running = state.get("running")
        except Exception:
            running = None

        # Restore pending jobs
        try:
            for item in (state.get("pending") or []):
                try:
                    jd = item.get("job") or {}
                    job = self._plannerjob_from_dict(jd)
                    if not job:
                        continue
                    pending_loaded.append({
                        "job": job,
                        "out_dir": str(item.get("out_dir") or ""),
                        "title": str(item.get("title") or ""),
                        "slug": str(item.get("slug") or ""),
                    })
                except Exception:
                    continue
        except Exception:
            pass

        try:
            self._pending_jobs = pending_loaded
        except Exception:
            pass

        # Validate running guard (if done_flag already exists, clear it)
        ext = None
        try:
            if isinstance(running, dict):
                done_flag = str(running.get("done_flag") or "")
                if done_flag and os.path.exists(done_flag):
                    running = None
                else:
                    ext = running
        except Exception:
            ext = None

        if ext is None:
            try:
                if bool(getattr(self, "_use_framevision_queue", True)):
                    ext = self._scan_for_external_running()
            except Exception:
                ext = None

        # Track for persistence + polling
        try:
            self._persist_running = ext
        except Exception:
            self._persist_running = ext

        self._set_external_running(ext)

        # Start a tiny poller if we have an external running guard.
        try:
            if getattr(self, "_external_running", None):
                if not hasattr(self, "_external_poll_timer") or self._external_poll_timer is None:
                    self._external_poll_timer = QTimer(self)
                    self._external_poll_timer.setInterval(1500)
                    self._external_poll_timer.timeout.connect(self._poll_external_running)
                try:
                    self._external_poll_timer.start()
                except Exception:
                    pass
        except Exception:
            pass

        # Refresh header status text if needed
        try:
            if getattr(self, "_external_running", None):
                self._update_header_running_text()
        except Exception:
            pass

    def _queue_count(self) -> int:
        """Planner mini-queue count + any externally-queued Planner lock jobs in the FrameVision queue.
        This prevents the Planner from showing 'Idle' (and thinking everything is done) after a restart
        while jobs are still pending/running in the FrameVision worker queue.
        """
        own = 0
        try:
            own = int(len(getattr(self, "_pending_jobs", []) or []))
        except Exception:
            own = 0
        ext = 0
        try:
            ext = int(self._external_queue_count() or 0)
        except Exception:
            ext = 0
        return int(own + ext)

    def _enqueue_job(self, job: "PlannerJob", out_dir: str, title: str = "", slug: str = "") -> None:
        try:
            if not hasattr(self, "_pending_jobs") or self._pending_jobs is None:
                self._pending_jobs = []
        except Exception:
            self._pending_jobs = []
        try:
            self._pending_jobs.append({
                "job": job,
                "out_dir": str(out_dir),
                "title": str(title or ""),
                "slug": str(slug or ""),
            })
        except Exception:
            pass
        try:
            qn = self._queue_count()
            self._append_log(f"[QUEUE] Added job {getattr(job,'job_id','')} → {out_dir} (queued: {qn})")
            try:
                self._save_planner_queue_state()
            except Exception:
                pass
        except Exception:
            pass
        try:
            # Update running header line to show queued count
            if bool(getattr(self, "_running", False)):
                self._update_header_running_text()
        except Exception:
            pass

    def _maybe_start_next_queued(self) -> None:
        """Start the next queued job (if any) after a run finishes/fails."""
        try:
            running = bool(getattr(self, "_running", False))
        except Exception:
            running = False
        if running:
            return
        try:
            q = getattr(self, "_pending_jobs", []) or []
        except Exception:
            q = []
        if not q:
            return
        try:
            nxt = q.pop(0)
        except Exception:
            return
        try:
            self._save_planner_queue_state()
        except Exception:
            pass
        try:
            job = nxt.get("job")
            out_dir = str(nxt.get("out_dir") or "")
            title = str(nxt.get("title") or "")
            slug = str(nxt.get("slug") or "")
            if job and out_dir:
                self._append_log("----")
                self._append_log(f"[QUEUE] Starting next queued job → {out_dir}")
                self._run_job(job, out_dir, title=title, slug=slug)
        except Exception as e:
            try:
                self._append_log(f"[QUEUE] Failed to start next job: {e}")
            except Exception:
                pass

    def _run_job(self, job: "PlannerJob", out_dir: str, title: str = "", slug: str = "", resume_note: str = "") -> None:
        """Start a planner run immediately (single active worker)."""
        # Optional: run Planner under FrameVision queue "lock" so other queued tools wait their turn.
        # IMPORTANT: Only enqueue the lock when the job actually starts (avoids deadlocking the FV worker).
        #
        # IMPORTANT: The Settings toggle must be authoritative:
        # - If Settings → Use FrameVision Queue is OFF, do NOT enqueue anything into the FrameVision queue,
        #   even if FV_PLANNER_QUEUE_LOCK is set in the environment.
        # - If Settings is ON, FV_PLANNER_QUEUE_LOCK may still explicitly force OFF/ON.
        use_queue_lock_env = str(os.environ.get("FV_PLANNER_QUEUE_LOCK", "") or "").strip().lower()

        # Master switch: Settings OFF always disables queue lock.
        use_queue_lock_setting = bool(getattr(self, "_use_framevision_queue", True))
        if not use_queue_lock_setting:
            use_queue_lock = False
        else:
            if use_queue_lock_env in ("0", "false", "no", "off"):
                use_queue_lock = False
            elif use_queue_lock_env in ("1", "true", "yes", "on"):
                use_queue_lock = True
            else:
                use_queue_lock = True


        self._queue_lock_done_flag = ""
        if use_queue_lock:
            try:
                from helpers import queue_adapter as _qa
                os.makedirs(out_dir, exist_ok=True)
                done_flag = os.path.join(out_dir, "_planner_done.flag")
                try:
                    if os.path.exists(done_flag):
                        os.remove(done_flag)
                except Exception:
                    pass

                args = {
                    "done_flag": done_flag,
                    "scan_dir": out_dir,
                    "scan_ext": ".mp4",
                    "log_tail_lines": 120,
                }
                _script = (
                    "import os,time,glob;"
                    f"done_flag=r'''{done_flag}''';"
                    f"scan_dir=r'''{out_dir}''';"
                    "scan_ext='.mp4';"
                    "while not os.path.exists(done_flag): time.sleep(0.5);"
                    "time.sleep(0.5);"
                    "c=glob.glob(os.path.join(scan_dir,'*'+scan_ext));"
                    "print(max(c,key=lambda p: os.path.getmtime(p)) if c else '')"
                )
                _cmd = [sys.executable, "-c", _script]
                _job_id = _qa.enqueue_tool_job(
                    "tools_ffmpeg",
                    input_path="",
                    out_dir=out_dir,
                    args=dict(args, **{"label": "Planner (queue lock)", "cmd": _cmd, "cwd": os.path.abspath(os.getcwd())}),
                    priority=120
                )
                try:
                    _pending_dir = _qa.jobs_dirs().get('pending')
                    _LOGGER.log_probe(f"planner_lock enqueued: id={_job_id} pending={_pending_dir} out_dir={out_dir}")
                    with open(os.path.join(out_dir, "_planner_lock_job.txt"), "w", encoding="utf-8") as _f:
                        _f.write(f"job_id={_job_id}\n")
                        _f.write(f"pending_dir={_pending_dir}\n")
                        _f.write(f"done_flag={done_flag}\n")
                except Exception:
                    pass

                self._queue_lock_done_flag = done_flag
            except Exception as e:
                try:
                    _LOGGER.log_probe(f"planner_lock enqueue failed: {e}")
                except Exception:
                    pass

        
        # Persist "running" guard so a Planner restart cannot start a second job while FV is still locked.
        try:
            active_flag = os.path.join(out_dir, "_planner_active.flag")
            try:
                Path(active_flag).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
            except Exception:
                try:
                    with open(active_flag, "w", encoding="utf-8") as f:
                        f.write("active")
                except Exception:
                    pass

            info = {
                "out_dir": str(out_dir),
                "job_id": str(getattr(job, "job_id", "") or ""),
                "title": str(title or ""),
                "slug": str(slug or ""),
                "started_at": float(time.time()),
                "done_flag": str(getattr(self, "_queue_lock_done_flag", "") or ""),
                "active_flag": str(active_flag),
            }
            self._persist_running = info
            self._set_external_running(info)
            self._save_planner_queue_state()
        except Exception:
            pass

# Preview thumbs: reset immediately when a new job starts (prevents stale thumbs).
        try:
            self._preview_reset_for_new_job(out_dir)
        except Exception:
            pass

        # Persist Planner-only upscaling settings into this project folder (no execution yet)
        try:
            self._active_project_dir = out_dir
            self._persist_upscale_settings_for_project(out_dir, force_write=False)
        except Exception:
            pass

        # Keep internal identity for manifests / resume
        try:
            if title:
                job.encoding.setdefault("title", title)
            if slug:
                job.encoding.setdefault("slug", slug)
        except Exception:
            pass

        # Reset UI for the new job
        try:
            self.progress.setValue(0)
            self.lbl_stage.setText("Stage: Starting")
        except Exception:
            pass
        try:
            self._header_stage = "Starting"
            self._header_pct = 0
        except Exception:
            pass
        try:
            self._set_header_status("Running: Starting", mode="running")
        except Exception:
            pass
        try:
            self.log.clear()
        except Exception:
            pass
        try:
            self._last_result = None
            self.btn_open_output.setEnabled(False)
        except Exception:
            pass

        self._set_running(True)

        self._worker = PipelineWorker(job, out_dir)
        self._worker.signals.log.connect(self._append_log)
        self._worker.signals.stage.connect(self._on_worker_stage)
        self._worker.signals.progress.connect(self._on_worker_progress)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.failed.connect(self._on_failed)
        self._worker.signals.request_image_review.connect(self._on_request_image_review)
        self._worker.signals.request_video_review.connect(self._on_request_video_review)
        self._worker.signals.image_regen_started.connect(self._on_image_regen_started)
        self._worker.signals.image_regen_done.connect(self._on_image_regen_done)
        self._worker.signals.image_regen_failed.connect(self._on_image_regen_failed)
        self._worker.signals.clip_regen_started.connect(self._on_clip_regen_started)
        self._worker.signals.clip_regen_done.connect(self._on_clip_regen_done)
        self._worker.signals.clip_regen_failed.connect(self._on_clip_regen_failed)
        self._worker.signals.asset_created.connect(self._on_asset_created)

        try:
            if resume_note:
                self._append_log(resume_note)
        except Exception:
            pass

        try:
            self._append_log("Starting pipeline…")
        except Exception:
            pass

        try:
            self._worker.start()
        except Exception:
            self._set_running(False)
            raise


    @Slot()
    def _cancel_pipeline(self) -> None:
        if self._worker and self._worker.isRunning():
            self._append_log("Cancel requested…")
            try:
                self._header_stage = "Cancelling…"
            except Exception:
                pass
            self._set_header_status("Running: Cancelling…", mode="running")
            self._worker.request_stop()

    @Slot(dict)
    def _on_finished(self, result: dict) -> None:
        self._last_result = result
        try:
            self._active_project_dir = str(result.get("output_dir") or "")
        except Exception:
            pass

        try:
            self._preview_add(str(result.get("final_video") or ""))
        except Exception:
            pass

        self._append_log("----")
        self._append_log("Done.")
        self._append_log(f"Final video: {result.get('final_video')}")
        # If we used a queue lock, signal the worker that the planner run is finished.
        try:
            flag = getattr(self, "_queue_lock_done_flag", "") or ""
            if flag:
                # touch the flag file
                try:
                    Path(flag).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
                except Exception:
                    try:
                        with open(flag, "w", encoding="utf-8") as f:
                            f.write("done")
                    except Exception:
                        pass
        except Exception:
            pass

        # Clear persisted running guard
        try:
            info = getattr(self, "_persist_running", None)
            if isinstance(info, dict):
                af = str(info.get("active_flag") or "")
                if af and os.path.exists(af):
                    try:
                        os.remove(af)
                    except Exception:
                        pass
            self._persist_running = None
        except Exception:
            pass
        try:
            self._set_external_running(None)
        except Exception:
            pass
        try:
            self._save_planner_queue_state()
        except Exception:
            pass


        self._set_header_status("Done", mode="done")
        self._set_running(False)
        self.btn_open_output.setEnabled(True)

        # Auto-run next queued job (if any)
        self._maybe_start_next_queued()

    @Slot(str)
    def _on_failed(self, err: str) -> None:
        self._append_log("----")
        self._append_log(f"FAILED: {err}")

        is_cancel = False
        try:
            is_cancel = ("cancel" in str(err).lower()) or ("aborted" in str(err).lower())
        except Exception:
            is_cancel = False

        # If we used a queue lock, signal the worker that the planner run is finished.
        try:
            flag = getattr(self, "_queue_lock_done_flag", "") or ""
            if flag:
                # touch the flag file
                try:
                    Path(flag).write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
                except Exception:
                    try:
                        with open(flag, "w", encoding="utf-8") as f:
                            f.write("done")
                    except Exception:
                        pass
        except Exception:
            pass

        # Clear persisted running guard
        try:
            info = getattr(self, "_persist_running", None)
            if isinstance(info, dict):
                af = str(info.get("active_flag") or "")
                if af and os.path.exists(af):
                    try:
                        os.remove(af)
                    except Exception:
                        pass
            self._persist_running = None
        except Exception:
            pass
        try:
            self._set_external_running(None)
        except Exception:
            pass
        try:
            self._save_planner_queue_state()
        except Exception:
            pass


        self._set_header_status("Done", mode="done")
        self._set_running(False)

        # Don't scare the user with an error popup for a normal Cancel action.
        if not is_cancel:
            QMessageBox.warning(self, "Pipeline failed", err)

        # Auto-run next queued job (if any)
        self._maybe_start_next_queued()

    @Slot()
    def _export_job_json(self) -> None:
        try:
            job = self._build_job()
        except Exception as e:
            QMessageBox.warning(self, "Cannot export", str(e))
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save job JSON", f"planner_job_{job.job_id}.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(job.to_json())
            QMessageBox.information(self, "Saved", f"Job JSON saved:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))

    @Slot()
    def _open_output_folder(self) -> None:
        if not self._last_result:
            return
        out_dir = self._last_result.get("output_dir", "")
        if not out_dir:
            return
        # Cross-platform open folder
        try:
            if os.name == "nt":
                os.startfile(out_dir)  # type: ignore[attr-defined]
            elif os.name == "posix":
                import subprocess
                subprocess.Popen(["xdg-open", out_dir])
            else:
                QMessageBox.information(self, "Output folder", out_dir)
        except Exception:
            QMessageBox.information(self, "Output folder", out_dir)

class ImageReviewDialog(QDialog):
    """Chunk 8A: Review all generated images and optionally regenerate individual shots with edited prompts."""

    def __init__(self, parent: QWidget, *, worker: "PipelineWorker", payload: Dict[str, Any]):
        super().__init__(parent)
        self.setWindowTitle("Image Review")
        self.setModal(True)
        self._worker = worker
        self._payload = dict(payload or {})
        self._sent_terminal_cmd = False
        self._current_sid: str = ""
        self._regen_busy: bool = False

        # Slightly larger default window so everything has breathing room
        self.resize(1180, 760)
        self.setMinimumSize(980, 640)

        self._manifest_path = str(self._payload.get("manifest_path") or "")
        self._images_dir = str(self._payload.get("images_dir") or "")
        # For resume / history workflows: we may need to invalidate / regenerate clips
        # after an image was regenerated.
        self._clips_dir = str(self._payload.get("clips_dir") or "")

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        top = QLabel("Inspect images now. Select a shot, edit its prompt, then click Recreate. When done, click Continue pipeline.")
        top.setWordWrap(True)
        root.addWidget(top)

        body = QHBoxLayout()
        body.setSpacing(10)

        self.lst = QListWidget()
        self.lst.setViewMode(QListWidget.IconMode)
        self.lst.setResizeMode(QListWidget.Adjust)
        self.lst.setMovement(QListWidget.Static)
        self.lst.setIconSize(QSize(256, 144))
        self.lst.setSpacing(8)
        self.lst.setMinimumWidth(520)
        self.lst.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        body.addWidget(self.lst, 2)

        right = QVBoxLayout()
        right.setSpacing(8)

        self.lbl_sel = QLabel("Selected: —")
        right.addWidget(self.lbl_sel)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Prompt used for this image (editable)…")
        self.prompt_edit.setMinimumHeight(120)
        right.addWidget(self.prompt_edit, 0)

        self.chk_retry_new_seed = QCheckBox("Retry with new seed")
        self.chk_retry_new_seed.setToolTip("When enabled, Recreate will pick a new random seed for this image.")
        right.addWidget(self.chk_retry_new_seed)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_recreate = QPushButton("Recreate")
        self.btn_reset = QPushButton("Reset prompt")
        btn_row.addWidget(self.btn_recreate)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)
        right.addLayout(btn_row)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        self.btn_continue = QPushButton("Continue pipeline")
        self.btn_cancel_job = QPushButton("Cancel job")
        bottom_row.addWidget(self.btn_continue)
        bottom_row.addWidget(self.btn_cancel_job)
        bottom_row.addStretch(1)
        right.addLayout(bottom_row)

        # If we're delegating to the FrameVision queue, cancellation is handled there.
        # Hide the Cancel button in review popups to avoid confusing/incorrect behavior.
        try:
            if bool(getattr(parent, "_use_framevision_queue", True)):
                self.btn_cancel_job.hide()
        except Exception:
            pass

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        right.addWidget(self.lbl_status)

        body.addLayout(right, 1)
        root.addLayout(body, 1)

        self.btn_recreate.clicked.connect(self._on_recreate)
        self.btn_reset.clicked.connect(self._on_reset)
        self.btn_continue.clicked.connect(self._on_continue)
        self.btn_cancel_job.clicked.connect(self._on_cancel_job)
        self.lst.currentItemChanged.connect(self._on_select)

        self._load_items()
        try:
            QTimer.singleShot(0, self._update_thumb_layout)
        except Exception:
            pass

    def closeEvent(self, ev):  # type: ignore[override]
        if not self._sent_terminal_cmd:
            try:
                self._worker.post_review_command({"type": "continue"})
            except Exception:
                pass
            self._sent_terminal_cmd = True
        super().closeEvent(ev)

    def _read_manifest(self) -> Dict[str, Any]:
        m = _safe_read_json(self._manifest_path) if self._manifest_path else None
        return m if isinstance(m, dict) else {}

    def _resolve_images(self, manifest: Dict[str, Any]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []

        try:
            items = (manifest.get("paths") or {}).get("images") or []
        except Exception:
            items = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("id") and it.get("file"):
                    out.append({"id": str(it.get("id")), "file": str(it.get("file"))})

        if not out:
            shots = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
            if isinstance(shots, dict):
                for sid, rec in shots.items():
                    if isinstance(rec, dict) and rec.get("file"):
                        out.append({"id": str(sid), "file": str(rec.get("file"))})

        if not out and self._images_dir and os.path.isdir(self._images_dir):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
                for p in Path(self._images_dir).glob(ext):
                    out.append({"id": p.stem, "file": str(p)})

        out.sort(key=lambda d: d.get("id", ""))
        return out

    def _load_items(self) -> None:
        self.lst.clear()
        self._item_by_sid: Dict[str, QListWidgetItem] = {}

        manifest = self._read_manifest()
        images = self._resolve_images(manifest)
        shots = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}

        for it in images:
            sid = str(it.get("id") or "").strip()
            fp = str(it.get("file") or "").strip()
            if not sid:
                continue

            icon = QIcon()
            if fp and os.path.exists(fp):
                try:
                    pm = QPixmap(fp)
                    if not pm.isNull():
                        icon = QIcon(pm)
                except Exception:
                    pass

            item = QListWidgetItem(icon, sid)
            item.setData(Qt.UserRole, {"id": sid, "file": fp})
            try:
                rec = shots.get(sid) if isinstance(shots, dict) else {}
                if isinstance(rec, dict):
                    tip = str(rec.get("prompt_compiled") or rec.get("prompt_used") or "").strip()
                    if tip:
                        item.setToolTip(tip)
            except Exception:
                pass

            self.lst.addItem(item)
            self._item_by_sid[sid] = item

        if self.lst.count() > 0:
            self.lst.setCurrentRow(0)
        try:
            self._update_thumb_layout()
        except Exception:
            pass



    def resizeEvent(self, ev):  # type: ignore[override]
        super().resizeEvent(ev)
        try:
            self._update_thumb_layout()
        except Exception:
            pass

    def _update_thumb_layout(self) -> None:
        """Scale thumbnails to use the available panel space (responsive grid).

        Thumbnails are sized primarily from the available *width* and we allow vertical
        scrolling. This avoids the common "tiny thumbs with lots of unused black space" issue
        that happens when the layout tries to fit all rows into the current viewport height.
        """
        try:
            n = int(self.lst.count() or 0)
            if n <= 0:
                return

            vw = int(self.lst.viewport().width() or 0)
            if vw <= 0:
                return

            spacing = int(self.lst.spacing() or 8)
            pad = 16

            # Choose column count based on width, then fill the width with big thumbs.
            # (Allow scrolling vertically; do NOT shrink thumbs just to fit the viewport height.)
            desired_min_w = 420
            cols = int(vw / max(1, (desired_min_w + spacing)))
            cols = max(1, min(cols, 3))

            label_h = 34
            avail_w = max(1, vw - (cols + 1) * spacing - pad)
            icon_w = int(avail_w / cols)
            icon_h = int(icon_w * 9 / 16)

            # Clamp to reasonable bounds.
            icon_w = max(320, min(icon_w, 980))
            icon_h = max(180, min(icon_h, 560))

            self.lst.setIconSize(QSize(icon_w, icon_h))
            # Grid size includes a little extra for padding + label.
            self.lst.setGridSize(QSize(icon_w + 34, icon_h + label_h + 16))
        except Exception:
            pass
    def _selected_sid(self) -> str:
        it = self.lst.currentItem()
        if not it:
            return ""
        try:
            data = it.data(Qt.UserRole) or {}
        except Exception:
            data = {}
        if isinstance(data, dict) and data.get("id"):
            return str(data.get("id"))
        return str(it.text() or "")

    def _on_select(self, _cur: Optional[QListWidgetItem], _prev: Optional[QListWidgetItem]) -> None:
        sid = self._selected_sid()
        self._current_sid = sid
        self.lbl_sel.setText(f"Selected: {sid or '—'}")

        manifest = self._read_manifest()
        shots = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
        rec = shots.get(sid) if isinstance(shots, dict) else {}
        if not isinstance(rec, dict):
            rec = {}

        prompt = str(rec.get("prompt_compiled") or rec.get("prompt_used") or "").strip()
        self.prompt_edit.blockSignals(True)
        self.prompt_edit.setPlainText(prompt)
        self.prompt_edit.blockSignals(False)

    def _on_reset(self) -> None:
        sid = self._selected_sid()
        if not sid:
            return
        manifest = self._read_manifest()
        shots = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
        rec = shots.get(sid) if isinstance(shots, dict) else {}
        if not isinstance(rec, dict):
            rec = {}
        base = str(rec.get("prompt_compiled_original") or rec.get("prompt_spec") or rec.get("prompt_compiled") or "").strip()
        if base:
            self.prompt_edit.setPlainText(base)
            self.lbl_status.setText("Prompt reset locally. Click Recreate to apply.")

    def _on_recreate(self) -> None:
        if self._regen_busy:
            return
        sid = self._selected_sid()
        if not sid:
            return
        new_prompt = (self.prompt_edit.toPlainText() or "").strip()
        if not new_prompt:
            QMessageBox.warning(self, "Empty prompt", "Prompt is empty.")
            return

        self._regen_busy = True
        self._set_busy(True, f"Regenerating {sid}…")

        seed_override = None
        try:
            if hasattr(self, "chk_retry_new_seed") and self.chk_retry_new_seed.isChecked():
                seed_override = int(random.SystemRandom().randint(1, 2147483647))
        except Exception:
            seed_override = None

        try:
            cmd = {"type": "regen", "sid": sid, "prompt": new_prompt}
            if seed_override is not None:
                cmd["seed"] = int(seed_override)
            self._worker.post_review_command(cmd)
        except Exception as e:
            self._regen_busy = False
            self._set_busy(False, f"Failed to send regen command: {e}")

    def _on_continue(self) -> None:
        if self._regen_busy:
            QMessageBox.information(self, "Busy", "Wait for regeneration to finish first.")
            return
        try:
            self._worker.post_review_command({"type": "continue"})
        except Exception:
            pass
        self._sent_terminal_cmd = True
        self.accept()

    def _on_cancel_job(self) -> None:
        try:
            self._worker.post_review_command({"type": "cancel"})
        except Exception:
            pass
        try:
            self._worker.request_stop()
        except Exception:
            pass
        self._sent_terminal_cmd = True
        self.reject()

    def _set_busy(self, busy: bool, status: str = "") -> None:
        self.btn_recreate.setEnabled(not busy)
        self.btn_reset.setEnabled(not busy)
        self.btn_continue.setEnabled(not busy)
        self.btn_cancel_job.setEnabled(True)
        if status:
            self.lbl_status.setText(status)

    def on_regen_started(self, sid: str) -> None:
        self._set_busy(True, f"Regenerating {sid}…")

    def on_regen_done(self, sid: str, path: str) -> None:
        self._regen_busy = False
        try:
            it = self._item_by_sid.get(str(sid))
            if it and path and os.path.exists(path):
                pm = QPixmap(path)
                if not pm.isNull():
                    it.setIcon(QIcon(pm))
                it.setData(Qt.UserRole, {"id": str(sid), "file": str(path)})
                it.setToolTip((self.prompt_edit.toPlainText() or "").strip())
        except Exception:
            pass

        # If this image was regenerated during a resume, the existing clip (if any)
        # is now potentially stale. Offer to delete it so the workflow regenerates it.
        try:
            self._offer_clip_regen_for_image(str(sid))
        except Exception:
            pass
        self._set_busy(False, f"Done: {sid}")

    def _resolve_clip_for_sid(self, sid: str) -> str:
        """Find an existing clip file for this shot id (best-effort)."""
        sid = str(sid or "").strip()
        if not sid:
            return ""

        # 1) manifest paths.clips
        try:
            manifest = self._read_manifest()
            items = (manifest.get("paths") or {}).get("clips") or []
            if isinstance(items, list):
                for it in items:
                    if isinstance(it, dict) and str(it.get("id") or "").strip() == sid:
                        fp = str(it.get("file") or "").strip()
                        if fp and os.path.isfile(fp):
                            return fp
        except Exception:
            pass

        # 2) conventional <clips_dir>/<sid>.mp4
        try:
            if self._clips_dir and os.path.isdir(self._clips_dir):
                cand = os.path.join(self._clips_dir, f"{sid}.mp4")
                if os.path.isfile(cand):
                    return cand
                # 3) legacy naming: shot_###_<sid>.mp4
                cands = sorted([str(p) for p in Path(self._clips_dir).glob(f"*_{sid}.mp4") if p.is_file()])
                if cands:
                    return cands[-1]
        except Exception:
            pass
        return ""

    def _invalidate_clips_cache(self) -> None:
        """Prevent clip stage from skipping due to a stale fingerprint/manifest.

        The clip stage may fast-skip based on clips_manifest.json and a coarse
        directory fingerprint. When an image is overwritten in-place, the images
        directory mtime may not change (notably on Windows), so we explicitly
        invalidate the clip manifest here when the user wants a new clip.
        """
        try:
            if not self._clips_dir:
                return
            cm = os.path.join(self._clips_dir, "clips_manifest.json")
            if os.path.exists(cm):
                os.remove(cm)
        except Exception:
            pass

        # Also create a "touch" marker in images_dir by delete+create to force a
        # directory mtime update in the common case.
        try:
            if not self._images_dir or not os.path.isdir(self._images_dir):
                return
            touch = os.path.join(self._images_dir, "_regen_touch.flag")
            try:
                if os.path.exists(touch):
                    os.remove(touch)
            except Exception:
                pass
            try:
                with open(touch, "w", encoding="utf-8") as f:
                    f.write(str(time.time()))
            except Exception:
                pass
        except Exception:
            pass

    def _remove_clip_from_manifest(self, sid: str) -> None:
        sid = str(sid or "").strip()
        if not sid or not self._manifest_path:
            return
        try:
            manifest = self._read_manifest()
            paths = manifest.get("paths") if isinstance(manifest.get("paths"), dict) else {}
            clips = paths.get("clips") if isinstance(paths.get("clips"), list) else []
            if clips:
                new = []
                for it in clips:
                    if isinstance(it, dict) and str(it.get("id") or "").strip() == sid:
                        continue
                    new.append(it)
                paths["clips"] = new
                manifest["paths"] = paths
                _safe_write_json(self._manifest_path, manifest)
        except Exception:
            pass

    def _offer_clip_regen_for_image(self, sid: str) -> None:
        sid = str(sid or "").strip()
        if not sid:
            return
        clip = self._resolve_clip_for_sid(sid)
        if not clip:
            return
        if not os.path.isfile(clip):
            return

        # Popup: image was recreated, offer to create a new clip.
        try:
            ans = QMessageBox.question(
                self,
                "Image recreated",
                "Image was re-created. Do you want to create a new clip with this image?\n\n"
                "Yes: delete the existing clip so the workflow can recreate it.\n"
                "No: keep the existing clip.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
        except Exception:
            ans = QMessageBox.Yes

        if ans != QMessageBox.Yes:
            return

        # Delete clip so it will be recreated when the workflow continues.
        try:
            os.remove(clip)
        except Exception:
            # If deletion fails, don't crash the review UI.
            pass

        # Ensure the clip stage won't skip due to cached manifests/fingerprints.
        try:
            self._invalidate_clips_cache()
        except Exception:
            pass

        # Keep manifest consistent (optional best-effort).
        try:
            self._remove_clip_from_manifest(sid)
        except Exception:
            pass

    def on_regen_failed(self, sid: str, err: str) -> None:
        self._regen_busy = False
        self._set_busy(False, f"Failed: {sid} — {err}")


class ClipReviewDialog(QDialog):
    """Chunk 8B: Review generated clips and optionally regenerate individual clips in the worker thread."""

    def _set_busy(self, busy: bool, status: str) -> None:
        """Enable/disable controls while a regeneration request is running.

        While regenerating, we still allow selecting and playing other clips.
        Only actions that would interfere with the running regen (Recreate / Continue)
        are disabled.
        """
        self._regen_busy = bool(busy)
        # Always allow browsing and playback.
        try:
            self.lst.setEnabled(True)
        except Exception:
            pass

        for attr, enabled in [
            ("btn_play", True),
            ("btn_stop", True),
            ("btn_recreate", (not busy) and bool(getattr(self, "_current_sid", ""))),
            ("btn_continue", not busy),
            ("btn_cancel_job", True),
        ]:
            w = getattr(self, attr, None)
            if w is not None:
                try:
                    w.setEnabled(bool(enabled))
                except Exception:
                    pass
        if status:
            lbl = getattr(self, "lbl_status", None)
            if lbl is not None:
                try:
                    lbl.setText(status)
                except Exception:
                    pass

    def __init__(self, parent: QWidget, *, worker: "PipelineWorker", payload: Dict[str, Any]):
        super().__init__(parent)
        self.setWindowTitle("Clip Review")
        self.setModal(True)
        self._worker = worker
        self._payload = dict(payload or {})
        self._sent_terminal_cmd = False
        self._regen_busy: bool = False

        # Slightly larger default window so everything has breathing room
        self.resize(1180, 760)
        self.setMinimumSize(980, 640)

        self._manifest_path = str(self._payload.get("manifest_path") or "")
        self._clips_dir = str(self._payload.get("clips_dir") or "")
        self._shots_path = str(self._payload.get("shots_path") or "")
        self._images_dir = str(self._payload.get("images_dir") or "")

        self._current_sid: str = ""
        self._current_clip: str = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        top = QLabel("Inspect clips now. Select a shot, then click Play to preview. Use Recreate to regenerate the selected clip. When done, click Continue pipeline.")
        top.setWordWrap(True)
        root.addWidget(top)

        body = QHBoxLayout()
        body.setSpacing(10)

        self.lst = QListWidget()
        self.lst.setMinimumWidth(360)
        self.lst.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        body.addWidget(self.lst, 2)

        right = QVBoxLayout()
        right.setSpacing(8)

        self.lbl_sel = QLabel("Selected: —")
        self.lbl_sel.setWordWrap(True)
        right.addWidget(self.lbl_sel)

        self.lbl_meta = QLabel("")
        self.lbl_meta.setWordWrap(True)
        right.addWidget(self.lbl_meta)

        labp = QLabel("Clip prompt (editable; used for Recreate):")
        labp.setWordWrap(True)
        right.addWidget(labp)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText("Edit the prompt for this clip, then click Recreate…")
        self.prompt_edit.setMinimumHeight(140)
        self.prompt_edit.setMaximumHeight(220)
        right.addWidget(self.prompt_edit)

        self.chk_retry_new_seed = QCheckBox("Retry with new seed")
        self.chk_retry_new_seed.setToolTip("When enabled, Recreate will pick a new random seed for this clip.")
        right.addWidget(self.chk_retry_new_seed)

        # Preview pane: always show the first frame as a poster image.
        # If QtMultimedia is available, Play will switch to the embedded player.
        self._thumb_cache: Dict[str, Tuple[float, str]] = {}
        self._thumb_dir = str((_root() / "output" / "planner" / "_thumbs").resolve())
        try:
            os.makedirs(self._thumb_dir, exist_ok=True)
        except Exception:
            self._thumb_dir = tempfile.gettempdir()

        self._thumb_label = QLabel("Select a clip…")
        try:
            self._thumb_label.setAlignment(Qt.AlignCenter)
        except Exception:
            pass
        self._thumb_label.setMinimumHeight(240)
        self._thumb_label.setStyleSheet("border: 1px solid #444; padding: 6px;")
        try:
            self._thumb_label.setScaledContents(False)
        except Exception:
            pass

        self._preview_stack = QStackedWidget()
        self._preview_stack.addWidget(self._thumb_label)  # index 0

        # Embedded player (optional)
        self._have_player = bool(_HAVE_QT_MEDIA)
        self._player = None
        self._video_widget = None
        self._audio = None

        if self._have_player:
            try:
                self._video_widget = QVideoWidget()
                self._video_widget.setMinimumHeight(240)
                self._player = QMediaPlayer(self)
                try:
                    self._audio = QAudioOutput(self)
                    self._player.setAudioOutput(self._audio)
                except Exception:
                    self._audio = None
                self._player.setVideoOutput(self._video_widget)
                self._preview_stack.addWidget(self._video_widget)  # index 1
            except Exception:
                self._have_player = False
                self._player = None
                self._video_widget = None
                self._audio = None

        if not self._have_player:
            lab = QLabel("Play will open your system player (QtMultimedia unavailable).")
            lab.setWordWrap(True)
            lab.setMinimumHeight(240)
            lab.setStyleSheet("border: 1px solid #444; padding: 8px;")
            self._preview_stack.addWidget(lab)  # index 1

        right.addWidget(self._preview_stack, 1)


        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_play = QPushButton("Play")
        self.btn_stop = QPushButton("Stop")
        self.btn_recreate = QPushButton("Recreate")
        btn_row.addWidget(self.btn_play)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_recreate)
        btn_row.addStretch(1)
        right.addLayout(btn_row)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        self.btn_continue = QPushButton("Continue pipeline")
        self.btn_cancel_job = QPushButton("Cancel job")
        bottom_row.addWidget(self.btn_continue)
        bottom_row.addWidget(self.btn_cancel_job)
        bottom_row.addStretch(1)
        right.addLayout(bottom_row)

        # If we're delegating to the FrameVision queue, cancellation is handled there.
        # Hide the Cancel button in review popups to avoid confusing/incorrect behavior.
        try:
            if bool(getattr(parent, "_use_framevision_queue", True)):
                self.btn_cancel_job.hide()
        except Exception:
            pass

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        right.addWidget(self.lbl_status)

        body.addLayout(right, 1)
        root.addLayout(body, 1)

        self.btn_play.clicked.connect(self._on_play)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_recreate.clicked.connect(self._on_recreate)
        def _do_continue():
            # Persist prompt edits from the current selection before resuming
            try:
                if getattr(self, "_current_sid", ""):
                    self._persist_prompt_for_sid(str(self._current_sid))
            except Exception:
                pass
            try:
                self._worker.post_review_command({"type": "continue"})
            except Exception:
                pass
            self._sent_terminal_cmd = True
            self.accept()

        def _do_cancel():
            try:
                self._worker.post_review_command({"type": "cancel"})
            except Exception:
                pass
            self._sent_terminal_cmd = True
            self.reject()

        self.btn_continue.clicked.connect(_do_continue)
        self.btn_cancel_job.clicked.connect(_do_cancel)
        self.lst.currentItemChanged.connect(self._on_select)

        self._load_items()
        self._set_busy(False, "")

    # NOTE: In a previous merge, the recreate handler was accidentally
    # dedented to module scope, so the dialog instance had no _on_recreate.
    # Keep this thin wrapper so existing projects don't crash.
    def _on_recreate(self) -> None:
        return _clipreview_on_recreate(self)

    def closeEvent(self, ev):  # type: ignore[override]
        try:
            self._on_stop()
        except Exception:
            pass

        # Persist any prompt edits from the current selection
        try:
            if getattr(self, "_current_sid", ""):
                self._persist_prompt_for_sid(str(self._current_sid))
        except Exception:
            pass

        if not self._sent_terminal_cmd:
            try:
                self._worker.post_review_command({"type": "continue"})
            except Exception:
                pass
            self._sent_terminal_cmd = True
        super().closeEvent(ev)

    def _read_manifest(self) -> Dict[str, Any]:
        m = _safe_read_json(self._manifest_path) if self._manifest_path else None
        return m if isinstance(m, dict) else {}

    def _read_shots(self) -> List[Dict[str, Any]]:
        try:
            return _load_shots_list(self._shots_path) if self._shots_path else []
        except Exception:
            return []

    def _shot_label(self, shots: List[Dict[str, Any]], sid: str) -> str:
        sid = str(sid or "")
        for sh in shots:
            if isinstance(sh, dict) and str(sh.get("id") or "") == sid:
                txt = str(sh.get("visual_description") or sh.get("notes") or "").strip()
                if txt:
                    return txt
        return ""

    def _resolve_clips(self, manifest: Dict[str, Any]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        try:
            items = (manifest.get("paths") or {}).get("clips") or []
        except Exception:
            items = []
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and it.get("id") and it.get("file"):
                    out.append({"id": str(it.get("id")), "file": str(it.get("file"))})

        if not out and self._clips_dir and os.path.isdir(self._clips_dir):
            try:
                for p in sorted(Path(self._clips_dir).glob("*.mp4")):
                    out.append({"id": p.stem, "file": str(p)})
            except Exception:
                pass

        out.sort(key=lambda d: d.get("id", ""))
        return out

    def _persist_prompt_for_sid(self, sid: str) -> None:
        """Persist the prompt editor text into manifest shots[sid].i2v_prompt (lightweight, UI thread)."""
        sid = str(sid or "").strip()
        if not sid or not self._manifest_path:
            return
        try:
            prompt = (self.prompt_edit.toPlainText() or "").strip()
        except Exception:
            return
        if not prompt:
            return
        try:
            manifest = self._read_manifest()
            shots = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
            rec = shots.get(sid) if isinstance(shots, dict) else {}
            if not isinstance(rec, dict):
                rec = {}
            old = str(rec.get("i2v_prompt") or "").strip()
            if old and old != prompt and "i2v_prompt_original" not in rec:
                rec["i2v_prompt_original"] = old
            rec["i2v_prompt"] = prompt
            rec["ts_i2v_prompt_edit"] = time.time()
            shots[sid] = rec
            manifest["shots"] = shots
            _safe_write_json(self._manifest_path, manifest)

            it = getattr(self, "_item_by_sid", {}).get(sid)
            if it:
                try:
                    it.setToolTip(prompt)
                except Exception:
                    pass
        except Exception:
            pass

    def _load_items(self) -> None:
        self.lst.clear()
        self._item_by_sid: Dict[str, QListWidgetItem] = {}

        manifest = self._read_manifest()
        shots = self._read_shots()
        clips = self._resolve_clips(manifest)

        for it in clips:
            sid = str(it.get("id") or "").strip()
            fp = str(it.get("file") or "").strip()
            if not sid:
                continue
            label = self._shot_label(shots, sid)
            display = sid
            if label:
                short = label.replace("\n", " ").strip()
                if len(short) > 80:
                    short = short[:77].rstrip() + "…"
                display = f"{sid} — {short}"

            item = QListWidgetItem(display)
            item.setData(Qt.UserRole, {"id": sid, "file": fp, "label": label})
            item.setToolTip(label or "")
            self.lst.addItem(item)
            self._item_by_sid[sid] = item

        if self.lst.count() > 0:
            self.lst.setCurrentRow(0)

    def _on_select(self, cur: QListWidgetItem, prev: QListWidgetItem) -> None:
            # Persist prompt edits from previous selection (so they survive switching items)
            try:
                if getattr(self, "_current_sid", ""):
                    self._persist_prompt_for_sid(str(self._current_sid))
            except Exception:
                pass

            try:
                self._on_stop()
            except Exception:
                pass

            data = cur.data(Qt.UserRole) if cur else {}
            sid = str((data or {}).get("id") or "").strip()
            fp = str((data or {}).get("file") or "").strip()
            label = str((data or {}).get("label") or "").strip()

            self._current_sid = sid
            self._current_clip = fp

            self.lbl_sel.setText(f"Selected: {sid}" if sid else "Selected: —")

            meta = []
            if label:
                meta.append(label)
            if fp:
                meta.append(fp)
            self.lbl_meta.setText("\n\n".join([m for m in meta if m]).strip())

            # Load i2v prompt for this shot (editable for clip regen)
            prompt = ""
            try:
                manifest = self._read_manifest()
                shots = manifest.get("shots") if isinstance(manifest.get("shots"), dict) else {}
                rec = shots.get(sid) if isinstance(shots, dict) else {}
                if isinstance(rec, dict):
                    prompt = str(rec.get("i2v_prompt") or "").strip()
            except Exception:
                prompt = ""
            if not prompt:
                prompt = "Slow camera move, subtle parallax, keep subject stable. Match the image exactly."

            try:
                self.prompt_edit.blockSignals(True)
                self.prompt_edit.setPlainText(prompt)
                self.prompt_edit.blockSignals(False)
            except Exception:
                pass

            # Update poster frame preview
            try:
                self._show_first_frame(str(fp))
            except Exception:
                pass

    def _on_play(self) -> None:
        path = str(self._current_clip or "").strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Missing clip", f"Clip not found:\n{path}")
            return

        if self._have_player and self._player is not None and QUrl is not None:
            try:
                self._player.setSource(QUrl.fromLocalFile(path))  # type: ignore[attr-defined]
                try:
                    self._preview_stack.setCurrentIndex(1)
                except Exception:
                    pass
                self._player.play()
                return
            except Exception:
                pass

        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            elif os.name == "posix":
                import subprocess
                subprocess.Popen(["xdg-open", path])
            else:
                QMessageBox.information(self, "Clip path", path)
        except Exception:
            QMessageBox.information(self, "Clip path", path)

    def _on_stop(self) -> None:
        if self._have_player and self._player is not None:
            try:
                self._player.stop()
            except Exception:
                pass
        try:
            self._preview_stack.setCurrentIndex(0)
        except Exception:
            pass



    def _ffmpeg_tool(self, exe_name: str) -> str:
        """Resolve ffmpeg from presets/bin first, then PATH."""
        b = _root() / "presets" / "bin"
        cands = []
        if os.name == "nt":
            cands += [b / f"{exe_name}.exe", b / f"{exe_name}.bat", b / exe_name]
        else:
            cands += [b / exe_name]
        for p in cands:
            try:
                if p.exists():
                    return str(p)
            except Exception:
                pass
        return exe_name

    def _ensure_first_frame_png(self, clip_path: str) -> Optional[str]:
        clip_path = str(clip_path or "").strip()
        if not clip_path or not os.path.isfile(clip_path):
            return None
        try:
            mtime = float(os.path.getmtime(clip_path))
        except Exception:
            mtime = 0.0

        cached = self._thumb_cache.get(clip_path)
        if cached and abs(float(cached[0]) - float(mtime)) < 0.0001 and os.path.isfile(cached[1]):
            return str(cached[1])

        # Stable name
        try:
            h = hashlib.sha1(clip_path.encode("utf-8", errors="ignore")).hexdigest()[:12]
        except Exception:
            h = str(uuid.uuid4()).replace("-", "")[:12]
        out_png = os.path.join(str(self._thumb_dir), f"clip_{h}.png")

        ffmpeg = self._ffmpeg_tool("ffmpeg")
        args = [
            ffmpeg, "-y",
            "-hide_banner", "-loglevel", "error",
            "-ss", "0",
            "-i", str(clip_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_png),
        ]
        try:
            subprocess.run(args, cwd=str(_root()), check=True)
        except Exception:
            # Some builds dislike -ss before -i; try a simpler select filter.
            try:
                args2 = [
                    ffmpeg, "-y",
                    "-hide_banner", "-loglevel", "error",
                    "-i", str(clip_path),
                    "-vf", "select=eq(n\,0)",
                    "-frames:v", "1",
                    "-q:v", "2",
                    str(out_png),
                ]
                subprocess.run(args2, cwd=str(_root()), check=True)
            except Exception:
                return None

        if os.path.isfile(out_png) and os.path.getsize(out_png) > 512:
            self._thumb_cache[clip_path] = (mtime, out_png)
            return str(out_png)
        return None

    def _show_first_frame(self, clip_path: str) -> None:
        png = self._ensure_first_frame_png(str(clip_path))
        if not png:
            try:
                self._thumb_label.setText("No preview available.")
            except Exception:
                pass
            try:
                self._preview_stack.setCurrentIndex(0)
            except Exception:
                pass
            return
        try:
            pm = QPixmap(png)
            if not pm.isNull():
                # Fit to label while preserving aspect ratio
                try:
                    w = max(1, int(self._thumb_label.width()))
                    h = max(1, int(self._thumb_label.height()))
                    pm2 = pm.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                except Exception:
                    pm2 = pm
                self._thumb_label.setPixmap(pm2)
                self._thumb_label.setText("")
        except Exception:
            try:
                self._thumb_label.setText("Preview load failed.")
            except Exception:
                pass
        try:
            self._preview_stack.setCurrentIndex(0)
        except Exception:
            pass

    def on_regen_started(self, sid: str) -> None:
        self._set_busy(True, f"Regenerating {sid}…")

    def on_regen_done(self, sid: str, path: str) -> None:
        # Update list item file pointer
        try:
            it = self._item_by_sid.get(str(sid))
            if it:
                data = it.data(Qt.UserRole) or {}
                data["file"] = str(path)
                it.setData(Qt.UserRole, data)
        except Exception:
            pass

        if str(sid) == str(getattr(self, "_current_sid", "")):
            self._current_clip = str(path)
            try:
                self._show_first_frame(str(path))
            except Exception:
                pass

        self._set_busy(False, f"Done: {sid} (recreated)")

    def on_regen_failed(self, sid: str, err: str) -> None:
        self._set_busy(False, f"Failed: {sid} — {err}")
        try:
            QMessageBox.warning(self, "Recreate failed", f"{sid}:\n{err}")
        except Exception:
            pass
# -----------------------------
# Wan 2.2 clip runner (CLI)
# -----------------------------
_WAN22_HELP_CACHE: Optional[str] = None
_WAN22_NEG_FLAG: Optional[str] = None
_WAN22_HAS_T5_CPU: Optional[bool] = None
_WAN22_LORA_CAPS: Optional[Dict[str, bool]] = None


def _wan22_python_exe() -> str:
    # Prefer Wan's dedicated venv if present, otherwise fall back.
    try:
        if os.name == "nt":
            cand = _root() / ".wan_venv" / "Scripts" / "python.exe"
        else:
            cand = _root() / ".wan_venv" / "bin" / "python"
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    return sys.executable


def _wan22_model_root() -> Path:
    return (_root() / "models" / "wan22").resolve()


def _wan22_generate_py() -> Path:
    return (_wan22_model_root() / "generate.py").resolve()


def _wan22_help_text(py_exe: str) -> str:
    global _WAN22_HELP_CACHE
    if isinstance(_WAN22_HELP_CACHE, str) and _WAN22_HELP_CACHE.strip():
        return _WAN22_HELP_CACHE
    gen = _wan22_generate_py()
    root_dir = _wan22_model_root()
    if not gen.exists():
        return ""
    try:
        p = subprocess.run(
            [py_exe, str(gen), "--help"],
            cwd=str(root_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=25,
        )
        _WAN22_HELP_CACHE = (p.stdout or "") + "\n" + (p.stderr or "")
    except Exception:
        _WAN22_HELP_CACHE = ""
    return _WAN22_HELP_CACHE or ""


def _wan22_negative_flag(py_exe: str) -> Optional[str]:
    global _WAN22_NEG_FLAG
    if _WAN22_NEG_FLAG is not None:
        return _WAN22_NEG_FLAG
    ht = _wan22_help_text(py_exe)
    for flag in ("--negative_prompt", "--negative_prompt_text", "--negative"):
        if flag in ht:
            _WAN22_NEG_FLAG = flag
            return _WAN22_NEG_FLAG
    _WAN22_NEG_FLAG = None
    return None


def _wan22_has_t5_cpu(py_exe: str) -> bool:
    global _WAN22_HAS_T5_CPU
    if isinstance(_WAN22_HAS_T5_CPU, bool):
        return _WAN22_HAS_T5_CPU
    ht = _wan22_help_text(py_exe)
    _WAN22_HAS_T5_CPU = ("--t5_cpu" in ht)
    return bool(_WAN22_HAS_T5_CPU)

def _wan22_subproc_env() -> dict:
    """Environment for running WAN 2.2 subprocesses (force UTF-8 on Windows)."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env



def _run_wan22_clip(*, prompt: str, negative: str, image_path: str, out_path: str,
                    fps: int, frames: int, steps: int, seed: Optional[int],
                    size_str: str, guidance: float = 4.0,
                    offload_model: bool = True, t5_cpu: bool = False,
                    log_path: Optional[str] = None) -> None:
    """Run Wan 2.2 generate.py (ti2v-5B) for one image->video clip."""
    py = _wan22_python_exe()
    gen = _wan22_generate_py()
    model_root = _wan22_model_root()

    if not gen.exists():
        raise RuntimeError(f"Wan 2.2 generate.py not found: {gen}")

    # build args aligned with helpers/wan22.py defaults
    args = [
        py,
        str(gen),
        "--task", "ti2v-5B",
        "--size", str(size_str),
        "--sample_steps", str(int(steps)),
        "--sample_guide_scale", str(float(guidance)),
        "--base_seed", str(int(seed) if seed is not None else 0),
        "--frame_num", str(int(frames)),
        "--fps", str(int(fps)),
        "--ckpt_dir", str(model_root),
        "--convert_model_dtype",
        "--prompt", str(prompt or ""),
        "--image", str(image_path),
    ]

    # Offload (best-effort): only pass when enabled to match WAN UI behavior
    if bool(offload_model):
        args += ["--offload_model", "True"]

    # Optional T5 CPU offload (best-effort; only if supported)
    if bool(t5_cpu) and _wan22_has_t5_cpu(py):
        args.append("--t5_cpu")

    # Negative prompt (best-effort; only if supported)
    if (negative or "").strip():
        nf = _wan22_negative_flag(py)
        if nf:
            args += [nf, str(negative)]

    
    # Output file
    args += ["--save_file", str(out_path)]
    # Ensure out dir exists
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # Run
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8", errors="ignore") as lf:
            lf.write("[planner] wan22 cmd:\n")
            lf.write(" ".join([str(x) for x in args]) + "\n\n")
            lf.flush()
            subprocess.run(args, cwd=str(model_root), stdout=lf, stderr=lf, check=True, env=_wan22_subproc_env())
    else:
        subprocess.run(args, cwd=str(model_root), check=True, env=_wan22_subproc_env())




def _clipreview_on_recreate(self) -> None:
    if self._regen_busy:
        return
    sid = str(self._current_sid or "").strip()
    if not sid:
        return

    prompt = ""
    try:
        prompt = (self.prompt_edit.toPlainText() or "").strip()
    except Exception:
        prompt = ""
    if not prompt:
        QMessageBox.warning(self, "Empty prompt", "Prompt is empty.")
        return

    # Persist prompt immediately so regen uses it and it survives restarts
    try:
        self._persist_prompt_for_sid(sid)
    except Exception:
        pass

    self._set_busy(True, f"Queued: {sid}…")

    seed_override = None
    try:
        if hasattr(self, "chk_retry_new_seed") and self.chk_retry_new_seed.isChecked():
            seed_override = int(random.SystemRandom().randint(1, 2147483647))
    except Exception:
        seed_override = None

    try:
        cmd = {"type": "clip_regen", "sid": sid, "prompt": prompt}
        if seed_override is not None:
            cmd["seed"] = int(seed_override)
        self._worker.post_review_command(cmd)
    except Exception as e:
        self._set_busy(False, f"Failed to send regen: {e}")
        try:
            QMessageBox.warning(self, "Cannot recreate", str(e))
        except Exception:
            pass

    def _on_continue(self) -> None:
        # Persist prompt edits from the current selection before resuming
        try:
            if getattr(self, "_current_sid", ""):
                self._persist_prompt_for_sid(str(self._current_sid))
        except Exception:
            pass
        try:
            self._worker.post_review_command({"type": "continue"})
        except Exception:
            pass
        self._sent_terminal_cmd = True
        self.accept()

    def _on_cancel_job(self) -> None:
        try:
            self._worker.post_review_command({"type": "cancel"})
        except Exception:
            pass
        self._sent_terminal_cmd = True
        self.reject()

    def _set_busy(self, busy: bool, status: str) -> None:
        self._regen_busy = bool(busy)
        self.btn_play.setEnabled(not busy)
        self.btn_stop.setEnabled(not busy)
        self.btn_recreate.setEnabled(not busy and bool(self._current_sid))
        self.btn_continue.setEnabled(not busy)
        self.btn_cancel_job.setEnabled(True)
        if status:
            self.lbl_status.setText(status)

    def on_regen_started(self, sid: str) -> None:
        self._set_busy(True, f"Regenerating {sid}…")

    def on_regen_done(self, sid: str, path: str) -> None:
        self._regen_busy = False
        try:
            it = self._item_by_sid.get(str(sid))
            if it:
                data = it.data(Qt.UserRole) or {}
                data["file"] = str(path)
                it.setData(Qt.UserRole, data)
        except Exception:
            pass

        if str(sid) == self._current_sid:
            self._current_clip = str(path)
        self._set_busy(False, f"Done: {sid} (recreated)")

    def on_regen_failed(self, sid: str, err: str) -> None:
        self._regen_busy = False
        self._set_busy(False, f"Failed: {sid} — {err}")
        try:
            QMessageBox.warning(self, "Recreate failed", f"{sid}:\n{err}")
        except Exception:
            pass



class VideoReviewPlaceholderDialog(QDialog):
    def __init__(self, parent: QWidget, *, worker: "PipelineWorker"):
        super().__init__(parent)
        self.setWindowTitle("Video Review (coming soon)")
        self.setModal(True)
        self._worker = worker
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)
        lab = QLabel("Video review/edit is not implemented yet .\n\nComing soon.")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        btn = QPushButton("Continue pipeline")
        btn.clicked.connect(self._on_continue)
        lay.addWidget(btn)

    def _on_continue(self) -> None:
        try:
            self._worker.post_review_command({"type": "continue"})
        except Exception:
            pass
        self.accept()


# -----------------------------
# Standalone window
# -----------------------------

class PlannerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Planner")
        self.setMinimumSize(1050, 650)

        pane = PlannerPane(self)
        self.setCentralWidget(pane)

        try:
            _LOGGER.log_probe("Window created")
        except Exception:
            pass

        # Simple menu
        m = self.menuBar().addMenu("File")
        act_export = QAction("Export job JSON…", self)
        act_export.triggered.connect(pane._export_job_json)
        m.addAction(act_export)

        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        m.addAction(act_quit)




def _qwen_text_call(step_name: str, system_prompt: str, user_prompt: str, log_path: str,
                    temperature: float = 0.3, max_new_tokens: int = 1024) -> str:
    """Call Qwen3-VL via a short-lived subprocess and return plain text.

    Writes a combined debug log (system/user/output/stderr) to log_path.

    VRAM rule:
    - Qwen is executed out-of-process so it cannot pin VRAM in the Planner process.
    """
    raw_text = ""
    try:
        if not _HAVE_QWEN_TEXT:
            raise RuntimeError(f"Qwen text generator not available: {_QWEN_IMPORT_ERROR!r}")
        model_path = Path(_qwen_model_dir())
        if not (model_path.exists() and any(model_path.iterdir())):
            raise RuntimeError(f"Qwen3-VL model folder not found or empty: {model_path}")

        out, err, rc = _run_qwen_text_subprocess(
            model_path=str(model_path),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=float(temperature),
            max_new_tokens=int(max_new_tokens),
        )
        raw_text = (out or "").strip()
        err = (err or "").strip()

        blob = []
        blob.append("[system]\n" + (system_prompt or "").strip())
        blob.append("\n[user]\n" + (user_prompt or "").strip())
        blob.append("\n[output]\n" + (raw_text or "").strip())
        if err:
            blob.append("\n[stderr]\n" + err)
        if rc != 0 and not raw_text:
            blob.append(f"\n[subprocess_rc]\n{rc}")
        _safe_write_text(log_path, "\n\n".join(blob).strip() + "\n")

        _vram_release("after qwen text (subprocess)")
        return raw_text
    except Exception:
        try:
            _safe_write_text(log_path, "[error]\n" + traceback.format_exc() + "\n")
        except Exception:
            pass
        raise

def main() -> int:
    try:
        _LOGGER.log_probe("App start")
    except Exception:
        pass
    app = QApplication([])
    w = PlannerWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())