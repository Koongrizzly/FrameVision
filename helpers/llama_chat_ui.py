# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import difflib
import json
import mimetypes
import os
import random
import re
import socket
import subprocess
import sys
import time
import uuid
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets
try:
    from PySide6 import QtMultimedia, QtMultimediaWidgets
except Exception:
    QtMultimedia = None
    QtMultimediaWidgets = None


# -----------------------------
# Paths / persistence helpers
# -----------------------------

def _find_fv_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, os.pardir)),
        os.path.abspath(os.path.join(here, os.pardir, os.pardir)),
        os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir)),
        here,
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "presets")):
            return c
    return os.path.abspath(os.path.join(here, os.pardir))


def _settings_path(fv_root: str) -> str:
    return os.path.join(fv_root, "presets", "setsave", "llama_chat_ui.json")


def _llm_gpu_lock_dir(fv_root: str) -> str:
    return os.path.join(fv_root, "temp", "runtime", "gpu_locks")


def _llm_gpu_lock_path(fv_root: str) -> str:
    return os.path.join(_llm_gpu_lock_dir(fv_root), "llm.lock")


def _write_llm_gpu_lock(fv_root: str, pid: int, model_path: str, runner_path: str = "") -> None:
    """Create a tiny lock file so the queue worker can avoid starting VRAM jobs while an LLM is loaded/loading."""
    try:
        lock_dir = _llm_gpu_lock_dir(fv_root)
        os.makedirs(lock_dir, exist_ok=True)
        payload = {
            "kind": "llm",
            "pid": int(pid or 0),
            "model": os.path.basename(str(model_path or "")),
            "model_path": str(model_path or ""),
            "runner": str(runner_path or ""),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "vram_heavy": True,
        }
        tmp = _llm_gpu_lock_path(fv_root) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
        os.replace(tmp, _llm_gpu_lock_path(fv_root))
    except Exception:
        pass


def _remove_llm_gpu_lock(fv_root: str) -> None:
    try:
        p = _llm_gpu_lock_path(fv_root)
        if os.path.exists(p):
            os.remove(p)
    except Exception:
        pass


def _default_llama_server_path(fv_root: str) -> str:
    """Preferred bundled llama.cpp server location."""
    return os.path.join(fv_root, "presets", "bin", "llama", "llama-server.exe")


def _find_default_llama_server(fv_root: str) -> str:
    """Find the bundled llama-server.exe without touching user-selected paths."""
    cands = [
        _default_llama_server_path(fv_root),
        os.path.join(fv_root, "presets", "bin", "llama-server.exe"),
    ]
    for cand in cands:
        if os.path.isfile(cand):
            return cand
    return ""


BUBBLE_COLOR_OPTIONS: List[Tuple[str, str]] = [
    ("Blue", "#3d6fd6"),
    ("Teal", "#2d8b8b"),
    ("Purple", "#7a54d1"),
    ("Pink", "#ba548e"),
    ("Green", "#3e8a52"),
    ("Orange", "#c97a2c"),
    ("Red", "#b65353"),
    ("Slate", "#56657f"),
]
DEFAULT_BUBBLE_BASE_COLOR = "#56657f"
DEFAULT_BUBBLE_USER_COLOR = "#33425f"
DEFAULT_BUBBLE_ASSISTANT_COLOR = "#212734"


def _normalize_hex_color(value: str, default: str) -> str:
    s = str(value or "").strip()
    if not s:
        return default
    if not s.startswith("#"):
        s = "#" + s
    if re.fullmatch(r"#[0-9a-fA-F]{6}", s):
        return s.lower()
    return default


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    color = _normalize_hex_color(value, "#000000")
    return int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)


def _mix_hex(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))
    ar, ag, ab = _hex_to_rgb(color_a)
    br, bg, bb = _hex_to_rgb(color_b)
    rr = int(round(ar + (br - ar) * ratio))
    rg = int(round(ag + (bg - ag) * ratio))
    rb = int(round(ab + (bb - ab) * ratio))
    return f"#{rr:02x}{rg:02x}{rb:02x}"


def _color_luminance(color: str) -> float:
    r, g, b = _hex_to_rgb(color)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _text_color_for_bg(color: str) -> str:
    return "#10151e" if _color_luminance(color) >= 0.64 else "#edf2fa"


def _role_color_for_bg(color: str) -> str:
    text = _text_color_for_bg(color)
    return _mix_hex(text, color, 0.42)


def _bubble_colors_from_settings(auto_mode: bool, base_color: str, assistant_color: str, user_color: str) -> Dict[str, str]:
    base_color = _normalize_hex_color(base_color, DEFAULT_BUBBLE_BASE_COLOR)
    assistant_color = _normalize_hex_color(assistant_color, DEFAULT_BUBBLE_ASSISTANT_COLOR)
    user_color = _normalize_hex_color(user_color, DEFAULT_BUBBLE_USER_COLOR)
    if auto_mode:
        user_bg = _mix_hex(base_color, "#ffffff", 0.18)
        assistant_bg = _mix_hex(base_color, "#000000", 0.46)
    else:
        user_bg = user_color
        assistant_bg = assistant_color
    return {
        "user_bg": user_bg,
        "assistant_bg": assistant_bg,
        "user_text": _text_color_for_bg(user_bg),
        "assistant_text": _text_color_for_bg(assistant_bg),
        "user_role": _role_color_for_bg(user_bg),
        "assistant_role": _role_color_for_bg(assistant_bg),
    }


def _palette_color_hex(role: QtGui.QPalette.ColorRole, fallback: str) -> str:
    try:
        app = QtWidgets.QApplication.instance()
        pal = app.palette() if app is not None else QtGui.QPalette()
        value = pal.color(role).name()
        return _normalize_hex_color(value, fallback)
    except Exception:
        return fallback


def _framevision_theme_colors() -> Dict[str, str]:
    """Palette-driven colors so the embedded chat follows FrameVision's active theme better."""
    window = _palette_color_hex(QtGui.QPalette.Window, "#16181d")
    base = _palette_color_hex(QtGui.QPalette.Base, "#1b2029")
    alt = _palette_color_hex(QtGui.QPalette.AlternateBase, _mix_hex(base, window, 0.35))
    button = _palette_color_hex(QtGui.QPalette.Button, "#2a3140")
    text = _palette_color_hex(QtGui.QPalette.WindowText, "#e8edf5")
    base_text = _palette_color_hex(QtGui.QPalette.Text, text)
    button_text = _text_color_for_bg(button)
    mid = _palette_color_hex(QtGui.QPalette.Mid, "#323848")
    dark = _palette_color_hex(QtGui.QPalette.Dark, "#252a36")
    highlight = _palette_color_hex(QtGui.QPalette.Highlight, DEFAULT_BUBBLE_BASE_COLOR)
    highlighted_text = _palette_color_hex(QtGui.QPalette.HighlightedText, "#ffffff")

    if _color_luminance(window) > 0.72:
        # Standalone fallback on a light OS palette: keep the chat readable instead of going pure white.
        window = "#16181d"
        base = "#1b2029"
        alt = "#21242d"
        button = "#2a3140"
        text = "#e8edf5"
        base_text = "#eff3f9"
        button_text = "#eff3f9"
        mid = "#323848"
        dark = "#252a36"
        highlighted_text = "#ffffff"

    panel = _mix_hex(base, window, 0.28)
    panel_2 = _mix_hex(button, window, 0.22)
    border = _mix_hex(mid, window, 0.18)
    hover = _mix_hex(button, highlight, 0.18)
    selected = _mix_hex(button, highlight, 0.28)
    subtle = _mix_hex(text, window, 0.42)
    return {
        "window": window,
        "base": base,
        "alt": alt,
        "panel": panel,
        "panel_2": panel_2,
        "button": button,
        "button_text": button_text,
        "text": text,
        "base_text": base_text,
        "subtle": subtle,
        "border": border,
        "dark": dark,
        "highlight": highlight,
        "highlighted_text": highlighted_text,
        "hover": hover,
        "selected": selected,
    }


def _soft_chat_bubble_colors(theme: Dict[str, str], auto_mode: bool, auto_color: str, assistant_color: str, user_color: str) -> Dict[str, str]:
    raw = _bubble_colors_from_settings(auto_mode, auto_color, assistant_color, user_color)
    highlight = _normalize_hex_color(theme.get("highlight", DEFAULT_BUBBLE_BASE_COLOR), DEFAULT_BUBBLE_BASE_COLOR)
    panel = _normalize_hex_color(theme.get("panel", "#21242d"), "#21242d")
    window = _normalize_hex_color(theme.get("window", "#16181d"), "#16181d")

    # Existing saved bubble colors can be very strong when the chat is embedded in FrameVision.
    # Blend them back into the app palette so they read as message areas, not loud theme blocks.
    user_seed = highlight if auto_mode else raw["user_bg"]
    assistant_seed = _mix_hex(highlight, panel, 0.82) if auto_mode else raw["assistant_bg"]

    # Keep message rows close to the host app surface. A small accent tint is enough;
    # stronger colors look like thick blocks inside FrameVision's own theme.
    user_bg = _mix_hex(panel, user_seed, 0.12)
    assistant_bg = _mix_hex(panel, assistant_seed, 0.07)

    user_border = _mix_hex(user_bg, highlight, 0.18)
    assistant_border = _mix_hex(assistant_bg, highlight, 0.13)

    user_text = _text_color_for_bg(user_bg)
    assistant_text = _text_color_for_bg(assistant_bg)
    return {
        "user_bg": user_bg,
        "assistant_bg": assistant_bg,
        "user_text": user_text,
        "assistant_text": assistant_text,
        "user_role": _mix_hex(user_text, user_bg, 0.46),
        "assistant_role": _mix_hex(assistant_text, assistant_bg, 0.50),
        "user_border": user_border,
        "assistant_border": assistant_border,
    }


def _chat_dir(fv_root: str) -> str:
    return os.path.join(fv_root, "data", "llama_chat", "chats")


def _attachment_temp_dir(fv_root: str) -> str:
    return os.path.join(fv_root, "data", "llama_chat", "attachments")


def _assistant_jobs_path(fv_root: str) -> str:
    return os.path.join(fv_root, "temp", "llama_chat_jobs.json")


def _llama_image_output_dir(fv_root: str) -> str:
    return os.path.join(fv_root, "output", "images")


def _templates_root(fv_root: str) -> str:
    cand1 = os.path.join(fv_root, "assets", "templates")
    cand2 = os.path.join(fv_root, "assests", "templates")
    if os.path.isdir(cand1):
        return cand1
    if os.path.isdir(cand2):
        return cand2
    if os.path.isdir(os.path.join(fv_root, "assets")):
        os.makedirs(cand1, exist_ok=True)
        return cand1
    os.makedirs(cand2, exist_ok=True)
    return cand2


# -----------------------------
# Local Knowledge & Memory helpers
# -----------------------------

def _knowledge_root(fv_root: str) -> str:
    # App-managed FrameVision knowledge. The chat may read/search this folder, but should not write memories here.
    return os.path.join(fv_root, "presets", "info")


def _memories_root(fv_root: str) -> str:
    # User-managed local memory root. This is intentionally outside presets/info so updates do not overwrite it.
    return os.path.join(fv_root, "assets", "memories")


def _memory_saved_notes_dir(fv_root: str) -> str:
    return os.path.join(_memories_root(fv_root), "saved_notes")


def _memory_user_files_dir(fv_root: str) -> str:
    return os.path.join(_memories_root(fv_root), "user_files")


def _memory_project_dir(fv_root: str) -> str:
    return os.path.join(_memories_root(fv_root), "project")


def _memory_llm_memory_dir(fv_root: str) -> str:
    return os.path.join(_memories_root(fv_root), "llm_memory")


def _ensure_memory_folders(fv_root: str) -> None:
    for d in (
        _memory_saved_notes_dir(fv_root),
        _memory_user_files_dir(fv_root),
        _memory_project_dir(fv_root),
        _memory_llm_memory_dir(fv_root),
    ):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


def _safe_memory_filename(title: str, fallback: str = "memory") -> str:
    text = unicodedata.normalize("NFKD", str(title or fallback)).encode("ascii", "ignore").decode("ascii", "ignore")
    text = re.sub(r"[^A-Za-z0-9._ -]+", "_", text).strip(" ._-")
    text = re.sub(r"\s+", "_", text)
    return (text[:72] or fallback).strip("._-") or fallback


def _memory_now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _iter_memory_files(root: str) -> List[str]:
    out: List[str] = []
    if not root or not os.path.isdir(root):
        return out
    allowed = {".txt", ".md", ".json", ".html", ".htm", ".py", ".log", ".csv", ".pdf"}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d.lower() not in {"__pycache__", "index"}]
        for name in files:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext in allowed:
                out.append(os.path.join(base, name))
    out.sort(key=lambda x: (os.path.getmtime(x) if os.path.exists(x) else 0), reverse=True)
    return out


def _read_pdf_text_best_effort(path: str, max_chars: int = 120000) -> str:
    try:
        import pypdf  # type: ignore
        reader = pypdf.PdfReader(path)
        parts = []
        for page in reader.pages[:60]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                pass
            if sum(len(x) for x in parts) >= max_chars:
                break
        return "\n".join(parts).strip()[:max_chars]
    except Exception:
        pass
    try:
        import PyPDF2  # type: ignore
        reader = PyPDF2.PdfReader(path)
        parts = []
        for page in reader.pages[:60]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                pass
            if sum(len(x) for x in parts) >= max_chars:
                break
        return "\n".join(parts).strip()[:max_chars]
    except Exception:
        return ""


def _read_memory_text_file(path: str, max_chars: int = 120000) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        text = _read_pdf_text_best_effort(path, max_chars=max_chars)
        return text or "[PDF has no readable text or PDF reader is unavailable.]"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(max_chars)
    except Exception as e:
        return f"[Could not read file: {e}]"


def _chunk_memory_text(text: str, chunk_chars: int = 2200) -> List[str]:
    text = re.sub(r"\r\n?", "\n", str(text or "")).strip()
    if not text:
        return []
    chunks: List[str] = []
    paras = re.split(r"\n\s*\n", text)
    buf = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(buf) + len(para) + 2 <= chunk_chars:
            buf = (buf + "\n\n" + para).strip()
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(para) <= chunk_chars:
            buf = para
        else:
            for i in range(0, len(para), chunk_chars):
                piece = para[i:i + chunk_chars].strip()
                if piece:
                    chunks.append(piece)
    if buf:
        chunks.append(buf)
    return chunks


def _keyword_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_\-]{2,}", str(text or "").lower()) if t not in {"the", "and", "for", "with", "that", "this", "from", "what", "about", "into", "your", "you", "are", "was", "were", "can", "could", "would", "should"}]


def _load_json(path: str, default):
    try:
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json_atomic(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# -----------------------------
# Discovery helpers
# -----------------------------

def _model_roots(fv_root: str) -> List[str]:
    cands = [
        os.path.join(fv_root, "models", "llama"),
        os.path.join(fv_root, "models", "llm_gguf"),
        os.path.join(fv_root, "models", "qwen35_gguf"),
        os.path.join(fv_root, "models"),
    ]
    out: List[str] = []
    seen = set()
    for p in cands:
        if not os.path.isdir(p):
            continue
        norm = os.path.normcase(os.path.normpath(p))
        if norm in seen:
            continue
        seen.add(norm)
        out.append(p)
    return out


def discover_models(fv_root: str, locked_root: str = "") -> List[Tuple[str, str]]:
    found: List[Tuple[str, str]] = []
    seen = set()
    locked_root = (locked_root or "").strip()
    if locked_root:
        if not os.path.isabs(locked_root):
            locked_root = os.path.abspath(os.path.join(fv_root, locked_root))
        search_roots = [locked_root] if os.path.isdir(locked_root) else []
        preferred_root = locked_root
    else:
        search_roots = _model_roots(fv_root)
        preferred_root = os.path.join(fv_root, "models", "llama")
    preferred_norm = os.path.normcase(os.path.normpath(preferred_root))
    for base in search_roots:
        for dirpath, _, filenames in os.walk(base):
            for fn in filenames:
                low = fn.lower()
                if not low.endswith(".gguf"):
                    continue
                # Skip multimodal projector GGUF files. These are companion files for vision models,
                # not standalone LLM chat models, and should not appear in the normal model picker.
                if "mmproj" in low:
                    continue
                full = os.path.join(dirpath, fn)
                norm = os.path.normcase(os.path.normpath(full))
                if norm in seen:
                    continue
                seen.add(norm)
                rel = os.path.relpath(full, fv_root)
                priority = 0 if os.path.normcase(os.path.normpath(base)) == preferred_norm else 1
                found.append((priority, rel, full))
    found.sort(key=lambda x: (x[0], x[1].lower()))
    return [(rel, full) for _, rel, full in found]


def _model_name_tokens(model_path: str) -> List[str]:
    stem = os.path.splitext(os.path.basename(model_path or ""))[0].lower()
    raw = re.split(r"[^a-z0-9]+", stem)
    generic = {
        "gguf", "q2", "q3", "q4", "q5", "q6", "q8", "k", "s", "m", "l", "xl",
        "iq1", "iq2", "iq3", "iq4", "iq5", "fp16", "bf16", "f16", "f32", "instruct",
        "chat", "it", "mlx", "vision", "text", "flash", "mini", "small", "medium", "large",
    }
    toks: List[str] = []
    for tok in raw:
        tok = tok.strip()
        if not tok or len(tok) < 3 or tok in generic:
            continue
        if tok.startswith("q") and any(ch.isdigit() for ch in tok):
            continue
        toks.append(tok)
    return toks


def _model_looks_multimodal(model_path: str) -> bool:
    hay = str(model_path or "").lower()
    keywords = (
        "vision", "-vl", "_vl", " vl ", "glm-4.1v", "glm-4v", "glm4v",
        "llava", "bakllava", "minicpm-v", "qwen2-vl", "qwen-vl", "internvl", "moondream",
    )
    return any(k in hay for k in keywords)


def _find_mmproj_for_model(model_path: str) -> str:
    model_path = os.path.abspath(model_path or "")
    if not model_path or not os.path.isfile(model_path):
        return ""
    if not _model_looks_multimodal(model_path):
        return ""

    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).lower()
    parent_dir = os.path.dirname(model_dir)
    candidate_dirs: List[str] = []
    seen_dirs = set()
    for d in [model_dir, os.path.join(model_dir, "mmproj"), os.path.join(model_dir, "vision"), parent_dir]:
        if d and os.path.isdir(d):
            norm = os.path.normcase(os.path.normpath(d))
            if norm not in seen_dirs:
                seen_dirs.add(norm)
                candidate_dirs.append(d)

    keyword_hits = [
        "mmproj",
        "mm-proj",
        "vision",
        "clip",
        "projector",
        "proj",
        "multimodal",
    ]
    tokens = _model_name_tokens(model_path)
    scored: List[Tuple[int, str]] = []

    for folder in candidate_dirs:
        try:
            names = os.listdir(folder)
        except Exception:
            continue
        for fn in names:
            low = fn.lower()
            if not low.endswith(".gguf"):
                continue
            full = os.path.join(folder, fn)
            if os.path.normcase(os.path.normpath(full)) == os.path.normcase(os.path.normpath(model_path)):
                continue
            if not (low.startswith("mmproj") or low.startswith("mm-proj") or any(k in low for k in keyword_hits)):
                continue
            score = 0
            if low.startswith("mmproj") or low.startswith("mm-proj"):
                score += 80
            if any(k in low for k in keyword_hits):
                score += 35
            if "proj" in low:
                score += 15
            for tok in tokens:
                if tok and tok in low:
                    score += 6
            if _model_looks_multimodal(model_name):
                score += 10
            if score > 0:
                scored.append((score, full))

    if not scored:
        return ""
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return scored[0][1]


def discover_templates(fv_root: str) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = [
        ("Auto (use model metadata / default)", "auto", ""),
        ("Smart guess from filename", "smart", ""),
        ("Use embedded template via jinja", "jinja", ""),
        ("Built-in: chatml", "builtin", "chatml"),
        ("Built-in: llama2", "builtin", "llama2"),
        ("Built-in: llama3", "builtin", "llama3"),
        ("Built-in: llama4", "builtin", "llama4"),
        ("Built-in: mistral-v1", "builtin", "mistral-v1"),
        ("Built-in: mistral-v3", "builtin", "mistral-v3"),
        ("Built-in: mistral-v7", "builtin", "mistral-v7"),
        ("Built-in: gemma", "builtin", "gemma"),
        ("Built-in: phi3", "builtin", "phi3"),
        ("Built-in: phi4", "builtin", "phi4"),
        ("Built-in: deepseek", "builtin", "deepseek"),
        ("Built-in: deepseek2", "builtin", "deepseek2"),
        ("Built-in: deepseek3", "builtin", "deepseek3"),
        ("Built-in: command-r", "builtin", "command-r"),
        ("Built-in: vicuna", "builtin", "vicuna"),
        ("Built-in: openchat", "builtin", "openchat"),
        ("Built-in: zephyr", "builtin", "zephyr"),
        ("Built-in: chatglm3", "builtin", "chatglm3"),
        ("Built-in: chatglm4", "builtin", "chatglm4"),
        ("Built-in: glmedge", "builtin", "glmedge"),
        ("Built-in: gpt-oss", "builtin", "gpt-oss"),
        ("Built-in: granite", "builtin", "granite"),
        ("Built-in: seed_oss", "builtin", "seed_oss"),
        ("Built-in: solar-open", "builtin", "solar-open"),
    ]
    root = _templates_root(fv_root)
    exts = {".jinja", ".j2", ".tmpl", ".txt"}
    files = []
    if os.path.isdir(root):
        for fn in os.listdir(root):
            p = os.path.join(root, fn)
            if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in exts:
                files.append(fn)
    files.sort(key=lambda s: s.lower())
    if files:
        items.append(("— Template files —", "sep", ""))
        for fn in files:
            items.append((f"File: {fn}", "file", fn))
    return items


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _resolve_server_executable(path: str) -> str:
    path = os.path.abspath(path)
    if not path:
        return path
    base = os.path.basename(path).lower()
    if "server" in base:
        return path
    folder = os.path.dirname(path)
    names = [
        "llama-server.exe",
        "llama-server",
        "server.exe",
        "server",
    ]
    for name in names:
        cand = os.path.join(folder, name)
        if os.path.isfile(cand):
            return cand
    return path


def _candidate_templates_from_model_path(model_path: str) -> List[Tuple[str, str]]:
    name = os.path.basename(model_path or "").lower()
    full = str(model_path or "").lower()
    hay = f"{name} {full}"

    out: List[Tuple[str, str]] = []

    def add(value: str):
        item = ("builtin", value)
        if item not in out:
            out.append(item)

    if any(n in hay for n in ("llama-4", "llama 4", "llama4")):
        add("llama4")
    if any(n in hay for n in ("llama-3", "llama 3", "llama3", "meta-llama-3", "meta_llama_3")):
        add("llama3")
    if any(n in hay for n in ("llama-2", "llama 2", "llama2")):
        add("llama2")
    if "qwen" in hay:
        add("chatml")
    if any(n in hay for n in ("deepseek-r1", "deepseek r1", "deepseek-v3", "deepseek v3", "deepseek3")):
        add("deepseek3")
    if any(n in hay for n in ("deepseek-v2", "deepseek v2", "deepseek2")):
        add("deepseek2")
    if "deepseek" in hay:
        add("deepseek")
    if any(n in hay for n in ("glm-4.5", "glm 4.5", "glm4.5", "glm-4.7", "glm 4.7", "glm4.7", "glmedge", "glm-edge", "venice", "flash")):
        add("glmedge")
        add("chatglm4")
    elif any(n in hay for n in ("chatglm4", "glm-4", "glm 4")):
        add("chatglm4")
        add("glmedge")
    if any(n in hay for n in ("chatglm3", "glm-3", "glm 3")):
        add("chatglm3")
    if "gemma" in hay:
        add("gemma")
    if any(n in hay for n in ("phi-4", "phi 4", "phi4")):
        add("phi4")
    if any(n in hay for n in ("phi-3", "phi 3", "phi3")):
        add("phi3")
    if any(n in hay for n in ("mistral", "mixtral", "ministral", "magistral")):
        add("mistral-v7")
        add("mistral-v3")
        add("chatml")
    if any(n in hay for n in ("command-r", "command r", "commandr")):
        add("command-r")
    if "vicuna" in hay:
        add("vicuna")
    if "zephyr" in hay:
        add("zephyr")
    if "openchat" in hay:
        add("openchat")
    if any(n in hay for n in ("gpt-oss", "gpt_oss", "gpt oss")):
        add("gpt-oss")
    if "granite" in hay:
        add("granite")
    if any(n in hay for n in ("seed-oss", "seed_oss", "seed oss")):
        add("seed_oss")
    if "solar" in hay:
        add("solar-open")

    return out


def _guess_template_from_model_path(model_path: str) -> Tuple[str, str]:
    return _candidate_templates_from_model_path(model_path)[0] if _candidate_templates_from_model_path(model_path) else ("auto", "")


def _looks_like_bad_assistant_reply(user_text: str, reply_text: str) -> bool:
    user = (user_text or "").strip().lower()
    reply = (reply_text or "").strip()
    low = reply.lower()

    if not reply:
        return True
    if len(reply) < 4:
        return True

    generic_markers = [
        "i'm an ai assistant",
        "i am an ai assistant",
        "i can help with a wide range of tasks",
        "here are some examples of things i can do",
        "i aim to be helpful",
        "i can assist with",
        "as an ai assistant",
    ]
    if any(m in low for m in generic_markers):
        return True

    if "capital of" in user:
        capitals = {
            "belgium": ["brussels"],
            "japan": ["tokyo"],
            "france": ["paris"],
            "germany": ["berlin"],
            "italy": ["rome"],
            "spain": ["madrid"],
            "netherlands": ["amsterdam"],
            "china": ["beijing"],
            "india": ["new delhi", "delhi"],
            "usa": ["washington"],
            "united states": ["washington"],
            "uk": ["london"],
            "united kingdom": ["london"],
        }
        for country, answers in capitals.items():
            if country in user and not any(a in low for a in answers):
                return True

    if user.endswith("?") and low.startswith(("i can help", "i'm here to help", "sure, i'd be happy to help")):
        return True

    return False


def _message_content_to_text(content) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).lower()
                if item_type in ("text", "output_text", "reasoning", "thinking"):
                    parts.append(str(item.get("text", item.get("content", ""))))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join([p for p in parts if str(p).strip()]).strip()
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text", "")).strip()
        if "content" in content:
            return str(content.get("content", "")).strip()
    return str(content or "").strip()


def _split_inline_reasoning(text: str) -> Tuple[str, str]:
    raw = str(text or "").strip()
    if not raw:
        return "", ""

    for pattern in (r"<think>(.*?)</think>", r"<thinking>(.*?)</thinking>", r"<reasoning>(.*?)</reasoning>"):
        m = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
        if m:
            answer = re.sub(pattern, "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
            extracted = m.group(1).strip()
            if answer:
                return answer, extracted
            return "[Model returned an empty response]", extracted

    m = re.search(
        r"^\s*(?:thinking|reasoning|thought process)\s*:\s*(.+?)\n\s*(?:answer|final answer|response)\s*:\s*(.+)$",
        raw,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(2).strip(), m.group(1).strip()

    return raw, ""


def _filename_words_from_prompt(prompt: str, max_words: int = 5) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", " ", str(prompt or "")).strip().lower()
    words = [w for w in clean.split() if w]
    if not words:
        return "image"
    return "_".join(words[:max_words])[:80].strip("_") or "image"


def _mime_to_extension(mime: str, fallback: str = ".png") -> str:
    mime = str(mime or "").strip().lower()
    mapping = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
    }
    return mapping.get(mime, fallback)


def _extract_images_from_response_payload(payload: Any) -> List[Dict[str, str]]:
    found: List[Dict[str, str]] = []
    seen = set()

    def add_image(b64_data: str, mime: str = "image/png", name_hint: str = ""):
        raw = str(b64_data or "").strip()
        if not raw:
            return
        if raw.startswith("data:image/"):
            header, _, encoded = raw.partition(",")
            mime_local = mime
            m = re.match(r"data:([^;]+);base64$", header, re.IGNORECASE)
            if m:
                mime_local = m.group(1).strip().lower() or mime_local
            raw = encoded.strip()
            mime = mime_local
        compact = re.sub(r"\s+", "", raw)
        try:
            base64.b64decode(compact, validate=True)
        except Exception:
            return
        key = (mime, compact[:128], len(compact))
        if key in seen:
            return
        seen.add(key)
        found.append({"data_b64": compact, "mime": str(mime or "image/png"), "name_hint": str(name_hint or "")})

    def walk_string(text: str):
        raw = str(text or "")
        if not raw:
            return
        for m in re.finditer(r"data:(image/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)", raw, re.IGNORECASE):
            add_image(m.group(2), m.group(1), "")
        for m in re.finditer(r'"(?:b64_json|image_base64|data)"\s*:\s*"([A-Za-z0-9+/=\\r\\n\\t ]{256,})"', raw, re.IGNORECASE):
            add_image(m.group(1), "image/png", "")

    def walk(node: Any):
        if isinstance(node, dict):
            node_type = str(node.get("type", "") or "").lower()
            if "b64_json" in node:
                add_image(node.get("b64_json", ""), str(node.get("mime", "") or "image/png"), str(node.get("name", "") or node.get("filename", "") or ""))
            if "image_base64" in node:
                add_image(node.get("image_base64", ""), str(node.get("mime", "") or "image/png"), str(node.get("name", "") or node.get("filename", "") or ""))
            if "data" in node and node_type.startswith("image") and isinstance(node.get("data"), str):
                add_image(node.get("data", ""), str(node.get("mime", "") or "image/png"), str(node.get("name", "") or node.get("filename", "") or ""))
            if "url" in node and isinstance(node.get("url"), str) and str(node.get("url", "")).startswith("data:image/"):
                add_image(node.get("url", ""), str(node.get("mime", "") or "image/png"), str(node.get("name", "") or node.get("filename", "") or ""))
            if "image_url" in node:
                image_url = node.get("image_url")
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", "") or "")
                    if url.startswith("data:image/"):
                        add_image(url, str(node.get("mime", "") or image_url.get("mime", "") or "image/png"), str(node.get("name", "") or node.get("filename", "") or ""))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            walk_string(node)

    walk(payload)
    return found

def _http_get_json(url: str, timeout: float = 10.0) -> Tuple[int, Dict]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.getcode()), json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw) if raw.strip() else {}
        except Exception:
            data = {"error": {"message": raw or str(e)}}
        return int(e.code), data


def _http_post_json(url: str, payload: Dict, timeout: float = 120.0) -> Tuple[int, Dict]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return int(resp.getcode()), json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(raw) if raw.strip() else {}
        except Exception:
            data = {"error": {"message": raw or str(e)}}
        return int(e.code), data


# -----------------------------
# Attachments
# -----------------------------
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus"}
TEXT_EXTS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".html", ".css", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".csv", ".log", ".bat", ".ps1", ".cpp", ".c", ".h", ".hpp", ".java", ".rs", ".go", ".php", ".sql"}
MAX_TEXT_ATTACHMENT_CHARS = 24000
MAX_IMAGE_INLINE_BYTES = 6 * 1024 * 1024

def _attachment_kind_from_path(path: str) -> str:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in TEXT_EXTS:
        return "text"
    return "file"

def _make_attachment_entry(path: str) -> Dict:
    path = os.path.abspath(path)
    kind = _attachment_kind_from_path(path)
    mime = mimetypes.guess_type(path)[0] or ("image/png" if kind == "image" else "text/plain" if kind == "text" else "application/octet-stream")
    return {
        "path": path,
        "name": os.path.basename(path),
        "kind": kind,
        "mime": mime,
    }

def _attachment_label(att: Dict) -> str:
    kind = str(att.get("kind", "file") or "file")
    prefix = "🖼" if kind == "image" else "🎬" if kind == "video" else "🎵" if kind == "audio" else "📄" if kind == "text" else "📎"
    return f"{prefix} {att.get('name', 'attachment')}"

def _read_text_attachment(path: str, max_chars: int = MAX_TEXT_ATTACHMENT_CHARS) -> str:
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        data = f.read(max_chars + 1)
    if len(data) > max_chars:
        data = data[:max_chars] + "\n\n[Attachment truncated to fit context]"
    return data

def _image_attachment_to_data_url(att: Dict) -> str:
    cached = str(att.get("data_url", "") or "")
    if cached:
        return cached
    path = str(att.get("path", "") or "")
    if not path or not os.path.isfile(path):
        return ""
    raw = Path(path).read_bytes()
    if len(raw) > MAX_IMAGE_INLINE_BYTES:
        raise ValueError(f"Image attachment too large for inline upload: {os.path.basename(path)}")
    mime = str(att.get("mime", "") or mimetypes.guess_type(path)[0] or "image/png")
    data_url = f"data:{mime};base64," + base64.b64encode(raw).decode('ascii')
    att["data_url"] = data_url
    return data_url

def _attachment_summary_lines(attachments: List[Dict]) -> List[str]:
    lines: List[str] = []
    for att in attachments or []:
        lines.append(_attachment_label(att))
    return lines


def _format_file_size(num_bytes: int) -> str:
    try:
        num = float(max(0, int(num_bytes)))
    except Exception:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while num >= 1024.0 and idx < len(units) - 1:
        num /= 1024.0
        idx += 1
    return f"{num:.1f} {units[idx]}" if idx else f"{int(num)} {units[idx]}"


def _attachment_display_text(att: Dict) -> str:
    name = str(att.get("name", "attachment") or "attachment")
    kind = str(att.get("kind", "file") or "file")
    try:
        size_text = _format_file_size(os.path.getsize(str(att.get("path", "") or "")))
    except Exception:
        size_text = "missing"
    label = "Image" if kind == "image" else "Video" if kind == "video" else "Audio" if kind == "audio" else "Text" if kind == "text" else "File"
    return f"{name}\n{label} • {size_text}"


def _make_attachment_thumbnail(att: Dict, size: int = 56) -> QtGui.QIcon:
    kind = str(att.get("kind", "file") or "file")
    path = str(att.get("path", "") or "")
    name = str(att.get("name", "attachment") or "attachment")
    canvas = QtGui.QPixmap(size, size)
    canvas.fill(QtCore.Qt.transparent)

    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
    painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)

    rect = QtCore.QRectF(0.5, 0.5, size - 1.0, size - 1.0)
    path_shape = QtGui.QPainterPath()
    path_shape.addRoundedRect(rect, 10, 10)

    if kind == "image":
        painter.fillPath(path_shape, QtGui.QColor(34, 40, 52))
        pix = QtGui.QPixmap(path) if path and os.path.isfile(path) else QtGui.QPixmap()
        if not pix.isNull():
            scaled = pix.scaled(size - 8, size - 8, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            clip = QtGui.QPainterPath()
            clip.addRoundedRect(QtCore.QRectF(2, 2, size - 4, size - 4), 8, 8)
            painter.setClipPath(clip)
            painter.drawPixmap(int((size - scaled.width()) / 2), int((size - scaled.height()) / 2), scaled)
            painter.setClipping(False)
        else:
            painter.setPen(QtGui.QPen(QtGui.QColor(196, 208, 231)))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(QtCore.QRect(0, 0, size, size), QtCore.Qt.AlignCenter, "IMG")
    else:
        bg = QtGui.QColor(70, 92, 135) if kind == "text" else QtGui.QColor(92, 84, 118)
        painter.fillPath(path_shape, bg)
        ext = os.path.splitext(name)[1].lower().lstrip(".")[:4].upper()
        short = ext or ("TXT" if kind == "text" else "FILE")
        painter.setPen(QtGui.QPen(QtGui.QColor(245, 247, 252)))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(11 if len(short) <= 3 else 9)
        painter.setFont(font)
        painter.drawText(QtCore.QRect(4, 6, size - 8, size - 12), QtCore.Qt.AlignCenter, short)

    painter.setPen(QtGui.QPen(QtGui.QColor(58, 67, 84), 1))
    painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, size - 1.0, size - 1.0), 10, 10)
    painter.end()
    return QtGui.QIcon(canvas)


def _open_local_path(path: str, open_with_dialog: bool = False) -> Tuple[bool, str]:
    path = os.path.abspath(str(path or ""))
    if not path or not os.path.exists(path):
        return False, "File no longer exists on disk."
    try:
        if open_with_dialog and sys.platform.startswith("win"):
            subprocess.Popen(["rundll32.exe", "shell32.dll,OpenAs_RunDLL", path])
            return True, ""
        if sys.platform.startswith("win"):
            os.startfile(path)
            return True, ""
        ok = QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))
        return (True, "") if ok else (False, "Could not hand file off to the operating system.")
    except Exception as e:
        return False, str(e)


class AttachmentPreviewDialog(QtWidgets.QDialog):
    def __init__(self, att: Dict, parent=None):
        super().__init__(parent)
        self.att = dict(att or {})
        self.path = os.path.abspath(str(self.att.get("path", "") or ""))
        self.kind = str(self.att.get("kind", "file") or "file")
        self.name = str(self.att.get("name", "attachment") or "attachment")
        self.mime = str(self.att.get("mime", "") or mimetypes.guess_type(self.path)[0] or "application/octet-stream")

        self.setWindowTitle(f"Preview - {self.name}")
        self.resize(920, 720)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)

        hdr = QtWidgets.QFrame()
        hdr_lay = QtWidgets.QVBoxLayout(hdr)
        hdr_lay.setContentsMargins(0, 0, 0, 0)
        hdr_lay.setSpacing(4)
        lbl_name = QtWidgets.QLabel(self.name)
        font = lbl_name.font()
        font.setPointSize(font.pointSize() + 1)
        font.setBold(True)
        lbl_name.setFont(font)
        hdr_lay.addWidget(lbl_name)

        try:
            size_text = _format_file_size(os.path.getsize(self.path))
        except Exception:
            size_text = "missing"
        lbl_meta = QtWidgets.QLabel(f"{self.kind} • {size_text} • {self.mime}")
        lbl_meta.setStyleSheet("color:#9aa3b3;")
        hdr_lay.addWidget(lbl_meta)

        lbl_path = QtWidgets.QLabel(self.path or "")
        lbl_path.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lbl_path.setWordWrap(True)
        lbl_path.setStyleSheet("color:#9aa3b3;")
        hdr_lay.addWidget(lbl_path)
        lay.addWidget(hdr)

        if self.kind == "image" and self.path and os.path.isfile(self.path):
            pix = QtGui.QPixmap(self.path)
            if not pix.isNull():
                area = QtWidgets.QScrollArea()
                area.setWidgetResizable(True)
                holder = QtWidgets.QWidget()
                holder_lay = QtWidgets.QVBoxLayout(holder)
                holder_lay.setContentsMargins(10, 10, 10, 10)
                holder_lay.addStretch(1)
                lbl_img = QtWidgets.QLabel()
                lbl_img.setAlignment(QtCore.Qt.AlignCenter)
                target = pix
                if pix.width() > 1400 or pix.height() > 900:
                    target = pix.scaled(1400, 900, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                lbl_img.setPixmap(target)
                holder_lay.addWidget(lbl_img, 0, QtCore.Qt.AlignCenter)
                holder_lay.addStretch(1)
                area.setWidget(holder)
                lay.addWidget(area, 1)
            else:
                msg = QtWidgets.QLabel("Could not decode this image for preview.")
                msg.setAlignment(QtCore.Qt.AlignCenter)
                lay.addWidget(msg, 1)
        elif self.kind == "text" and self.path and os.path.isfile(self.path):
            viewer = QtWidgets.QPlainTextEdit()
            viewer.setReadOnly(True)
            viewer.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
            try:
                viewer.setPlainText(_read_text_attachment(self.path, max_chars=120000))
            except Exception as e:
                viewer.setPlainText(f"Could not read file:\n{e}")
            lay.addWidget(viewer, 1)
        else:
            info = QtWidgets.QPlainTextEdit()
            info.setReadOnly(True)
            info.setPlainText(
                f"Name: {self.name}\n"
                f"Kind: {self.kind}\n"
                f"Mime: {self.mime}\n"
                f"Path: {self.path}\n"
                f"Status: {'Found on disk' if self.path and os.path.exists(self.path) else 'Missing'}"
            )
            lay.addWidget(info, 1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.btn_open = QtWidgets.QPushButton("Open")
        self.btn_close = QtWidgets.QPushButton("Close")
        btn_row.addWidget(self.btn_open)
        if self.kind in {"text", "file"}:
            self.btn_open_with = QtWidgets.QPushButton("Open with…")
            btn_row.addWidget(self.btn_open_with)
            self.btn_open_with.clicked.connect(self._open_with)
        self.btn_close.clicked.connect(self.accept)
        self.btn_open.clicked.connect(self._open_default)
        btn_row.addWidget(self.btn_close)
        lay.addLayout(btn_row)

    def _open_default(self):
        ok, err = _open_local_path(self.path, open_with_dialog=False)
        if ok:
            self.accept()
            return
        QtWidgets.QMessageBox.warning(self, "Open failed", err or "Could not open file.")

    def _open_with(self):
        ok, err = _open_local_path(self.path, open_with_dialog=True)
        if ok:
            return
        QtWidgets.QMessageBox.warning(self, "Open with failed", err or "Could not open file chooser.")

# -----------------------------
# Data models
# -----------------------------
@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: str


@dataclass
class ChatSession:
    id: str
    title: str
    created_at: str
    updated_at: str
    model_path: str
    template_kind: str
    template_value: str
    system_prompt: str
    messages: List[Dict]

    @staticmethod
    def create_default() -> "ChatSession":
        now = datetime.now().isoformat(timespec="seconds")
        return ChatSession(
            id=str(uuid.uuid4()),
            title="New Chat",
            created_at=now,
            updated_at=now,
            model_path="",
            template_kind="auto",
            template_value="",
            system_prompt="",
            messages=[],
        )


# -----------------------------
# UI building blocks
# -----------------------------
class PromptEdit(QtWidgets.QPlainTextEdit):
    sendRequested = QtCore.Signal()
    imagePasted = QtCore.Signal(object)
    fileUrlsPasted = QtCore.Signal(object)

    def _try_emit_attachment_paste(self, mime: Optional[QtCore.QMimeData]) -> bool:
        if mime is None:
            return False
        urls = []
        try:
            if mime.hasUrls():
                for url in mime.urls():
                    if url.isLocalFile():
                        local = url.toLocalFile()
                        if local:
                            urls.append(local)
        except Exception:
            urls = []
        if urls:
            self.fileUrlsPasted.emit(urls)
            return True
        try:
            if mime.hasImage():
                image = QtGui.QImage(mime.imageData())
                if not image.isNull():
                    self.imagePasted.emit(image)
                    return True
        except Exception:
            pass
        return False

    def insertFromMimeData(self, source: QtCore.QMimeData) -> None:
        if self._try_emit_attachment_paste(source):
            return
        super().insertFromMimeData(source)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if event.matches(QtGui.QKeySequence.Paste) or (event.key() == QtCore.Qt.Key_Insert and event.modifiers() & QtCore.Qt.ShiftModifier):
            mime = QtWidgets.QApplication.clipboard().mimeData()
            if self._try_emit_attachment_paste(mime):
                event.accept()
                return
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if event.modifiers() & QtCore.Qt.ShiftModifier:
                return super().keyPressEvent(event)
            self.sendRequested.emit()
            event.accept()
            return
        return super().keyPressEvent(event)



def _chat_media_time_text(ms: int) -> str:
    try:
        sec = max(0, int(ms // 1000))
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}"
        return f"{m:d}:{s:02d}"
    except Exception:
        return "0:00"


class ChatVideoPopup(QtWidgets.QDialog):
    def __init__(self, path: str, start_position: int = 0, parent=None):
        super().__init__(parent)
        self.path = os.path.abspath(str(path or ""))
        self.setWindowTitle(os.path.basename(self.path) or "Video preview")
        self.resize(980, 620)
        self._dragging = False

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        if QtMultimedia is None or QtMultimediaWidgets is None or not self.path or not os.path.isfile(self.path):
            msg = QtWidgets.QLabel("Video preview is not available. Use Open external player instead.")
            msg.setAlignment(QtCore.Qt.AlignCenter)
            lay.addWidget(msg, 1)
            return

        self.video = QtMultimediaWidgets.QVideoWidget(self)
        self.video.setMinimumSize(640, 360)
        lay.addWidget(self.video, 1)

        row = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QToolButton(); self.btn_play.setText("▶")
        self.btn_pause = QtWidgets.QToolButton(); self.btn_pause.setText("⏸")
        self.btn_stop = QtWidgets.QToolButton(); self.btn_stop.setText("■")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.time_lbl = QtWidgets.QLabel("0:00 / 0:00")
        self.time_lbl.setMinimumWidth(110)
        row.addWidget(self.btn_play); row.addWidget(self.btn_pause); row.addWidget(self.btn_stop)
        row.addWidget(self.slider, 1); row.addWidget(self.time_lbl)
        lay.addLayout(row)

        self.player = QtMultimedia.QMediaPlayer(self)
        self.audio = QtMultimedia.QAudioOutput(self)
        try:
            self.audio.setVolume(0.85)
            self.player.setAudioOutput(self.audio)
            self.player.setVideoOutput(self.video)
            self.player.setSource(QtCore.QUrl.fromLocalFile(self.path))
        except Exception:
            pass

        self.btn_play.clicked.connect(self.player.play)
        self.btn_pause.clicked.connect(self.player.pause)
        self.btn_stop.clicked.connect(self.player.stop)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.positionChanged.connect(self._position_changed)
        self.slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
        self.slider.sliderReleased.connect(self._seek_from_slider)
        self.slider.sliderMoved.connect(lambda v: self.time_lbl.setText(f"{_chat_media_time_text(v)} / {_chat_media_time_text(self.player.duration())}"))
        if start_position:
            QtCore.QTimer.singleShot(150, lambda: self.player.setPosition(int(start_position)))

    def _duration_changed(self, duration: int):
        try:
            self.slider.setRange(0, max(0, int(duration)))
            self.time_lbl.setText(f"{_chat_media_time_text(self.player.position())} / {_chat_media_time_text(duration)}")
        except Exception:
            pass

    def _position_changed(self, position: int):
        try:
            if not self._dragging:
                self.slider.setValue(max(0, int(position)))
            self.time_lbl.setText(f"{_chat_media_time_text(position)} / {_chat_media_time_text(self.player.duration())}")
        except Exception:
            pass

    def _seek_from_slider(self):
        try:
            self._dragging = False
            self.player.setPosition(int(self.slider.value()))
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            self.player.stop()
        except Exception:
            pass
        super().closeEvent(event)


class ChatMediaPlayerWidget(QtWidgets.QFrame):
    def __init__(self, path: str, kind: str = "audio", parent=None):
        super().__init__(parent)
        self.path = os.path.abspath(str(path or ""))
        self.kind = str(kind or "audio").lower()
        self._dragging = False
        self._popup_refs: list[QtWidgets.QDialog] = []
        self.setObjectName("ChatMediaPlayer")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        name = os.path.basename(self.path) or "media"
        title = QtWidgets.QLabel(("🎬 " if self.kind == "video" else "🎵 ") + name)
        title.setObjectName("AttachmentText")
        title.setWordWrap(True)
        title.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(title)

        self.player = None
        self.audio = None
        self.btn_popup = None

        # Video: keep the attachment info in chat, but do not embed the small
        # inline player. Some video backends work better in the dedicated popup.
        # Audio keeps the mini player because that path has been tested working.
        if self.kind == "video":
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            row.addStretch(1)
            self.btn_popup = QtWidgets.QPushButton("Show in popup player")
            self.btn_popup.setToolTip("Open this video in a resizable chat popup player")
            self.btn_popup.clicked.connect(self._open_popup)
            row.addWidget(self.btn_popup, 0)
            lay.addLayout(row)
            return

        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self.btn_play = QtWidgets.QToolButton(); self.btn_play.setText("▶"); self.btn_play.setToolTip("Play")
        self.btn_pause = QtWidgets.QToolButton(); self.btn_pause.setText("⏸"); self.btn_pause.setToolTip("Pause")
        self.btn_stop = QtWidgets.QToolButton(); self.btn_stop.setText("■"); self.btn_stop.setToolTip("Stop")
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.time_lbl = QtWidgets.QLabel("0:00 / 0:00")
        self.time_lbl.setMinimumWidth(105)
        row.addWidget(self.btn_play); row.addWidget(self.btn_pause); row.addWidget(self.btn_stop)
        row.addWidget(self.slider, 1); row.addWidget(self.time_lbl)
        lay.addLayout(row)

        if QtMultimedia is None or not self.path or not os.path.isfile(self.path):
            self.btn_play.setEnabled(False); self.btn_pause.setEnabled(False); self.btn_stop.setEnabled(False); self.slider.setEnabled(False)
            fallback = QtWidgets.QPushButton("Open external player")
            fallback.clicked.connect(lambda: _open_local_path(self.path, open_with_dialog=False))
            lay.addWidget(fallback, 0, QtCore.Qt.AlignLeft)
            return

        try:
            self.player = QtMultimedia.QMediaPlayer(self)
            self.audio = QtMultimedia.QAudioOutput(self)
            self.audio.setVolume(0.85)
            self.player.setAudioOutput(self.audio)
            self.player.setSource(QtCore.QUrl.fromLocalFile(self.path))
            self.btn_play.clicked.connect(self.player.play)
            self.btn_pause.clicked.connect(self.player.pause)
            self.btn_stop.clicked.connect(self.player.stop)
            self.player.durationChanged.connect(self._duration_changed)
            self.player.positionChanged.connect(self._position_changed)
            self.slider.sliderPressed.connect(lambda: setattr(self, "_dragging", True))
            self.slider.sliderReleased.connect(self._seek_from_slider)
            self.slider.sliderMoved.connect(lambda v: self.time_lbl.setText(f"{_chat_media_time_text(v)} / {_chat_media_time_text(self.player.duration() if self.player else 0)}"))
        except Exception:
            self.btn_play.setEnabled(False); self.btn_pause.setEnabled(False); self.btn_stop.setEnabled(False); self.slider.setEnabled(False)

    def _duration_changed(self, duration: int):
        try:
            self.slider.setRange(0, max(0, int(duration)))
            pos = self.player.position() if self.player else 0
            self.time_lbl.setText(f"{_chat_media_time_text(pos)} / {_chat_media_time_text(duration)}")
        except Exception:
            pass

    def _position_changed(self, position: int):
        try:
            if not self._dragging:
                self.slider.setValue(max(0, int(position)))
            dur = self.player.duration() if self.player else 0
            self.time_lbl.setText(f"{_chat_media_time_text(position)} / {_chat_media_time_text(dur)}")
        except Exception:
            pass

    def _seek_from_slider(self):
        try:
            self._dragging = False
            if self.player:
                self.player.setPosition(int(self.slider.value()))
        except Exception:
            pass

    def _open_popup(self):
        try:
            pos = self.player.position() if self.player else 0
        except Exception:
            pos = 0
        dlg = ChatVideoPopup(self.path, pos, self.window())
        self._popup_refs.append(dlg)
        dlg.finished.connect(lambda *_: self._popup_refs.remove(dlg) if dlg in self._popup_refs else None)
        dlg.show()

    def stop(self):
        try:
            if self.player:
                self.player.stop()
        except Exception:
            pass

class MessageBubble(QtWidgets.QFrame):
    editRequested = QtCore.Signal(str)
    editSaved = QtCore.Signal(str, str)
    retryRequested = QtCore.Signal(str)
    saveToMemoryRequested = QtCore.Signal(str)
    saveToProjectRequested = QtCore.Signal(str)
    versionPrevRequested = QtCore.Signal(str)
    versionNextRequested = QtCore.Signal(str)

    def __init__(self, role: str, text: str, thinking: str = "", attachments: Optional[List[Dict]] = None, loading: bool = False, parent=None, message_id: str = ""):
        super().__init__(parent)
        self.role = role
        self.message_id = str(message_id or "")
        self._attachments: List[Dict[str, Any]] = list(attachments or [])
        self._edit_box: Optional[QtWidgets.QPlainTextEdit] = None
        self._edit_controls: Optional[QtWidgets.QWidget] = None
        self._version_frame: Optional[QtWidgets.QWidget] = None
        self._version_label: Optional[QtWidgets.QLabel] = None
        self._version_prev_btn: Optional[QtWidgets.QToolButton] = None
        self._version_next_btn: Optional[QtWidgets.QToolButton] = None

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)

        self.bubble = QtWidgets.QFrame()
        self.bubble.setObjectName(f"Bubble_{role}")
        self.bubble_lay = QtWidgets.QVBoxLayout(self.bubble)
        self.bubble_lay.setContentsMargins(14, 10, 14, 10)
        self.bubble_lay.setSpacing(8)

        role_lbl = QtWidgets.QLabel("You" if role == "user" else ("Assistant" if role == "assistant" else "Info"))
        role_lbl.setObjectName("BubbleRole")
        self.bubble_lay.addWidget(role_lbl)

        self.att_frame: Optional[QtWidgets.QFrame] = None
        self.think_frame: Optional[QtWidgets.QFrame] = None
        self.think_lbl: Optional[QtWidgets.QLabel] = None

        if role == "assistant" and (thinking or "").strip():
            self._ensure_thinking_frame()
            self.think_lbl.setText((thinking or "").strip())

        self.text_lbl = QtWidgets.QLabel(text or "")
        if role == "assistant":
            self.text_lbl.setObjectName("AnswerText")
        elif role == "user":
            self.text_lbl.setObjectName("UserText")
        else:
            self.text_lbl.setObjectName("InfoText")
        self.text_lbl.setWordWrap(True)
        self.text_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        try:
            self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.bubble.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.text_lbl.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.customContextMenuRequested.connect(lambda pos: self._show_message_menu(pos))
            self.bubble.customContextMenuRequested.connect(lambda pos: self._show_message_menu(pos))
            self.text_lbl.customContextMenuRequested.connect(lambda pos: self._show_message_menu(pos))
        except Exception:
            pass

        self._rebuild_attachments(self._attachments)

        self.bubble_lay.addWidget(self.text_lbl)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setVisible(bool(loading))
        self.bubble_lay.addWidget(self.progress_bar)

        self.bubble.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        outer.addWidget(self.bubble, 1)

    def _ensure_thinking_frame(self):
        if self.think_frame is not None:
            return
        self.think_frame = QtWidgets.QFrame()
        self.think_frame.setObjectName("ThinkingFrame")
        think_lay = QtWidgets.QVBoxLayout(self.think_frame)
        think_lay.setContentsMargins(10, 8, 10, 8)
        think_lay.setSpacing(4)

        think_title = QtWidgets.QLabel("Thinking")
        think_title.setObjectName("ThinkingTitle")
        think_lay.addWidget(think_title)

        self.think_lbl = QtWidgets.QLabel("")
        self.think_lbl.setObjectName("ThinkingText")
        self.think_lbl.setWordWrap(True)
        self.think_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        think_lay.addWidget(self.think_lbl)
        self.bubble_lay.addWidget(self.think_frame)

    def _clear_attachment_frame(self):
        if self.att_frame is not None:
            self.att_frame.deleteLater()
            self.att_frame = None

    def _rebuild_attachments(self, attachments: Optional[List[Dict]] = None):
        attachments = list(attachments or [])
        self._attachments = attachments
        self._clear_attachment_frame()
        if not attachments:
            return

        self.att_frame = QtWidgets.QFrame()
        self.att_frame.setObjectName("AttachmentFrame")
        att_lay = QtWidgets.QVBoxLayout(self.att_frame)
        att_lay.setContentsMargins(10, 8, 10, 8)
        att_lay.setSpacing(6)

        att_title = QtWidgets.QLabel("Attachments")
        att_title.setObjectName("AttachmentTitle")
        att_lay.addWidget(att_title)

        for att in attachments:
            kind = str(att.get("kind", "") or "")
            path = str(att.get("path", "") or "")
            name = str(att.get("name", os.path.basename(path) or "attachment") or "attachment")
            if kind == "image" and path and os.path.isfile(path):
                holder = QtWidgets.QFrame()
                holder_lay = QtWidgets.QVBoxLayout(holder)
                holder_lay.setContentsMargins(0, 0, 0, 0)
                holder_lay.setSpacing(4)

                name_lbl = QtWidgets.QLabel(name)
                name_lbl.setObjectName("AttachmentText")
                name_lbl.setWordWrap(True)
                holder_lay.addWidget(name_lbl)

                preview = QtWidgets.QLabel()
                preview.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
                preview.setMinimumHeight(120)
                pm = QtGui.QPixmap(path)
                if not pm.isNull():
                    preview.setPixmap(pm.scaled(360, 260, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                else:
                    preview.setText("[Could not load image preview]")
                    preview.setObjectName("AttachmentText")
                try:
                    preview.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
                    preview.customContextMenuRequested.connect(lambda pos, _p=path, _w=preview: self._attachment_image_context_menu(_p, _w, pos))
                    preview.setToolTip(path)
                except Exception:
                    pass
                holder_lay.addWidget(preview)
                att_lay.addWidget(holder)
            elif kind in {"video", "audio"} and path and os.path.isfile(path):
                holder = QtWidgets.QFrame()
                holder.setObjectName("AttachmentMediaFrame")
                holder_lay = QtWidgets.QVBoxLayout(holder)
                holder_lay.setContentsMargins(0, 0, 0, 0)
                holder_lay.setSpacing(6)
                player = ChatMediaPlayerWidget(path, kind, holder)
                holder_lay.addWidget(player)
                try:
                    holder.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
                    player.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
                    holder.customContextMenuRequested.connect(lambda pos, _p=path, _w=holder: self._attachment_media_context_menu(_p, _w, pos))
                    player.customContextMenuRequested.connect(lambda pos, _p=path, _w=player: self._attachment_media_context_menu(_p, _w, pos))
                    holder.setToolTip(path)
                    player.setToolTip(path)
                except Exception:
                    pass
                att_lay.addWidget(holder)
            else:
                lbl = QtWidgets.QLabel(_attachment_label(att))
                lbl.setObjectName("AttachmentText")
                lbl.setWordWrap(True)
                lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                att_lay.addWidget(lbl)

        self.bubble_lay.insertWidget(1, self.att_frame)

    def _attachment_image_context_menu(self, path: str, widget: QtWidgets.QWidget, pos: QtCore.QPoint):
        try:
            host = self
            while host is not None:
                fn = getattr(host, "_show_chat_image_context_menu", None)
                if callable(fn):
                    fn(str(path or ""), widget, pos)
                    return
                host = host.parent()
            host = self.window()
            fn = getattr(host, "_show_chat_image_context_menu", None)
            if callable(fn):
                fn(str(path or ""), widget, pos)
        except Exception:
            pass

    def _attachment_media_context_menu(self, path: str, widget: QtWidgets.QWidget, pos: QtCore.QPoint):
        try:
            host = self
            while host is not None:
                fn = getattr(host, "_show_chat_media_context_menu", None)
                if callable(fn):
                    fn(str(path or ""), widget, pos)
                    return
                host = host.parent()
            host = self.window()
            fn = getattr(host, "_show_chat_media_context_menu", None)
            if callable(fn):
                fn(str(path or ""), widget, pos)
        except Exception:
            pass

    def _show_message_menu(self, pos: QtCore.QPoint):
        menu = QtWidgets.QMenu(self)
        if self.role == "user":
            act_edit = menu.addAction("Edit")
        else:
            act_edit = None
        if self.role == "assistant":
            act_retry = menu.addAction("Retry")
        else:
            act_retry = None
        if self.role in ("user", "assistant"):
            menu.addSeparator()
            act_save_memory = menu.addAction("Save to memory")
            act_save_project = menu.addAction("Save to project")
            menu.addSeparator()
        else:
            act_save_memory = None
            act_save_project = None
        act_copy = menu.addAction("Copy")
        action = menu.exec(QtGui.QCursor.pos())
        if action is None:
            return
        if act_edit is not None and action == act_edit:
            self.start_edit()
            self.editRequested.emit(self.message_id)
            return
        if act_retry is not None and action == act_retry:
            self.retryRequested.emit(self.message_id)
            return
        if act_save_memory is not None and action == act_save_memory:
            self.saveToMemoryRequested.emit(self.message_id)
            return
        if act_save_project is not None and action == act_save_project:
            self.saveToProjectRequested.emit(self.message_id)
            return
        if action == act_copy:
            try:
                QtWidgets.QApplication.clipboard().setText(self.text_lbl.text())
            except Exception:
                pass

    def start_edit(self):
        if self.role != "user" or self._edit_box is not None:
            return
        self.text_lbl.setVisible(False)
        self._edit_box = QtWidgets.QPlainTextEdit(self.text_lbl.text())
        self._edit_box.setObjectName("MessageEditBox")
        self._edit_box.setMinimumHeight(90)
        self._edit_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        idx = self.bubble_lay.indexOf(self.text_lbl)
        self.bubble_lay.insertWidget(max(0, idx), self._edit_box)

        self._edit_controls = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self._edit_controls)
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        btn_save = QtWidgets.QPushButton("Save")
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_save.setObjectName("SmallActionButton")
        btn_cancel.setObjectName("SmallActionButton")
        row.addWidget(btn_save)
        row.addWidget(btn_cancel)
        self.bubble_lay.insertWidget(max(0, idx + 1), self._edit_controls)
        btn_save.clicked.connect(self._finish_edit_save)
        btn_cancel.clicked.connect(self.cancel_edit)
        self._edit_box.setFocus()

    def _finish_edit_save(self):
        if self._edit_box is None:
            return
        new_text = self._edit_box.toPlainText()
        self.cancel_edit(keep_text=True)
        self.editSaved.emit(self.message_id, new_text)

    def cancel_edit(self, keep_text: bool = False):
        if self._edit_box is not None:
            self._edit_box.deleteLater()
            self._edit_box = None
        if self._edit_controls is not None:
            self._edit_controls.deleteLater()
            self._edit_controls = None
        self.text_lbl.setVisible(True)

    def set_version_controls(self, label: str = "", can_prev: bool = False, can_next: bool = False):
        label = str(label or "").strip()
        if not label:
            if self._version_frame is not None:
                self._version_frame.deleteLater()
                self._version_frame = None
            self._version_label = None
            self._version_prev_btn = None
            self._version_next_btn = None
            return
        if self._version_frame is None:
            self._version_frame = QtWidgets.QWidget()
            lay = QtWidgets.QHBoxLayout(self._version_frame)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(6)
            lay.addStretch(1)
            self._version_prev_btn = QtWidgets.QToolButton()
            self._version_prev_btn.setText("‹")
            self._version_prev_btn.setObjectName("VersionButton")
            self._version_label = QtWidgets.QLabel("")
            self._version_label.setObjectName("VersionLabel")
            self._version_next_btn = QtWidgets.QToolButton()
            self._version_next_btn.setText("›")
            self._version_next_btn.setObjectName("VersionButton")
            lay.addWidget(self._version_prev_btn)
            lay.addWidget(self._version_label)
            lay.addWidget(self._version_next_btn)
            self.bubble_lay.addWidget(self._version_frame)
            self._version_prev_btn.clicked.connect(lambda: self.versionPrevRequested.emit(self.message_id))
            self._version_next_btn.clicked.connect(lambda: self.versionNextRequested.emit(self.message_id))
        if self._version_label is not None:
            self._version_label.setText(label)
        if self._version_prev_btn is not None:
            self._version_prev_btn.setEnabled(bool(can_prev))
        if self._version_next_btn is not None:
            self._version_next_btn.setEnabled(bool(can_next))

    def update_message(self, text: Optional[str] = None, thinking: Optional[str] = None, attachments: Optional[List[Dict]] = None, loading: Optional[bool] = None):
        if text is not None:
            self.text_lbl.setText(text or "")
        if thinking is not None:
            if thinking.strip():
                self._ensure_thinking_frame()
                self.think_lbl.setText(thinking.strip())
            elif self.think_frame is not None:
                self.think_frame.deleteLater()
                self.think_frame = None
                self.think_lbl = None
        if attachments is not None:
            self._rebuild_attachments(attachments)
        if loading is not None:
            self.progress_bar.setVisible(bool(loading))


class ChatScrollArea(QtWidgets.QScrollArea):
    editMessageRequested = QtCore.Signal(str)
    editMessageSaved = QtCore.Signal(str, str)
    retryMessageRequested = QtCore.Signal(str)
    saveToMemoryRequested = QtCore.Signal(str)
    saveToProjectRequested = QtCore.Signal(str)
    versionPrevRequested = QtCore.Signal(str)
    versionNextRequested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        inner = QtWidgets.QWidget()
        self.vbox = QtWidgets.QVBoxLayout(inner)
        self.vbox.setContentsMargins(18, 18, 18, 18)
        self.vbox.setSpacing(10)
        self.vbox.addStretch(1)
        self.setWidget(inner)
        self._message_widgets: Dict[str, MessageBubble] = {}

    def clear_messages(self):
        self._message_widgets = {}
        while self.vbox.count() > 1:
            item = self.vbox.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def add_message(self, role: str, text: str, thinking: str = "", attachments: Optional[List[Dict]] = None, message_id: Optional[str] = None, loading: bool = False):
        mid = str(message_id or uuid.uuid4())
        bubble = MessageBubble(role, text, thinking, attachments, loading, message_id=mid)
        bubble.editRequested.connect(self.editMessageRequested.emit)
        bubble.editSaved.connect(self.editMessageSaved.emit)
        bubble.retryRequested.connect(self.retryMessageRequested.emit)
        bubble.saveToMemoryRequested.connect(self.saveToMemoryRequested.emit)
        bubble.saveToProjectRequested.connect(self.saveToProjectRequested.emit)
        bubble.versionPrevRequested.connect(self.versionPrevRequested.emit)
        bubble.versionNextRequested.connect(self.versionNextRequested.emit)
        self.vbox.insertWidget(self.vbox.count() - 1, bubble)
        self._message_widgets[mid] = bubble
        self.scroll_to_bottom(force=True)
        QtCore.QTimer.singleShot(0, lambda: self.scroll_to_bottom(force=True))
        QtCore.QTimer.singleShot(35, lambda: self.scroll_to_bottom(force=True))
        QtCore.QTimer.singleShot(120, lambda: self.scroll_to_bottom(force=True))
        return mid, bubble

    def add_memory_choice(self, title: str, detail: str, on_this_chat, on_saved_memories):
        frame = QtWidgets.QFrame()
        frame.setObjectName("MemoryChoiceFrame")
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(8)
        lbl = QtWidgets.QLabel(str(title or "Memory question"))
        lbl.setObjectName("BubbleRole")
        lbl.setWordWrap(True)
        lay.addWidget(lbl)
        info = QtWidgets.QLabel(str(detail or "Choose where the answer should come from."))
        info.setObjectName("InfoText")
        info.setWordWrap(True)
        lay.addWidget(info)
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        btn_chat = QtWidgets.QPushButton("This chat")
        btn_saved = QtWidgets.QPushButton("Saved memories")
        btn_chat.setObjectName("SmallActionButton")
        btn_saved.setObjectName("SmallActionButton")
        row.addWidget(btn_chat)
        row.addWidget(btn_saved)
        lay.addLayout(row)

        def _lock(text: str):
            try:
                btn_chat.setEnabled(False)
                btn_saved.setEnabled(False)
                info.setText(text)
            except Exception:
                pass

        def _chat_clicked():
            _lock("Using this chat only…")
            if callable(on_this_chat):
                on_this_chat()

        def _saved_clicked():
            _lock("Using saved memories…")
            if callable(on_saved_memories):
                on_saved_memories()

        btn_chat.clicked.connect(_chat_clicked)
        btn_saved.clicked.connect(_saved_clicked)
        self.vbox.insertWidget(self.vbox.count() - 1, frame)
        self.scroll_to_bottom(force=True)
        return frame

    def update_message(self, message_id: str, text: Optional[str] = None, thinking: Optional[str] = None, attachments: Optional[List[Dict]] = None, loading: Optional[bool] = None):
        bubble = self._message_widgets.get(str(message_id or ""))
        if bubble is None:
            return False
        bubble.update_message(text=text, thinking=thinking, attachments=attachments, loading=loading)
        self.scroll_to_bottom(force=True)
        return True

    def set_version_controls(self, message_id: str, label: str = "", can_prev: bool = False, can_next: bool = False):
        bubble = self._message_widgets.get(str(message_id or ""))
        if bubble is None:
            return False
        bubble.set_version_controls(label, can_prev, can_next)
        return True

    def scroll_to_bottom(self, force: bool = False):
        if self.widget() is not None:
            self.widget().adjustSize()
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents, 5)
        bar = self.verticalScrollBar()
        if force:
            bar.setValue(bar.maximum())
        else:
            bar.triggerAction(QtWidgets.QAbstractSlider.SliderToMaximum)


# -----------------------------
# Background workers
# -----------------------------
class ServerBootThread(QtCore.QThread):
    statusChanged = QtCore.Signal(str)
    ready = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, base_url: str, timeout_s: int = 240, parent=None):
        super().__init__(parent)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def run(self):
        start = time.time()
        while not self.isInterruptionRequested():
            if (time.time() - start) > self.timeout_s:
                self.failed.emit("Timed out while waiting for llama-server to load the model.")
                return
            try:
                code, payload = _http_get_json(f"{self.base_url}/health", timeout=3.0)
                if code == 200:
                    self.ready.emit()
                    return
                if code == 503:
                    msg = payload.get("error", {}).get("message", "Loading model…")
                    self.statusChanged.emit(str(msg))
                else:
                    self.statusChanged.emit(f"Waiting for server… ({code})")
            except Exception:
                self.statusChanged.emit("Starting local llama-server…")
            self.msleep(900)


class ChatCompletionThread(QtCore.QThread):
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, base_url: str, messages: List[Dict], max_tokens: int, temperature: float, top_p: float, parent=None):
        super().__init__(parent)
        self.base_url = base_url.rstrip("/")
        self.messages = messages
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def run(self):
        try:
            payload = {
                "model": "local-model",
                "messages": self.messages,
                "stream": False,
                "max_tokens": int(self.max_tokens),
                "temperature": float(self.temperature),
                "top_p": float(self.top_p),
                "reasoning_format": "none",
            }
            code, data = _http_post_json(f"{self.base_url}/v1/chat/completions", payload, timeout=360.0)
            if code >= 400:
                msg = data.get("error", {}).get("message", f"HTTP {code}")
                self.failed.emit(str(msg))
                return
            choices = data.get("choices") or []
            if not choices:
                self.failed.emit("No choices returned by the local server.")
                return
            message = choices[0].get("message") or {}
            content = _message_content_to_text(message.get("content", ""))
            reasoning = _message_content_to_text(
                message.get("reasoning")
                or message.get("reasoning_content")
                or message.get("thinking")
                or message.get("reasoning_text")
                or ""
            )
            content, inline_reasoning = _split_inline_reasoning(content)
            if not reasoning:
                reasoning = inline_reasoning
            images = _extract_images_from_response_payload(data)
            if not content and images:
                content = f"[Model returned {len(images)} image{'s' if len(images) != 1 else ''}]"
            if not content:
                content = "[Model returned an empty response]"
            self.succeeded.emit({"content": content, "thinking": reasoning, "images": images})
        except Exception as e:
            self.failed.emit(str(e))


class BubbleColorButton(QtWidgets.QPushButton):
    colorChanged = QtCore.Signal(str)

    def __init__(self, color: str, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._color = _normalize_hex_color(color, DEFAULT_BUBBLE_BASE_COLOR)
        self.clicked.connect(self._show_picker)
        self.setMinimumWidth(170)
        self._refresh_appearance()

    def color(self) -> str:
        return self._color

    def setColor(self, color: str, emit: bool = False):
        color = _normalize_hex_color(color, self._color or DEFAULT_BUBBLE_BASE_COLOR)
        changed = (color != self._color)
        self._color = color
        self._refresh_appearance()
        if emit and changed:
            self.colorChanged.emit(self._color)

    def _color_name(self) -> str:
        for label, value in BUBBLE_COLOR_OPTIONS:
            if value.lower() == self._color.lower():
                return label
        return self._color.upper()

    def _refresh_appearance(self):
        swatch = QtGui.QPixmap(18, 18)
        swatch.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(swatch)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(QtGui.QPen(QtGui.QColor("#2a3140"), 1))
        painter.setBrush(QtGui.QColor(self._color))
        painter.drawRoundedRect(QtCore.QRectF(0.5, 0.5, 17.0, 17.0), 4, 4)
        painter.end()
        self.setIcon(QtGui.QIcon(swatch))
        self.setText(self._color_name())
        self.setToolTip(f"{self._title}: {self._color}")

    def _show_picker(self):
        menu = QtWidgets.QMenu(self)
        host = QtWidgets.QWidget(menu)
        grid = QtWidgets.QGridLayout(host)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for idx, (label, value) in enumerate(BUBBLE_COLOR_OPTIONS):
            btn = QtWidgets.QToolButton(host)
            btn.setToolTip(f"{label} {value}")
            btn.setAutoRaise(True)
            btn.setFixedSize(34, 34)
            border = "3px solid #eff3f9" if value.lower() == self._color.lower() else "1px solid #2f3645"
            btn.setStyleSheet(
                f"QToolButton {{ background: {value}; border: {border}; border-radius: 9px; }}"
                "QToolButton:hover { border-color: #ffffff; }"
            )
            btn.clicked.connect(lambda _=False, c=value, m=menu: (m.close(), self.setColor(c, emit=True)))
            grid.addWidget(btn, idx // 4, idx % 4)
        action = QtWidgets.QWidgetAction(menu)
        action.setDefaultWidget(host)
        menu.addAction(action)
        menu.exec(self.mapToGlobal(QtCore.QPoint(0, self.height())))


# -----------------------------
# Settings dialog
# -----------------------------
class SettingsDialog(QtWidgets.QDialog):
    settingsChanged = QtCore.Signal()

    def __init__(self, fv_root: str, parent=None):
        super().__init__(parent)
        self.fv_root = fv_root
        self.setWindowTitle("Chat Settings")
        self.resize(760, 680)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)

        root_lay = QtWidgets.QVBoxLayout(self)
        root_lay.setContentsMargins(10, 10, 10, 10)
        root_lay.setSpacing(10)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_content = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.scroll_content)
        root_lay.addWidget(self.scroll_area, 1)

        lay = QtWidgets.QVBoxLayout(self.scroll_content)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)
        form.setColumnStretch(0, 0)
        form.setColumnStretch(1, 0)
        form.setColumnStretch(2, 1)

        self.ed_runner = QtWidgets.QLineEdit()
        self.btn_browse_runner = QtWidgets.QPushButton("Browse…")
        self.cmb_model = QtWidgets.QComboBox()
        self.btn_refresh_models = QtWidgets.QPushButton("Refresh")
        self.ed_model_root = QtWidgets.QLineEdit()
        self.ed_model_root.setPlaceholderText(os.path.join("models", "llama"))
        self.btn_browse_model_root = QtWidgets.QPushButton("Browse…")
        self.cmb_template = QtWidgets.QComboBox()

        for _btn in (self.btn_browse_runner, self.btn_refresh_models, self.btn_browse_model_root):
            _btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
            _btn.setMinimumWidth(110)

        for _wide in (self.ed_runner, self.cmb_model, self.ed_model_root, self.cmb_template):
            _wide.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            _wide.setMinimumWidth(260)

        self.sp_ctx_size = QtWidgets.QSpinBox()
        self.sp_ctx_size.setRange(256, 1048576)
        self.sp_ctx_size.setSingleStep(256)
        self.sp_ctx_size.setValue(8192)

        self.sp_max_tokens = QtWidgets.QSpinBox()
        self.sp_max_tokens.setRange(1, 500000)
        self.sp_max_tokens.setValue(1200)

        self.sp_temp = QtWidgets.QDoubleSpinBox()
        self.sp_temp.setRange(0.0, 2.0)
        self.sp_temp.setSingleStep(0.05)
        self.sp_temp.setValue(0.7)

        self.sp_top_p = QtWidgets.QDoubleSpinBox()
        self.sp_top_p.setRange(0.0, 1.0)
        self.sp_top_p.setSingleStep(0.05)
        self.sp_top_p.setValue(0.9)

        self.ed_system = QtWidgets.QPlainTextEdit()
        self.ed_system.setPlaceholderText("Optional system prompt…")
        self.ed_system.setMaximumHeight(120)

        self.lbl_templates_dir = QtWidgets.QLabel(_templates_root(fv_root))
        self.lbl_templates_dir.setWordWrap(True)

        self.chk_bubble_auto = QtWidgets.QCheckBox("Auto pair")
        self.chk_bubble_auto.setChecked(False)
        self.btn_bubble_auto_color = BubbleColorButton(DEFAULT_BUBBLE_BASE_COLOR, "Auto bubble color")
        self.btn_bubble_assistant_color = BubbleColorButton(DEFAULT_BUBBLE_ASSISTANT_COLOR, "Chat bubble color")
        self.btn_bubble_user_color = BubbleColorButton(DEFAULT_BUBBLE_USER_COLOR, "User bubble color")
        self.chk_results_chat_only = QtWidgets.QCheckBox("Show generated results only in chat")
        self.chk_results_chat_only.setChecked(True)
        self.chk_results_chat_only.setToolTip("For jobs started from this chat, prevent the Queue tab's Play last result toggle from opening the result in the main FrameVision player. The result still appears in this chat.")

        default_runner = _default_llama_server_path(self.fv_root).replace("/", "\\")
        runner_tip = (
            f"Default llama.cpp server location:\n{default_runner}\n\n"
            "If this file exists and Runner is empty, FrameVision selects it automatically."
        )
        model_tip = "GGUF model file. Coding models are best for patches/code; general/story models are better for creative Planner writing."
        model_folder_tip = "Folder scanned for .gguf models. Default: models\\llama"
        template_tip = "Auto is usually best. Pick a template only when a model replies strangely or uses the wrong chat format."
        ctx_tip = (
            "How much text the model can see. Good defaults:\n"
            "Coding: 32768-65536\n"
            "Planner / storymode: 65536"
        )
        max_tokens_tip = (
            "Maximum answer length. This is reserved inside the context window. Good defaults:\n"
            "Coding: 4096-8192\n"
            "Planner / storymode long answers: 12000-16000\n"
            "Do not set this equal to Context size."
        )
        temp_tip = (
            "Creativity/randomness. Good defaults:\n"
            "Coding / patches: 0.20-0.40\n"
            "Strict JSON / shotlists: 0.30-0.50\n"
            "Storymode ideas: 0.60-0.80"
        )
        top_p_tip = "Sampling range. Good default: 0.90. Use 1.00 for more open creative writing; lower values are stricter."
        system_tip = "Optional instruction that stays active for the chat. Keep it short for coding; use Planner rules here for story/shotlist work."

        self.ed_runner.setToolTip(runner_tip)
        self.btn_browse_runner.setToolTip(runner_tip)
        self.cmb_model.setToolTip(model_tip)
        self.btn_refresh_models.setToolTip("Refresh the model list after adding or moving GGUF files.")
        self.ed_model_root.setToolTip(model_folder_tip)
        self.btn_browse_model_root.setToolTip(model_folder_tip)
        self.cmb_template.setToolTip(template_tip)
        self.sp_ctx_size.setToolTip(ctx_tip)
        self.sp_max_tokens.setToolTip(max_tokens_tip)
        self.sp_temp.setToolTip(temp_tip)
        self.sp_top_p.setToolTip(top_p_tip)
        self.ed_system.setToolTip(system_tip)

        r = 0
        lbl_runner = QtWidgets.QLabel("Runner")
        lbl_runner.setToolTip(runner_tip)
        form.addWidget(lbl_runner, r, 0)
        form.addWidget(self.btn_browse_runner, r, 1)
        form.addWidget(self.ed_runner, r, 2)
        r += 1

        lbl_model = QtWidgets.QLabel("Model")
        lbl_model.setToolTip(model_tip)
        form.addWidget(lbl_model, r, 0)
        form.addWidget(self.btn_refresh_models, r, 1)
        form.addWidget(self.cmb_model, r, 2)
        r += 1

        lbl_model_folder = QtWidgets.QLabel("Model folder")
        lbl_model_folder.setToolTip(model_folder_tip)
        form.addWidget(lbl_model_folder, r, 0)
        form.addWidget(self.btn_browse_model_root, r, 1)
        form.addWidget(self.ed_model_root, r, 2)
        r += 1

        lbl_template = QtWidgets.QLabel("Template")
        lbl_template.setToolTip(template_tip)
        form.addWidget(lbl_template, r, 0)
        form.addWidget(self.cmb_template, r, 1, 1, 2)
        r += 1

        lbl_ctx_size = QtWidgets.QLabel("Context size")
        lbl_ctx_size.setToolTip(ctx_tip)
        form.addWidget(lbl_ctx_size, r, 0)
        form.addWidget(self.sp_ctx_size, r, 1, 1, 2)
        r += 1

        lbl_max_tokens = QtWidgets.QLabel("Max tokens")
        lbl_max_tokens.setToolTip(max_tokens_tip)
        form.addWidget(lbl_max_tokens, r, 0)
        form.addWidget(self.sp_max_tokens, r, 1, 1, 2)
        r += 1

        lbl_temp = QtWidgets.QLabel("Temperature")
        lbl_temp.setToolTip(temp_tip)
        form.addWidget(lbl_temp, r, 0)
        form.addWidget(self.sp_temp, r, 1, 1, 2)
        r += 1

        lbl_top_p = QtWidgets.QLabel("Top-p")
        lbl_top_p.setToolTip(top_p_tip)
        form.addWidget(lbl_top_p, r, 0)
        form.addWidget(self.sp_top_p, r, 1, 1, 2)

        lay.addLayout(form)
        lbl_system = QtWidgets.QLabel("System prompt")
        lbl_system.setToolTip(system_tip)
        lay.addWidget(lbl_system)
        lay.addWidget(self.ed_system)
        lay.addWidget(QtWidgets.QLabel("Template folder"))
        lay.addWidget(self.lbl_templates_dir)

        bubble_box = QtWidgets.QGroupBox("Chat bubble colors")
        bubble_lay = QtWidgets.QGridLayout(bubble_box)
        bubble_lay.setHorizontalSpacing(10)
        bubble_lay.setVerticalSpacing(8)
        bubble_lay.addWidget(self.chk_bubble_auto, 0, 0, 1, 2)
        bubble_lay.addWidget(QtWidgets.QLabel("Auto color"), 1, 0)
        bubble_lay.addWidget(self.btn_bubble_auto_color, 1, 1)
        bubble_lay.addWidget(QtWidgets.QLabel("Chat bubble"), 2, 0)
        bubble_lay.addWidget(self.btn_bubble_assistant_color, 2, 1)
        bubble_lay.addWidget(QtWidgets.QLabel("User bubble"), 3, 0)
        bubble_lay.addWidget(self.btn_bubble_user_color, 3, 1)
        bubble_hint = QtWidgets.QLabel("Auto uses one color and creates a darker chat bubble plus a lighter user bubble.")
        bubble_hint.setWordWrap(True)
        bubble_hint.setObjectName("SubtleLabel")
        bubble_lay.addWidget(bubble_hint, 4, 0, 1, 2)
        lay.addWidget(bubble_box)

        result_box = QtWidgets.QGroupBox("Generated results")
        result_lay = QtWidgets.QVBoxLayout(result_box)
        result_lay.setContentsMargins(10, 8, 10, 8)
        result_lay.addWidget(self.chk_results_chat_only)
        lay.addWidget(result_box)

        self.chk_memory_enabled = QtWidgets.QCheckBox("Use Knowledge & Memory")
        self.chk_memory_enabled.setChecked(True)
        self.chk_memory_enabled.setToolTip("Allow the chat to search /presets/info and /assets/memories when your message asks for saved knowledge, memories, notes, projects, or user files.")
        self.chk_memory_sources = QtWidgets.QCheckBox("Show memory sources in replies")
        self.chk_memory_sources.setChecked(True)
        self.chk_memory_sources.setToolTip("Add a short source list when memory or knowledge files were injected into the prompt.")
        memory_box = QtWidgets.QGroupBox("Knowledge & Memory")
        memory_lay = QtWidgets.QGridLayout(memory_box)
        memory_lay.setContentsMargins(10, 8, 10, 8)
        memory_lay.setHorizontalSpacing(8)
        memory_lay.setVerticalSpacing(8)
        memory_lay.addWidget(self.chk_memory_enabled, 0, 0, 1, 2)
        memory_lay.addWidget(self.chk_memory_sources, 1, 0, 1, 2)
        self.btn_open_info_folder = QtWidgets.QPushButton("Open Knowledge")
        self.btn_open_memory_folder = QtWidgets.QPushButton("Open Memories")
        self.btn_open_user_files_folder = QtWidgets.QPushButton("Open User Files")
        self.btn_open_project_folder = QtWidgets.QPushButton("Open Project")
        self.btn_open_llm_memory_folder = QtWidgets.QPushButton("Open LLM Memory")
        memory_lay.addWidget(self.btn_open_info_folder, 2, 0)
        memory_lay.addWidget(self.btn_open_memory_folder, 2, 1)
        memory_lay.addWidget(self.btn_open_user_files_folder, 3, 0)
        memory_lay.addWidget(self.btn_open_project_folder, 3, 1)
        memory_lay.addWidget(self.btn_open_llm_memory_folder, 4, 0, 1, 2)
        memory_hint = QtWidgets.QLabel("Reads /presets/info and /assets/memories. Saves only to /assets/memories/saved_notes or /assets/memories/project. LLM Memory is loaded into chat context at startup.")
        memory_hint.setWordWrap(True)
        memory_hint.setObjectName("SubtleLabel")
        memory_lay.addWidget(memory_hint, 5, 0, 1, 2)
        lay.addWidget(memory_box)

        hint = QtWidgets.QLabel(
            "Good defaults: Coding = 65k ctx / 4k-8k answer. Planner storymode = 65k ctx / 12k-16k answer."
        )
        hint.setWordWrap(True)
        hint.setObjectName("SubtleLabel")
        lay.addWidget(hint)

        lay.addStretch(1)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        self.btn_chat_info = QtWidgets.QPushButton("Info")
        self.btn_chat_info.setToolTip("Show FrameVision chat tool options and queue/VRAM behavior.")
        btns.addButton(self.btn_chat_info, QtWidgets.QDialogButtonBox.HelpRole)
        root_lay.addWidget(btns)
        btns.rejected.connect(self.reject)
        self.btn_chat_info.clicked.connect(self._show_framevision_chat_info)

        self.btn_browse_runner.clicked.connect(self._browse_runner)
        self.btn_browse_model_root.clicked.connect(self._browse_model_root)
        self.btn_refresh_models.clicked.connect(self.refresh_models)
        self.ed_model_root.editingFinished.connect(self._on_model_root_edited)
        self.chk_bubble_auto.toggled.connect(self._update_bubble_color_mode)
        self.chk_bubble_auto.toggled.connect(lambda *_: self.settingsChanged.emit())
        self.chk_results_chat_only.toggled.connect(lambda *_: self.settingsChanged.emit())
        self.chk_memory_enabled.toggled.connect(lambda *_: self.settingsChanged.emit())
        self.chk_memory_sources.toggled.connect(lambda *_: self.settingsChanged.emit())
        self.btn_open_info_folder.clicked.connect(lambda: _open_local_path(_knowledge_root(self.fv_root)))
        self.btn_open_memory_folder.clicked.connect(lambda: _open_local_path(_memories_root(self.fv_root)))
        self.btn_open_user_files_folder.clicked.connect(lambda: _open_local_path(_memory_user_files_dir(self.fv_root)))
        self.btn_open_project_folder.clicked.connect(lambda: _open_local_path(_memory_project_dir(self.fv_root)))
        self.btn_open_llm_memory_folder.clicked.connect(lambda: _open_local_path(_memory_llm_memory_dir(self.fv_root)))
        self.btn_bubble_auto_color.colorChanged.connect(lambda *_: self.settingsChanged.emit())
        self.btn_bubble_assistant_color.colorChanged.connect(lambda *_: self.settingsChanged.emit())
        self.btn_bubble_user_color.colorChanged.connect(lambda *_: self.settingsChanged.emit())

        for w in [
            self.ed_runner,
            self.ed_model_root,
            self.cmb_model,
            self.cmb_template,
            self.sp_ctx_size,
            self.sp_max_tokens,
            self.sp_temp,
            self.sp_top_p,
        ]:
            if isinstance(w, QtWidgets.QLineEdit):
                w.textChanged.connect(self.settingsChanged)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(self.settingsChanged)
            elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                w.valueChanged.connect(self.settingsChanged)
        self.ed_system.textChanged.connect(self.settingsChanged)
        self._update_bubble_color_mode()

    def _update_bubble_color_mode(self):
        auto_mode = self.chk_bubble_auto.isChecked()
        self.btn_bubble_auto_color.setEnabled(auto_mode)
        self.btn_bubble_assistant_color.setEnabled(not auto_mode)
        self.btn_bubble_user_color.setEnabled(not auto_mode)

    def bubble_color_settings(self) -> Tuple[bool, str, str, str]:
        return (
            bool(self.chk_bubble_auto.isChecked()),
            self.btn_bubble_auto_color.color(),
            self.btn_bubble_assistant_color.color(),
            self.btn_bubble_user_color.color(),
        )

    def set_bubble_color_settings(self, auto_mode: bool, auto_color: str, assistant_color: str, user_color: str):
        widgets = [
            self.chk_bubble_auto,
            self.btn_bubble_auto_color,
            self.btn_bubble_assistant_color,
            self.btn_bubble_user_color,
        ]
        blocked = [w.blockSignals(True) for w in widgets]
        try:
            self.chk_bubble_auto.setChecked(bool(auto_mode))
            self.btn_bubble_auto_color.setColor(_normalize_hex_color(auto_color, DEFAULT_BUBBLE_BASE_COLOR), emit=False)
            self.btn_bubble_assistant_color.setColor(_normalize_hex_color(assistant_color, DEFAULT_BUBBLE_ASSISTANT_COLOR), emit=False)
            self.btn_bubble_user_color.setColor(_normalize_hex_color(user_color, DEFAULT_BUBBLE_USER_COLOR), emit=False)
        finally:
            for w, old_block in zip(widgets, blocked):
                w.blockSignals(old_block)
        self._update_bubble_color_mode()


    def _show_framevision_chat_info(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("FrameVision Chat Info")
        dlg.resize(760, 620)
        dlg.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint, True)

        root = QtWidgets.QVBoxLayout(dlg)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QtWidgets.QLabel("Exclusive FrameVision chat options")
        title.setObjectName("SectionTitle")
        try:
            f = title.font()
            f.setPointSize(max(f.pointSize() + 2, 13))
            f.setBold(True)
            title.setFont(f)
        except Exception:
            pass
        root.addWidget(title)

        info = QtWidgets.QTextBrowser()
        info.setOpenExternalLinks(False)
        info.setReadOnly(True)
        info.setHtml(
            """
            <style>
                body { font-size: 12pt; line-height: 1.35; }
                h3 { margin: 10px 0 4px 0; }
                p { margin: 4px 0 10px 0; }
                ul { margin-top: 4px; }
            </style>

            <h3>Chat Memory</h3>
            <p>User can save to memory or save to projects (both with right mouse click or directly typing in the chat (save to memory, save to project </p>
            <p>User can put files in assets/memories/user_files for specific memory tasks.</p>
            <p>Give your llm extra powers, a name, a template (you are an expert in..)... by adding something in assets/memories/llm_memory -> it will be loaded everytime the model is loaded</p>
            <p>Chat also has access to the framevision knowledge base to be able to answer questions about the app.</p>
            
            <h3>Chat Wizards</h3>
            <ul>
            <p>cancel</b>: Stops the current wizard and returns to normal chat.</p>
            <p>undo</b>: Goes back one step in the current wizard</p>

            <h3>Create image</h3>
            <p>The chat checks for available image models and lets you select a model to create an image.
            Supported aspect presets include 1:1, 9:16 and 16:9, up to 2048 x 2048.</p>

            <h3>Edit image</h3>
            <p>Currently supports Flux Klein for smaller edits and HiDream for bigger edits or reference-image based edits.</p>

            <h3>Upscale image/ upscale video</h3>
            <p>Uses SeedVR2. Available output for images : 1920, 1440 and 2160p. For video it is 720, 1920 and 1440p (will be very slow)</p>

            <h3>Create video</h3>
            <p>This triggers the wizard for text-to-video, image-to-video, or video-to-video.
            You can select resolution, frames, FPS, prompt and other needed details, including a finetuned safetensors file.
            For image/video-to-video requests, matching aspect ratio and/or FPS (frames per second) is recommended to avoid bad or failed results.</p>

            <h3>Create music</h3>
            <p>Uses Ace Step Music 1.5. The chat checks the genre preset list for available genres and subgenres when present.
            It can also ask for a custom genre description when no preset is found, then collect duration, title, lyrics or instrumental choice, and other needed details.</p>

            </ul>

            <h3>Wizard behavior a other info</h3>
            <ul>
                <li>Models need to be installed before the chat can use them.(find them in the 'optional downloads' menu</li>
                <li>Usage is easy : use the trigger words above to start a wizard for a job</li>
                <li>keep answers short and to the point to avoid llm getting confused</li>
                <li>The Framevision extra chat options work with any loaded GGUF LLM model.</li>
                <li>When the queue is busy creating a heavy FrameVision job, the LLM is unloaded from VRAM into cache.</li>
                <li>A settings toggle lets generated results open in the original FrameVision player or stay only in the chat. Images can show in both.</li>
                <li>Drag the splitter completely to the left to hide the internal mediaplayer and get a full sized llm chat program.</li>
                <li>You can keep typing while a job is running, but sending is blocked until the job is finished to avoid overloading VRAM.</li>
                <li>Right click on a response from the chat to retry</li>
                <li>Edit sent message and save them to have the chat try again with the updated request</li>
            </ul>
            """
        )
        root.addWidget(info, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        root.addWidget(buttons)
        dlg.exec()

    def _browse_runner(self):
        start = self.ed_runner.text().strip()
        start_dir = os.path.dirname(start) if start and os.path.isdir(os.path.dirname(start)) else self.fv_root
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select runner",
            start_dir,
            "Executables (*.exe);;All files (*.*)",
        )
        if path:
            self.ed_runner.setText(path)


    def _browse_model_root(self):
        start = self.ed_model_root.text().strip()
        if start and not os.path.isabs(start):
            start = os.path.join(self.fv_root, start)
        start_dir = start if start and os.path.isdir(start) else os.path.join(self.fv_root, "models", "llama")
        if not os.path.isdir(start_dir):
            start_dir = os.path.join(self.fv_root, "models")
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select model folder", start_dir)
        if not path:
            return
        try:
            rel = os.path.relpath(path, self.fv_root)
            if not rel.startswith(".."):
                path = rel
        except Exception:
            pass
        self.ed_model_root.setText(path)
        self.refresh_models()
        self.settingsChanged.emit()

    def _on_model_root_edited(self):
        self.refresh_models()
        self.settingsChanged.emit()

    def refresh_models(self):
        prev = self.current_model_path()
        locked_root = self.ed_model_root.text().strip()
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        for label, path in discover_models(self.fv_root, locked_root=locked_root):
            self.cmb_model.addItem(label, userData=path)
        self.cmb_model.blockSignals(False)
        if prev:
            self.select_model_by_path(prev)

    def refresh_templates(self):
        prev_kind, prev_val = self.current_template_choice()
        self.cmb_template.blockSignals(True)
        self.cmb_template.clear()
        for label, kind, value in discover_templates(self.fv_root):
            self.cmb_template.addItem(label, userData=(kind, value))
        self.cmb_template.blockSignals(False)
        self.select_template(prev_kind, prev_val)

    def current_model_path(self) -> str:
        data = self.cmb_model.currentData()
        return str(data) if data else ""

    def current_template_choice(self) -> Tuple[str, str]:
        data = self.cmb_template.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return str(data[0]), str(data[1])
        return ("auto", "")

    def select_model_by_path(self, path: str, emit: bool = False) -> bool:
        if not path:
            return False
        old_block = self.cmb_model.blockSignals(True)
        try:
            for i in range(self.cmb_model.count()):
                if (self.cmb_model.itemData(i) or "") == path:
                    changed = (self.cmb_model.currentIndex() != i)
                    self.cmb_model.setCurrentIndex(i)
                    if emit and changed:
                        self.settingsChanged.emit()
                    return changed
        finally:
            self.cmb_model.blockSignals(old_block)
        return False

    def select_template(self, kind: str, value: str, emit: bool = False) -> bool:
        target = -1
        for i in range(self.cmb_template.count()):
            data = self.cmb_template.itemData(i)
            if isinstance(data, tuple) and len(data) == 2 and data[0] == kind and data[1] == value:
                target = i
                break
        if target < 0:
            for i in range(self.cmb_template.count()):
                data = self.cmb_template.itemData(i)
                if isinstance(data, tuple) and len(data) == 2 and data[0] == "auto":
                    target = i
                    break
        if target < 0:
            return False
        old_block = self.cmb_template.blockSignals(True)
        try:
            changed = (self.cmb_template.currentIndex() != target)
            self.cmb_template.setCurrentIndex(target)
        finally:
            self.cmb_template.blockSignals(old_block)
        if emit and changed:
            self.settingsChanged.emit()
        return changed


# -----------------------------
# Main window
# -----------------------------
class LlamaChatWindow(QtWidgets.QMainWindow):
    framevisionFullscreenRequested = QtCore.Signal(bool)

    def __init__(self):
        super().__init__()
        self.fv_root = _find_fv_root()
        self.chat_dir = _chat_dir(self.fv_root)
        self.attachment_temp_dir = _attachment_temp_dir(self.fv_root)
        self.settings_path = _settings_path(self.fv_root)
        self.assistant_jobs_path = _assistant_jobs_path(self.fv_root)

        os.makedirs(self.chat_dir, exist_ok=True)
        os.makedirs(self.attachment_temp_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.assistant_jobs_path), exist_ok=True)
        _ensure_memory_folders(self.fv_root)

        self.sessions: List[ChatSession] = []
        self.current_session_id: Optional[str] = None
        self._saved_model_path = ""
        self._saved_template = ("auto", "")
        self._bubble_color_auto = False
        self._bubble_color_base = DEFAULT_BUBBLE_BASE_COLOR
        self._bubble_color_assistant = DEFAULT_BUBBLE_ASSISTANT_COLOR
        self._bubble_color_user = DEFAULT_BUBBLE_USER_COLOR

        self.server_process: Optional[QtCore.QProcess] = None
        self.server_boot_thread: Optional[ServerBootThread] = None
        self.chat_thread: Optional[ChatCompletionThread] = None
        self.server_port: Optional[int] = None
        self.server_url: str = ""
        self.server_ready: bool = False
        self.loaded_model_path: str = ""
        self.loaded_template: Tuple[str, str] = ("auto", "")
        self._llm_gpu_lock_active: bool = False
        self.loaded_ctx_size: int = 0
        self.server_log_tail: List[str] = []
        self.pending_generate_session_id: str = ""
        self.pending_retry_generation: bool = False
        self._pending_generation_update: Optional[Dict[str, Any]] = None
        self._last_user_text_sent: str = ""
        self._auto_retry_templates: List[Tuple[str, str]] = []
        self._auto_retry_original_selection: Tuple[str, str] = ("auto", "")
        self.effective_template_on_server: Tuple[str, str] = ("auto", "")
        self.pending_attachments: List[Dict[str, Any]] = []
        self._pending_seedvr2_request: Optional[Dict[str, Any]] = None
        self._pending_ace15_music_request: Optional[Dict[str, Any]] = None
        self._framevision_fullscreen_active = False
        try:
            from helpers.fv_assistant_router import FrameVisionAssistantRouter
        except Exception:
            try:
                from fv_assistant_router import FrameVisionAssistantRouter  # type: ignore
            except Exception:
                FrameVisionAssistantRouter = None  # type: ignore
        self._fv_assistant_router = FrameVisionAssistantRouter(self.fv_root) if FrameVisionAssistantRouter else None

        self._syncing_model_selectors = False
        self._loading_session_state = False
        self._last_ui_model_path = ""
        self._startup_llm_memory_context: str = ""
        self._last_memory_sources: List[str] = []
        self._memory_context_mode_for_next_generation: str = "auto"

        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_all)

        self._assistant_jobs: List[Dict[str, Any]] = []
        self._assistant_job_timer = QtCore.QTimer(self)
        self._assistant_job_timer.setInterval(1200)
        self._assistant_job_timer.timeout.connect(self._poll_assistant_jobs)

        self.setWindowTitle("FrameVision - Llama Chat UI")
        self.resize(1480, 900)

        self._build_ui()
        self._apply_style()
        self._load_settings()
        self._reload_startup_llm_memory()
        self._apply_style()
        self._load_sessions()
        self._load_assistant_jobs()
        self._apply_send_button_guard()
        self._assistant_job_timer.start()
        self.settings_dialog.refresh_models()
        self.settings_dialog.refresh_templates()

        if self._saved_model_path:
            self.settings_dialog.select_model_by_path(self._saved_model_path)
        self.settings_dialog.select_template(*self._saved_template)
        self._last_ui_model_path = self.current_model_path()
        self._rebuild_quick_model_combo()

        if not self.sessions:
            self._new_chat()
        else:
            self._select_session(self.current_session_id or self.sessions[0].id)

    # ---------- UI ----------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        root.addWidget(splitter)

        sidebar = QtWidgets.QFrame()
        sidebar.setObjectName("Sidebar")
        s_lay = QtWidgets.QVBoxLayout(sidebar)
        s_lay.setContentsMargins(12, 12, 12, 12)
        s_lay.setSpacing(10)

        row = QtWidgets.QHBoxLayout()
        self.lbl_chats = QtWidgets.QLabel("Chats")
        self.lbl_chats.setObjectName("SidebarTitle")
        row.addWidget(self.lbl_chats)
        row.addStretch(1)
        s_lay.addLayout(row)

        self.btn_new_chat = QtWidgets.QPushButton("New Chat")
        s_lay.addWidget(self.btn_new_chat)

        self.ed_search = QtWidgets.QLineEdit()
        self.ed_search.setPlaceholderText("Search chats…")
        s_lay.addWidget(self.ed_search)

        self.lst_chats = QtWidgets.QListWidget()
        self.lst_chats.setObjectName("ChatList")
        s_lay.addWidget(self.lst_chats, 1)

        self.btn_delete_chat = QtWidgets.QPushButton("Delete Chat")
        s_lay.addWidget(self.btn_delete_chat)

        self.btn_framevision_fullscreen = QtWidgets.QPushButton("Full screen")
        self.btn_framevision_fullscreen.setCheckable(True)
        self.btn_framevision_fullscreen.setObjectName("LlamaFullscreenButton")
        self.btn_framevision_fullscreen.setToolTip("Expands the LLM Chat tab by moving FrameVision's main splitter to the left. You can also resize the splitter manually by dragging it.")
        # Hidden in FrameVision: fullscreen is controlled by the main app splitter, not by this sidebar button.
        self.btn_framevision_fullscreen.setVisible(False)

        self.btn_settings = QtWidgets.QPushButton("Settings")
        s_lay.addWidget(self.btn_settings)

        main = QtWidgets.QWidget()
        m_lay = QtWidgets.QVBoxLayout(main)
        m_lay.setContentsMargins(0, 0, 0, 0)
        m_lay.setSpacing(0)

        topbar = QtWidgets.QFrame()
        topbar.setObjectName("TopBar")
        tb_lay = QtWidgets.QHBoxLayout(topbar)
        tb_lay.setContentsMargins(18, 12, 18, 12)
        tb_lay.setSpacing(10)

        self.lbl_title = QtWidgets.QLabel("New Chat")
        self.lbl_title.setObjectName("ChatHeader")
        self.cmb_model_quick = QtWidgets.QComboBox()
        self.cmb_model_quick.setMinimumWidth(420)
        self.lbl_status = QtWidgets.QLabel("Idle")
        self.lbl_status.setObjectName("StatusIdle")
        self.btn_load = QtWidgets.QPushButton("Load")
        self.btn_unload = QtWidgets.QPushButton("Unload")
        self.btn_unload.setEnabled(False)
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setEnabled(False)

        tb_lay.addWidget(self.lbl_title)
        tb_lay.addStretch(1)
        tb_lay.addWidget(self.cmb_model_quick, 0)
        tb_lay.addWidget(self.lbl_status, 0)
        tb_lay.addWidget(self.btn_load)
        tb_lay.addWidget(self.btn_unload)
        tb_lay.addWidget(self.btn_stop)

        self.chat_view = ChatScrollArea()
        self.chat_view.editMessageSaved.connect(self._on_chat_message_edit_saved)
        self.chat_view.retryMessageRequested.connect(self._on_chat_message_retry_requested)
        self.chat_view.saveToMemoryRequested.connect(self._on_save_message_to_memory_requested)
        self.chat_view.saveToProjectRequested.connect(self._on_save_message_to_project_requested)
        self.chat_view.versionPrevRequested.connect(lambda mid: self._switch_message_version(mid, -1))
        self.chat_view.versionNextRequested.connect(lambda mid: self._switch_message_version(mid, 1))

        composer_wrap = QtWidgets.QWidget()
        cw_lay = QtWidgets.QVBoxLayout(composer_wrap)
        cw_lay.setContentsMargins(24, 0, 24, 22)
        cw_lay.setSpacing(8)

        self.lbl_model_info = QtWidgets.QLabel("")
        self.lbl_model_info.setObjectName("SubtleLabel")
        cw_lay.addWidget(self.lbl_model_info)

        composer = QtWidgets.QFrame()
        composer.setObjectName("ComposerFrame")
        comp_lay = QtWidgets.QVBoxLayout(composer)
        comp_lay.setContentsMargins(14, 12, 14, 12)
        comp_lay.setSpacing(10)

        self.ed_prompt = PromptEdit()
        self.ed_prompt.setPlaceholderText("Send a message to the model…")
        # Let the composer be resized with the vertical splitter.
        # Minimum is roughly one text line; maximum is roughly twenty lines.
        _line_h = max(18, self.ed_prompt.fontMetrics().lineSpacing())
        self.ed_prompt.setMinimumHeight(_line_h + 18)
        self.ed_prompt.setMaximumHeight((_line_h * 20) + 28)
        self.ed_prompt.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        self.lst_attachments = QtWidgets.QListWidget()
        self.lst_attachments.setObjectName("AttachmentList")
        self.lst_attachments.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.lst_attachments.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.lst_attachments.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.lst_attachments.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.lst_attachments.setIconSize(QtCore.QSize(56, 56))
        self.lst_attachments.setResizeMode(QtWidgets.QListView.Adjust)
        self.lst_attachments.setMovement(QtWidgets.QListView.Static)
        self.lst_attachments.setUniformItemSizes(False)
        self.lst_attachments.setSpacing(4)
        self.lst_attachments.setWordWrap(False)
        self.lst_attachments.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.lst_attachments.setMaximumHeight(156)
        self.lst_attachments.setVisible(False)

        send_row = QtWidgets.QHBoxLayout()
        self.btn_attach = QtWidgets.QPushButton("+")
        self.btn_attach.setFixedWidth(42)
        self.btn_send = QtWidgets.QPushButton("Send")
        self.btn_send.setDefault(True)
        send_row.addWidget(self.btn_attach, 0)
        send_row.addStretch(1)
        send_row.addWidget(self.btn_send, 0)

        comp_lay.addWidget(self.lst_attachments)
        comp_lay.addWidget(self.ed_prompt)
        comp_lay.addLayout(send_row)
        cw_lay.addWidget(composer)

        self.main_vertical_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.main_vertical_splitter.setObjectName("ChatComposerSplitter")
        self.main_vertical_splitter.setChildrenCollapsible(False)
        self.main_vertical_splitter.setHandleWidth(8)
        self.main_vertical_splitter.addWidget(self.chat_view)
        self.main_vertical_splitter.addWidget(composer_wrap)
        self.main_vertical_splitter.setStretchFactor(0, 1)
        self.main_vertical_splitter.setStretchFactor(1, 0)
        self.main_vertical_splitter.setSizes([760, 150])
        self.main_vertical_splitter.splitterMoved.connect(self._queue_save)

        m_lay.addWidget(topbar, 0)
        m_lay.addWidget(self.main_vertical_splitter, 1)

        sidebar.setMinimumWidth(180)
        main.setMinimumWidth(760)
        splitter.addWidget(sidebar)
        splitter.addWidget(main)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1200])

        self.settings_dialog = SettingsDialog(self.fv_root, self)
        self.settings_dialog.settingsChanged.connect(self._settings_changed)

        self.btn_new_chat.clicked.connect(self._new_chat)
        self.btn_delete_chat.clicked.connect(self._delete_current_chat)
        self.lst_chats.currentRowChanged.connect(self._on_chat_row_changed)
        self.ed_search.textChanged.connect(self._apply_chat_filter)
        self.btn_send.clicked.connect(self._send_message)
        self.ed_prompt.sendRequested.connect(self._send_message)
        self.ed_prompt.imagePasted.connect(self._handle_pasted_image)
        self.ed_prompt.fileUrlsPasted.connect(self._handle_pasted_file_urls)
        self.btn_stop.clicked.connect(self._stop_generation)
        self.btn_settings.clicked.connect(self._show_settings)
        self.btn_framevision_fullscreen.clicked.connect(self._toggle_framevision_fullscreen)
        self.btn_load.clicked.connect(self._load_selected_model)
        self.btn_unload.clicked.connect(self._unload_model)
        self.cmb_model_quick.currentIndexChanged.connect(self._quick_model_changed)
        self.settings_dialog.btn_refresh_models.clicked.connect(self._sync_models_from_settings)
        self.btn_attach.clicked.connect(self._pick_attachments)
        self.lst_attachments.itemClicked.connect(self._preview_attachment_item)
        self.lst_attachments.customContextMenuRequested.connect(self._attachment_context_menu)
        self._refresh_attachment_list()

    def _toggle_framevision_fullscreen(self, checked: bool):
        self._framevision_fullscreen_active = bool(checked)
        self._update_framevision_fullscreen_button()
        try:
            self.framevisionFullscreenRequested.emit(bool(checked))
        except Exception:
            pass
        try:
            win = self.window()
            if win is not None and win is not self and hasattr(win, "set_llm_chat_splitter_fullscreen"):
                win.set_llm_chat_splitter_fullscreen(bool(checked))
        except Exception:
            pass

    def set_framevision_fullscreen_active(self, active: bool):
        """Called by FrameVision when the user exits pseudo-fullscreen by dragging the splitter."""
        self._framevision_fullscreen_active = bool(active)
        self._update_framevision_fullscreen_button()

    def _update_framevision_fullscreen_button(self):
        btn = getattr(self, "btn_framevision_fullscreen", None)
        if btn is None:
            return
        active = bool(getattr(self, "_framevision_fullscreen_active", False))
        try:
            btn.blockSignals(True)
            btn.setChecked(active)
        finally:
            try:
                btn.blockSignals(False)
            except Exception:
                pass
        if active:
            try:
                pal = QtWidgets.QApplication.palette()
                bg = pal.color(QtGui.QPalette.Highlight).name()
                fg = pal.color(QtGui.QPalette.HighlightedText).name()
            except Exception:
                bg, fg = "#3d6fd6", "#ffffff"
            btn.setStyleSheet(
                "QPushButton#LlamaFullscreenButton {"
                f"background: {bg}; color: {fg}; border: 1px solid {bg}; "
                "font-weight: 700; border-radius: 10px; padding: 8px 12px; }"
                "QPushButton#LlamaFullscreenButton:hover { filter: brightness(1.1); }"
            )
        else:
            btn.setStyleSheet("")

    def _apply_style(self):
        auto_mode, auto_color, assistant_color, user_color = self.settings_dialog.bubble_color_settings()
        theme = _framevision_theme_colors()
        bubble_theme = _soft_chat_bubble_colors(theme, auto_mode, auto_color, assistant_color, user_color)
        thinking_bg = _mix_hex(theme['panel'], theme['window'], 0.16)
        attachment_bg = _mix_hex(theme['panel'], theme['highlight'], 0.08)
        info_bg = _mix_hex(theme['panel'], '#607040', 0.18)
        disabled_bg = _mix_hex(theme['button'], theme['window'], 0.45)
        disabled_fg = _mix_hex(_text_color_for_bg(disabled_bg), disabled_bg, 0.38)
        status_idle_bg = _mix_hex(theme['button'], theme['window'], 0.18)
        status_loading_bg = _mix_hex(theme['button'], '#9a6a22', 0.30)
        status_ready_bg = _mix_hex(theme['button'], '#17814f', 0.30)
        status_error_bg = _mix_hex(theme['button'], '#9b2f2f', 0.34)

        self.setStyleSheet(f"""
        QMainWindow, QWidget {{
            background: {theme['window']};
            color: {theme['text']};
            font-size: 13px;
        }}
        #Sidebar {{
            background: {theme['window']};
            border-right: 1px solid {theme['border']};
        }}
        #TopBar {{
            background: {theme['window']};
            border-bottom: 1px solid {theme['border']};
        }}
        #ComposerFrame {{
            background: {theme['panel']};
            border: 1px solid {theme['border']};
            border-radius: 18px;
        }}
        #AttachmentList {{
            background: {theme['base']};
            border: 1px solid {theme['border']};
            border-radius: 10px;
            padding: 4px;
        }}
        #AttachmentList::item {{
            padding: 6px 8px;
            border-radius: 8px;
        }}
        #AttachmentList::item:selected {{
            background: {theme['selected']};
        }}
        #ChatList {{
            background: {theme['base']};
            border: 1px solid {theme['border']};
            border-radius: 12px;
            padding: 4px;
        }}
        #SidebarTitle {{
            font-size: 21px;
            font-weight: 600;
        }}
        #ChatHeader {{
            font-size: 17px;
            font-weight: 600;
        }}
        #SubtleLabel, #BubbleRole {{
            color: {theme['subtle']};
        }}
        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {theme['base']};
            color: {theme['base_text']};
            border: 1px solid {theme['border']};
            border-radius: 10px;
            padding: 8px;
        }}
        QListWidget::item {{
            padding: 10px 10px;
            border-radius: 8px;
            margin: 2px 0px;
        }}
        QListWidget::item:selected {{
            background: {theme['selected']};
            color: {theme['highlighted_text']};
        }}
        QListWidget::item:hover {{
            background: {theme['hover']};
        }}
        QPushButton {{
            background: {theme['button']};
            color: {_text_color_for_bg(theme['button'])};
            border: 1px solid {theme['border']};
            border-radius: 10px;
            padding: 9px 14px;
            font-weight: 400;
        }}
        QPushButton:hover {{
            background: {theme['hover']};
        }}
        QPushButton:disabled {{
            color: {disabled_fg};
            background: {disabled_bg};
            border: 1px solid {_mix_hex(theme['border'], theme['window'], 0.35)};
        }}
        QScrollArea {{
            border: none;
            background: {theme['window']};
        }}
        QFrame#Bubble_user {{
            background: {bubble_theme['user_bg']};
            border: 1px solid {bubble_theme['user_border']};
            border-radius: 14px;
        }}
        QFrame#Bubble_user QLabel#BubbleRole {{
            color: {bubble_theme['user_role']};
            font-weight: 400;
        }}
        QFrame#Bubble_user QLabel#UserText {{
            color: {bubble_theme['user_text']};
            font-weight: 400;
        }}
        QFrame#Bubble_assistant {{
            background: {bubble_theme['assistant_bg']};
            border: 1px solid {bubble_theme['assistant_border']};
            border-radius: 14px;
        }}
        QFrame#Bubble_assistant QLabel#BubbleRole {{
            color: {bubble_theme['assistant_role']};
            font-weight: 400;
        }}
        QFrame#Bubble_assistant QLabel#AnswerText {{
            color: {bubble_theme['assistant_text']};
            font-weight: 400;
        }}
        QFrame#ThinkingFrame {{
            background: {thinking_bg};
            border: 1px solid {theme['border']};
            border-radius: 10px;
        }}
        QFrame#AttachmentFrame {{
            background: {attachment_bg};
            border: 1px solid {theme['border']};
            border-radius: 10px;
        }}
        QLabel#AttachmentTitle {{
            color: {theme['text']};
            font-weight: 500;
        }}
        QLabel#AttachmentText {{
            color: {theme['text']};
        }}
        QLabel#ThinkingTitle {{
            color: {theme['subtle']};
            font-weight: 500;
        }}
        QLabel#ThinkingText {{
            color: {theme['subtle']};
        }}
        QLabel#AnswerText {{
            color: {bubble_theme['assistant_text']};
            font-weight: 400;
        }}
        QLabel#UserText {{
            color: {bubble_theme['user_text']};
            font-weight: 400;
        }}
        QFrame#Bubble_info {{
            background: {info_bg};
            border: 1px solid {theme['border']};
            border-radius: 14px;
        }}
        QLabel#InfoText {{
            color: {theme['text']};
        }}
        QLabel#StatusIdle, QLabel#StatusLoading, QLabel#StatusReady, QLabel#StatusError {{
            border-radius: 12px;
            padding: 6px 10px;
            font-weight: 500;
        }}
        QLabel#StatusIdle {{
            background: {status_idle_bg};
            color: {theme['text']};
        }}
        QLabel#StatusLoading {{
            background: {status_loading_bg};
            color: #ffdf9c;
        }}
        QLabel#StatusReady {{
            background: {status_ready_bg};
            color: #9cf0c8;
        }}
        QLabel#StatusError {{
            background: {status_error_bg};
            color: #ffb1b1;
        }}
        """)

    # ---------- persistence ----------
    def _load_settings(self):
        data = _load_json(self.settings_path, {})
        if not isinstance(data, dict):
            data = {}
        runner = data.get("runner_path", "")
        if isinstance(runner, str):
            runner = runner.strip()
            if not runner:
                runner = _find_default_llama_server(self.fv_root)
            self.settings_dialog.ed_runner.setText(runner)
        try:
            self.settings_dialog.sp_ctx_size.setValue(int(data.get("ctx_size", 8192)))
        except Exception:
            pass
        try:
            self.settings_dialog.sp_max_tokens.setValue(int(data.get("max_tokens", 1200)))
        except Exception:
            pass
        try:
            self.settings_dialog.sp_temp.setValue(float(data.get("temp", 0.7)))
        except Exception:
            pass
        try:
            self.settings_dialog.sp_top_p.setValue(float(data.get("top_p", 0.9)))
        except Exception:
            pass
        model_root = data.get("model_root", os.path.join("models", "llama"))
        if isinstance(model_root, str):
            self.settings_dialog.ed_model_root.setText(model_root)
        if isinstance(data.get("system_prompt"), str):
            self.settings_dialog.ed_system.setPlainText(data.get("system_prompt", ""))
        self._bubble_color_auto = bool(data.get("bubble_color_auto", False))
        self._bubble_color_base = _normalize_hex_color(str(data.get("bubble_color_base", DEFAULT_BUBBLE_BASE_COLOR)), DEFAULT_BUBBLE_BASE_COLOR)
        self._bubble_color_assistant = _normalize_hex_color(str(data.get("bubble_color_assistant", DEFAULT_BUBBLE_ASSISTANT_COLOR)), DEFAULT_BUBBLE_ASSISTANT_COLOR)
        self._bubble_color_user = _normalize_hex_color(str(data.get("bubble_color_user", DEFAULT_BUBBLE_USER_COLOR)), DEFAULT_BUBBLE_USER_COLOR)
        self.settings_dialog.set_bubble_color_settings(
            self._bubble_color_auto,
            self._bubble_color_base,
            self._bubble_color_assistant,
            self._bubble_color_user,
        )
        try:
            self.settings_dialog.chk_results_chat_only.setChecked(bool(data.get("assistant_results_chat_only", True)))
        except Exception:
            pass
        try:
            self.settings_dialog.chk_memory_enabled.setChecked(bool(data.get("memory_enabled", True)))
        except Exception:
            pass
        try:
            self.settings_dialog.chk_memory_sources.setChecked(bool(data.get("memory_show_sources", True)))
        except Exception:
            pass
        self.current_session_id = data.get("last_session_id") if isinstance(data.get("last_session_id"), str) else None
        self._saved_model_path = data.get("model_path", "") if isinstance(data.get("model_path"), str) else ""
        self._saved_template = (data.get("template_kind", "auto"), data.get("template_value", ""))
        try:
            sizes = data.get("chat_composer_splitter_sizes")
            if isinstance(sizes, list) and len(sizes) >= 2 and getattr(self, "main_vertical_splitter", None) is not None:
                a = max(120, int(sizes[0]))
                b = max(64, int(sizes[1]))
                self.main_vertical_splitter.setSizes([a, b])
        except Exception:
            pass

    def _save_all(self):
        bubble_auto, bubble_base, bubble_assistant, bubble_user = self.settings_dialog.bubble_color_settings()
        settings = {
            "runner_path": self.settings_dialog.ed_runner.text().strip(),
            "model_root": self.settings_dialog.ed_model_root.text().strip(),
            "model_path": self.current_model_path(),
            "template_kind": self.current_template_choice()[0],
            "template_value": self.current_template_choice()[1],
            "ctx_size": int(self.settings_dialog.sp_ctx_size.value()),
            "max_tokens": int(self.settings_dialog.sp_max_tokens.value()),
            "temp": float(self.settings_dialog.sp_temp.value()),
            "top_p": float(self.settings_dialog.sp_top_p.value()),
            "system_prompt": self.settings_dialog.ed_system.toPlainText(),
            "bubble_color_auto": bool(bubble_auto),
            "bubble_color_base": bubble_base,
            "bubble_color_assistant": bubble_assistant,
            "bubble_color_user": bubble_user,
            "assistant_results_chat_only": bool(getattr(self.settings_dialog, "chk_results_chat_only", None).isChecked()) if getattr(self.settings_dialog, "chk_results_chat_only", None) is not None else True,
            "memory_enabled": bool(getattr(self.settings_dialog, "chk_memory_enabled", None).isChecked()) if getattr(self.settings_dialog, "chk_memory_enabled", None) is not None else True,
            "memory_show_sources": bool(getattr(self.settings_dialog, "chk_memory_sources", None).isChecked()) if getattr(self.settings_dialog, "chk_memory_sources", None) is not None else True,
            "last_session_id": self.current_session_id or "",
            "chat_composer_splitter_sizes": list(getattr(self, "main_vertical_splitter", None).sizes()) if getattr(self, "main_vertical_splitter", None) is not None else [760, 150],
        }
        _save_json_atomic(self.settings_path, settings)
        for s in self.sessions:
            _save_json_atomic(os.path.join(self.chat_dir, f"{s.id}.json"), asdict(s))

    def _queue_save(self):
        self._save_timer.start(250)

    def _load_assistant_jobs(self):
        data = _load_json(self.assistant_jobs_path, [])
        jobs = list(data) if isinstance(data, list) else []
        now = time.time()
        kept = []
        for tr in jobs:
            try:
                status = str(tr.get("status") or "").lower()
                ts = self._parse_timestamp_seconds(tr.get("queued_at") or tr.get("created_at"), now)
                if status != "done" and (now - ts) > 60 * 60 * 12:
                    continue
            except Exception:
                pass
            kept.append(tr)
        # Do not preserve endless active spinners across restarts forever.
        for tr in kept:
            try:
                status = str(tr.get("status") or "").lower()
                ts = self._parse_timestamp_seconds(tr.get("queued_at") or tr.get("created_at"), now)
                if status != "done" and (now - ts) > 60 * 30:
                    tr["status"] = "stale"
            except Exception:
                pass
        self._assistant_jobs = kept[-100:]

    def _save_assistant_jobs(self):
        try:
            _save_json_atomic(self.assistant_jobs_path, self._assistant_jobs)
        except Exception:
            pass

    def _assistant_jobs_active(self) -> bool:
        for tr in list(getattr(self, "_assistant_jobs", []) or []):
            try:
                status = str(tr.get("status") or "").strip().lower()
            except Exception:
                status = ""
            if status and status not in {"done", "stale"}:
                return True
        return False

    def _apply_send_button_guard(self) -> bool:
        """Disable chat actions while the local LLM is busy or a queued FV job is still running."""
        allow = True
        try:
            if self._assistant_jobs_active():
                allow = False
            elif self.chat_thread and self.chat_thread.isRunning():
                allow = False
            elif self.server_boot_thread and self.server_boot_thread.isRunning():
                allow = False
            elif self.server_process and self.server_process.state() != QtCore.QProcess.NotRunning and not self.server_ready:
                allow = False
        except Exception:
            pass
        try:
            self.btn_send.setEnabled(bool(allow))
        except Exception:
            pass
        try:
            # Loading/reloading a GGUF model can use a lot of VRAM. Keep it disabled
            # for the same period as Send, so a FrameVision generation job can finish
            # before the user starts another model load.
            self.btn_load.setEnabled(bool(allow))
        except Exception:
            pass
        return bool(allow)

    def _normalize_prompt_for_match(self, text: str) -> str:
        s = str(text or "").strip().lower()
        s = re.sub(r"^[\s:;,.\-–—>]+", "", s)
        s = re.sub(r"\s+", " ", s)
        return s

    def _extract_job_prompt_for_match(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        for val in (data.get("prompt"), data.get("input"), args.get("prompt"), args.get("label"), data.get("title")):
            s = str(val or "").strip()
            if s:
                return self._normalize_prompt_for_match(s)
        return ""

    def _extract_job_image_path(self, data: Dict[str, Any]) -> str:
        for key in ("produced", "output_path", "output_image", "result_path", "image_path"):
            produced = str(data.get(key) or "").strip()
            if produced and os.path.isfile(produced):
                return produced
        args_obj = data.get("args")
        if isinstance(args_obj, dict):
            for key in ("output_path", "output_image", "out", "output"):
                produced = str(args_obj.get(key) or "").strip()
                if produced and os.path.isfile(produced):
                    return produced
        files = data.get("files") or []
        if isinstance(files, list):
            for item in files:
                p = str(item or "").strip()
                if p and os.path.isfile(p) and _attachment_kind_from_path(p) == "image":
                    return p
        return ""

    def _assistant_output_dirs_for_track(self, track: Dict[str, Any]) -> List[str]:
        """Fallback output folders for assistant-created image previews.

        Some queue adapters finish correctly and update the media player, but the
        done-job JSON may not contain a path shape the chat tracker recognizes.
        In that case, scan the expected output folder for a new image created
        after this assistant request was queued.
        """
        model_id = str(track.get("model_id") or "").strip().lower()
        dirs = []
        # Prefer known output folders first. Include older/common variants because
        # some helpers/queue adapters ignore the registry output_dir and write to
        # their own historical output folder.
        if model_id == "zimage_gguf":
            dirs.extend([
                os.path.join(self.fv_root, "output", "photo", "txt2img"),
                os.path.join(self.fv_root, "output", "txt2img"),
                os.path.join(self.fv_root, "output", "images", "txt2img"),
                os.path.join(self.fv_root, "output", "photo"),
                os.path.join(self.fv_root, "output", "images"),
            ])
        elif model_id == "lens":
            dirs.extend([
                os.path.join(self.fv_root, "output", "lens_turbo_u4"),
                os.path.join(self.fv_root, "output", "images", "lens_turbo_u4"),
                os.path.join(self.fv_root, "output", "images"),
            ])
        elif model_id == "flux_klein":
            dirs.extend([
                os.path.join(self.fv_root, "output", "edits", "flux_klein"),
                os.path.join(self.fv_root, "output", "klein4b_gguf"),
            ])
        elif model_id == "chroma":
            dirs.extend([
                os.path.join(self.fv_root, "output", "images", "chroma"),
                os.path.join(self.fv_root, "output", "chroma"),
            ])
        elif model_id == "hidream":
            dirs.extend([
                os.path.join(self.fv_root, "output", "hidream"),
                os.path.join(self.fv_root, "output", "images", "hidream"),
                os.path.join(self.fv_root, "models", "hidream_bf16", "results"),
                os.path.join(self.fv_root, "models", "hidream", "results"),
            ])
        elif model_id == "seedvr2":
            dirs.extend([
                os.path.join(self.fv_root, "output", "photo"),
                os.path.join(self.fv_root, "output", "upscaled"),
            ])
        elif model_id == "ace_step_15":
            dirs.extend([
                os.path.join(self.fv_root, "output", "audio", "ace15"),
                os.path.join(self.fv_root, "output", "ace_step_15"),
                os.path.join(self.fv_root, "output", "audio"),
            ])
        elif model_id == "ltx23":
            dirs.extend([
                os.path.join(self.fv_root, "output", "video", "ltx23", "chat"),
                os.path.join(self.fv_root, "output", "video", "ltx23"),
                os.path.join(self.fv_root, "output", "ltx_ui"),
                os.path.join(self.fv_root, "output", "video"),
            ])

        # Keep this broad as final fallback. This is what prevents a permanent
        # spinner when the queue finished and the media player sees the result,
        # but the done-job JSON did not contain a path the chat could parse.
        dirs.append(os.path.join(self.fv_root, "output"))

        seen = set()
        clean = []
        for d in dirs:
            try:
                nd = os.path.normcase(os.path.abspath(d))
            except Exception:
                nd = d.lower()
            if nd not in seen and os.path.isdir(d):
                seen.add(nd)
                clean.append(d)
        return clean

    def _scan_output_images_for_track(self, track: Dict[str, Any]) -> str:
        queued_at = self._parse_timestamp_seconds(track.get("queued_at"), 0.0)
        # Do not accept already-finished images from an older request. The old
        # broad grace window could instantly attach the previous edit result while
        # the new queue job had barely started.
        min_time = max(0.0, queued_at + 0.25)
        try:
            age = time.time() - float(queued_at or 0.0)
        except Exception:
            age = 999.0
        # Give the worker a few seconds before scanning folders by timestamp.
        # Expected output paths are checked before this function, so known paths
        # can still resolve immediately when they really exist.
        if age < 4.0:
            return ""
        used_paths = self._assistant_used_image_paths(str(track.get("id") or ""))
        best_path = ""
        best_mtime = -1.0
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        model_id = str(track.get("model_id") or "").strip().lower()

        def _name_score(fn: str) -> int:
            low = fn.lower()
            if model_id == "zimage_gguf" and (low.startswith("z_img_") or "zimage" in low or "z-image" in low):
                return 20
            if model_id == "flux_klein" and ("flux_klein" in low or "klein" in low):
                return 20
            if model_id == "lens" and "lens" in low:
                return 20
            if model_id == "chroma" and "chroma" in low:
                return 20
            if model_id == "hidream" and "hidream" in low:
                return 20
            return 0

        for folder in self._assistant_output_dirs_for_track(track):
            try:
                for root, _dirs, files in os.walk(folder):
                    for fn in files:
                        if Path(fn).suffix.lower() not in image_exts:
                            continue
                        path = os.path.join(root, fn)
                        try:
                            mt = os.path.getmtime(path)
                            if mt < min_time:
                                continue
                            try:
                                norm = os.path.normcase(os.path.abspath(path))
                            except Exception:
                                norm = path.lower()
                            if norm in used_paths:
                                continue
                            score = (mt, _name_score(fn))
                            best_score = (best_mtime, _name_score(os.path.basename(best_path))) if best_path else (-1.0, -1)
                            if score >= best_score:
                                best_path = path
                                best_mtime = mt
                        except Exception:
                            pass
            except Exception:
                pass
        return best_path

    def _audio_exts_for_assistant(self) -> set:
        return {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    def _extract_job_audio_path(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        keys = ("produced", "output", "out_path", "output_path", "result_path", "audio_path", "music_path")
        for key in keys:
            produced = str(data.get(key) or "").strip()
            if produced and os.path.isfile(produced) and Path(produced).suffix.lower() in self._audio_exts_for_assistant():
                return produced
        args_obj = data.get("args")
        if isinstance(args_obj, dict):
            for key in keys:
                produced = str(args_obj.get(key) or "").strip()
                if produced and os.path.isfile(produced) and Path(produced).suffix.lower() in self._audio_exts_for_assistant():
                    return produced
        files = data.get("files") or []
        if isinstance(files, list):
            for item in files:
                p = str(item or "").strip()
                if p and os.path.isfile(p) and Path(p).suffix.lower() in self._audio_exts_for_assistant():
                    return p
        return ""

    def _video_exts_for_assistant(self) -> set:
        return {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

    def _extract_job_video_path(self, data: Dict[str, Any]) -> str:
        if not isinstance(data, dict):
            return ""
        keys = ("produced", "output", "out_path", "output_path", "result_path", "video_path", "outfile")
        for key in keys:
            produced = str(data.get(key) or "").strip()
            if produced and os.path.isfile(produced) and Path(produced).suffix.lower() in self._video_exts_for_assistant():
                return produced
        args_obj = data.get("args")
        if isinstance(args_obj, dict):
            for key in keys:
                produced = str(args_obj.get(key) or "").strip()
                if produced and os.path.isfile(produced) and Path(produced).suffix.lower() in self._video_exts_for_assistant():
                    return produced
        files = data.get("files") or []
        if isinstance(files, list):
            for item in files:
                p = str(item or "").strip()
                if p and os.path.isfile(p) and Path(p).suffix.lower() in self._video_exts_for_assistant():
                    return p
        return ""

    def _scan_output_video_for_track(self, track: Dict[str, Any]) -> str:
        queued_at = self._parse_timestamp_seconds(track.get("queued_at"), 0.0)
        min_time = max(0.0, queued_at + 0.25)
        try:
            age = time.time() - float(queued_at or 0.0)
        except Exception:
            age = 999.0
        if age < 4.0:
            return ""
        expected = str(track.get("expected_output_path") or "").strip()
        if expected and os.path.isfile(expected) and Path(expected).suffix.lower() in self._video_exts_for_assistant():
            return expected
        best_path = ""
        best_mtime = -1.0
        model_id = str(track.get("model_id") or "").strip().lower()
        def _name_score(fn: str) -> int:
            low = fn.lower()
            score = 0
            if model_id == "ltx23" and ("ltx" in low or "glued" in low or "seed" in low):
                score += 15
            return score
        for folder in self._assistant_output_dirs_for_track(track):
            try:
                for root, _dirs, files in os.walk(folder):
                    for fn in files:
                        if Path(fn).suffix.lower() not in self._video_exts_for_assistant():
                            continue
                        path = os.path.join(root, fn)
                        try:
                            mt = os.path.getmtime(path)
                            if mt < min_time:
                                continue
                            score = (mt, _name_score(fn))
                            best_score = (best_mtime, _name_score(os.path.basename(best_path))) if best_path else (-1.0, -1)
                            if score >= best_score:
                                best_path = path
                                best_mtime = mt
                        except Exception:
                            pass
            except Exception:
                pass
        return best_path

    def _scan_output_audio_for_track(self, track: Dict[str, Any]) -> str:
        queued_at = self._parse_timestamp_seconds(track.get("queued_at"), 0.0)
        min_time = max(0.0, queued_at + 0.25)
        try:
            age = time.time() - float(queued_at or 0.0)
        except Exception:
            age = 999.0
        if age < 4.0:
            return ""
        expected = str(track.get("expected_output_path") or "").strip()
        if expected and os.path.isfile(expected) and Path(expected).suffix.lower() in self._audio_exts_for_assistant():
            return expected
        best_path = ""
        best_mtime = -1.0
        title_hint = str(track.get("title_hint") or "").strip().lower()
        sub_hint = str(track.get("subgenre") or "").strip().lower()

        def _name_score(fn: str) -> int:
            low = fn.lower()
            score = 0
            if "ace" in low or "acestep" in low or "ace_step" in low:
                score += 10
            if title_hint and title_hint in low:
                score += 25
            if sub_hint and sub_hint.replace(" ", "_") in low:
                score += 8
            return score

        for folder in self._assistant_output_dirs_for_track(track):
            try:
                for root, _dirs, files in os.walk(folder):
                    for fn in files:
                        if Path(fn).suffix.lower() not in self._audio_exts_for_assistant():
                            continue
                        path = os.path.join(root, fn)
                        try:
                            mt = os.path.getmtime(path)
                            if mt < min_time:
                                continue
                            score = (mt, _name_score(fn))
                            best_score = (best_mtime, _name_score(os.path.basename(best_path))) if best_path else (-1.0, -1)
                            if score >= best_score:
                                best_path = path
                                best_mtime = mt
                        except Exception:
                            pass
            except Exception:
                pass
        return best_path

    def _parse_timestamp_seconds(self, value, default: float = 0.0) -> float:
        """Best-effort timestamp parser for assistant/queue matching."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value or "").strip()
            if not s:
                return float(default)
            try:
                return float(s)
            except Exception:
                pass
            # Queue JSON usually uses either ISO format or "YYYY-MM-DD HH:MM:SS".
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
            except Exception:
                pass
            try:
                return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S").timestamp()
            except Exception:
                return float(default)
        except Exception:
            return float(default)

    def _assistant_used_image_paths(self, except_track_id: str = "") -> set:
        used = set()
        except_track_id = str(except_track_id or "")
        for tr in getattr(self, "_assistant_jobs", []) or []:
            if except_track_id and str(tr.get("id") or "") == except_track_id:
                continue
            p = str(tr.get("image_path") or "").strip()
            if p:
                try:
                    used.add(os.path.normcase(os.path.abspath(p)))
                except Exception:
                    used.add(p.lower())
        return used

    def _job_matches_assistant_track(self, data: Dict[str, Any], track: Dict[str, Any]) -> bool:
        prompt_norm = self._normalize_prompt_for_match(str(track.get("prompt") or ""))
        if not prompt_norm:
            return False
        job_prompt = self._extract_job_prompt_for_match(data)
        if prompt_norm not in job_prompt and job_prompt not in prompt_norm:
            return False
        model_id = str(track.get("model_id") or "").strip().lower()
        args = data.get("args") if isinstance(data.get("args"), dict) else {}
        blob = " ".join([
            str(data.get("engine") or ""),
            str(data.get("backend") or ""),
            str(data.get("type") or ""),
            str(data.get("model") or ""),
            str(args.get("engine") or ""),
        ]).lower()
        if model_id == "lens":
            return "lens" in blob
        if model_id == "zimage_gguf":
            return ("zimage_gguf" in blob) or ("z-image" in blob) or ("txt2img" in blob)
        if model_id == "seedvr2":
            return ("seedvr2" in blob) or ("seedvr" in blob) or ("external_cmd" in blob) or ("upscale" in blob)
        if model_id == "ace_step_15":
            return ("ace_step_15" in blob) or ("ace-step" in blob) or ("acestep" in blob) or ("music" in blob)
        if model_id == "ltx23":
            return ("ltx23" in blob) or ("ltx" in blob) or ("tools_ffmpeg" in blob)
        return True

    def _scan_done_jobs_for_track(self, track: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        done_dir = os.path.join(self.fv_root, "jobs", "done")
        if not os.path.isdir(done_dir):
            return None

        # Critical: do not let a new request reuse an older completed job.
        # The previous version matched only by prompt/model, so after the first
        # generated image it could immediately pick the same old done JSON again.
        queued_at = self._parse_timestamp_seconds(track.get("queued_at"), 0.0)
        try:
            age = time.time() - float(queued_at or 0.0)
        except Exception:
            age = 999.0
        # Do not scan done JSON immediately. Some workers touch/move old done
        # files, and prompt/model matching alone can pick the previous result.
        if age < 4.0:
            return None
        min_time = max(0.0, queued_at + 0.25)
        used_paths = self._assistant_used_image_paths(str(track.get("id") or ""))

        best = None
        best_mtime = -1.0
        for fn in os.listdir(done_dir):
            if not fn.lower().endswith(".json"):
                continue
            path = os.path.join(done_dir, fn)
            try:
                mt = os.path.getmtime(path)
                if mt < min_time:
                    continue

                data = _load_json(path, None)
                if not isinstance(data, dict):
                    continue

                # Prefer queue timestamps too when available.
                finished_ts = self._parse_timestamp_seconds(data.get("finished_at") or data.get("updated_at") or data.get("created_at"), mt)
                if finished_ts < min_time:
                    continue

                if not self._job_matches_assistant_track(data, track):
                    continue

                image_path = self._extract_job_image_path(data)
                if image_path:
                    try:
                        norm_img = os.path.normcase(os.path.abspath(image_path))
                    except Exception:
                        norm_img = image_path.lower()
                    if norm_img in used_paths:
                        continue

                if mt >= best_mtime:
                    best = data
                    best_mtime = mt
            except Exception:
                pass
        return best

    def _find_session_message(self, session_id: str, message_id: str) -> Tuple[Optional[ChatSession], Optional[Dict[str, Any]]]:
        s = self._find_session(session_id)
        if not s:
            return None, None
        for m in s.messages:
            if str(m.get("id") or "") == str(message_id or ""):
                return s, m
        return s, None

    def _update_session_message(self, session_id: str, message_id: str, **fields) -> bool:
        s, msg = self._find_session_message(session_id, message_id)
        if not s or not msg:
            return False
        msg.update(fields)
        s.updated_at = datetime.now().isoformat(timespec="seconds")
        if self.current_session_id == session_id:
            self.chat_view.update_message(message_id, text=fields.get("content"), thinking=fields.get("thinking"), attachments=fields.get("attachments"), loading=fields.get("loading"))
        self._queue_save()
        return True

    def _append_assistant_progress_message(self, text: str, attachments: Optional[List[Dict]] = None, loading: bool = True) -> str:
        s = self._current_session()
        if not s:
            return ""
        now = datetime.now().isoformat(timespec="seconds")
        mid = str(uuid.uuid4())
        msg = {
            "id": mid,
            "role": "assistant",
            "content": str(text or "").strip(),
            "thinking": "",
            "attachments": list(attachments or []),
            "timestamp": now,
            "loading": bool(loading),
        }
        s.messages.append(msg)
        s.updated_at = now
        self.chat_view.add_message("assistant", msg["content"], "", attachments=msg["attachments"], message_id=mid, loading=bool(loading))
        self._queue_save()
        return mid

    def _track_assistant_image_job(self, route) -> None:
        s = self._current_session()
        if not s:
            return
        prompt = str(getattr(route, "prompt", "") or "").strip()
        model_id = str(getattr(route, "model", "") or "").strip()
        model_label = str(getattr(route, "model_label", model_id) or model_id).strip()
        width = int(getattr(route, "width", 0) or 0)
        height = int(getattr(route, "height", 0) or 0)
        # Use the chat-side queue time as the matching boundary, so old done jobs
        # cannot be picked for a brand-new request.
        queued_at = time.time()
        route_mode = str(getattr(route, "mode", "") or "")
        if route_mode == "video":
            progress_text = "Hang on, creating the requested video with LTX 2.3"
        elif route_mode == "edit":
            progress_text = "Hang on, editing the image"
        else:
            progress_text = "Hang on, creating the requested image"
        msg_id = self._append_assistant_progress_message(progress_text, loading=True)
        if not msg_id:
            return
        self._assistant_jobs.append({
            "id": str(uuid.uuid4()),
            "session_id": s.id,
            "message_id": msg_id,
            "prompt": prompt,
            "prompt_norm": self._normalize_prompt_for_match(prompt),
            "model_id": model_id,
            "model_label": model_label,
            "width": width,
            "height": height,
            "queued_at": queued_at,
            "status": "queued",
            "mode": str(getattr(route, "mode", "") or "create"),
            "input_images": list(getattr(route, "input_images", ()) or ()),
            "expected_output_path": str(getattr(route, "output_path", "") or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
        self._save_assistant_jobs()
        self._apply_send_button_guard()

    def _track_seedvr2_upscale_job(self, source_path: str, model_name: str, resolution: int, output_path: str, source_kind: str = "image") -> None:
        s = self._current_session()
        if not s:
            return
        queued_at = time.time()
        is_video_src = str(source_kind or "image").lower() == "video"
        msg_id = self._append_assistant_progress_message("Hang on, upscaling the video with SeedVR2" if is_video_src else "Hang on, upscaling the image", loading=True)
        if not msg_id:
            return
        prompt = f"Upscale to {int(resolution)}p"
        self._assistant_jobs.append({
            "id": str(uuid.uuid4()),
            "session_id": s.id,
            "message_id": msg_id,
            "prompt": prompt,
            "prompt_norm": self._normalize_prompt_for_match(prompt),
            "model_id": "seedvr2",
            "model_label": str(model_name or "SeedVR2"),
            "width": 0,
            "height": 0,
            "queued_at": queued_at,
            "status": "queued",
            "mode": "video" if is_video_src else "upscale",
            "input_images": [] if is_video_src else [str(source_path or "")],
            "input_videos": [str(source_path or "")] if is_video_src else [],
            "expected_output_path": str(output_path or ""),
            "resolution": int(resolution or 0),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
        self._save_assistant_jobs()
        self._apply_send_button_guard()

    def _track_ace15_music_job(self, title_hint: str, genre: str, subgenre: str, duration: float, cfg_path: str, out_dir: str) -> None:
        s = self._current_session()
        if not s:
            return
        queued_at = time.time()
        msg_id = self._append_assistant_progress_message("Hang on, creating the music with Ace-Step 1.5", loading=True)
        if not msg_id:
            return
        title_hint = self._ace15_sanitize_filename_part(title_hint or "")
        prompt = f"Ace Step music: {genre} / {subgenre}".strip()
        self._assistant_jobs.append({
            "id": str(uuid.uuid4()),
            "session_id": s.id,
            "message_id": msg_id,
            "prompt": prompt,
            "prompt_norm": self._normalize_prompt_for_match(prompt),
            "model_id": "ace_step_15",
            "model_label": "Ace-Step 1.5",
            "width": 0,
            "height": 0,
            "queued_at": queued_at,
            "status": "queued",
            "mode": "music",
            "genre": str(genre or ""),
            "subgenre": str(subgenre or ""),
            "duration": float(duration or 0.0),
            "title_hint": title_hint,
            "cfg_path": str(cfg_path or ""),
            "expected_output_path": "",
            "expected_output_dir": str(out_dir or ""),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        })
        self._save_assistant_jobs()
        self._apply_send_button_guard()

    def _poll_assistant_jobs(self):
        if not getattr(self, "_assistant_jobs", None):
            return
        changed = False
        keep = []
        running_queue_jobs = self._running_queue_job_files()
        for track in self._assistant_jobs:
            status_now = str(track.get("status") or "").lower()
            if status_now == "done":
                keep.append(track)
                continue
            if status_now == "stale":
                model_label = str(track.get("model_label") or "Image model")
                mode_now = str(track.get("mode") or "")
                if mode_now == "edit":
                    unknown_text = f"Image edit status is unknown for {model_label}. The job may have finished, but the chat could not find the output image."
                elif mode_now == "upscale":
                    unknown_text = f"Image upscale status is unknown for {model_label}. The job may have finished, but the chat could not find the output image."
                else:
                    unknown_text = f"Image creation status is unknown for {model_label}. The job may have finished, but the chat could not find the output image."
                self._update_session_message(
                    str(track.get("session_id") or ""),
                    str(track.get("message_id") or ""),
                    content=unknown_text,
                    attachments=[],
                    loading=False,
                )
                try:
                    self._set_status("Ready", "ready")
                except Exception:
                    pass
                track["status"] = "done"
                changed = True
                keep.append(track)
                continue
            mode_now = str(track.get("mode") or "")
            expected_path = str(track.get("expected_output_path") or "").strip()
            done_job = self._scan_done_jobs_for_track(track)

            audio_path = ""
            image_path = ""
            video_path = ""
            if mode_now == "music":
                audio_path = expected_path if expected_path and os.path.isfile(expected_path) and Path(expected_path).suffix.lower() in self._audio_exts_for_assistant() else ""
                if not audio_path:
                    audio_path = self._extract_job_audio_path(done_job) if done_job else ""
                if not audio_path:
                    audio_path = self._scan_output_audio_for_track(track)
            elif mode_now == "video":
                video_path = expected_path if expected_path and os.path.isfile(expected_path) and Path(expected_path).suffix.lower() in self._video_exts_for_assistant() else ""
                if not video_path:
                    video_path = self._extract_job_video_path(done_job) if done_job else ""
                if not video_path:
                    video_path = self._scan_output_video_for_track(track)
            else:
                image_path = expected_path if expected_path and os.path.isfile(expected_path) else ""
                if not image_path:
                    image_path = self._extract_job_image_path(done_job) if done_job else ""
                if not image_path:
                    image_path = self._scan_output_images_for_track(track)

            model_label = str(track.get("model_label") or "FrameVision model")
            prompt = str(track.get("prompt") or "").strip()

            # Safer queue-state rule:
            # If FrameVision still has a real job file in jobs/running (>1.5 KB),
            # a generation is still running and the chat must keep waiting. This
            # avoids accepting half-written video/audio/image outputs too early.
            if running_queue_jobs and not done_job:
                keep.append(track)
                continue

            if not done_job and not image_path and not audio_path and not video_path:
                # When a queued job was cancelled/removed, the chat-side tracker can
                # otherwise wait forever. After a small startup grace period, no real
                # running job file means the worker is not processing it anymore.
                queued_at = self._parse_timestamp_seconds(track.get("queued_at") or track.get("created_at"), time.time())
                try:
                    age = time.time() - float(queued_at or 0.0)
                except Exception:
                    age = 999.0
                if age >= 10.0:
                    if mode_now == "music":
                        stopped_text = f"Music job for {model_label} is no longer running. It may have been cancelled, removed, or finished without returning an audio file."
                    elif mode_now == "video":
                        stopped_text = f"Video job for {model_label} is no longer running. It may have been cancelled, removed, or finished without returning a video file."
                    elif mode_now == "upscale":
                        stopped_text = f"Image upscale job for {model_label} is no longer running. It may have been cancelled, removed, or finished without returning an image file."
                    elif mode_now == "edit":
                        stopped_text = f"Image edit job for {model_label} is no longer running. It may have been cancelled, removed, or finished without returning an image file."
                    else:
                        stopped_text = f"Image creation job for {model_label} is no longer running. It may have been cancelled, removed, or finished without returning an image file."
                    self._update_session_message(
                        str(track.get("session_id") or ""),
                        str(track.get("message_id") or ""),
                        content=stopped_text,
                        attachments=[],
                        loading=False,
                    )
                    try:
                        self._set_status("Ready", "ready")
                    except Exception:
                        pass
                    track["status"] = "done"
                    track["finished_at"] = datetime.now().isoformat(timespec="seconds")
                    changed = True
                    keep.append(track)
                    continue

                keep.append(track)
                continue

            if audio_path:
                attachments = [_make_attachment_entry(audio_path)]
                genre = str(track.get("genre") or "").strip()
                sub = str(track.get("subgenre") or "").strip()
                dur = float(track.get("duration") or 0.0)
                dur_txt = f" ({int(round(dur))}s)" if dur > 0 else ""
                style_txt = f"{genre} / {sub}" if genre and sub else (sub or genre or "music")
                text_done = f"Created music with {model_label}: {style_txt}{dur_txt}."
                self._update_session_message(str(track.get("session_id") or ""), str(track.get("message_id") or ""), content=text_done, attachments=attachments, loading=False)
                try:
                    self._set_status("Ready", "ready")
                except Exception:
                    pass
            elif video_path:
                attachments = [_make_attachment_entry(video_path)]
                text_done = f"Created video with {model_label}: {prompt}" if prompt else f"Created video with {model_label}."
                self._update_session_message(str(track.get("session_id") or ""), str(track.get("message_id") or ""), content=text_done, attachments=attachments, loading=False)
                try:
                    self._set_status("Ready", "ready")
                except Exception:
                    pass
            elif image_path:
                attachments = [_make_attachment_entry(image_path)]
                if mode_now == "edit":
                    text_done = f"Edited image with {model_label}: {prompt}" if prompt else f"Edited image with {model_label}."
                elif mode_now == "upscale":
                    res = int(track.get("resolution") or 0)
                    src_name = os.path.basename(str((track.get("input_images") or [""])[0] or ""))
                    if res > 0 and src_name:
                        text_done = f"Upscaled image with {model_label} to {res}p: {src_name}"
                    elif res > 0:
                        text_done = f"Upscaled image with {model_label} to {res}p."
                    else:
                        text_done = f"Upscaled image with {model_label}."
                else:
                    text_done = f"Created image with {model_label}: {prompt}" if prompt else f"Created image with {model_label}."
                self._update_session_message(str(track.get("session_id") or ""), str(track.get("message_id") or ""), content=text_done, attachments=attachments, loading=False)
                try:
                    self._set_status("Ready", "ready")
                except Exception:
                    pass
            else:
                stage = str((done_job or {}).get("stage") or "").strip()
                if mode_now == "music":
                    fail_text = f"Music creation did not produce an audio file for {model_label}." + (f"\n\nStage: {stage}" if stage else "")
                elif mode_now == "video":
                    fail_text = f"Video creation did not produce a video file for {model_label}." + (f"\n\nStage: {stage}" if stage else "")
                elif mode_now == "upscale":
                    fail_text = f"Image upscale did not produce an image for {model_label}." + (f"\n\nStage: {stage}" if stage else "")
                else:
                    fail_text = f"Image creation did not produce an image for {model_label}." + (f"\n\nStage: {stage}" if stage else "")
                self._update_session_message(str(track.get("session_id") or ""), str(track.get("message_id") or ""), content=fail_text, attachments=[], loading=False)
                try:
                    self._set_status("Ready", "ready")
                except Exception:
                    pass
            track["status"] = "done"
            track["finished_at"] = datetime.now().isoformat(timespec="seconds")
            track["image_path"] = image_path
            track["audio_path"] = audio_path
            track["video_path"] = video_path
            changed = True
            keep.append(track)
        self._assistant_jobs = keep[-100:]
        if changed:
            self._save_assistant_jobs()
        self._apply_send_button_guard()

    # ---------- session management ----------
    def _maybe_reset_template_after_model_change(self) -> bool:
        current_model = self.current_model_path()
        previous_model = self._last_ui_model_path
        if current_model == previous_model:
            return False
        self._last_ui_model_path = current_model
        if not current_model:
            return False

        tk, tv = self.current_template_choice()
        if tk != "builtin" or not tv:
            return False

        guessed_kind, guessed_value = _guess_template_from_model_path(current_model)
        if guessed_kind != "builtin" or not guessed_value:
            return False
        if tv == guessed_value:
            return False

        self.settings_dialog.select_template("auto", "")
        self._set_status(f"Switched template to Auto for this model ({guessed_value}).", "idle")
        return True

    def _settings_changed(self):
        if self._syncing_model_selectors or self._loading_session_state:
            return
        self._maybe_reset_template_after_model_change()
        self._apply_settings_to_current_session()
        self._apply_style()
        self._rebuild_quick_model_combo()
        self._update_header()
        self._queue_save()

    def _load_sessions(self):
        self.sessions = []
        if os.path.isdir(self.chat_dir):
            for fn in os.listdir(self.chat_dir):
                if not fn.lower().endswith(".json"):
                    continue
                data = _load_json(os.path.join(self.chat_dir, fn), None)
                if not isinstance(data, dict):
                    continue
                try:
                    self.sessions.append(ChatSession(
                        id=str(data.get("id", os.path.splitext(fn)[0])),
                        title=str(data.get("title", "Chat")),
                        created_at=str(data.get("created_at", "")),
                        updated_at=str(data.get("updated_at", "")),
                        model_path=str(data.get("model_path", "")),
                        template_kind=str(data.get("template_kind", "auto")),
                        template_value=str(data.get("template_value", "")),
                        system_prompt=str(data.get("system_prompt", "")),
                        messages=list(data.get("messages", [])),
                    ))
                except Exception:
                    pass
        self.sessions.sort(key=lambda s: s.updated_at or "", reverse=True)
        self._refresh_chat_list()

    def _refresh_chat_list(self):
        filter_text = self.ed_search.text().strip().lower()
        self.lst_chats.blockSignals(True)
        self.lst_chats.clear()
        for s in self.sessions:
            if filter_text and filter_text not in s.title.lower():
                continue
            item = QtWidgets.QListWidgetItem(s.title)
            item.setData(QtCore.Qt.UserRole, s.id)
            self.lst_chats.addItem(item)
        self.lst_chats.blockSignals(False)

    def _apply_chat_filter(self):
        self._refresh_chat_list()

    def _find_session(self, session_id: str) -> Optional[ChatSession]:
        for s in self.sessions:
            if s.id == session_id:
                return s
        return None

    def _current_session(self) -> Optional[ChatSession]:
        return self._find_session(self.current_session_id) if self.current_session_id else None

    def _new_chat(self):
        s = ChatSession.create_default()
        s.model_path = self.current_model_path()
        tk, tv = self.current_template_choice()
        s.template_kind = tk
        s.template_value = tv
        s.system_prompt = self.settings_dialog.ed_system.toPlainText()
        self.sessions.insert(0, s)
        self._refresh_chat_list()
        self._select_session(s.id)
        self._queue_save()

    def _delete_current_chat(self):
        s = self._current_session()
        if not s:
            return
        if QtWidgets.QMessageBox.question(self, "Delete chat", f"Delete '{s.title}'?") != QtWidgets.QMessageBox.Yes:
            return
        self.sessions = [x for x in self.sessions if x.id != s.id]
        try:
            os.remove(os.path.join(self.chat_dir, f"{s.id}.json"))
        except Exception:
            pass
        self._refresh_chat_list()
        if self.sessions:
            self._select_session(self.sessions[0].id)
        else:
            self._new_chat()
        self._queue_save()

    def _select_session(self, session_id: str):
        s = self._find_session(session_id)
        if not s:
            return
        self.current_session_id = s.id
        self._load_session_into_ui(s)
        self._render_session(s)
        for i in range(self.lst_chats.count()):
            item = self.lst_chats.item(i)
            if item.data(QtCore.Qt.UserRole) == session_id:
                self.lst_chats.setCurrentRow(i)
                break
        self._update_header()
        self._queue_save()

    def _on_chat_row_changed(self, row: int):
        if row < 0 or row >= self.lst_chats.count():
            return
        sid = self.lst_chats.item(row).data(QtCore.Qt.UserRole)
        if sid and sid != self.current_session_id:
            self._select_session(sid)

    def _load_session_into_ui(self, s: ChatSession):
        self._loading_session_state = True
        try:
            self.settings_dialog.select_model_by_path(s.model_path)
            self.settings_dialog.select_template(s.template_kind, s.template_value)
            self._last_ui_model_path = self.current_model_path()
            self.settings_dialog.ed_system.blockSignals(True)
            self.settings_dialog.ed_system.setPlainText(s.system_prompt or "")
            self.settings_dialog.ed_system.blockSignals(False)
            self._rebuild_quick_model_combo()
        finally:
            self._loading_session_state = False

    def _render_session(self, s: ChatSession):
        self.chat_view.clear_messages()
        if not s.messages:
            self.chat_view.add_message(
                "info",
                "Pick a local GGUF model, load it, then start chatting. use trigger words to enable extra chat options (check info in settings).",
            )
            return
        for m in s.messages:
            role = str(m.get("role", "assistant"))
            if role not in ("user", "assistant", "info"):
                role = "assistant"
            mid = str(m.get("id") or uuid.uuid4())
            m["id"] = mid
            self.chat_view.add_message(role, str(m.get("content", "")), str(m.get("thinking", "")), attachments=list(m.get("attachments", []) or []), message_id=mid, loading=bool(m.get("loading", False)))
            self._refresh_one_message_version_controls(m)

    def _apply_settings_to_current_session(self):
        s = self._current_session()
        if not s:
            return
        s.model_path = self.current_model_path()
        tk, tv = self.current_template_choice()
        s.template_kind = tk
        s.template_value = tv
        s.system_prompt = self.settings_dialog.ed_system.toPlainText()
        s.updated_at = datetime.now().isoformat(timespec="seconds")

    def _sync_models_from_settings(self):
        self.settings_dialog.refresh_models()
        self._rebuild_quick_model_combo()
        self._queue_save()

    def _rebuild_quick_model_combo(self):
        current = self.current_model_path()
        self._syncing_model_selectors = True
        old_block = self.cmb_model_quick.blockSignals(True)
        try:
            self.cmb_model_quick.clear()
            for i in range(self.settings_dialog.cmb_model.count()):
                self.cmb_model_quick.addItem(
                    self.settings_dialog.cmb_model.itemText(i),
                    self.settings_dialog.cmb_model.itemData(i),
                )
            if current:
                for i in range(self.cmb_model_quick.count()):
                    if (self.cmb_model_quick.itemData(i) or "") == current:
                        self.cmb_model_quick.setCurrentIndex(i)
                        break
        finally:
            self.cmb_model_quick.blockSignals(old_block)
            self._syncing_model_selectors = False

    def _quick_model_changed(self):
        if self._syncing_model_selectors or self._loading_session_state:
            return
        path = self.cmb_model_quick.currentData()
        if not path:
            return
        self._syncing_model_selectors = True
        try:
            self.settings_dialog.select_model_by_path(str(path))
        finally:
            self._syncing_model_selectors = False
        self._settings_changed()

    def _show_settings(self):
        self.settings_dialog.refresh_models()
        self.settings_dialog.refresh_templates()
        self.settings_dialog.showMaximized()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def current_model_path(self) -> str:
        return self.settings_dialog.current_model_path()

    def current_template_choice(self) -> Tuple[str, str]:
        return self.settings_dialog.current_template_choice()

    # ---------- runner / loading ----------
    def _set_status(self, text: str, kind: str = "idle"):
        mapping = {
            "idle": "StatusIdle",
            "loading": "StatusLoading",
            "ready": "StatusReady",
            "error": "StatusError",
        }
        self.lbl_status.setObjectName(mapping.get(kind, "StatusIdle"))
        self.lbl_status.setText(text)
        self.style().unpolish(self.lbl_status)
        self.style().polish(self.lbl_status)

    def _append_server_log(self, text: str):
        if not text:
            return
        for line in text.replace("\r", "\n").split("\n"):
            line = line.strip()
            if not line:
                continue
            self.server_log_tail.append(line)
            if len(self.server_log_tail) > 40:
                self.server_log_tail = self.server_log_tail[-40:]

    def _running_queue_job_files(self) -> List[str]:
        """Return running queue marker/job files above 1.5 KB.

        FrameVision queue jobs can leave tiny stale/finished marker files behind.
        Those are ignored here so only real active running jobs block LLM loading.
        """
        running_dir = os.path.join(self.fv_root, "jobs", "running")
        found: List[str] = []
        try:
            if not os.path.isdir(running_dir):
                return found
            for name in os.listdir(running_dir):
                path = os.path.join(running_dir, name)
                if not os.path.isfile(path):
                    continue
                try:
                    if os.path.getsize(path) > 1536:
                        found.append(path)
                except OSError:
                    continue
        except Exception:
            return []
        found.sort(key=lambda x: os.path.basename(x).lower())
        return found

    def _confirm_no_running_queue_jobs_before_llm_load(self) -> bool:
        """Block VRAM-heavy LLM loading while FrameVision queue jobs are running.

        OK cancels the current load attempt. Refresh checks jobs/running again and
        allows the load to continue only after no real running job files remain.
        """
        while True:
            jobs = self._running_queue_job_files()
            if not jobs:
                return True

            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Queue jobs running")
            msg.setText(
                "Running jobs found in the queue. Please finish or cancel running jobs before loading a Vram heavy large language model."
            )
            detail_lines = [
                f"Checked: {os.path.join(self.fv_root, 'jobs', 'running')}",
                "Ignored files smaller than or equal to 1.5 KB.",
                "",
                "Blocking files:",
            ]
            for path in jobs[:20]:
                try:
                    size_kb = os.path.getsize(path) / 1024.0
                    detail_lines.append(f"- {os.path.basename(path)} ({size_kb:.1f} KB)")
                except OSError:
                    detail_lines.append(f"- {os.path.basename(path)}")
            if len(jobs) > 20:
                detail_lines.append(f"- ... and {len(jobs) - 20} more")
            msg.setDetailedText("\n".join(detail_lines))
            ok_btn = msg.addButton("OK", QtWidgets.QMessageBox.RejectRole)
            refresh_btn = msg.addButton("Refresh", QtWidgets.QMessageBox.ActionRole)
            msg.setDefaultButton(refresh_btn)
            msg.exec()
            if msg.clickedButton() == refresh_btn:
                continue
            return False

    def _same_loaded_config(self) -> bool:
        return bool(
            self.server_ready
            and self.loaded_model_path
            and os.path.normcase(os.path.normpath(self.loaded_model_path))
            == os.path.normcase(os.path.normpath(self.current_model_path() or ""))
            and self.loaded_template == self.current_template_choice()
            and int(self.loaded_ctx_size or 0) == int(self.settings_dialog.sp_ctx_size.value())
        )

    def _effective_template_choice(self, model_path: str, template_kind: str, template_value: str) -> Tuple[str, str]:
        if template_kind == "smart":
            guessed_kind, guessed_value = _guess_template_from_model_path(model_path)
            if guessed_kind != "auto":
                return guessed_kind, guessed_value
            return ("auto", "")
        return (template_kind, template_value)

    def _build_server_args(self, model_path: str, template_kind: str, template_value: str, port: int) -> List[str]:
        effective_kind, effective_value = self._effective_template_choice(model_path, template_kind, template_value)
        self.effective_template_on_server = (effective_kind, effective_value)
        args = [
            "-m", model_path,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-c", str(int(self.settings_dialog.sp_ctx_size.value())),
            "--reasoning-budget", "0",
        ]
        mmproj_path = _find_mmproj_for_model(model_path)
        self.active_mmproj_path = mmproj_path
        if mmproj_path:
            args.extend(["--mmproj", mmproj_path])
        if effective_kind == "jinja":
            args.append("--jinja")
        elif effective_kind == "builtin" and effective_value:
            args.extend(["--chat-template", effective_value])
        elif effective_kind == "file" and effective_value:
            args.append("--jinja")
            args.extend(["--chat-template-file", os.path.join(_templates_root(self.fv_root), effective_value)])
        return args

    def _runner_path(self) -> str:
        return self.settings_dialog.ed_runner.text().strip()

    def _validate_runner_and_model(self) -> Tuple[str, str]:
        runner = self._runner_path()
        model = self.current_model_path()
        if not runner:
            raise RuntimeError("Select a llama-server runner in Settings first.")
        if not model:
            raise RuntimeError("Select a GGUF model first.")
        server_exe = _resolve_server_executable(runner)
        if not os.path.isfile(server_exe):
            raise RuntimeError(f"Runner not found: {server_exe}")
        if not os.path.isfile(model):
            raise RuntimeError(f"Model not found: {model}")
        return server_exe, model

    def _cleanup_boot_thread(self):
        if self.server_boot_thread:
            try:
                self.server_boot_thread.requestInterruption()
                self.server_boot_thread.wait(500)
            except Exception:
                pass
            self.server_boot_thread = None

    def _load_selected_model(self):
        try:
            server_exe, model = self._validate_runner_and_model()
        except Exception as e:
            self._set_status(str(e), "error")
            QtWidgets.QMessageBox.warning(self, "Cannot load model", str(e))
            return

        if self._same_loaded_config():
            self._set_status("Ready", "ready")
            return

        if not self._confirm_no_running_queue_jobs_before_llm_load():
            self._set_status("LLM load cancelled: queue job running", "idle")
            return

        self._cleanup_boot_thread()
        self._stop_process_only()

        self.server_port = _pick_free_port()
        self.server_url = f"http://127.0.0.1:{self.server_port}"
        template_kind, template_value = self.current_template_choice()
        args = self._build_server_args(model, template_kind, template_value, self.server_port)

        proc = QtCore.QProcess(self)
        proc.setProgram(server_exe)
        proc.setArguments(args)
        proc.setWorkingDirectory(os.path.dirname(server_exe))
        proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_server_output)
        proc.finished.connect(self._on_server_finished)
        proc.errorOccurred.connect(self._on_server_error)

        self.server_process = proc
        self.server_ready = False
        self.loaded_model_path = ""
        self.loaded_template = ("auto", "")
        self.effective_template_on_server = ("auto", "")
        self.active_mmproj_path = ""
        self.active_mmproj_path = ""
        self.server_log_tail = []
        self.pending_generate_session_id = self.pending_generate_session_id or ""
        self.pending_retry_generation = False if not self.pending_retry_generation else self.pending_retry_generation

        self.btn_load.setEnabled(False)
        self.btn_unload.setEnabled(True)
        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status("Loading…", "loading")
        self._update_header()

        proc.start()
        if not proc.waitForStarted(2500):
            err = proc.errorString() or "Failed to start the local runner."
            self._remove_llm_lock()
            self._set_status(err, "error")
            self.btn_load.setEnabled(True)
            self.btn_unload.setEnabled(False)
            self._apply_send_button_guard()
            self.btn_stop.setEnabled(False)
            return

        self._write_llm_lock(model, server_exe)

        self.server_boot_thread = ServerBootThread(self.server_url, 240, self)
        self.server_boot_thread.statusChanged.connect(self._on_boot_status)
        self.server_boot_thread.ready.connect(self._on_server_ready)
        self.server_boot_thread.failed.connect(self._on_server_failed)
        self.server_boot_thread.start()

    def _write_llm_lock(self, model_path: str = "", runner_path: str = "") -> None:
        try:
            pid = 0
            if self.server_process is not None:
                try:
                    pid = int(self.server_process.processId() or 0)
                except Exception:
                    pid = 0
            if pid > 0:
                _write_llm_gpu_lock(self.fv_root, pid, model_path or self.current_model_path(), runner_path or self._runner_path())
                self._llm_gpu_lock_active = True
        except Exception:
            pass

    def _remove_llm_lock(self) -> None:
        _remove_llm_gpu_lock(self.fv_root)
        self._llm_gpu_lock_active = False

    def _on_server_output(self):
        if not self.server_process:
            return
        data = bytes(self.server_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_server_log(data)

    def _on_server_error(self, _err):
        if self.server_ready:
            return
        msg = self.server_process.errorString() if self.server_process else "Local runner error."
        self._set_status(msg, "error")

    def _on_server_finished(self, *_args):
        self._remove_llm_lock()
        self._cleanup_boot_thread()
        was_ready = self.server_ready
        self.server_ready = False
        self.loaded_ctx_size = 0
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(False)
        self._apply_send_button_guard()
        self.btn_stop.setEnabled(False)
        if was_ready:
            self._set_status("Unloaded", "idle")
        else:
            tail = self.server_log_tail[-1] if self.server_log_tail else "Local server stopped."
            self._set_status(tail, "error")
            if self.server_log_tail:
                detail = "\n".join(self.server_log_tail[-12:])
                try:
                    QtWidgets.QMessageBox.warning(self, "Model load failed", detail)
                except Exception:
                    pass
        self._update_header()
        self.server_process = None

    def _on_boot_status(self, text: str):
        self._set_status(text or "Loading…", "loading")

    def _on_server_ready(self):
        self.server_ready = True
        self.loaded_model_path = self.current_model_path()
        self.loaded_template = self.current_template_choice()
        self.loaded_ctx_size = int(self.settings_dialog.sp_ctx_size.value())
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(True)
        self._apply_send_button_guard()
        self.btn_stop.setEnabled(False)
        self._set_status("Ready", "ready")
        self._update_header()
        if self.pending_retry_generation:
            self.pending_retry_generation = False
            QtCore.QTimer.singleShot(0, self._start_reply_request_for_current_session)
            return
        if self.pending_generate_session_id:
            target_session_id = self.pending_generate_session_id
            self.pending_generate_session_id = ""
            QtCore.QTimer.singleShot(0, lambda sid=target_session_id: self._start_reply_request_for_session_id(sid))

    def _on_server_failed(self, message: str):
        self._remove_llm_lock()
        tail = self.server_log_tail[-1] if self.server_log_tail else ""
        detail = message if not tail else f"{message}\n\nLast log line: {tail}"
        self._set_status(detail.splitlines()[0], "error")
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(False)
        self._apply_send_button_guard()
        self.btn_stop.setEnabled(False)
        self.pending_generate_session_id = ""

    def _stop_process_only(self):
        self._remove_llm_lock()
        if self.server_process:
            try:
                if self.server_process.state() != QtCore.QProcess.NotRunning:
                    self.server_process.terminate()
                    if not self.server_process.waitForFinished(2500):
                        self.server_process.kill()
                        self.server_process.waitForFinished(2500)
            except Exception:
                pass
        self.server_process = None
        self.server_ready = False
        self.loaded_model_path = ""
        self.loaded_ctx_size = 0
        self.loaded_template = ("auto", "")
        self.effective_template_on_server = ("auto", "")

    def _unload_model(self):
        self.pending_generate_session_id = ""
        self._cleanup_boot_thread()
        self._stop_process_only()
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(False)
        self._apply_send_button_guard()
        self.btn_stop.setEnabled(False)
        self._set_status("Unloaded", "idle")
        self._update_header()

    # ---------- message versions / edit + retry ----------
    def _ensure_message_versions(self, msg: Dict[str, Any]) -> List[Dict[str, Any]]:
        versions = msg.get("versions")
        if not isinstance(versions, list) or not versions:
            versions = [{
                "content": str(msg.get("content", "") or ""),
                "thinking": str(msg.get("thinking", "") or ""),
                "attachments": list(msg.get("attachments", []) or []),
                "timestamp": str(msg.get("timestamp", "") or datetime.now().isoformat(timespec="seconds")),
            }]
            msg["versions"] = versions
            msg["active_version"] = 0
        try:
            idx = int(msg.get("active_version", 0) or 0)
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(versions) - 1))
        msg["active_version"] = idx
        return versions

    def _apply_active_message_version(self, msg: Dict[str, Any]) -> None:
        versions = self._ensure_message_versions(msg)
        idx = int(msg.get("active_version", 0) or 0)
        v = versions[idx] if 0 <= idx < len(versions) else versions[0]
        msg["content"] = str(v.get("content", "") or "")
        msg["thinking"] = str(v.get("thinking", "") or "")
        msg["attachments"] = list(v.get("attachments", []) or [])

    def _message_version_state(self, msg: Dict[str, Any]) -> Tuple[str, bool, bool]:
        versions = msg.get("versions")
        if not isinstance(versions, list) or len(versions) <= 1:
            return "", False, False
        try:
            idx = int(msg.get("active_version", 0) or 0)
        except Exception:
            idx = 0
        idx = max(0, min(idx, len(versions) - 1))
        return f"{idx + 1}/{len(versions)}", idx > 0, idx < len(versions) - 1

    def _refresh_one_message_version_controls(self, msg: Dict[str, Any]) -> None:
        if not isinstance(msg, dict):
            return
        mid = str(msg.get("id") or "")
        if not mid:
            return
        label, can_prev, can_next = self._message_version_state(msg)
        try:
            self.chat_view.set_version_controls(mid, label, can_prev, can_next)
        except Exception:
            pass

    def _refresh_all_message_version_controls(self, session: Optional[ChatSession] = None) -> None:
        s = session or self._current_session()
        if not s:
            return
        for msg in s.messages:
            if isinstance(msg, dict):
                self._refresh_one_message_version_controls(msg)

    def _find_message_index(self, session: ChatSession, message_id: str) -> int:
        for i, msg in enumerate(session.messages):
            if str(msg.get("id") or "") == str(message_id or ""):
                return i
        return -1

    def _build_api_messages_until(self, session: ChatSession, stop_before_message_id: str) -> List[Dict]:
        idx = self._find_message_index(session, stop_before_message_id)
        if idx < 0:
            return self._build_api_messages(session)
        shadow = ChatSession(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            model_path=session.model_path,
            template_kind=session.template_kind,
            template_value=session.template_value,
            system_prompt=session.system_prompt,
            messages=[dict(m) for m in session.messages[:idx]],
        )
        return self._build_api_messages(shadow)

    def _start_generation_for_message_update(self, assistant_message_id: str):
        s = self._current_session()
        if not s:
            return
        try:
            self._validate_runner_and_model()
        except Exception as e:
            self._set_status(str(e), "error")
            QtWidgets.QMessageBox.warning(self, "Cannot retry", str(e))
            return
        self._pending_generation_update = {
            "session_id": s.id,
            "assistant_message_id": str(assistant_message_id or ""),
        }
        tk, tv = self.current_template_choice()
        self._prepare_auto_retry_templates(self.current_model_path(), tk, tv)
        if not self._same_loaded_config() or not self.server_ready:
            self.pending_generate_session_id = s.id
            self._load_selected_model()
            return
        self._start_reply_request_for_current_session()

    def _on_chat_message_retry_requested(self, message_id: str):
        if self._assistant_jobs_active() or (self.chat_thread and self.chat_thread.isRunning()):
            self._apply_send_button_guard()
            self._set_status("Busy — retry is disabled until the current job is finished", "loading")
            return
        s = self._current_session()
        if not s:
            return
        idx = self._find_message_index(s, message_id)
        if idx < 0:
            return
        msg = s.messages[idx]
        if str(msg.get("role") or "") != "assistant":
            return
        self._ensure_message_versions(msg)
        msg["loading"] = True
        self.chat_view.update_message(str(msg.get("id") or ""), loading=True)
        self._refresh_one_message_version_controls(msg)
        self._start_generation_for_message_update(str(msg.get("id") or ""))

    def _on_chat_message_edit_saved(self, message_id: str, new_text: str):
        if self._assistant_jobs_active() or (self.chat_thread and self.chat_thread.isRunning()):
            self._apply_send_button_guard()
            self._set_status("Busy — edit/save is disabled until the current job is finished", "loading")
            return
        s = self._current_session()
        if not s:
            return
        idx = self._find_message_index(s, message_id)
        if idx < 0:
            return
        msg = s.messages[idx]
        if str(msg.get("role") or "") != "user":
            return
        new_text = str(new_text or "").strip()
        if not new_text:
            self._set_status("Edited message is empty", "error")
            return

        versions = self._ensure_message_versions(msg)
        cur_idx = int(msg.get("active_version", 0) or 0)
        cur_text = str(versions[cur_idx].get("content", "") if 0 <= cur_idx < len(versions) else msg.get("content", "") or "")
        if new_text != cur_text:
            versions.append({
                "content": new_text,
                "thinking": "",
                "attachments": list(msg.get("attachments", []) or []),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            })
            msg["active_version"] = len(versions) - 1
        self._apply_active_message_version(msg)
        self.chat_view.update_message(str(msg.get("id") or ""), text=str(msg.get("content", "")), attachments=list(msg.get("attachments", []) or []), loading=False)
        self._refresh_one_message_version_controls(msg)

        # One edited user turn gets one matching assistant turn. Keep earlier context,
        # keep the old answer as version 1, and remove later active-branch messages.
        answer_idx = idx + 1 if idx + 1 < len(s.messages) and str(s.messages[idx + 1].get("role") or "") == "assistant" else -1
        if answer_idx >= 0:
            answer_msg = s.messages[answer_idx]
            self._ensure_message_versions(answer_msg)
            if len(s.messages) > answer_idx + 1:
                del s.messages[answer_idx + 1:]
                self._render_session(s)
            answer_msg["loading"] = True
            self.chat_view.update_message(str(answer_msg.get("id") or ""), loading=True)
            self._refresh_one_message_version_controls(answer_msg)
            target_assistant_id = str(answer_msg.get("id") or "")
        else:
            now = datetime.now().isoformat(timespec="seconds")
            target_assistant_id = str(uuid.uuid4())
            answer_msg = {
                "id": target_assistant_id,
                "role": "assistant",
                "content": "",
                "thinking": "",
                "attachments": [],
                "timestamp": now,
                "loading": True,
                "versions": [],
                "active_version": 0,
            }
            s.messages.insert(idx + 1, answer_msg)
            if len(s.messages) > idx + 2:
                del s.messages[idx + 2:]
            self.chat_view.add_message("assistant", "Regenerating…", "", attachments=[], message_id=target_assistant_id, loading=True)

        s.updated_at = datetime.now().isoformat(timespec="seconds")
        self._last_user_text_sent = new_text
        self._refresh_chat_list()
        self._update_header()
        self._queue_save()
        self._start_generation_for_message_update(target_assistant_id)

    def _switch_message_version(self, message_id: str, direction: int):
        s = self._current_session()
        if not s:
            return
        idx = self._find_message_index(s, message_id)
        if idx < 0:
            return
        msg = s.messages[idx]
        versions = self._ensure_message_versions(msg)
        if len(versions) <= 1:
            return
        cur = int(msg.get("active_version", 0) or 0)
        new_idx = max(0, min(len(versions) - 1, cur + int(direction or 0)))
        if new_idx == cur:
            return
        msg["active_version"] = new_idx
        self._apply_active_message_version(msg)
        self.chat_view.update_message(str(msg.get("id") or ""), text=str(msg.get("content", "")), thinking=str(msg.get("thinking", "")), attachments=list(msg.get("attachments", []) or []), loading=bool(msg.get("loading", False)))
        self._refresh_one_message_version_controls(msg)

        # If this is a user message and the next assistant answer has matching
        # versions, move that answer to the same version number too. This gives
        # the expected ChatGPT-style user edit + answer 1/2, 2/2 behavior.
        if str(msg.get("role") or "") == "user" and idx + 1 < len(s.messages):
            nxt = s.messages[idx + 1]
            if str(nxt.get("role") or "") == "assistant":
                nvers = self._ensure_message_versions(nxt)
                if len(nvers) > new_idx:
                    nxt["active_version"] = new_idx
                    self._apply_active_message_version(nxt)
                    self.chat_view.update_message(str(nxt.get("id") or ""), text=str(nxt.get("content", "")), thinking=str(nxt.get("thinking", "")), attachments=list(nxt.get("attachments", []) or []), loading=bool(nxt.get("loading", False)))
                    self._refresh_one_message_version_controls(nxt)
        s.updated_at = datetime.now().isoformat(timespec="seconds")
        self._queue_save()

    def _complete_pending_generation_update(self, reply: str, thinking: str, attachments: Optional[List[Dict]] = None) -> bool:
        pending = self._pending_generation_update if isinstance(self._pending_generation_update, dict) else None
        if not pending:
            return False
        session_id = str(pending.get("session_id") or "")
        assistant_message_id = str(pending.get("assistant_message_id") or "")
        self._pending_generation_update = None
        s, msg = self._find_session_message(session_id, assistant_message_id)
        if not s or not msg:
            return False
        existing_versions = msg.get("versions")
        if isinstance(existing_versions, list) and len(existing_versions) == 0 and not str(msg.get("content", "") or "").strip():
            versions = existing_versions
        else:
            versions = self._ensure_message_versions(msg)
        versions.append({
            "content": str(reply or "").strip(),
            "thinking": str(thinking or "").strip(),
            "attachments": list(attachments or []),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })
        msg["versions"] = versions
        msg["active_version"] = len(versions) - 1
        msg["loading"] = False
        self._apply_active_message_version(msg)
        s.updated_at = datetime.now().isoformat(timespec="seconds")
        if self.current_session_id == session_id:
            self.chat_view.update_message(assistant_message_id, text=str(msg.get("content", "")), thinking=str(msg.get("thinking", "")), attachments=list(msg.get("attachments", []) or []), loading=False)
            self._refresh_one_message_version_controls(msg)
            self.chat_view.scroll_to_bottom(force=True)
        self._refresh_chat_list()
        self._update_header()
        self._queue_save()
        self._set_status("Ready", "ready")
        return True

    # ---------- Knowledge & Memory ----------
    def _memory_enabled(self) -> bool:
        try:
            return bool(self.settings_dialog.chk_memory_enabled.isChecked())
        except Exception:
            return True

    def _memory_show_sources(self) -> bool:
        try:
            return bool(self.settings_dialog.chk_memory_sources.isChecked())
        except Exception:
            return True

    def _reload_startup_llm_memory(self) -> None:
        _ensure_memory_folders(self.fv_root)
        parts: List[str] = []
        for path in _iter_memory_files(_memory_llm_memory_dir(self.fv_root))[:20]:
            try:
                text = _read_memory_text_file(path, max_chars=16000).strip()
            except Exception:
                text = ""
            if not text:
                continue
            rel = os.path.relpath(path, self.fv_root)
            parts.append(f"[llm_memory: {rel}]\n{text[:12000]}")
            if sum(len(x) for x in parts) > 30000:
                break
        self._startup_llm_memory_context = "\n\n".join(parts).strip()

    def _message_text_by_id(self, message_id: str) -> str:
        s = self._current_session()
        if not s:
            return ""
        idx = self._find_message_index(s, message_id)
        if idx < 0:
            return ""
        return str(s.messages[idx].get("content", "") or "")

    def _write_memory_note(self, title: str, text: str, folder: str, extra: Optional[Dict[str, Any]] = None) -> str:
        _ensure_memory_folders(self.fv_root)
        os.makedirs(folder, exist_ok=True)
        stamp = _memory_now_stamp()
        name = f"{stamp}_{_safe_memory_filename(title or 'memory')}.json"
        path = os.path.join(folder, name)
        payload = {
            "title": str(title or "Memory"),
            "created": datetime.now().isoformat(timespec="seconds"),
            "text": str(text or ""),
            "source": "llm_chat",
        }
        if isinstance(extra, dict):
            payload.update(extra)
        _save_json_atomic(path, payload)
        return path

    def _extract_memory_save_text(self, text: str) -> Tuple[bool, str, str]:
        raw = str(text or "").strip()
        if not raw:
            return False, "", ""
        patterns = [
            r"^\s*(?:save\s+to\s+memory|remember\s+this|remember|note\s+this|save\s+this\s+to\s+memory)\s*[:\-]?\s*(.+)$",
            r"^\s*(?:can\s+you\s+)?(?:save|store)\s+(?:this\s+)?(?:in|to)\s+memory\s*[:\-]?\s*(.+)$",
        ]
        for pat in patterns:
            m = re.match(pat, raw, re.IGNORECASE | re.DOTALL)
            if m:
                body = (m.group(1) or "").strip()
                if body:
                    title = body.splitlines()[0].strip()[:60]
                    return True, title or "Saved memory", body
        return False, "", ""

    def _on_save_message_to_memory_requested(self, message_id: str):
        text = self._message_text_by_id(message_id).strip()
        if not text:
            return
        default_title = text.splitlines()[0].strip()[:60] or "Saved chat message"
        title, ok = QtWidgets.QInputDialog.getText(self, "Save to memory", "Memory name:", text=default_title)
        if not ok:
            return
        title = str(title or default_title).strip() or default_title
        try:
            path = self._write_memory_note(title, text, _memory_saved_notes_dir(self.fv_root), {"from_message_id": str(message_id or "")})
            self._set_status(f"Saved memory: {os.path.basename(path)}", "ready")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save to memory failed", str(e))

    def _pick_or_create_project_folder(self) -> Tuple[str, str]:
        root = _memory_project_dir(self.fv_root)
        os.makedirs(root, exist_ok=True)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Save to project")
        lay = QtWidgets.QVBoxLayout(dlg)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)
        info = QtWidgets.QLabel("Create a new project folder or continue an existing one.")
        info.setWordWrap(True)
        lay.addWidget(info)
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("New project name"))
        ed_name = QtWidgets.QLineEdit()
        ed_name.setPlaceholderText("my_project")
        name_row.addWidget(ed_name, 1)
        lay.addLayout(name_row)
        chosen_label = QtWidgets.QLabel("Existing project: none selected")
        chosen_label.setObjectName("SubtleLabel")
        lay.addWidget(chosen_label)
        picked = {"path": ""}
        def browse():
            path = QtWidgets.QFileDialog.getExistingDirectory(dlg, "Select existing project folder", root)
            if path:
                picked["path"] = path
                chosen_label.setText("Existing project: " + path)
        btn_browse = QtWidgets.QPushButton("Continue existing project…")
        btn_browse.clicked.connect(browse)
        lay.addWidget(btn_browse)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        lay.addWidget(btns)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        if dlg.exec() != QtWidgets.QDialog.Accepted:
            return "", ""
        if picked.get("path"):
            path = str(picked["path"])
            return path, os.path.basename(path.rstrip(os.sep)) or "project"
        name = _safe_memory_filename(ed_name.text().strip() or "project", "project")
        path = os.path.join(root, name)
        os.makedirs(path, exist_ok=True)
        return path, name

    def _on_save_message_to_project_requested(self, message_id: str):
        text = self._message_text_by_id(message_id).strip()
        if not text:
            return
        folder, project_name = self._pick_or_create_project_folder()
        if not folder:
            return
        default_title = text.splitlines()[0].strip()[:60] or "Saved chat message"
        title, ok = QtWidgets.QInputDialog.getText(self, "Save to project", "Note name:", text=default_title)
        if not ok:
            return
        try:
            path = self._write_memory_note(str(title or default_title), text, folder, {"project": project_name, "from_message_id": str(message_id or "")})
            self._set_status(f"Saved project note: {os.path.basename(path)}", "ready")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Save to project failed", str(e))

    def _memory_search_roots_for_query(self, query: str, mode: str = "auto") -> List[Tuple[str, str]]:
        low = str(query or "").lower()
        mode = str(mode or "auto").lower().strip()
        roots: List[Tuple[str, str]] = []

        if mode in {"forced", "saved", "saved_memories", "memories"}:
            # Saved memories means the full user memory area, not the current chat history.
            roots.extend([
                ("saved_notes", _memory_saved_notes_dir(self.fv_root)),
                ("user_files", _memory_user_files_dir(self.fv_root)),
                ("project", _memory_project_dir(self.fv_root)),
            ])
        else:
            if any(x in low for x in ("user files", "user_files", "my files", "look in files", "look in user files")):
                roots.append(("user_files", _memory_user_files_dir(self.fv_root)))
            if any(x in low for x in ("project", "projects")):
                roots.append(("project", _memory_project_dir(self.fv_root)))
            if any(x in low for x in ("memory", "memories", "remember", "saved", "notes", "what did i save", "what did you save")):
                roots.append(("saved_notes", _memory_saved_notes_dir(self.fv_root)))
                roots.append(("project", _memory_project_dir(self.fv_root)))
            if any(x in low for x in ("knowledge", "knowledge base", "framevision", "feature guide", "how does", "what does", "planner", "queue", "settings")):
                roots.append(("knowledge", _knowledge_root(self.fv_root)))
        if not roots:
            return []
        # De-duplicate while preserving order.
        seen = set()
        deduped = []
        for label, root in roots:
            key = os.path.abspath(root).lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append((label, root))
        return deduped

    def _search_memory_files(self, query: str, roots: List[Tuple[str, str]], max_hits: int = 6, include_recent_if_no_hits: bool = False) -> List[Dict[str, Any]]:
        q_tokens = _keyword_tokens(query)
        hits: List[Dict[str, Any]] = []
        recent: List[Dict[str, Any]] = []
        for label, root in roots:
            for path in _iter_memory_files(root)[:200]:
                text = _read_memory_text_file(path, max_chars=90000)
                if not text or text.startswith("[Could not read"):
                    continue
                rel = os.path.relpath(path, self.fv_root)
                file_mtime = os.path.getmtime(path) if os.path.exists(path) else 0
                recent.append({"score": 1, "label": label, "path": path, "rel": rel, "text": text[:2200], "mtime": file_mtime})
                if not q_tokens:
                    continue
                for chunk in _chunk_memory_text(text, chunk_chars=2200)[:60]:
                    low = chunk.lower()
                    score = 0
                    for tok in q_tokens:
                        if tok in low:
                            score += 2
                        if tok in os.path.basename(path).lower():
                            score += 4
                    # Identity/name questions should still find notes that mention name/persona.
                    if any(t in {"name", "called", "identity", "persona"} for t in q_tokens):
                        if re.search(r"\b(name|called|identity|persona)\b", low):
                            score += 3
                    if score <= 0:
                        continue
                    hits.append({"score": score, "label": label, "path": path, "rel": rel, "text": chunk, "mtime": file_mtime})
        hits.sort(key=lambda h: (int(h.get("score", 0)), float(h.get("mtime", 0))), reverse=True)
        if hits:
            return hits[:max_hits]
        if include_recent_if_no_hits:
            recent.sort(key=lambda h: float(h.get("mtime", 0)), reverse=True)
            return recent[:max_hits]
        return []

    def _memory_context_for_query(self, query: str, mode: str = "auto") -> Tuple[str, List[str]]:
        if not self._memory_enabled():
            return "", []
        mode = str(mode or "auto").lower().strip()
        if mode in {"none", "chat", "this_chat"}:
            return "", []

        # Reload here, not only at application startup. Users may add/edit files while the chat is open.
        try:
            self._reload_startup_llm_memory()
        except Exception:
            pass

        roots = self._memory_search_roots_for_query(query, mode=mode)
        include_recent = mode in {"forced", "saved", "saved_memories", "memories"}
        hits = self._search_memory_files(query, roots, max_hits=8 if include_recent else 6, include_recent_if_no_hits=include_recent) if roots else []
        sources: List[str] = []
        blocks: List[str] = []
        if self._startup_llm_memory_context and mode in {"auto", "forced", "saved", "saved_memories", "memories"}:
            blocks.append(
                "Persistent local LLM memory / persona instructions loaded from assets/memories/llm_memory. "
                "Treat these as user-provided persistent instructions unless the current user message clearly overrides them:\n"
                + self._startup_llm_memory_context[:30000]
            )
            sources.append(os.path.relpath(_memory_llm_memory_dir(self.fv_root), self.fv_root))
        for h in hits:
            rel = str(h.get("rel") or "")
            label = str(h.get("label") or "memory")
            txt = str(h.get("text") or "").strip()
            if not txt:
                continue
            sources.append(rel)
            blocks.append(f"Source: {rel} ({label})\n{txt}")
        if not blocks:
            return "", []
        # Keep duplicates out of the visible source list.
        uniq_sources = []
        seen = set()
        for src in sources:
            if src and src not in seen:
                seen.add(src)
                uniq_sources.append(src)
        context = (
            "Use the following local FrameVision Knowledge & Memory context when it is relevant. "
            "If the user asks about saved memories, answer from these sources instead of explaining generic AI memory. "
            "If a source states your name/persona or expertise, follow it as local user-provided context. "
            "Do not pretend it contains information that is not present. Mention sources briefly when helpful.\n\n" +
            "\n\n---\n\n".join(blocks)
        )
        return context[:52000], uniq_sources[:12]

    def _looks_like_memory_choice_request(self, text: str) -> bool:
        raw = str(text or "").strip().lower()
        if not raw:
            return False
        save_mem, _, _ = self._extract_memory_save_text(text)
        if save_mem:
            return False
        # These are ambiguous because the user may mean either the current chat history or saved memory files.
        patterns = [
            r"\bcheck\s+(my\s+)?memories\b",
            r"\bwhat\s+(do|did)\s+(you|i)\s+(remember|save|saved)\b",
            r"\bwhat\s+is\s+in\s+(my\s+)?memories\b",
            r"\bshow\s+(my\s+)?memories\b",
            r"\bsearch\s+(my\s+)?memories\b",
            r"\blook\s+in\s+(my\s+)?memories\b",
            r"\brecall\s+(my\s+)?memories\b",
            r"\bsaved\s+memories\b",
            r"\bmemory\s+folder\b",
        ]
        return any(re.search(p, raw, flags=re.IGNORECASE) for p in patterns)

    def _continue_memory_choice_generation(self, session_id: str, mode: str) -> None:
        mode = str(mode or "auto").lower().strip()
        if mode not in {"auto", "none", "forced"}:
            mode = "auto"
        self._memory_context_mode_for_next_generation = mode
        try:
            self._validate_runner_and_model()
        except Exception as e:
            self._memory_context_mode_for_next_generation = "auto"
            self._set_status(str(e), "error")
            QtWidgets.QMessageBox.warning(self, "Cannot send", str(e))
            return
        s = self._current_session()
        if not s or s.id != session_id:
            self._select_session(session_id)
            s = self._current_session()
        if not s:
            self._memory_context_mode_for_next_generation = "auto"
            return
        tk, tv = self.current_template_choice()
        self._prepare_auto_retry_templates(self.current_model_path(), tk, tv)
        if not self._same_loaded_config() or not self.server_ready:
            self.pending_generate_session_id = s.id
            self._load_selected_model()
            return
        self._start_reply_request_for_current_session()

    def _ask_memory_source_choice(self, session_id: str, user_text: str) -> None:
        self.chat_view.add_memory_choice(
            "Memory question detected",
            "Do you mean this current chat history, or the saved memory folders?",
            lambda sid=session_id: self._continue_memory_choice_generation(sid, "none"),
            lambda sid=session_id: self._continue_memory_choice_generation(sid, "forced"),
        )
        self._set_status("Choose memory source", "ready")
    # ---------- chat ----------
    def _build_api_messages(self, session: ChatSession) -> List[Dict]:
        messages: List[Dict] = []
        system_prompt = (session.system_prompt or "").strip()
        last_user_text = ""
        try:
            for _m in reversed(session.messages):
                if str(_m.get("role") or "") == "user":
                    last_user_text = str(_m.get("content", "") or "")
                    break
        except Exception:
            last_user_text = ""
        mode = str(getattr(self, "_memory_context_mode_for_next_generation", "auto") or "auto")
        memory_context, memory_sources = self._memory_context_for_query(last_user_text, mode=mode)
        self._last_memory_sources = memory_sources
        system_parts = []
        if system_prompt:
            system_parts.append(system_prompt)
        if memory_context:
            system_parts.append(memory_context)
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts).strip()})
        for m in session.messages:
            role = str(m.get("role", "assistant"))
            if role not in ("user", "assistant"):
                continue
            content_text = str(m.get("content", "") or "")
            attachments = list(m.get("attachments", []) or [])
            if role != "user" or not attachments:
                messages.append({"role": role, "content": content_text})
                continue

            parts: List[Dict] = []
            text_chunks: List[str] = []
            if content_text.strip():
                text_chunks.append(content_text.strip())
            for att in attachments:
                kind = str(att.get("kind", "file") or "file")
                name = str(att.get("name", "attachment") or "attachment")
                path = str(att.get("path", "") or "")
                if kind == "image":
                    try:
                        data_url = _image_attachment_to_data_url(att)
                        if data_url:
                            parts.append({"type": "image_url", "image_url": {"url": data_url}})
                            text_chunks.append(f"[Attached image: {name}]")
                        else:
                            text_chunks.append(f"[Image attachment missing on disk: {name}]")
                    except Exception as e:
                        text_chunks.append(f"[Image attachment could not be embedded: {name} ({e})]")
                elif kind == "text":
                    if path and os.path.isfile(path):
                        try:
                            file_text = _read_text_attachment(path)
                            text_chunks.append(f"Attached file: {name}\n```\n{file_text}\n```")
                        except Exception as e:
                            text_chunks.append(f"[Text attachment could not be read: {name} ({e})]")
                    else:
                        text_chunks.append(f"[Text attachment missing on disk: {name}]")
                else:
                    text_chunks.append(f"[Attached file: {name}]")
            text_payload = "\n\n".join([x for x in text_chunks if x.strip()]).strip()
            if text_payload:
                parts.insert(0, {"type": "text", "text": text_payload})
            messages.append({"role": role, "content": parts if parts else content_text})
        return messages
    def _prepare_auto_retry_templates(self, model_path: str, template_kind: str, template_value: str):
        self._auto_retry_templates = []
        self._auto_retry_original_selection = (template_kind, template_value)
        if template_kind not in ("auto", "smart"):
            return
        for item in _candidate_templates_from_model_path(model_path):
            if item != self._effective_template_choice(model_path, template_kind, template_value):
                self._auto_retry_templates.append(item)

    def _set_template_silently(self, kind: str, value: str):
        self._loading_session_state = True
        try:
            self.settings_dialog.select_template(kind, value)
            self._apply_settings_to_current_session()
            self._update_header()
            self._queue_save()
        finally:
            self._loading_session_state = False

    def _try_auto_retry_after_bad_reply(self, reply: str) -> bool:
        if not self._auto_retry_templates:
            return False
        if not _looks_like_bad_assistant_reply(self._last_user_text_sent, reply):
            self._auto_retry_templates = []
            return False

        next_kind, next_value = self._auto_retry_templates.pop(0)
        self._set_template_silently(next_kind, next_value)
        self.pending_retry_generation = True
        self._set_status(f"Retrying with template {next_value}…", "loading")
        self._load_selected_model()
        return True

    def _append_user_message_to_current_session(self, text: str, attachments: Optional[List[Dict]] = None) -> Optional[ChatSession]:
        s = self._current_session()
        if not s:
            self._new_chat()
            s = self._current_session()
            if not s:
                return None

        attachments = list(attachments or [])
        title_seed = text[:48].strip() if text.strip() else (attachments[0].get("name", "New Chat") if attachments else "New Chat")
        if s.title == "New Chat":
            s.title = title_seed or "New Chat"

        now = datetime.now().isoformat(timespec="seconds")
        mid = str(uuid.uuid4())
        s.messages.append({
            "id": mid,
            "role": "user",
            "content": text,
            "attachments": attachments,
            "timestamp": now,
            "loading": False,
        })
        s.updated_at = now
        self.chat_view.add_message("user", text, attachments=attachments, message_id=mid, loading=False)
        self.chat_view.scroll_to_bottom(force=True)
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 25)
        self.ed_prompt.clear()
        self.pending_attachments = []
        self._refresh_attachment_list()
        self._refresh_chat_list()
        self._update_header()
        self._queue_save()
        if attachments and text.strip():
            self._last_user_text_sent = text + " " + " ".join([att.get("name", "attachment") for att in attachments])
        elif attachments:
            self._last_user_text_sent = " ".join([att.get("name", "attachment") for att in attachments])
        else:
            self._last_user_text_sent = text
        return s

    def _start_reply_request_for_session_id(self, session_id: str):
        if not session_id:
            return
        if session_id != self.current_session_id:
            self._select_session(session_id)
        self._start_reply_request_for_current_session()

    def _start_reply_request_for_current_session(self):
        s = self._current_session()
        if not s or not self.server_ready:
            return

        self.btn_send.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._set_status("Generating…", "loading")

        pending_update = self._pending_generation_update if isinstance(self._pending_generation_update, dict) else None
        if pending_update and str(pending_update.get("session_id") or "") == s.id:
            api_messages = self._build_api_messages_until(s, str(pending_update.get("assistant_message_id") or ""))
        else:
            api_messages = self._build_api_messages(s)
        self.chat_thread = ChatCompletionThread(
            self.server_url,
            api_messages,
            self.settings_dialog.sp_max_tokens.value(),
            self.settings_dialog.sp_temp.value(),
            self.settings_dialog.sp_top_p.value(),
            self,
        )
        self.chat_thread.succeeded.connect(self._on_chat_succeeded)
        self.chat_thread.failed.connect(self._on_chat_failed)
        self.chat_thread.finished.connect(self._on_chat_finished)
        self.chat_thread.start()

    def _save_generated_images(self, images: List[Dict], prompt_text: str) -> List[Dict]:
        saved: List[Dict] = []
        out_dir = _llama_image_output_dir(self.fv_root)
        os.makedirs(out_dir, exist_ok=True)
        prompt_stub = _filename_words_from_prompt(prompt_text)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for idx, img in enumerate(images or [], start=1):
            b64_data = str(img.get("data_b64", "") or "").strip()
            if not b64_data:
                continue
            mime = str(img.get("mime", "") or "image/png")
            ext = _mime_to_extension(mime, ".png")
            suffix = f"_{idx}" if len(images or []) > 1 else ""
            filename = f"llama_{prompt_stub}_{stamp}{suffix}{ext}"
            out_path = os.path.join(out_dir, filename)
            raw = base64.b64decode(b64_data)
            with open(out_path, "wb") as f:
                f.write(raw)
            saved.append(_make_attachment_entry(out_path))
        return saved


    def _append_framevision_assistant_reply(self, text: str, thinking: str = "") -> None:
        """Append a deterministic FrameVision Assistant reply to the current chat."""
        s = self._current_session()
        if not s:
            return
        reply = str(text or "").strip()
        if not reply:
            return
        now = datetime.now().isoformat(timespec="seconds")
        mid = str(uuid.uuid4())
        s.messages.append({
            "id": mid,
            "role": "assistant",
            "content": reply,
            "thinking": thinking or "",
            "attachments": [],
            "timestamp": now,
            "loading": False,
        })
        s.updated_at = now
        self.chat_view.add_message("assistant", reply, thinking or "", attachments=[], message_id=mid, loading=False)
        self.chat_view.scroll_to_bottom(force=True)
        self._refresh_chat_list()
        self._update_header()
        self._queue_save()


    # -----------------------------
    # Ace-Step 1.5 music assistant flow
    # -----------------------------
    def _assistant_registry(self) -> Dict[str, Any]:
        path = Path(self.fv_root) / "scripts" / "fv_assistant_image_models.json"
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _ace15_config(self) -> Dict[str, Any]:
        reg = self._assistant_registry()
        tools = reg.get("tools") if isinstance(reg.get("tools"), dict) else {}
        cfg = tools.get("ace_step_15") if isinstance(tools.get("ace_step_15"), dict) else {}
        defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
        out = {
            "label": str(cfg.get("label") or "Ace-Step 1.5"),
            "env_win": str(cfg.get("env_win") or "environments/.ace_15/Scripts/python.exe"),
            "env_posix": str(cfg.get("env_posix") or "environments/.ace_15/bin/python"),
            "project_root": str(cfg.get("project_root") or "models/ace_step_15/repo/ACE-Step-1.5"),
            "cli": str(cfg.get("cli") or "models/ace_step_15/repo/ACE-Step-1.5/cli.py"),
            "preset_manager": str(cfg.get("preset_manager") or "presets/setsave/ace15presets/presetmanager.json"),
            "output_dir": str(cfg.get("output_dir") or "output/audio/ace15"),
            "audio_format": str(defaults.get("audio_format") or "mp3"),
            "task_type": str(defaults.get("task_type") or "text2music"),
            "batch_size": int(defaults.get("batch_size", 1) or 1),
            "log_level": str(defaults.get("log_level") or "INFO"),
            "default_backend": str(defaults.get("backend") or "vllm"),
            "default_shift": float(defaults.get("shift", 3.0) or 3.0),
            "sft_steps": int(defaults.get("sft_steps", 60) or 60),
            "base_steps": int(defaults.get("base_steps", 60) or 60),
            "turbo_steps": int(defaults.get("turbo_steps", 20) or 20),
            "preferred_models": list(defaults.get("preferred_models") or ["acestep-v15-sft", "acestep-v15-base", "acestep-v15-turbo"]),
            "lm_model": str(defaults.get("lm_model") or "acestep-5Hz-lm-1.7B"),
            "hide_console": bool(defaults.get("hide_console", True)),
        }
        return out

    def _ace15_env_python(self) -> Path:
        cfg = self._ace15_config()
        return self._root_join(str(cfg.get("env_win") if os.name == "nt" else cfg.get("env_posix")))

    def _ace15_project_root(self) -> Path:
        return self._root_join(str(self._ace15_config().get("project_root") or "models/ace_step_15/repo/ACE-Step-1.5"))

    def _ace15_cli_path(self) -> Path:
        return self._root_join(str(self._ace15_config().get("cli") or "models/ace_step_15/repo/ACE-Step-1.5/cli.py"))

    def _ace15_preset_manager_path(self) -> Path:
        return self._root_join(str(self._ace15_config().get("preset_manager") or "presets/setsave/ace15presets/presetmanager.json"))

    def _ace15_output_dir(self) -> Path:
        return self._root_join(str(self._ace15_config().get("output_dir") or "output/audio/ace15"))

    def _ace15_load_presets(self) -> Dict[str, Any]:
        path = self._ace15_preset_manager_path()
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8") or "{}")
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {"version": 1, "genres": {}}

    def _ace15_genres_dict(self) -> Dict[str, Any]:
        data = self._ace15_load_presets()
        genres = data.get("genres") if isinstance(data.get("genres"), dict) else {}
        return genres if isinstance(genres, dict) else {}

    def _ace15_sanitize_filename_part(self, value: str) -> str:
        s = re.sub(r"[^A-Za-z0-9._ -]+", "_", str(value or "")).strip(" ._-+")
        s = re.sub(r"\s+", "_", s)
        return s[:80] or ""

    def _ace15_normalize_text(self, value: str) -> str:
        s = str(value or "").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def _ace15_music_intent(self, text: str) -> bool:
        low = self._ace15_normalize_text(text)
        if not low:
            return False
        if re.search(r"\b(image|picture|photo|render|drawing|upscale|edit image)\b", low):
            return False
        action = bool(re.search(r"\b(create|generate|make|compose|queue|start|write|produce)\b", low))
        music = bool(re.search(r"\b(music|song|track|beat|instrumental|rap|hip hop|house|techno|rock|reggae|pop|jazz|metal|lyrics)\b", low))
        if action and music:
            return True
        return bool(re.search(r"\b(i want|new)\s+(some\s+)?(music|song|track|beat)\b", low))

    def _ace15_yes_no(self, text: str) -> str:
        low = self._ace15_normalize_text(text)
        if low in {"yes", "y", "yeah", "yep", "sure", "ok", "okay"} or low.startswith("yes "):
            return "yes"
        if low in {"no", "n", "nope", "nah"} or low.startswith("no "):
            return "no"
        return ""

    def _ace15_parse_lyrics_mode(self, text: str) -> str:
        low = self._ace15_normalize_text(text)
        if re.search(r"\b(instrumental|no lyrics|without lyrics|no vocal|no vocals)\b", low):
            return "instrumental"
        if re.search(r"\b(lyrics|with lyrics|vocals|vocal|sing|rap)\b", low):
            return "lyrics"
        return ""

    def _ace15_parse_duration_seconds(self, text: str) -> int:
        raw = str(text or "").strip().lower().replace(",", ".")
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:minutes|minute|min|mins|m)\b", raw)
        if m:
            return int(max(5, min(600, round(float(m.group(1)) * 60))))
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds|second|sec|secs|s)\b", raw)
        if m:
            return int(max(5, min(600, round(float(m.group(1))))))
        m = re.search(r"\b(\d{1,2})\s*[:.]\s*(\d{2})\b", raw)
        if m:
            return int(max(5, min(600, int(m.group(1)) * 60 + int(m.group(2)))))
        m = re.search(r"\b(\d{2,3})\b", raw)
        if m:
            n = int(m.group(1))
            if 5 <= n <= 600:
                return n
        return 0

    def _ace15_parse_bpm(self, text: str) -> int:
        raw = str(text or "").strip().lower().replace(",", ".")
        if not raw:
            return 0
        if raw in {"auto", "automatic", "default", "skip", "no", "none", "not sure"}:
            return -1
        m = re.search(r"(\d{2,3})(?:\.\d+)?\s*(?:bpm|beats\s+per\s+minute)?\b", raw)
        if not m:
            return 0
        bpm = int(m.group(1))
        if 40 <= bpm <= 220:
            return bpm
        return 0

    def _ace15_bpm_help_text(self) -> str:
        return (
            "What BPM should it use? You can also say `auto` if you are not sure.\n\n"
            "Average examples:\n"
            "- Rap / hip hop: about 80-100 BPM\n"
            "- House: about 120-128 BPM\n"
            "- Rock: about 110-150 BPM"
        )

    def _ace15_find_genre(self, text: str) -> str:
        genres = self._ace15_genres_dict()
        q = self._ace15_normalize_text(text)
        if not q:
            return ""
        best = (0, "")
        q_words = set(q.split())
        for g in genres.keys():
            gn = self._ace15_normalize_text(g)
            if not gn:
                continue
            score = 0
            gn_words = set(gn.split())
            if gn == q:
                score += 100
            elif q_words and q_words.issubset(gn_words):
                score += 45
            elif gn_words and gn_words.issubset(q_words):
                score += 40
            elif len(q) >= 5 and (gn in q or q in gn):
                score += 25
            score += len(q_words.intersection(gn_words)) * 12
            if score > best[0]:
                best = (score, g)
        return best[1] if best[0] >= 40 else ""

    def _ace15_subgenres_for_genre(self, genre: str) -> Dict[str, Any]:
        gd = self._ace15_genres_dict().get(str(genre or "")) or {}
        subs = gd.get("subgenres") if isinstance(gd, dict) else {}
        return subs if isinstance(subs, dict) else {}

    def _ace15_find_subgenre(self, text: str, genre: str = "") -> Tuple[str, str]:
        matches = self._ace15_music_match_candidates(text, genre=genre, include_genres=False, include_subgenres=True, limit=1)
        if not matches:
            return "", ""
        m = matches[0]
        return (str(m.get("genre") or ""), str(m.get("subgenre") or "")) if int(m.get("score") or 0) >= 40 else ("", "")

    def _ace15_music_match_candidates(self, text: str, genre: str = "", include_genres: bool = True, include_subgenres: bool = True, limit: int = 8) -> List[Dict[str, Any]]:
        genres = self._ace15_genres_dict()
        q = self._ace15_normalize_text(text)
        if not q:
            return []
        q_words = set(q.split())

        def _score_name(name: str, caption: str = "") -> int:
            nn = self._ace15_normalize_text(name)
            if not nn:
                return 0
            n_words = set(nn.split())
            score = 0
            if nn == q:
                score += 120
            elif q_words and q_words.issubset(n_words):
                score += 65
            elif n_words and n_words.issubset(q_words):
                score += 55
            elif len(q) >= 5 and (nn in q or q in nn):
                score += 45
            score += len(q_words.intersection(n_words)) * 18
            try:
                ratio = difflib.SequenceMatcher(None, q, nn).ratio()
                if ratio >= 0.82:
                    score += int(ratio * 35)
                elif ratio >= 0.72 and len(q) >= 4:
                    score += int(ratio * 18)
            except Exception:
                pass
            cap = self._ace15_normalize_text(caption)
            if q_words and cap:
                score += min(18, len(q_words.intersection(set(cap.split()))) * 3)
            return int(score)

        out: List[Dict[str, Any]] = []
        if include_genres:
            for g in genres.keys():
                score = _score_name(str(g or ""))
                if score >= 35:
                    out.append({"kind": "genre", "genre": str(g), "subgenre": "", "score": score, "label": str(g)})

        if include_subgenres:
            genre_items = []
            if genre:
                genre_items = [(genre, {"subgenres": self._ace15_subgenres_for_genre(genre)})]
            else:
                genre_items = list(genres.items())
            for g, gd in genre_items:
                subs = (gd or {}).get("subgenres") if isinstance(gd, dict) else {}
                if not isinstance(subs, dict):
                    continue
                for sub, payload in subs.items():
                    cap = str((payload or {}).get("caption") or "") if isinstance(payload, dict) else ""
                    score = _score_name(str(sub or ""), cap)
                    if score >= 35:
                        out.append({"kind": "subgenre", "genre": str(g), "subgenre": str(sub), "score": score, "label": f"{g} / {sub}"})

        # Deduplicate by visible label while preserving best score.
        best_by_label: Dict[str, Dict[str, Any]] = {}
        for item in out:
            label = str(item.get("label") or "")
            if not label:
                continue
            old = best_by_label.get(label)
            if old is None or int(item.get("score") or 0) > int(old.get("score") or 0):
                best_by_label[label] = item
        out = list(best_by_label.values())
        out.sort(key=lambda m: (-int(m.get("score") or 0), str(m.get("label") or "").lower()))
        return out[:max(1, int(limit or 8))]

    def _ace15_match_choices_text(self, matches: List[Dict[str, Any]]) -> str:
        lines = []
        for i, item in enumerate(matches or [], 1):
            label = str(item.get("label") or item.get("genre") or item.get("subgenre") or "").strip()
            if label:
                lines.append(f"{i}. `{label}`")
        return "\n".join(lines)

    def _ace15_select_match_from_reply(self, text: str, matches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        raw = str(text or "").strip()
        low = self._ace15_normalize_text(raw)
        if not matches:
            return None
        if low in {"yes", "y", "correct", "that", "that one", "first", "1"}:
            return matches[0]
        m = re.search(r"\b(\d{1,2})\b", raw)
        if m:
            try:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(matches):
                    return matches[idx]
            except Exception:
                pass
        best = None
        best_score = 0
        for item in matches:
            label = self._ace15_normalize_text(str(item.get("label") or ""))
            genre = self._ace15_normalize_text(str(item.get("genre") or ""))
            sub = self._ace15_normalize_text(str(item.get("subgenre") or ""))
            score = 0
            if low and low in {label, genre, sub}:
                score = 100
            elif low and any(low == part for part in label.replace("/", " ").split()):
                score = 80
            elif len(low) >= 4 and low and (low in label or label in low):
                score = 55
            if score > best_score:
                best_score = score
                best = item
        return best if best_score >= 55 else None

    def _ace15_apply_music_match(self, state: Dict[str, Any], match: Dict[str, Any]) -> str:
        kind = str(match.get("kind") or "")
        genre = str(match.get("genre") or "").strip()
        sub = str(match.get("subgenre") or "").strip()
        if kind == "subgenre" and genre and sub:
            state["genre"] = genre
            state["subgenre"] = sub
            state["preset"] = self._ace15_preset_payload(genre, sub)
            state["stage"] = "duration"
            self._set_pending_ace15_music_state(state)
            return f"Okay, `{genre} / {sub}`. How long should it be? Example: 3 minutes."
        if genre:
            state["genre"] = genre
            subs_txt = self._ace15_list_subgenres(genre)
            if subs_txt == "no subgenres found":
                state["subgenre"] = "Custom"
                state["stage"] = "caption"
                self._set_pending_ace15_music_state(state)
                return f"I found `{genre}`, but no subgenre presets. Describe the music style you want, for example: instruments, tempo/BPM, mood, vocal style, and anything to avoid."
            state["stage"] = "subgenre"
            self._set_pending_ace15_music_state(state)
            return f"I found `{genre}`. Subgenres: {subs_txt}.\n\nWhich one? If none of these fit, describe the style instead."
        state["genre"] = str(state.get("genre") or "Custom")
        state["subgenre"] = "Custom"
        state["stage"] = "caption"
        self._set_pending_ace15_music_state(state)
        return "Describe the music style you want, for example: instruments, tempo/BPM, mood, vocal style, and anything to avoid."

    def _ace15_list_genres(self) -> str:
        names = sorted(self._ace15_genres_dict().keys(), key=lambda s: s.lower())
        return ", ".join(names) if names else "no genres found"

    def _ace15_list_subgenres(self, genre: str) -> str:
        subs = sorted(self._ace15_subgenres_for_genre(genre).keys(), key=lambda s: s.lower())
        return ", ".join(subs) if subs else "no subgenres found"

    def _ace15_preset_payload(self, genre: str, subgenre: str) -> Dict[str, Any]:
        payload = self._ace15_subgenres_for_genre(genre).get(subgenre) or {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _ace15_default_music_preset(self, caption: str = "") -> Dict[str, Any]:
        cfg = self._ace15_config()
        return {
            "caption": str(caption or "").strip(),
            "backend": str(cfg.get("default_backend") or "vllm"),
            "shift": float(cfg.get("default_shift") or 3.0),
            "thinking": True,
            "parallel_thinking": True,
            "enable_lm": True,
            "lm_enhance_prompt": True,
            "use_cot_caption": True,
            "use_cot_language": True,
            "use_cot_metas": True,
            "use_cot_lyrics": False,
            "lm_model_path": str(cfg.get("lm_model") or "acestep-5Hz-lm-1.7B"),
            "infer_method": "ode",
        }

    def _ace15_model_dir_has_files(self, model_name: str) -> bool:
        try:
            p = self._ace15_project_root() / "checkpoints" / str(model_name or "")
            return p.exists() and p.is_dir() and any(p.iterdir())
        except Exception:
            return False

    def _ace15_choose_main_model(self) -> Tuple[str, int]:
        cfg = self._ace15_config()
        prefs = [str(x) for x in (cfg.get("preferred_models") or []) if str(x or "").strip()]
        if not prefs:
            prefs = ["acestep-v15-sft", "acestep-v15-base", "acestep-v15-turbo"]
        for name in prefs:
            if self._ace15_model_dir_has_files(name):
                low = name.lower()
                if "turbo" in low:
                    return name, int(cfg.get("turbo_steps") or 20)
                if "base" in low:
                    return name, int(cfg.get("base_steps") or 60)
                return name, int(cfg.get("sft_steps") or 60)
        first = prefs[0]
        return first, int(cfg.get("sft_steps") or 60)

    def _ace15_toml_escape_str(self, value: str) -> str:
        s = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return s.replace("\n", "\\n")

    def _ace15_toml_dumps_flat(self, data: Dict[str, Any]) -> str:
        lines = []
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, bool):
                val = "true" if v else "false"
            elif isinstance(v, int):
                val = str(v)
            elif isinstance(v, float):
                val = ("%.6f" % v).rstrip("0").rstrip(".") or "0"
            else:
                val = f'"{self._ace15_toml_escape_str(str(v))}"'
            lines.append(f"{k} = {val}")
        return "\n".join(lines) + "\n"

    def _ace15_preset_get(self, preset: Dict[str, Any], *names, default=None):
        for name in names:
            if name in preset and preset.get(name) not in (None, ""):
                return preset.get(name)
        return default

    def _ace15_write_config(self, state: Dict[str, Any]) -> Path:
        cfg = self._ace15_config()
        out_dir = self._ace15_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        title_hint = self._ace15_sanitize_filename_part(str(state.get("title") or ""))
        suffix = f"_{title_hint}" if title_hint else ""
        cfg_path = out_dir / f"ace_step_chat_{ts}{suffix}.toml"
        preset = dict(state.get("preset") or {})
        main_model, steps = self._ace15_choose_main_model()
        lm_model = str(self._ace15_preset_get(preset, "lm_model_path", "lm_model", default=cfg.get("lm_model") or "acestep-5Hz-lm-1.7B"))
        backend = str(self._ace15_preset_get(preset, "backend", default=cfg.get("default_backend") or "vllm"))
        infer_method = str(self._ace15_preset_get(preset, "infer_method", default="ode"))
        if infer_method.lower() == "auto":
            infer_method = ""
        lyrics_mode = str(state.get("lyrics_mode") or "instrumental")
        lyrics_text = str(state.get("lyrics") or "").strip()
        instrumental = lyrics_mode == "instrumental"
        caption = str(self._ace15_preset_get(preset, "caption", default=state.get("caption") or "")).strip()
        neg = str(self._ace15_preset_get(preset, "lm_negative_prompt", "negatives", default="")).strip()
        config = {
            "project_root": str(self._ace15_project_root().resolve()),
            "backend": backend,
            "log_level": str(cfg.get("log_level") or "INFO"),
            "device": "auto",
            "shift": float(self._ace15_preset_get(preset, "shift", default=cfg.get("default_shift") or 3.0) or 3.0),
            "save_dir": str(out_dir.resolve()),
            "audio_format": str(cfg.get("audio_format") or "mp3"),
            "task_type": str(cfg.get("task_type") or "text2music"),
            "caption": caption,
            "duration": float(state.get("duration") or 60.0),
            "batch_size": int(cfg.get("batch_size") or 1),
            "num_outputs": int(cfg.get("batch_size") or 1),
            "output_count": int(cfg.get("batch_size") or 1),
            "num_samples": int(cfg.get("batch_size") or 1),
            "seed": int(state.get("seed") or random.randint(1, 2147483647)),
            "inference_steps": int(steps),
            "main_model_path": main_model,
            "config_path": main_model,
            "main_model": main_model,
            "dit_model": main_model,
            "lm_model_path": lm_model,
            "lm_model": lm_model,
            "instrumental": bool(instrumental),
            "lyrics": "[Instrumental]" if instrumental else (lyrics_text or None),
            "use_cot_lyrics": False,
        }
        if infer_method:
            config["infer_method"] = infer_method
        for key, aliases in {
            "bpm": ("bpm",),
            "timesignature": ("timesignature", "time_sig"),
            "keyscale": ("keyscale", "key_scale"),
            "vocal_language": ("vocal_language",),
            "guidance_scale": ("guidance_scale", "guidance"),
            "thinking": ("thinking",),
            "parallel_thinking": ("parallel_thinking",),
            "enable_lm": ("enable_lm",),
            "use_cot_caption": ("use_cot_caption", "lm_enhance_prompt"),
            "use_cot_language": ("use_cot_language",),
            "use_cot_metas": ("use_cot_metas",),
            "lm_temperature": ("lm_temperature",),
            "lm_top_p": ("lm_top_p",),
            "lm_top_k": ("lm_top_k",),
            "offload_to_cpu": ("offload_to_cpu",),
            "offload_dit_to_cpu": ("offload_dit_to_cpu",),
            "use_flash_attention": ("use_flash_attention",),
        }.items():
            val = self._ace15_preset_get(preset, *aliases, default=None)
            if val not in (None, "", "auto"):
                config[key] = val
        if "bpm" not in config:
            try:
                state_bpm = int(state.get("bpm") or 0)
            except Exception:
                state_bpm = 0
            if state_bpm > 0:
                config["bpm"] = state_bpm
        if neg:
            config["lm_negative_prompt"] = neg
        clean = {k: v for k, v in config.items() if v is not None}
        cfg_path.write_text(self._ace15_toml_dumps_flat(clean), encoding="utf-8")
        return cfg_path

    def _assistant_chat_only_results_enabled(self) -> bool:
        try:
            chk = getattr(self.settings_dialog, "chk_results_chat_only", None)
            if chk is not None and hasattr(chk, "isChecked"):
                return bool(chk.isChecked())
        except Exception:
            pass
        try:
            data = _load_json(self.settings_path, {})
            if isinstance(data, dict):
                return bool(data.get("assistant_results_chat_only", True))
        except Exception:
            pass
        return True

    def _assistant_chat_result_flags(self) -> Dict[str, Any]:
        return {
            "assistant_origin": "llama_chat",
            "assistant_chat_only": bool(self._assistant_chat_only_results_enabled()),
        }

    def _queue_ace15_music(self, state: Dict[str, Any]) -> Tuple[bool, str, str]:
        env_py = self._ace15_env_python()
        cli_py = self._ace15_cli_path()
        project_root = self._ace15_project_root()
        if not env_py.exists():
            return False, f"Ace-Step environment python was not found: {env_py}", ""
        if not cli_py.exists():
            return False, f"Ace-Step CLI was not found: {cli_py}", ""
        if not project_root.exists():
            return False, f"Ace-Step project root was not found: {project_root}", ""
        cfg_path = self._ace15_write_config(state)
        out_dir = self._ace15_output_dir()
        title = str(state.get("title") or "").strip()
        sub = str(state.get("subgenre") or "Custom").strip() or "Custom"
        seed = int(state.get("seed") or 0)
        label_title = title or sub
        label = f"Ace-Step 1.5: {label_title}" + (f" (seed {seed})" if seed else "")
        args = {
            "label": label,
            "env_python": str(env_py.resolve()),
            "cli_py": str(cli_py.resolve()),
            "project_root": str(project_root.resolve()),
            "cfg_path": str(cfg_path.resolve()),
            "hide_console": bool(self._ace15_config().get("hide_console", True)),
        }
        try:
            args.update(self._assistant_chat_result_flags())
        except Exception:
            pass
        try:
            from helpers.queue_adapter import enqueue_tool_job  # type: ignore
            enqueue_tool_job(job_type="ace_step_15", input_path="", out_dir=str(out_dir.resolve()), args=args, priority=620)
            return True, str(cfg_path.resolve()), str(out_dir.resolve())
        except Exception:
            job = {
                "id": uuid.uuid4().hex,
                "type": "tool",
                "job_type": "ace_step_15",
                "title": label,
                "name": label,
                "category": "music",
                "engine": "ace_step_15",
                "input_path": "",
                "out_dir": str(out_dir.resolve()),
                "args": args,
                "priority": 620,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "pending",
            }
            ok = self._assistant_write_pending_job(job)
            return ok, str(cfg_path.resolve()) if ok else "Could not write Ace-Step job to jobs/pending.", str(out_dir.resolve())

    # ----------------------------- wizard undo trigger words -----------------------------
    def _wizard_is_cancel_command(self, text: str) -> bool:
        return self._ace15_normalize_text(text) in {"cancel", "cancel it", "stop", "nevermind", "never mind"}

    def _wizard_is_undo_command(self, text: str) -> bool:
        return self._ace15_normalize_text(text) in {"undo", "go back", "back", "one step back", "previous", "previous step"}

    def _wizard_clone_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return json.loads(json.dumps(state or {}, ensure_ascii=False))
        except Exception:
            return dict(state or {})

    def _wizard_state_without_undo(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in dict(state or {}).items() if k not in {"_undo_stack"}}

    def _set_pending_ace15_music_state(self, state: Optional[Dict[str, Any]]) -> None:
        if state is None:
            self._pending_ace15_music_request = None
            return
        new_state = dict(state or {})
        old_state = self._pending_ace15_music_request
        if isinstance(old_state, dict):
            old_clean = self._wizard_state_without_undo(old_state)
            new_clean = self._wizard_state_without_undo(new_state)
            if old_clean and old_clean != new_clean:
                stack = old_state.get("_undo_stack") if isinstance(old_state.get("_undo_stack"), list) else []
                if not stack or stack[-1] != old_clean:
                    stack = list(stack) + [old_clean]
                new_state["_undo_stack"] = stack[-20:]
        self._pending_ace15_music_request = new_state

    def _set_pending_seedvr2_state(self, state: Optional[Dict[str, Any]]) -> None:
        if state is None:
            self._pending_seedvr2_request = None
            return
        new_state = dict(state or {})
        old_state = self._pending_seedvr2_request
        if isinstance(old_state, dict):
            old_clean = self._wizard_state_without_undo(old_state)
            new_clean = self._wizard_state_without_undo(new_state)
            if old_clean and old_clean != new_clean:
                stack = old_state.get("_undo_stack") if isinstance(old_state.get("_undo_stack"), list) else []
                if not stack or stack[-1] != old_clean:
                    stack = list(stack) + [old_clean]
                new_state["_undo_stack"] = stack[-20:]
        self._pending_seedvr2_request = new_state

    def _ace15_question_for_state(self, state: Dict[str, Any]) -> str:
        stage = str((state or {}).get("stage") or "")
        if stage == "lyrics_mode":
            return "Do you want lyrics or instrumental music?"
        if stage == "lyrics_text":
            return "Do you have lyrics ready? Paste them in the chat. You can also say `undo` or `cancel`."
        if stage == "genre":
            return f"What genre?\n\nAvailable genres: {self._ace15_list_genres()}."
        if stage == "genre_confirm":
            matches = state.get("genre_candidates") if isinstance(state.get("genre_candidates"), list) else []
            choices = self._ace15_match_choices_text(matches)
            return "Please choose one of these by number/name, or describe a custom style:" + ("\n" + choices if choices else "")
        if stage == "subgenre":
            return f"Subgenres: {self._ace15_list_subgenres(str(state.get('genre') or ''))}.\n\nWhich one? If none of these fit, describe the style instead."
        if stage == "caption":
            return "Describe the music style you want, for example: instruments, tempo/BPM, mood, vocal style, and anything to avoid."
        if stage == "duration":
            return "How long should it be? Example: `3 minutes`, `180 seconds`, or `2:30`."
        if stage == "title":
            return "Optional title for the track? Type `default` to skip."
        return "Restored the previous Ace-Step music step."

    def _seedvr2_question_for_state(self, state: Dict[str, Any]) -> str:
        stage = str((state or {}).get("stage") or "")
        source_kind = str((state or {}).get("source_kind") or "image")
        if stage == "choose_model":
            families = list((state or {}).get("families") or [])
            opts = ", ".join(families) if families else "3B"
            return f"Please choose one of these SeedVR2 model groups: {opts}."
        if stage == "choose_resolution":
            allowed = [720, 1080, 1440] if source_kind == "video" else self._seedvr2_config().get("resolutions", [1080, 1440, 2160])
            return f"Choose the target resolution: {', '.join(str(x) + 'p' for x in allowed)}."
        return "Restored the previous SeedVR2 upscale step."

    def _undo_ace15_music_state(self, state: Dict[str, Any]) -> bool:
        stack = state.get("_undo_stack") if isinstance(state.get("_undo_stack"), list) else []
        if not stack:
            self._append_framevision_assistant_reply("Nothing to undo in this music wizard yet. Type `cancel` to stop it.")
            return True
        prev = dict(stack.pop() or {})
        prev["_undo_stack"] = stack
        self._pending_ace15_music_request = prev
        self._append_framevision_assistant_reply("Undone one step.\n\n" + self._ace15_question_for_state(prev))
        try:
            self._set_status("Waiting for Ace-Step music details", "ready")
        except Exception:
            pass
        return True

    def _undo_seedvr2_state(self, state: Dict[str, Any]) -> bool:
        stack = state.get("_undo_stack") if isinstance(state.get("_undo_stack"), list) else []
        if not stack:
            self._append_framevision_assistant_reply("Nothing to undo in this upscale wizard yet. Type `cancel` to stop it.")
            return True
        prev = dict(stack.pop() or {})
        prev["_undo_stack"] = stack
        self._pending_seedvr2_request = prev
        self._append_framevision_assistant_reply("Undone one step.\n\n" + self._seedvr2_question_for_state(prev))
        try:
            self._set_status("Waiting for SeedVR2 details", "ready")
        except Exception:
            pass
        return True

    def _start_ace15_music_flow(self, requested_text: str = ""):
        mode = self._ace15_parse_lyrics_mode(requested_text)
        self._set_pending_ace15_music_state({
            "stage": "lyrics_text" if mode == "lyrics" else "genre" if mode == "instrumental" else "lyrics_mode",
            "lyrics_mode": mode,
            "lyrics": "",
            "genre": "",
            "subgenre": "",
            "duration": 0,
            "title": "",
            "seed": random.randint(1, 2147483647),
        })
        if mode == "lyrics":
            self._append_framevision_assistant_reply("Do you have lyrics ready? Paste them in the chat. You can also say `undo` or `cancel`.")
        elif mode == "instrumental":
            self._append_framevision_assistant_reply(f"Instrumental selected. What genre?\n\nAvailable genres: {self._ace15_list_genres()}.")
        else:
            self._append_framevision_assistant_reply("Do you want lyrics or instrumental music? You can also say `undo` or `cancel`.")
        try:
            self._set_status("Waiting for Ace-Step music details", "ready")
        except Exception:
            pass

    def _handle_pending_ace15_music_message(self, text: str) -> bool:
        state = getattr(self, "_pending_ace15_music_request", None)
        if not isinstance(state, dict):
            return False
        raw = str(text or "").strip()
        low = self._ace15_normalize_text(raw)
        if not raw:
            return True
        if self._wizard_is_undo_command(raw):
            return self._undo_ace15_music_state(state)
        if self._wizard_is_cancel_command(raw):
            self._set_pending_ace15_music_state(None)
            self._append_framevision_assistant_reply("Cancelled the pending Ace-Step music request.")
            try:
                self._set_status("Ready", "ready")
            except Exception:
                pass
            return True
        state = self._wizard_clone_state(state)
        stage = str(state.get("stage") or "")
        if stage == "lyrics_mode":
            mode = self._ace15_parse_lyrics_mode(raw)
            if not mode:
                self._append_framevision_assistant_reply("Please choose: lyrics or instrumental. You can also say cancel.")
                return True
            state["lyrics_mode"] = mode
            if mode == "lyrics":
                state["stage"] = "lyrics_text"
                self._set_pending_ace15_music_state(state)
                self._append_framevision_assistant_reply("Ok, paste the lyrics in the chat.")
                return True
            state["stage"] = "genre"
            self._set_pending_ace15_music_state(state)
            self._append_framevision_assistant_reply(f"Instrumental selected. What genre?\n\nAvailable genres: {self._ace15_list_genres()}.")
            return True
        if stage == "lyrics_text":
            state["lyrics"] = raw
            state["stage"] = "genre"
            self._set_pending_ace15_music_state(state)
            self._append_framevision_assistant_reply(f"Lyrics received. What genre?\n\nAvailable genres: {self._ace15_list_genres()}.")
            return True
        if stage == "genre_confirm":
            matches = state.get("genre_candidates") if isinstance(state.get("genre_candidates"), list) else []
            chosen = self._ace15_select_match_from_reply(raw, matches)
            if chosen is None:
                if low in {"no", "none", "custom", "something else", "other"}:
                    state["genre"] = "Custom"
                    state["subgenre"] = "Custom"
                    state["stage"] = "caption"
                    self._set_pending_ace15_music_state(state)
                    self._append_framevision_assistant_reply("Describe the music style you want, for example: instruments, tempo/BPM, mood, vocal style, and anything to avoid.")
                    return True
                choices = self._ace15_match_choices_text(matches)
                self._append_framevision_assistant_reply(f"Please choose one of these by number/name, or describe a custom style:\n{choices}")
                return True
            state.pop("genre_candidates", None)
            self._append_framevision_assistant_reply(self._ace15_apply_music_match(state, chosen))
            return True
        if stage == "subgenre_confirm":
            matches = state.get("subgenre_candidates") if isinstance(state.get("subgenre_candidates"), list) else []
            chosen = self._ace15_select_match_from_reply(raw, matches)
            if chosen is None:
                if low in {"no", "none", "custom", "something else", "other"}:
                    state["subgenre"] = "Custom"
                    state["stage"] = "caption"
                    self._set_pending_ace15_music_state(state)
                    self._append_framevision_assistant_reply("Describe the music style you want, for example: instruments, tempo/BPM, mood, vocal style, and anything to avoid.")
                    return True
                choices = self._ace15_match_choices_text(matches)
                self._append_framevision_assistant_reply(f"Please choose one of these by number/name, or describe a custom style:\n{choices}")
                return True
            state.pop("subgenre_candidates", None)
            self._append_framevision_assistant_reply(self._ace15_apply_music_match(state, chosen))
            return True
        if stage == "genre":
            matches = self._ace15_music_match_candidates(raw, genre="", include_genres=True, include_subgenres=True, limit=8)
            if not matches:
                state["genre"] = raw
                state["subgenre"] = "Custom"
                state["stage"] = "caption"
                self._set_pending_ace15_music_state(state)
                self._append_framevision_assistant_reply("I could not find that in the Ace-Step presets. Describe the music style you want, for example: instruments, tempo/BPM, mood, vocal style, and anything to avoid.")
                return True
            qn = self._ace15_normalize_text(raw)
            exact = [m for m in matches if self._ace15_normalize_text(str(m.get("label") or m.get("genre") or m.get("subgenre") or "")) == qn or self._ace15_normalize_text(str(m.get("genre") or "")) == qn or self._ace15_normalize_text(str(m.get("subgenre") or "")) == qn]
            # For short/ambiguous words such as rap/reggae/trap, never silently jump to a fuzzy subgenre.
            # Ask the user to pick from nearby preset names instead.
            ask_confirm = False
            if len(matches) > 1:
                try:
                    top = int(matches[0].get("score") or 0)
                    second = int(matches[1].get("score") or 0)
                    ask_confirm = (len(qn) <= 6) or (second >= top - 35) or (not exact)
                except Exception:
                    ask_confirm = True
            if ask_confirm:
                state["stage"] = "genre_confirm"
                state["genre_candidates"] = matches
                self._set_pending_ace15_music_state(state)
                choices = self._ace15_match_choices_text(matches)
                first = str(matches[0].get("label") or "")
                self._append_framevision_assistant_reply(f"I think you mean `{first}`, or did you mean one of these?\n{choices}\n\nReply with the number/name, or describe a custom style.")
                return True
            self._append_framevision_assistant_reply(self._ace15_apply_music_match(state, matches[0]))
            return True
        if stage == "subgenre":
            genre_now = str(state.get("genre") or "")
            matches = self._ace15_music_match_candidates(raw, genre=genre_now, include_genres=False, include_subgenres=True, limit=8)
            if not matches:
                state["subgenre"] = "Custom"
                state["stage"] = "caption"
                self._set_pending_ace15_music_state(state)
                self._append_framevision_assistant_reply("I could not find that subgenre in the presets. Describe the music style you want, for example: instruments, tempo/BPM, mood, vocal style, and anything to avoid.")
                return True
            qn = self._ace15_normalize_text(raw)
            exact = [m for m in matches if self._ace15_normalize_text(str(m.get("subgenre") or "")) == qn or self._ace15_normalize_text(str(m.get("label") or "")) == qn]
            ask_confirm = False
            if len(matches) > 1:
                try:
                    top = int(matches[0].get("score") or 0)
                    second = int(matches[1].get("score") or 0)
                    ask_confirm = (len(qn) <= 6) or (second >= top - 35) or (not exact)
                except Exception:
                    ask_confirm = True
            if ask_confirm:
                state["stage"] = "subgenre_confirm"
                state["subgenre_candidates"] = matches
                self._set_pending_ace15_music_state(state)
                choices = self._ace15_match_choices_text(matches)
                first = str(matches[0].get("label") or "")
                self._append_framevision_assistant_reply(f"I think you mean `{first}`, or did you mean one of these?\n{choices}\n\nReply with the number/name, or describe a custom style.")
                return True
            self._append_framevision_assistant_reply(self._ace15_apply_music_match(state, matches[0]))
            return True
        if stage == "caption":
            state["caption"] = raw
            state["preset"] = self._ace15_default_music_preset(raw)
            state["stage"] = "bpm"
            self._set_pending_ace15_music_state(state)
            self._append_framevision_assistant_reply("Got the style description. " + self._ace15_bpm_help_text())
            return True
        if stage == "bpm":
            bpm = self._ace15_parse_bpm(raw)
            if bpm == 0:
                self._append_framevision_assistant_reply(self._ace15_bpm_help_text())
                return True
            state["bpm"] = 0 if bpm < 0 else bpm
            state["stage"] = "duration"
            self._set_pending_ace15_music_state(state)
            self._append_framevision_assistant_reply("BPM set to auto. How long should it be? Example: 3 minutes." if bpm < 0 else f"BPM set to {bpm}. How long should it be? Example: 3 minutes.")
            return True
        if stage == "duration":
            seconds = self._ace15_parse_duration_seconds(raw)
            if seconds <= 0:
                self._append_framevision_assistant_reply("Please give the duration, for example `3 minutes`, `180 seconds`, or `2:30`. You can also say cancel.")
                return True
            state["duration"] = seconds
            state["stage"] = "title_yes_no"
            self._set_pending_ace15_music_state(state)
            self._append_framevision_assistant_reply("Ok, I have what I need. Do you want a title for the filename?")
            return True
        if stage == "title_yes_no":
            yn = self._ace15_yes_no(raw)
            if yn == "yes":
                state["stage"] = "title_text"
                self._set_pending_ace15_music_state(state)
                self._append_framevision_assistant_reply("What should the filename title be?")
                return True
            if yn == "no":
                state["title"] = ""
                return self._finish_ace15_music_flow(state)
            state["title"] = raw
            return self._finish_ace15_music_flow(state)
        if stage == "title_text":
            state["title"] = raw
            return self._finish_ace15_music_flow(state)
        return True

    def _finish_ace15_music_flow(self, state: Dict[str, Any]) -> bool:
        genre = str(state.get("genre") or "")
        sub = str(state.get("subgenre") or "")
        if not genre or not sub:
            self._append_framevision_assistant_reply("Missing genre/subgenre, so I cannot queue the Ace-Step job yet.")
            return True
        ok, info, out_dir = self._queue_ace15_music(state)
        self._set_pending_ace15_music_state(None)
        if ok:
            dur = int(float(state.get("duration") or 0))
            title = str(state.get("title") or "").strip()
            title_line = f"\nTitle: `{title}`" if title else ""
            self._append_framevision_assistant_reply(
                f"Queued Ace-Step 1.5 music: `{genre} / {sub}` for {dur} seconds.{title_line}\n\nConfig: `{info}`"
            )
            self._track_ace15_music_job(title, genre, sub, float(state.get("duration") or 0), info, out_dir)
            try:
                self._unload_llm_for_queue_job()
            except Exception:
                pass
            try:
                self._set_status("Ace-Step music queued", "ready")
            except Exception:
                pass
        else:
            self._append_framevision_assistant_reply(f"I could not queue Ace-Step music.\n\nError: {info}")
            try:
                self._set_status("Ace-Step queue failed", "error")
            except Exception:
                pass
        return True

    def _image_attachment_paths_for_seedvr2(self, attachments: Optional[List[Dict[str, Any]]]) -> List[str]:
        out: List[str] = []
        for att in list(attachments or []):
            try:
                if str(att.get("kind", "") or "") != "image":
                    continue
                path = os.path.abspath(str(att.get("path", "") or ""))
                if path and os.path.isfile(path):
                    out.append(path)
            except Exception:
                pass
        return out

    def _media_attachment_paths_for_seedvr2(self, attachments: Optional[List[Dict[str, Any]]]) -> List[str]:
        out: List[str] = []
        for att in list(attachments or []):
            try:
                kind = str(att.get("kind", "") or "").lower()
                if kind not in {"image", "video"}:
                    continue
                path = os.path.abspath(str(att.get("path", "") or ""))
                if path and os.path.isfile(path):
                    out.append(path)
            except Exception:
                pass
        return out

    def _looks_like_seedvr2_upscale_request(self, text: str, attachments: Optional[List[Dict[str, Any]]]) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return False
        if not self._media_attachment_paths_for_seedvr2(attachments):
            return False
        return bool(re.search(r"\b(upscale|upscaler|seed\s*vr\s*2|seedvr2|enlarge|increase\s+resolution)\b", low, flags=re.IGNORECASE))

    def _seedvr2_requested_family_from_text(self, text: str, families: List[str]) -> str:
        low = str(text or "").strip().lower()
        fams = list(families or [])
        for fam in fams:
            n = fam.lower()
            if n in low or n.replace("b", " billion") in low:
                return fam
        if any(tok in low for tok in ("lighter", "less vram", "low vram", "fast", "faster", "small")):
            ranked = [fam for fam in self._seedvr2_config().get("model_families_order", ["3B", "7B", "9B"]) if fam in fams]
            return ranked[0] if ranked else ""
        if any(tok in low for tok in ("best", "quality", "heavy", "heavier", "large", "bigger", "big")):
            ranked = [fam for fam in self._seedvr2_config().get("model_families_order", ["3B", "7B", "9B"]) if fam in fams]
            return ranked[-1] if ranked else ""
        return ""

    def _send_message(self):
        text = (self.ed_prompt.toPlainText() or "").strip()
        attachments = list(self.pending_attachments)
        if not text and not attachments:
            return

        if self._assistant_jobs_active():
            self._apply_send_button_guard()
            self._set_status("Queue job running — sending is disabled until the job is finished", "loading")
            return

        save_mem, mem_title, mem_body = self._extract_memory_save_text(text)
        if save_mem and mem_body:
            s = self._append_user_message_to_current_session(text, attachments)
            if not s:
                return
            try:
                path = self._write_memory_note(mem_title or "Saved memory", mem_body, _memory_saved_notes_dir(self.fv_root), {"source_text": text})
                self._append_framevision_assistant_reply(f"Saved to memory: {os.path.relpath(path, self.fv_root)}")
                self._set_status("Saved to memory", "ready")
            except Exception as e:
                self._append_framevision_assistant_reply(f"Could not save to memory: {e}")
                self._set_status("Memory save failed", "error")
            return

        if self._looks_like_memory_choice_request(text):
            s = self._append_user_message_to_current_session(text, attachments)
            if not s:
                return
            self._ask_memory_source_choice(s.id, text)
            return

        if getattr(self, "_pending_ace15_music_request", None) is not None and text:
            s = self._append_user_message_to_current_session(text, attachments)
            if not s:
                return
            if self._handle_pending_ace15_music_message(text):
                return

        if self._ace15_music_intent(text):
            s = self._append_user_message_to_current_session(text, attachments)
            if not s:
                return
            self._start_ace15_music_flow(requested_text=text)
            return

        if getattr(self, "_pending_seedvr2_request", None) is not None and text:
            s = self._append_user_message_to_current_session(text, attachments)
            if not s:
                return
            if self._handle_pending_seedvr2_message(text):
                return

        if self._looks_like_seedvr2_upscale_request(text, attachments):
            s = self._append_user_message_to_current_session(text, attachments)
            if not s:
                return
            src_paths = self._media_attachment_paths_for_seedvr2(attachments)
            is_video_src = bool(src_paths and Path(src_paths[0]).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})
            requested_resolution = self._seedvr2_parse_resolution_choice(text, [720, 1080, 1440] if is_video_src else None)
            self._start_seedvr2_upscale_flow(src_paths[0], requested_text=text, requested_resolution=requested_resolution)
            return

        # FrameVision Assistant command router.
        # Image-create commands are handled before requiring a loaded local LLM,
        # so users can queue FrameVision image jobs instead of getting a text-only
        # "I cannot create images" response. Normal chat still goes to the LLM.
        if (text or attachments) and getattr(self, "_fv_assistant_router", None) is not None:
            try:
                route = self._fv_assistant_router.handle_user_text(text, attachments=attachments)
            except Exception as e:
                route = None
                try:
                    print("[fv assistant] router error:", e)
                except Exception:
                    pass
            if route is not None and getattr(route, "handled", False):
                s = self._append_user_message_to_current_session(text, attachments)
                if not s:
                    return
                self._append_framevision_assistant_reply(getattr(route, "message", "") or "Done.")
                if getattr(route, "queued", False):
                    try:
                        self._track_assistant_image_job(route)
                    except Exception as e:
                        print("[fv assistant] tracking error:", e)
                    try:
                        self._unload_llm_for_queue_job()
                    except Exception:
                        pass
                try:
                    self._set_status(("Video job queued" if getattr(route, "mode", "") == "video" and getattr(route, "queued", False) else "Image edit queued" if getattr(route, "mode", "") == "edit" and getattr(route, "queued", False) else "Image job queued" if getattr(route, "queued", False) else "Ready"), "ready")
                except Exception:
                    pass
                return

        try:
            self._validate_runner_and_model()
        except Exception as e:
            self._set_status(str(e), "error")
            QtWidgets.QMessageBox.warning(self, "Cannot send", str(e))
            return

        s = self._append_user_message_to_current_session(text, attachments)
        if not s:
            return

        tk, tv = self.current_template_choice()
        self._prepare_auto_retry_templates(self.current_model_path(), tk, tv)

        if not self._same_loaded_config() or not self.server_ready:
            self.pending_generate_session_id = s.id
            self._load_selected_model()
            return

        self._start_reply_request_for_current_session()

    def _on_chat_succeeded(self, payload):
        s = self._current_session()
        if not s:
            return
        images_payload: List[Dict] = []
        if isinstance(payload, dict):
            reply = str(payload.get("content", "") or "").strip()
            thinking = str(payload.get("thinking", "") or "").strip()
            images_payload = list(payload.get("images", []) or [])
        else:
            reply, thinking = _split_inline_reasoning(str(payload or ""))
        if self._try_auto_retry_after_bad_reply(reply):
            return
        self._auto_retry_templates = []
        saved_attachments: List[Dict] = []
        if images_payload:
            try:
                prompt_seed = self._last_user_text_sent or (s.messages[-1].get("content", "") if s.messages else "")
                saved_attachments = self._save_generated_images(images_payload, prompt_seed)
                if saved_attachments:
                    reply = (reply + "\n\n" if reply else "") + f"Saved image{'s' if len(saved_attachments) != 1 else ''} to output/images"
            except Exception as e:
                reply = (reply + "\n\n" if reply else "") + f"[Image save failed: {e}]"
        elif re.search(r"\b(?:here is|i generated|generated|creating|rendered)\b", reply or "", re.IGNORECASE) and re.search(r"\bimage\b", reply or "", re.IGNORECASE):
            reply = (reply + "\n\n" if reply else "") + "[No real image data was returned by the model/backend, so nothing could be saved. This reply is text only.]"
        try:
            sources = list(getattr(self, "_last_memory_sources", []) or [])
            if sources and self._memory_show_sources():
                shown = "\n".join([f"- {src}" for src in sources[:8]])
                reply = (reply + "\n\n" if reply else "") + "Memory/knowledge used:\n" + shown
        except Exception:
            pass
        if self._complete_pending_generation_update(reply, thinking, saved_attachments):
            self._auto_retry_templates = []
            return
        now = datetime.now().isoformat(timespec="seconds")
        mid = str(uuid.uuid4())
        s.messages.append({
            "id": mid,
            "role": "assistant",
            "content": reply,
            "thinking": thinking,
            "attachments": saved_attachments,
            "timestamp": now,
            "loading": False,
        })
        s.updated_at = now
        self.chat_view.add_message("assistant", reply, thinking, attachments=saved_attachments, message_id=mid, loading=False)
        self.chat_view.scroll_to_bottom(force=True)
        self._refresh_chat_list()
        self._update_header()
        self._queue_save()
        self._set_status("Ready", "ready")

    def _on_chat_failed(self, message: str):
        self._auto_retry_templates = []
        self.pending_retry_generation = False
        msg = str(message or "").strip()
        low = msg.lower()
        if "image input is not supported" in low and not self.active_mmproj_path:
            msg += "\n\nNo mmproj file was auto-detected next to this GGUF. Put the matching mmproj .gguf in the same model folder (or a mmproj/ subfolder) and reload the model."
        elif "image input is not supported" in low and self.active_mmproj_path:
            msg += f"\n\nmmproj currently attached: {self.active_mmproj_path}"
        pending_update = self._pending_generation_update if isinstance(self._pending_generation_update, dict) else None
        if pending_update:
            self._pending_generation_update = None
            try:
                _s, _m = self._find_session_message(str(pending_update.get("session_id") or ""), str(pending_update.get("assistant_message_id") or ""))
                if _m:
                    _m["loading"] = False
                    self.chat_view.update_message(str(_m.get("id") or ""), loading=False)
                    self._refresh_one_message_version_controls(_m)
                    self._queue_save()
            except Exception:
                pass
        self.chat_view.add_message("info", f"Generation failed: {msg}")
        if self.server_process and self.server_process.state() != QtCore.QProcess.NotRunning:
            self._set_status("Ready", "ready")
        else:
            self._set_status("Runner stopped", "error")

    def _on_chat_finished(self):
        self.chat_thread = None
        self._memory_context_mode_for_next_generation = "auto"
        self._apply_send_button_guard()
        self.btn_stop.setEnabled(False)

    def _stop_generation(self):
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_view.add_message("info", "Stopping local generation and unloading the current model…")
            self.pending_generate_session_id = ""
            self._cleanup_boot_thread()
            self._stop_process_only()
            self.btn_load.setEnabled(True)
            self.btn_unload.setEnabled(False)
            self._apply_send_button_guard()
            self.btn_stop.setEnabled(False)
            self._set_status("Stopped", "idle")
            self._update_header()
            return
        if self.server_process and self.server_process.state() != QtCore.QProcess.NotRunning:
            self._unload_model()
            return
        self._set_status("Nothing running", "idle")

    def _refresh_attachment_list(self):
        self.lst_attachments.clear()
        for att in self.pending_attachments:
            item = QtWidgets.QListWidgetItem(_make_attachment_thumbnail(att), _attachment_display_text(att))
            item.setToolTip(str(att.get("path", "") or att.get("name", "")))
            item.setData(QtCore.Qt.UserRole, dict(att))
            item.setSizeHint(QtCore.QSize(0, 64))
            self.lst_attachments.addItem(item)
        self.lst_attachments.setVisible(bool(self.pending_attachments))

    def _attachment_from_item(self, item: Optional[QtWidgets.QListWidgetItem]) -> Optional[Dict[str, Any]]:
        if item is None:
            return None
        data = item.data(QtCore.Qt.UserRole)
        return dict(data) if isinstance(data, dict) else None

    def _preview_attachment(self, att: Optional[Dict[str, Any]]):
        if not isinstance(att, dict):
            return
        dlg = AttachmentPreviewDialog(att, self)
        dlg.exec()

    def _preview_attachment_item(self, item: Optional[QtWidgets.QListWidgetItem]):
        att = self._attachment_from_item(item)
        if att:
            self._preview_attachment(att)

    def _open_attachment_externally(self, att: Optional[Dict[str, Any]], open_with_dialog: bool = False):
        if not isinstance(att, dict):
            return
        ok, err = _open_local_path(str(att.get("path", "") or ""), open_with_dialog=open_with_dialog)
        if not ok:
            self._set_status(err or "Could not open file", "error")

    def _add_attachment_paths(self, paths: List[str], announce: bool = True) -> int:
        existing = {str(att.get("path", "") or "") for att in self.pending_attachments}
        added = 0
        for path in paths or []:
            norm = os.path.abspath(str(path or ""))
            if not norm or not os.path.isfile(norm) or norm in existing:
                continue
            self.pending_attachments.append(_make_attachment_entry(norm))
            existing.add(norm)
            added += 1
        self._refresh_attachment_list()
        if added and announce:
            self._set_status(f"Attached {added} file(s)", "idle")
        return added

    def _save_pasted_image(self, image: QtGui.QImage) -> str:
        if image is None or image.isNull():
            return ""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"clipboard_{ts}_{uuid.uuid4().hex[:8]}.png"
        out_path = os.path.join(self.attachment_temp_dir, name)
        os.makedirs(self.attachment_temp_dir, exist_ok=True)
        if not image.save(out_path, "PNG"):
            raise RuntimeError("Could not save pasted image to attachment cache.")
        return out_path

    def _handle_pasted_image(self, image_obj):
        image = image_obj if isinstance(image_obj, QtGui.QImage) else QtGui.QImage(image_obj)
        if image.isNull():
            self._set_status("Clipboard image was empty", "error")
            return
        try:
            out_path = self._save_pasted_image(image)
        except Exception as e:
            self._set_status(f"Paste failed: {e}", "error")
            return
        self._add_attachment_paths([out_path], announce=False)
        self._set_status("Clipboard screenshot attached", "idle")

    def _handle_pasted_file_urls(self, paths_obj):
        paths = []
        if isinstance(paths_obj, (list, tuple)):
            paths = [str(p) for p in paths_obj]
        added = self._add_attachment_paths(paths, announce=False)
        if added:
            self._set_status(f"Attached {added} pasted file(s)", "idle")

    def _pick_attachments(self):
        start_dir = self.fv_root if os.path.isdir(self.fv_root) else os.getcwd()
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Attach files",
            start_dir,
            "Supported files (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.mp4 *.mov *.mkv *.webm *.avi *.m4v *.mp3 *.wav *.flac *.m4a *.aac *.ogg *.txt *.md *.py *.js *.ts *.json *.html *.css *.xml *.yaml *.yml *.ini *.cfg *.csv *.log *.bat *.ps1 *.cpp *.c *.h *.hpp *.java *.rs *.go *.php *.sql);;All files (*.*)",
        )
        if not paths:
            return
        self._add_attachment_paths(paths)

    def _remove_selected_attachment(self):
        row = self.lst_attachments.currentRow()
        if row < 0 or row >= len(self.pending_attachments):
            return
        del self.pending_attachments[row]
        self._refresh_attachment_list()

    def _show_chat_image_context_menu(self, path: str, widget: Optional[QtWidgets.QWidget], pos: QtCore.QPoint):
        img_path = os.path.abspath(str(path or ""))
        if not img_path:
            return
        menu = QtWidgets.QMenu(self)
        act_open = menu.addAction("Open the image")
        act_edit = menu.addAction("Use this image for an edit")
        act_video = menu.addAction("Use image to create a video")
        act_delete = menu.addAction("Delete this image")
        menu.addSeparator()
        act_seedvr2 = menu.addAction("Upscale with SeedVR2")
        base = widget if widget is not None else self
        chosen = menu.exec(base.mapToGlobal(pos))
        if chosen == act_open:
            ok, err = _open_local_path(img_path, open_with_dialog=False)
            if not ok:
                self._set_status(err or "Could not open image", "error")
            return
        if chosen == act_edit:
            self._start_context_image_edit(img_path)
            return
        if chosen == act_video:
            self._start_context_image_to_video(img_path)
            return
        if chosen == act_delete:
            self._delete_chat_image(img_path, widget)
            return
        if chosen == act_seedvr2:
            self._start_seedvr2_upscale_flow(img_path)
            return


    def _show_chat_media_context_menu(self, path: str, widget: Optional[QtWidgets.QWidget], pos: QtCore.QPoint):
        media_path = os.path.abspath(str(path or ""))
        if not media_path:
            return
        ext = Path(media_path).suffix.lower()
        is_video = ext in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        menu = QtWidgets.QMenu(self)
        if is_video:
            act_continue = menu.addAction("Continue this video")
            act_seedvr2 = menu.addAction("Upscale with SeedVR2")
            menu.addSeparator()
        else:
            act_continue = None
            act_seedvr2 = None
        act_open = menu.addAction("Open in external player")
        act_delete = menu.addAction("Delete from disk")
        base = widget if widget is not None else self
        chosen = menu.exec(base.mapToGlobal(pos))
        if is_video and chosen == act_continue:
            self._start_context_video_continue(media_path)
            return
        if is_video and chosen == act_seedvr2:
            self._start_seedvr2_upscale_flow(media_path)
            return
        if chosen == act_open:
            ok, err = _open_local_path(media_path, open_with_dialog=False)
            if not ok:
                self._set_status(err or "Could not open media", "error")
            return
        if chosen == act_delete:
            self._delete_chat_media(media_path, widget)
            return

    def _start_context_video_continue(self, video_path: str):
        vid_path = os.path.abspath(str(video_path or ""))
        if not vid_path or not os.path.isfile(vid_path):
            self._append_framevision_assistant_reply("That video could not be found on disk anymore.")
            return
        self._set_pending_seedvr2_state(None)
        self._set_pending_ace15_music_state(None)
        att = _make_attachment_entry(vid_path)
        route = None
        if getattr(self, "_fv_assistant_router", None) is not None:
            try:
                route = self._fv_assistant_router.handle_user_text("continue this video", attachments=[att])
            except Exception as e:
                try:
                    print("[fv assistant] context-video router error:", e)
                except Exception:
                    pass
        if route is not None and getattr(route, "handled", False):
            self._append_framevision_assistant_reply(getattr(route, "message", "") or "Which LTX model should I use?")
            try:
                self._set_status("Video selected for LTX continuation", "ready")
            except Exception:
                pass
            return
        self._append_framevision_assistant_reply("I could not start the LTX continue-video flow from that video.")

    def _delete_chat_media(self, media_path: str, widget: Optional[QtWidgets.QWidget] = None):
        path = os.path.abspath(str(media_path or ""))
        if not path or not os.path.exists(path):
            self._append_framevision_assistant_reply("That media file was already missing on disk.")
            return
        name = os.path.basename(path)
        if QtWidgets.QMessageBox.question(self, "Delete media", f"Delete this file from disk?\n\n{name}") != QtWidgets.QMessageBox.Yes:
            return
        try:
            if widget is not None:
                widget.setEnabled(False)
        except Exception:
            pass
        try:
            self.pending_attachments = [att for att in self.pending_attachments if os.path.abspath(str(att.get("path", "") or "")) != path]
            self._refresh_attachment_list()
        except Exception:
            pass
        try:
            os.remove(path)
        except Exception as e:
            self._append_framevision_assistant_reply(f"Could not delete the media file: {e}")
            try:
                self._set_status("Delete failed", "error")
            except Exception:
                pass
            return
        self._append_framevision_assistant_reply(f"Deleted media file: `{name}`")
        try:
            self._set_status("Media deleted", "ready")
        except Exception:
            pass
        self._queue_save()

    def _start_context_image_to_video(self, image_path: str):
        img_path = os.path.abspath(str(image_path or ""))
        if not img_path or not os.path.isfile(img_path):
            self._append_framevision_assistant_reply("That image could not be found on disk anymore.")
            return
        self._set_pending_seedvr2_state(None)
        self._set_pending_ace15_music_state(None)
        att = _make_attachment_entry(img_path)
        route = None
        if getattr(self, "_fv_assistant_router", None) is not None:
            try:
                route = self._fv_assistant_router.handle_user_text("animate this image", attachments=[att])
            except Exception as e:
                try:
                    print("[fv assistant] context-image-video router error:", e)
                except Exception:
                    pass
        if route is not None and getattr(route, "handled", False):
            self._append_framevision_assistant_reply(getattr(route, "message", "") or "Let's create a video from this image.")
            try:
                self._set_status("Image selected for video creation", "ready")
            except Exception:
                pass
            return
        self.pending_attachments = [att]
        self._refresh_attachment_list()
        self._append_framevision_assistant_reply("I attached that image for video creation. What would you like the video to show?")
        try:
            self._set_status("Image attached for video creation", "ready")
        except Exception:
            pass

    def _start_context_image_edit(self, image_path: str):
        img_path = os.path.abspath(str(image_path or ""))
        if not img_path or not os.path.isfile(img_path):
            self._append_framevision_assistant_reply("That image could not be found on disk anymore.")
            return
        self._set_pending_seedvr2_state(None)
        self._set_pending_ace15_music_state(None)
        att = _make_attachment_entry(img_path)
        route = None
        if getattr(self, "_fv_assistant_router", None) is not None:
            try:
                route = self._fv_assistant_router.handle_user_text("edit this image", attachments=[att])
            except Exception as e:
                try:
                    print("[fv assistant] context-edit router error:", e)
                except Exception:
                    pass
        if route is not None and getattr(route, "handled", False):
            self._append_framevision_assistant_reply(getattr(route, "message", "") or "What would you like to change in this image?")
            try:
                self._set_status("Image selected for edit", "ready")
            except Exception:
                pass
            return
        self.pending_attachments = [att]
        self._refresh_attachment_list()
        self._append_framevision_assistant_reply("I attached that image for editing. What would you like to change?")
        try:
            self._set_status("Image attached for edit", "ready")
        except Exception:
            pass

    def _delete_chat_image(self, image_path: str, widget: Optional[QtWidgets.QWidget] = None):
        img_path = os.path.abspath(str(image_path or ""))
        if not img_path or not os.path.exists(img_path):
            self._append_framevision_assistant_reply("That image was already missing on disk.")
            return
        name = os.path.basename(img_path)
        if QtWidgets.QMessageBox.question(self, "Delete image", f"Delete this image from disk?\n\n{name}") != QtWidgets.QMessageBox.Yes:
            return
        try:
            if widget is not None and hasattr(widget, "clear"):
                widget.clear()
                if hasattr(widget, "setText"):
                    widget.setText("[Image deleted]")
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents, 25)
        except Exception:
            pass
        try:
            self.pending_attachments = [att for att in self.pending_attachments if os.path.abspath(str(att.get("path", "") or "")) != img_path]
            self._refresh_attachment_list()
        except Exception:
            pass
        try:
            os.remove(img_path)
        except Exception as e:
            self._append_framevision_assistant_reply(f"Could not delete the image: {e}")
            try:
                self._set_status("Delete failed", "error")
            except Exception:
                pass
            return
        self._append_framevision_assistant_reply(f"Deleted image: `{name}`")
        try:
            self._set_status("Image deleted", "ready")
        except Exception:
            pass
        self._queue_save()

    def _fv_assistant_registry(self) -> Dict[str, Any]:
        path = os.path.join(self.fv_root, "scripts", "fv_assistant_image_models.json")
        data = _load_json(path, {})
        return data if isinstance(data, dict) else {}

    def _seedvr2_config(self) -> Dict[str, Any]:
        reg = self._fv_assistant_registry()
        tools = reg.get("tools") if isinstance(reg.get("tools"), dict) else {}
        cfg = tools.get("seedvr2") if isinstance(tools.get("seedvr2"), dict) else {}
        defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
        out = {
            "label": str(cfg.get("label") or "SeedVR2"),
            "folder": str(cfg.get("folder") or "models/SEEDVR2"),
            "output_dir": str(cfg.get("output_dir") or "output/photo"),
            "cli": str(cfg.get("cli") or "presets/extra_env/seedvr2_src/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py"),
            "env_win": str(cfg.get("env_win") or "environments/.seedvr2/Scripts/python.exe"),
            "env_posix": str(cfg.get("env_posix") or "environments/.seedvr2/bin/python"),
            "runner": str(cfg.get("runner") or "helpers/seedvr2_runner.py"),
            "resolutions": list(cfg.get("resolutions") or [1080, 1440, 2160]),
            "model_families_order": list(cfg.get("model_families_order") or ["3B", "7B", "9B"]),
            "model_family_labels": dict(cfg.get("model_family_labels") or {
                "3B": "3B (good quality, lighter on VRAM)",
                "7B": "7B (best quality, slower and VRAM heavier)",
                "9B": "9B (detected model, slower and VRAM heavier)",
            }),
            "batch_size": int(defaults.get("batch_size", 2) or 2),
            "chunk_size": int(defaults.get("chunk_size", 20) or 20),
            "temporal_overlap": int(defaults.get("temporal_overlap", 0) or 0),
            "prepend_frames": int(defaults.get("prepend_frames", 0) or 0),
            "color_correction": str(defaults.get("color_correction") or "lab"),
            "attention_mode": str(defaults.get("attention_mode") or "auto"),
            "output_format": str(defaults.get("output_format") or "png"),
            "video_backend": str(defaults.get("video_backend") or "ffmpeg"),
            "open_on_success": bool(defaults.get("open_on_success", True)),
        }
        try:
            out["resolutions"] = [int(x) for x in out["resolutions"] if int(x) > 0]
        except Exception:
            out["resolutions"] = [1080, 1440, 2160]
        return out

    def _root_join(self, rel_or_abs: str) -> Path:
        p = Path(str(rel_or_abs or ""))
        if p.is_absolute():
            return p
        return Path(self.fv_root) / p

    def _unload_llm_for_queue_job(self):
        """Free VRAM after FrameVision queues a heavy job. This must not cancel chat job tracking."""
        try:
            running = bool(self.server_process and self.server_process.state() != QtCore.QProcess.NotRunning)
        except Exception:
            running = False
        try:
            booting = bool(self.server_boot_thread and self.server_boot_thread.isRunning())
        except Exception:
            booting = False
        if not running and not booting and not self.server_ready:
            return
        try:
            self.pending_generate_session_id = ""
            self._cleanup_boot_thread()
            self._stop_process_only()
            self.btn_load.setEnabled(True)
            self.btn_unload.setEnabled(False)
            self._apply_send_button_guard()
            self.btn_stop.setEnabled(False)
            self._update_header()
        except Exception:
            pass

    def _seedvr2_models_dir(self) -> Path:
        return self._root_join(str(self._seedvr2_config().get("folder") or "models/SEEDVR2"))

    def _seedvr2_cli_path(self) -> Path:
        return self._root_join(str(self._seedvr2_config().get("cli") or "presets/extra_env/seedvr2_src/ComfyUI-SeedVR2_VideoUpscaler/inference_cli.py"))

    def _seedvr2_env_python(self) -> Path:
        cfg = self._seedvr2_config()
        if os.name == "nt":
            return self._root_join(str(cfg.get("env_win") or "environments/.seedvr2/Scripts/python.exe"))
        return self._root_join(str(cfg.get("env_posix") or "environments/.seedvr2/bin/python"))

    def _seedvr2_runner_path(self) -> Optional[Path]:
        candidates = []
        try:
            candidates.append(Path(__file__).resolve().with_name("seedvr2_runner.py"))
        except Exception:
            pass
        try:
            candidates.append(self._root_join(str(self._seedvr2_config().get("runner") or "helpers/seedvr2_runner.py")).resolve())
        except Exception:
            pass
        try:
            candidates.append((Path(self.fv_root) / "helpers" / "seedvr2_runner.py").resolve())
        except Exception:
            pass
        try:
            if getattr(sys, "frozen", False) and getattr(sys, "executable", None):
                candidates.append((Path(sys.executable).resolve().parent / "helpers" / "seedvr2_runner.py"))
        except Exception:
            pass
        try:
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                candidates.append(Path(meipass) / "helpers" / "seedvr2_runner.py")
        except Exception:
            pass
        try:
            candidates.append(Path.cwd() / "helpers" / "seedvr2_runner.py")
        except Exception:
            pass
        for cand in candidates:
            try:
                if cand and cand.exists():
                    return cand
            except Exception:
                pass
        return None

    def _seedvr2_scan_models(self) -> List[Path]:
        out: List[Path] = []
        try:
            base = self._seedvr2_models_dir()
            if base.exists():
                out = sorted(base.rglob("*.gguf"))
        except Exception:
            out = []
        return out

    def _seedvr2_model_family(self, model_path: Path) -> str:
        low = model_path.name.lower()
        for fam in ("9B", "7B", "3B"):
            n = fam[:-1]
            if re.search(rf"(^|[^0-9]){n}b([^0-9]|$)", low, flags=re.IGNORECASE):
                return fam
        return "other"

    def _seedvr2_group_models(self, models: List[Path]) -> Dict[str, List[Path]]:
        groups: Dict[str, List[Path]] = {}
        for model_path in models or []:
            fam = self._seedvr2_model_family(model_path)
            groups.setdefault(fam, []).append(model_path)
        for fam, items in list(groups.items()):
            try:
                items.sort(key=lambda p: (p.stat().st_size, p.name.lower()))
            except Exception:
                items.sort(key=lambda p: p.name.lower())
            groups[fam] = items
        return groups

    def _seedvr2_pick_best_model(self, family: str, groups: Dict[str, List[Path]]) -> Optional[Path]:
        items = list(groups.get(family) or [])
        if not items:
            return None
        return items[-1]

    def _seedvr2_parse_family_choice(self, text: str, available: List[str]) -> str:
        low = str(text or "").strip().lower()
        fams = list(available or [])
        for fam in fams:
            if fam.lower() in low or fam.lower().replace("b", " billion") in low:
                return fam
        ranked = [fam for fam in self._seedvr2_config().get("model_families_order", ["3B", "7B", "9B"]) if fam in fams]
        if any(tok in low for tok in ("light", "lighter", "fast", "faster", "low vram", "less vram", "small", "3b")):
            return ranked[0] if ranked else ""
        if any(tok in low for tok in ("best", "quality", "heavier", "heavy", "large", "bigger", "big", "9b", "7b")):
            return ranked[-1] if ranked else ""
        if len(fams) == 1:
            return fams[0]
        return ""

    def _seedvr2_parse_resolution_choice(self, text: str, allowed_values: Optional[List[int]] = None) -> int:
        low = str(text or "").strip().lower().replace(" ", "")
        allowed = set(int(x) for x in (allowed_values if allowed_values is not None else self._seedvr2_config().get("resolutions", [1080, 1440, 2160])))
        checks = [
            (2160, ("2160", "2160p", "4k", "uhd")),
            (1440, ("1440", "1440p", "2k", "qhd")),
            (1080, ("1080", "1080p", "fullhd", "fhd")),
            (720, ("720", "720p", "hd")),
        ]
        for value, toks in checks:
            if value in allowed and any(tok in low for tok in toks):
                return value
        for value in sorted(allowed, reverse=True):
            if str(value) in low or f"{value}p" in low:
                return int(value)
        return 0

    def _seedvr2_output_path_for_image(self, src_path: Path, resolution: int) -> Path:
        outd = self._root_join(str(self._seedvr2_config().get("output_dir") or "output/photo"))
        outd.mkdir(parents=True, exist_ok=True)
        try:
            raw_stem = (src_path.stem or "image")[:15]
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_stem).strip("_")
            if not safe:
                safe = "image"
            stamp = time.strftime("%y%m%d", time.localtime())
            tag = f"{safe}_seedVR2_{resolution}_{stamp}"
        except Exception:
            tag = f"{src_path.stem}_seedVR2_{resolution}"
        outfile = outd / f"{tag}.png"
        if outfile.exists():
            n = 1
            while True:
                cand = outd / f"{tag}_{n:02d}.png"
                if not cand.exists():
                    outfile = cand
                    break
                n += 1
        return outfile


    def _seedvr2_output_path_for_video(self, src_path: Path, resolution: int) -> Path:
        outd = self._root_join("output/upscaled/seedvr2/video")
        outd.mkdir(parents=True, exist_ok=True)
        try:
            raw_stem = (src_path.stem or "video")[:18]
            safe = re.sub(r"[^A-Za-z0-9_-]+", "_", raw_stem).strip("_") or "video"
            stamp = time.strftime("%y%m%d", time.localtime())
            tag = f"{safe}_seedVR2_{resolution}_{stamp}"
        except Exception:
            tag = f"{src_path.stem}_seedVR2_{resolution}"
        outfile = outd / f"{tag}.mp4"
        if outfile.exists():
            n = 1
            while True:
                cand = outd / f"{tag}_{n:02d}.mp4"
                if not cand.exists():
                    outfile = cand
                    break
                n += 1
        return outfile

    def _assistant_find_enqueue_callable(self):
        main_obj = None
        try:
            win = self.window()
            if win is not None and win is not self:
                main_obj = win
        except Exception:
            main_obj = None
        if main_obj is None:
            main_obj = getattr(self, "_main", None)
        candidates = []
        for name in ("enqueue", "enqueue_job", "enqueue_external", "enqueue_single_action", "queue_add", "add_job"):
            fn = getattr(main_obj, name, None) if main_obj is not None else None
            if callable(fn):
                candidates.append((fn, f"main.{name}"))
        qa = getattr(main_obj, "queue_adapter", None) if main_obj is not None else None
        for name in ("enqueue", "add", "put"):
            fn = getattr(qa, name, None) if qa is not None else None
            if callable(fn):
                candidates.append((fn, f"main.queue_adapter.{name}"))
        q = getattr(main_obj, "queue", None) if main_obj is not None else None
        for name in ("enqueue", "add", "put", "add_job"):
            fn = getattr(q, name, None) if q is not None else None
            if callable(fn):
                candidates.append((fn, f"main.queue.{name}"))
        try:
            import helpers.queue_adapter as _qa  # type: ignore
            for name in ("enqueue", "add", "put", "add_job"):
                fn = getattr(_qa, name, None)
                if callable(fn):
                    candidates.append((fn, f"helpers.queue_adapter.{name}"))
        except Exception:
            pass
        for prefer in ("helpers.queue_adapter.enqueue", "main.queue_adapter.enqueue", "main.enqueue", "main.enqueue_job"):
            for fn, label in candidates:
                if label.endswith(prefer):
                    return fn, label
        return candidates[0] if candidates else (None, "")

    def _assistant_write_pending_job(self, job: Dict[str, Any]) -> bool:
        pending = Path(self.fv_root) / "jobs" / "pending"
        pending.mkdir(parents=True, exist_ok=True)
        jid = str(job.get("id") or uuid.uuid4().hex)
        job["id"] = jid
        job.setdefault("status", "pending")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = pending / f"assistant_{stamp}_{jid[:8]}.json"
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(job, f, ensure_ascii=False, indent=2)
            os.replace(tmp, out_path)
            return True
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return False

    def _assistant_enqueue_external_job(self, job: Dict[str, Any]) -> bool:
        enq, _label = self._assistant_find_enqueue_callable()
        if callable(enq):
            try:
                enq(job)
                return True
            except Exception:
                pass
        return self._assistant_write_pending_job(job)

    def _queue_seedvr2_upscale(self, image_path: str, model_path: Path, resolution: int, source_kind: str = "image") -> Tuple[bool, str]:
        src = Path(image_path).resolve()
        cli = self._seedvr2_cli_path()
        env_py = self._seedvr2_env_python()
        runner = self._seedvr2_runner_path()
        if not src.exists():
            return False, "That source file no longer exists on disk."
        if not cli.exists():
            return False, f"SeedVR2 CLI was not found: {cli}"
        if not env_py.exists():
            return False, f"SeedVR2 environment python was not found: {env_py}"
        is_video_src = str(source_kind or "image").lower() == "video" or src.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        outfile = self._seedvr2_output_path_for_video(src, int(resolution)) if is_video_src else self._seedvr2_output_path_for_image(src, int(resolution))
        ffmpeg = Path(self.fv_root) / "presets" / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        ffprobe = Path(self.fv_root) / "presets" / "bin" / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
        cfg = self._seedvr2_config()
        if is_video_src:
            batch_size = 4 if int(resolution) == 720 else 2
            chunk_size = 50 if int(resolution) == 720 else 20
            temporal_overlap = 1
            prepend_frames = 1
            output_format = "mp4"
        else:
            batch_size = int(cfg.get("batch_size", 2) or 2)
            chunk_size = int(cfg.get("chunk_size", 20) or 20)
            temporal_overlap = int(cfg.get("temporal_overlap", 0) or 0)
            prepend_frames = int(cfg.get("prepend_frames", 0) or 0)
            output_format = str(cfg.get("output_format") or "png")
        seed_forward = [
            "--output", str(outfile),
            "--output_format", str(output_format),
            "--video_backend", str(cfg.get("video_backend") or "ffmpeg"),
            "--model_dir", str(model_path.parent),
            "--dit_model", model_path.name,
            "--resolution", str(int(resolution)),
            "--batch_size", str(int(batch_size)),
            "--chunk_size", str(int(chunk_size)),
            "--temporal_overlap", str(int(temporal_overlap)),
            "--prepend_frames", str(int(prepend_frames)),
            "--color_correction", str(cfg.get("color_correction") or "lab"),
            "--attention_mode", str(cfg.get("attention_mode") or "auto"),
        ]
        if runner is not None:
            cmd = [str(env_py), "-X", "utf8", str(runner),
                   "--cli", str(cli),
                   "--input", str(src),
                   "--ffmpeg", str(ffmpeg),
                   "--ffprobe", str(ffprobe),
                   "--work_root", str(self.fv_root),
                   "--"] + seed_forward
            cwd = str(cli.parent)
        else:
            cmd = [str(env_py), "-X", "utf8", str(cli), str(src)] + seed_forward
            cwd = str(cli.parent)
        env = os.environ.copy()
        try:
            if os.name == "nt":
                env["PATH"] = str(Path(self.fv_root) / "presets" / "bin") + ";" + env.get("PATH", "")
            else:
                env["PATH"] = str(Path(self.fv_root) / "presets" / "bin") + ":" + env.get("PATH", "")
        except Exception:
            pass
        env.setdefault("PYTHONUTF8", "1")
        env.setdefault("PYTHONIOENCODING", "utf-8")
        job = {
            "id": uuid.uuid4().hex,
            "type": "external_cmd",
            "title": f"SeedVR2 {'video ' if is_video_src else ''}upscale: {src.name}",
            "name": "Upscale (SeedVR2)",
            "category": "upscale",
            "engine": "seedvr2",
            "cmd": cmd,
            "cwd": cwd,
            "env": env,
            "open_on_success": False if bool(self._assistant_chat_only_results_enabled()) else bool(self._seedvr2_config().get("open_on_success", True)),
            "assistant_chat_only": bool(self._assistant_chat_only_results_enabled()),
            "output": str(outfile),
            "out_path": str(outfile),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending",
        }
        ok = self._assistant_enqueue_external_job(job)
        return ok, str(outfile)

    def _start_seedvr2_upscale_flow(self, image_path: str, requested_text: str = "", requested_resolution: int = 0):
        img_path = os.path.abspath(str(image_path or ""))
        if not img_path or not os.path.isfile(img_path):
            self._append_framevision_assistant_reply("That source file could not be found on disk anymore.")
            return
        source_kind = "video" if Path(img_path).suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"} else "image"
        models = self._seedvr2_scan_models()
        if not models:
            self._append_framevision_assistant_reply(
                "I could not find any SeedVR2 GGUF models in `models/SEEDVR2`. Please install or copy a SeedVR2 model there first."
            )
            return
        groups = self._seedvr2_group_models(models)
        families = [fam for fam in self._seedvr2_config().get("model_families_order", ["3B", "7B", "9B"]) if fam in groups]
        if not families:
            families = list(groups.keys())
        requested_family = self._seedvr2_requested_family_from_text(requested_text, families)
        chosen_family = requested_family if requested_family in families else (families[0] if len(families) == 1 else "")
        self._set_pending_seedvr2_state({
            "stage": "choose_model" if len(families) > 1 and not chosen_family else "choose_resolution",
            "source_path": img_path,
            "source_name": os.path.basename(img_path),
            "source_kind": source_kind,
            "groups": {k: [str(p) for p in v] for k, v in groups.items()},
            "families": families,
            "family": chosen_family,
        })
        allowed_initial_res = ([720, 1080, 1440] if source_kind == "video" else [int(x) for x in self._seedvr2_config().get("resolutions", [1080, 1440, 2160])])
        if chosen_family and requested_resolution in allowed_initial_res:
            model_path = self._seedvr2_pick_best_model(chosen_family, {k: [Path(x) for x in v] for k, v in self._pending_seedvr2_request.get("groups", {}).items()})
            if model_path is not None:
                ok, out_info = self._queue_seedvr2_upscale(img_path, model_path, requested_resolution, source_kind=source_kind)
                self._set_pending_seedvr2_state(None)
                if ok:
                    self._append_framevision_assistant_reply(
                        f"Queued SeedVR2 upscale for `{os.path.basename(img_path)}` using `{model_path.name}` at {requested_resolution}p.\n\n"
                        f"Output: `{out_info}`"
                    )
                    self._track_seedvr2_upscale_job(img_path, model_path.name, requested_resolution, out_info, source_kind=source_kind)
                    try:
                        self._unload_llm_for_queue_job()
                    except Exception:
                        pass
                    try:
                        self._set_status("SeedVR2 job queued", "ready")
                    except Exception:
                        pass
                else:
                    self._append_framevision_assistant_reply(f"I could not queue the SeedVR2 upscale.\n\nError: {out_info}")
                    try:
                        self._set_status("SeedVR2 queue failed", "error")
                    except Exception:
                        pass
                return
        if len(families) > 1 and not chosen_family:
            labels = []
            family_labels = dict(self._seedvr2_config().get("model_family_labels") or {})
            for fam in families:
                labels.append(str(family_labels.get(fam) or fam))
            self._append_framevision_assistant_reply(
                f"SeedVR2 upscale selected for `{os.path.basename(img_path)}`.\n\n"
                f"Which model do you want to use: {' or '.join(labels)}? After that I will ask for one of the configured output sizes. You can also say `undo` or `cancel`."
            )
            try:
                self._set_status("Waiting for SeedVR2 model choice", "ready")
            except Exception:
                pass
            return
        fam = self._pending_seedvr2_request.get("family", "") or (families[0] if families else "")
        chosen = self._seedvr2_pick_best_model(fam, {k: [Path(x) for x in v] for k, v in self._pending_seedvr2_request.get("groups", {}).items()})
        model_name = chosen.name if chosen is not None else "the detected model"
        self._append_framevision_assistant_reply(
            f"SeedVR2 upscale selected for `{os.path.basename(img_path)}` using `{model_name}`.\n\n"
            f"Choose the target resolution: {', '.join(str(x) + 'p' for x in ([720, 1080, 1440] if source_kind == 'video' else self._seedvr2_config().get('resolutions', [1080, 1440, 2160])))}. You can also say `undo` or `cancel`."
        )
        try:
            self._set_status("Waiting for SeedVR2 resolution", "ready")
        except Exception:
            pass

    def _handle_pending_seedvr2_message(self, text: str) -> bool:
        state = getattr(self, "_pending_seedvr2_request", None)
        if not isinstance(state, dict):
            return False
        raw = str(text or "").strip()
        low = raw.lower()
        if not raw:
            return True
        if self._wizard_is_undo_command(raw):
            return self._undo_seedvr2_state(state)
        if self._wizard_is_cancel_command(raw):
            self._set_pending_seedvr2_state(None)
            self._append_framevision_assistant_reply("Cancelled the pending SeedVR2 upscale.")
            try:
                self._set_status("Ready", "ready")
            except Exception:
                pass
            return True

        state = self._wizard_clone_state(state)
        groups = {k: [Path(x) for x in v] for k, v in (state.get("groups", {}) or {}).items()}
        families = list(state.get("families") or [])
        stage = str(state.get("stage") or "")
        source_kind = str(state.get("source_kind") or "image")

        if stage == "choose_model":
            fam = self._seedvr2_parse_family_choice(raw, families)
            if not fam:
                opts = ", ".join(families) if families else "3B"
                self._append_framevision_assistant_reply(f"Please choose one of these SeedVR2 model groups: {opts}. You can also say cancel.")
                return True
            state["family"] = fam
            state["stage"] = "choose_resolution"
            self._set_pending_seedvr2_state(state)
            chosen = self._seedvr2_pick_best_model(fam, groups)
            model_name = chosen.name if chosen is not None else fam
            self._append_framevision_assistant_reply(
                f"Okay, I will use `{model_name}`.\n\nChoose the target resolution: {', '.join(str(x) + 'p' for x in self._seedvr2_config().get('resolutions', [1080, 1440, 2160]))}."
            )
            try:
                self._set_status("Waiting for SeedVR2 resolution", "ready")
            except Exception:
                pass
            return True

        if stage == "choose_resolution":
            allowed_resolutions = [720, 1080, 1440] if source_kind == "video" else [int(x) for x in self._seedvr2_config().get("resolutions", [1080, 1440, 2160])]
            resolution = self._seedvr2_parse_resolution_choice(raw, allowed_resolutions)
            if resolution not in allowed_resolutions:
                self._append_framevision_assistant_reply(
                    f"Please choose one of the configured target resolutions: {', '.join(str(x) + 'p' for x in allowed_resolutions)}. You can also say cancel."
                )
                return True
            fam = str(state.get("family") or "")
            if not fam and families:
                fam = families[0]
            model_path = self._seedvr2_pick_best_model(fam, groups)
            if model_path is None:
                flat = []
                for arr in groups.values():
                    flat.extend(arr)
                model_path = flat[-1] if flat else None
            if model_path is None:
                self._set_pending_seedvr2_state(None)
                self._append_framevision_assistant_reply("I could not find a usable SeedVR2 model anymore.")
                return True
            ok, out_info = self._queue_seedvr2_upscale(str(state.get("source_path") or ""), model_path, resolution, source_kind=source_kind)
            self._set_pending_seedvr2_state(None)
            if ok:
                self._append_framevision_assistant_reply(
                    f"Queued SeedVR2 upscale for `{state.get('source_name') or os.path.basename(str(state.get('source_path') or 'image'))}` using `{model_path.name}` at {resolution}p.\n\n"
                    f"Output: `{out_info}`"
                )
                source_path = str(state.get("source_path") or "")
                self._track_seedvr2_upscale_job(source_path, model_path.name, resolution, out_info, source_kind=source_kind)
                try:
                    self._unload_llm_for_queue_job()
                except Exception:
                    pass
                try:
                    self._set_status("SeedVR2 job queued", "ready")
                except Exception:
                    pass
            else:
                self._append_framevision_assistant_reply(f"I could not queue the SeedVR2 upscale.\n\nError: {out_info}")
                try:
                    self._set_status("SeedVR2 queue failed", "error")
                except Exception:
                    pass
            return True

        return False

    def _attachment_context_menu(self, pos):
        if not self.pending_attachments:
            return
        row = self.lst_attachments.indexAt(pos).row()
        att = None
        if row >= 0:
            self.lst_attachments.setCurrentRow(row)
            att = dict(self.pending_attachments[row])
        menu = QtWidgets.QMenu(self)
        act_preview = menu.addAction("Preview") if att else None
        act_open = menu.addAction("Open") if att else None
        act_open_with = menu.addAction("Open with…") if att and str(att.get("kind", "file") or "file") in {"text", "file"} else None
        if att:
            menu.addSeparator()
        act_remove = menu.addAction("Remove attachment") if att else None
        act_clear = menu.addAction("Clear all attachments")
        chosen = menu.exec(self.lst_attachments.mapToGlobal(pos))
        if chosen == act_preview and att:
            self._preview_attachment(att)
        elif chosen == act_open and att:
            self._open_attachment_externally(att, open_with_dialog=False)
        elif chosen == act_open_with and att:
            self._open_attachment_externally(att, open_with_dialog=True)
        elif chosen == act_remove:
            self._remove_selected_attachment()
        elif chosen == act_clear:
            self.pending_attachments = []
            self._refresh_attachment_list()

    def _update_header(self):
        s = self._current_session()
        selected_model = os.path.basename(self.current_model_path()) if self.current_model_path() else "No model selected"
        loaded_model = os.path.basename(self.loaded_model_path) if self.loaded_model_path else "Not loaded"
        tk, tv = self.current_template_choice()
        template = tv if tv else tk
        eff_k, eff_v = self._effective_template_choice(self.current_model_path(), tk, tv)
        effective_template = eff_v if eff_v else eff_k
        if tk == 'auto':
            template_text = 'auto(metadata/default)'
        elif tk == 'smart':
            template_text = f"smart→{effective_template}" if effective_template and effective_template != 'auto' else 'smart'
        else:
            template_text = template or 'auto'

        if self.server_ready and self.loaded_model_path:
            state_text = f"loaded: {loaded_model}"
        elif self.server_process and self.server_process.state() != QtCore.QProcess.NotRunning:
            state_text = f"loading: {selected_model}"
        else:
            state_text = f"selected: {selected_model}"

        self.lbl_title.setText(s.title if s else "New Chat")
        self.lbl_model_info.setText(
            f"{state_text}   •   template: {template_text}   •   ctx: {int(self.settings_dialog.sp_ctx_size.value())}   •   max tokens: {int(self.settings_dialog.sp_max_tokens.value())}"
        )

    # ---------- window ----------
    def closeEvent(self, event: QtGui.QCloseEvent):
        self.pending_generate_session_id = ""
        self._cleanup_boot_thread()
        self._stop_process_only()
        self._apply_settings_to_current_session()
        self._save_all()
        super().closeEvent(event)




# -----------------------------
# Embeddable FrameVision tab
# -----------------------------
class LlamaChatPane(LlamaChatWindow):
    """Embeddable version of the local LLM chat UI for the main FrameVision tab widget.

    It intentionally runs in the same Python environment as FrameVision and uses the
    bundled llama-server.exe path from presets/bin/llama when available.
    """

    def __init__(self, parent=None):
        super().__init__()
        if parent is not None:
            try:
                self.setParent(parent)
            except Exception:
                pass
        try:
            self.setWindowFlags(QtCore.Qt.Widget)
        except Exception:
            pass
        try:
            self.setObjectName("tab_llm_chat")
        except Exception:
            pass

# -----------------------------
# entry point
# -----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LlamaChatWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
