# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import socket
import sys
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


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


def _chat_dir(fv_root: str) -> str:
    return os.path.join(fv_root, "data", "llama_chat", "chats")


def _attachment_temp_dir(fv_root: str) -> str:
    return os.path.join(fv_root, "data", "llama_chat", "attachments")


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
                if low.startswith("mmproj-") or ".mmproj" in low:
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


def _find_mmproj_for_model(model_path: str) -> str:
    model_path = os.path.abspath(model_path or "")
    if not model_path or not os.path.isfile(model_path):
        return ""

    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).lower()
    parent_dir = os.path.dirname(model_dir)
    candidate_dirs: List[str] = []
    for d in [model_dir, os.path.join(model_dir, "mmproj"), os.path.join(model_dir, "vision"), parent_dir]:
        if d and os.path.isdir(d):
            norm = os.path.normcase(os.path.normpath(d))
            if norm not in {os.path.normcase(os.path.normpath(x)) for x in candidate_dirs}:
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
            if os.path.normcase(os.path.normpath(os.path.join(folder, fn))) == os.path.normcase(os.path.normpath(model_path)):
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
            if any(k in model_name for k in ("vision", "vl", "glm-4.1v", "glm-4v", "glm4v", "llava", "bakllava", "minicpm-v", "qwen2-vl", "qwen-vl", "internvl", "moondream")):
                score += 10
            if score > 0:
                scored.append((score, os.path.join(folder, fn)))

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
TEXT_EXTS = {".txt", ".md", ".py", ".js", ".ts", ".json", ".html", ".css", ".xml", ".yaml", ".yml", ".ini", ".cfg", ".csv", ".log", ".bat", ".ps1", ".cpp", ".c", ".h", ".hpp", ".java", ".rs", ".go", ".php", ".sql"}
MAX_TEXT_ATTACHMENT_CHARS = 24000
MAX_IMAGE_INLINE_BYTES = 6 * 1024 * 1024

def _attachment_kind_from_path(path: str) -> str:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
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
    prefix = "🖼" if kind == "image" else "📄" if kind == "text" else "📎"
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


class MessageBubble(QtWidgets.QFrame):
    def __init__(self, role: str, text: str, thinking: str = "", attachments: Optional[List[Dict]] = None, parent=None):
        super().__init__(parent)
        attachments = attachments or []
        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)

        bubble = QtWidgets.QFrame()
        bubble.setObjectName(f"Bubble_{role}")
        bubble_lay = QtWidgets.QVBoxLayout(bubble)
        bubble_lay.setContentsMargins(14, 10, 14, 10)
        bubble_lay.setSpacing(8)

        role_lbl = QtWidgets.QLabel("You" if role == "user" else ("Assistant" if role == "assistant" else "Info"))
        role_lbl.setObjectName("BubbleRole")
        bubble_lay.addWidget(role_lbl)

        if attachments:
            att_frame = QtWidgets.QFrame()
            att_frame.setObjectName("AttachmentFrame")
            att_lay = QtWidgets.QVBoxLayout(att_frame)
            att_lay.setContentsMargins(10, 8, 10, 8)
            att_lay.setSpacing(4)
            att_title = QtWidgets.QLabel("Attachments")
            att_title.setObjectName("AttachmentTitle")
            att_lay.addWidget(att_title)
            for line in _attachment_summary_lines(attachments):
                lbl = QtWidgets.QLabel(line)
                lbl.setObjectName("AttachmentText")
                lbl.setWordWrap(True)
                lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                att_lay.addWidget(lbl)
            bubble_lay.addWidget(att_frame)

        if role == "assistant" and (thinking or "").strip():
            think_frame = QtWidgets.QFrame()
            think_frame.setObjectName("ThinkingFrame")
            think_lay = QtWidgets.QVBoxLayout(think_frame)
            think_lay.setContentsMargins(10, 8, 10, 8)
            think_lay.setSpacing(4)

            think_title = QtWidgets.QLabel("Thinking")
            think_title.setObjectName("ThinkingTitle")
            think_lay.addWidget(think_title)

            think_lbl = QtWidgets.QLabel((thinking or "").strip())
            think_lbl.setObjectName("ThinkingText")
            think_lbl.setWordWrap(True)
            think_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            think_lay.addWidget(think_lbl)
            bubble_lay.addWidget(think_frame)

        if text:
            text_lbl = QtWidgets.QLabel(text)
            if role == "assistant":
                text_lbl.setObjectName("AnswerText")
            text_lbl.setWordWrap(True)
            text_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            bubble_lay.addWidget(text_lbl)

        if role == "user":
            outer.addStretch(1)
            outer.addWidget(bubble, 0)
            bubble.setMaximumWidth(700)
        elif role == "assistant":
            outer.addWidget(bubble, 0)
            outer.addStretch(1)
            bubble.setMaximumWidth(860)
        else:
            outer.addStretch(1)
            outer.addWidget(bubble, 0)
            outer.addStretch(1)
            bubble.setMaximumWidth(860)


class ChatScrollArea(QtWidgets.QScrollArea):
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

    def clear_messages(self):
        while self.vbox.count() > 1:
            item = self.vbox.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def add_message(self, role: str, text: str, thinking: str = "", attachments: Optional[List[Dict]] = None):
        self.vbox.insertWidget(self.vbox.count() - 1, MessageBubble(role, text, thinking, attachments))
        self.scroll_to_bottom(force=True)
        QtCore.QTimer.singleShot(0, lambda: self.scroll_to_bottom(force=True))
        QtCore.QTimer.singleShot(35, lambda: self.scroll_to_bottom(force=True))
        QtCore.QTimer.singleShot(120, lambda: self.scroll_to_bottom(force=True))

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

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(12)

        form = QtWidgets.QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        self.ed_runner = QtWidgets.QLineEdit()
        self.btn_browse_runner = QtWidgets.QPushButton("Browse…")
        self.cmb_model = QtWidgets.QComboBox()
        self.btn_refresh_models = QtWidgets.QPushButton("Refresh")
        self.ed_model_root = QtWidgets.QLineEdit()
        self.ed_model_root.setPlaceholderText(os.path.join("models", "llama"))
        self.btn_browse_model_root = QtWidgets.QPushButton("Browse…")
        self.cmb_template = QtWidgets.QComboBox()

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

        r = 0
        form.addWidget(QtWidgets.QLabel("Runner"), r, 0)
        form.addWidget(self.ed_runner, r, 1)
        form.addWidget(self.btn_browse_runner, r, 2)
        r += 1

        form.addWidget(QtWidgets.QLabel("Model"), r, 0)
        form.addWidget(self.cmb_model, r, 1)
        form.addWidget(self.btn_refresh_models, r, 2)
        r += 1

        form.addWidget(QtWidgets.QLabel("Model folder"), r, 0)
        form.addWidget(self.ed_model_root, r, 1)
        form.addWidget(self.btn_browse_model_root, r, 2)
        r += 1

        form.addWidget(QtWidgets.QLabel("Template"), r, 0)
        form.addWidget(self.cmb_template, r, 1, 1, 2)
        r += 1

        form.addWidget(QtWidgets.QLabel("Context size"), r, 0)
        form.addWidget(self.sp_ctx_size, r, 1, 1, 2)
        r += 1

        form.addWidget(QtWidgets.QLabel("Max tokens"), r, 0)
        form.addWidget(self.sp_max_tokens, r, 1, 1, 2)
        r += 1

        form.addWidget(QtWidgets.QLabel("Temperature"), r, 0)
        form.addWidget(self.sp_temp, r, 1, 1, 2)
        r += 1

        form.addWidget(QtWidgets.QLabel("Top-p"), r, 0)
        form.addWidget(self.sp_top_p, r, 1, 1, 2)

        lay.addLayout(form)
        lay.addWidget(QtWidgets.QLabel("System prompt"))
        lay.addWidget(self.ed_system)
        lay.addWidget(QtWidgets.QLabel("Template folder"))
        lay.addWidget(self.lbl_templates_dir)

        hint = QtWidgets.QLabel(
            "Tip: pick llama-server.exe directly when possible. If you pick another llama.cpp binary, this UI will also look for a sibling llama-server executable in the same folder."
        )
        hint.setWordWrap(True)
        hint.setObjectName("SubtleLabel")
        lay.addWidget(hint)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        lay.addWidget(btns)
        btns.rejected.connect(self.reject)

        self.btn_browse_runner.clicked.connect(self._browse_runner)
        self.btn_browse_model_root.clicked.connect(self._browse_model_root)
        self.btn_refresh_models.clicked.connect(self.refresh_models)
        self.ed_model_root.editingFinished.connect(self._on_model_root_edited)

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
    def __init__(self):
        super().__init__()
        self.fv_root = _find_fv_root()
        self.chat_dir = _chat_dir(self.fv_root)
        self.attachment_temp_dir = _attachment_temp_dir(self.fv_root)
        self.settings_path = _settings_path(self.fv_root)

        os.makedirs(self.chat_dir, exist_ok=True)
        os.makedirs(self.attachment_temp_dir, exist_ok=True)

        self.sessions: List[ChatSession] = []
        self.current_session_id: Optional[str] = None
        self._saved_model_path = ""
        self._saved_template = ("auto", "")

        self.server_process: Optional[QtCore.QProcess] = None
        self.server_boot_thread: Optional[ServerBootThread] = None
        self.chat_thread: Optional[ChatCompletionThread] = None
        self.server_port: Optional[int] = None
        self.server_url: str = ""
        self.server_ready: bool = False
        self.loaded_model_path: str = ""
        self.loaded_template: Tuple[str, str] = ("auto", "")
        self.server_log_tail: List[str] = []
        self.pending_generate_session_id: str = ""
        self.pending_retry_generation: bool = False
        self._last_user_text_sent: str = ""
        self._auto_retry_templates: List[Tuple[str, str]] = []
        self._auto_retry_original_selection: Tuple[str, str] = ("auto", "")
        self.effective_template_on_server: Tuple[str, str] = ("auto", "")
        self.pending_attachments: List[Dict[str, Any]] = []

        self._syncing_model_selectors = False
        self._loading_session_state = False
        self._last_ui_model_path = ""

        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._save_all)

        self.setWindowTitle("FrameVision - Llama Chat UI")
        self.resize(1480, 900)

        self._build_ui()
        self._apply_style()
        self._load_settings()
        self._load_sessions()
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
        self.ed_prompt.setFixedHeight(96)

        self.lst_attachments = QtWidgets.QListWidget()
        self.lst_attachments.setObjectName("AttachmentList")
        self.lst_attachments.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.lst_attachments.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.lst_attachments.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
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

        m_lay.addWidget(topbar, 0)
        m_lay.addWidget(self.chat_view, 1)
        m_lay.addWidget(composer_wrap, 0)

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
        self.btn_load.clicked.connect(self._load_selected_model)
        self.btn_unload.clicked.connect(self._unload_model)
        self.cmb_model_quick.currentIndexChanged.connect(self._quick_model_changed)
        self.settings_dialog.btn_refresh_models.clicked.connect(self._sync_models_from_settings)
        self.btn_attach.clicked.connect(self._pick_attachments)
        self.lst_attachments.customContextMenuRequested.connect(self._attachment_context_menu)
        self._refresh_attachment_list()

    def _apply_style(self):
        self.setStyleSheet("""
        QMainWindow, QWidget {
            background: #16181d;
            color: #e8edf5;
            font-size: 13px;
        }
        #Sidebar {
            background: #171920;
            border-right: 1px solid #252a36;
        }
        #TopBar {
            background: #16181d;
            border-bottom: 1px solid #232836;
        }
        #ComposerFrame {
            background: #21242d;
            border: 1px solid #323848;
            border-radius: 18px;
        }
        #AttachmentList {
            background: #1b2029;
            border: 1px solid #31384a;
            border-radius: 10px;
            padding: 4px;
        }
        #AttachmentList::item {
            padding: 6px 8px;
            border-radius: 8px;
        }
        #AttachmentList::item:selected {
            background: #353d4f;
        }
        #ChatList {
            background: #1a1d25;
            border: 1px solid #2c3342;
            border-radius: 12px;
            padding: 4px;
        }
        #SidebarTitle {
            font-size: 21px;
            font-weight: 600;
        }
        #ChatHeader {
            font-size: 17px;
            font-weight: 600;
        }
        #SubtleLabel, #BubbleRole {
            color: #9aa3b3;
        }
        QLineEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background: #1b2029;
            color: #eff3f9;
            border: 1px solid #31384a;
            border-radius: 10px;
            padding: 8px;
        }
        QListWidget::item {
            padding: 10px 10px;
            border-radius: 8px;
            margin: 2px 0px;
        }
        QListWidget::item:selected {
            background: #353d4f;
            color: white;
        }
        QListWidget::item:hover {
            background: #2a3140;
        }
        QPushButton {
            background: #2a3140;
            color: #eff3f9;
            border: 1px solid #3b4456;
            border-radius: 10px;
            padding: 9px 14px;
        }
        QPushButton:hover {
            background: #34405a;
        }
        QPushButton:disabled {
            color: #868fa0;
            background: #1d2230;
        }
        QScrollArea {
            border: none;
            background: #16181d;
        }
        QFrame#Bubble_user {
            background: #33425f;
            border-radius: 14px;
        }
        QFrame#Bubble_assistant {
            background: #212734;
            border-radius: 14px;
        }
        QFrame#ThinkingFrame {
            background: #1b2030;
            border: 1px solid #2f3850;
            border-radius: 10px;
        }
        QFrame#AttachmentFrame {
            background: #202634;
            border: 1px solid #33405a;
            border-radius: 10px;
        }
        QLabel#AttachmentTitle {
            color: #c4d0e7;
            font-weight: 600;
        }
        QLabel#AttachmentText {
            color: #cdd7e8;
        }
        QLabel#ThinkingTitle {
            color: #aeb7c8;
            font-weight: 600;
        }
        QLabel#ThinkingText {
            color: #9aa3b3;
        }
        QLabel#AnswerText {
            color: #edf2fa;
        }
        QFrame#Bubble_info {
            background: #2d3127;
            border-radius: 14px;
        }
        QLabel#StatusIdle, QLabel#StatusLoading, QLabel#StatusReady, QLabel#StatusError {
            border-radius: 12px;
            padding: 6px 10px;
            font-weight: 600;
        }
        QLabel#StatusIdle {
            background: #252b38;
            color: #cfd7e6;
        }
        QLabel#StatusLoading {
            background: #5a4727;
            color: #ffdf9c;
        }
        QLabel#StatusReady {
            background: #1d4a36;
            color: #9cf0c8;
        }
        QLabel#StatusError {
            background: #532828;
            color: #ffb1b1;
        }
        """)

    # ---------- persistence ----------
    def _load_settings(self):
        data = _load_json(self.settings_path, {})
        if not isinstance(data, dict):
            data = {}
        runner = data.get("runner_path", "")
        if isinstance(runner, str):
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
        self.current_session_id = data.get("last_session_id") if isinstance(data.get("last_session_id"), str) else None
        self._saved_model_path = data.get("model_path", "") if isinstance(data.get("model_path"), str) else ""
        self._saved_template = (data.get("template_kind", "auto"), data.get("template_value", ""))

    def _save_all(self):
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
            "last_session_id": self.current_session_id or "",
        }
        _save_json_atomic(self.settings_path, settings)
        for s in self.sessions:
            _save_json_atomic(os.path.join(self.chat_dir, f"{s.id}.json"), asdict(s))

    def _queue_save(self):
        self._save_timer.start(250)

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
                "Pick a local GGUF model, load it, then start chatting. This version now talks to a real local llama-server instead of returning placeholder text.",
            )
            return
        for m in s.messages:
            role = str(m.get("role", "assistant"))
            if role not in ("user", "assistant", "info"):
                role = "assistant"
            self.chat_view.add_message(role, str(m.get("content", "")), str(m.get("thinking", "")))

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
        self.settings_dialog.show()
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

    def _same_loaded_config(self) -> bool:
        return bool(
            self.server_ready
            and self.loaded_model_path
            and os.path.normcase(os.path.normpath(self.loaded_model_path))
            == os.path.normcase(os.path.normpath(self.current_model_path() or ""))
            and self.loaded_template == self.current_template_choice()
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
            self._set_status(err, "error")
            self.btn_load.setEnabled(True)
            self.btn_unload.setEnabled(False)
            self.btn_send.setEnabled(True)
            self.btn_stop.setEnabled(False)
            return

        self.server_boot_thread = ServerBootThread(self.server_url, 240, self)
        self.server_boot_thread.statusChanged.connect(self._on_boot_status)
        self.server_boot_thread.ready.connect(self._on_server_ready)
        self.server_boot_thread.failed.connect(self._on_server_failed)
        self.server_boot_thread.start()

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
        self._cleanup_boot_thread()
        was_ready = self.server_ready
        self.server_ready = False
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(False)
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if was_ready:
            self._set_status("Unloaded", "idle")
        else:
            tail = self.server_log_tail[-1] if self.server_log_tail else "Local server stopped."
            self._set_status(tail, "error")
        self._update_header()
        self.server_process = None

    def _on_boot_status(self, text: str):
        self._set_status(text or "Loading…", "loading")

    def _on_server_ready(self):
        self.server_ready = True
        self.loaded_model_path = self.current_model_path()
        self.loaded_template = self.current_template_choice()
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(True)
        self.btn_send.setEnabled(True)
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
        tail = self.server_log_tail[-1] if self.server_log_tail else ""
        detail = message if not tail else f"{message}\n\nLast log line: {tail}"
        self._set_status(detail.splitlines()[0], "error")
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(False)
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.pending_generate_session_id = ""

    def _stop_process_only(self):
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
        self.loaded_template = ("auto", "")
        self.effective_template_on_server = ("auto", "")

    def _unload_model(self):
        self.pending_generate_session_id = ""
        self._cleanup_boot_thread()
        self._stop_process_only()
        self.btn_load.setEnabled(True)
        self.btn_unload.setEnabled(False)
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("Unloaded", "idle")
        self._update_header()

    # ---------- chat ----------
    def _build_api_messages(self, session: ChatSession) -> List[Dict]:
        messages: List[Dict] = []
        system_prompt = (session.system_prompt or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
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
        s.messages.append({
            "role": "user",
            "content": text,
            "attachments": attachments,
            "timestamp": now,
        })
        s.updated_at = now
        self.chat_view.add_message("user", text, attachments=attachments)
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


    def _send_message(self):
        text = (self.ed_prompt.toPlainText() or "").strip()
        attachments = list(self.pending_attachments)
        if not text and not attachments:
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
        now = datetime.now().isoformat(timespec="seconds")
        s.messages.append({
            "role": "assistant",
            "content": reply,
            "thinking": thinking,
            "attachments": saved_attachments,
            "timestamp": now,
        })
        s.updated_at = now
        self.chat_view.add_message("assistant", reply, thinking, attachments=saved_attachments)
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
        self.chat_view.add_message("info", f"Generation failed: {msg}")
        if self.server_process and self.server_process.state() != QtCore.QProcess.NotRunning:
            self._set_status("Ready", "ready")
        else:
            self._set_status("Runner stopped", "error")

    def _on_chat_finished(self):
        self.btn_send.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.chat_thread = None

    def _stop_generation(self):
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_view.add_message("info", "Stopping local generation and unloading the current model…")
            self.pending_generate_session_id = ""
            self._cleanup_boot_thread()
            self._stop_process_only()
            self.btn_load.setEnabled(True)
            self.btn_unload.setEnabled(False)
            self.btn_send.setEnabled(True)
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
            item = QtWidgets.QListWidgetItem(_attachment_label(att))
            item.setToolTip(str(att.get("path", "") or att.get("name", "")))
            item.setData(QtCore.Qt.UserRole, dict(att))
            self.lst_attachments.addItem(item)
        self.lst_attachments.setVisible(bool(self.pending_attachments))

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
            "Supported files (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.txt *.md *.py *.js *.ts *.json *.html *.css *.xml *.yaml *.yml *.ini *.cfg *.csv *.log *.bat *.ps1 *.cpp *.c *.h *.hpp *.java *.rs *.go *.php *.sql);;All files (*.*)",
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

    def _attachment_context_menu(self, pos):
        if not self.pending_attachments:
            return
        row = self.lst_attachments.indexAt(pos).row()
        if row >= 0:
            self.lst_attachments.setCurrentRow(row)
        menu = QtWidgets.QMenu(self)
        act_remove = menu.addAction("Remove attachment")
        act_clear = menu.addAction("Clear all attachments")
        chosen = menu.exec(self.lst_attachments.mapToGlobal(pos))
        if chosen == act_remove:
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
# entry point
# -----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    w = LlamaChatWindow()
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
