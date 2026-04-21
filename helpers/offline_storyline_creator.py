from __future__ import annotations

import json
import os
import queue
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Tkinter is required to run this app: {exc}")

APP_NAME = "Offline Storyline Creator"
APP_DIR = Path.home() / ".offline_storyline_creator"
SETTINGS_PATH = APP_DIR / "settings.json"
DEFAULT_OUTPUT_DIR = APP_DIR / "exports"


# -----------------------------
# Data model
# -----------------------------
@dataclass
class StoryProject:
    title: str
    idea: str
    shot_count: int
    story_outline: List[str]
    character_bibles: List[str]
    object_bibles: List[str]
    text_to_image_prompts: List[str]
    image_to_video_prompts: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------
# Local llama-server client
# Based on the same overall flow as the uploaded planner reference:
# - resolve runner/model
# - start local llama-server
# - wait for /health
# - call /v1/chat/completions
# -----------------------------
class LocalLlamaClient:
    def __init__(self, runner_path: str, model_path: str, ctx_size: int = 8192, top_p: float = 0.9):
        self.runner_path = self._resolve_server_executable(runner_path)
        self.model_path = os.path.abspath(model_path.strip()) if model_path else ""
        self.ctx_size = int(ctx_size)
        self.top_p = float(top_p)
        self.proc: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.base_url: Optional[str] = None
        self.log_path = APP_DIR / "llama_server.log"

    @staticmethod
    def _resolve_server_executable(path: str) -> str:
        raw = os.path.abspath(str(path or "").strip())
        if not raw:
            return ""
        base = os.path.basename(raw).lower()
        if "server" in base:
            return raw
        folder = os.path.dirname(raw)
        for name in ("llama-server.exe", "llama-server", "server.exe", "server"):
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate):
                return candidate
        return raw

    @staticmethod
    def _pick_free_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
        sock.close()
        return port

    @staticmethod
    def _http_get_json(url: str, timeout: float = 8.0) -> Tuple[int, Dict[str, Any]]:
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return int(resp.getcode()), json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw) if raw.strip() else {}
            except Exception:
                data = {"error": {"message": raw or str(exc)}}
            return int(exc.code), data

    @staticmethod
    def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 300.0) -> Tuple[int, Dict[str, Any]]:
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
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(raw) if raw.strip() else {}
            except Exception:
                data = {"error": {"message": raw or str(exc)}}
            return int(exc.code), data

    @staticmethod
    def _extract_message_text(message: Any) -> str:
        if not isinstance(message, dict):
            return ""
        content = message.get("content", "")
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text", item.get("content", ""))
                    if txt:
                        parts.append(str(txt))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(p for p in parts if p.strip()).strip()
        if isinstance(content, dict):
            return str(content.get("text", content.get("content", "")) or "").strip()
        return str(content or "").strip()

    def start(self) -> None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        if not self.runner_path or not os.path.isfile(self.runner_path):
            raise RuntimeError(f"llama-server executable not found: {self.runner_path or '[empty]'}")
        if not self.model_path or not os.path.isfile(self.model_path):
            raise RuntimeError(f"GGUF model not found: {self.model_path or '[empty]'}")
        if self.proc and self.proc.poll() is None:
            return

        self.port = self._pick_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"

        args = [
            self.runner_path,
            "-m", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.ctx_size),
            "--reasoning-budget", "0",
        ]

        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        with open(self.log_path, "w", encoding="utf-8", errors="replace") as fh:
            fh.write("[offline_storyline_creator]\n")
            fh.write(f"runner={self.runner_path}\n")
            fh.write(f"model={self.model_path}\n")
            fh.write(f"args={json.dumps(args, ensure_ascii=False)}\n\n")

        log_handle = open(self.log_path, "a", encoding="utf-8", errors="replace")
        try:
            self.proc = subprocess.Popen(
                args,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(self.runner_path),
                creationflags=creationflags,
            )
        finally:
            log_handle.close()

        start = time.time()
        last_status = "Starting local llama-server..."
        while time.time() - start <= 240:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited before becoming ready. Check {self.log_path}."
                )
            try:
                code, payload = self._http_get_json(f"{self.base_url}/health", timeout=4.0)
                if code == 200:
                    return
                if code == 503:
                    last_status = str(((payload or {}).get("error") or {}).get("message") or "Loading model...")
                else:
                    last_status = f"Waiting for server... ({code})"
            except Exception:
                pass
            time.sleep(1.0)

        raise RuntimeError(f"Timed out waiting for llama-server. Last status: {last_status}")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=8)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        finally:
            self.proc = None

    def generate(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.6, max_tokens: int = 4096) -> str:
        if not self.base_url or not self.proc or self.proc.poll() is not None:
            self.start()

        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(self.top_p),
            "reasoning_format": "none",
        }
        code, data = self._http_post_json(f"{self.base_url}/v1/chat/completions", payload)
        if code >= 400:
            msg = ((data or {}).get("error") or {}).get("message") or f"HTTP {code}"
            raise RuntimeError(str(msg))
        choices = (data or {}).get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by llama-server.")
        message = choices[0].get("message") or {}
        text = self._extract_message_text(message)
        return text.strip()


# -----------------------------
# Story pipeline
# -----------------------------
class StorylineGenerator:
    def __init__(self, client: LocalLlamaClient, log_callback: Optional[Callable[[str], None]] = None):
        self.client = client
        self.log_callback = log_callback

    def _log(self, text: str) -> None:
        if callable(self.log_callback):
            try:
                self.log_callback(str(text))
            except Exception:
                pass

    @staticmethod
    def _format_lines_for_log(title: str, lines: List[str]) -> str:
        if not lines:
            return f"{title}: [empty]"
        body = "\n".join(f"[{idx+1:02d}] {line}" for idx, line in enumerate(lines))
        return f"{title}:\n{body}"

    @staticmethod
    def _extract_numbered_lines(text: str, expected_count: int) -> List[str]:
        raw_text = str(text or "").replace("\r", "").strip()
        if not raw_text:
            raise RuntimeError(f"Model returned 0 usable items, expected {expected_count}.\n\nRaw output:\n{text}")

        lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
        prompts: List[str] = []
        line_rx = re.compile(r"^\s*(?:\[?\(?)(\d{1,3})(?:\]?\)?)\s*[:.\-\]]?\s*(.+)$")

        for line in lines:
            m = line_rx.match(line)
            if m:
                prompts.append(m.group(2).strip())

        if len(prompts) >= expected_count:
            return prompts[:expected_count]

        block_rx = re.compile(
            r"(?:^|[\n\t ])(?:\[?\(?)(\d{1,3})(?:\]?\)?)\s*[:.\-\]]?\s*(.+?)(?=(?:[\n\t ]+(?:\[?\(?\d{1,3}(?:\]?\)?))\s*[:.\-\]]?)|$)",
            re.DOTALL,
        )
        block_matches = [m.group(2).strip() for m in block_rx.finditer(raw_text) if m.group(2).strip()]
        if len(block_matches) >= expected_count:
            return block_matches[:expected_count]

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        if len(paragraphs) >= expected_count:
            return paragraphs[:expected_count]

        if len(lines) >= expected_count:
            cleaned = [line_rx.sub(r"\2", ln).strip() for ln in lines]
            return cleaned[:expected_count]

        usable_count = max(len(prompts), len(block_matches), len(paragraphs), len(lines))
        raise RuntimeError(
            f"Model returned {usable_count} usable items, expected {expected_count}.\n\nRaw output:\n{text}"
        )

    @staticmethod
    def _clean_style_hint(style_hint: str) -> str:
        raw = re.sub(r"\s+", " ", str(style_hint or "").strip())
        if not raw:
            return ""
        low = raw.lower()
        if "pixar" in low and "style" not in low:
            if any(tok in low for tok in ("animated", "animation", "3d", "rendered", "render")):
                return "Pixar animated style"
            return "Pixar style"
        return raw

    @staticmethod
    def _normalize_prompt_text(text: str) -> str:
        s = re.sub(r"\s+", " ", str(text or "").strip())
        s = re.sub(r"\s+([,.;:!?])", r"\1", s)
        s = re.sub(r"([,.;:!?]){2,}", lambda m: m.group(1), s)
        return s.strip(" ,")

    @staticmethod
    def _style_equivalent_already_present(base_text: str, style_hint: str) -> bool:
        base_low = str(base_text or "").lower()
        style_low = str(style_hint or "").lower().strip()
        if not base_low or not style_low:
            return False
        if style_low in base_low:
            return True

        # Smarter dedupe for common Pixar variants so we do not append
        # ", pixar style" to lines that already say things like
        # "Pixar-style illustration" or "Pixar animation style".
        if "pixar" in style_low:
            pixar_markers = (
                "pixar style",
                "pixar-style",
                "pixar animated style",
                "pixar animation style",
                "pixar animated",
                "pixar animation",
                "pixar-style animation",
                "pixar-style animated",
                "pixar-style illustration",
                "in pixar style",
                "rendered in pixar",
            )
            if any(marker in base_low for marker in pixar_markers):
                return True
        return False

    @staticmethod
    def _apply_style_to_prompt(prompt: str, style_hint: str) -> str:
        base = StorylineGenerator._normalize_prompt_text(prompt)
        style = StorylineGenerator._normalize_prompt_text(style_hint)
        if not base or not style:
            return base
        if StorylineGenerator._style_equivalent_already_present(base, style):
            return base
        return f"{base}, {style}"

    def _apply_style_to_prompt_list(self, prompts: List[str], style_hint: str) -> List[str]:
        style = self._clean_style_hint(style_hint)
        normalized = [self._normalize_prompt_text(p) for p in prompts]
        if not style:
            return normalized
        return [self._apply_style_to_prompt(p, style) for p in normalized]

    @staticmethod
    def _extract_character_bible_lines(text: str) -> List[str]:
        lines = [ln.strip() for ln in text.replace("\r", "").split("\n") if ln.strip()]
        cleaned: List[str] = []
        rx = re.compile(r"^\s*(?:[-*•]|\[?\(?\d{1,3}\]?\)?[.:\-]?)\s*(.+)$")
        seen: set[str] = set()
        for line in lines:
            body = line
            m = rx.match(line)
            if m:
                body = m.group(1).strip()
            body = re.sub(r"\s+", " ", body).strip(" ,")
            if not body or "(" not in body or ")" not in body:
                continue
            key = body.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(body)
        return cleaned

    @staticmethod
    def _split_character_bible_entry(entry: str) -> Tuple[str, str]:
        m = re.match(r"^\s*(.+?)\s*\((.+)\)\s*$", str(entry or "").strip())
        if not m:
            return "", ""
        label = re.sub(r"\s+", " ", m.group(1)).strip(" ,")
        detail = re.sub(r"\s+", " ", m.group(2)).strip(" ,")
        return label, detail

    @staticmethod
    def _normalize_character_label(label: str) -> str:
        raw = re.sub(r"\s+", " ", str(label or "").strip(" ,"))
        if not raw:
            return ""
        low = raw.lower()
        if low.startswith(("his ", "her ", "their ", "its ", "my ", "our ")):
            return low
        if raw[0].isupper() and " " not in raw and "'" not in raw:
            return raw
        if low.startswith(("the ", "a ", "an ")):
            words = low.split()
            if words[0] != "the":
                low = "the " + " ".join(words[1:])
            return low
        return "the " + low

    @staticmethod
    def _looks_like_character_label(label: str, detail: str) -> bool:
        low = f"{label} {detail}".lower()
        object_terms = {
            "slab", "rock", "rubble", "briefcase", "bag", "cape", "sword", "gun", "car", "truck", "bench",
            "lamppost", "jewel", "jewels", "building", "tower", "door", "window", "helmet", "mask", "concrete",
            "fire", "explosion", "smoke", "road", "street", "chair", "table", "weapon", "stone", "crystal",
            "artifact", "device", "machine", "vehicle", "ship"
        }
        living_terms = {
            "man", "men", "woman", "women", "boy", "girl", "child", "kid", "teen", "adult", "person", "people",
            "citizen", "citizens", "hero", "superhero", "villain", "henchman", "dog", "cat", "alien", "creature",
            "monster", "dragon", "bird", "wolf", "bear", "fox", "rabbit", "horse", "cow", "baby", "mother",
            "father", "king", "queen", "soldier", "guard", "driver", "pilot", "wizard", "witch", "robot", "android",
            "detective", "teacher", "student", "sidekick", "giant", "elf", "orc", "demon", "angel"
        }
        if any(term in low for term in living_terms):
            return True
        if any(term in low for term in object_terms):
            return False
        return False

    @classmethod
    def _clean_character_bible_entries(cls, entries: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()
        for entry in entries:
            label, detail = cls._split_character_bible_entry(entry)
            if not label or not detail:
                continue
            if not cls._looks_like_character_label(label, detail):
                continue
            norm_label = cls._normalize_character_label(label)
            norm_detail = cls._clean_bible_detail(detail, kind="character")
            if not norm_detail:
                continue
            full = f"{norm_label} ({norm_detail})"
            key = full.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(full)
        return cleaned


    @staticmethod
    def _character_detail_category_flags(label: str, detail: str) -> Dict[str, bool]:
        low = f"{label} {detail}".lower()
        detail_low = str(detail or '').lower()
        face_terms = (
            'face', 'facial', 'eyes', 'eye ', 'eyebrow', 'eyebrows', 'lashes', 'cheek', 'cheekbones', 'jaw', 'jawline',
            'nose', 'lips', 'mouth', 'chin', 'forehead', 'freckles', 'scar', 'wrinkle', 'skin', 'oval face', 'round face',
            'square face', 'heart-shaped face', 'almond-shaped eyes'
        )
        hair_terms = (
            'hair', 'hairstyle', 'hairline', 'bangs', 'fringe', 'ponytail', 'bun', 'braid', 'braids', 'curl', 'curly',
            'wavy', 'straight hair', 'short hair', 'long hair', 'bob cut', 'pixie cut', 'beard', 'mustache', 'moustache',
            'clean-shaven', 'sideburns', 'fur', 'mane', 'coat pattern'
        )
        clothing_terms = (
            'jacket', 'coat', 'hoodie', 'shirt', 't-shirt', 'blouse', 'dress', 'skirt', 'jeans', 'pants', 'trousers',
            'boots', 'shoes', 'sneakers', 'scarf', 'hat', 'cap', 'gloves', 'armor', 'armour', 'uniform', 'robe', 'suit',
            'vest', 'sweater', 'cardigan', 'kimono', 'cape', 'apron', 'necklace', 'earrings', 'bracelet', 'belt'
        )
        species_terms = ('dog', 'cat', 'alien', 'creature', 'monster', 'dragon', 'wolf', 'bear', 'fox', 'rabbit', 'horse', 'bird')
        is_humanish = not any(term in low for term in species_terms)
        return {
            'face': any(term in detail_low for term in face_terms),
            'hair': any(term in detail_low for term in hair_terms),
            'clothing': any(term in detail_low for term in clothing_terms),
            'humanish': is_humanish,
        }

    @classmethod
    def _character_entry_needs_strengthening(cls, entry: str) -> bool:
        label, detail = cls._split_character_bible_entry(entry)
        if not label or not detail:
            return False
        flags = cls._character_detail_category_flags(label, detail)
        comma_parts = [p.strip() for p in re.split(r",|;", detail) if p.strip()]
        if flags['humanish']:
            score = int(flags['face']) + int(flags['hair']) + int(flags['clothing'])
            if score < 3:
                return True
            if len(comma_parts) < 5:
                return True
            return False
        # for non-human characters, still require at least body/fur/pattern + one style anchor
        if len(comma_parts) < 4:
            return True
        return not (flags['hair'] or flags['clothing'] or flags['face'])

    def _strengthen_character_bibles(self, *, entries: List[str], idea: str, story_outline: List[str], draft_prompts: List[str], shot_count: int, style_hint: str, t2i_model_hint: str) -> List[str]:
        cleaned = self._clean_character_bible_entries(entries)
        if not cleaned:
            return []
        weak = [entry for entry in cleaned if self._character_entry_needs_strengthening(entry)]
        if not weak:
            return cleaned
        self._log(f"Strengthening {len(weak)} weak character bible entr{'y' if len(weak)==1 else 'ies'}...")
        source_parts: List[str] = []
        if story_outline:
            source_parts.append("Story beats:\n" + "\n".join(f"{idx+1}. {beat}" for idx, beat in enumerate(story_outline)))
        if draft_prompts:
            source_parts.append("Draft text-to-image prompts:\n" + "\n".join(f"{idx+1}. {prompt}" for idx, prompt in enumerate(draft_prompts)))
        source_block = "\n\n".join(source_parts) if source_parts else f"User idea:\n{idea.strip()}"
        weak_block = "\n".join(f"- {entry}" for entry in weak)
        rewrite_prompt = f"""
Rewrite only the weak character bible entries below so they become strong visual identity anchors for text-to-image consistency.

{source_block}

Weak character bible entries:
{weak_block}

Global style hint: {style_hint or 'none'}
Target text-to-image model: {t2i_model_hint or 'none'}

Rules:
- Return exactly {len(weak)} rewritten character bible lines.
- Keep the same character label for each rewritten line.
- Keep each line in this exact format: Name or label (detailed visual identity)
- For human or human-like characters, the identity inside parentheses must include all three:
  1) facial identity and face structure
  2) hairstyle or facial hair
  3) clothing style or outfit anchors
- Good facial identity details include: face shape, eyes, eyebrows, nose, lips, jawline, cheekbones, skin details, freckles, scars, wrinkles.
- Good hair details include: hair length, hairstyle, color, texture, beard, mustache, clean-shaven.
- Good clothing details include: jacket, shirt, dress, trousers, boots, accessories, colors, materials, signature outfit pieces.
- Do not write vague filler like handsome face, pretty face, casual clothes, nice outfit.
- Do not include actions, poses, camera framing, emotion, or location.
- Keep the label stable and natural.
- Return only the rewritten character bible lines.
""".strip()
        raw = self.client.generate(
            system_prompt=(
                "You are an offline story and prompt engine. "
                "Follow the requested format exactly. "
                "Do not output chain-of-thought, thinking, explanations, JSON, markdown fences, or commentary. "
                "Return only the requested lines."
            ),
            user_prompt=rewrite_prompt,
            temperature=0.35,
            max_tokens=max(500, len(weak) * 180),
        )
        rewritten = self._clean_character_bible_entries(self._extract_character_bible_lines(raw))
        by_label = {}
        for entry in rewritten:
            label, detail = self._split_character_bible_entry(entry)
            if label and detail:
                by_label[self._normalize_character_label(label)] = entry
        merged = []
        for entry in cleaned:
            label, detail = self._split_character_bible_entry(entry)
            key = self._normalize_character_label(label)
            replacement = by_label.get(key)
            if replacement and not self._character_entry_needs_strengthening(replacement):
                merged.append(replacement)
            else:
                merged.append(entry)
        return self._clean_character_bible_entries(merged)
    @staticmethod
    def _looks_like_object_label(label: str, detail: str) -> bool:
        low = f"{label} {detail}".lower()
        recurring_object_terms = {
            "guitar", "sword", "helmet", "mask", "briefcase", "backpack", "bag", "camera", "book", "journal",
            "necklace", "ring", "amulet", "artifact", "orb", "crystal", "staff", "wand", "microphone", "coffee cup",
            "mug", "bicycle", "bike", "motorcycle", "car", "truck", "van", "ship", "spaceship", "robot", "drone",
            "doll", "toy", "key", "watch", "communicator", "phone", "tablet", "laptop", "lantern", "umbrella",
            "suitcase", "case", "crown", "shield", "hammer", "glasses", "sneakers", "boots"
        }
        background_terms = {
            "rock", "rubble", "bench", "lamppost", "building", "tower", "road", "street", "window", "door",
            "wall", "floor", "ground", "smoke", "fire", "explosion", "sky", "cloud", "tree", "grass", "chair",
            "table", "box", "boxes", "cardboard", "concrete", "bridge", "cart", "stall", "vendor cart", "city"
        }
        if any(term in low for term in background_terms):
            return False
        return any(term in low for term in recurring_object_terms)

    @staticmethod
    def _clean_bible_detail(detail: str, kind: str) -> str:
        raw = re.sub(r"\s+", " ", str(detail or "")).strip(" ,")
        if not raw:
            return ""
        parts = [p.strip(" ,") for p in re.split(r",|;", raw) if p.strip(" ,")]
        if not parts:
            return ""
        banned_common = {
            "in the background", "background", "shown as", "close-up", "wide shot", "low angle", "high angle",
            "silhouette", "against the", "looking up", "looking at", "watching", "pointing", "running",
            "walking", "sitting", "standing", "floating", "drifting", "holding", "firing", "mid-yawn",
            "on the surface", "on the moon", "at the horizon", "in the sky", "earth", "horizon", "cratered",
        }
        kept: List[str] = []
        for part in parts:
            low = part.lower()
            if any(term in low for term in banned_common):
                continue
            if kind == "character":
                if low.startswith(("wearing a ", "wearing an ")):
                    part = re.sub(r"^wearing\s+", "", part, flags=re.I)
                if low.startswith(("shown ", "looking ", "pointing ", "standing ", "sitting ", "running ", "walking ")):
                    continue
            kept.append(part)
        return ", ".join(kept[:6]).strip(" ,")

    @classmethod
    def _clean_object_bible_entries(cls, entries: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen: set[str] = set()
        for entry in entries:
            label, detail = cls._split_character_bible_entry(entry)
            if not label or not detail:
                continue
            norm_label = re.sub(r"\s+", " ", label).strip(" ,").lower()
            norm_detail = cls._clean_bible_detail(detail, kind="object")
            if not norm_label or not norm_detail:
                continue
            if not cls._looks_like_object_label(norm_label, norm_detail):
                continue
            full = f"{norm_label} ({norm_detail})"
            key = full.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(full)
        return cleaned

    @staticmethod
    def _strip_leading_article(text: str) -> str:
        return re.sub(r"^(?:the|a|an)\s+", "", str(text or "").strip(), flags=re.I).strip()

    @classmethod
    def _infer_character_roles(cls, label: str, detail: str) -> set[str]:
        low = f"{label} {detail}".lower()
        roles: set[str] = set()
        role_groups = {
            "man": {" man", "male", "gentleman", "husband", "father", "king", "boyfriend"},
            "woman": {" woman", "female", "lady", "wife", "mother", "queen", "girlfriend"},
            "boy": {" boy", "young boy", "little boy"},
            "girl": {" girl", "young girl", "little girl"},
            "dog": {" dog", "puppy", "hound", "canine"},
            "cat": {" cat", "kitten", "feline"},
            "alien": {" alien", "extraterrestrial"},
            "robot": {" robot", "android", "machine being"},
        }
        padded = f" {low} "
        for role, markers in role_groups.items():
            if any(marker in padded for marker in markers):
                roles.add(role)
        return roles

    @classmethod
    def _character_entry_variants(cls, label: str, detail: str) -> List[str]:
        raw_label = re.sub(r"\s+", " ", str(label or "").strip(" ,"))
        if not raw_label:
            return []
        stripped = cls._strip_leading_article(raw_label)
        variants = [raw_label]
        low = raw_label.lower()
        if stripped and stripped.lower() != low:
            variants.append(stripped)
        if stripped:
            variants.extend([
                f"the {stripped}",
                f"a {stripped}",
                f"an {stripped}",
            ])
        for role in cls._infer_character_roles(label, detail):
            if role == "man":
                variants.extend(["the man", "a man", "man"])
            elif role == "woman":
                variants.extend(["the woman", "a woman", "woman"])
            elif role == "boy":
                variants.extend(["the boy", "a boy", "boy"])
            elif role == "girl":
                variants.extend(["the girl", "a girl", "girl"])
            elif role == "dog":
                variants.extend(["the dog", "a dog", "dog"])
            elif role == "cat":
                variants.extend(["the cat", "a cat", "cat"])
            elif role == "alien":
                variants.extend(["the alien", "an alien", "alien"])
            elif role == "robot":
                variants.extend(["the robot", "a robot", "robot"])
        unique: List[str] = []
        seen: set[str] = set()
        for item in variants:
            cleaned = re.sub(r"\s+", " ", str(item or "").strip(" ,")).lower()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique.append(cleaned)
        unique.sort(key=len, reverse=True)
        return unique

    @classmethod
    def _inject_single_character_bible(cls, text: str, label: str, detail: str) -> str:
        updated = str(text or "")
        if not updated.strip() or not label or not detail:
            return updated
        inline = f"{label} ({detail}),"
        inline_low = inline.lower().rstrip(',')
        if inline_low in updated.lower():
            return cls._postprocess_inline_bible_prompt(updated)
        for variant in cls._character_entry_variants(label, detail):
            pattern = re.compile(rf"(?<!\w){re.escape(variant)}(?!\w)", re.I)
            match = pattern.search(updated)
            if not match:
                continue
            replacement = inline
            if variant in {"man", "woman", "boy", "girl", "dog", "cat", "alien", "robot"}:
                prev = updated[max(0, match.start()-4):match.start()].lower()
                if prev.endswith(("a ", "an ", "the ")):
                    continue
            updated = updated[:match.start()] + replacement + updated[match.end():]
            return cls._postprocess_inline_bible_prompt(updated)
        return cls._postprocess_inline_bible_prompt(updated)

    @classmethod
    def _expand_gendered_pair_reference(cls, text: str, character_bibles: List[str]) -> str:
        if len(character_bibles) != 2:
            return text
        parsed = []
        for entry in character_bibles:
            label, detail = cls._split_character_bible_entry(entry)
            if label and detail:
                parsed.append((label, detail, cls._infer_character_roles(label, detail)))
        if len(parsed) != 2:
            return text
        male = next((item for item in parsed if "man" in item[2] or "boy" in item[2]), None)
        female = next((item for item in parsed if "woman" in item[2] or "girl" in item[2]), None)
        if not male or not female:
            return text
        replacement = f"{male[0]} ({male[1]}), and {female[0]} ({female[1]}),"
        patterns = [
            r"\ba man and a woman\b",
            r"\bthe man and the woman\b",
            r"\bman and woman\b",
            r"\bwoman and man\b",
            r"\ba woman and a man\b",
            r"\bthe woman and the man\b",
        ]
        updated = str(text or "")
        for pat in patterns:
            updated2 = re.sub(pat, replacement, updated, count=1, flags=re.I)
            if updated2 != updated:
                return cls._postprocess_inline_bible_prompt(updated2)
        return updated

    @classmethod
    def _force_character_bibles_in_prompts(cls, prompts: List[str], character_bibles: List[str]) -> List[str]:
        if not prompts or not character_bibles:
            return prompts
        cleaned_entries = cls._clean_character_bible_entries(character_bibles)
        if not cleaned_entries:
            return prompts
        pairs: List[Tuple[str, str]] = []
        for entry in cleaned_entries:
            label, detail = cls._split_character_bible_entry(entry)
            if label and detail:
                pairs.append((label, detail))
        forced: List[str] = []
        for prompt in prompts:
            updated = cls._expand_gendered_pair_reference(prompt, cleaned_entries)
            for label, detail in pairs:
                updated = cls._inject_single_character_bible(updated, label, detail)
            forced.append(cls._postprocess_inline_bible_prompt(updated))
        return forced


    @staticmethod
    def _object_label_occurrence_count(label: str, texts: List[str]) -> int:
        norm_label = re.sub(r"\s+", " ", str(label or "").strip().lower())
        if not norm_label:
            return 0
        variants = {norm_label}
        for prefix in ("the ", "a ", "an "):
            if norm_label.startswith(prefix):
                variants.add(norm_label[len(prefix):].strip())
        count = 0
        for txt in texts or []:
            low = re.sub(r"\s+", " ", str(txt or "").lower())
            if not low:
                continue
            if any(re.search(r"\b" + re.escape(v) + r"\b", low) for v in variants if v):
                count += 1
        return count

    @classmethod
    def _filter_object_bibles_by_recurrence(cls, entries: List[str], texts: List[str], min_hits: int = 2) -> List[str]:
        if not entries:
            return []
        kept: List[str] = []
        seen: set[str] = set()
        for entry in entries:
            label, detail = cls._split_character_bible_entry(entry)
            if not label or not detail:
                continue
            if cls._object_label_occurrence_count(label, texts) < min_hits:
                continue
            normalized_label = re.sub(r"\s+", " ", label).strip(" ,").lower()
            full = f"{normalized_label} ({detail})"
            key = full.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(full)
        return kept
    @staticmethod
    def _ensure_inline_bible_commas(text: str) -> str:
        if not text:
            return text
        updated = str(text)
        updated = re.sub(r"([A-Za-z][^.,;:!?()]{0,80})\s*\(([^)]+)\)\s*(['’]s)", lambda m: f"{m.group(1).strip()} ({m.group(2).strip()}),", updated)
        updated = re.sub(r"\b([^.,;:!?()]{1,80}?)\s*\(([^)]+)\)(?!\s*,)", lambda m: f"{m.group(1).strip()} ({m.group(2).strip()}),", updated)
        updated = re.sub(r"\),\s*,+", "), ", updated)
        updated = re.sub(r"\),\s+([.!?])", r")\1", updated)
        updated = re.sub(r"\s{2,}", " ", updated).strip(" ,")
        return updated

    @classmethod
    def _dedupe_inline_bibles_per_prompt(cls, prompts: List[str], bible_entries: List[str]) -> List[str]:
        if not prompts or not bible_entries:
            return prompts
        pairs = []
        for entry in bible_entries:
            label, detail = cls._split_character_bible_entry(entry)
            if label and detail:
                pairs.append((label, detail))
        pairs.sort(key=lambda item: len(item[0]), reverse=True)
        fixed: List[str] = []
        for prompt in prompts:
            updated = cls._ensure_inline_bible_commas(prompt)
            for label, detail in pairs:
                inline_with_comma = f"{label} ({detail}),"
                inline_without_comma = f"{label} ({detail})"
                pattern = re.compile(re.escape(inline_with_comma) + r"|" + re.escape(inline_without_comma), re.IGNORECASE)
                first = True
                def repl(match):
                    nonlocal first
                    if first:
                        first = False
                        return inline_with_comma
                    return label
                updated = pattern.sub(repl, updated)
            updated = cls._ensure_inline_bible_commas(updated)
            fixed.append(updated)
        return fixed

    @staticmethod
    def _postprocess_inline_bible_prompt(prompt: str) -> str:
        text = StorylineGenerator._ensure_inline_bible_commas(prompt)
        text = re.sub(r"\)\s*,\s*'s\s+", "), ", text)
        text = re.sub(r"\)\s*,\s*’s\s+", "), ", text)

        def _drop_stacked_article(match: re.Match) -> str:
            label = match.group(1).strip()
            prefix = match.group(2) or ""
            if prefix:
                return f"{prefix}{label[:1].upper()}{label[1:]}"
            return label

        text = re.sub(
            r"(?i)(?:(?:a|an|the)\s+)(the\s+[A-Za-z][^,.;:!?]{0,120}?\([^)]*\),?)",
            lambda m: _drop_stacked_article(m),
            text,
        )
        text = re.sub(
            r"(^|[.!?]\s+)(the\s+[A-Za-z][^,.;:!?]{0,120}?\([^)]*\),?)",
            lambda m: _drop_stacked_article(m),
            text,
        )
        text = re.sub(r"([Tt]he)\s+", r"", text)
        text = re.sub(r"([Aa]n?)\s+", r"", text)
        text = re.sub(r"\s+,", ",", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ,")
        return text

    @classmethod
    def _expand_known_two_character_groups(cls, prompts: List[str], character_bibles: List[str]) -> List[str]:
        if not prompts or not character_bibles:
            return prompts
        pairs = []
        for entry in character_bibles:
            label, detail = cls._split_character_bible_entry(entry)
            if label and detail:
                pairs.append((label, detail))
        if len(pairs) != 2:
            return prompts
        left = f"{pairs[0][0]} ({pairs[0][1]}), and {pairs[1][0]} ({pairs[1][1]}),"
        starts = ("the ", "a ", "an ")
        vague = {"the people", "the group", "the crowd", "the team", "the family"}
        lowered_labels = {pairs[0][0].lower(), pairs[1][0].lower()}
        if lowered_labels & vague:
            return prompts
        group_patterns = [
            r"the\s+couple",
            r"a\s+couple",
            r"couple",
            r"the\s+pair",
            r"a\s+pair",
            r"pair",
            r"the\s+duo",
            r"a\s+duo",
            r"duo",
        ]
        fixed=[]
        for prompt in prompts:
            updated = str(prompt or "")
            # only expand if the prompt does not already contain both known characters inline
            low = updated.lower()
            if not all(lbl in low for lbl in lowered_labels):
                for pat in group_patterns:
                    updated2 = re.sub(pat, left, updated, count=1, flags=re.I)
                    if updated2 != updated:
                        updated = updated2
                        break
            updated = cls._postprocess_inline_bible_prompt(updated)
            fixed.append(updated)
        return fixed

    def _generate_object_bibles(self, *, idea: str, story_outline: List[str], draft_prompts: List[str], shot_count: int, style_hint: str, t2i_model_hint: str) -> List[str]:
        source_parts: List[str] = []
        if story_outline:
            source_parts.append("Story beats:\n" + "\n".join(f"{idx+1}. {beat}" for idx, beat in enumerate(story_outline)))
        if draft_prompts:
            source_parts.append("Draft text-to-image prompts:\n" + "\n".join(f"{idx+1}. {prompt}" for idx, prompt in enumerate(draft_prompts)))
        source_block = "\n\n".join(source_parts) if source_parts else f"User idea:\n{idea.strip()}"
        bible_prompt = f"""
Create an object bible only for distinct recurring or clearly reused story objects that should stay visually consistent across the story.

{source_block}

Global style hint: {style_hint or 'none'}
Target text-to-image model: {t2i_model_hint or 'none'}

Rules:
- Return one entry per line.
- Include only recurring or reused important objects, props, instruments, vehicles, weapons, accessories, or artifacts that matter visually across multiple beats.
- Good examples: a guitar used through the story, a flying motorcycle, a magic sword, a special helmet, a briefcase, a communicator, a red bicycle.
- Do not create entries for generic scenery or one-off background items such as rocks, rubble, buildings, roads, benches, lampposts, smoke, fire, trees, or random boxes.
- Write each line in this exact format: object label (detailed visual identity)
- Put the descriptive identity inside parentheses immediately after the object label.
- Keep the object label short and stable, such as the guitar, the flying motorcycle, the briefcase, the silver helmet.
- The text inside parentheses must be visual and specific: material, color, shape, size, wear, lights, markings, special parts, and other identifying details.
- Do not write separate explanations, keys, numbers, or summaries.
- Return only the object bible lines.
""".strip()
        raw = self.client.generate(
            system_prompt=(
                "You are an offline story and prompt engine. "
                "Follow the requested format exactly. "
                "Do not output chain-of-thought, thinking, explanations, JSON, markdown fences, or commentary. "
                "Return only the requested lines."
            ),
            user_prompt=bible_prompt,
            temperature=0.4,
            max_tokens=max(800, shot_count * 160),
        )
        entries = self._extract_character_bible_lines(raw)
        cleaned = self._clean_object_bible_entries(entries)
        recurrence_texts = list(story_outline or []) + list(draft_prompts or [])
        return self._filter_object_bibles_by_recurrence(cleaned, recurrence_texts, min_hits=2)

    def _inject_object_bibles_into_t2i(self, *, story_outline: List[str], draft_prompts: List[str], object_bibles: List[str], shot_count: int, style_hint: str, negative_hint: str, t2i_model_hint: str) -> List[str]:
        if not draft_prompts or not object_bibles:
            return draft_prompts
        object_bibles = self._clean_object_bible_entries(object_bibles)
        if not object_bibles:
            return draft_prompts
        beats_block = "\n".join(f"{idx+1}. {beat}" for idx, beat in enumerate(story_outline)) if story_outline else "none"
        draft_block = "\n".join(f"{idx+1}. {draft}" for idx, draft in enumerate(draft_prompts))
        bible_block = "\n".join(f"- {entry}" for entry in object_bibles)
        inject_prompt = f"""
Rewrite the numbered text-to-image prompts below so each prompt injects the matching recurring object bible inline.

Story beats:
{beats_block}

Object bible:
{bible_block}

Draft text-to-image prompts:
{draft_block}

Global style hint: {style_hint or 'none'}
Negative notes to avoid: {negative_hint or 'none'}
Target text-to-image model: {t2i_model_hint or 'none'}

Rules:
- Return exactly {shot_count} numbered prompts.
- Keep each prompt faithful to its matching story beat and draft prompt.
- Use the exact same object label from the object bible every time that object appears.
- When a listed object appears in a prompt, write the label followed immediately by its parenthetical bible and then a comma, for example: the guitar (red electric guitar, white pickguard, worn strap), ...
- Inject each object bible only once per prompt, even if that same object is mentioned more than once.
- Only inject recurring important objects from the object bible. Do not start tagging random background scenery.
- Keep wording natural and rewrite the sentence if needed so the object bible sits cleanly inline.
- Do not convert the prompt into a legend, key, or explanation.
- Do not append a separate object section at the end of any prompt.
- Keep each prompt direct, visual, and usable for text-to-image.
- Return only the numbered prompts.
""".strip()
        prompts = self._generate_numbered_list_with_retry(
            system_prompt=(
                "You are an offline story and prompt engine. "
                "Follow the requested format exactly. "
                "Do not output chain-of-thought, thinking, explanations, JSON, markdown fences, or commentary. "
                "Return only the requested numbered lines."
            ),
            user_prompt=inject_prompt,
            expected_count=shot_count,
            temperature=0.4,
            max_tokens=max(1200, shot_count * 260),
            item_kind="numbered prompts",
        )
        prompts = self._dedupe_inline_bibles_per_prompt(prompts, object_bibles)
        return [self._postprocess_inline_bible_prompt(p) for p in prompts]

    def _generate_character_bibles(self, *, idea: str, story_outline: List[str], draft_prompts: List[str], shot_count: int, style_hint: str, t2i_model_hint: str) -> List[str]:
        source_parts: List[str] = []
        if story_outline:
            source_parts.append("Story beats:\n" + "\n".join(f"{idx+1}. {beat}" for idx, beat in enumerate(story_outline)))
        if draft_prompts:
            source_parts.append("Draft text-to-image prompts:\n" + "\n".join(f"{idx+1}. {prompt}" for idx, prompt in enumerate(draft_prompts)))
        source_block = "\n\n".join(source_parts) if source_parts else f"User idea:\n{idea.strip()}"
        bible_prompt = f"""
Create a character bible for every distinct recurring or important living character, person, animal, creature, alien, or named figure that appears in the source below.

{source_block}

Global style hint: {style_hint or 'none'}
Target text-to-image model: {t2i_model_hint or 'none'}

Rules:
- Return one entry per line.
- Include humans, animals, aliens, creatures, and named figures that matter visually in the story.
- If the story clearly has multiple recurring characters with the same role, keep them separate with stable labels such as the first astronaut and the second astronaut.
- Write each line in this exact format: Name or label (detailed visual identity)
- Put the descriptive identity inside parentheses immediately after the name or label.
- Keep the name or label short and natural, such as Peter, his dog, the green alien, the old woman, the taxi driver.
- The text inside parentheses must be detailed and visual: age or life stage when relevant, build, face, hair or fur, clothing or accessories, colors, species traits, and other stable identifying details.
- Only include stable appearance traits. Do not include actions, poses, camera framing, current emotion, location, or shot-specific details.
- Do not write separate explanations, keys, character numbers, or summaries.
- Do not write lines like character 1 is ..., character 2 is ...
- Do not group different characters together in one line.
- Do not create bible entries for props, objects, rubble, vehicles, buildings, weapons, scenery, or other non-living things.
- Use one stable label per character and keep it natural, for example: the superhero, the young child, his dog, Peter, the old woman.
- Return only the character bible lines.
""".strip()
        raw = self.client.generate(
            system_prompt=(
                "You are an offline story and prompt engine. "
                "Follow the requested format exactly. "
                "Do not output chain-of-thought, thinking, explanations, JSON, markdown fences, or commentary. "
                "Return only the requested lines."
            ),
            user_prompt=bible_prompt,
            temperature=0.5,
            max_tokens=max(600, shot_count * 220),
        )
        initial_entries = self._clean_character_bible_entries(self._extract_character_bible_lines(raw))
        return self._strengthen_character_bibles(
            entries=initial_entries,
            idea=idea,
            story_outline=story_outline,
            draft_prompts=draft_prompts,
            shot_count=shot_count,
            style_hint=style_hint,
            t2i_model_hint=t2i_model_hint,
        )

    def _inject_character_bibles_into_t2i(self, *, story_outline: List[str], draft_prompts: List[str], character_bibles: List[str], shot_count: int, style_hint: str, negative_hint: str, t2i_model_hint: str) -> List[str]:
        if not draft_prompts or not character_bibles:
            return draft_prompts
        character_bibles = self._clean_character_bible_entries(character_bibles)
        if not character_bibles:
            return draft_prompts
        beats_block = "\n".join(f"{idx+1}. {beat}" for idx, beat in enumerate(story_outline)) if story_outline else "none"
        draft_block = "\n".join(f"{idx+1}. {draft}" for idx, draft in enumerate(draft_prompts))
        bible_block = "\n".join(f"- {entry}" for entry in character_bibles)
        inject_prompt = f"""
Rewrite the numbered text-to-image prompts below so each prompt injects the matching character bible inline.

Story beats:
{beats_block}

Character bible:
{bible_block}

Draft text-to-image prompts:
{draft_block}

Global style hint: {style_hint or 'none'}
Negative notes to avoid: {negative_hint or 'none'}
Target text-to-image model: {t2i_model_hint or 'none'}

Rules:
- Return exactly {shot_count} numbered prompts.
- Keep each prompt faithful to its matching story beat and draft prompt.
- Use the exact same character label from the character bible every time that character appears.
- Keep labels consistent across all prompts. Do not switch between forms like a superhero, superhero, and the superhero. Pick the bible label and reuse it.
- Only inject a character bible when that character is a clear visible subject or important visible entity in that shot. Do not force a character bible into environment-only or object-focused shots.
- When a listed character appears in a prompt, write the label followed immediately by its parenthetical bible and then a comma, for example: the superhero (detail...), moves through the scene.
- Inject each character bible only once per prompt, even if that same character is mentioned more than once in the sentence.
- Do not create character bibles for props, objects, rubble, vehicles, buildings, weapons, or scenery.
- Do not convert the prompt into a legend, key, cast list, or explanation.
- Do not add lines like character 1 is ..., character 2 is ...
- Do not append a separate character section at the end of any prompt.
- Keep possessive wording natural and readable. Rewrite the sentence if needed so the inline bible does not become awkward.
- Keep each prompt direct, visual, and usable for text-to-image.
- Return only the numbered prompts.
""".strip()
        prompts = self._generate_numbered_list_with_retry(
            system_prompt=(
                "You are an offline story and prompt engine. "
                "Follow the requested format exactly. "
                "Do not output chain-of-thought, thinking, explanations, JSON, markdown fences, or commentary. "
                "Return only the requested numbered lines."
            ),
            user_prompt=inject_prompt,
            expected_count=shot_count,
            temperature=0.4,
            max_tokens=max(1200, shot_count * 260),
            item_kind="numbered prompts",
        )
        prompts = self._force_character_bibles_in_prompts(prompts, character_bibles)
        prompts = self._dedupe_inline_bibles_per_prompt(prompts, character_bibles)
        return [self._postprocess_inline_bible_prompt(p) for p in prompts]

    def _generate_numbered_list_with_retry(self, *, system_prompt: str, user_prompt: str, expected_count: int, temperature: float, max_tokens: int, item_kind: str) -> List[str]:
        raw = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature, max_tokens=max_tokens)
        try:
            return self._extract_numbered_lines(raw, expected_count)
        except RuntimeError as first_error:
            first_pass: List[str] = []
            for probe in (expected_count - 1, max(1, expected_count // 2), 1):
                if probe < 1:
                    continue
                try:
                    first_pass = self._extract_numbered_lines(raw, probe)
                    break
                except Exception:
                    continue
            have = len(first_pass)
            if have <= 0 or have >= expected_count:
                raise first_error
            missing = expected_count - have
            self._log(f"Model returned {have}/{expected_count} {item_kind}; requesting the remaining {missing} item(s)...")
            continuation_prompt = f"""
The previous answer for {item_kind} stopped early.
You already returned the first {have} items.
Now continue and return exactly the missing items only.

Continue from item {have + 1} through item {expected_count}.
Do not repeat items 1 through {have}.
Do not restart from 1.
Keep the same format and return only numbered lines.

Original request:
{user_prompt}

Partial output already received:
{raw}
""".strip()
            continuation_raw = self.client.generate(
                system_prompt=system_prompt,
                user_prompt=continuation_prompt,
                temperature=max(0.2, min(temperature, 0.45)),
                max_tokens=max_tokens,
            )
            combined = (raw.strip() + "\n" + continuation_raw.strip()).strip()
            return self._extract_numbered_lines(combined, expected_count)

    def _refine_prompt_list(self, *, kind: str, source_beats: List[str], draft_prompts: List[str], shot_count: int, style_hint: str, negative_hint: str, t2i_model_hint: str = "", i2v_model_hint: str = "") -> List[str]:
        if not source_beats or not draft_prompts:
            return draft_prompts
        beats_block = "\n".join(f"{idx+1}. {beat}" for idx, beat in enumerate(source_beats))
        draft_block = "\n".join(f"{idx+1}. {draft}" for idx, draft in enumerate(draft_prompts))
        style_hint = self._clean_style_hint(style_hint)
        negative_hint = self._normalize_prompt_text(negative_hint)
        if kind == "t2i":
            refine_prompt = f"""
Rewrite the numbered text-to-image prompts below so each line stays faithful to its matching story beat.

Story beats:
{beats_block}

Draft prompts:
{draft_block}

Global style hint: {style_hint or 'none'}
Negative notes to avoid: {negative_hint or 'none'}
Target text-to-image model: {t2i_model_hint or 'none'}

Rules:
- Return exactly {shot_count} numbered prompts.
- Keep each prompt faithful to its matching story beat.
- Do not invent outcomes, props, or resolutions that are not in the beat.
- Do not change the core action, subject, or setting.
- Keep each line image-focused and concrete.
- Do not add camera movement.
- Keep the wording compact and usable.
- Return only the numbered prompts.
""".strip()
            max_tokens = shot_count * 170
        else:
            refine_prompt = f"""
Rewrite the numbered image-to-video prompts below so each line stays faithful to its matching story beat and source image prompt.

Story beats:
{beats_block}

Draft prompts:
{draft_block}

Global style hint: {style_hint or 'none'}
Target image-to-video model: {i2v_model_hint or 'none'}

Rules:
- Return exactly {shot_count} numbered prompts.
- This is for image-to-video: start from an already existing image and animate what is visible in that image.
- Keep each prompt faithful to its matching story beat.
- Focus on one main visible action per shot.
- Optional: add one small secondary motion or one simple camera move only if useful.
- Prefer actions over poses. Do not write poster captions, static pose descriptions, or appearance-only lines.
- Do not invent resolutions, props, or actions that are not in the beat.
- Do not flatten the action into idle breathing, standing, or drifting unless the beat itself is calm.
- Keep each line compact, motion-first, and directly usable for animation.
- Return only the numbered prompts.
""".strip()
            max_tokens = shot_count * 150
        return self._generate_numbered_list_with_retry(system_prompt=(
            "You are an offline story and prompt engine. "
            "Follow the requested format exactly. "
            "Do not output chain-of-thought, thinking, explanations, JSON, markdown fences, or commentary. "
            "Return only the requested numbered lines. "
            "Keep the writing concrete, visual, direct, and faithful to the source beats. "
            "For image-to-video, favor visible action over static pose or appearance descriptions."
        ), user_prompt=refine_prompt, expected_count=shot_count, temperature=0.45, max_tokens=max_tokens, item_kind=f"{kind} prompts")

    def generate_project(
        self,
        *,
        title: str,
        idea: str,
        shot_count: int,
        include_story_outline: bool,
        generate_t2i: bool,
        generate_i2v: bool,
        style_hint: str,
        negative_hint: str,
        use_character_bible: bool,
        use_object_bible: bool,
        t2i_model_hint: str,
        i2v_model_hint: str,
    ) -> StoryProject:
        if shot_count < 1:
            raise ValueError("Shot count must be at least 1.")
        if not idea.strip():
            raise ValueError("Idea is empty.")
        if not (generate_t2i or generate_i2v):
            raise ValueError("Enable at least text-to-image or image-to-video.")

        story_outline: List[str] = []
        character_bibles: List[str] = []
        object_bibles: List[str] = []
        t2i_prompts: List[str] = []
        i2v_prompts: List[str] = []

        system = (
            "You are an offline story and prompt engine. "
            "Follow the requested format exactly. "
            "Do not output chain-of-thought, thinking, explanations, JSON, markdown fences, or commentary. "
            "Return only the requested numbered lines. "
            "Keep the writing concrete, visual, and direct."
        )

        style_hint = self._clean_style_hint(style_hint)
        negative_hint = re.sub(r"\s+", " ", str(negative_hint or "").strip())
        t2i_model_hint = re.sub(r"\s+", " ", str(t2i_model_hint or "").strip())
        i2v_model_hint = re.sub(r"\s+", " ", str(i2v_model_hint or "").strip())

        self._log(f"Project: {title.strip() or 'Untitled project'}")
        self._log(f"Idea: {idea.strip()}")
        self._log(f"Shot count: {shot_count}")
        self._log(f"Active style: {style_hint or 'none'}")
        self._log(f"Negative notes: {negative_hint or 'none'}")
        self._log(f"Text-to-image model: {t2i_model_hint or 'none'}")
        self._log(f"Image-to-video model: {i2v_model_hint or 'none'}")
        self._log(f"Character bible: {'on' if use_character_bible else 'off'}")

        if include_story_outline:
            self._log("Generating story outline...")
            outline_prompt = f"""
Create exactly {shot_count} numbered story beats for a visual story.

User idea: {idea.strip()}
Style notes: {style_hint.strip() or 'none'}
Negative notes to avoid: {negative_hint.strip() or 'none'}
Target text-to-image model: {t2i_model_hint or 'none'}
Target image-to-video model: {i2v_model_hint or 'none'}

Rules:
- Write exactly {shot_count} numbered beats.
- Each beat must describe one clear event or moment.
- Keep each beat grounded, visual, and specific.
- Build actual progression from beginning to ending.
- Avoid generic filler, repeated mood labels, and poetic padding.
- Avoid camera language unless the user asked for it.
- No commentary before or after the list.
""".strip()
            story_outline = self._generate_numbered_list_with_retry(
                system_prompt=system,
                user_prompt=outline_prompt,
                expected_count=shot_count,
                temperature=0.7,
                max_tokens=shot_count * 120,
                item_kind="story beats",
            )
            self._log(self._format_lines_for_log("Story outline", story_outline))

        seed_text = "\n".join(f"{idx+1}. {beat}" for idx, beat in enumerate(story_outline)) if story_outline else idea.strip()

        if generate_t2i:
            self._log("Generating text-to-image prompts...")
            t2i_prompt = f"""
Using the source below, create exactly {shot_count} numbered prompts for text-to-image.

Source:
{seed_text}

Global style hint: {style_hint.strip() or 'none'}
Negative notes to avoid: {negative_hint.strip() or 'none'}
Target text-to-image model: {t2i_model_hint or 'none'}

Rules:
- Each prompt must describe one strong image.
- Focus on subject, setting, visible action, composition, and important details.
- Keep each prompt direct and usable.
- Do not write camera movement.
- Do not mention previous or next prompts.
- Avoid repeated filler such as "the mood is" unless truly needed.
- Return only the {shot_count} numbered prompts.
""".strip()
            t2i_prompts = self._generate_numbered_list_with_retry(
                system_prompt=system,
                user_prompt=t2i_prompt,
                expected_count=shot_count,
                temperature=0.65,
                max_tokens=shot_count * 180,
                item_kind="text-to-image prompts",
            )
            self._log("Refining text-to-image prompts for beat accuracy...")
            t2i_prompts = self._refine_prompt_list(
                kind="t2i",
                source_beats=story_outline if story_outline else [idea.strip()] * shot_count,
                draft_prompts=t2i_prompts,
                shot_count=shot_count,
                style_hint=style_hint,
                negative_hint=negative_hint,
                t2i_model_hint=t2i_model_hint,
                i2v_model_hint=i2v_model_hint,
            )
            if use_character_bible:
                self._log("Generating character bible...")
                character_bibles = self._generate_character_bibles(
                    idea=idea,
                    story_outline=story_outline,
                    draft_prompts=t2i_prompts,
                    shot_count=shot_count,
                    style_hint=style_hint,
                    t2i_model_hint=t2i_model_hint,
                )
                self._log(self._format_lines_for_log("Character bible", character_bibles))
                if character_bibles:
                    self._log("Injecting character bible into text-to-image prompts...")
                    t2i_prompts = self._inject_character_bibles_into_t2i(
                        story_outline=story_outline if story_outline else [idea.strip()] * shot_count,
                        draft_prompts=t2i_prompts,
                        character_bibles=character_bibles,
                        shot_count=shot_count,
                        style_hint=style_hint,
                        negative_hint=negative_hint,
                        t2i_model_hint=t2i_model_hint,
                    )
            if use_object_bible:
                self._log("Generating recurring object bible...")
                object_bibles = self._generate_object_bibles(
                    idea=idea,
                    story_outline=story_outline,
                    draft_prompts=t2i_prompts,
                    shot_count=shot_count,
                    style_hint=style_hint,
                    t2i_model_hint=t2i_model_hint,
                )
                self._log(self._format_lines_for_log("Object bible", object_bibles))
                if object_bibles:
                    self._log("Injecting recurring object bible into text-to-image prompts...")
                    t2i_prompts = self._inject_object_bibles_into_t2i(
                        story_outline=story_outline if story_outline else [idea.strip()] * shot_count,
                        draft_prompts=t2i_prompts,
                        object_bibles=object_bibles,
                        shot_count=shot_count,
                        style_hint=style_hint,
                        negative_hint=negative_hint,
                        t2i_model_hint=t2i_model_hint,
                    )
            if use_character_bible and character_bibles:
                t2i_prompts = self._expand_known_two_character_groups(t2i_prompts, character_bibles)
            t2i_prompts = self._apply_style_to_prompt_list(t2i_prompts, style_hint)
            self._log(self._format_lines_for_log("Text-to-image prompts", t2i_prompts))

        if generate_i2v:
            self._log("Generating image-to-video prompts...")
            source_for_i2v = t2i_prompts if t2i_prompts else story_outline
            if not source_for_i2v:
                source_for_i2v = [idea.strip()] * shot_count

            i2v_seed = "\n".join(f"{idx+1}. {item}" for idx, item in enumerate(source_for_i2v))
            i2v_prompt = f"""
Use the same scene order below to create exactly {shot_count} numbered prompts for image-to-video.

Source prompts:
{i2v_seed}

Global style hint: {style_hint.strip() or 'none'}
Target image-to-video model: {i2v_model_hint or 'none'}

Rules:
- This is for image-to-video, not text-to-image. Start from an already existing image and describe how it animates.
- Each line must describe what actually changes or moves in the shot.
- Keep the same main subject and setting from the source prompt.
- Focus on visible action first, not appearance or pose.
- Use one main action, plus at most one small secondary motion if useful.
- Add only useful camera motion, and keep it simple.
- Do not write poster captions, pose descriptions, heroic pose language, or still-image wording like illustration caption.
- Do not add poetic filler, emotional commentary, or repeated mood labels.
- Do not pad with lighting essays or cinematic wording unless it affects visible motion.
- Do not say static, keep the same frame, no movement, gentle wonder, tranquil mood, or similar filler.
- Return one compact motion prompt per line.
- Return only the {shot_count} numbered prompts.
""".strip()
            i2v_prompts = self._generate_numbered_list_with_retry(
                system_prompt=system,
                user_prompt=i2v_prompt,
                expected_count=shot_count,
                temperature=0.65,
                max_tokens=shot_count * 140,
                item_kind="image-to-video prompts",
            )
            self._log("Refining image-to-video prompts for beat accuracy and stronger action...")
            i2v_prompts = self._refine_prompt_list(
                kind="i2v",
                source_beats=story_outline if story_outline else source_for_i2v,
                draft_prompts=i2v_prompts,
                shot_count=shot_count,
                style_hint=style_hint,
                negative_hint=negative_hint,
                t2i_model_hint=t2i_model_hint,
                i2v_model_hint=i2v_model_hint,
            )
            i2v_prompts = self._apply_style_to_prompt_list(i2v_prompts, style_hint)
            self._log(self._format_lines_for_log("Image-to-video prompts", i2v_prompts))

        metadata = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "app": APP_NAME,
            "runner_path": self.client.runner_path,
            "model_path": self.client.model_path,
            "ctx_size": self.client.ctx_size,
            "top_p": self.client.top_p,
            "style_hint": style_hint,
            "negative_hint": negative_hint,
            "include_story_outline": include_story_outline,
            "generate_t2i": generate_t2i,
            "generate_i2v": generate_i2v,
            "use_character_bible": use_character_bible,
            "use_object_bible": use_object_bible,
            "t2i_model_hint": t2i_model_hint,
            "i2v_model_hint": i2v_model_hint,
        }

        return StoryProject(
            title=title.strip() or "Untitled project",
            idea=idea.strip(),
            shot_count=shot_count,
            story_outline=story_outline,
            character_bibles=character_bibles,
            object_bibles=object_bibles,
            text_to_image_prompts=t2i_prompts,
            image_to_video_prompts=i2v_prompts,
            metadata=metadata,
        )


# -----------------------------
# JSON persistence
# -----------------------------
def save_project_json(project: StoryProject, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as fh:
        json.dump(project.to_dict(), fh, indent=2, ensure_ascii=False)


def load_project_json(path: str) -> StoryProject:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return StoryProject(
        title=str(data.get("title") or "Untitled project"),
        idea=str(data.get("idea") or ""),
        shot_count=int(data.get("shot_count") or 1),
        story_outline=list(data.get("story_outline") or []),
        character_bibles=list(data.get("character_bibles") or []),
        object_bibles=list(data.get("object_bibles") or []),
        text_to_image_prompts=list(data.get("text_to_image_prompts") or []),
        image_to_video_prompts=list(data.get("image_to_video_prompts") or []),
        metadata=dict(data.get("metadata") or {}),
    )


# -----------------------------
# UI
# -----------------------------
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1320x900")
        self.minsize(1100, 760)

        APP_DIR.mkdir(parents=True, exist_ok=True)
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        self.worker_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.current_project: Optional[StoryProject] = None
        self.settings = self._load_settings()

        self._build_ui()
        self._apply_settings_to_ui()
        self.after(120, self._poll_worker_queue)

    def _load_settings(self) -> Dict[str, Any]:
        if SETTINGS_PATH.exists():
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        return {
            "runner_path": "",
            "model_path": "",
            "ctx_size": 8192,
            "top_p": 0.9,
            "output_dir": str(DEFAULT_OUTPUT_DIR),
            "last_title": "",
            "last_idea": "",
            "last_shot_count": 8,
            "last_style_hint": "",
            "last_negative_hint": "",
            "include_story_outline": True,
            "generate_t2i": True,
            "generate_i2v": True,
            "use_character_bible": False,
            "use_object_bible": False,
            "last_t2i_model_hint": "",
            "last_i2v_model_hint": "",
        }

    def _save_settings(self) -> None:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(self.settings, fh, indent=2, ensure_ascii=False)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_generate = ttk.Frame(notebook)
        self.tab_output = ttk.Frame(notebook)
        notebook.add(self.tab_generate, text="Generate")
        notebook.add(self.tab_output, text="Output")

        self.tab_generate.columnconfigure(0, weight=1)
        self.tab_generate.rowconfigure(0, weight=1)
        self.generate_canvas = tk.Canvas(self.tab_generate, highlightthickness=0)
        self.generate_scrollbar = ttk.Scrollbar(self.tab_generate, orient="vertical", command=self.generate_canvas.yview)
        self.generate_canvas.configure(yscrollcommand=self.generate_scrollbar.set)
        self.generate_canvas.grid(row=0, column=0, sticky="nsew")
        self.generate_scrollbar.grid(row=0, column=1, sticky="ns")
        self.generate_content = ttk.Frame(self.generate_canvas)
        self.generate_canvas_window = self.generate_canvas.create_window((0, 0), window=self.generate_content, anchor="nw")
        self.generate_content.bind("<Configure>", self._on_generate_content_configure)
        self.generate_canvas.bind("<Configure>", self._on_generate_canvas_configure)
        self.generate_canvas.bind_all("<MouseWheel>", self._on_generate_mousewheel, add="+")

        self._build_generate_tab()
        self._build_output_tab()

    def _build_generate_tab(self) -> None:
        tab = self.generate_content
        for c in range(2):
            tab.columnconfigure(c, weight=1)
        tab.rowconfigure(3, weight=1)

        cfg = ttk.LabelFrame(tab, text="Offline LLM")
        cfg.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        cfg.columnconfigure(1, weight=1)

        self.runner_var = tk.StringVar()
        self.model_var = tk.StringVar()
        self.ctx_var = tk.IntVar(value=8192)
        self.top_p_var = tk.DoubleVar(value=0.9)

        ttk.Label(cfg, text="llama-server path").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(cfg, textvariable=self.runner_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(cfg, text="Browse", command=self._browse_runner).grid(row=0, column=2, padx=6, pady=6)

        ttk.Label(cfg, text="GGUF model path").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(cfg, textvariable=self.model_var).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(cfg, text="Browse", command=self._browse_model).grid(row=1, column=2, padx=6, pady=6)

        ttk.Label(cfg, text="Context size").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Spinbox(cfg, from_=2048, to=65536, increment=1024, textvariable=self.ctx_var, width=12).grid(row=2, column=1, sticky="w", padx=6, pady=6)

        ttk.Label(cfg, text="Top-p").grid(row=2, column=2, sticky="e", padx=6, pady=6)
        ttk.Spinbox(cfg, from_=0.1, to=1.0, increment=0.05, textvariable=self.top_p_var, width=8).grid(row=2, column=3, sticky="w", padx=6, pady=6)

        prompt = ttk.LabelFrame(tab, text="Story setup")
        prompt.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        prompt.columnconfigure(1, weight=1)
        prompt.rowconfigure(4, weight=1)

        self.title_var = tk.StringVar()
        self.shot_count_var = tk.IntVar(value=8)
        self.style_var = tk.StringVar()
        self.negative_var = tk.StringVar()
        self.t2i_model_hint_var = tk.StringVar()
        self.i2v_model_hint_var = tk.StringVar()
        self.include_story_var = tk.BooleanVar(value=True)
        self.gen_t2i_var = tk.BooleanVar(value=True)
        self.gen_i2v_var = tk.BooleanVar(value=True)
        self.character_bible_var = tk.BooleanVar(value=False)
        self.object_bible_var = tk.BooleanVar(value=False)

        ttk.Label(prompt, text="Project title").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(prompt, textvariable=self.title_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)

        ttk.Label(prompt, text="Shot count").grid(row=0, column=2, sticky="w", padx=6, pady=6)
        ttk.Spinbox(prompt, from_=1, to=100, textvariable=self.shot_count_var, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=6)

        ttk.Label(prompt, text="Style hint").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(prompt, textvariable=self.style_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=6)

        ttk.Label(prompt, text="Negative hint").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(prompt, textvariable=self.negative_var).grid(row=2, column=1, columnspan=3, sticky="ew", padx=6, pady=6)

        ttk.Label(prompt, text="Text-to-image model").grid(row=3, column=0, sticky="w", padx=6, pady=6)
        ttk.Entry(prompt, textvariable=self.t2i_model_hint_var).grid(row=3, column=1, sticky="ew", padx=6, pady=6)
        ttk.Label(prompt, text="Image-to-video model").grid(row=3, column=2, sticky="w", padx=6, pady=6)
        ttk.Entry(prompt, textvariable=self.i2v_model_hint_var).grid(row=3, column=3, sticky="ew", padx=6, pady=6)

        checks = ttk.Frame(prompt)
        checks.grid(row=4, column=0, columnspan=4, sticky="w", padx=6, pady=6)
        ttk.Checkbutton(checks, text="Create story outline", variable=self.include_story_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(checks, text="Create text-to-image prompts", variable=self.gen_t2i_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(checks, text="Create image-to-video prompts", variable=self.gen_i2v_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(checks, text="Character bible", variable=self.character_bible_var).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(checks, text="Recurring object bible", variable=self.object_bible_var).pack(side="left", padx=(0, 10))

        ttk.Label(prompt, text="Idea / story input").grid(row=5, column=0, sticky="nw", padx=6, pady=6)
        self.idea_text = ScrolledText(prompt, height=10, wrap="word")
        self.idea_text.grid(row=5, column=1, columnspan=3, sticky="nsew", padx=6, pady=6)

        actions = ttk.Frame(tab)
        actions.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        actions.columnconfigure(4, weight=1)
        ttk.Button(actions, text="Test connection", command=self._test_connection).grid(row=0, column=0, padx=(0, 8))
        self.generate_button = ttk.Button(actions, text="Generate storyline + prompts", command=self._start_generation)
        self.generate_button.grid(row=0, column=1, padx=(0, 8))
        ttk.Button(actions, text="Load JSON", command=self._load_project_dialog).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(actions, text="Save JSON", command=self._save_project_dialog).grid(row=0, column=3, padx=(0, 8))
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(actions, textvariable=self.status_var).grid(row=0, column=4, sticky="e")

        logs = ttk.LabelFrame(tab, text="Log")
        logs.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=1)
        self.log_text = ScrolledText(logs, height=18, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.log_menu = tk.Menu(self, tearoff=0)
        self.log_menu.add_command(label="Copy", command=self._copy_log_selection)
        self.log_menu.add_command(label="Clear logs", command=self._clear_logs)
        self.log_text.bind("<Button-3>", self._show_log_context_menu)

    def _on_generate_content_configure(self, _event: tk.Event) -> None:
        self.generate_canvas.configure(scrollregion=self.generate_canvas.bbox("all"))

    def _on_generate_canvas_configure(self, event: tk.Event) -> None:
        self.generate_canvas.itemconfigure(self.generate_canvas_window, width=event.width)

    def _on_generate_mousewheel(self, event: tk.Event) -> None:
        widget = self.winfo_containing(event.x_root, event.y_root)
        if widget is None:
            return
        current = widget
        while current is not None:
            if current is self.generate_content:
                delta = int(-event.delta / 120) if getattr(event, "delta", 0) else 0
                if delta:
                    self.generate_canvas.yview_scroll(delta, "units")
                return
            current = getattr(current, "master", None)

    def _build_output_tab(self) -> None:
        tab = self.tab_output
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(2, weight=1)
        tab.rowconfigure(1, weight=1)

        top = ttk.Frame(tab)
        top.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        top.columnconfigure(1, weight=1)
        self.output_dir_var = tk.StringVar()
        ttk.Label(top, text="JSON export folder").grid(row=0, column=0, sticky="w", padx=6)
        ttk.Entry(top, textvariable=self.output_dir_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="Browse", command=self._browse_output_dir).grid(row=0, column=2, padx=6)

        self.story_box = self._make_output_box(tab, "Story outline", 0)
        self.t2i_box = self._make_output_box(tab, "Text-to-image prompts", 1)
        self.i2v_box = self._make_output_box(tab, "Image-to-video prompts", 2)

    def _make_output_box(self, parent: ttk.Frame, title: str, column: int) -> ScrolledText:
        frame = ttk.LabelFrame(parent, text=title)
        frame.grid(row=1, column=column, sticky="nsew", padx=10, pady=(0, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        box = ScrolledText(frame, wrap="word")
        box.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        return box

    def _apply_settings_to_ui(self) -> None:
        self.runner_var.set(str(self.settings.get("runner_path") or ""))
        self.model_var.set(str(self.settings.get("model_path") or ""))
        self.ctx_var.set(int(self.settings.get("ctx_size") or 8192))
        self.top_p_var.set(float(self.settings.get("top_p") or 0.9))
        self.output_dir_var.set(str(self.settings.get("output_dir") or DEFAULT_OUTPUT_DIR))
        self.title_var.set(str(self.settings.get("last_title") or ""))
        self.shot_count_var.set(int(self.settings.get("last_shot_count") or 8))
        self.style_var.set(str(self.settings.get("last_style_hint") or ""))
        self.negative_var.set(str(self.settings.get("last_negative_hint") or ""))
        self.t2i_model_hint_var.set(str(self.settings.get("last_t2i_model_hint") or ""))
        self.i2v_model_hint_var.set(str(self.settings.get("last_i2v_model_hint") or ""))
        self.include_story_var.set(bool(self.settings.get("include_story_outline", True)))
        self.gen_t2i_var.set(bool(self.settings.get("generate_t2i", True)))
        self.gen_i2v_var.set(bool(self.settings.get("generate_i2v", True)))
        self.character_bible_var.set(bool(self.settings.get("use_character_bible", False)))
        self.object_bible_var.set(bool(self.settings.get("use_object_bible", False)))
        self.idea_text.delete("1.0", "end")
        self.idea_text.insert("1.0", str(self.settings.get("last_idea") or ""))

    def _sync_settings_from_ui(self) -> None:
        self.settings.update(
            {
                "runner_path": self.runner_var.get().strip(),
                "model_path": self.model_var.get().strip(),
                "ctx_size": int(self.ctx_var.get()),
                "top_p": float(self.top_p_var.get()),
                "output_dir": self.output_dir_var.get().strip() or str(DEFAULT_OUTPUT_DIR),
                "last_title": self.title_var.get().strip(),
                "last_idea": self.idea_text.get("1.0", "end").strip(),
                "last_shot_count": int(self.shot_count_var.get()),
                "last_style_hint": self.style_var.get().strip(),
                "last_negative_hint": self.negative_var.get().strip(),
                "last_t2i_model_hint": self.t2i_model_hint_var.get().strip(),
                "last_i2v_model_hint": self.i2v_model_hint_var.get().strip(),
                "include_story_outline": bool(self.include_story_var.get()),
                "generate_t2i": bool(self.gen_t2i_var.get()),
                "generate_i2v": bool(self.gen_i2v_var.get()),
                "use_character_bible": bool(self.character_bible_var.get()),
                "use_object_bible": bool(self.object_bible_var.get()),
            }
        )
        self._save_settings()

    def _browse_runner(self) -> None:
        path = filedialog.askopenfilename(title="Select llama-server executable")
        if path:
            self.runner_var.set(path)

    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(title="Select GGUF model", filetypes=[("GGUF", "*.gguf"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Select export folder")
        if path:
            self.output_dir_var.set(path)

    def _show_log_context_menu(self, event: tk.Event) -> str:
        try:
            self.log_text.tag_ranges("sel")
            selected = bool(self.log_text.tag_ranges("sel"))
        except tk.TclError:
            selected = False
        self.log_menu.entryconfigure("Copy", state=("normal" if selected or self.log_text.get("1.0", "end-1c").strip() else "disabled"))
        self.log_menu.tk_popup(event.x_root, event.y_root)
        self.log_menu.grab_release()
        return "break"

    def _copy_log_selection(self) -> None:
        try:
            if self.log_text.tag_ranges("sel"):
                text = self.log_text.get("sel.first", "sel.last")
            else:
                text = self.log_text.get("1.0", "end-1c")
        except tk.TclError:
            text = self.log_text.get("1.0", "end-1c")
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)

    def _clear_logs(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_outputs(self, project: StoryProject) -> None:
        self._fill_box(self.story_box, project.story_outline)
        self._fill_box(self.t2i_box, project.text_to_image_prompts)
        self._fill_box(self.i2v_box, project.image_to_video_prompts)

    @staticmethod
    def _fill_box(box: ScrolledText, lines: List[str]) -> None:
        box.delete("1.0", "end")
        if not lines:
            return
        box.insert("1.0", "\n\n".join(f"[{idx+1:02d}] {line}" for idx, line in enumerate(lines)))

    def _make_client(self) -> LocalLlamaClient:
        self._sync_settings_from_ui()
        return LocalLlamaClient(
            runner_path=self.runner_var.get(),
            model_path=self.model_var.get(),
            ctx_size=int(self.ctx_var.get()),
            top_p=float(self.top_p_var.get()),
        )

    def _test_connection(self) -> None:
        self._sync_settings_from_ui()
        self.generate_button.configure(state="disabled")
        self.status_var.set("Testing local llama-server...")

        def work() -> None:
            client = self._make_client()
            try:
                client.start()
                self.worker_queue.put(("log", f"Connected. llama-server is ready on {client.base_url}"))
                self.worker_queue.put(("status", "Connection test succeeded."))
            except Exception as exc:
                self.worker_queue.put(("error", str(exc)))
            finally:
                client.stop()
                self.worker_queue.put(("done", None))

        self.worker_thread = threading.Thread(target=work, daemon=True)
        self.worker_thread.start()

    def _start_generation(self) -> None:
        self._sync_settings_from_ui()
        self.generate_button.configure(state="disabled")
        self.status_var.set("Generating...")
        self._log("Starting generation...")
        self._log("Generated output will be printed into the log before you save any JSON.")

        title = self.title_var.get().strip()
        idea = self.idea_text.get("1.0", "end").strip()
        shot_count = int(self.shot_count_var.get())
        include_story = bool(self.include_story_var.get())
        gen_t2i = bool(self.gen_t2i_var.get())
        gen_i2v = bool(self.gen_i2v_var.get())
        use_character_bible = bool(self.character_bible_var.get())
        use_object_bible = bool(self.object_bible_var.get())
        style_hint = self.style_var.get().strip()
        negative_hint = self.negative_var.get().strip()
        t2i_model_hint = self.t2i_model_hint_var.get().strip()
        i2v_model_hint = self.i2v_model_hint_var.get().strip()

        def work() -> None:
            client = self._make_client()
            generator = StorylineGenerator(client, log_callback=lambda msg: self.worker_queue.put(("log", msg)))
            try:
                self.worker_queue.put(("log", "Launching local llama-server..."))
                project = generator.generate_project(
                    title=title,
                    idea=idea,
                    shot_count=shot_count,
                    include_story_outline=include_story,
                    generate_t2i=gen_t2i,
                    generate_i2v=gen_i2v,
                    style_hint=style_hint,
                    negative_hint=negative_hint,
                    use_character_bible=use_character_bible,
                    use_object_bible=use_object_bible,
                    t2i_model_hint=t2i_model_hint,
                    i2v_model_hint=i2v_model_hint,
                )
                self.worker_queue.put(("project", project))
                self.worker_queue.put(("log", "Result is visible above in the log and in the output panes. Save to JSON only when you want to keep it."))
                self.worker_queue.put(("status", "Generation finished."))
            except Exception as exc:
                self.worker_queue.put(("error", str(exc)))
            finally:
                client.stop()
                self.worker_queue.put(("done", None))

        self.worker_thread = threading.Thread(target=work, daemon=True)
        self.worker_thread.start()

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                if kind == "log":
                    self._log(str(payload))
                elif kind == "status":
                    self.status_var.set(str(payload))
                elif kind == "project":
                    self.current_project = payload
                    self._set_outputs(payload)
                    self._log("Generation completed and output panes updated.")
                elif kind == "error":
                    self._log(f"ERROR: {payload}")
                    self.status_var.set("Failed.")
                    messagebox.showerror(APP_NAME, str(payload))
                elif kind == "done":
                    self.generate_button.configure(state="normal")
        except queue.Empty:
            pass
        self.after(120, self._poll_worker_queue)

    def _save_project_dialog(self) -> None:
        if not self.current_project:
            messagebox.showinfo(APP_NAME, "Nothing to save yet. Generate or load a project first.")
            return
        self._sync_settings_from_ui()
        initial_dir = self.output_dir_var.get().strip() or str(DEFAULT_OUTPUT_DIR)
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", self.current_project.title).strip("_") or "story_project"
        path = filedialog.asksaveasfilename(
            title="Save project JSON",
            defaultextension=".json",
            initialdir=initial_dir,
            initialfile=f"{safe_name}.json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        save_project_json(self.current_project, path)
        self.status_var.set(f"Saved: {path}")
        self._log(f"Saved JSON: {path}")

    def _load_project_dialog(self) -> None:
        path = filedialog.askopenfilename(title="Load project JSON", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            project = load_project_json(path)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Failed to load JSON:\n{exc}")
            return
        self.current_project = project
        self.title_var.set(project.title)
        self.shot_count_var.set(project.shot_count)
        self.idea_text.delete("1.0", "end")
        self.idea_text.insert("1.0", project.idea)
        self._set_outputs(project)
        self.character_bible_var.set(bool((project.metadata or {}).get("use_character_bible", False)))
        self.object_bible_var.set(bool((project.metadata or {}).get("use_object_bible", False)))
        self.t2i_model_hint_var.set(str((project.metadata or {}).get("t2i_model_hint", "") or ""))
        self.i2v_model_hint_var.set(str((project.metadata or {}).get("i2v_model_hint", "") or ""))
        self.status_var.set(f"Loaded: {path}")
        self._log(f"Loaded JSON: {path}")
        if getattr(project, "character_bibles", None):
            self._log(StorylineGenerator._format_lines_for_log("Character bible", project.character_bibles))
        if getattr(project, "object_bibles", None):
            self._log(StorylineGenerator._format_lines_for_log("Object bible", project.object_bibles))


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
