from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

APP_DIR = Path.home() / ".offline_storyline_creator"


def _strip_llm_protocol_artifacts(text: str) -> str:
    s = str(text or "")
    if not s:
        return ""
    s = re.sub(r"(?is)<think\b[^>]*>.*?</think>", "", s)
    s = re.sub(r"(?i)<\|/?(?:begin_of_text|end_of_text|eot_id|im_start|im_end|start_header_id|end_header_id|channel|message|assistant|user|system|final|analysis|thought|reasoning)[^>]*\|?>", "", s)
    s = re.sub(r"(?im)^\s*(?:final|answer|response)\s*[:：]\s*", "", s)
    return s.replace("\r\n", "\n").replace("\r", "\n").strip()


def _clean_line(value: Any) -> str:
    s = _strip_llm_protocol_artifacts(str(value or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("`*_ \t\r\n")


def _parse_json(text: str) -> Any:
    raw = _strip_llm_protocol_artifacts(text).strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    dec = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch not in "[{":
            continue
        try:
            obj, _ = dec.raw_decode(raw[i:])
            return obj
        except Exception:
            continue
    raise RuntimeError("Model response did not contain parseable JSON.\n\nRaw output:\n" + raw[:6000])


def _sig(text: str) -> set[str]:
    stop = {"the","a","an","and","or","to","of","in","on","at","with","as","for","from","into","his","her","their","he","she","they","it","is","are","was","were","then","while","this","that","shot","scene","camera"}
    return {w for w in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(w) > 2 and w not in stop}


def _name_key(value: Any) -> str:
    s = _clean_line(value).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _subject_present(text: str, names: List[str], role_tags: List[str]) -> bool:
    hay = _name_key(text)
    if not hay:
        return False
    for token in list(names or []) + list(role_tags or []):
        tk = _name_key(token)
        if tk and tk in hay:
            return True
    return False


def _compose_identity_line(item: Dict[str, Any], kind: str = "character") -> str:
    name = _clean_line(item.get("display_name") or item.get("name") or item.get("id") or kind.title())
    tags = [_clean_line(x) for x in (item.get("role_tags") or []) if _clean_line(x)]
    identity = _clean_line(item.get("identity_anchor") or item.get("visual_identity") or item.get("continuity_role") or "")
    wardrobe = _clean_line(item.get("wardrobe_anchor") or item.get("wardrobe") or "")
    parts: List[str] = []
    if tags:
        parts.append(f"role/look: {', '.join(tags)}")
    if identity:
        parts.append(identity)
    if wardrobe:
        parts.append(f"wardrobe/material: {wardrobe}")
    detail = "; ".join([p for p in parts if p])
    if detail:
        return f"{name}: {detail}"
    return name


def _ensure_unique_identity_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for line in lines or []:
        s = _clean_line(line)
        if not s:
            continue
        key = _name_key(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _identity_prefix_for_shot(characters: List[Dict[str, Any]], objects: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for c in characters or []:
        line = _compose_identity_line(c, "character")
        if line:
            parts.append(line)
    for o in objects or []:
        line = _compose_identity_line(o, "object")
        if line:
            parts.append(line)
    if not parts:
        return ""
    return "Maintain continuity for visible recurring subjects: " + " | ".join(parts)


def _inline_identity_details(item: Dict[str, Any]) -> str:
    identity = _clean_line(item.get("identity_anchor") or item.get("visual_identity") or item.get("continuity_role") or "")
    wardrobe = _clean_line(item.get("wardrobe_anchor") or item.get("wardrobe") or "")
    parts: List[str] = []
    if identity:
        parts.append(identity)
    if wardrobe:
        parts.append(wardrobe)
    if not parts:
        tags = [_clean_line(x) for x in (item.get("role_tags") or []) if _clean_line(x)]
        if tags:
            parts.append(", ".join(tags))
    return "; ".join([p for p in parts if p])


def _item_aliases(item: Dict[str, Any]) -> List[str]:
    raw = [item.get("display_name"), item.get("name"), item.get("id")] + list(item.get("role_tags") or [])
    out: List[str] = []
    seen: set[str] = set()
    for value in raw:
        alias = _clean_line(value)
        key = _name_key(alias)
        if not alias or not key or key in seen:
            continue
        seen.add(key)
        out.append(alias)
    out.sort(key=lambda x: (-len(_name_key(x)), x.lower()))
    return out


def _inject_identity_once(prompt: str, item: Dict[str, Any]) -> Tuple[str, bool]:
    s = _clean_line(prompt)
    details = _inline_identity_details(item)
    if not s or not details:
        return s, False
    details_key = _name_key(details)
    if details_key and details_key in _name_key(s):
        return s, False
    aliases = _item_aliases(item)
    chosen = None
    for alias in aliases:
        pattern = re.compile(rf"(?<!\w)({re.escape(alias)})(?!\w)", re.IGNORECASE)
        m = pattern.search(s)
        if not m:
            continue
        if chosen is None or m.start() < chosen[0].start() or (m.start() == chosen[0].start() and len(alias) > len(chosen[1])):
            chosen = (m, alias)
    if chosen is None:
        return s, False
    m, _alias = chosen
    expanded = f"{m.group(1)} ({details})"
    return s[:m.start()] + expanded + s[m.end():], True


def _inject_bound_identities(prompt: str, characters: List[Dict[str, Any]], objects: List[Dict[str, Any]]) -> str:
    s = _clean_line(prompt)
    if not s:
        return s
    missing_prefixes: List[str] = []
    for item in list(characters or []) + list(objects or []):
        s, changed = _inject_identity_once(s, item)
        if changed:
            continue
        details = _inline_identity_details(item)
        name = _clean_line(item.get("display_name") or item.get("name") or item.get("id") or "")
        if details and name:
            token = f"{name} ({details})"
            if _name_key(token) not in _name_key(s):
                missing_prefixes.append(token)
    if missing_prefixes:
        s = ". ".join(missing_prefixes) + ". " + s
    return s


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
    story_bible: List[str] = field(default_factory=list)
    narrative_beats: List[str] = field(default_factory=list)
    shot_plan: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        if "server" in os.path.basename(raw).lower():
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
    def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 360.0) -> Tuple[int, Dict[str, Any]]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
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
            parts = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text", item.get("content", ""))
                    if txt:
                        parts.append(str(txt))
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()
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
        args = [self.runner_path, "-m", self.model_path, "--host", "127.0.0.1", "--port", str(self.port), "-c", str(self.ctx_size), "--reasoning-budget", "0"]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(self.log_path, "w", encoding="utf-8", errors="replace")
        try:
            self.proc = subprocess.Popen(args, stdout=log_handle, stderr=subprocess.STDOUT, cwd=os.path.dirname(self.runner_path), creationflags=creationflags)
        finally:
            log_handle.close()
        started = time.time()
        while time.time() - started <= 240:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited before becoming ready. Check {self.log_path}.")
            try:
                code, _ = self._http_get_json(f"{self.base_url}/health", timeout=4.0)
                if code == 200:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("Timed out waiting for llama-server.")

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

    def generate(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.55, max_tokens: int = 4096, json_mode: bool = False) -> str:
        if not self.base_url or not self.proc or self.proc.poll() is not None:
            self.start()
        direct_system = str(system_prompt or "")
        direct_user = str(user_prompt or "")
        if json_mode:
            direct_system = (
                "DIRECT STRUCTURED OUTPUT MODE. Do not reveal reasoning or analysis. "
                "Do not emit <think> tags. Start the response with { and return only the requested JSON object. "
                + direct_system
            )
            # Qwen3-family chat templates understand /no_think; models that do not simply see it as an extra instruction.
            direct_user = direct_user.rstrip() + "\n\n/no_think"
        payload = {
            "model": "local-model",
            "messages": [{"role": "system", "content": direct_system}, {"role": "user", "content": direct_user}],
            "stream": False,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(self.top_p),
            "reasoning_format": "none",
        }
        if json_mode:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["response_format"] = {"type": "json_object"}
        code, data = self._http_post_json(f"{self.base_url}/v1/chat/completions", payload)
        if code >= 400 and json_mode:
            # Compatibility fallback for older llama.cpp servers that do not accept one of the optional JSON/thinking fields.
            payload.pop("chat_template_kwargs", None)
            payload.pop("response_format", None)
            code, data = self._http_post_json(f"{self.base_url}/v1/chat/completions", payload)
        if code >= 400:
            msg = ((data or {}).get("error") or {}).get("message") or f"HTTP {code}"
            raise RuntimeError(str(msg))
        choices = (data or {}).get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by llama-server.")
        return _strip_llm_protocol_artifacts(self._extract_message_text(choices[0].get("message") or {})).strip()


class StorylineGenerator:
    """Agent-style Planner story engine.

    This intentionally does NOT inherit the old offline-storyline pipeline.  It
    mirrors the Telegram Agent architecture: locked whole-story blueprint ->
    chunked chronological beats -> separate shot-direction pass -> prompt output.
    """

    def __init__(self, client: LocalLlamaClient, log_callback: Optional[Callable[[str], None]] = None):
        self.client = client
        self.log_callback = log_callback

    def _log(self, text: str) -> None:
        if callable(self.log_callback):
            try:
                self.log_callback(str(text))
            except Exception:
                pass

    def _json_call(self, system: str, user: str, label: str, *, max_tokens: int = 5000, retries: int = 3) -> Any:
        last = None
        feedback = ""
        for attempt in range(1, retries + 1):
            try:
                text = self.client.generate(system, user + feedback, temperature=0.30 if attempt == 1 else 0.18, max_tokens=max_tokens, json_mode=True)
                return _parse_json(text)
            except Exception as exc:
                last = exc
                self._log(f"[story] {label} attempt {attempt}/{retries} failed: {exc}")
                feedback = "\n\nYour previous response was invalid. Return ONLY the requested JSON shape with every required item present."
        raise RuntimeError(f"{label} failed after {retries} attempts: {last}")

    @staticmethod
    def _lock_section_budget(sections: List[Dict[str, Any]], total: int) -> List[Dict[str, Any]]:
        if not sections:
            raise RuntimeError("Blueprint returned no story_sections.")
        sections = [dict(s) for s in sections if isinstance(s, dict)]
        counts = [max(1, int(s.get("clip_count") or 1)) for s in sections]
        # If too many sections for clips, keep the architecture valid by refusing.
        if len(counts) > total:
            raise RuntimeError(f"Blueprint created {len(counts)} sections for only {total} clips.")
        delta = total - sum(counts)
        if delta > 0:
            order = list(range(1, max(1, len(counts) - 1))) or [0]
            pos = 0
            while delta > 0:
                counts[order[pos % len(order)]] += 1
                delta -= 1
                pos += 1
        elif delta < 0:
            need = -delta
            # Trim resolution first, then largest development sections; never kill a section.
            order = [len(counts)-1] + sorted(range(max(0, len(counts)-1)), key=lambda i: (-counts[i], i))
            while need > 0:
                changed = False
                for i in order:
                    if need <= 0:
                        break
                    if counts[i] > 1:
                        counts[i] -= 1
                        need -= 1
                        changed = True
                if not changed:
                    break
            if need:
                raise RuntimeError("Could not reconcile blueprint section budget with requested shot count.")
        for sec, count in zip(sections, counts):
            sec["clip_count"] = count
        return sections

    @staticmethod
    def _slot_targets(sections: List[Dict[str, Any]], total: int) -> List[Dict[str, Any]]:
        out = []
        slot = 1
        for sec_i, sec in enumerate(sections, 1):
            count = int(sec.get("clip_count") or 0)
            for local_i in range(1, count + 1):
                if slot > total:
                    break
                out.append({
                    "slot": slot,
                    "section_index": sec_i,
                    "section": str(sec.get("title") or f"Section {sec_i}"),
                    "role": str(sec.get("role") or "development"),
                    "purpose": str(sec.get("purpose") or ""),
                    "must_achieve": str(sec.get("must_achieve") or ""),
                    "section_beat": f"{local_i}/{count}",
                })
                slot += 1
        if len(out) != total:
            raise RuntimeError(f"Blueprint produced {len(out)} slot targets; expected {total}.")
        return out

    @staticmethod
    def _duplicate_issue(beats: List[Dict[str, Any]]) -> Optional[str]:
        sigs = [_sig(str(b.get("beat") or "")) for b in beats]
        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                if not sigs[i] or not sigs[j]:
                    continue
                inter = len(sigs[i] & sigs[j])
                union = len(sigs[i] | sigs[j]) or 1
                if inter >= 5 and inter / union >= 0.72:
                    return f"story slots {i+1} and {j+1} are near-duplicate events"
        return None

    @staticmethod
    def _story_section_guidance(target_duration: float, shot_count: int) -> Tuple[int, int]:
        """Return a duration-aware *guidance* range for blueprint section count.

        This deliberately does not force one fixed act template.  Longer runtimes get
        permission to create more narrative movements so a 10- or 30-minute story is
        not stretched across the same five sections used by a short film.
        """
        duration = max(1.0, float(target_duration or 0.0))
        shots = max(1, int(shot_count or 1))
        if duration <= 60:
            lo, hi = 3, 5
        elif duration <= 120:
            lo, hi = 4, 6
        elif duration <= 300:
            lo, hi = 5, 8
        elif duration <= 600:
            lo, hi = 6, 10
        elif duration <= 1200:
            lo, hi = 8, 14
        else:
            lo, hi = 10, 18
        hi = min(hi, shots)
        lo = min(lo, hi)
        return max(1, lo), max(1, hi)

    def _build_blueprint(self, idea: str, style: str, shot_count: int, target_duration: float, reference_guidance: str = "", audio_context: str = "") -> Dict[str, Any]:
        section_min, section_max = self._story_section_guidance(target_duration, shot_count)
        blueprint_max_tokens = int(min(12000, max(4500, 2800 + (section_max * 360) + (shot_count * 8))))
        system = (
            "You are the senior story director inside FrameVision. Design the COMPLETE narrative architecture BEFORE any individual shots are written. "
            "The story must use the user's actual idea as authority, not replace it with a generic interpretation. The style brief controls presentation but must not erase plot requirements. "
            "Plan setup, development, escalation, climax and resolution across the full runtime. Do not finish early and fill remaining clips with repeated reactions, alternate camera angles, scenery, poses, or establishing shots. "
            "A clip slot exists because something narratively changes. Camera coverage alone is never a reason for another slot. "
            "Every explicitly named action, obstacle, set piece, location change, reveal, and ending condition in the user's idea is a story obligation unless it is physically impossible. Do not silently replace them with easier generic events. "
            "Reference guidance constrains identity/appearance only and must not rewrite the plot. Audio/lyrics context may influence pacing but must not replace the user's story. "
            "Return JSON only with title, story_summary, recurring_subjects, required_story_elements, and story_sections. recurring_subjects contains only characters/creatures/important recurring objects that require identity continuity. "
            "required_story_elements is a concise list of the user's explicit plot/set-piece obligations that the later beats must cover. Each story section must contain title, role, purpose, must_achieve and clip_count."
        )
        user = f"""USER IDEA (AUTHORITATIVE — preserve every explicit plot requirement):
{idea}

STYLE / PRESENTATION BRIEF (AUTHORITATIVE — do not invent a conflicting style):
{style or '[none]'}

REFERENCE GUIDANCE (identity/appearance constraints only):
{reference_guidance or '[none]'}

AUDIO / LYRIC CONTEXT (pacing context only):
{audio_context or '[none]'}

TARGET: exactly {shot_count} video clips across about {target_duration:.1f} seconds.
The sum of story_sections.clip_count MUST equal {shot_count}. The definitive resolution must occur in the final section near the end.
For this runtime, aim for about {section_min}-{section_max} meaningful story sections when the idea supports them. This is creative guidance, not a quota: use fewer if extra sections would be filler, or more only when genuinely useful. Longer runtimes should exploit the premise with additional developments, discoveries, obstacles, reversals, location changes, decisions or mini-payoffs instead of stretching a few events across many clips.
Before allocating sections, identify the explicit story obligations from USER IDEA and put them in required_story_elements. The beat writer will be required to cover them.

Return exactly this JSON shape:
{{
  "title": "...",
  "story_summary": "complete beginning-to-ending summary that preserves the user's plot",
  "recurring_subjects": [{{"id":"char_1","name":"...","type":"character|creature|object","continuity_role":"..."}}],
  "required_story_elements": ["explicit obligation 1", "explicit obligation 2"],
  "story_sections": [{{"title":"...","role":"setup|development|escalation|climax|resolution","purpose":"...","must_achieve":"...","clip_count":1}}]
}}"""
        obj = self._json_call(system, user, "blueprint", max_tokens=blueprint_max_tokens)
        if not isinstance(obj, dict):
            raise RuntimeError("Blueprint response was not an object.")
        raw_sections = obj.get("story_sections")
        if not isinstance(raw_sections, list):
            raise RuntimeError("Blueprint is missing story_sections.")
        sections = []
        for sec in raw_sections:
            if not isinstance(sec, dict):
                continue
            title = _clean_line(sec.get("title"))
            purpose = _clean_line(sec.get("purpose"))
            must = _clean_line(sec.get("must_achieve") or purpose)
            role = _clean_line(sec.get("role") or "development").lower()
            try:
                count = int(sec.get("clip_count") or 0)
            except Exception:
                count = 0
            if title and purpose and count > 0:
                sections.append({"title": title, "role": role, "purpose": purpose, "must_achieve": must, "clip_count": count})
        sections = self._lock_section_budget(sections, shot_count)
        severe_min = 5 if shot_count >= 10 else 1
        if target_duration >= 600:
            severe_min = max(severe_min, max(5, section_min - 2))
        severe_min = min(severe_min, shot_count)
        if len(sections) < severe_min:
            raise RuntimeError(
                f"Blueprint is too compressed for {target_duration:.0f}s: returned {len(sections)} sections; "
                f"need at least {severe_min} meaningful sections before shot expansion."
            )
        res_start = sum(int(s["clip_count"]) for s in sections[:-1]) + 1
        if shot_count >= 10 and res_start < int(math.floor(shot_count * 0.80)) + 1:
            raise RuntimeError(f"Blueprint resolves too early at slot {res_start} of {shot_count}.")
        subjects = [dict(x) for x in (obj.get("recurring_subjects") or []) if isinstance(x, dict)]
        required = [_clean_line(x) for x in (obj.get("required_story_elements") or []) if _clean_line(x)]
        if not required:
            required = [idea]
        return {"title": _clean_line(obj.get("title") or "Planner Story"), "story_summary": _clean_line(obj.get("story_summary") or idea), "recurring_subjects": subjects, "required_story_elements": required, "story_sections": sections}

    def _build_beats(self, idea: str, style: str, blueprint: Dict[str, Any], shot_count: int, quality_feedback: str = "") -> List[Dict[str, Any]]:
        """Create the locked shot list.

        The built-in 2B model occasionally returns fewer list items than requested even
        when the JSON itself is valid.  A wrong item count is therefore treated as a
        recoverable generation error, not as a fatal workflow error.  We retry the
        batch once, then fall back to one slot at a time while preserving the exact
        locked blueprint.
        """
        targets = self._slot_targets(blueprint["story_sections"], shot_count)
        all_beats: List[Dict[str, Any]] = []

        def _one_beat(target: Dict[str, Any], previous_beats: List[Dict[str, Any]]) -> Dict[str, Any]:
            system = (
                "You are the story-beat writer inside FrameVision. The story architecture and this slot are LOCKED. "
                "Write one concrete NEW chronological event for this slot only. Do not redesign the story or repeat an earlier event. "
                "The event must create a visible or narrative state change. Return JSON only as "
                "{\"beat\":\"...\",\"reference_ids\":[\"...\"]}."
            )
            user = (
                f"USER IDEA:\n{idea}\n\nSTYLE:\n{style or '[none]'}\n\nLOCKED STORY SUMMARY:\n{blueprint['story_summary']}\n\n"
                f"MANDATORY STORY ELEMENTS:\n{json.dumps(blueprint.get('required_story_elements') or [idea], ensure_ascii=False, indent=2)}\n\n"
                f"LOCKED SLOT:\n{json.dumps(target, ensure_ascii=False, indent=2)}\n\n"
                + (f"PREVIOUS BEATS — continue after these and do not repeat them:\n{json.dumps(previous_beats, ensure_ascii=False, indent=2)}\n\n" if previous_beats else "")
                + "Return one beat object only."
            )
            obj = self._json_call(system, user, f"shot {target['slot']}", max_tokens=1800)
            if isinstance(obj, dict) and isinstance(obj.get("beat"), str):
                item = obj
            elif isinstance(obj, dict) and isinstance(obj.get("beats"), list) and len(obj["beats"]) == 1 and isinstance(obj["beats"][0], dict):
                item = obj["beats"][0]
            else:
                raise RuntimeError(f"Shot {target['slot']} did not return one beat object.")
            beat = _clean_line(item.get("beat"))
            if not beat:
                raise RuntimeError(f"Shot {target['slot']} returned an empty beat.")
            return {
                "slot": int(target["slot"]), "section": target["section"], "role": target["role"],
                "purpose": target["purpose"], "must_achieve": target["must_achieve"], "beat": beat,
                "reference_ids": [str(x) for x in (item.get("reference_ids") or []) if str(x).strip()],
            }

        for start in range(1, shot_count + 1, 5):
            end = min(shot_count, start + 4)
            chunk_targets = targets[start-1:end]
            next_targets = targets[end:min(shot_count, end+3)]
            system = (
                "You are the story-beat writer inside FrameVision. The complete story architecture is already locked. "
                "Write exactly one concrete NEW chronological event for each supplied slot target. Do not redesign the story, rush ahead, repeat an earlier event, or create another clip merely to show the same event from a new angle. "
                "Every beat must cause a visible or narrative state change: an action begins or completes, new information changes behavior, an obstacle alters the plan, a location transition advances the objective, or a consequence forces the next event. "
                "If several slots belong to one section they must develop that phase through distinct cause-and-effect events. Return JSON only: {\"beats\":[{\"beat\":\"...\",\"reference_ids\":[\"...\"]}]} ."
            )
            base_user = (
                f"USER IDEA:\n{idea}\n\nSTYLE:\n{style or '[none]'}\n\nLOCKED STORY SUMMARY:\n{blueprint['story_summary']}\n\n"
                f"LOCKED RECURRING SUBJECTS:\n{json.dumps(blueprint['recurring_subjects'], ensure_ascii=False, indent=2)}\n\n"
                f"MANDATORY STORY ELEMENTS FROM USER IDEA — the complete beat list must cover these:\n{json.dumps(blueprint.get('required_story_elements') or [idea], ensure_ascii=False, indent=2)}\n\n"
                f"LOCKED FULL BLUEPRINT:\n{json.dumps(blueprint['story_sections'], ensure_ascii=False, indent=2)}\n\n"
                f"EXACT TARGETS FOR SLOTS {start}-{end}:\n{json.dumps(chunk_targets, ensure_ascii=False, indent=2)}\n\n"
                + (f"NEXT TARGETS FOR CONTEXT ONLY:\n{json.dumps(next_targets, ensure_ascii=False, indent=2)}\n\n" if next_targets else "")
                + (f"ALL PREVIOUS BEATS — continue after these and do not retell them:\n{json.dumps(all_beats, ensure_ascii=False, indent=2)}\n\n" if all_beats else "")
                + (f"QUALITY-GATE CORRECTION FROM PREVIOUS PASS:\n{quality_feedback}\n\n" if quality_feedback else "")
            )

            raw = None
            for semantic_attempt in range(1, 3):
                correction = "" if semantic_attempt == 1 else (
                    f"\n\nCORRECTION: Your previous valid JSON had the wrong number of items. "
                    f"Return EXACTLY {len(chunk_targets)} beat objects, one for every supplied slot, in the same order. Do not omit or merge slots."
                )
                try:
                    obj = self._json_call(system, base_user + correction + f"\n\nReturn exactly {len(chunk_targets)} beat objects in chronological order.", f"shots {start}-{end}", max_tokens=4200)
                except Exception:
                    self._log(f"[story] Shot list batch {start}-{end} could not produce valid JSON; switching this batch to individual shots.")
                    break
                candidate = obj.get("beats") if isinstance(obj, dict) else None
                if isinstance(candidate, list) and len(candidate) == len(chunk_targets) and all(isinstance(x, dict) for x in candidate):
                    raw = candidate
                    break
                got = len(candidate) if isinstance(candidate, list) else 0
                self._log(f"[story] Shot list batch {start}-{end} returned {got}/{len(chunk_targets)} items; retrying.")

            if raw is None:
                self._log(f"[story] Shot list batch {start}-{end} is incomplete; generating those slots individually.")
                for target in chunk_targets:
                    all_beats.append(_one_beat(target, all_beats))
                continue

            for idx, item in enumerate(raw):
                beat = _clean_line(item.get("beat"))
                if not beat:
                    target = chunk_targets[idx]
                    self._log(f"[story] Shot {target['slot']} beat was empty; regenerating that slot.")
                    all_beats.append(_one_beat(target, all_beats))
                    continue
                target = chunk_targets[idx]
                all_beats.append({
                    "slot": int(target["slot"]), "section": target["section"], "role": target["role"],
                    "purpose": target["purpose"], "must_achieve": target["must_achieve"], "beat": beat,
                    "reference_ids": [str(x) for x in (item.get("reference_ids") or []) if str(x).strip()],
                })
        issue = self._duplicate_issue(all_beats)
        if issue:
            raise RuntimeError("Story quality check rejected plan: " + issue)
        return all_beats

    def _build_bibles(self, idea: str, style: str, blueprint: Dict[str, Any], beats: List[Dict[str, Any]], use_character_bible: bool, use_object_bible: bool, predefined: Optional[List[str]], predefined_entries: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Build one continuity bible and bind it to the already-locked story.

        The blueprint owns recurring-subject IDs. The LLM may describe those subjects,
        but it is not allowed to invent a second ID namespace or a second shot-cast plan.
        Shot bindings come from the locked beat reference_ids plus explicit subject-name
        mentions in each beat. This makes the bible a consumer of the story rather than
        a competing planner.
        """
        predefined = [_clean_line(x) for x in (predefined or []) if _clean_line(x)]
        predefined_entries = [x for x in (predefined_entries or []) if isinstance(x, dict)]
        recurring = [x for x in (blueprint.get("recurring_subjects") or []) if isinstance(x, dict)]

        def _aliases(name: str) -> List[str]:
            full = _clean_line(name)
            out: List[str] = [full] if full else []
            words = [w for w in re.findall(r"[A-Za-z0-9]+", full) if len(w) >= 4]
            # A final role noun such as "guard" in "Night Guard" is useful because
            # story beats often shorten the recurring subject name after introduction.
            if len(words) >= 2:
                out.append(words[-1])
            return list(dict.fromkeys([x for x in out if x]))

        def _predefined_character_items() -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            if predefined_entries:
                for idx, rec in enumerate(predefined_entries, 1):
                    prompt = _clean_line(rec.get("prompt") or "")
                    if not prompt:
                        continue
                    codeword = _clean_line(rec.get("codeword") or "")
                    out.append({
                        "id": f"U{idx}",
                        "display_name": codeword or f"Character {idx}",
                        "role_tags": ([codeword] if codeword else []),
                        "identity_anchor": prompt,
                        "wardrobe_anchor": "",
                        "source": "user",
                    })
                return out
            for idx, line in enumerate(predefined or [], 1):
                name = _clean_line(str(line).split(":", 1)[0]) or f"Character {idx}"
                out.append({
                    "id": f"U{idx}",
                    "display_name": name,
                    "role_tags": [name],
                    "identity_anchor": _clean_line(line),
                    "wardrobe_anchor": "",
                    "source": "user",
                })
            return out

        manual_characters = _predefined_character_items()
        bundle: Dict[str, Any] = {
            "characters": list(manual_characters),
            "objects": [],
            "shot_bindings": {},
            "source_subjects": recurring,
            "mode": "manual_override" if manual_characters and not use_character_bible else "automatic",
        }

        # Own Character Bible is the escape hatch/override. When normal Character Bible
        # is OFF, do not generate a second automatic character identity set.
        want_auto_characters = bool(use_character_bible)
        want_auto_objects = bool(use_object_bible)

        if want_auto_characters or want_auto_objects:
            system = (
                "You are the continuity editor inside FrameVision. The story and recurring-subject IDs are already LOCKED. "
                "Your only job is to define a stable reusable VISUAL identity for each supplied recurring subject. "
                "Do not invent plot events, do not invent new recurring subjects, do not rename IDs, and do not decide which shots contain which subjects. "
                "Return JSON only with keys characters and objects. Every returned item must use the exact id supplied in RECURRING SUBJECTS and include display_name, role_tags, identity_anchor, and optional wardrobe_anchor. "
                "Describe the visible identity appropriate to the subject itself. A person should receive stable physical and clothing details; an animal, creature, alien, vehicle, machine, prop, or other subject should receive the appropriate stable visual traits for that subject rather than human anatomy. "
                "Make identity_anchor specific enough that an image model can reproduce the same subject across many independent images. "
                "User-provided character bibles are authoritative and must not be replaced."
            )
            base_user = f"""USER IDEA:\n{idea}\n\nSTYLE:\n{style or '[none]'}\n\nLOCKED STORY SUMMARY:\n{blueprint['story_summary']}\n\nRECURRING SUBJECTS - IDs ARE IMMUTABLE:\n{json.dumps(recurring, ensure_ascii=False, indent=2)}\n\nUSER-PROVIDED CHARACTER BIBLES (authoritative):\n{json.dumps(predefined, ensure_ascii=False, indent=2)}\n\nReturn JSON only."""

            parsed: Dict[str, Any] = {}
            missing_ids: List[str] = []
            expected_ids = {str(x.get("id") or "").strip() for x in recurring if str(x.get("id") or "").strip()}
            for semantic_attempt in range(1, 3):
                correction = ""
                if missing_ids:
                    correction = "\n\nCORRECTION: You omitted these locked recurring subject IDs: " + ", ".join(missing_ids) + ". Return them using exactly those IDs."
                # This is the one story stage whose JSON output grows with the total
                # clip count because it contains recurring identities plus shot bindings.
                # Keep short jobs at the old budget, but scale long stories instead of
                # truncating a 10-minute continuity bible at 4200 tokens.
                _bible_subjects = len((blueprint or {}).get("recurring_subjects") or [])
                _bible_max_tokens = int(min(16000, max(4200, 2200 + (len(beats) * 110) + (_bible_subjects * 180))))
                obj = self._json_call(system, base_user + correction, "continuity bibles", max_tokens=_bible_max_tokens)
                parsed = obj if isinstance(obj, dict) else {}
                returned_ids = set()
                for key in ("characters", "objects"):
                    for raw in (parsed.get(key) or []) if isinstance(parsed.get(key), list) else []:
                        if isinstance(raw, dict):
                            rid = str(raw.get("id") or "").strip()
                            if rid:
                                returned_ids.add(rid)
                # Only require locked subjects that the selected bible types intend to keep.
                # If both are on (normal Planner mode), every recurring subject must survive.
                if want_auto_characters and want_auto_objects:
                    missing_ids = sorted(expected_ids - returned_ids)
                else:
                    missing_ids = []
                if not missing_ids:
                    break
                self._log(f"[story] Continuity bible omitted {len(missing_ids)} recurring subject(s); retrying.")

            recurring_by_id = {str(x.get("id") or "").strip(): x for x in recurring if str(x.get("id") or "").strip()}
            recurring_by_name = {_name_key(x.get("name") or ""): x for x in recurring if _name_key(x.get("name") or "")}

            def _normalize(raw_items: Any, source_kind: str) -> List[Dict[str, Any]]:
                out: List[Dict[str, Any]] = []
                seen: set[str] = set()
                for raw in raw_items if isinstance(raw_items, list) else []:
                    if not isinstance(raw, dict):
                        continue
                    rid = str(raw.get("id") or "").strip()
                    name = _clean_line(raw.get("display_name") or raw.get("name") or "")
                    src = recurring_by_id.get(rid)
                    if src is None and name:
                        src = recurring_by_name.get(_name_key(name))
                    if src is None:
                        # Ignore invented subjects. This is a continuity bible, not a cast creator.
                        continue
                    rid = str(src.get("id") or "").strip()
                    if not rid or rid in seen:
                        continue
                    seen.add(rid)
                    display_name = _clean_line(src.get("name") or name or rid)
                    role_tags = [_clean_line(x) for x in (raw.get("role_tags") or []) if _clean_line(x)]
                    for a in _aliases(display_name):
                        if a not in role_tags:
                            role_tags.append(a)
                    identity = _clean_line(raw.get("identity_anchor") or raw.get("visual_identity") or raw.get("description") or "")
                    wardrobe = _clean_line(raw.get("wardrobe_anchor") or raw.get("wardrobe") or "")
                    if not identity:
                        identity = _clean_line(src.get("continuity_role") or display_name)
                    out.append({
                        "id": rid,
                        "display_name": display_name,
                        "role_tags": role_tags,
                        "identity_anchor": identity,
                        "wardrobe_anchor": wardrobe,
                        "source": "llm",
                        "source_type": _clean_line(src.get("type") or source_kind),
                    })
                return out

            auto_chars = _normalize(parsed.get("characters"), "character") if want_auto_characters else []
            auto_objs = _normalize(parsed.get("objects"), "object") if want_auto_objects else []

            # If a model put a recurring subject in the opposite array, preserve the
            # identity instead of losing it. IDs/names still come from the locked blueprint.
            normalized_all = {str(x.get("id")): x for x in auto_chars + auto_objs}
            for src in recurring:
                sid = str(src.get("id") or "").strip()
                if not sid or sid in normalized_all:
                    continue
                # Search the opposite/raw arrays by ID or name.
                found = None
                for key in ("characters", "objects"):
                    for raw in (parsed.get(key) or []) if isinstance(parsed.get(key), list) else []:
                        if not isinstance(raw, dict):
                            continue
                        if str(raw.get("id") or "").strip() == sid or _name_key(raw.get("display_name") or raw.get("name") or "") == _name_key(src.get("name") or ""):
                            found = raw
                            break
                    if found:
                        break
                if found:
                    item = _normalize([found], "subject")
                    if item:
                        normalized_all[sid] = item[0]

            if want_auto_characters:
                # Character/creature-like entries returned in characters stay characters.
                bundle["characters"] = list(manual_characters) + [x for x in normalized_all.values() if str(x.get("id")) in {str(c.get("id")) for c in auto_chars}]
            if want_auto_objects:
                bundle["objects"] = [x for x in normalized_all.values() if str(x.get("id")) in {str(o.get("id")) for o in auto_objs}]

            # Ensure no locked recurring subject disappears just because the LLM put it
            # into an unexpected array. Use the blueprint type only to choose storage;
            # visual identity itself remains entirely LLM-authored.
            have = {str(x.get("id")) for x in bundle["characters"] + bundle["objects"]}
            for sid, item in normalized_all.items():
                if sid in have:
                    continue
                src_type = _name_key(item.get("source_type") or "")
                if src_type in {"object", "prop", "item", "vehicle", "machine", "device"}:
                    if want_auto_objects:
                        bundle["objects"].append(item)
                elif want_auto_characters:
                    bundle["characters"].append(item)

        char_map = {str(c.get("id")): c for c in bundle["characters"] if isinstance(c, dict)}
        obj_map = {str(o.get("id")): o for o in bundle["objects"] if isinstance(o, dict)}
        source_to_kind: Dict[str, str] = {}
        for cid in char_map:
            source_to_kind[cid] = "character"
        for oid in obj_map:
            source_to_kind[oid] = "object"

        # Build shot bindings from the LOCKED story, not from a second LLM cast guess.
        for beat in beats or []:
            slot = int(beat.get("slot") or 0)
            beat_text = _clean_line(beat.get("beat") or "")
            refs = [str(x).strip() for x in (beat.get("reference_ids") or []) if str(x).strip()]
            cids: List[str] = []
            oids: List[str] = []

            def _add_subject(sid: str) -> None:
                if sid in char_map and sid not in cids:
                    cids.append(sid)
                if sid in obj_map and sid not in oids:
                    oids.append(sid)

            for sid in refs:
                _add_subject(sid)

            # Beat writers occasionally omit a reference_id even while explicitly
            # naming the subject in the beat (the museum guard case). Reconcile that
            # omission deterministically from the locked recurring-subject names.
            for sid, item in char_map.items():
                aliases = [item.get("display_name"), item.get("id")] + list(item.get("role_tags") or [])
                if _subject_present(beat_text, [str(x) for x in aliases if x], []):
                    _add_subject(sid)
            for sid, item in obj_map.items():
                aliases = [item.get("display_name"), item.get("id")] + list(item.get("role_tags") or [])
                if _subject_present(beat_text, [str(x) for x in aliases if x], []):
                    _add_subject(sid)

            # Manual override: bind only by the user's codeword/name. Do not run an
            # automatic character detector behind the user's back.
            if manual_characters and not use_character_bible:
                for item in manual_characters:
                    aliases = [item.get("display_name")] + list(item.get("role_tags") or [])
                    if _subject_present(beat_text, [str(x) for x in aliases if x], []):
                        uid = str(item.get("id") or "")
                        if uid and uid not in cids:
                            cids.append(uid)

            bundle["shot_bindings"][f"S{slot:02d}"] = {"character_ids": cids, "object_ids": oids}

        char_lines = _ensure_unique_identity_lines([_compose_identity_line(c, "character") for c in bundle["characters"]])
        object_lines = _ensure_unique_identity_lines([_compose_identity_line(o, "object") for o in bundle["objects"]])
        return char_lines, object_lines, bundle

    def _build_shot_prompts(self, idea: str, style: str, blueprint: Dict[str, Any], beats: List[Dict[str, Any]], character_bibles: List[str], object_bibles: List[str], t2i_model_hint: str, i2v_model_hint: str, continuity_bundle: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Translate locked beats into image/video prompts without ever dropping a slot.

        Batch generation is kept for speed, but exact item count is a hard workflow
        invariant.  If a small model returns an incomplete batch, retry once and then
        generate only the affected shots individually.
        """
        out: List[Dict[str, Any]] = []
        total = len(beats)
        continuity_bundle = continuity_bundle if isinstance(continuity_bundle, dict) else {}
        character_map = {str(c.get("id")): c for c in (continuity_bundle.get("characters") or []) if isinstance(c, dict)}
        object_map = {str(o.get("id")): o for o in (continuity_bundle.get("objects") or []) if isinstance(o, dict)}
        shot_bindings = continuity_bundle.get("shot_bindings") if isinstance(continuity_bundle.get("shot_bindings"), dict) else {}

        system = (
            "You are the shot director inside FrameVision. Story events are LOCKED; your job is to translate each event into an image start frame and a video-motion prompt without changing or duplicating the event. "
            "The start_frame_prompt describes the exact visible state at the BEGINNING of the clip, before the key action is already finished. It must look like an action-ready film frame, not a portrait, fashion photo, posed group shot, poster, or completed-result tableau. "
            "The video_prompt describes what visibly changes DURING the clip and must execute the locked beat through cause and effect. Do not substitute generic camera drift, blinking, breathing, hair movement, or another angle for story action. "
            "Identity continuity from the supplied shot cast and continuity bibles is mandatory whenever those recurring subjects are present. Use the supplied canonical identities directly instead of re-inventing the appearance. The user's style remains authoritative. "
            "Return JSON only. Every requested locked event must get exactly one result."
        )

        def _shot_subjects(shot_no: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            rec = shot_bindings.get(f"S{shot_no:02d}") if isinstance(shot_bindings.get(f"S{shot_no:02d}"), dict) else {}
            chars = [character_map[cid] for cid in (rec.get("character_ids") or []) if cid in character_map]
            objs = [object_map[oid] for oid in (rec.get("object_ids") or []) if oid in object_map]
            return chars, objs

        def _shot_cast_block(shot_no: int) -> str:
            chars, objs = _shot_subjects(shot_no)
            data = {
                "present_characters": chars,
                "present_objects": objs,
                "continuity_requirements": _identity_prefix_for_shot(chars, objs),
            }
            return json.dumps(data, ensure_ascii=False, indent=2)

        def _normalize(item: Dict[str, Any], beat: Dict[str, Any], shot_no: int) -> Dict[str, Any]:
            sf = _clean_line(item.get("start_frame_prompt"))
            vp = _clean_line(item.get("video_prompt"))
            if not sf or not vp:
                raise RuntimeError(f"Shot {shot_no} is missing start_frame_prompt or video_prompt.")
            chars, objs = _shot_subjects(shot_no)
            sf = _inject_bound_identities(sf, chars, objs)
            vp = _clean_line(vp)
            return {
                "section": beat["section"], "purpose": _clean_line(item.get("purpose") or beat["purpose"]),
                "change": _clean_line(item.get("continuity_change") or beat["beat"]),
                "visual": sf, "i2v": vp, "beat": beat["beat"], "reference_ids": beat.get("reference_ids") or [],
                "present_character_ids": [str(c.get("id")) for c in chars],
                "present_object_ids": [str(o.get("id")) for o in objs],
            }
        def _one_shot(beat: Dict[str, Any], shot_no: int, previous: List[Dict[str, Any]]) -> Dict[str, Any]:
            prev_block = f"""PREVIOUS LOCKED EVENTS FOR CONTINUITY ONLY:
{json.dumps(previous, ensure_ascii=False, indent=2)}

""" if previous else ""
            event_block = f"""LOCKED EVENT FOR SHOT {shot_no}:
{json.dumps(beat, ensure_ascii=False, indent=2)}

"""
            user = (
                f"""USER IDEA:
{idea}

STYLE:
{style or '[none]'}

LOCKED STORY SUMMARY:
{blueprint['story_summary']}

CHARACTER BIBLES:
{json.dumps(character_bibles, ensure_ascii=False, indent=2)}

OBJECT BIBLES:
{json.dumps(object_bibles, ensure_ascii=False, indent=2)}

STRUCTURED CONTINUITY BIBLE:
{json.dumps(continuity_bundle, ensure_ascii=False, indent=2)}

THIS SHOT CAST REQUIREMENT:
{_shot_cast_block(shot_no)}

TARGET IMAGE MODEL: {t2i_model_hint or '[unspecified]'}
TARGET VIDEO MODEL: {i2v_model_hint or '[unspecified]'}

"""
                + prev_block
                + event_block
                + 'Return exactly this JSON object: {"shot":{"purpose":"...","continuity_change":"...","start_frame_prompt":"...","video_prompt":"..."}}'
            )
            obj = self._json_call(system, user, f"shot prompts {shot_no}", max_tokens=3200)
            item = None
            if isinstance(obj, dict) and isinstance(obj.get("shot"), dict):
                item = obj["shot"]
            elif isinstance(obj, dict) and isinstance(obj.get("shots"), list) and len(obj["shots"]) == 1 and isinstance(obj["shots"][0], dict):
                item = obj["shots"][0]
            elif isinstance(obj, dict) and (obj.get("start_frame_prompt") or obj.get("video_prompt")):
                item = obj
            if not isinstance(item, dict):
                raise RuntimeError(f"Shot {shot_no} did not return one shot object.")
            return _normalize(item, beat, shot_no)
        for start in range(0, total, 5):
            chunk = beats[start:start+5]
            previous = beats[max(0, start-2):start]
            cast_requirements = [{
                "shot": start + i + 1,
                "binding": json.loads(_shot_cast_block(start + i + 1)),
            } for i, _ in enumerate(chunk)]
            prev_block = f"""PREVIOUS LOCKED EVENTS FOR CONTINUITY ONLY:
{json.dumps(previous, ensure_ascii=False, indent=2)}

""" if previous else ""
            base_user = (
                f"""USER IDEA:
{idea}

STYLE:
{style or '[none]'}

LOCKED STORY SUMMARY:
{blueprint['story_summary']}

CHARACTER BIBLES:
{json.dumps(character_bibles, ensure_ascii=False, indent=2)}

OBJECT BIBLES:
{json.dumps(object_bibles, ensure_ascii=False, indent=2)}

STRUCTURED CONTINUITY BIBLE:
{json.dumps(continuity_bundle, ensure_ascii=False, indent=2)}

SHOT CAST REQUIREMENTS BY SHOT:
{json.dumps(cast_requirements, ensure_ascii=False, indent=2)}

TARGET IMAGE MODEL: {t2i_model_hint or '[unspecified]'}
TARGET VIDEO MODEL: {i2v_model_hint or '[unspecified]'}

"""
                + prev_block
                + f"""LOCKED EVENTS TO DIRECT NOW:
{json.dumps(chunk, ensure_ascii=False, indent=2)}

"""
            )

            raw = None
            for semantic_attempt in range(1, 3):
                correction = "" if semantic_attempt == 1 else (
                    f"\n\nCORRECTION: Your previous valid JSON had the wrong number of shots. "
                    f"Return EXACTLY {len(chunk)} shot objects, one per locked event, in the same order. Do not omit, merge, or combine events."
                )
                try:
                    obj = self._json_call(
                        system,
                        base_user + correction + f"\n\nReturn JSON as {{\"shots\":[...]}} with exactly {len(chunk)} shot objects in the same order.",
                        f"shot prompts {start+1}-{start+len(chunk)}",
                        max_tokens=6800,
                    )
                except Exception:
                    self._log(f"[story] Prompt batch {start+1}-{start+len(chunk)} could not produce valid JSON; switching this batch to individual shots.")
                    break
                candidate = obj.get("shots") if isinstance(obj, dict) else None
                if isinstance(candidate, list) and len(candidate) == len(chunk) and all(isinstance(x, dict) for x in candidate):
                    raw = candidate
                    break
                got = len(candidate) if isinstance(candidate, list) else 0
                self._log(f"[story] Prompt batch {start+1}-{start+len(chunk)} returned {got}/{len(chunk)} shots; retrying.")

            if raw is None:
                self._log(f"[story] Prompt batch {start+1}-{start+len(chunk)} is incomplete; generating those shots individually.")
                for i, beat in enumerate(chunk):
                    shot_no = start + i + 1
                    prev = beats[max(0, shot_no-3):shot_no-1]
                    out.append(_one_shot(beat, shot_no, prev))
                continue

            for i, item in enumerate(raw):
                beat = chunk[i]
                shot_no = start + i + 1
                try:
                    out.append(_normalize(item, beat, shot_no))
                except Exception:
                    self._log(f"[story] Prompt for shot {shot_no} is incomplete; regenerating that shot.")
                    prev = beats[max(0, shot_no-3):shot_no-1]
                    out.append(_one_shot(beat, shot_no, prev))

        if len(out) != total:
            raise RuntimeError(f"Prompt creation produced {len(out)} shots; expected {total}.")
        return out

    def generate_project(self, *, title: str, idea: str, shot_count: int, include_story_outline: bool = True,
                         generate_t2i: bool = True, generate_i2v: bool = True, style_hint: str = "",
                         negative_hint: str = "", use_character_bible: bool = True, use_object_bible: bool = True,
                         t2i_model_hint: str = "", i2v_model_hint: str = "", predefined_character_bibles: Optional[List[str]] = None,
                         target_duration_sec: float = 0.0, **extra: Any) -> StoryProject:
        reference_guidance = _clean_line(extra.get("reference_guidance") or "")
        audio_context = _clean_line(extra.get("audio_context") or "")
        idea = _clean_line(idea)
        style_hint = _clean_line(style_hint)
        if not idea:
            raise RuntimeError("Planner idea is empty.")
        shot_count = max(1, int(shot_count))
        target_duration_sec = float(target_duration_sec or shot_count * 5.0)

        self._log(f"[story] Story pipeline: blueprint -> shot list -> continuity bibles -> image/video prompts ({shot_count} clips)")
        self._log("[story] Creating blueprint")
        blueprint = None
        last_blueprint_error = None
        for blueprint_attempt in range(1, 3):
            try:
                blueprint = self._build_blueprint(idea, style_hint, shot_count, target_duration_sec, reference_guidance, audio_context)
                break
            except Exception as exc:
                last_blueprint_error = exc
                self._log(f"[story] blueprint semantic validation attempt {blueprint_attempt}/2 failed: {exc}")
        if not isinstance(blueprint, dict):
            raise RuntimeError(f"Blueprint failed semantic validation: {last_blueprint_error}")
        self._log("[story] Locked blueprint: " + " | ".join(f"{s['title']}={s['clip_count']}" for s in blueprint['story_sections']))

        self._log("[story] Creating shot list")
        beats = None
        beat_feedback = ""
        last_beat_error = None
        for beat_attempt in range(1, 3):
            try:
                beats = self._build_beats(idea, style_hint, blueprint, shot_count, beat_feedback)
                break
            except Exception as exc:
                last_beat_error = exc
                beat_feedback = str(exc)
                self._log(f"[story] beat quality attempt {beat_attempt}/2 failed: {exc}")
        if not isinstance(beats, list) or len(beats) != shot_count:
            raise RuntimeError(f"Beat generation failed quality validation: {last_beat_error}")
        for b in beats:
            self._log(f"[story] beat {b['slot']:02d} [{b['section']}]: {b['beat']}")

        self._log("[story] Creating continuity bibles")
        chars, objects, continuity_bundle = self._build_bibles(idea, style_hint, blueprint, beats, use_character_bible, use_object_bible, predefined_character_bibles, extra.get("predefined_character_entries"))

        self._log("[story] Creating image and video prompts")
        directed = self._build_shot_prompts(idea, style_hint, blueprint, beats, chars, objects, t2i_model_hint, i2v_model_hint, continuity_bundle)

        t2i = [d["visual"] for d in directed] if generate_t2i else []
        i2v = [d["i2v"] for d in directed] if generate_i2v else []
        outline = [b["beat"] for b in beats] if include_story_outline else []
        story_bible = [f"{s['title']} [{s['role']}]: {s['purpose']} MUST: {s['must_achieve']}" for s in blueprint["story_sections"]]
        shot_plan = []
        per = target_duration_sec / max(1, shot_count)
        for idx, d in enumerate(directed, 1):
            shot_plan.append({
                "shot": idx, "beat_index": idx, "section": d["section"], "purpose": d["purpose"],
                "change": d["change"], "visual": d["visual"], "duration_sec": round(per, 3),
                "reference_ids": d.get("reference_ids") or [],
                "present_character_ids": d.get("present_character_ids") or [],
                "present_object_ids": d.get("present_object_ids") or [],
            })

        return StoryProject(
            title=_clean_line(blueprint.get("title") or title or "Planner Story"),
            idea=idea,
            shot_count=shot_count,
            story_outline=outline,
            character_bibles=chars,
            object_bibles=objects,
            text_to_image_prompts=t2i,
            image_to_video_prompts=i2v,
            metadata={
                "engine": "agent_story_replacement_v1",
                "architecture": "blueprint->locked_beats->continuity_bibles->shot_prompts",
                "style_hint": style_hint,
                "negative_hint": negative_hint,
                "t2i_model_hint": t2i_model_hint,
                "i2v_model_hint": i2v_model_hint,
                "blueprint": blueprint,
                "reference_guidance_present": bool(reference_guidance),
                "audio_context_present": bool(audio_context),
                "continuity_bundle": continuity_bundle,
                "story_scale": {
                    "shot_count": int(shot_count),
                    "target_duration_sec": float(target_duration_sec),
                    "continuity_bible_max_tokens": int(min(16000, max(4200, 2200 + (len(beats) * 110) + (len((blueprint or {}).get("recurring_subjects") or []) * 180)))),
                "story_section_guidance": list(self._story_section_guidance(target_duration_sec, shot_count)),
                },
            },
            story_bible=story_bible,
            narrative_beats=[b["beat"] for b in beats],
            shot_plan=shot_plan,
        )
