from __future__ import annotations

"""Shared MiniMax H3 music-video creative director.

This module contains the LLM/creative planning layer only. It intentionally knows
nothing about PySide6, FrameVision queueing, MiniMax generation, or final assembly.
That makes the same director reusable by both the standalone MiniMax music creator
and the FrameVision-imported widget.
"""

import json
import math
import os
import random
import re
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Neutral camera vocabulary used only when Auto Fill Camera / Effects is enabled.
# Keep these descriptions camera-only: no people, characters, performers, choreography,
# props, or scene events. This avoids turning camera direction into a second action
# prompt, which can destabilize realistic MiniMax H3 subjects.
AUTO_CAMERA_LIBRARY: Tuple[str, ...] = (
    "low-angle forward tracking shot",
    "low-angle lateral tracking shot",
    "high-angle lateral tracking shot",
    "smooth circular orbit around the central focal area",
    "slow controlled push-in",
    "slow controlled pull-back reveal",
    "steady backward tracking movement",
    "smooth forward dolly movement",
    "smooth lateral dolly movement",
    "diagonal tracking movement",
    "vertical crane rise revealing more of the environment",
    "slow vertical crane descent",
    "overhead top-down camera drift",
    "locked-off symmetrical wide frame",
    "wide static establishing frame",
    "close framing with subtle parallax movement",
    "shallow-depth close detail with gentle rack focus",
    "controlled handheld camera drift",
    "smooth stabilized glide through the environment",
    "subtle dutch-angle tracking shot",
    "slow cinematic arc movement",
    "circular orbit while gradually pulling back",
    "circular orbit while gradually pushing in",
    "rapid push-in timed to a musical impact",
    "long-lens compressed tracking shot",
)


def _randomized_camera_toolbox() -> str:
    items = list(AUTO_CAMERA_LIBRARY)
    random.shuffle(items)
    return "\n".join(f"- {item}" for item in items)


def _fallback_camera_choice() -> str:
    return random.choice(AUTO_CAMERA_LIBRARY) if AUTO_CAMERA_LIBRARY else ""


def _music_director_llama_settings(root: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """Read the same own-llama selection used by the new offline Planner.

    MiniMax Music Clip Creator intentionally does not import planner.py: planner.py is
    a very large GUI module and importing it from this standalone helper would create
    unnecessary coupling and circular-import risk inside FrameVision.  We share only
    its small persisted own-llama contract instead.
    """
    root = Path(root).resolve()
    planner_settings_path = root / "presets" / "setsave" / "planner_settings.json"
    data: Dict[str, Any] = {}
    try:
        if planner_settings_path.is_file():
            raw = json.loads(planner_settings_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = raw
    except Exception as exc:
        return None, f"could not read Planner LLM settings: {exc}"
    if not bool(data.get("own_llama_enabled", False)):
        return None, "Planner own llama is not enabled"

    runner = str(data.get("own_llama_runner_path") or "").strip()
    if not runner:
        try:
            bin_dir = root / "presets" / "bin"
            for name in ("llama-server.exe", "llama-server", "server.exe", "server"):
                found = next((x for x in bin_dir.rglob(name) if x.is_file()), None) if bin_dir.is_dir() else None
                if found is not None:
                    runner = str(found.resolve())
                    break
        except Exception:
            runner = ""
    if runner and "server" not in Path(runner).name.lower():
        folder = Path(runner).parent
        for name in ("llama-server.exe", "llama-server", "server.exe", "server"):
            candidate = folder / name
            if candidate.is_file():
                runner = str(candidate.resolve())
                break

    model = str(data.get("own_llama_model_path") or "").strip()
    if not runner or not Path(runner).is_file():
        return None, f"Planner llama-server was not found: {runner or '[empty]'}"
    if not model or not Path(model).is_file():
        return None, f"Planner GGUF model was not found: {model or '[empty]'}"
    return {
        "runner": str(Path(runner).resolve()),
        "model": str(Path(model).resolve()),
        "template_kind": str(data.get("own_llama_template_kind") or "smart").strip().lower() or "smart",
        "template_value": str(data.get("own_llama_template_value") or "").strip(),
        "top_p": float(data.get("own_llama_top_p", 0.9) or 0.9),
        "log_path": str(root / "logs" / "minimax_music_director_llama.log"),
    }, ""


def _music_director_template(model_path: str, kind: str, value: str) -> Tuple[str, str]:
    if kind != "smart":
        return kind, value
    hay = str(model_path or "").lower()
    checks = [
        (("llama-4", "llama4"), ("builtin", "llama4")),
        (("llama-3", "llama3", "meta-llama-3"), ("builtin", "llama3")),
        (("qwen",), ("builtin", "chatml")),
        (("deepseek-r1", "deepseek-v3", "deepseek3"), ("builtin", "deepseek3")),
        (("chatglm4", "glm-4", "glm4"), ("builtin", "chatglm4")),
        (("gemma",), ("builtin", "gemma")),
        (("mistral", "mixtral", "ministral", "magistral"), ("builtin", "mistral-v7")),
        (("gpt-oss", "gpt_oss"), ("builtin", "gpt-oss")),
    ]
    for needles, result in checks:
        if any(x in hay for x in needles):
            return result
    return "auto", ""


def _music_http_get_json(url: str, timeout: float = 8.0) -> Tuple[int, Dict[str, Any]]:
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


def _music_http_post_json(url: str, payload: Dict[str, Any], timeout: float = 600.0) -> Tuple[int, Dict[str, Any]]:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=raw, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return int(resp.getcode()), json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body) if body.strip() else {}
        except Exception:
            data = {"error": {"message": body or str(exc)}}
        return int(exc.code), data


def _music_strip_protocol(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<analysis>.*?</analysis>", "", text)
    text = text.replace("```json", "```")
    return text.strip()


def _music_parse_json(text: str) -> Dict[str, Any]:
    clean = _music_strip_protocol(text).strip()
    if clean.startswith("```") and clean.endswith("```"):
        clean = clean[3:-3].strip()
    try:
        obj = json.loads(clean)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = clean.find("{")
    end = clean.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(clean[start:end + 1])
        if isinstance(obj, dict):
            return obj
    raise RuntimeError("Model response did not contain parseable JSON.")


class _MusicDirectorLlamaSession:
    """One temporary llama-server process reused for all retries of one music plan."""

    def __init__(self, cfg: Dict[str, Any], shot_count: int):
        self.cfg = dict(cfg)
        self.shot_count = max(1, int(shot_count or 1))
        self.proc: Optional[subprocess.Popen] = None
        self.base_url = ""
        self.log_path = Path(str(self.cfg.get("log_path") or "minimax_music_director_llama.log"))

    def __enter__(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
        sock.close()
        self.base_url = f"http://127.0.0.1:{port}"

        # The plan is compact compared with a full storyline, but leave enough room
        # for a three-to-five minute song with many clip directives.
        ctx_size = max(8192, min(32768, int(math.ceil((7000 + self.shot_count * 320) / 1024.0) * 1024)))
        template_kind, template_value = _music_director_template(
            self.cfg["model"], self.cfg.get("template_kind", "smart"), self.cfg.get("template_value", "")
        )
        base = ["-m", self.cfg["model"], "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx_size)]
        template_args: List[str] = []
        if template_kind == "jinja":
            template_args.append("--jinja")
        elif template_kind == "builtin" and template_value:
            template_args += ["--chat-template", str(template_value)]
        attempts = [base + ["--reasoning-budget", "0"] + template_args, base + template_args]
        if template_args:
            attempts.append(base)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        last_tail = ""
        for attempt_no, args in enumerate(attempts, 1):
            try:
                if self.proc is not None and self.proc.poll() is None:
                    self.proc.terminate(); self.proc.wait(timeout=3)
            except Exception:
                pass
            with open(self.log_path, "w", encoding="utf-8", errors="replace") as log:
                log.write(f"[minimax music director] attempt {attempt_no}\n")
                log.write(f"runner: {self.cfg['runner']}\nmodel: {self.cfg['model']}\n")
                log.write("args: " + json.dumps(args, ensure_ascii=False) + "\n\n")
            log_handle = open(self.log_path, "a", encoding="utf-8", errors="replace")
            try:
                self.proc = subprocess.Popen(
                    [self.cfg["runner"]] + args,
                    stdout=log_handle, stderr=subprocess.STDOUT,
                    cwd=str(Path(self.cfg["runner"]).parent), creationflags=creationflags,
                )
            finally:
                log_handle.close()
            start = time.time()
            while time.time() - start <= 240.0:
                if self.proc.poll() is not None:
                    try:
                        last_tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2200:]
                    except Exception:
                        last_tail = ""
                    break
                try:
                    code, _payload = _music_http_get_json(self.base_url + "/health", timeout=3.0)
                    if code == 200:
                        return self
                except Exception:
                    pass
                time.sleep(0.8)
        raise RuntimeError("Local llama-server could not start for MiniMax Music Director. " + (last_tail or "Check the director log."))

    def generate_json(self, system: str, user: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
        sys_text = (
            "DIRECT STRUCTURED OUTPUT MODE. Never reveal reasoning. Never emit <think> tags. "
            "Start with { and return only one valid JSON object. " + str(system or "")
        )
        payload = {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": str(user or "").rstrip() + "\n\n/no_think"},
            ],
            "stream": False,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(self.cfg.get("top_p", 0.9) or 0.9),
            "reasoning_format": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_object"},
        }
        timeout = float(max(150.0, min(300.0, 120.0 + max_tokens * 0.03)))
        code, data = _music_http_post_json(self.base_url + "/v1/chat/completions", payload, timeout=timeout)
        if code >= 400:
            payload.pop("chat_template_kwargs", None)
            payload.pop("response_format", None)
            code, data = _music_http_post_json(self.base_url + "/v1/chat/completions", payload, timeout=timeout)
        if code >= 400:
            msg = ((data or {}).get("error") or {}).get("message") or f"HTTP {code}"
            raise RuntimeError(str(msg))
        choices = (data or {}).get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by local llama-server.")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(str(x.get("text") or "") if isinstance(x, dict) else str(x) for x in content)
        return _music_parse_json(str(content))

    def __exit__(self, exc_type, exc, tb):
        if self.proc is not None:
            try:
                self.proc.terminate(); self.proc.wait(timeout=8)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None


def _clean_line(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _safe_progress(progress: Callable[[str], None], message: str) -> None:
    try:
        progress(str(message))
    except Exception:
        pass


def _director_output_dir(project: Any, root: Path) -> Path:
    raw = str(getattr(project, "output_dir", "") or "").strip()
    if raw:
        out = Path(raw)
    else:
        title = re.sub(r"[^A-Za-z0-9._-]+", "_", str(getattr(project, "title", "") or "music_video")).strip("_") or "music_video"
        out = Path(root) / "output" / "video" / "Minimax" / title
    try:
        out.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return out


def _save_stage(project: Any, root: Path, name: str, payload: Any) -> None:
    """Best-effort debug/inspection artifact. Never fail the director because saving failed."""
    try:
        path = _director_output_dir(project, root) / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _music_section_count(duration: float, shot_count: int) -> int:
    """Creative section count, intentionally independent from raw analyzer micro-sections."""
    duration = max(1.0, float(duration or 0.0))
    shots = max(1, int(shot_count or 1))
    if duration < 75:
        target = 4
    elif duration < 120:
        target = 5
    elif duration < 180:
        target = 7
    elif duration < 240:
        target = 8
    elif duration < 330:
        target = 9
    else:
        target = max(9, min(12, int(round(duration / 35.0))))
    return max(1, min(shots, target))


def _lock_section_budget(sections: List[Dict[str, Any]], total: int, target_count: int) -> List[Dict[str, Any]]:
    """Same invariant used by the storyline planner: sections survive, clip total is exact."""
    cleaned = [dict(x) for x in sections if isinstance(x, dict)]
    if not cleaned:
        raise RuntimeError("Blueprint returned no music sections.")
    if len(cleaned) != target_count:
        raise RuntimeError(f"Blueprint returned {len(cleaned)}/{target_count} creative sections.")
    if len(cleaned) > total:
        raise RuntimeError(f"Blueprint created {len(cleaned)} sections for only {total} clips.")

    counts = []
    for sec in cleaned:
        try:
            counts.append(max(1, int(sec.get("clip_count") or 1)))
        except Exception:
            counts.append(1)
    delta = int(total) - sum(counts)
    if delta > 0:
        # Prefer adding clips to development/middle sections; then cycle all sections.
        order = list(range(1, max(1, len(counts) - 1))) or list(range(len(counts)))
        pos = 0
        while delta > 0:
            counts[order[pos % len(order)]] += 1
            delta -= 1
            pos += 1
    elif delta < 0:
        need = -delta
        order = [len(counts) - 1] + sorted(range(max(0, len(counts) - 1)), key=lambda i: (-counts[i], i))
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
            raise RuntimeError("Could not reconcile music blueprint section budget with requested clip count.")

    for idx, (sec, count) in enumerate(zip(cleaned, counts), 1):
        sec["section_index"] = idx
        sec["clip_count"] = int(count)
        sec["name"] = _clean_line(sec.get("name")) or f"Section {idx}"
        sec["purpose"] = _clean_line(sec.get("purpose"))
        sec["location"] = _clean_line(sec.get("location"))
        sec["performance_goal"] = _clean_line(sec.get("performance_goal"))
        sec["visual_progression"] = _clean_line(sec.get("visual_progression"))
    return cleaned


def _slot_targets(sections: List[Dict[str, Any]], shot_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    slot = 1
    total = len(shot_rows)
    for sec in sections:
        count = int(sec.get("clip_count") or 0)
        for local_i in range(1, count + 1):
            if slot > total:
                break
            row = shot_rows[slot - 1]
            targets.append({
                "shot": slot,
                "section_index": int(sec.get("section_index") or 1),
                "section_name": str(sec.get("name") or "Section"),
                "section_beat": f"{local_i}/{count}",
                "section_purpose": str(sec.get("purpose") or ""),
                "section_location": str(sec.get("location") or ""),
                "section_performance_goal": str(sec.get("performance_goal") or ""),
                "section_visual_progression": str(sec.get("visual_progression") or ""),
                "start": row["start"],
                "end": row["end"],
                "duration": row["duration"],
                "analysis_kind": row.get("analysis_kind", ""),
                "lyrics": row.get("lyrics", ""),
                "internal_cut_times": row.get("internal_cut_times", []),
            })
            slot += 1
    if len(targets) != total:
        raise RuntimeError(f"Music blueprint produced {len(targets)} slot targets; expected {total}.")
    return targets


def _fallback_blueprint(project: Any, shot_count: int, section_count: int) -> Dict[str, Any]:
    """Only used if blueprint JSON fails; later LLM stages can still run."""
    locations = [x.strip() for x in re.split(r"[\n;]+", str(getattr(project, "locations_world", "") or "")) if x.strip()]
    base = shot_count // section_count
    extra = shot_count % section_count
    sections = []
    for i in range(section_count):
        count = base + (1 if i < extra else 0)
        loc = locations[i % len(locations)] if locations else ""
        sections.append({
            "section_index": i + 1,
            "name": f"Music section {i + 1}",
            "clip_count": count,
            "purpose": "Develop the music-video performance with visible progression.",
            "location": loc,
            "performance_goal": "Keep performers physically engaged with the music and vary staging as energy develops.",
            "visual_progression": "Increase or relax visual intensity according to the local song energy.",
        })
    return {
        "concept_summary": _clean_line(getattr(project, "main_idea", "")) or "Music-driven performance video directed from the song structure.",
        "visual_rules": [_clean_line(getattr(project, "style_theme", ""))] if _clean_line(getattr(project, "style_theme", "")) else [],
        "music_sections": sections,
        "fallback": True,
    }


def _music_director_task(progress, project: Any, *, root: Path, clean_lyric: Callable[[str], str], normalize_ref_kind: Callable[[str], str]) -> Dict[str, Any]:
    """Planner-style MiniMax music director.

    Mirrors the proven storyline workflow instead of inventing a second monolithic
    planner: blueprint -> locked creative sections -> shot list in batches -> prompt
    directions in batches. Successful earlier stages are kept when a later batch fails.
    """
    cfg, reason = _music_director_llama_settings(root)
    if cfg is None:
        _safe_progress(progress, "Music Director unavailable; using deterministic fallback (" + reason + ").")
        return {"used_llm": False, "warning": reason, "shots": []}

    total = len(getattr(project, "shots", []) or [])
    if total <= 0:
        return {"used_llm": False, "warning": "No locked music clips exist.", "shots": []}

    refs = []
    for ref in getattr(project, "references", []) or []:
        if not getattr(ref, "enabled", False):
            continue
        refs.append({
            "name": str(getattr(ref, "name", "") or ""),
            "kind": normalize_ref_kind(getattr(ref, "kind", "")),
            "description": _clean_line(getattr(ref, "description", "")),
        })
    valid_ref_names = {r["name"] for r in refs if r.get("name")}

    shot_rows: List[Dict[str, Any]] = []
    for shot in project.shots:
        shot_rows.append({
            "shot": int(shot.index),
            "start": round(float(shot.edit_start), 3),
            "end": round(float(shot.edit_end), 3),
            "duration": round(float(shot.edit_duration), 3),
            "analysis_kind": str(getattr(shot, "section", "") or ""),
            "lyrics": clean_lyric(getattr(shot, "lyrics", "")),
            "internal_cut_times": [round(float(x - shot.generation_start), 3) for x in list(getattr(shot, "internal_cuts", []) or [])[:4]],
        })

    lyric_lines: List[str] = []
    if bool(getattr(project, "whisper_timing_enabled", False)):
        for seg in getattr(project, "lyrics", []) or []:
            text = clean_lyric(getattr(seg, "text", ""))
            if text:
                lyric_lines.append(f"{float(seg.start):.2f}-{float(seg.end):.2f}: {text}")
    lyric_context = "\n".join(lyric_lines)

    duration = float(getattr(getattr(project, "analysis", None), "duration", 0.0) or 0.0)
    if duration <= 0 and shot_rows:
        duration = float(shot_rows[-1]["end"])
    section_count = _music_section_count(duration, total)

    creative_brief = f"""USER IDEA (may be empty):
{getattr(project, 'main_idea', '') or '[empty - derive visual direction from the song and lyrics]'}

STYLE / THEME (may be empty):
{getattr(project, 'style_theme', '') or '[empty]'}

CHARACTERS / SUBJECT RULES:
{getattr(project, 'characters_subjects', '') or '[none]'}

USER LOCATION MATERIAL:
{getattr(project, 'locations_world', '') or '[none]'}
AUTO FILL LOCATIONS: {'ON' if bool(getattr(project, 'auto_fill_locations', False)) else 'OFF'}

USER CAMERA / CHOREOGRAPHY MATERIAL:
{getattr(project, 'camera_choreography', '') or '[none]'}
AUTO FILL CAMERA / EFFECTS: {'ON' if bool(getattr(project, 'auto_fill_camera', False)) else 'OFF'}

AVAILABLE REFERENCES:
{json.dumps(refs, ensure_ascii=False, indent=2)}

WHISPER LYRIC CONTEXT:
{lyric_context or '[no Whisper lyrics - direct from musical structure and energy]'}
"""

    system = (
        "You are the music-video planner inside FrameVision. The song timing and clip count are already locked. "
        "Music performance is the priority: favor convincing dancing, singing/lip-sync when lyrics exist, band performance, rhythmic body movement, prop interaction or other beat-driven physical action. "
        "Do not fill clips with passive people staring at the camera. User instructions are authoritative. Expand sparse input intelligently without replacing a detailed user idea. "
        "Lyrics are semantic inspiration, not mandatory literal illustration. Repeated choruses may intentionally reuse a signature location, choreography motif or framing with escalation. "
        "Camera text must describe camera geometry/motion only. Never name, reference, follow, approach, reveal, switch to, or introduce a person, character, performer, dancer, subject, prop, or new scene event inside the camera field. "
        "Never write MiniMax token syntax. Return JSON only and never reveal reasoning."
    )

    llm_generated_any = False
    fallback_shots: List[int] = []

    with _MusicDirectorLlamaSession(cfg, total) as session:
        # ---------- Stage 1: BLUEPRINT ----------
        _safe_progress(progress, f"Music Director: {duration:.1f}s track -> {total} locked MiniMax clips.")
        _safe_progress(progress, f"Music Director: creating blueprint with {section_count} creative sections...")
        blueprint_prompt = creative_brief + f"""
LOCKED CLIP TIMELINE SUMMARY:
{json.dumps(shot_rows, ensure_ascii=False, indent=2)}

Create the WHOLE music-video blueprint only. Do NOT write individual clip actions yet.
The raw audio analyzer may contain many micro-changes; ignore those as creative sections. Create EXACTLY {section_count} coherent creative sections for the {total} locked clips.

Rules:
1. music_sections must contain exactly {section_count} entries in chronological order.
2. clip_count values must sum to exactly {total}. Every section gets at least one clip.
3. A section is a creative movement of the video, not every tiny beat/energy change.
4. With Auto Fill Locations ON, create/choose one primary environment per creative section, guided by the user's idea/theme and supplied locations. Supplied locations are ingredients, not a forced round-robin list.
5. With Auto Fill Locations OFF, stay within user supplied location material; leave location empty when none was supplied.
6. performance_goal must describe what makes that section musically active. Dancing/performance should escalate or relax with the song rather than become generic posing.
7. visual_progression explains how this section differs from the previous one.
8. If the brief is empty, derive the concept from lyrics when available, otherwise from the song structure/energy.

Return exactly:
{{"concept_summary":"...","visual_rules":["..."],"music_sections":[{{"name":"...","clip_count":1,"purpose":"...","location":"...","performance_goal":"...","visual_progression":"..."}}]}}
"""
        blueprint: Optional[Dict[str, Any]] = None
        last_error: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                if attempt > 1:
                    _safe_progress(progress, f"Music Director: blueprint attempt {attempt}/3...")
                obj = session.generate_json(system, blueprint_prompt, max_tokens=2400, temperature=0.34 if attempt == 1 else 0.20)
                raw_sections = obj.get("music_sections") if isinstance(obj, dict) else None
                if not isinstance(raw_sections, list):
                    raise RuntimeError("blueprint is missing music_sections")
                locked_sections = _lock_section_budget(raw_sections, total, section_count)
                blueprint = {
                    "concept_summary": _clean_line(obj.get("concept_summary")),
                    "visual_rules": [_clean_line(x) for x in (obj.get("visual_rules") or []) if _clean_line(x)],
                    "music_sections": locked_sections,
                    "fallback": False,
                }
                llm_generated_any = True
                break
            except Exception as exc:
                last_error = exc
                _safe_progress(progress, f"Music Director: blueprint attempt {attempt}/3 failed: {exc}")
        if blueprint is None:
            _safe_progress(progress, f"Music Director: blueprint LLM failed; using a simple {section_count}-section scaffold and continuing ({last_error}).")
            blueprint = _fallback_blueprint(project, total, section_count)

        _save_stage(project, root, "music_director_blueprint.json", blueprint)
        _save_stage(project, root, "music_director_sections.json", blueprint.get("music_sections") or [])
        _safe_progress(progress, "Music Director: blueprint locked.")
        for sec in blueprint["music_sections"]:
            _safe_progress(progress, f"  Section {sec['section_index']}: {sec['name']} -> {sec['clip_count']} clip{'s' if int(sec['clip_count']) != 1 else ''} | {sec.get('location') or '[location inherited/unspecified]'}")

        targets = _slot_targets(blueprint["music_sections"], shot_rows)

        # ---------- Stage 2: SHOT LIST ----------
        _safe_progress(progress, "Music Director: creating locked shot list...")
        beats: List[Dict[str, Any]] = []

        def normalize_beat(item: Any, target: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(item, dict):
                raise RuntimeError(f"Clip {target['shot']} beat is not an object")
            try:
                idx = int(item.get("shot"))
            except Exception:
                raise RuntimeError(f"Clip {target['shot']} beat has no shot number")
            if idx != int(target["shot"]):
                raise RuntimeError(f"Expected clip {target['shot']} but received {idx}")
            beat = _clean_line(item.get("beat") or item.get("action"))
            if not beat:
                raise RuntimeError(f"Clip {idx} returned an empty beat")
            mode = _clean_line(item.get("performance_mode"))
            refs_out = [str(x).strip() for x in (item.get("reference_names") or []) if str(x).strip() in valid_ref_names][:9]
            return {
                "shot": idx,
                "section_index": int(target["section_index"]),
                "section_name": target["section_name"],
                "section_beat": target["section_beat"],
                "beat": beat,
                "performance_mode": mode,
                "reference_names": refs_out,
            }

        def fallback_beat(target: Dict[str, Any]) -> Dict[str, Any]:
            mode = "music-driven physical performance"
            lyric = _clean_line(target.get("lyrics"))
            if lyric:
                action = f"Perform the current lyric phrase with convincing rhythmic body movement while visibly developing the {target['section_name']} section."
                mode = "vocal performance with rhythmic movement"
            else:
                action = f"Perform energetic beat-driven full-body action that visibly develops the {target['section_name']} section rather than posing for the camera."
            return {
                "shot": int(target["shot"]), "section_index": int(target["section_index"]), "section_name": target["section_name"],
                "section_beat": target["section_beat"], "beat": action, "performance_mode": mode, "reference_names": [], "fallback": True,
            }

        def one_beat(target: Dict[str, Any], previous: List[Dict[str, Any]]) -> Dict[str, Any]:
            user = creative_brief + f"""
LOCKED BLUEPRINT:
{json.dumps(blueprint, ensure_ascii=False, indent=2)}

PREVIOUS LOCKED SHOTS FOR CONTINUITY / VARIETY:
{json.dumps(previous[-3:], ensure_ascii=False, indent=2) if previous else '[none]'}

LOCKED TARGET:
{json.dumps(target, ensure_ascii=False, indent=2)}

Write ONE concrete music-video event/action for this clip. It must begin visibly and provide a real performance/staging change, not generic 'moves to the music'. Respect the section location and performance goal. Use only exact available reference names genuinely present.
Return JSON only: {{"beat":{{"shot":{int(target['shot'])},"beat":"...","performance_mode":"...","reference_names":["..."]}}}}
"""
            obj = session.generate_json(system, user, max_tokens=1200, temperature=0.28)
            item = obj.get("beat") if isinstance(obj, dict) else None
            return normalize_beat(item, target)

        for start in range(0, total, 5):
            chunk = targets[start:start + 5]
            first_no, last_no = int(chunk[0]["shot"]), int(chunk[-1]["shot"])
            _safe_progress(progress, f"Music Director: shot list {first_no}-{last_no} of {total}...")
            next_targets = targets[start + len(chunk):start + len(chunk) + 2]
            user = creative_brief + f"""
LOCKED BLUEPRINT:
{json.dumps(blueprint, ensure_ascii=False, indent=2)}

PREVIOUS LOCKED SHOTS FOR CONTINUITY / VARIETY:
{json.dumps(beats[-3:], ensure_ascii=False, indent=2) if beats else '[none]'}

LOCKED TARGETS TO WRITE NOW:
{json.dumps(chunk, ensure_ascii=False, indent=2)}

NEXT TARGETS FOR CONTEXT ONLY (do not return them):
{json.dumps(next_targets, ensure_ascii=False, indent=2) if next_targets else '[end of song]'}

Write exactly one concrete NEW music-video beat for every target. Each beat must create visible performance/staging progression. Favor dancing, singing/lip-sync when lyrics exist, band performance, rhythmic movement, prop interaction, formation changes, environment interaction or other beat-driven physical action. Do not merely change camera angle or describe passive posing. Repeated chorus motifs are allowed when intentionally escalated.

STRICT OUTPUT CONTRACT:
Fill the string/list values in this exact skeleton. Do not add, remove, reorder, duplicate or renumber entries. Keep every shot number exactly as shown.
{json.dumps({"beats": [{"shot": int(t["shot"]), "beat": "", "performance_mode": "", "reference_names": []} for t in chunk]}, ensure_ascii=False, indent=2)}
Return JSON only.
"""
            raw_items: Optional[List[Any]] = None
            batch_error: Optional[Exception] = None
            for attempt in range(1, 4):
                try:
                    if attempt > 1:
                        reason = f" — {batch_error}" if batch_error else ""
                        _safe_progress(progress, f"Music Director: shot list {first_no}-{last_no} retry {attempt}/3{reason}...")
                    obj = session.generate_json(system, user, max_tokens=3000, temperature=0.34 if attempt == 1 else 0.20)
                    cand = obj.get("beats") if isinstance(obj, dict) else None
                    if not isinstance(cand, list) or len(cand) != len(chunk):
                        raise RuntimeError(f"returned {len(cand) if isinstance(cand, list) else 0}/{len(chunk)} beats")
                    normalized = [normalize_beat(item, target) for target, item in zip(chunk, cand)]
                    raw_items = normalized
                    llm_generated_any = True
                    break
                except Exception as exc:
                    batch_error = exc
            if raw_items is None:
                _safe_progress(progress, f"Music Director: shot-list batch {first_no}-{last_no} failed ({batch_error}); generating only this batch individually.")
                raw_items = []
                for target in chunk:
                    try:
                        raw_items.append(one_beat(target, beats + raw_items))
                        llm_generated_any = True
                    except Exception as exc:
                        shot_no = int(target["shot"])
                        _safe_progress(progress, f"Music Director: clip {shot_no} shot-list fallback used: {exc}")
                        raw_items.append(fallback_beat(target))
                        fallback_shots.append(shot_no)
            beats.extend(raw_items)
            _save_stage(project, root, "music_director_shotlist.json", beats)

        _safe_progress(progress, f"Music Director: shot list locked ({len(beats)}/{total}).")

        # ---------- Stage 3: PROMPT DIRECTIONS ----------
        # This mirrors Planner's separate shot-direction pass. The existing MiniMax
        # compiler still owns actual H3 syntax; this stage only supplies creative fields.
        _safe_progress(progress, "Music Director: creating MiniMax clip directions from the locked shot list...")
        directed: List[Dict[str, Any]] = []

        section_by_index = {int(s["section_index"]): s for s in blueprint["music_sections"]}

        def normalize_direction(item: Any, beat: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(item, dict):
                raise RuntimeError(f"Clip {beat['shot']} direction is not an object")
            try:
                idx = int(item.get("shot"))
            except Exception:
                raise RuntimeError(f"Clip {beat['shot']} direction has no shot number")
            if idx != int(beat["shot"]):
                raise RuntimeError(f"Expected clip {beat['shot']} but received {idx}")
            action = _clean_line(item.get("action")) or _clean_line(beat.get("beat"))
            if not action:
                raise RuntimeError(f"Clip {idx} direction has no action")
            refs_out = [str(x).strip() for x in (item.get("reference_names") or beat.get("reference_names") or []) if str(x).strip() in valid_ref_names][:9]
            sec = section_by_index.get(int(beat["section_index"]), {})
            return {
                "shot": idx,
                "section_name": _clean_line(item.get("section_name")) or str(beat.get("section_name") or sec.get("name") or ""),
                "action": action,
                "location": _clean_line(item.get("location")) or str(sec.get("location") or ""),
                "camera": _clean_line(item.get("camera")),
                "performance_mode": _clean_line(item.get("performance_mode")) or _clean_line(beat.get("performance_mode")),
                "reference_names": refs_out,
            }

        def fallback_direction(beat: Dict[str, Any]) -> Dict[str, Any]:
            sec = section_by_index.get(int(beat["section_index"]), {})
            return {
                "shot": int(beat["shot"]), "section_name": str(beat.get("section_name") or sec.get("name") or ""),
                "action": str(beat.get("beat") or ""), "location": str(sec.get("location") or ""),
                "camera": _fallback_camera_choice() if bool(getattr(project, "auto_fill_camera", False)) else "",
                "performance_mode": str(beat.get("performance_mode") or "music-driven performance"),
                "reference_names": list(beat.get("reference_names") or []), "fallback": True,
            }

        def one_direction(beat: Dict[str, Any], previous: List[Dict[str, Any]]) -> Dict[str, Any]:
            sec = section_by_index.get(int(beat["section_index"]), {})
            timeline = shot_rows[int(beat["shot"]) - 1]
            user = creative_brief + f"""
LOCKED BLUEPRINT SECTION:
{json.dumps(sec, ensure_ascii=False, indent=2)}

LOCKED SHOT EVENT:
{json.dumps(beat, ensure_ascii=False, indent=2)}

LOCKED TIMING / LYRICS:
{json.dumps(timeline, ensure_ascii=False, indent=2)}

RECENT DIRECTED CLIPS FOR VARIETY:
{json.dumps(previous[-3:], ensure_ascii=False, indent=2) if previous else '[none]'}

AUTO CAMERA TOOLBOX (randomized order; use only when Auto Fill Camera / Effects is ON):
{_randomized_camera_toolbox() if bool(getattr(project, 'auto_fill_camera', False)) else '[disabled]'}

Translate the locked beat into ONE concrete MiniMax music-video direction without changing its event. Action must describe visible physical movement/performance. Location follows the locked section. With Auto Fill Camera ON, choose one suitable treatment from the neutral camera toolbox and avoid repeating recent camera choices. The camera field describes camera motion/position only: never mention or target people, characters, performers, dancers, subjects, props, or new scene events. With Auto Fill Camera OFF, use only supplied camera/choreography material; camera may be empty.
Return JSON only: {{"shot":{{"shot":{int(beat['shot'])},"section_name":"...","action":"...","location":"...","camera":"...","performance_mode":"...","reference_names":["..."]}}}}
"""
            obj = session.generate_json(system, user, max_tokens=1400, temperature=0.26)
            return normalize_direction(obj.get("shot") if isinstance(obj, dict) else None, beat)

        for start in range(0, total, 5):
            chunk_beats = beats[start:start + 5]
            first_no, last_no = int(chunk_beats[0]["shot"]), int(chunk_beats[-1]["shot"])
            _safe_progress(progress, f"Music Director: prompts {first_no}-{last_no} of {total}...")
            packed = []
            for beat in chunk_beats:
                sec = section_by_index.get(int(beat["section_index"]), {})
                timeline = shot_rows[int(beat["shot"]) - 1]
                packed.append({"beat": beat, "section": sec, "timing": timeline})
            user = creative_brief + f"""
LOCKED WHOLE-VIDEO BLUEPRINT:
{json.dumps(blueprint, ensure_ascii=False, indent=2)}

PREVIOUS DIRECTED CLIPS FOR CONTINUITY / CAMERA VARIETY:
{json.dumps(directed[-3:], ensure_ascii=False, indent=2) if directed else '[none]'}

LOCKED SHOTS TO DIRECT NOW:
{json.dumps(packed, ensure_ascii=False, indent=2)}

AUTO CAMERA TOOLBOX (randomized order; use only when Auto Fill Camera / Effects is ON):
{_randomized_camera_toolbox() if bool(getattr(project, 'auto_fill_camera', False)) else '[disabled]'}

For every locked shot, translate the beat into a concrete MiniMax music-video direction. Do NOT redesign the event or combine shots.
- action: visible physical performance/movement that starts immediately; never generic 'moves to the music'.
- location: follow the locked creative section. Auto Fill Locations may expand within the user's concept; otherwise stay within supplied location material.
- camera: when Auto Fill Camera is ON, choose one suitable treatment from the neutral camera toolbox and vary adjacent clips. Camera text must describe ONLY camera position/motion; never mention or target people, characters, performers, dancers, subjects, props, or new scene events. When OFF, use only user camera material and it may be empty.
- performance_mode: concise description of the intended music performance mode.
- reference_names: only exact available reference names genuinely needed in that clip.

STRICT OUTPUT CONTRACT:
Fill the string/list values in this exact skeleton. Do not add, remove, reorder, duplicate or renumber entries. Keep every shot number exactly as shown.
{json.dumps({"shots": [{"shot": int(b["shot"]), "section_name": "", "action": "", "location": "", "camera": "", "performance_mode": "", "reference_names": []} for b in chunk_beats]}, ensure_ascii=False, indent=2)}
Return JSON only.
"""
            batch_dirs: Optional[List[Dict[str, Any]]] = None
            batch_error: Optional[Exception] = None
            for attempt in range(1, 4):
                try:
                    if attempt > 1:
                        reason = f" — {batch_error}" if batch_error else ""
                        _safe_progress(progress, f"Music Director: prompts {first_no}-{last_no} retry {attempt}/3{reason}...")
                    obj = session.generate_json(system, user, max_tokens=3400, temperature=0.32 if attempt == 1 else 0.18)
                    cand = obj.get("shots") if isinstance(obj, dict) else None
                    if not isinstance(cand, list) or len(cand) != len(chunk_beats):
                        raise RuntimeError(f"returned {len(cand) if isinstance(cand, list) else 0}/{len(chunk_beats)} directions")
                    batch_dirs = [normalize_direction(item, beat) for beat, item in zip(chunk_beats, cand)]
                    llm_generated_any = True
                    break
                except Exception as exc:
                    batch_error = exc
            if batch_dirs is None:
                _safe_progress(progress, f"Music Director: prompt batch {first_no}-{last_no} failed ({batch_error}); directing only this batch individually.")
                batch_dirs = []
                for beat in chunk_beats:
                    try:
                        batch_dirs.append(one_direction(beat, directed + batch_dirs))
                        llm_generated_any = True
                    except Exception as exc:
                        shot_no = int(beat["shot"])
                        _safe_progress(progress, f"Music Director: clip {shot_no} prompt-direction fallback used: {exc}")
                        batch_dirs.append(fallback_direction(beat))
                        if shot_no not in fallback_shots:
                            fallback_shots.append(shot_no)
            directed.extend(batch_dirs)
            _save_stage(project, root, "music_director_prompts.json", directed)

    if len(directed) != total:
        # This should not happen because every failed item gets a local deterministic
        # fallback, but never throw away already good work if an invariant is violated.
        by_shot = {int(x.get("shot")): x for x in directed if isinstance(x, dict) and str(x.get("shot", "")).isdigit()}
        for beat in beats:
            idx = int(beat["shot"])
            if idx not in by_shot:
                by_shot[idx] = fallback_direction(beat)
                if idx not in fallback_shots:
                    fallback_shots.append(idx)
        directed = [by_shot[i] for i in range(1, total + 1)]

    result = {
        "used_llm": bool(llm_generated_any),
        "concept_summary": blueprint.get("concept_summary", ""),
        "visual_rules": blueprint.get("visual_rules", []),
        "section_plan": blueprint.get("music_sections", []),
        "shot_list": beats,
        "shots": directed,
        "fallback_shots": sorted(set(int(x) for x in fallback_shots)),
        "applied_count": len(directed),
    }
    _save_stage(project, root, "music_director_result.json", result)
    _safe_progress(progress, f"Music Director complete: blueprint + {len(blueprint.get('music_sections', []))} sections + {len(beats)}/{total} shot-list items + {len(directed)}/{total} prompt directions.")
    if fallback_shots:
        _safe_progress(progress, "Music Director: local fallback used only for clips " + ", ".join(str(x) for x in sorted(set(fallback_shots))) + ".")
    return result


def music_director_safe_task(progress, project: Any, *, root: Path, clean_lyric: Callable[[str], str], normalize_ref_kind: Callable[[str], str]) -> Dict[str, Any]:
    """Top-level guard. Unlike the old version, never erase partial result files silently."""
    try:
        return _music_director_task(progress, project, root=root, clean_lyric=clean_lyric, normalize_ref_kind=normalize_ref_kind)
    except Exception as exc:
        _safe_progress(progress, f"Music Director failed before a usable plan could be returned: {exc}")
        return {"used_llm": False, "warning": str(exc), "shots": []}


def apply_music_director_result(project: Any, result: Dict[str, Any]) -> bool:
    """Bind returned directions to the existing locked MusicShot objects.

    Valid per-shot rows are applied even when some rows used local fallback. This is
    intentionally not all-or-nothing.
    """
    if not isinstance(result, dict):
        return False
    rows = result.get("shots") if isinstance(result.get("shots"), list) else []
    by_index: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("shot"))
        except Exception:
            continue
        by_index[idx] = row

    applied = 0
    for shot in getattr(project, "shots", []) or []:
        row = by_index.get(int(shot.index))
        if not row:
            continue
        action = str(row.get("action") or "").strip()
        if not action:
            continue
        shot.director_action = action
        shot.director_location = str(row.get("location") or "").strip()
        shot.director_camera = str(row.get("camera") or "").strip()
        shot.director_section = str(row.get("section_name") or "").strip()
        shot.director_performance_mode = str(row.get("performance_mode") or "").strip()
        shot.director_reference_names = [str(x).strip() for x in (row.get("reference_names") or []) if str(x).strip()][:9]
        applied += 1
    result["applied_count"] = applied
    return applied > 0


def assign_music_shot_references(
    project: Any,
    shot: Any,
    *,
    normalize_ref_kind: Callable[[str], str],
    fallback_assign: Callable[[Any, Any], List[str]],
) -> List[str]:
    """Use the director's locked ref set; fall back only for non-directed shots."""
    allowed = {r.name for r in project.references if r.enabled and r.name and Path(r.path).is_file()}
    directed = [x for x in getattr(shot, "director_reference_names", []) if x in allowed]
    if str(getattr(shot, "director_action", "") or "").strip():
        chosen = list(directed)
        # Style/Mood references are safe project-wide guidance and do not create cast.
        for ref in project.references:
            if not ref.enabled or not ref.name or not Path(ref.path).is_file():
                continue
            if normalize_ref_kind(ref.kind) == "Style / Mood" and ref.name not in chosen:
                chosen.append(ref.name)
        return chosen[:9]
    return fallback_assign(project, shot)


__all__ = [
    "music_director_safe_task",
    "apply_music_director_result",
    "assign_music_shot_references",
]
