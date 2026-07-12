"""Shot-timed narration helper for FrameVision Planner.

This module intentionally contains no UI code.  The planner supplies its existing
LLM JSON callback and Qwen3-TTS worker paths, while this helper handles:

* turning the assembled shot timeline into fixed narration windows;
* requesting one coherent, visual-grounded line per window;
* validating/falling back when a small model returns weak JSON;
* synthesising each beat separately;
* fitting and placing each beat on the real video timeline;
* writing a full-length narration WAV, transcript, timed JSON and SRT.

The public entry point is :func:`create_story_narration`.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

SPEECH_ENGINE_VERSION = "planner_speech_v2.1"


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(text)


def _write_json(path: str, obj: Any) -> None:
    _write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _read_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return json.load(handle)
    except Exception:
        return None


def _collapse(text: Any) -> str:
    value = str(text or "").replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _clean_spoken_text(text: Any) -> str:
    value = _collapse(text)
    value = re.sub(r"^```(?:json|text)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```$", "", value)
    value = re.sub(
        r"^\s*(?:beat|segment|scene|shot|narration|voice[- ]?over)\s*[A-Za-z0-9_-]*\s*[:\-]\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = value.strip(" \t\"'")
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    if value and value[-1] not in ".!?…":
        value += "."
    return value


def _words(text: str) -> List[str]:
    return re.findall(r"\S+", _collapse(text))


def _trim_words(text: str, limit: int) -> str:
    limit = max(1, int(limit))
    words = _words(text)
    if len(words) <= limit:
        return _clean_spoken_text(text)
    clipped = " ".join(words[:limit]).rstrip(" ,;:-")
    # Prefer a complete sentence when one exists reasonably near the limit.
    sentence_end = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    if sentence_end >= max(10, int(len(clipped) * 0.58)):
        clipped = clipped[: sentence_end + 1]
    else:
        # A hard cut can leave phrases such as "...path and", "...pursuers
        # have" or "..., leaving". Remove a trailing connector, but close at
        # the previous clause when the final fragment still clearly needs a
        # verb or continuation.
        tokens = clipped.split()
        terminal = re.sub(r"[^A-Za-z']", "", tokens[-1]).lower() if tokens else ""
        connectors = {
            "a", "an", "the", "and", "or", "but", "because", "while", "although",
            "that", "which", "who", "whose", "where", "when", "as", "to", "of",
            "for", "with", "from", "into", "through", "over", "under",
        }
        auxiliaries = {
            "have", "has", "had", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "can", "could", "will", "would", "shall", "should",
            "may", "might", "must",
        }
        if terminal in connectors:
            while tokens:
                tail = re.sub(r"[^A-Za-z']", "", tokens[-1]).lower()
                if tail not in connectors:
                    break
                tokens.pop()
            clipped = " ".join(tokens).rstrip(" ,;:-")
        elif terminal in auxiliaries or (len(terminal) > 5 and terminal.endswith("ing")):
            clause_end = max(clipped.rfind(","), clipped.rfind(";"), clipped.rfind(":"))
            candidate = clipped[:clause_end].rstrip(" ,;:-") if clause_end >= 0 else ""
            floor = max(4, int(round(limit * 0.4)))
            if len(_words(candidate)) >= floor:
                clipped = candidate
            elif tokens:
                tokens.pop()
                clipped = " ".join(tokens).rstrip(" ,;:-")
    return _clean_spoken_text(clipped)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _normalise_shots(shots_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(shots_obj, dict):
        shots_obj = shots_obj.get("shots") or []
    if not isinstance(shots_obj, list):
        return []
    return [dict(item) for item in shots_obj if isinstance(item, dict)]


def _normalise_timeline(timeline_obj: Any) -> List[Dict[str, Any]]:
    if isinstance(timeline_obj, dict):
        timeline_obj = timeline_obj.get("timeline") or []
    if not isinstance(timeline_obj, list):
        return []
    out: List[Dict[str, Any]] = []
    cursor = 0.0
    for index, item in enumerate(timeline_obj, start=1):
        if not isinstance(item, dict):
            continue
        start = _safe_float(item.get("start_sec"), cursor)
        duration = _safe_float(item.get("duration_sec"), 0.0)
        end = _safe_float(item.get("end_sec"), start + duration)
        if end <= start:
            end = start + max(0.1, duration)
        duration = max(0.1, end - start)
        sid = _collapse(item.get("shot_id") or item.get("id") or f"S{index:02d}")
        out.append(
            {
                **item,
                "index": int(item.get("index") or index),
                "shot_id": sid,
                "start_sec": round(start, 3),
                "duration_sec": round(duration, 3),
                "end_sec": round(end, 3),
            }
        )
        cursor = end
    return out


def _shot_summary(shot: Dict[str, Any]) -> str:
    candidates = (
        shot.get("visual_description"),
        shot.get("seed"),
        shot.get("prompt_used"),
        shot.get("notes"),
        shot.get("shot_purpose"),
    )
    text = next((_collapse(v) for v in candidates if _collapse(v)), "")
    # Narration needs content, not render metadata.
    text = re.sub(
        r"\b(?:wide|medium|close[- ]?up|tracking|dolly|pan|tilt|cinematic)\s+(?:shot|view|camera)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:camera|lighting|lens|aspect ratio|fps)\s*[:=][^.;]+[.;]?", "", text, flags=re.IGNORECASE)
    return _collapse(text)[:700]


def _scene_records(shots_obj: Any, timeline_obj: Any) -> List[Dict[str, Any]]:
    shots = _normalise_shots(shots_obj)
    timeline = _normalise_timeline(timeline_obj)
    by_id = {_collapse(item.get("id") or item.get("shot_id")): item for item in shots}
    scenes: List[Dict[str, Any]] = []
    for item in timeline:
        sid = _collapse(item.get("shot_id"))
        shot = by_id.get(sid, {})
        progression = shot.get("story_progression") if isinstance(shot.get("story_progression"), dict) else {}
        scenes.append(
            {
                "shot_id": sid,
                "start_sec": _safe_float(item.get("start_sec")),
                "end_sec": _safe_float(item.get("end_sec")),
                "duration_sec": _safe_float(item.get("duration_sec")),
                "story_section": _collapse(shot.get("story_section") or shot.get("phase") or progression.get("section")),
                "story_role": _collapse(shot.get("story_role") or shot.get("shot_purpose") or shot.get("purpose")),
                "section_change": _collapse(shot.get("section_change") or progression.get("change")),
                "visual": _shot_summary(shot),
            }
        )
    return scenes


def _target_beat_seconds(total_duration: float) -> float:
    if total_duration <= 15.0:
        return 5.0
    if total_duration <= 40.0:
        return 7.0
    if total_duration <= 120.0:
        return 9.0
    return 11.0


def build_narration_windows(shots_obj: Any, timeline_obj: Any, target_duration_sec: float = 0.0) -> List[Dict[str, Any]]:
    """Create deterministic, non-overlapping narration beats from the actual timeline."""
    scenes = _scene_records(shots_obj, timeline_obj)
    if not scenes:
        return []

    timeline_total = max(scene["end_sec"] for scene in scenes)
    total = timeline_total if timeline_total > 0.01 else max(0.1, _safe_float(target_duration_sec))
    # Keep TTS model starts reasonable: narration is timed by story acts, not every tiny clip.
    if total <= 15.0:
        max_beats = 2
    elif total <= 30.0:
        max_beats = 4
    elif total <= 60.0:
        max_beats = 5
    elif total <= 120.0:
        max_beats = 6
    else:
        max_beats = 8
    target = max(_target_beat_seconds(total), total / float(max_beats))
    max_scenes_per_beat = 5 if total <= 60.0 else (7 if total <= 180.0 else 9)
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    for scene in scenes:
        if not current:
            current = [scene]
            continue
        current_duration = scene["end_sec"] - current[0]["start_sec"]
        prev_section = _collapse(current[-1].get("story_section")).lower()
        new_section = _collapse(scene.get("story_section")).lower()
        section_break = bool(prev_section and new_section and prev_section != new_section)
        should_break = (
            (section_break and (current[-1]["end_sec"] - current[0]["start_sec"]) >= 3.5)
            or current_duration > max(12.5, target * 1.45)
            or len(current) >= max_scenes_per_beat
        )
        if should_break:
            groups.append(current)
            current = [scene]
        else:
            current.append(scene)
        if (current[-1]["end_sec"] - current[0]["start_sec"]) >= target and len(current) >= 2:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    # Avoid a tiny orphan beat at the end when it can sensibly join the previous one.
    if len(groups) >= 2:
        last_duration = groups[-1][-1]["end_sec"] - groups[-1][0]["start_sec"]
        combined_duration = groups[-1][-1]["end_sec"] - groups[-2][0]["start_sec"]
        if last_duration < 3.0 and combined_duration <= max(14.5, target * 1.6):
            groups[-2].extend(groups[-1])
            groups.pop()

    # Story-section boundaries are useful, but they must not cause dozens of separate
    # Qwen3-TTS model starts. Merge the shortest adjacent acts until the duration-based
    # ceiling is respected.
    while len(groups) > max_beats:
        best_index = 0
        best_duration = float("inf")
        for idx in range(len(groups) - 1):
            combined = groups[idx + 1][-1]["end_sec"] - groups[idx][0]["start_sec"]
            if combined < best_duration:
                best_duration = combined
                best_index = idx
        groups[best_index].extend(groups[best_index + 1])
        groups.pop(best_index + 1)

    windows: List[Dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        start = group[0]["start_sec"]
        end = group[-1]["end_sec"]
        # Keep tiny natural margins around cuts. The final beat is allowed almost to the end.
        speech_start = start + (0.20 if index == 1 else 0.12)
        speech_end = end - (0.18 if index == len(groups) else 0.12)
        if speech_end - speech_start < 1.0:
            speech_start = start + 0.05
            speech_end = max(speech_start + 0.7, end - 0.05)
        available = max(0.7, speech_end - speech_start)

        # Conservative Qwen3-TTS budget. It is better to leave a brief natural gap than clip speech.
        max_words = max(4, int(math.floor(available * 1.95)))
        min_words = max(3, int(math.floor(max_words * (0.58 if len(groups) > 1 else 0.72))))
        if index == len(groups):
            min_words = max(min_words, int(math.floor(max_words * 0.68)))

        visuals = [scene["visual"] for scene in group if scene.get("visual")]
        windows.append(
            {
                "beat_id": f"B{index:02d}",
                "index": index,
                "start_sec": round(speech_start, 3),
                "end_sec": round(speech_end, 3),
                "available_sec": round(available, 3),
                "shot_ids": [scene["shot_id"] for scene in group],
                "story_section": next((scene["story_section"] for scene in group if scene.get("story_section")), ""),
                "story_role": next((scene["story_role"] for scene in group if scene.get("story_role")), ""),
                "section_change": next((scene["section_change"] for scene in group if scene.get("section_change")), ""),
                "visuals": visuals[:5],
                "min_words": min_words,
                "max_words": max_words,
                "is_first": index == 1,
                "is_final": index == len(groups),
            }
        )
    return windows


def _compact_plan(plan_obj: Any) -> Dict[str, Any]:
    if not isinstance(plan_obj, dict):
        return {}
    keys = (
        "title",
        "title_idea",
        "logline",
        "summary",
        "story_engine",
        "setting",
        "tone",
        "characters",
        "story_arc",
        "arc_sections",
        "beats",
    )
    out: Dict[str, Any] = {}
    for key in keys:
        value = plan_obj.get(key)
        if value not in (None, "", [], {}):
            out[key] = value
    # Limit prompt size for the bundled 2B model.
    encoded = json.dumps(out, ensure_ascii=False)
    if len(encoded) > 9000:
        out = {key: out[key] for key in ("title", "logline", "summary", "setting", "tone") if key in out}
    return out


def _prompt_payload(windows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for window in windows:
        payload.append(
            {
                "beat_id": window["beat_id"],
                "time_window": [window["start_sec"], window["end_sec"]],
                "shot_ids": window["shot_ids"],
                "story_section": window.get("story_section") or "",
                "story_role": window.get("story_role") or "",
                "change": window.get("section_change") or "",
                "visible_events": window.get("visuals") or [],
                "min_words": window["min_words"],
                "max_words": window["max_words"],
                "position": "opening" if window.get("is_first") else ("ending" if window.get("is_final") else "middle"),
            }
        )
    return payload


def _extract_segments(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        value = parsed.get("segments") or parsed.get("narration_segments") or parsed.get("beats")
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if parsed.get("narration_text"):
            return [{"text": parsed.get("narration_text")}]
    return []


def _split_raw_across_windows(raw_text: str, windows: Sequence[Dict[str, Any]]) -> List[str]:
    text = _clean_spoken_text(raw_text)
    words = _words(text)
    if not words:
        return []
    budgets = [max(1, int(window["max_words"])) for window in windows]
    total_budget = max(1, sum(budgets))
    cursor = 0
    result: List[str] = []
    for index, budget in enumerate(budgets):
        if index == len(budgets) - 1:
            take = len(words) - cursor
        else:
            take = max(1, int(round(len(words) * (budget / total_budget))))
        result.append(_clean_spoken_text(" ".join(words[cursor : cursor + take])))
        cursor += take
    return result


def _fallback_line(window: Dict[str, Any], prompt: str, previous: str = "") -> str:
    visuals = [v for v in (window.get("visuals") or []) if _collapse(v)]
    # Open on the first visible action and close on the final one. This keeps the
    # deterministic fallback aligned even when a narration window spans shots.
    if visuals:
        base = visuals[0] if window.get("is_first") else visuals[-1]
    else:
        base = _collapse(prompt)
    base = re.split(r"(?<=[.!?])\s+|[;]", base)[0]
    base = _collapse(base)
    if not base:
        base = "the story moves into its next decisive moment"
    if window.get("is_first"):
        line = f"The story begins as {base[0].lower() + base[1:] if len(base) > 1 else base.lower()}"
    elif window.get("is_final"):
        line = f"At last, {base[0].lower() + base[1:] if len(base) > 1 else base.lower()}, bringing the journey to its close"
    else:
        change = _collapse(window.get("section_change"))
        if change:
            line = f"Then {change[0].lower() + change[1:] if len(change) > 1 else change.lower()}, while {base[0].lower() + base[1:] if len(base) > 1 else base.lower()}"
        else:
            line = f"Next, {base[0].lower() + base[1:] if len(base) > 1 else base.lower()}"
    line = _trim_words(line, int(window.get("max_words") or 12))
    if previous and line.lower() == previous.lower():
        line = _trim_words("The situation changes, and " + base.lower(), int(window.get("max_words") or 12))
    return line


def _validated_script(parsed: Any, raw_text: str, windows: Sequence[Dict[str, Any]], prompt: str) -> List[Dict[str, Any]]:
    incoming = _extract_segments(parsed)
    by_id: Dict[str, str] = {}
    positional: List[str] = []
    for item in incoming:
        text = _clean_spoken_text(item.get("text") or item.get("narration_text") or item.get("line") or item.get("voiceover"))
        if not text:
            continue
        beat_id = _collapse(item.get("beat_id") or item.get("id")).upper()
        if beat_id:
            by_id[beat_id] = text
        positional.append(text)

    if len(positional) == 1 and len(windows) > 1:
        positional = _split_raw_across_windows(positional[0], windows)
    elif not positional and raw_text:
        positional = _split_raw_across_windows(raw_text, windows)

    result: List[Dict[str, Any]] = []
    previous = ""
    for index, window in enumerate(windows):
        text = by_id.get(str(window["beat_id"]).upper())
        if not text and index < len(positional):
            text = positional[index]
        if not text:
            text = _fallback_line(window, prompt, previous)
        text = _trim_words(text, int(window["max_words"]))
        if len(_words(text)) < int(window["min_words"]):
            fallback = _fallback_line(window, prompt, previous)
            if len(_words(fallback)) > len(_words(text)):
                text = fallback
        result.append({**window, "text": text, "word_count": len(_words(text))})
        previous = text
    return result


def _request_script(
    *,
    windows: Sequence[Dict[str, Any]],
    plan_obj: Any,
    prompt: str,
    extra_info: str,
    language: str,
    llm_json_call: Callable[..., Tuple[Optional[Any], str]],
    llm_settings: Optional[Dict[str, Any]],
    prompts_dir: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    os.makedirs(prompts_dir, exist_ok=True)
    raw_path = os.path.join(prompts_dir, "narration_timeline_raw.txt")
    used_path = os.path.join(prompts_dir, "narration_timeline_prompts_used.txt")
    error_path = os.path.join(prompts_dir, "narration_timeline_error.txt")

    system_prompt = (
        "You are the voice-over writer for a generated story video. "
        "Write one coherent narration line for each fixed visual time window. "
        "The narration must tell one continuous story and must match only what the listed shots visibly show. "
        "Return strict JSON only, with no markdown."
    )
    user_prompt = (
        "Write the complete timed voice-over.\n"
        "Rules:\n"
        "- Return exactly one segment for every supplied beat_id, in the same order.\n"
        "- Do not change, invent, merge, or omit beat_id values.\n"
        "- Each text must stay between its min_words and max_words.\n"
        "- Use natural spoken prose, not captions, bullet points, screenplay directions, or image prompts.\n"
        "- Never mention shots, scenes, cameras, prompts, models, transitions, or timestamps.\n"
        "- Ground each line in visible_events; do not describe unseen thoughts or events as facts.\n"
        "- The first line hooks the viewer without repeating the project prompt.\n"
        "- Every middle line advances the story; do not restate the setup.\n"
        "- The final line resolves or deliberately closes the story and is written to be spoken near the video ending.\n"
        "- Keep names, identities, locations, tense, and causal logic consistent across all lines.\n"
        "- Output JSON schema: {\"story_summary\":\"...\",\"segments\":[{\"beat_id\":\"B01\",\"text\":\"...\"}]}.\n\n"
        f"LANGUAGE: {language or 'auto'}\n"
        f"PROJECT_PROMPT: {_collapse(prompt)}\n"
        + (f"EXTRA_CONTEXT: {_collapse(extra_info)}\n" if _collapse(extra_info) else "")
        + f"PLAN: {json.dumps(_compact_plan(plan_obj), ensure_ascii=False)}\n"
        + f"FIXED_BEATS: {json.dumps(_prompt_payload(windows), ensure_ascii=False)}\n"
    )

    parsed: Any = None
    raw_text = ""
    call_error = ""
    try:
        try:
            parsed, raw_text = llm_json_call(
                "Timed narration script",
                system_prompt,
                user_prompt,
                raw_path,
                used_path,
                error_path,
                temperature=0.35,
                max_new_tokens=max(900, len(windows) * 190),
                planner_llama_settings=llm_settings,
            )
        except TypeError:
            parsed, raw_text = llm_json_call(
                "Timed narration script",
                system_prompt,
                user_prompt,
                raw_path,
                used_path,
                error_path,
                0.35,
                max(900, len(windows) * 190),
                llm_settings,
            )
    except Exception as exc:
        call_error = str(exc)
        parsed = None
        raw_text = raw_text or ""

    segments = _validated_script(parsed, raw_text, windows, prompt)

    def _quality(items: Sequence[Dict[str, Any]]) -> float:
        score = 0.0
        seen = set()
        for item in items:
            wc = max(0, int(item.get("word_count") or len(_words(str(item.get("text") or "")))))
            minimum = max(1, int(item.get("min_words") or 1))
            maximum = max(minimum, int(item.get("max_words") or minimum))
            score += min(1.0, wc / float(minimum))
            if wc > maximum:
                score -= 0.25
            key = _collapse(item.get("text")).lower()
            if key in seen:
                score -= 0.5
            seen.add(key)
        return score

    # Small local models occasionally return valid JSON but ignore several word budgets.
    # One focused repair pass is cheaper than discovering the problem after TTS synthesis.
    underfilled = any(int(item.get("word_count") or 0) < int(item.get("min_words") or 0) for item in segments)
    duplicate_count = len({_collapse(item.get("text")).lower() for item in segments}) < len(segments)
    repair_used = False
    if (underfilled or duplicate_count) and segments:
        repair_raw = os.path.join(prompts_dir, "narration_timeline_repair_raw.txt")
        repair_used_path = os.path.join(prompts_dir, "narration_timeline_repair_prompts_used.txt")
        repair_error = os.path.join(prompts_dir, "narration_timeline_repair_error.txt")
        repair_system = (
            "You repair a timed video voice-over. Return strict JSON only. "
            "Preserve every beat_id and rewrite only the spoken text."
        )
        repair_user = (
            "Repair this draft so every segment obeys its own min_words and max_words, remains grounded in visible_events, "
            "does not repeat another segment, and forms one continuous story with a real ending. "
            "Return {\"segments\":[{\"beat_id\":\"B01\",\"text\":\"...\"}]}.\n\n"
            f"LANGUAGE: {language or 'auto'}\n"
            f"FIXED_BEATS: {json.dumps(_prompt_payload(windows), ensure_ascii=False)}\n"
            f"DRAFT_SEGMENTS: {json.dumps([{k: item.get(k) for k in ('beat_id', 'text', 'word_count', 'min_words', 'max_words')} for item in segments], ensure_ascii=False)}\n"
        )
        try:
            try:
                repaired_obj, repaired_raw = llm_json_call(
                    "Timed narration repair",
                    repair_system,
                    repair_user,
                    repair_raw,
                    repair_used_path,
                    repair_error,
                    temperature=0.22,
                    max_new_tokens=max(800, len(windows) * 180),
                    planner_llama_settings=llm_settings,
                )
            except TypeError:
                repaired_obj, repaired_raw = llm_json_call(
                    "Timed narration repair",
                    repair_system,
                    repair_user,
                    repair_raw,
                    repair_used_path,
                    repair_error,
                    0.22,
                    max(800, len(windows) * 180),
                    llm_settings,
                )
            repaired = _validated_script(repaired_obj, repaired_raw, windows, prompt)
            if _quality(repaired) > _quality(segments):
                segments = repaired
                repair_used = True
        except Exception:
            pass

    meta = {
        "llm_error": call_error,
        "raw_path": raw_path,
        "prompts_used_path": used_path,
        "error_path": error_path,
        "used_fallback": bool(call_error or not _extract_segments(parsed)),
        "repair_used": bool(repair_used),
    }
    return segments, meta


def _probe_duration(path: str, ffprobe_path: str) -> float:
    commands = [
        [ffprobe_path, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        [ffprobe_path, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
    ]
    for command in commands:
        try:
            cp = subprocess.run(command, capture_output=True, text=True)
            if cp.returncode == 0:
                value = _safe_float((cp.stdout or "").strip().splitlines()[-1], 0.0)
                if value > 0:
                    return value
        except Exception:
            continue
    return 0.0


def _parse_worker_result(stdout: str) -> str:
    matches = re.findall(r"__RESULT__\s*(\{[^\r\n]*\})", stdout or "")
    for raw in reversed(matches):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("out_path"):
                return str(obj.get("out_path"))
        except Exception:
            pass
    for raw in reversed(re.findall(r"(\{[^\r\n]*\})", stdout or "")):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and obj.get("out_path"):
                return str(obj.get("out_path"))
        except Exception:
            pass
    return ""


def _latest_candidate(folder: str, stem: str, started: float) -> str:
    candidates: List[Path] = []
    root = Path(folder)
    if root.exists():
        for ext in (".wav", ".flac", ".mp3", ".ogg", ".m4a"):
            candidates.extend(root.glob(f"{stem}*{ext}"))
    candidates = [p for p in candidates if p.is_file()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            if path.stat().st_mtime >= started - 3.0:
                return str(path)
        except Exception:
            continue
    return str(candidates[0]) if candidates else ""


def _convert_to_wav(src: str, dst: str, ffmpeg_path: str, log_handle: Any) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if str(src).lower().endswith(".wav"):
        try:
            shutil.copy2(src, dst)
            if os.path.isfile(dst) and os.path.getsize(dst) > 256:
                return
        except Exception:
            pass
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        src,
        "-vn",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-c:a",
        "pcm_s16le",
        dst,
    ]
    log_handle.write("[convert] " + " ".join(command) + "\n")
    cp = subprocess.run(command, capture_output=True, text=True)
    if cp.stdout:
        log_handle.write(cp.stdout + "\n")
    if cp.stderr:
        log_handle.write(cp.stderr + "\n")
    if cp.returncode != 0 or not os.path.isfile(dst):
        raise RuntimeError(f"Could not convert TTS output to WAV: {src}")


def _run_tts_segment(
    *,
    segment: Dict[str, Any],
    job_id: str,
    language: str,
    narration_mode: str,
    voice: str,
    voice_sample_path: str,
    ref_text: str,
    model_path: str,
    tokenizer_path: str,
    qwentts_python: str,
    qwentts_script: str,
    segments_dir: str,
    ffmpeg_path: str,
    log_handle: Any,
    stop_requested: Optional[Callable[[], bool]],
    retry_suffix: str = "",
) -> str:
    if stop_requested and stop_requested():
        raise RuntimeError("Cancelled by user.")
    beat_id = str(segment["beat_id"])
    safe_job = re.sub(r"[^A-Za-z0-9_-]+", "_", str(job_id or "job"))[-48:]
    output_name = f"narration_{safe_job}_{beat_id}{retry_suffix}"
    mode = "clone" if str(narration_mode or "").lower() == "clone" else "custom"
    payload: Dict[str, Any] = {
        "mode": mode,
        "payload": {
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "text": str(segment["text"]),
            "language": str(language or "auto"),
            "speaker": str(voice or "ryan"),
            "ref_audio_path": str(voice_sample_path or ""),
            "common": {
                "output_name": output_name,
                "add_timestamp": False,
                "output_dir": str(segments_dir),
                "output_format": "wav",
            },
        },
    }
    if mode == "clone":
        payload["payload"]["ref_text"] = str(ref_text or "")
        payload["payload"]["x_vector_only_mode"] = not bool(str(ref_text or "").strip())

    command = [str(qwentts_python), "-u", str(qwentts_script), "--worker", "--task", "generate"]
    started = time.time()
    log_handle.write(f"\n[{beat_id}] text={segment['text']}\n")
    log_handle.write(f"[{beat_id}] cmd=" + " ".join(command) + "\n")
    cp = subprocess.run(command, input=json.dumps(payload), capture_output=True, text=True)
    if cp.stdout:
        log_handle.write(cp.stdout + "\n")
    if cp.stderr:
        log_handle.write("[stderr]\n" + cp.stderr + "\n")
    if cp.returncode != 0:
        raise RuntimeError(f"Qwen3 TTS failed for {beat_id} (exit={cp.returncode}).")

    source = _parse_worker_result(cp.stdout or "")
    if source and not os.path.isabs(source):
        source = os.path.abspath(os.path.join(segments_dir, source))
    if not source or not os.path.isfile(source):
        source = _latest_candidate(segments_dir, output_name, started)
    if not source or not os.path.isfile(source):
        raise RuntimeError(f"Qwen3 TTS returned no audio file for {beat_id}.")

    destination = os.path.join(segments_dir, f"{beat_id}{retry_suffix}_raw.wav")
    _convert_to_wav(source, destination, ffmpeg_path, log_handle)
    if not os.path.isfile(destination) or os.path.getsize(destination) < 256:
        raise RuntimeError(f"TTS WAV is missing or empty for {beat_id}.")
    return destination


def _atempo_chain(value: float) -> str:
    value = max(0.5, min(2.0, float(value)))
    return f"atempo={value:.6f}"


def _fit_audio(
    *,
    source: str,
    destination: str,
    source_duration: float,
    available: float,
    ffmpeg_path: str,
    log_handle: Any,
) -> Tuple[float, float, bool]:
    """Fit audio to a beat. Returns (final_duration, tempo, hard_trimmed)."""
    available = max(0.45, float(available))
    source_duration = max(0.01, float(source_duration))
    tempo = 1.0
    hard_trimmed = False

    if source_duration > available * 0.98:
        tempo = min(1.28, source_duration / (available * 0.96))
    elif source_duration < available * 0.62 and source_duration > 1.0:
        # A very small slowdown sounds more natural than a large silent hole.
        tempo = max(0.92, source_duration / (available * 0.70))

    expected = source_duration / tempo
    filters = [_atempo_chain(tempo)] if abs(tempo - 1.0) > 0.005 else []
    if expected > available:
        hard_trimmed = True
        fade_start = max(0.0, available - 0.10)
        filters.extend([f"atrim=0:{available:.4f}", f"afade=t=out:st={fade_start:.4f}:d=0.10"])
        expected = available
    filters.append("aresample=44100")
    filters.append("aformat=sample_fmts=s16:channel_layouts=stereo")

    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        source,
        "-af",
        ",".join(filters),
        "-c:a",
        "pcm_s16le",
        destination,
    ]
    log_handle.write("[fit] " + " ".join(command) + "\n")
    cp = subprocess.run(command, capture_output=True, text=True)
    if cp.stdout:
        log_handle.write(cp.stdout + "\n")
    if cp.stderr:
        log_handle.write(cp.stderr + "\n")
    if cp.returncode != 0 or not os.path.isfile(destination):
        raise RuntimeError(f"Could not fit narration segment: {source}")
    return max(0.01, expected), tempo, hard_trimmed


def _srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(float(seconds) * 1000.0)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _write_srt(path: str, segments: Sequence[Dict[str, Any]]) -> None:
    blocks: List[str] = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            f"{index}\n{_srt_time(segment['scheduled_start_sec'])} --> {_srt_time(segment['scheduled_end_sec'])}\n{segment['text']}"
        )
    _write_text(path, "\n\n".join(blocks).strip() + "\n")


def _compose_track(
    *,
    segments: Sequence[Dict[str, Any]],
    target_duration: float,
    destination: str,
    ffmpeg_path: str,
    log_handle: Any,
) -> None:
    command: List[str] = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-t",
        f"{target_duration:.4f}",
        "-i",
        "anullsrc=r=44100:cl=stereo",
    ]
    for segment in segments:
        command.extend(["-i", str(segment["audio_file"])])

    filters = ["[0:a]volume=0[base]"]
    labels = ["[base]"]
    for index, segment in enumerate(segments, start=1):
        delay_ms = max(0, int(round(float(segment["scheduled_start_sec"]) * 1000.0)))
        label = f"s{index}"
        filters.append(
            f"[{index}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,adelay={delay_ms}|{delay_ms}[{label}]"
        )
        labels.append(f"[{label}]")
    filters.append(
        "".join(labels)
        + f"amix=inputs={len(labels)}:duration=first:dropout_transition=0:normalize=0[out]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-t",
            f"{target_duration:.4f}",
            "-c:a",
            "pcm_s16le",
            destination,
        ]
    )
    log_handle.write("[compose] " + " ".join(command) + "\n")
    cp = subprocess.run(command, capture_output=True, text=True)
    if cp.stdout:
        log_handle.write(cp.stdout + "\n")
    if cp.stderr:
        log_handle.write(cp.stderr + "\n")
    if cp.returncode != 0 or not os.path.isfile(destination) or os.path.getsize(destination) < 256:
        raise RuntimeError("Could not assemble the timed narration track.")


def create_story_narration(
    *,
    root_dir: str,
    job_id: str,
    prompt: str,
    extra_info: str,
    language: str,
    narration_mode: str,
    voice: str,
    voice_sample_path: str,
    ref_text: str,
    plan_obj: Any,
    shots_obj: Any,
    timeline_obj: Any,
    target_duration_sec: float,
    audio_dir: str,
    prompts_dir: str,
    narration_wav: str,
    narration_txt: str,
    narration_json: str,
    transcript_path: str,
    ffmpeg_path: str,
    ffprobe_path: str,
    qwentts_python: str,
    qwentts_script: str,
    tts_model_path: str,
    tts_tokenizer_path: str,
    llm_json_call: Callable[..., Tuple[Optional[Any], str]],
    llm_settings: Optional[Dict[str, Any]] = None,
    log_path: str = "",
    stop_requested: Optional[Callable[[], bool]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Generate a coherent, shot-timed narration track for an assembled story video."""
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)
    segments_dir = os.path.join(audio_dir, "narration_segments")
    os.makedirs(segments_dir, exist_ok=True)
    log_path = log_path or os.path.join(audio_dir, "tts_log.txt")

    timeline = _normalise_timeline(timeline_obj)
    timeline_duration = max((item["end_sec"] for item in timeline), default=0.0)
    # The assembled timeline is the source of truth. A stale requested/manifest duration
    # must never push the final narration beyond the real video ending.
    target_duration = timeline_duration if timeline_duration > 0.01 else _safe_float(target_duration_sec)
    if target_duration <= 0.01:
        raise RuntimeError("Narration cannot be timed because the assembled video duration is unknown.")

    windows = build_narration_windows(shots_obj, timeline, target_duration)
    if not windows:
        raise RuntimeError("Narration cannot be timed because the assembled shot timeline is empty.")

    if log_callback:
        log_callback(f"[speech] planning {len(windows)} timed narration beats across {target_duration:.1f}s")

    script, llm_meta = _request_script(
        windows=windows,
        plan_obj=plan_obj,
        prompt=prompt,
        extra_info=extra_info,
        language=language,
        llm_json_call=llm_json_call,
        llm_settings=llm_settings,
        prompts_dir=prompts_dir,
    )

    spoken_text = "\n\n".join(segment["text"] for segment in script).strip()
    _write_text(narration_txt, spoken_text + "\n")
    # Replace only the old planner placeholder; never overwrite a real Whisper/music transcript.
    transcript_was_placeholder = False
    try:
        old_transcript = ""
        if transcript_path and os.path.isfile(transcript_path):
            with open(transcript_path, "r", encoding="utf-8", errors="replace") as handle:
                old_transcript = handle.read()
        transcript_lower = old_transcript.strip().lower()
        placeholder = (
            not transcript_lower
            or transcript_lower.startswith("transcript (will be generated")
            or transcript_lower.startswith("pending (transcription")
            or ("transcript" in transcript_lower and "generated" in transcript_lower and "later" in transcript_lower)
        )
        if transcript_path and placeholder:
            _write_text(transcript_path, spoken_text + "\n")
            transcript_was_placeholder = True
    except Exception:
        pass

    fitted: List[Dict[str, Any]] = []
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"\n=== {SPEECH_ENGINE_VERSION} ===\n")
        log_handle.write(f"target_duration={target_duration:.3f}s beats={len(script)}\n")
        for index, segment in enumerate(script):
            if stop_requested and stop_requested():
                raise RuntimeError("Cancelled by user.")
            if log_callback:
                log_callback(f"[speech] voice-over beat {index + 1}/{len(script)}: {segment['beat_id']}")

            raw_wav = _run_tts_segment(
                segment=segment,
                job_id=job_id,
                language=language,
                narration_mode=narration_mode,
                voice=voice,
                voice_sample_path=voice_sample_path,
                ref_text=ref_text,
                model_path=tts_model_path,
                tokenizer_path=tts_tokenizer_path,
                qwentts_python=qwentts_python,
                qwentts_script=qwentts_script,
                segments_dir=segments_dir,
                ffmpeg_path=ffmpeg_path,
                log_handle=log_handle,
                stop_requested=stop_requested,
            )
            raw_duration = _probe_duration(raw_wav, ffprobe_path)
            if raw_duration <= 0.01:
                raise RuntimeError(f"Could not measure TTS duration for {segment['beat_id']}.")

            available = max(0.45, float(segment["available_sec"]))
            # A severe overflow is best fixed by one shorter re-synthesis, not a chipmunk tempo.
            if raw_duration > available * 1.32 and len(_words(segment["text"])) > 4:
                reduced_limit = max(4, int(math.floor(len(_words(segment["text"])) * (available / raw_duration) * 0.94)))
                shortened = _trim_words(segment["text"], reduced_limit)
                if shortened and shortened != segment["text"]:
                    segment["text"] = shortened
                    segment["word_count"] = len(_words(shortened))
                    raw_wav = _run_tts_segment(
                        segment=segment,
                        job_id=job_id,
                        language=language,
                        narration_mode=narration_mode,
                        voice=voice,
                        voice_sample_path=voice_sample_path,
                        ref_text=ref_text,
                        model_path=tts_model_path,
                        tokenizer_path=tts_tokenizer_path,
                        qwentts_python=qwentts_python,
                        qwentts_script=qwentts_script,
                        segments_dir=segments_dir,
                        ffmpeg_path=ffmpeg_path,
                        log_handle=log_handle,
                        stop_requested=stop_requested,
                        retry_suffix="_fit",
                    )
                    raw_duration = _probe_duration(raw_wav, ffprobe_path)

            fitted_wav = os.path.join(segments_dir, f"{segment['beat_id']}_fitted.wav")
            final_duration, tempo, hard_trimmed = _fit_audio(
                source=raw_wav,
                destination=fitted_wav,
                source_duration=raw_duration,
                available=available,
                ffmpeg_path=ffmpeg_path,
                log_handle=log_handle,
            )

            start = float(segment["start_sec"])
            end = float(segment["end_sec"])
            if segment.get("is_final"):
                scheduled_start = max(start, end - final_duration)
            else:
                spare = max(0.0, available - final_duration)
                scheduled_start = min(end - final_duration, start + min(0.30, spare * 0.20))
            scheduled_start = max(0.0, scheduled_start)
            scheduled_end = min(target_duration, scheduled_start + final_duration)

            fitted.append(
                {
                    **segment,
                    "audio_file": fitted_wav,
                    "raw_audio_file": raw_wav,
                    "raw_duration_sec": round(raw_duration, 3),
                    "audio_duration_sec": round(final_duration, 3),
                    "tempo": round(tempo, 4),
                    "hard_trimmed": bool(hard_trimmed),
                    "scheduled_start_sec": round(scheduled_start, 3),
                    "scheduled_end_sec": round(scheduled_end, 3),
                }
            )

        _compose_track(
            segments=fitted,
            target_duration=target_duration,
            destination=narration_wav,
            ffmpeg_path=ffmpeg_path,
            log_handle=log_handle,
        )

    # Re-write transcript after any duration-driven text shortening.
    spoken_text = "\n\n".join(segment["text"] for segment in fitted).strip()
    _write_text(narration_txt, spoken_text + "\n")
    if transcript_was_placeholder and transcript_path:
        try:
            _write_text(transcript_path, spoken_text + "\n")
        except Exception:
            pass
    srt_path = os.path.join(audio_dir, "narration.srt")
    timeline_path = os.path.join(audio_dir, "narration_timeline.json")
    _write_srt(srt_path, fitted)

    spoken_seconds = sum(float(segment["audio_duration_sec"]) for segment in fitted)
    last_end = max((float(segment["scheduled_end_sec"]) for segment in fitted), default=0.0)
    result = {
        "engine": SPEECH_ENGINE_VERSION,
        "target_duration_sec": round(target_duration, 3),
        "spoken_duration_sec": round(spoken_seconds, 3),
        "spoken_coverage_ratio": round(spoken_seconds / max(0.001, target_duration), 4),
        "last_spoken_end_sec": round(last_end, 3),
        "last_spoken_end_ratio": round(last_end / max(0.001, target_duration), 4),
        "segment_count": len(fitted),
        "segments": fitted,
        "llm": llm_meta,
        "paths": {
            "narration_wav": narration_wav,
            "narration_txt": narration_txt,
            "narration_json": narration_json,
            "narration_timeline_json": timeline_path,
            "narration_srt": srt_path,
            "segments_dir": segments_dir,
            "tts_log": log_path,
        },
    }
    _write_json(timeline_path, result)
    _write_json(narration_json, result)

    if log_callback:
        log_callback(
            f"[speech] timed narration ready: {len(fitted)} beats, final line ends at {last_end:.1f}/{target_duration:.1f}s"
        )
    return result
