"""FrameVision Planner LTX narration lip-sync helper.

This module intentionally owns only the speech/timing preparation layer.  It does
not launch LTX itself; planner.py keeps using the proven FrameVision LTX CLI and
passes the per-shot WAV files produced here through the official audio-conditioned
arguments.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


LIPSYNC_ENGINE_VERSION = "planner_lipsync_v1.2"
_MAX_TEMPO = 1.28
_BASE_WORDS_PER_SEC = 1.55
# Use the same practical headroom as the real audio fitter.  The old planner
# rejected text at 1.55 words/sec before TTS, even though the allowed gentle
# tempo correction could safely fit almost 2 words/sec.
_ALLOCATION_WORDS_PER_SEC = _BASE_WORDS_PER_SEC * _MAX_TEMPO * 0.98
_PREFLIGHT_USABLE_RATIO = 0.96


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _collapse(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        return handle.read()


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(text)


def _write_json(path: str, obj: Any) -> None:
    _write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


def preflight_uploaded_transcript(transcript_path: str, target_duration_sec: float) -> Dict[str, Any]:
    """Cheap Generate-click validation before story planning or model loading.

    This deliberately blocks only transcripts that cannot fit even with the same
    gentle tempo allowance used by the real fitter.  Borderline-but-feasible text
    is allowed through; exact per-shot fitting still happens later after the shot
    list exists.
    """
    path = str(transcript_path or "").strip()
    target = max(0.0, _safe_float(target_duration_sec))
    if not path or not os.path.isfile(path):
        return {
            "ok": False,
            "reason": "missing",
            "message": "The selected lip-sync transcript file could not be found.",
            "word_count": 0,
            "target_duration_sec": target,
        }

    raw = _read_text(path)
    cues = _parse_timed_transcript(raw)
    if cues:
        spoken_text = _collapse(" ".join(str(item.get("text") or "") for item in cues))
        timed_end = max((_safe_float(item.get("end_sec")) for item in cues), default=0.0)
    else:
        spoken_text = _collapse(" ".join(_sentence_units(raw)))
        timed_end = 0.0
    words = len(spoken_text.split())
    if words <= 0:
        return {
            "ok": False,
            "reason": "empty",
            "message": "The selected lip-sync transcript contains no readable spoken text.",
            "word_count": 0,
            "target_duration_sec": target,
        }

    normal_seconds = words / _BASE_WORDS_PER_SEC
    minimum_text_seconds = words / max(0.1, _ALLOCATION_WORDS_PER_SEC)
    minimum_project_seconds = minimum_text_seconds / _PREFLIGHT_USABLE_RATIO
    if timed_end > 0.0:
        minimum_project_seconds = max(minimum_project_seconds, timed_end)

    # Round the useful recommendation up to the Planner's normal 5-second step.
    recommended = int(math.ceil(max(5.0, minimum_project_seconds) / 5.0) * 5)
    too_long = target <= 0.0 or minimum_project_seconds > target + 0.25

    if too_long:
        if timed_end > target + 0.25:
            detail = (
                f"Its last timed subtitle ends at {timed_end:.1f}s, but the selected project duration is "
                f"{target:.1f}s."
            )
        else:
            detail = (
                f"It has {words} words and needs at least about {minimum_project_seconds:.1f}s, even with "
                f"the allowed gentle speech fitting. The selected project duration is {target:.1f}s."
            )
        message = (
            detail
            + f"\n\nSet the Planner duration to at least {recommended} seconds, or shorten the transcript, "
              "before starting. Nothing has been generated yet."
        )
    else:
        message = ""

    return {
        "ok": not too_long,
        "reason": "too_long" if too_long else "ok",
        "message": message,
        "word_count": words,
        "target_duration_sec": round(target, 3),
        "estimated_normal_speech_sec": round(normal_seconds, 3),
        "minimum_project_duration_sec": round(minimum_project_seconds, 3),
        "recommended_duration_sec": recommended,
        "timed_end_sec": round(timed_end, 3),
        "has_timing": bool(cues),
    }


def _read_json(path: str) -> Any:
    try:
        return json.loads(_read_text(path))
    except Exception:
        return {}


def _sha1_text(text: str) -> str:
    return hashlib.sha1(str(text or "").encode("utf-8", errors="replace")).hexdigest()


def _file_fingerprint(path: str) -> Dict[str, Any]:
    try:
        p = Path(path)
        st = p.stat()
        return {"path": str(p.resolve()), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    except Exception:
        return {"path": str(path or ""), "size": 0, "mtime_ns": 0}


def _file_ok(path: str, minimum: int = 256) -> bool:
    try:
        return bool(path and os.path.isfile(path) and os.path.getsize(path) >= int(minimum))
    except Exception:
        return False


def _stable_llm_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep only narration-writing settings that can change generated speech.

    The Planner encoding dictionary also contains transient resume/review flags.
    Including those flags in the lip-sync fingerprint made every Resume look like
    a new narration job, which restarted TTS and invalidated every clip intent.
    """
    src = settings if isinstance(settings, dict) else {}
    keys = (
        "own_llama_enabled",
        "own_llama_runner_path",
        "own_llama_model_path",
        "own_llama_template_kind",
        "own_llama_template_value",
        "own_llama_ctx_size",
        "own_llama_top_p",
    )
    return {key: src.get(key) for key in keys if key in src}


def _prepared_plan_assets_ok(prior: Any, narration_wav: str) -> bool:
    if not isinstance(prior, dict) or not _file_ok(narration_wav, 512):
        return False
    shot_items = prior.get("shots") if isinstance(prior.get("shots"), list) else []
    if not shot_items:
        return False
    for item in shot_items:
        if not isinstance(item, dict):
            return False
        if bool(item.get("audio_conditioned")) and not _file_ok(str(item.get("audio_file") or "")):
            return False
    return True


def _clean_visual_text(value: Any) -> str:
    text = _collapse(value)
    if not text:
        return ""
    text = re.sub(
        r"\b(?:start from the uploaded image|use this as the visual anchor|focus on this beat|end by)\b[^.;]*[.;]?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    pieces = re.split(r"(?<=[.!?])\s+|\s*;\s*", text)
    kept: List[str] = []
    for piece in pieces:
        piece = _collapse(piece)
        low = piece.lower()
        if not piece:
            continue
        if any(token in low for token in ("negative prompt", "camera move", "transition lora", "aspect ratio", "fps", "render prompt")):
            continue
        if re.match(r"^(?:include|exclude|avoid|do not|don't|enter|show|add|remove|ensure|make sure|end with|start with|focus on)\b", low):
            continue
        kept.append(piece.rstrip(" ."))
        if len(kept) >= 3:
            break
    return _collapse(". ".join(kept))


def _infer_speaker_name(plan_obj: Any, prompt: str = "") -> str:
    plan = plan_obj if isinstance(plan_obj, dict) else {}
    characters = plan.get("characters") if isinstance(plan.get("characters"), list) else []
    for item in characters:
        if not isinstance(item, dict):
            continue
        taxonomy = _collapse(item.get("taxonomy") or item.get("type") or item.get("species")).lower()
        name = _collapse(item.get("name"))
        if name and any(token in taxonomy for token in ("human", "person", "man", "woman", "boy", "girl")):
            return name
    for item in characters:
        if isinstance(item, dict) and _collapse(item.get("name")):
            return _collapse(item.get("name"))
    low = _collapse(prompt).lower()
    for token, label in (("woman", "Woman"), ("man", "Man"), ("girl", "Girl"), ("boy", "Boy"), ("presenter", "Presenter"), ("narrator", "Narrator")):
        if re.search(rf"\b{token}\b", low):
            return label
    return "Primary character"


def _speaker_visual_label(name: str) -> str:
    value = _collapse(name) or "the primary character"
    generic = {
        "man": "the man", "woman": "the woman", "boy": "the boy",
        "girl": "the girl", "presenter": "the presenter", "narrator": "the narrator",
        "primary character": "the primary character",
    }
    return generic.get(value.lower(), value)


def _onscreen_storyteller_indices(count: int, source_mode: str) -> List[int]:
    count = max(0, int(count))
    if count <= 0:
        return []
    if str(source_mode or "").strip().lower().startswith("upload"):
        return list(range(count))
    if count <= 2:
        return list(range(count))
    selected = {0, count - 1}
    if count >= 5:
        step = 3
        for idx in range(step, count - 1, step):
            selected.add(idx)
    return sorted(selected)


def _speaker_ready_from_existing_shot(shot: Dict[str, Any]) -> bool:
    delivery = _collapse(shot.get("lipsync_delivery")).lower()
    if delivery:
        return delivery == "onscreen"
    camera = _collapse(shot.get("camera") or (shot.get("stage_directions") or {}).get("camera")).lower()
    visual = _clean_visual_text(
        shot.get("lipsync_original_visual")
        or shot.get("visual_description")
        or shot.get("seed")
        or shot.get("prompt")
    ).lower()
    if any(token in camera for token in ("over-the-shoulder", "rear", "back view", "wide", "establishing")):
        return False
    human = any(re.search(rf"\b{token}\b", visual) for token in ("man", "woman", "boy", "girl", "person", "presenter", "speaker"))
    speech = any(token in visual for token in ("speaks", "speaking", "talks", "talking", "tells the story", "addresses the camera"))
    readable = any(token in camera for token in ("close-up", "close up", "medium", "portrait"))
    return bool(human and (speech or readable))


def stage_shots_for_lipsync(
    shots_obj: Any,
    plan_obj: Any,
    prompt: str = "",
    source_mode: str = "generated",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Stage selected shots as visible storyteller shots before images are generated."""
    shots = _normalise_shots(shots_obj)
    if not shots:
        return [], {"speaker_name": "", "onscreen_shot_ids": []}
    speaker = _infer_speaker_name(plan_obj, prompt)
    speaker_visual = _speaker_visual_label(speaker)
    setting = _collapse((plan_obj or {}).get("setting") if isinstance(plan_obj, dict) else "")
    setting = re.sub(r"^outdoors\s*,?\s*", "", setting, flags=re.IGNORECASE).strip()
    selected = set(_onscreen_storyteller_indices(len(shots), source_mode))
    onscreen_ids: List[str] = []
    for index, shot in enumerate(shots):
        original = _clean_visual_text(
            shot.get("lipsync_original_visual")
            or shot.get("visual_description")
            or shot.get("seed")
            or shot.get("prompt")
        )
        shot["lipsync_original_visual"] = original
        shot["lipsync_speaker_name"] = speaker
        shot["lipsync_speaker_visual_label"] = speaker_visual
        if index in selected:
            sid = _collapse(shot.get("id") or f"S{index + 1:02d}")
            onscreen_ids.append(sid)
            if setting:
                location = f" {setting}" if re.match(r"^(?:in|at|near|by|beside|on|inside|outside)\b", setting, flags=re.IGNORECASE) else f" in {setting}"
            else:
                location = " in the established location"
            context = original or "the story setting and its important visual details"
            staged = (
                f"Medium close-up of {speaker_visual}{location}, facing the camera in a natural three-quarter view with a clear, "
                f"unobstructed face and fully visible mouth, speaking as the on-screen storyteller. "
                f"The surrounding story context remains visible behind the speaker: {context}. "
                "One speaking character only; do not duplicate the speaker or cover the mouth."
            )
            shot["lipsync_delivery"] = "onscreen"
            shot["visual_description"] = staged
            shot["seed"] = staged
            shot["camera"] = "medium close-up"
            stage = shot.get("stage_directions") if isinstance(shot.get("stage_directions"), dict) else {}
            stage = dict(stage)
            stage["camera"] = "medium close-up"
            stage["lipsync_delivery"] = "onscreen storyteller"
            shot["stage_directions"] = stage
            note = _collapse(shot.get("notes"))
            shot["notes"] = _collapse(
                (note + " " if note else "")
                + f"On-screen storyteller: {speaker} speaks with a readable face and visible mouth; preserve the story context as background detail."
            )
        else:
            shot["lipsync_delivery"] = "voiceover_broll"
            note = _collapse(shot.get("notes"))
            shot["notes"] = _collapse(
                (note + " " if note else "")
                + f"Voice-over B-roll for {speaker}; no visible character should appear to speak in this shot."
            )
    return shots, {
        "speaker_name": speaker,
        "onscreen_shot_ids": onscreen_ids,
        "voiceover_shot_ids": [str(item.get("id") or "") for item in shots if item.get("lipsync_delivery") == "voiceover_broll"],
        "source_mode": "uploaded" if str(source_mode or "").lower().startswith("upload") else "generated",
    }


def _normalise_shots(shots_obj: Any, fallback_duration: float = 5.0) -> List[Dict[str, Any]]:
    if isinstance(shots_obj, dict):
        shots_obj = shots_obj.get("shots") or []
    if not isinstance(shots_obj, list):
        return []
    out: List[Dict[str, Any]] = []
    for index, raw in enumerate(shots_obj, start=1):
        if not isinstance(raw, dict):
            continue
        shot = dict(raw)
        sid = _collapse(shot.get("id") or shot.get("shot_id") or f"S{index:02d}")
        duration = _safe_float(shot.get("duration_sec"), fallback_duration)
        if duration <= 0.05:
            duration = fallback_duration
        shot["id"] = sid
        shot["duration_sec"] = round(max(0.1, duration), 3)
        out.append(shot)
    return out


def build_shot_timeline(shots_obj: Any, target_duration_sec: float = 0.0) -> List[Dict[str, Any]]:
    shots = _normalise_shots(shots_obj)
    if not shots:
        return []
    total_raw = sum(_safe_float(item.get("duration_sec"), 0.0) for item in shots)
    requested = max(0.0, _safe_float(target_duration_sec))
    scale = 1.0
    if requested > 0.1 and total_raw > 0.1 and abs(requested - total_raw) / max(requested, total_raw) > 0.20:
        scale = requested / total_raw
    cursor = 0.0
    timeline: List[Dict[str, Any]] = []
    for index, shot in enumerate(shots, start=1):
        duration = max(0.1, _safe_float(shot.get("duration_sec"), 5.0) * scale)
        timeline.append(
            {
                "index": index,
                "shot_id": str(shot.get("id") or f"S{index:02d}"),
                "start_sec": round(cursor, 3),
                "duration_sec": round(duration, 3),
                "end_sec": round(cursor + duration, 3),
            }
        )
        cursor += duration
    return timeline



def _shot_lookup(shots_obj: Any) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("id") or ""): item for item in _normalise_shots(shots_obj) if str(item.get("id") or "")}


def _shot_visual_text(shot: Dict[str, Any]) -> str:
    for key in (
        "lipsync_original_visual", "visual_description", "seed", "prompt_used",
        "i2v_prompt", "video_prompt", "prompt", "visual", "description", "action", "image_prompt",
    ):
        value = _clean_visual_text(shot.get(key))
        if value:
            return value
    return ""


def _build_lipsync_windows(shots_obj: Any, timeline: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One speech window per usable LTX shot, so TTS never crosses a clip cut."""
    lookup = _shot_lookup(shots_obj)
    windows: List[Dict[str, Any]] = []
    usable = [item for item in timeline if _safe_float(item.get("duration_sec")) >= 1.15]
    for index, item in enumerate(usable, start=1):
        sid = str(item.get("shot_id") or "")
        shot = lookup.get(sid, {})
        shot_start = _safe_float(item.get("start_sec"))
        shot_end = _safe_float(item.get("end_sec"))
        duration = max(0.1, shot_end - shot_start)
        first = index == 1
        final = index == len(usable)
        left_margin = 0.18 if first else 0.10
        right_margin = 0.08 if final else 0.12
        start = shot_start + min(left_margin, duration * 0.12)
        end = shot_end - min(right_margin, duration * 0.10)
        if end - start < 0.72:
            start = shot_start + 0.04
            end = shot_end - 0.04
        available = max(0.55, end - start)
        # Conservative budget: the line should fit without cutting or extreme tempo.
        max_words = max(3, min(20, int(math.floor(available * _ALLOCATION_WORDS_PER_SEC))))
        min_words = max(2, int(math.floor(max_words * (0.55 if not final else 0.65))))
        windows.append(
            {
                "beat_id": f"L{index:02d}",
                "index": index,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "available_sec": round(available, 3),
                "shot_ids": [sid],
                "story_section": _collapse(shot.get("story_section") or shot.get("section")),
                "story_role": _collapse(shot.get("story_role") or shot.get("role")),
                "section_change": _collapse(shot.get("section_change") or shot.get("change")),
                "visuals": [_shot_visual_text(shot)] if _shot_visual_text(shot) else [],
                "speech_delivery": _collapse(shot.get("lipsync_delivery") or ("onscreen" if _speaker_ready_from_existing_shot(shot) else "voiceover_broll")),
                "speaker_name": _collapse(shot.get("lipsync_speaker_name") or ""),
                "min_words": min_words,
                "max_words": max_words,
                "is_first": first,
                "is_final": final,
            }
        )
    return windows


def _choose_word_boundary(tokens: Sequence[str], start: int, desired: int, maximum: int) -> int:
    maximum = max(1, min(int(maximum), len(tokens) - start))
    desired = max(1, min(int(desired), maximum))
    for distance in range(0, 5):
        for candidate in (desired + distance, desired - distance):
            if 1 <= candidate <= maximum:
                token = str(tokens[start + candidate - 1])
                if re.search(r"[.!?…][\"')\]]?$", token):
                    return candidate
    return desired


def _even_window_subset(windows: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    if not windows or count <= 0:
        return []
    count = min(len(windows), max(1, int(count)))
    if count == len(windows):
        return [dict(item) for item in windows]
    if count == 1:
        return [dict(windows[-1])]
    indices: List[int] = []
    for pos in range(count):
        idx = int(round(pos * (len(windows) - 1) / float(count - 1)))
        if idx not in indices:
            indices.append(idx)
    return [dict(windows[idx]) for idx in indices]


def _allocate_plain_text_to_windows(text: str, windows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokens = _collapse(text).split()
    if not tokens or not windows:
        return []
    units = _sentence_units(text)
    desired_segments = max(1, min(len(windows), max(len(units), int(math.ceil(len(tokens) / 7.0)))))
    selected = _even_window_subset(windows, desired_segments)
    capacity = sum(int(item.get("max_words") or 0) for item in selected)
    # A sentence-count-based subset can be too small for a perfectly usable
    # transcript.  Expand across every planned shot before declaring failure.
    if len(tokens) > capacity and len(selected) < len(windows):
        selected = [dict(item) for item in windows]
        capacity = sum(int(item.get("max_words") or 0) for item in selected)
    if len(tokens) > capacity:
        approximate = len(tokens) / _BASE_WORDS_PER_SEC
        minimum = len(tokens) / max(0.1, _ALLOCATION_WORDS_PER_SEC)
        available = sum(_safe_float(item.get("available_sec")) for item in selected)
        raise RuntimeError(
            f"The transcript still cannot fit the completed shot timeline: {len(tokens)} words need at least about "
            f"{minimum:.1f}s after gentle fitting, while {available:.1f}s is available. "
            "This should normally be caught before generation; increase the Planner duration and resume. "
            "No words were removed."
        )
    segments: List[Dict[str, Any]] = []
    cursor = 0
    remaining_words = len(tokens)
    remaining_capacity = capacity
    for index, window in enumerate(selected, start=1):
        if cursor >= len(tokens):
            break
        cap = max(1, int(window.get("max_words") or 1))
        windows_left = len(selected) - index
        if windows_left <= 0:
            take = remaining_words
        else:
            proportional = int(round(remaining_words * cap / float(max(1, remaining_capacity))))
            minimum_left = windows_left
            take = max(1, min(cap, proportional, remaining_words - minimum_left if remaining_words > minimum_left else 1))
        take = _choose_word_boundary(tokens, cursor, take, min(cap, remaining_words))
        chunk = _collapse(" ".join(tokens[cursor : cursor + take]))
        cursor += take
        remaining_words = len(tokens) - cursor
        remaining_capacity -= cap
        segments.append({**window, "beat_id": f"U{index:02d}", "index": index, "text": chunk, "word_count": len(chunk.split())})
    if cursor < len(tokens) and segments:
        tail = _collapse(" ".join(tokens[cursor:]))
        segments[-1]["text"] = _collapse(str(segments[-1].get("text") or "") + " " + tail)
        segments[-1]["word_count"] = len(str(segments[-1]["text"]).split())
    for index, segment in enumerate(segments, start=1):
        segment["is_first"] = index == 1
        segment["is_final"] = index == len(segments)
    return segments


def _split_tokens_by_weights(tokens: Sequence[str], weights: Sequence[float]) -> List[List[str]]:
    if not tokens or not weights:
        return []
    positive = [max(0.001, float(value)) for value in weights]
    total_weight = sum(positive)
    output: List[List[str]] = []
    cursor = 0
    for index, weight in enumerate(positive):
        remaining = len(tokens) - cursor
        pieces_left = len(positive) - index
        if remaining <= 0:
            output.append([])
            continue
        if pieces_left <= 1:
            take = remaining
        else:
            ideal = int(round(len(tokens) * weight / total_weight))
            take = max(1, min(remaining - (pieces_left - 1), ideal)) if remaining >= pieces_left else 1
        output.append(list(tokens[cursor : cursor + take]))
        cursor += take
    if cursor < len(tokens) and output:
        output[-1].extend(tokens[cursor:])
    return output


def _timed_cues_to_shot_segments(
    cues: Sequence[Dict[str, Any]],
    timeline: Sequence[Dict[str, Any]],
    windows: Sequence[Dict[str, Any]],
    target_duration: float,
) -> List[Dict[str, Any]]:
    if not cues or not timeline or not windows:
        return []
    window_map = {str(item.get("shot_ids", [""])[0]): item for item in windows if item.get("shot_ids")}
    last_end = max(_safe_float(cue.get("end_sec")) for cue in cues)
    scale = (max(0.7, target_duration - 0.12) / max(0.1, last_end)) if last_end > 0.1 else 1.0
    scale = max(0.25, min(4.0, scale))
    segments: List[Dict[str, Any]] = []
    serial = 0
    for cue_index, cue in enumerate(cues, start=1):
        cue_start = max(0.02, _safe_float(cue.get("start_sec")) * scale)
        cue_end = min(target_duration - 0.02, _safe_float(cue.get("end_sec")) * scale)
        if cue_end <= cue_start + 0.15:
            cue_end = min(target_duration - 0.01, cue_start + 0.45)
        overlaps: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
        for shot in timeline:
            sid = str(shot.get("shot_id") or "")
            window = window_map.get(sid)
            if not window:
                continue
            overlap = min(cue_end, _safe_float(shot.get("end_sec"))) - max(cue_start, _safe_float(shot.get("start_sec")))
            if overlap > 0.03:
                overlaps.append((shot, window, overlap))
        if not overlaps:
            midpoint = (cue_start + cue_end) * 0.5
            nearest = min(
                ((shot, window_map.get(str(shot.get("shot_id") or ""))) for shot in timeline),
                key=lambda pair: abs(((_safe_float(pair[0].get("start_sec")) + _safe_float(pair[0].get("end_sec"))) * 0.5) - midpoint),
            )
            if nearest[1]:
                overlaps = [(nearest[0], nearest[1], 1.0)]
        tokens = _collapse(cue.get("text")).split()
        pieces = _split_tokens_by_weights(tokens, [item[2] for item in overlaps])
        for piece_index, ((shot, window, _weight), words) in enumerate(zip(overlaps, pieces), start=1):
            if not words:
                continue
            start = max(cue_start, _safe_float(window.get("start_sec")))
            end = min(cue_end, _safe_float(window.get("end_sec")))
            if end - start < 0.55:
                start = _safe_float(window.get("start_sec"))
                end = _safe_float(window.get("end_sec"))
            serial += 1
            text = _collapse(" ".join(words))
            segments.append(
                {
                    **window,
                    "beat_id": f"U{serial:02d}",
                    "index": serial,
                    "text": text,
                    "word_count": len(words),
                    "start_sec": round(start, 3),
                    "end_sec": round(end, 3),
                    "available_sec": round(max(0.45, end - start), 3),
                    "shot_ids": [str(shot.get("shot_id") or "")],
                    "source_cue": cue_index,
                    "source_piece": piece_index,
                }
            )
    segments.sort(key=lambda item: (_safe_float(item.get("start_sec")), int(item.get("index") or 0)))
    for index, segment in enumerate(segments, start=1):
        segment["index"] = index
        segment["is_first"] = index == 1
        segment["is_final"] = index == len(segments)
    return segments

def _timestamp_seconds(token: str) -> float:
    token = str(token or "").strip().replace(",", ".")
    parts = token.split(":")
    try:
        if len(parts) == 3:
            return max(0.0, float(parts[0]) * 3600.0 + float(parts[1]) * 60.0 + float(parts[2]))
        if len(parts) == 2:
            return max(0.0, float(parts[0]) * 60.0 + float(parts[1]))
        return max(0.0, float(parts[0]))
    except Exception:
        return 0.0


def _parse_timed_transcript(text: str) -> List[Dict[str, Any]]:
    clean = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    clean = re.sub(r"^\ufeff?WEBVTT[^\n]*\n", "", clean, flags=re.IGNORECASE)
    blocks = re.split(r"\n\s*\n", clean)
    cues: List[Dict[str, Any]] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        timing_index = next((idx for idx, line in enumerate(lines) if "-->" in line), -1)
        if timing_index < 0:
            continue
        left, right = [part.strip() for part in lines[timing_index].split("-->", 1)]
        right = right.split()[0] if right.split() else right
        start = _timestamp_seconds(left)
        end = _timestamp_seconds(right)
        cue_text = _collapse(" ".join(lines[timing_index + 1 :]))
        cue_text = re.sub(r"<[^>]+>", "", cue_text).strip()
        if cue_text and end > start + 0.05:
            cues.append({"text": cue_text, "start_sec": start, "end_sec": end})
    return cues


def _sentence_units(text: str) -> List[str]:
    clean = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    clean = re.sub(r"^\ufeff?WEBVTT[^\n]*\n", "", clean, flags=re.IGNORECASE)
    units: List[str] = []
    for paragraph in re.split(r"\n\s*\n+", clean):
        paragraph = _collapse(paragraph)
        if not paragraph:
            continue
        parts = re.split(r"(?<=[.!?…])\s+|\s*\n\s*", paragraph)
        for part in parts:
            part = _collapse(re.sub(r"^\d+\s*$", "", part))
            if part and "-->" not in part:
                units.append(part)
    if not units and _collapse(clean):
        units = [_collapse(clean)]
    expanded: List[str] = []
    for unit in units:
        words = unit.split()
        if len(words) <= 32:
            expanded.append(unit)
            continue
        for start in range(0, len(words), 28):
            expanded.append(" ".join(words[start : start + 28]).strip())
    return [item for item in expanded if item]


def _allocate_units_to_windows(units: Sequence[str], windows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not units or not windows:
        return []
    word_counts = [max(1, len(item.split())) for item in units]
    total_words = sum(word_counts)
    total_available = sum(max(0.1, _safe_float(window.get("available_sec"), 0.1)) for window in windows)
    segments: List[Dict[str, Any]] = []
    unit_index = 0
    words_used = 0
    for index, window in enumerate(windows, start=1):
        if unit_index >= len(units):
            break
        if index == len(windows):
            take_to = len(units)
        else:
            cumulative_available = sum(
                max(0.1, _safe_float(windows[pos].get("available_sec"), 0.1)) for pos in range(index)
            )
            target_words = max(words_used + 1, int(round(total_words * cumulative_available / max(0.1, total_available))))
            take_to = unit_index
            running = words_used
            while take_to < len(units):
                next_words = word_counts[take_to]
                if take_to > unit_index and running + next_words > target_words:
                    break
                running += next_words
                take_to += 1
            if take_to <= unit_index:
                take_to = unit_index + 1
        text = _collapse(" ".join(units[unit_index:take_to]))
        words_used += sum(word_counts[unit_index:take_to])
        unit_index = take_to
        segments.append(
            {
                "beat_id": f"U{index:02d}",
                "index": index,
                "text": text,
                "start_sec": _safe_float(window.get("start_sec")),
                "end_sec": _safe_float(window.get("end_sec")),
                "available_sec": max(0.45, _safe_float(window.get("available_sec"), 0.45)),
                "shot_ids": list(window.get("shot_ids") or []),
                "is_final": index == len(windows) or unit_index >= len(units),
            }
        )
    if unit_index < len(units) and segments:
        segments[-1]["text"] = _collapse(segments[-1]["text"] + " " + " ".join(units[unit_index:]))
    if segments:
        for idx, item in enumerate(segments, start=1):
            item["is_final"] = idx == len(segments)
    return segments


def _strict_fit_audio(
    *,
    source: str,
    destination: str,
    source_duration: float,
    available: float,
    ffmpeg_path: str,
    log_handle: Any,
    label: str,
) -> Tuple[float, float]:
    available = max(0.45, float(available))
    source_duration = max(0.01, float(source_duration))
    # Keep a small end margin, but do not reject a line that genuinely fits.
    # The old 3% margin could fail a 9.4s line inside a 9.6s shot even though
    # the allowed gentle tempo adjustment was sufficient.
    fit_window = max(0.01, available * 0.99)
    tempo = max(1.0, source_duration / fit_window)
    if tempo > _MAX_TEMPO:
        minimum = source_duration / (_MAX_TEMPO * 0.99)
        raise RuntimeError(
            f"Speech section {label} needs about {minimum:.1f}s including the lip-sync safety margin, "
            f"but its video window is only {available:.1f}s. Increase the Planner duration or shorten "
            "that transcript section; words were not cut."
        )
    filters: List[str] = []
    if abs(tempo - 1.0) > 0.005:
        filters.append(f"atempo={tempo:.6f}")
    filters += ["aresample=44100", "aformat=sample_fmts=s16:channel_layouts=stereo"]
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
    log_handle.write("[strict-fit] " + " ".join(command) + "\n")
    cp = subprocess.run(command, capture_output=True, text=True)
    if cp.stdout:
        log_handle.write(cp.stdout + "\n")
    if cp.stderr:
        log_handle.write(cp.stderr + "\n")
    if cp.returncode != 0 or not _file_ok(destination):
        raise RuntimeError(f"Could not fit uploaded transcript audio for {label}.")
    return source_duration / tempo, tempo


def _shorten_generated_line(text: str, target_words: int) -> str:
    words = _collapse(text).split()
    target_words = max(3, min(len(words), int(target_words)))
    if len(words) <= target_words:
        return _collapse(text)
    shortened = " ".join(words[:target_words]).rstrip(" ,;:-")
    if shortened and shortened[-1] not in ".!?…":
        shortened += "."
    return shortened


def _synthesise_segments(
    *,
    planner_speech: Any,
    segments: Sequence[Dict[str, Any]],
    target_duration: float,
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
    audio_dir: str,
    narration_wav: str,
    narration_txt: str,
    narration_json: str,
    ffmpeg_path: str,
    ffprobe_path: str,
    log_path: str,
    strategy: str,
    llm_meta: Optional[Dict[str, Any]],
    allow_generated_shorten: bool,
    stop_requested: Optional[Callable[[], bool]],
    log_callback: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    if not segments:
        raise RuntimeError("No usable lip-sync speech sections were created.")
    segments_dir = os.path.join(audio_dir, "narration_segments")
    os.makedirs(segments_dir, exist_ok=True)
    fitted: List[Dict[str, Any]] = []
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"\n=== {LIPSYNC_ENGINE_VERSION} {strategy} ===\n")
        for index, raw_segment in enumerate(segments, start=1):
            if stop_requested and stop_requested():
                raise RuntimeError("Cancelled by user.")
            segment = dict(raw_segment)
            if log_callback:
                log_callback(f"[lip-sync] voice section {index}/{len(segments)}: {segment.get('shot_ids') or []}")
            raw_wav = planner_speech._run_tts_segment(
                segment=segment,
                job_id=job_id,
                language=language,
                narration_mode=narration_mode,
                voice=voice,
                voice_sample_path=voice_sample_path,
                ref_text=ref_text,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                qwentts_python=qwentts_python,
                qwentts_script=qwentts_script,
                segments_dir=segments_dir,
                ffmpeg_path=ffmpeg_path,
                log_handle=log_handle,
                stop_requested=stop_requested,
            )
            raw_duration = planner_speech._probe_duration(raw_wav, ffprobe_path)
            if raw_duration <= 0.01:
                raise RuntimeError(f"Could not measure TTS duration for {segment['beat_id']}.")
            available = max(0.45, float(segment.get("available_sec") or 0.45))
            if (
                allow_generated_shorten
                and raw_duration > available * 0.99 * _MAX_TEMPO
                and len(str(segment.get("text") or '').split()) > 3
            ):
                ratio = (available * 1.16) / max(0.1, raw_duration)
                target_words = max(3, int(math.floor(len(str(segment["text"]).split()) * ratio)))
                shortened = _shorten_generated_line(str(segment["text"]), target_words)
                if shortened and shortened != segment["text"]:
                    segment["text"] = shortened
                    segment["word_count"] = len(shortened.split())
                    raw_wav = planner_speech._run_tts_segment(
                        segment=segment,
                        job_id=job_id,
                        language=language,
                        narration_mode=narration_mode,
                        voice=voice,
                        voice_sample_path=voice_sample_path,
                        ref_text=ref_text,
                        model_path=model_path,
                        tokenizer_path=tokenizer_path,
                        qwentts_python=qwentts_python,
                        qwentts_script=qwentts_script,
                        segments_dir=segments_dir,
                        ffmpeg_path=ffmpeg_path,
                        log_handle=log_handle,
                        stop_requested=stop_requested,
                        retry_suffix="_fit",
                    )
                    raw_duration = planner_speech._probe_duration(raw_wav, ffprobe_path)
            fitted_wav = os.path.join(segments_dir, f"{segment['beat_id']}_fitted.wav")
            final_duration, tempo = _strict_fit_audio(
                source=raw_wav,
                destination=fitted_wav,
                source_duration=raw_duration,
                available=available,
                ffmpeg_path=ffmpeg_path,
                log_handle=log_handle,
                label=str(segment["beat_id"]),
            )
            start = float(segment.get("start_sec") or 0.0)
            end = float(segment.get("end_sec") or (start + available))
            spare = max(0.0, end - start - final_duration)
            if segment.get("is_final"):
                scheduled_start = max(start, end - final_duration)
            else:
                scheduled_start = min(end - final_duration, start + min(0.22, spare * 0.18))
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
                    "hard_trimmed": False,
                    "scheduled_start_sec": round(scheduled_start, 3),
                    "scheduled_end_sec": round(scheduled_end, 3),
                }
            )
        planner_speech._compose_track(
            segments=fitted,
            target_duration=target_duration,
            destination=narration_wav,
            ffmpeg_path=ffmpeg_path,
            log_handle=log_handle,
        )
    spoken_text = "\n\n".join(str(segment.get("text") or '').strip() for segment in fitted).strip()
    _write_text(narration_txt, spoken_text + "\n")
    srt_path = os.path.join(audio_dir, "narration.srt")
    timeline_path = os.path.join(audio_dir, "narration_timeline.json")
    planner_speech._write_srt(srt_path, fitted)
    last_end = max((float(item["scheduled_end_sec"]) for item in fitted), default=0.0)
    result = {
        "engine": LIPSYNC_ENGINE_VERSION,
        "strategy": strategy,
        "target_duration_sec": round(target_duration, 3),
        "spoken_duration_sec": round(sum(float(item["audio_duration_sec"]) for item in fitted), 3),
        "last_spoken_end_sec": round(last_end, 3),
        "last_spoken_end_ratio": round(last_end / max(0.001, target_duration), 4),
        "segment_count": len(fitted),
        "segments": fitted,
        "llm": dict(llm_meta or {}),
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
    return result


def _synthesise_generated_narration(
    *,
    planner_speech: Any,
    windows: Sequence[Dict[str, Any]],
    plan_obj: Any,
    prompt: str,
    extra_info: str,
    language: str,
    llm_json_call: Callable[..., Tuple[Optional[Any], str]],
    llm_settings: Optional[Dict[str, Any]],
    prompts_dir: str,
    speaker_name: str = "",
    **speech_kwargs: Any,
) -> Dict[str, Any]:
    script, llm_meta = planner_speech._request_script(
        windows=windows,
        plan_obj=plan_obj,
        prompt=prompt,
        extra_info=extra_info,
        language=language,
        llm_json_call=llm_json_call,
        llm_settings=llm_settings,
        prompts_dir=prompts_dir,
        delivery_mode="storyteller_monologue",
        speaker_name=speaker_name,
    )
    return _synthesise_segments(
        planner_speech=planner_speech,
        segments=script,
        strategy="ltx_shot_bound_narration",
        llm_meta=llm_meta,
        allow_generated_shorten=True,
        language=language,
        **speech_kwargs,
    )


def _synthesise_uploaded_transcript(
    *,
    planner_speech: Any,
    transcript_path: str,
    shots_obj: Any,
    timeline: Sequence[Dict[str, Any]],
    target_duration: float,
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
    audio_dir: str,
    narration_wav: str,
    narration_txt: str,
    narration_json: str,
    ffmpeg_path: str,
    ffprobe_path: str,
    log_path: str,
    stop_requested: Optional[Callable[[], bool]],
    log_callback: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    raw_text = _read_text(transcript_path)
    if not _collapse(raw_text):
        raise RuntimeError("The uploaded transcript is empty.")
    windows = _build_lipsync_windows(shots_obj, timeline)
    if not windows:
        raise RuntimeError("The Planner has no video shot long enough to hold lip-sync speech.")
    cues = _parse_timed_transcript(raw_text)
    if cues:
        segments = _timed_cues_to_shot_segments(cues, timeline, windows, target_duration)
        strategy = "uploaded_timed_transcript_shot_bound"
    else:
        segments = _allocate_plain_text_to_windows(raw_text, windows)
        strategy = "uploaded_plain_transcript_shot_bound"
    return _synthesise_segments(
        planner_speech=planner_speech,
        segments=segments,
        target_duration=target_duration,
        job_id=job_id,
        language=language,
        narration_mode=narration_mode,
        voice=voice,
        voice_sample_path=voice_sample_path,
        ref_text=ref_text,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        qwentts_python=qwentts_python,
        qwentts_script=qwentts_script,
        audio_dir=audio_dir,
        narration_wav=narration_wav,
        narration_txt=narration_txt,
        narration_json=narration_json,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        log_path=log_path,
        strategy=strategy,
        llm_meta=None,
        allow_generated_shorten=False,
        stop_requested=stop_requested,
        log_callback=log_callback,
    )


def ensure_clone_reference_text(
    *,
    root_dir: str,
    voice_sample_path: str,
    audio_dir: str,
    log_path: str,
) -> Dict[str, str]:
    sample = str(voice_sample_path or "").strip()
    if not sample or not os.path.isfile(sample):
        raise RuntimeError("Narration is set to a cloned voice but its voice sample file is missing.")
    ext = os.path.splitext(sample)[1] or ".wav"
    stable_sample = os.path.join(audio_dir, f"voice_sample{ext}")
    try:
        if os.path.abspath(sample) != os.path.abspath(stable_sample):
            shutil.copy2(sample, stable_sample)
        sample = stable_sample
    except Exception:
        pass

    transcript_path = os.path.join(audio_dir, "voice_sample_transcript.txt")
    fp_path = os.path.join(audio_dir, "voice_sample_transcript_meta.json")
    sample_fp = _file_fingerprint(sample)
    meta = _read_json(fp_path) if os.path.isfile(fp_path) else {}
    if _file_ok(transcript_path, 2) and isinstance(meta, dict) and meta.get("sample_fingerprint") == sample_fp:
        text = _collapse(_read_text(transcript_path))
        if text:
            return {"voice_sample_path": sample, "ref_text": text, "transcript_path": transcript_path}

    try:
        from helpers import whisper as whisper_helper  # type: ignore
    except Exception:
        import whisper as whisper_helper  # type: ignore
    env_py = whisper_helper._whisper_env_python()
    if not env_py or not os.path.isfile(str(env_py)):
        raise RuntimeError("Whisper environment not found. Install Whisper via Optional Installs.")
    runner = whisper_helper._ensure_whisper_runner_file()
    model_dir = ""
    candidates = [
        os.path.join(root_dir, "models", "whisper"),
        os.path.join(root_dir, "models", "faster_whisper", "medium"),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            model_dir = candidate
            break
    if not model_dir:
        raise RuntimeError("Whisper model folder was not found for voice-clone transcription.")
    try:
        device = whisper_helper._guess_device()
    except Exception:
        device = "cpu"
    try:
        ffprobe_path = str(whisper_helper._find_binary("ffprobe") or "")
    except Exception:
        ffprobe_path = ""
    payload = {
        "root": root_dir,
        "media_path": sample,
        "model_dir": model_dir,
        "device": device,
        "compute_type": "float16" if device == "cuda" else "int8",
        "language": "auto",
        "task": "transcribe",
        "ffprobe_path": ffprobe_path,
    }
    temp_dir = os.path.join(root_dir, "output", "_temp")
    os.makedirs(temp_dir, exist_ok=True)
    payload_file = os.path.join(temp_dir, f"_whisper_payload_lipsync_{int(time.time() * 1000)}.json")
    _write_json(payload_file, payload)
    command = [str(env_py), str(runner), payload_file]
    cp = subprocess.run(command, cwd=root_dir, capture_output=True, text=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write("[lip-sync clone] whisper cmd: " + " ".join(command) + "\n")
        if cp.stdout:
            log_handle.write(cp.stdout + "\n")
        if cp.stderr:
            log_handle.write(cp.stderr + "\n")
    if cp.returncode != 0:
        raise RuntimeError("Whisper failed while transcribing the cloned-voice sample.")
    result_obj: Dict[str, Any] = {}
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict) and parsed.get("type") == "result":
                result_obj = parsed.get("data") or {}
        except Exception:
            continue
    source_text_path = str(result_obj.get("text_path") or "").strip()
    if not source_text_path or not os.path.isfile(source_text_path):
        raise RuntimeError("Whisper returned no transcript for the cloned-voice sample.")
    text = _collapse(_read_text(source_text_path))
    if not text:
        raise RuntimeError("Whisper produced an empty cloned-voice transcript.")
    _write_text(transcript_path, text + "\n")
    _write_json(fp_path, {"sample_fingerprint": sample_fp, "transcript_path": transcript_path})
    return {"voice_sample_path": sample, "ref_text": text, "transcript_path": transcript_path}


def _cut_shot_audio(
    *,
    source: str,
    destination: str,
    start_sec: float,
    duration_sec: float,
    ffmpeg_path: str,
) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    duration = max(0.1, float(duration_sec))
    command = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(start_sec)):.4f}",
        "-i",
        source,
        "-t",
        f"{duration:.4f}",
        "-af",
        f"apad=pad_dur={duration:.4f},atrim=0:{duration:.4f},aresample=44100,aformat=sample_fmts=s16:channel_layouts=stereo",
        "-c:a",
        "pcm_s16le",
        destination,
    ]
    cp = subprocess.run(command, capture_output=True, text=True)
    if cp.returncode != 0 or not _file_ok(destination):
        raise RuntimeError(f"Could not create LTX lip-sync audio chunk: {destination}")


def _segments_for_shot(segments: Sequence[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for segment in segments:
        seg_start = _safe_float(segment.get("scheduled_start_sec"), _safe_float(segment.get("start_sec")))
        seg_end = _safe_float(segment.get("scheduled_end_sec"), _safe_float(segment.get("end_sec")))
        if min(end, seg_end) - max(start, seg_start) > 0.04:
            out.append(segment)
    return out


def prepare_lipsync_assets(
    *,
    root_dir: str,
    job_id: str,
    prompt: str,
    extra_info: str,
    language: str,
    narration_mode: str,
    voice: str,
    voice_sample_path: str,
    plan_obj: Any,
    shots_obj: Any,
    target_duration_sec: float,
    audio_dir: str,
    prompts_dir: str,
    narration_wav: str,
    narration_txt: str,
    narration_json: str,
    transcript_path: str,
    source_mode: str,
    uploaded_transcript_path: str,
    ffmpeg_path: str,
    ffprobe_path: str,
    qwentts_python: str,
    qwentts_script: str,
    tts_model_path: str,
    tts_tokenizer_path: str,
    llm_json_call: Callable[..., Tuple[Optional[Any], str]],
    llm_settings: Optional[Dict[str, Any]] = None,
    resume_existing: bool = False,
    log_path: str = "",
    stop_requested: Optional[Callable[[], bool]] = None,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)
    log_path = log_path or os.path.join(audio_dir, "tts_log.txt")
    shots = _normalise_shots(shots_obj)
    timeline = build_shot_timeline(shots, target_duration_sec)
    if not timeline:
        raise RuntimeError("LTX lip-sync cannot start because the Planner shot list is empty.")
    total_duration = max(_safe_float(item.get("end_sec")) for item in timeline)
    source_mode = "uploaded" if str(source_mode or "").strip().lower().startswith("upload") else "generated"
    speaker_name = _infer_speaker_name(plan_obj, prompt)
    uploaded = str(uploaded_transcript_path or "").strip()
    if source_mode == "uploaded" and (not uploaded or not os.path.isfile(uploaded)):
        raise RuntimeError("LTX lip-sync is set to Uploaded transcript, but no transcript file was selected.")

    ref_text = ""
    stable_voice_sample = str(voice_sample_path or "")
    clone_info: Dict[str, str] = {}
    if str(narration_mode or "").strip().lower() == "clone":
        clone_info = ensure_clone_reference_text(
            root_dir=root_dir,
            voice_sample_path=stable_voice_sample,
            audio_dir=audio_dir,
            log_path=log_path,
        )
        stable_voice_sample = clone_info.get("voice_sample_path") or stable_voice_sample
        ref_text = clone_info.get("ref_text") or ""

    source_fp: Any = _file_fingerprint(uploaded) if source_mode == "uploaded" else {
        "prompt": str(prompt or ""),
        "extra_info": str(extra_info or ""),
        "plan": plan_obj if isinstance(plan_obj, dict) else {},
    }
    fingerprint = _sha1_text(
        json.dumps(
            {
                "engine": LIPSYNC_ENGINE_VERSION,
                "source_mode": source_mode,
                "source": source_fp,
                "speaker_name": speaker_name,
                "staging_version": "storyteller_stage_v1",
                "shots": shots,
                "timeline": timeline,
                "language": language,
                "narration_mode": narration_mode,
                "voice": voice,
                "voice_sample": _file_fingerprint(stable_voice_sample) if stable_voice_sample else {},
                "tts_model": _file_fingerprint(tts_model_path),
                "tts_tokenizer": _file_fingerprint(tts_tokenizer_path),
                "llm_settings": _stable_llm_settings(llm_settings) if source_mode == "generated" else {},
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    plan_path = os.path.join(audio_dir, "lipsync_plan.json")
    prior = _read_json(plan_path) if os.path.isfile(plan_path) else {}
    if _prepared_plan_assets_ok(prior, narration_wav):
        if prior.get("fingerprint") == fingerprint:
            if log_callback:
                log_callback("[lip-sync] reusing prepared narration and per-shot audio")
            return prior
        if bool(resume_existing):
            # Compatibility path for jobs created before resume flags were removed
            # from the fingerprint. Preserve the old plan fingerprint so already
            # rendered LTX clips keep their matching intent fingerprints.
            if log_callback:
                log_callback("[lip-sync] resume: reusing existing narration/audio; transient Planner flags were ignored")
            return prior

    try:
        from helpers import planner_speech  # type: ignore
    except Exception:
        import planner_speech  # type: ignore

    if log_callback:
        label = "uploaded transcript" if source_mode == "uploaded" else "Planner narration"
        log_callback(f"[lip-sync] preparing {label} before LTX clips")

    if source_mode == "uploaded":
        ext = os.path.splitext(uploaded)[1] or ".txt"
        stable_transcript = os.path.join(audio_dir, f"lipsync_transcript_original{ext}")
        try:
            shutil.copy2(uploaded, stable_transcript)
        except Exception:
            stable_transcript = uploaded
        speech_result = _synthesise_uploaded_transcript(
            planner_speech=planner_speech,
            transcript_path=stable_transcript,
            shots_obj=shots,
            timeline=timeline,
            target_duration=total_duration,
            job_id=job_id,
            language=language,
            narration_mode=narration_mode,
            voice=voice,
            voice_sample_path=stable_voice_sample,
            ref_text=ref_text,
            model_path=tts_model_path,
            tokenizer_path=tts_tokenizer_path,
            qwentts_python=qwentts_python,
            qwentts_script=qwentts_script,
            audio_dir=audio_dir,
            narration_wav=narration_wav,
            narration_txt=narration_txt,
            narration_json=narration_json,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            log_path=log_path,
            stop_requested=stop_requested,
            log_callback=log_callback,
        )
    else:
        windows = _build_lipsync_windows(shots, timeline)
        if not windows:
            raise RuntimeError("The Planner has no video shot long enough to hold lip-sync speech.")
        speech_result = _synthesise_generated_narration(
            planner_speech=planner_speech,
            windows=windows,
            plan_obj=plan_obj,
            prompt=prompt,
            extra_info=extra_info,
            language=language,
            llm_json_call=llm_json_call,
            llm_settings=llm_settings,
            prompts_dir=prompts_dir,
            speaker_name=speaker_name,
            target_duration=total_duration,
            job_id=job_id,
            narration_mode=narration_mode,
            voice=voice,
            voice_sample_path=stable_voice_sample,
            ref_text=ref_text,
            model_path=tts_model_path,
            tokenizer_path=tts_tokenizer_path,
            qwentts_python=qwentts_python,
            qwentts_script=qwentts_script,
            audio_dir=audio_dir,
            narration_wav=narration_wav,
            narration_txt=narration_txt,
            narration_json=narration_json,
            ffmpeg_path=ffmpeg_path,
            ffprobe_path=ffprobe_path,
            log_path=log_path,
            stop_requested=stop_requested,
            log_callback=log_callback,
        )

    try:
        current_transcript = _read_text(transcript_path) if transcript_path and os.path.isfile(transcript_path) else ""
        placeholder = not _collapse(current_transcript) or "will be generated" in current_transcript.lower() or "generated later" in current_transcript.lower()
        if transcript_path and (source_mode == "uploaded" or placeholder):
            _write_text(transcript_path, _read_text(narration_txt))
    except Exception:
        pass

    if not _file_ok(narration_wav, 512):
        raise RuntimeError("LTX lip-sync narration preparation produced no usable master WAV.")
    segments = speech_result.get("segments") if isinstance(speech_result, dict) and isinstance(speech_result.get("segments"), list) else []
    chunks_dir = os.path.join(audio_dir, "lipsync_shots")
    os.makedirs(chunks_dir, exist_ok=True)
    shot_items: List[Dict[str, Any]] = []
    shot_lookup = _shot_lookup(shots)
    for item in timeline:
        sid = str(item.get("shot_id") or "")
        start = _safe_float(item.get("start_sec"))
        end = _safe_float(item.get("end_sec"))
        duration = max(0.1, end - start)
        overlapping = _segments_for_shot(segments, start, end)
        has_speech = bool(overlapping)
        shot = shot_lookup.get(sid, {})
        delivery = _collapse(shot.get("lipsync_delivery")).lower()
        if delivery not in ("onscreen", "voiceover_broll"):
            delivery = "onscreen" if _speaker_ready_from_existing_shot(shot) else "voiceover_broll"
        shot_speaker = _collapse(shot.get("lipsync_speaker_name") or speaker_name) or "the storyteller"
        shot_speaker_visual = _collapse(shot.get("lipsync_speaker_visual_label") or _speaker_visual_label(shot_speaker))
        audio_conditioned = bool(has_speech and delivery == "onscreen")
        chunk_path = os.path.join(chunks_dir, f"{sid}.wav") if audio_conditioned else ""
        if audio_conditioned:
            _cut_shot_audio(
                source=narration_wav,
                destination=chunk_path,
                start_sec=start,
                duration_sec=duration,
                ffmpeg_path=ffmpeg_path,
            )
        prompt_suffix = ""
        audio_condition_duration = 0.0
        if audio_conditioned:
            try:
                last_spoken_end = max(
                    _safe_float(segment.get("scheduled_end_sec"), _safe_float(segment.get("end_sec")))
                    for segment in overlapping
                )
                audio_condition_duration = max(0.1, min(duration, (last_spoken_end - start) + 0.20))
            except Exception:
                audio_condition_duration = duration
            prompt_suffix = (
                f"{shot_speaker_visual} is the only speaking subject and visibly says the supplied words with natural articulation. "
                f"Keep {shot_speaker_visual}'s face in a frontal or natural three-quarter readable view with the mouth unobstructed. "
                "Follow the supplied speech timing, but when the supplied speech ends, immediately close the mouth and return to a relaxed neutral expression; "
                "do not continue speaking, mouthing words, or inventing extra dialogue. Other people, animals, and background subjects remain silent. "
                "Preserve the planned action, setting, and camera movement without turning away from the speaking face."
            )
        shot_items.append(
            {
                "shot_id": sid,
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "duration_sec": round(duration, 3),
                "has_speech": has_speech,
                "audio_conditioned": audio_conditioned,
                "speech_delivery": delivery,
                "speaker_name": shot_speaker,
                "audio_file": chunk_path,
                "audio_condition_duration_sec": round(audio_condition_duration, 3) if audio_conditioned else 0.0,
                "text": _collapse(" ".join(str(segment.get("text") or "") for segment in overlapping)),
                "prompt_suffix": prompt_suffix,
                "audio_fingerprint": _file_fingerprint(chunk_path) if chunk_path else {},
            }
        )


    result = {
        "engine": LIPSYNC_ENGINE_VERSION,
        "fingerprint": fingerprint,
        "source_mode": source_mode,
        "speaker_name": speaker_name,
        "target_duration_sec": round(total_duration, 3),
        "voice_sample_path": stable_voice_sample,
        "ref_text": ref_text,
        "voice_sample_transcript_path": clone_info.get("transcript_path", ""),
        "speech": speech_result,
        "timeline": timeline,
        "shots": shot_items,
        "paths": {
            "lipsync_plan_json": plan_path,
            "narration_wav": narration_wav,
            "narration_txt": narration_txt,
            "narration_json": narration_json,
            "narration_srt": str((speech_result.get("paths") or {}).get("narration_srt") or "") if isinstance(speech_result, dict) else "",
            "narration_timeline_json": str((speech_result.get("paths") or {}).get("narration_timeline_json") or "") if isinstance(speech_result, dict) else "",
            "lipsync_shots_dir": chunks_dir,
        },
    }
    _write_json(plan_path, result)
    return result


def load_shot_map(plan_or_path: Any) -> Dict[str, Dict[str, Any]]:
    obj = _read_json(str(plan_or_path)) if isinstance(plan_or_path, (str, os.PathLike)) else plan_or_path
    shots = obj.get("shots") if isinstance(obj, dict) and isinstance(obj.get("shots"), list) else []
    return {
        str(item.get("shot_id") or ""): dict(item)
        for item in shots
        if isinstance(item, dict) and str(item.get("shot_id") or "").strip()
    }
