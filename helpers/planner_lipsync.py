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


LIPSYNC_ENGINE_VERSION = "planner_lipsync_v2.0_audio_driven_ui_mirror"
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


def inspect_uploaded_transcript(
    transcript_path: str,
    *,
    tail_sec: float = 0.30,
    min_clip_sec: float = 3.0,
    max_clip_sec: float = 10.041667,
) -> Dict[str, Any]:
    """Inspect source text without forcing it into a user-selected project length."""
    path = str(transcript_path or "").strip()
    if not path or not os.path.isfile(path):
        return {"ok": False, "reason": "missing", "message": "The selected lip-sync transcript file could not be found.", "line_count": 0}
    raw = _read_text(path)
    cues = _parse_timed_transcript(raw)
    units = [_collapse(item.get("text")) for item in cues] if cues else _uploaded_plain_lines(raw)
    units = [unit for unit in units if unit]
    if not units:
        return {"ok": False, "reason": "empty", "message": "The selected lip-sync transcript contains no readable spoken lines.", "line_count": 0}
    tail = max(0.0, min(2.0, _safe_float(tail_sec, 0.30)))
    minimum = max(0.1, _safe_float(min_clip_sec, 3.0))
    maximum = max(minimum, _safe_float(max_clip_sec, 10.041667))
    estimates: List[float] = []
    for unit in units:
        words = max(1, len(unit.split()))
        speech = max(0.65, words / _BASE_WORDS_PER_SEC)
        estimates.append(round(min(maximum, max(minimum, speech + tail)), 3))
    return {
        "ok": True,
        "reason": "ok",
        "message": "",
        "line_count": len(units),
        "lines": units,
        "has_timing": bool(cues),
        "tail_sec": round(tail, 3),
        "min_clip_sec": round(minimum, 3),
        "max_clip_sec": round(maximum, 3),
        "estimated_clip_durations_sec": estimates,
        "estimated_total_duration_sec": round(sum(estimates), 3),
        "word_count": sum(len(unit.split()) for unit in units),
    }


def preflight_uploaded_transcript(transcript_path: str, target_duration_sec: float = 0.0) -> Dict[str, Any]:
    """Compatibility wrapper: validate the file, never compare it with a slider."""
    return inspect_uploaded_transcript(transcript_path)

def _read_json(path: str) -> Any:
    try:
        return json.loads(_read_text(path))
    except Exception:
        return {}


def _transcript_spoken_text(raw_text: str) -> str:
    """Return only the spoken words from TXT/SRT/VTT input, without timestamps."""
    cues = _parse_timed_transcript(str(raw_text or ""))
    if cues:
        return _collapse(" ".join(str(item.get("text") or "") for item in cues))
    return _collapse(" ".join(_sentence_units(str(raw_text or ""))))


def _spoken_word_signature(text: str) -> str:
    """Stable word-only signature used to prove uploaded text was not rewritten."""
    words = re.findall(r"[^\W_]+(?:['’-][^\W_]+)?", str(text or "").lower(), flags=re.UNICODE)
    return _sha1_text(" ".join(words))


def _find_stable_uploaded_transcript(audio_dir: str) -> str:
    """Recover the project-local uploaded transcript during Resume."""
    try:
        for name in sorted(os.listdir(audio_dir)):
            low = str(name).lower()
            if low.startswith("lipsync_transcript_original") and low.endswith((".txt", ".srt", ".vtt")):
                candidate = os.path.join(audio_dir, name)
                if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
                    return candidate
    except Exception:
        pass
    return ""


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
    """Mark speaking shots without rewriting user-authored uploaded-transcript visuals."""
    shots = _normalise_shots(shots_obj)
    if not shots:
        return [], {"speaker_name": "", "onscreen_shot_ids": []}
    speaker = _infer_speaker_name(plan_obj, prompt)
    speaker_visual = _speaker_visual_label(speaker)
    uploaded_mode = str(source_mode or "").strip().lower().startswith("upload")
    selected = set(range(len(shots))) if uploaded_mode else set(_onscreen_storyteller_indices(len(shots), source_mode))
    onscreen_ids: List[str] = []
    for index, shot in enumerate(shots):
        sid = _collapse(shot.get("id") or f"S{index + 1:02d}")
        original = _clean_visual_text(shot.get("lipsync_original_visual") or shot.get("visual_description") or shot.get("seed") or shot.get("prompt"))
        shot["lipsync_original_visual"] = original
        shot["lipsync_speaker_name"] = speaker
        shot["lipsync_speaker_visual_label"] = speaker_visual
        if index in selected:
            onscreen_ids.append(sid)
            shot["lipsync_delivery"] = "onscreen"
            if not uploaded_mode:
                # Planner-created narration can still stage a readable storyteller.
                setting = _collapse((plan_obj or {}).get("setting") if isinstance(plan_obj, dict) else "")
                setting = re.sub(r"^outdoors\s*,?\s*", "", setting, flags=re.IGNORECASE).strip()
                location = (f" in {setting}" if setting else " in the established location")
                context = original or "the story setting and its important visual details"
                staged = (
                    f"Medium close-up of {speaker_visual}{location}, facing the camera in a natural three-quarter view with a clear, "
                    f"unobstructed face and fully visible mouth, speaking as the on-screen storyteller. "
                    f"The surrounding story context remains visible behind the speaker: {context}. One speaking character only."
                )
                shot["visual_description"] = staged
                shot["seed"] = staged
                shot["camera"] = "medium close-up"
        else:
            shot["lipsync_delivery"] = "voiceover_broll"
    return shots, {
        "speaker_name": speaker,
        "onscreen_shot_ids": onscreen_ids,
        "voiceover_shot_ids": [str(item.get("id") or "") for item in shots if item.get("lipsync_delivery") == "voiceover_broll"],
        "source_mode": "uploaded" if uploaded_mode else "generated",
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


def _uploaded_plain_lines(text: str) -> List[str]:
    """Return physical non-empty TXT lines as hard speech units.

    Uploaded plain-text transcripts are user-authored shot scripts. A newline is
    therefore an explicit boundary, not a suggestion. Never merge adjacent lines
    and never move overflow words into the next line/shot.
    """
    clean = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    clean = clean.lstrip("\ufeff")
    lines: List[str] = []
    for raw_line in clean.split("\n"):
        line = _collapse(raw_line)
        if not line:
            continue
        if line.upper() == "WEBVTT" or "-->" in line:
            continue
        # Ignore standalone subtitle cue numbers without discarding ordinary
        # lines that begin with a number.
        if re.fullmatch(r"\d+", line):
            continue
        lines.append(line)
    return lines


def _allocate_plain_text_to_windows(text: str, windows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bind each uploaded TXT line to one Planner shot, in order.

    Older builds collapsed the whole transcript into one token stream and then
    redistributed words according to shot capacity. That merged neighbouring
    lines and could start a new shot with words stolen from the previous line.
    This function deliberately treats every non-empty physical line as atomic.
    """
    lines = _uploaded_plain_lines(text)
    if not lines or not windows:
        return []
    if len(lines) > len(windows):
        raise RuntimeError(
            f"The uploaded transcript has {len(lines)} non-empty lines but the Planner created only "
            f"{len(windows)} usable video shots. Each line is locked to one shot, so add more shots/images "
            "or remove transcript lines. No lines were merged or split."
        )

    selected = [dict(item) for item in windows[: len(lines)]]
    segments: List[Dict[str, Any]] = []
    for index, (line, window) in enumerate(zip(lines, selected), start=1):
        word_count = len(line.split())
        # Do not redistribute or reject a user-authored line based on a rough
        # words-per-second estimate. The real TTS result is measured below and
        # the existing strict audio fitter decides whether that exact line can
        # fit the shot. If it cannot, the run stops with the line/shot intact.
        segments.append(
            {
                **window,
                "beat_id": f"U{index:02d}",
                "index": index,
                "text": line,
                "word_count": word_count,
                "source_line_index": index,
                "line_locked": True,
                "is_first": index == 1,
                "is_final": index == len(lines),
            }
        )
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
            f"Speech section {label} needs about {minimum:.1f}s, but one LTX clip can provide only "
            f"{available:.1f}s for speech. Shorten that single line; words were not cut or moved."
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


def _convert_audio_without_tempo(source: str, destination: str, ffmpeg_path: str, log_handle: Any) -> None:
    command = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", "-i", source,
        "-af", "aresample=44100,aformat=sample_fmts=s16:channel_layouts=stereo",
        "-c:a", "pcm_s16le", destination,
    ]
    log_handle.write("[convert] " + " ".join(command) + "\n")
    cp = subprocess.run(command, capture_output=True, text=True)
    if cp.stderr:
        log_handle.write(cp.stderr + "\n")
    if cp.returncode != 0 or not _file_ok(destination):
        raise RuntimeError(f"Could not convert narration audio: {source}")


def _pad_audio_to_duration(source: str, destination: str, duration_sec: float, ffmpeg_path: str, log_handle: Any) -> None:
    duration = max(0.1, float(duration_sec))
    command = [
        ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error", "-i", source,
        "-af", f"apad=pad_dur={duration:.6f},atrim=0:{duration:.6f},aresample=44100,aformat=sample_fmts=s16:channel_layouts=stereo",
        "-t", f"{duration:.6f}", "-c:a", "pcm_s16le", destination,
    ]
    log_handle.write("[shot-pad] " + " ".join(command) + "\n")
    cp = subprocess.run(command, capture_output=True, text=True)
    if cp.stderr:
        log_handle.write(cp.stderr + "\n")
    if cp.returncode != 0 or not _file_ok(destination):
        raise RuntimeError(f"Could not create audio-driven shot WAV: {destination}")


def _synthesise_uploaded_audio_driven(
    *, planner_speech: Any, transcript_path: str, shots_obj: Any, job_id: str,
    language: str, narration_mode: str, voice: str, voice_sample_path: str,
    ref_text: str, model_path: str, tokenizer_path: str, qwentts_python: str,
    qwentts_script: str, audio_dir: str, narration_wav: str, narration_txt: str,
    narration_json: str, ffmpeg_path: str, ffprobe_path: str, log_path: str,
    tail_sec: float, min_clip_sec: float, max_clip_sec: float,
    stop_requested: Optional[Callable[[], bool]], log_callback: Optional[Callable[[str], None]],
) -> Dict[str, Any]:
    inspection = inspect_uploaded_transcript(
        transcript_path, tail_sec=tail_sec, min_clip_sec=min_clip_sec, max_clip_sec=max_clip_sec
    )
    if not inspection.get("ok"):
        raise RuntimeError(str(inspection.get("message") or "The uploaded transcript is invalid."))
    lines = list(inspection.get("lines") or [])
    shots = _normalise_shots(shots_obj)
    if len(lines) != len(shots):
        raise RuntimeError(
            f"Uploaded transcript has {len(lines)} non-empty lines, but the Planner has {len(shots)} shots/images. "
            "Uploaded lip-sync is one line = one image = one video clip. Make the counts equal; no line will be split, merged, or moved."
        )
    tail = float(inspection["tail_sec"])
    minimum = float(inspection["min_clip_sec"])
    maximum = float(inspection["max_clip_sec"])
    segments_dir = os.path.join(audio_dir, "narration_segments")
    chunks_dir = os.path.join(audio_dir, "lipsync_shots")
    os.makedirs(segments_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    fitted: List[Dict[str, Any]] = []
    updated_shots: List[Dict[str, Any]] = []
    cursor = 0.0
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write(f"\n=== {LIPSYNC_ENGINE_VERSION} uploaded_audio_driven ===\n")
        for index, (line, shot_raw) in enumerate(zip(lines, shots), start=1):
            if stop_requested and stop_requested():
                raise RuntimeError("Cancelled by user.")
            sid = str(shot_raw.get("id") or f"S{index:02d}")
            beat_id = f"U{index:02d}"
            segment = {"beat_id": beat_id, "index": index, "text": line, "shot_ids": [sid]}
            if log_callback:
                log_callback(f"[lip-sync] TTS {sid} line {index}/{len(lines)}")
            raw_wav = planner_speech._run_tts_segment(
                segment=segment, job_id=job_id, language=language, narration_mode=narration_mode,
                voice=voice, voice_sample_path=voice_sample_path, ref_text=ref_text,
                model_path=model_path, tokenizer_path=tokenizer_path, qwentts_python=qwentts_python,
                qwentts_script=qwentts_script, segments_dir=segments_dir, ffmpeg_path=ffmpeg_path,
                log_handle=log_handle, stop_requested=stop_requested,
            )
            raw_duration = float(planner_speech._probe_duration(raw_wav, ffprobe_path) or 0.0)
            if raw_duration <= 0.01:
                raise RuntimeError(f"Could not measure TTS duration for transcript line {index} ({sid}).")
            max_speech = max(0.45, maximum - tail)
            fitted_wav = os.path.join(segments_dir, f"{beat_id}_fitted.wav")
            if raw_duration > max_speech:
                try:
                    speech_duration, tempo = _strict_fit_audio(
                        source=raw_wav, destination=fitted_wav, source_duration=raw_duration,
                        available=max_speech, ffmpeg_path=ffmpeg_path, log_handle=log_handle, label=sid,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Transcript line {index} ({sid}) produces {raw_duration:.2f}s of speech, longer than one LTX clip can hold. "
                        f"Shorten only that line; the Planner did not split or cut it. Details: {exc}"
                    ) from exc
            else:
                _convert_audio_without_tempo(raw_wav, fitted_wav, ffmpeg_path, log_handle)
                speech_duration, tempo = raw_duration, 1.0
            clip_duration = min(maximum, max(minimum, speech_duration + tail))
            shot_wav = os.path.join(chunks_dir, f"{sid}.wav")
            _pad_audio_to_duration(fitted_wav, shot_wav, clip_duration, ffmpeg_path, log_handle)
            scheduled_start = cursor
            scheduled_end = cursor + speech_duration
            clip_end = cursor + clip_duration
            fitted.append({
                **segment,
                "audio_file": fitted_wav,
                "shot_audio_file": shot_wav,
                "raw_audio_file": raw_wav,
                "raw_duration_sec": round(raw_duration, 3),
                "audio_duration_sec": round(speech_duration, 3),
                "clip_duration_sec": round(clip_duration, 3),
                "tail_sec": round(max(0.0, clip_duration - speech_duration), 3),
                "tempo": round(float(tempo), 4),
                "scheduled_start_sec": round(scheduled_start, 3),
                "scheduled_end_sec": round(scheduled_end, 3),
                "clip_start_sec": round(cursor, 3),
                "clip_end_sec": round(clip_end, 3),
            })
            shot = dict(shot_raw)
            shot["duration_sec"] = round(clip_duration, 3)
            shot["lipsync_audio_duration_sec"] = round(speech_duration, 3)
            shot["lipsync_tail_sec"] = round(max(0.0, clip_duration - speech_duration), 3)
            shot["lipsync_delivery"] = "onscreen"
            updated_shots.append(shot)
            if log_callback:
                log_callback(
                    f"[lip-sync] {sid}: speech {speech_duration:.3f}s -> video {clip_duration:.3f}s "
                    f"({int(round(clip_duration * 24.0))} frames at 24 fps before model alignment)"
                )
            cursor = clip_end
        planner_speech._compose_track(
            segments=fitted, target_duration=cursor, destination=narration_wav,
            ffmpeg_path=ffmpeg_path, log_handle=log_handle,
        )
    _write_text(narration_txt, "\n".join(lines).strip() + "\n")
    srt_path = os.path.join(audio_dir, "narration.srt")
    timeline_path = os.path.join(audio_dir, "narration_timeline.json")
    planner_speech._write_srt(srt_path, fitted)
    timeline = [
        {"index": i, "shot_id": str(shot.get("id") or f"S{i:02d}"), "start_sec": round(float(seg["clip_start_sec"]), 3),
         "duration_sec": round(float(seg["clip_duration_sec"]), 3), "end_sec": round(float(seg["clip_end_sec"]), 3)}
        for i, (shot, seg) in enumerate(zip(updated_shots, fitted), start=1)
    ]
    result = {
        "engine": LIPSYNC_ENGINE_VERSION,
        "strategy": "uploaded_audio_driven_one_line_per_shot",
        "target_duration_sec": round(cursor, 3),
        "spoken_duration_sec": round(sum(float(item["audio_duration_sec"]) for item in fitted), 3),
        "segment_count": len(fitted),
        "segments": fitted,
        "timeline": timeline,
        "updated_shots": updated_shots,
        "inspection": inspection,
        "paths": {
            "narration_wav": narration_wav, "narration_txt": narration_txt,
            "narration_json": narration_json, "narration_timeline_json": timeline_path,
            "narration_srt": srt_path, "segments_dir": segments_dir,
            "lipsync_shots_dir": chunks_dir, "tts_log": log_path,
        },
    }
    _write_json(timeline_path, result)
    _write_json(narration_json, result)
    return result


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
    *, root_dir: str, job_id: str, prompt: str, extra_info: str, language: str,
    narration_mode: str, voice: str, voice_sample_path: str, plan_obj: Any,
    shots_obj: Any, target_duration_sec: float, audio_dir: str, prompts_dir: str,
    narration_wav: str, narration_txt: str, narration_json: str, transcript_path: str,
    source_mode: str, uploaded_transcript_path: str, ffmpeg_path: str, ffprobe_path: str,
    qwentts_python: str, qwentts_script: str, tts_model_path: str, tts_tokenizer_path: str,
    llm_json_call: Callable[..., Tuple[Optional[Any], str]], llm_settings: Optional[Dict[str, Any]] = None,
    resume_existing: bool = False, log_path: str = "", stop_requested: Optional[Callable[[], bool]] = None,
    log_callback: Optional[Callable[[str], None]] = None, uploaded_tail_sec: float = 0.30,
    uploaded_min_clip_sec: float = 3.0, uploaded_max_clip_sec: float = 10.041667,
) -> Dict[str, Any]:
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(prompts_dir, exist_ok=True)
    log_path = log_path or os.path.join(audio_dir, "tts_log.txt")
    shots = _normalise_shots(shots_obj)
    if not shots:
        raise RuntimeError("LTX lip-sync cannot start because the Planner shot list is empty.")
    source_mode = "uploaded" if str(source_mode or "").strip().lower().startswith("upload") else "generated"
    # Uploaded speech is not scaled to the old duration slider. Generated narration keeps its legacy timeline.
    timeline = build_shot_timeline(shots, 0.0 if source_mode == "uploaded" else target_duration_sec)
    total_duration = max(_safe_float(item.get("end_sec")) for item in timeline)
    speaker_name = _infer_speaker_name(plan_obj, prompt)
    uploaded = str(uploaded_transcript_path or "").strip()
    if source_mode == "uploaded" and (not uploaded or not os.path.isfile(uploaded)):
        uploaded = _find_stable_uploaded_transcript(audio_dir)
        if not uploaded:
            raise RuntimeError("LTX lip-sync is set to Uploaded transcript, but no transcript file was selected or recovered.")
        if log_callback:
            log_callback(f"[lip-sync] recovered project-local uploaded transcript: {uploaded}")

    ref_text = ""
    stable_voice_sample = str(voice_sample_path or "")
    clone_info: Dict[str, str] = {}
    if str(narration_mode or "").strip().lower() == "clone":
        clone_info = ensure_clone_reference_text(root_dir=root_dir, voice_sample_path=stable_voice_sample, audio_dir=audio_dir, log_path=log_path)
        stable_voice_sample = clone_info.get("voice_sample_path") or stable_voice_sample
        ref_text = clone_info.get("ref_text") or ""

    uploaded_spoken_text = ""
    uploaded_spoken_signature = ""
    if source_mode == "uploaded":
        inspection = inspect_uploaded_transcript(
            uploaded, tail_sec=uploaded_tail_sec, min_clip_sec=uploaded_min_clip_sec,
            max_clip_sec=uploaded_max_clip_sec,
        )
        if not inspection.get("ok"):
            raise RuntimeError(str(inspection.get("message") or "The uploaded transcript is invalid."))
        if int(inspection.get("line_count") or 0) != len(shots):
            raise RuntimeError(
                f"Uploaded transcript has {int(inspection.get('line_count') or 0)} lines, but the Planner has {len(shots)} shots/images. "
                "Uploaded lip-sync requires an exact one-to-one count."
            )
        uploaded_spoken_text = _transcript_spoken_text(_read_text(uploaded))
        uploaded_spoken_signature = _spoken_word_signature(uploaded_spoken_text)
        source_fp: Any = {**_file_fingerprint(uploaded), "spoken_word_signature": uploaded_spoken_signature}
    else:
        inspection = {}
        source_fp = {"prompt": str(prompt or ""), "extra_info": str(extra_info or ""), "plan": plan_obj if isinstance(plan_obj, dict) else {}}

    fingerprint = _sha1_text(json.dumps({
        "engine": LIPSYNC_ENGINE_VERSION, "source_mode": source_mode, "source": source_fp,
        "speaker_name": speaker_name, "shots": shots,
        "language": language, "narration_mode": narration_mode, "voice": voice,
        "voice_sample": _file_fingerprint(stable_voice_sample) if stable_voice_sample else {},
        "tts_model": _file_fingerprint(tts_model_path), "tts_tokenizer": _file_fingerprint(tts_tokenizer_path),
        "uploaded_tail_sec": round(float(uploaded_tail_sec), 3) if source_mode == "uploaded" else None,
        "uploaded_min_clip_sec": round(float(uploaded_min_clip_sec), 3) if source_mode == "uploaded" else None,
        "uploaded_max_clip_sec": round(float(uploaded_max_clip_sec), 3) if source_mode == "uploaded" else None,
        "llm_settings": _stable_llm_settings(llm_settings) if source_mode == "generated" else {},
    }, sort_keys=True, ensure_ascii=False))
    plan_path = os.path.join(audio_dir, "lipsync_plan.json")
    prior = _read_json(plan_path) if os.path.isfile(plan_path) else {}
    if _prepared_plan_assets_ok(prior, narration_wav) and prior.get("fingerprint") == fingerprint:
        if log_callback:
            log_callback("[lip-sync] reusing prepared narration and per-shot audio (exact source match)")
        return prior

    try:
        from helpers import planner_speech  # type: ignore
    except Exception:
        import planner_speech  # type: ignore
    if log_callback:
        log_callback(f"[lip-sync] preparing {'uploaded transcript' if source_mode == 'uploaded' else 'Planner narration'} before LTX clips")

    if source_mode == "uploaded":
        ext = os.path.splitext(uploaded)[1] or ".txt"
        stable_transcript = os.path.join(audio_dir, f"lipsync_transcript_original{ext}")
        try:
            shutil.copy2(uploaded, stable_transcript)
        except Exception:
            stable_transcript = uploaded
        speech_result = _synthesise_uploaded_audio_driven(
            planner_speech=planner_speech, transcript_path=stable_transcript, shots_obj=shots,
            job_id=job_id, language=language, narration_mode=narration_mode, voice=voice,
            voice_sample_path=stable_voice_sample, ref_text=ref_text, model_path=tts_model_path,
            tokenizer_path=tts_tokenizer_path, qwentts_python=qwentts_python, qwentts_script=qwentts_script,
            audio_dir=audio_dir, narration_wav=narration_wav, narration_txt=narration_txt,
            narration_json=narration_json, ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path,
            log_path=log_path, tail_sec=uploaded_tail_sec, min_clip_sec=uploaded_min_clip_sec,
            max_clip_sec=uploaded_max_clip_sec, stop_requested=stop_requested, log_callback=log_callback,
        )
        rendered_spoken_text = _collapse(_read_text(narration_txt)) if os.path.isfile(narration_txt) else ""
        if _spoken_word_signature(rendered_spoken_text) != uploaded_spoken_signature:
            raise RuntimeError("Planner narration text no longer matches the uploaded transcript. Generation stopped before LTX clips.")
        shots = list(speech_result.get("updated_shots") or shots)
        timeline = list(speech_result.get("timeline") or build_shot_timeline(shots, 0.0))
        total_duration = float(speech_result.get("target_duration_sec") or sum(_safe_float(x.get("duration_sec")) for x in shots))
        segments = list(speech_result.get("segments") or [])
        shot_items: List[Dict[str, Any]] = []
        for shot, segment in zip(shots, segments):
            sid = str(shot.get("id") or "")
            shot_wav = str(segment.get("shot_audio_file") or "")
            if not _file_ok(shot_wav):
                raise RuntimeError(f"Missing audio-driven per-shot WAV for {sid}: {shot_wav}")
            duration = float(segment.get("clip_duration_sec") or shot.get("duration_sec") or 0.0)
            shot_items.append({
                "shot_id": sid, "start_sec": float(segment.get("clip_start_sec") or 0.0),
                "end_sec": float(segment.get("clip_end_sec") or duration), "duration_sec": round(duration, 3),
                "has_speech": True, "audio_conditioned": True, "speech_delivery": "onscreen",
                "speaker_name": _collapse(shot.get("lipsync_speaker_name") or speaker_name) or "the storyteller",
                "speaker_visual_label": _collapse(shot.get("lipsync_speaker_visual_label") or _speaker_visual_label(speaker_name)),
                "audio_file": shot_wav, "audio_condition_duration_sec": round(duration, 3),
                "speech_duration_sec": round(float(segment.get("audio_duration_sec") or 0.0), 3),
                "text": str(segment.get("text") or ""), "prompt_suffix": "",
                "audio_fingerprint": _file_fingerprint(shot_wav),
            })
    else:
        windows = _build_lipsync_windows(shots, timeline)
        if not windows:
            raise RuntimeError("The Planner has no video shot long enough to hold lip-sync speech.")
        speech_result = _synthesise_generated_narration(
            planner_speech=planner_speech, windows=windows, plan_obj=plan_obj, prompt=prompt,
            extra_info=extra_info, language=language, llm_json_call=llm_json_call,
            llm_settings=llm_settings, prompts_dir=prompts_dir, speaker_name=speaker_name,
            target_duration=total_duration, job_id=job_id, narration_mode=narration_mode,
            voice=voice, voice_sample_path=stable_voice_sample, ref_text=ref_text,
            model_path=tts_model_path, tokenizer_path=tts_tokenizer_path,
            qwentts_python=qwentts_python, qwentts_script=qwentts_script, audio_dir=audio_dir,
            narration_wav=narration_wav, narration_txt=narration_txt, narration_json=narration_json,
            ffmpeg_path=ffmpeg_path, ffprobe_path=ffprobe_path, log_path=log_path,
            stop_requested=stop_requested, log_callback=log_callback,
        )
        segments = speech_result.get("segments") if isinstance(speech_result, dict) and isinstance(speech_result.get("segments"), list) else []
        chunks_dir = os.path.join(audio_dir, "lipsync_shots")
        os.makedirs(chunks_dir, exist_ok=True)
        shot_items = []
        shot_lookup = _shot_lookup(shots)
        for item in timeline:
            sid = str(item.get("shot_id") or "")
            start = _safe_float(item.get("start_sec")); end = _safe_float(item.get("end_sec")); duration = max(0.1, end-start)
            overlapping = _segments_for_shot(segments, start, end)
            shot = shot_lookup.get(sid, {})
            delivery = _collapse(shot.get("lipsync_delivery")).lower() or ("onscreen" if _speaker_ready_from_existing_shot(shot) else "voiceover_broll")
            audio_conditioned = bool(overlapping and delivery == "onscreen")
            chunk_path = os.path.join(chunks_dir, f"{sid}.wav") if audio_conditioned else ""
            if audio_conditioned:
                _cut_shot_audio(source=narration_wav, destination=chunk_path, start_sec=start, duration_sec=duration, ffmpeg_path=ffmpeg_path)
            shot_items.append({
                "shot_id": sid, "start_sec": round(start,3), "end_sec": round(end,3), "duration_sec": round(duration,3),
                "has_speech": bool(overlapping), "audio_conditioned": audio_conditioned, "speech_delivery": delivery,
                "speaker_name": _collapse(shot.get("lipsync_speaker_name") or speaker_name) or "the storyteller",
                "speaker_visual_label": _collapse(shot.get("lipsync_speaker_visual_label") or _speaker_visual_label(speaker_name)),
                "audio_file": chunk_path, "audio_condition_duration_sec": round(duration,3) if audio_conditioned else 0.0,
                "text": _collapse(" ".join(str(segment.get("text") or "") for segment in overlapping)),
                "prompt_suffix": "", "audio_fingerprint": _file_fingerprint(chunk_path) if chunk_path else {},
            })

    try:
        if transcript_path and (source_mode == "uploaded" or not _collapse(_read_text(transcript_path) if os.path.isfile(transcript_path) else "")):
            _write_text(transcript_path, _read_text(narration_txt))
    except Exception:
        pass
    if not _file_ok(narration_wav, 512):
        raise RuntimeError("LTX lip-sync narration preparation produced no usable master WAV.")
    chunks_dir = os.path.join(audio_dir, "lipsync_shots")
    result = {
        "engine": LIPSYNC_ENGINE_VERSION, "fingerprint": fingerprint, "source_mode": source_mode,
        "source_transcript_path": uploaded if source_mode == "uploaded" else "", "source_fingerprint": source_fp,
        "source_spoken_word_signature": uploaded_spoken_signature if source_mode == "uploaded" else "",
        "speaker_name": speaker_name, "target_duration_sec": round(total_duration, 3),
        "audio_driven_duration": bool(source_mode == "uploaded"),
        "voice_sample_path": stable_voice_sample, "ref_text": ref_text,
        "voice_sample_transcript_path": clone_info.get("transcript_path", ""),
        "speech": speech_result, "timeline": timeline, "shots": shot_items,
        "updated_shots": shots if source_mode == "uploaded" else [],
        "paths": {
            "lipsync_plan_json": plan_path, "narration_wav": narration_wav,
            "narration_txt": narration_txt, "narration_json": narration_json,
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
