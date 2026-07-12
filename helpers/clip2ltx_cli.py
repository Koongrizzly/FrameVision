"""
Optional Music Clip Creator -> Planner bridge foundation for FrameVision.

This helper is deliberately lightweight and private/optional:
- stdlib only
- no PySide dependency
- no Planner import
- no LTX/LLM/generation call

It only exports a clean scene-plan JSON that later Planner/LTX work can consume.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


_VERSION = 1
_SOURCE = "framevision_music_clip_creator"


def is_available() -> bool:
    """Return True when this optional bridge module is importable and usable."""
    return True


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[1]
    except Exception:
        return Path.cwd().resolve()




def _default_ltx_videoclip_output_dir(root: Optional[Path] = None) -> Path:
    """Default base folder for LTX Music Clip Creator job folders."""
    base_root = Path(root).resolve() if root is not None else _project_root()
    return (base_root / "output" / "videoclips" / "ltx").resolve()

def _ltx_job_output_dir_from_payload(payload: Dict[str, Any], audio_path: str = "") -> Path:
    """Return a fresh per-job LTX folder below the selected/default output base.

    The UI passes output_dir as the base folder, not as the exact job folder.
    Keep every scene/prompt/shot/director JSON plus chunks under one named job folder.
    """
    root_value = _safe_str(payload.get("root_dir")) if isinstance(payload, dict) else ""
    root = Path(root_value).resolve() if root_value else _project_root()
    base_raw = _safe_str(payload.get("output_dir")) if isinstance(payload, dict) else ""
    base = Path(base_raw).expanduser().resolve() if base_raw else _default_ltx_videoclip_output_dir(root)
    safe_song = _safe_stem((payload or {}).get("safe_song_stem") or audio_path or "musicclip_ltx")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = (base / f"{safe_song}_{timestamp}").resolve()
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    for idx in range(2, 1000):
        candidate = (base / f"{safe_song}_{timestamp}_{idx:02d}").resolve()
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError("Could not create a unique LTX music-clip job output folder.")

def _safe_str(value: Any, default: str = "") -> str:
    try:
        text = str(value if value is not None else "")
    except Exception:
        return default
    return text.strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    except Exception:
        return bool(default)


def _safe_stem(value: Any) -> str:
    text = _safe_str(value, "musicclip") or "musicclip"
    text = os.path.splitext(os.path.basename(text))[0] or text
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text[:80] or "musicclip"


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _normalize_creative_brief(value: Any) -> Dict[str, str]:
    src = value if isinstance(value, dict) else {}
    return {
        "main_idea": _safe_str(src.get("main_idea")),
        "style_theme": _safe_str(src.get("style_theme")),
        "characters_subjects": _safe_str(src.get("characters_subjects")),
        "locations_world": _safe_str(src.get("locations_world")),
        "camera_choreography": _safe_str(src.get("camera_choreography")),
    }


def _character_reference_slot_keys() -> List[str]:
    return [f"char_{idx:02d}" for idx in range(1, 6)]


def _character_reference_existing_path(path: Any) -> str:
    p = _safe_str(path).strip().strip('"')
    try:
        return p if p and os.path.isfile(p) else ""
    except Exception:
        return ""


def _character_reference_source_has_real_content(ref: Any) -> bool:
    """Return True only when a character-reference payload carries usable intent.

    Earlier builds treated any empty ``character_reference_sheets`` dict as a
    real source.  That let an inactive UI/review payload shadow the director
    plan, so later start-image generation could lose the loaded sheet paths and
    silently drop back to text-only.
    """
    if not isinstance(ref, dict):
        return False
    if _safe_bool(ref.get("enabled"), False):
        return True
    if _safe_str(ref.get("reference_sheet_path")) or _safe_str(ref.get("global_reference_sheet_path")):
        return True
    sheets = ref.get("character_reference_sheets")
    if isinstance(sheets, dict):
        for value in sheets.values():
            if _safe_str(value):
                return True
    for key in _character_reference_slot_keys():
        if _safe_str(ref.get(key)) or _safe_str(ref.get(f"reference_sheet_path_{key}")):
            return True
    selected = ref.get("selected_reference_sheet_paths")
    if isinstance(selected, list) and any(_safe_str(x) for x in selected):
        return True
    return False


def _character_reference_ref_label(src: Any, index: int) -> str:
    if isinstance(src, dict):
        if isinstance(src.get("character_reference"), dict):
            if src.get("ltx_shots") is not None or src.get("shots") is not None or src.get("creative_brief") is not None:
                return f"director_plan[{index}]"
            if src.get("ltx_director_plan_path") is not None or src.get("shot_id") is not None:
                return f"payload[{index}]"
    return f"source[{index}]"


def _character_reference_from_sources(*sources: Any) -> Dict[str, Any]:
    """Normalize and merge Character Bible V2 reference-sheet metadata.

    This deliberately merges all usable sources instead of letting the first
    review/payload source win.  Review payloads often carry only
    selected_reference_sheet_paths=[char_01]; those selected paths are useful,
    but they must not erase the full director-plan character_reference_sheets.
    """
    warnings: List[str] = []
    refs: List[tuple[str, Dict[str, Any]]] = []
    for idx, src in enumerate(sources):
        if not isinstance(src, dict):
            continue
        ref = src.get("character_reference")
        if _character_reference_source_has_real_content(ref):
            refs.append((_character_reference_ref_label(src, idx), dict(ref)))

    chosen: Dict[str, Any] = dict(refs[0][1]) if refs else {}
    source_labels = [label for label, _ref in refs]
    sheets_raw: Dict[str, Any] = {}
    selected_direct_paths: List[str] = []
    selected_direct_source = ""
    available_direct_paths: List[str] = []
    global_candidates: List[str] = []
    source_had_full_character_reference_sheets = False
    richest_reference_source = source_labels[0] if source_labels else "none"
    richest_valid_count = 0

    def _append_path(target: List[str], value: Any) -> None:
        path = _safe_str(value)
        if path and path not in target:
            target.append(path)

    # First pass: preserve explicit per-character slots from every source.
    # Valid/richer sources win over empty or stale single-selected sources.
    for label, ref in refs:
        source_raw_slots: Dict[str, Any] = {}
        if isinstance(ref.get("character_reference_sheets"), dict):
            source_raw_slots.update(ref.get("character_reference_sheets") or {})
        for key in _character_reference_slot_keys():
            if key not in source_raw_slots and _safe_str(ref.get(key)):
                source_raw_slots[key] = ref.get(key)
            flat = f"reference_sheet_path_{key}"
            if key not in source_raw_slots and _safe_str(ref.get(flat)):
                source_raw_slots[key] = ref.get(flat)

        source_valid_count = 0
        for key in _character_reference_slot_keys():
            raw = _safe_str(source_raw_slots.get(key))
            if not raw:
                continue
            valid = _character_reference_existing_path(raw)
            if valid:
                source_valid_count += 1
            existing_raw = _safe_str(sheets_raw.get(key))
            existing_valid = _character_reference_existing_path(existing_raw)
            if not existing_raw or (valid and not existing_valid):
                sheets_raw[key] = raw

        if source_valid_count >= 2:
            source_had_full_character_reference_sheets = True
        if source_valid_count > richest_valid_count:
            richest_valid_count = source_valid_count
            richest_reference_source = label

        raw_global = _safe_str(ref.get("global_reference_sheet_path"))
        if raw_global:
            _append_path(global_candidates, raw_global)
        # Legacy single-sheet payloads may store the one-person path under
        # reference_sheet_path.  Keep it as an available direct path unless the
        # source explicitly marks it as global/group.
        legacy_ref = _safe_str(ref.get("reference_sheet_path"))
        if legacy_ref and not raw_global:
            _append_path(available_direct_paths, legacy_ref)

        source_selected = [_safe_str(x) for x in _as_list(ref.get("selected_reference_sheet_paths")) if _safe_str(x)]
        if not source_selected:
            source_selected = [_safe_str(x) for x in _as_list(ref.get("actual_image_model_reference_paths_passed")) if _safe_str(x)]
        if source_selected and not selected_direct_source:
            selected_direct_source = label
        for path in source_selected:
            _append_path(selected_direct_paths, path)

        for path in _as_list(ref.get("available_reference_sheet_paths")):
            _append_path(available_direct_paths, path)

        for w in _as_list(ref.get("warnings")):
            if _safe_str(w):
                warnings.append(_safe_str(w))

    # If a source only had selected/available direct paths, promote those into
    # empty char slots.  Do this after explicit sheets are merged so a review
    # payload with only char_01 cannot overwrite char_02/char_03 from the plan.
    for raw_path in selected_direct_paths + available_direct_paths:
        if not _safe_str(raw_path):
            continue
        if any(_safe_str(sheets_raw.get(key)) == _safe_str(raw_path) for key in _character_reference_slot_keys()):
            continue
        for key in _character_reference_slot_keys():
            if not _safe_str(sheets_raw.get(key)):
                sheets_raw[key] = raw_path
                break

    global_valid = ""
    for raw_global in global_candidates:
        valid = _character_reference_existing_path(raw_global)
        if valid:
            global_valid = valid
            break
        if _safe_str(raw_global):
            warnings.append("Global character reference sheet path is missing; global fallback was not used.")

    sheets: Dict[str, str] = {}
    for key in _character_reference_slot_keys():
        raw = _safe_str(sheets_raw.get(key))
        valid = _character_reference_existing_path(raw)
        sheets[key] = valid if valid else ""
        if raw and not valid:
            warnings.append(f"Character reference sheet {key} is missing or unreadable; that slot was ignored.")

    available: List[str] = []
    if global_valid:
        available.append(global_valid)
    for key in _character_reference_slot_keys():
        if sheets.get(key) and sheets[key] not in available:
            available.append(sheets[key])
    for raw_path in available_direct_paths + selected_direct_paths:
        valid = _character_reference_existing_path(raw_path)
        if valid and valid not in available:
            available.append(valid)
    available = _dedupe_texts(available, max_items=6, max_len=4000) if "_dedupe_texts" in globals() else list(dict.fromkeys(available))[:6]

    selected_valid = []
    for raw_path in selected_direct_paths:
        valid = _character_reference_existing_path(raw_path)
        if valid and valid not in selected_valid:
            selected_valid.append(valid)
    selected_valid = selected_valid[:5]

    enabled = bool(_safe_bool(chosen.get("enabled"), False) and available) or bool(available)
    first_character_reference = next((sheets.get(k) for k in _character_reference_slot_keys() if sheets.get(k)), "")
    legacy_reference = first_character_reference or global_valid
    if global_valid:
        warnings.append(
            "Global/group character reference sheet is stored as context only for HiDream multi-reference; use separate one-person character sheets for direct reference input."
        )
    return {
        "enabled": bool(enabled),
        "reference_sheet_path": legacy_reference,
        "global_reference_sheet_path": global_valid,
        "global_reference_direct_model_use": False,
        "global_reference_usage": "metadata_text_context_only",
        "reference_type": _safe_str(chosen.get("reference_type") or "per_character_reference_sheets"),
        "source": _safe_str(chosen.get("source") or ("user_loaded_image" if enabled else "none")),
        "mode": "reference_sheet",
        "hidream_multi_reference_policy": "per_character_sheets_only",
        "future_hidream_group_image_workflow": "hidream_image_edit",
        "max_character_reference_slots": 5,
        "character_reference_sheets": sheets,
        "available_reference_sheet_paths": available,
        "selected_reference_sheet_paths": selected_valid,
        "selected_reference_sheet_paths_source": selected_direct_source or "none",
        "loaded_reference_count": len([p for p in available if p and (not global_valid or os.path.normcase(p) != os.path.normcase(global_valid))]),
        "source_had_full_character_reference_sheets": bool(source_had_full_character_reference_sheets),
        "richest_reference_source": richest_reference_source,
        "merged_reference_sources": source_labels,
        "warnings": warnings[:20],
    }

def _character_reference_for_character_id(ref: Dict[str, Any], character_id: Any) -> str:
    cid = _safe_str(character_id)
    if not cid:
        return ""
    m = re.search(r"(\d+)$", cid)
    if not m:
        return ""
    try:
        idx = int(m.group(1))
    except Exception:
        idx = 0
    if idx < 1 or idx > 5:
        return ""
    sheets = ref.get("character_reference_sheets") if isinstance(ref.get("character_reference_sheets"), dict) else {}
    return _safe_str(sheets.get(f"char_{idx:02d}"))


def _select_character_reference_paths_for_item(item: Dict[str, Any], ref: Dict[str, Any]) -> tuple[List[str], str, List[str]]:
    """Select direct reference sheets from final shot subject intent.

    The old logic treated broad labels such as b-roll/background as a reason to
    suppress refs.  That broke subject shots that were still labelled as b-roll.
    This version only suppresses refs when the final shot concept/prompt has no
    visible human subject evidence.
    """
    warnings: List[str] = []
    if not _safe_bool(ref.get("enabled"), False):
        return [], "text_only_no_reference_enabled", warnings

    valid_slots, slot_warnings = _reference_valid_slot_paths(ref)
    warnings.extend(slot_warnings)

    global_path = _safe_str(ref.get("global_reference_sheet_path"))
    global_loaded = False
    if global_path:
        try:
            global_loaded = bool(os.path.isfile(global_path))
        except Exception:
            global_loaded = False
        if global_loaded:
            warnings.append("Global/group reference is loaded and kept as context; primary direct HiDream references come from slots 1-5.")
        else:
            warnings.append(f"Global/group reference path is missing or unreadable: {global_path}")

    if not valid_slots:
        if global_loaded:
            return [], "global_reference_loaded_context_only_no_slot_references", warnings
        return [], "no_valid_reference_sheets", warnings

    concept = item.get("director_scene_concept") if isinstance(item.get("director_scene_concept"), dict) else {}
    subject_info = _shot_subject_mode(item, concept=concept)
    if subject_info.get("shot_subject_mode") == "environment_only":
        return [], "environment_only_no_direct_subject_refs", warnings

    loaded_paths = [p for _k, p in valid_slots]
    loaded_count = len(loaded_paths)
    wanted_count = _reference_count_for_subject_shot(item, concept, loaded_count)
    char_ids = [_safe_str(x) for x in _as_list(item.get("character_ids")) if _safe_str(x)]
    selected: List[str] = []
    for cid in char_ids:
        pth = _character_reference_for_character_id(ref, cid)
        if pth:
            selected.append(pth)
        if len(selected) >= wanted_count:
            break
    if len(selected) < wanted_count:
        for pth in loaded_paths:
            if pth not in selected:
                selected.append(pth)
            if len(selected) >= wanted_count:
                break
    selected = list(dict.fromkeys(selected))[:max(1, min(5, wanted_count or 1))]
    if not selected:
        return [], "subject_present_but_no_reference_selected", warnings

    if len(selected) == 1 and loaded_count > 1:
        mode = "direct_reference_image_solo_deterministic"
    elif len(selected) == 2:
        mode = "direct_reference_image_duo"
    else:
        mode = "direct_reference_image_group"
    return selected, mode, warnings


def _character_reference_guidance(brief: Dict[str, str], ref: Dict[str, Any]) -> str:
    if not _safe_bool(ref.get("enabled"), False):
        return ""
    selected = [_safe_str(x) for x in _as_list(ref.get("selected_reference_sheet_paths")) if _safe_str(x)]
    if not selected:
        if _safe_str(ref.get("global_reference_sheet_path")):
            return "keep the stored identity context consistent in one clean full-frame scene"
        return ""
    if len(selected) > 1:
        return f"keep the {len(selected)} loaded identities distinct and consistent in one clean full-frame scene"
    return "keep the loaded identity consistent in one clean full-frame scene"

def _inject_character_reference_prompt(prompt: Any, brief: Dict[str, str], ref: Dict[str, Any], *, max_len: int = 1800) -> str:
    text = _safe_str(prompt)
    guide = _character_reference_guidance(brief, ref)
    if not text or not guide:
        return text[:max_len]
    low = text.lower()
    if guide.lower() in low or "loaded character reference" in low or "loaded reference sheet" in low:
        return text[:max_len]
    # Keep identity guidance before the compact safety line.  Appending after the
    # safety line can split the line and create duplicated "no collage" fragments
    # during final word limiting.
    safety = globals().get("_DIRECTOR_VISUAL_SAFETY_LINE", "single coherent full-frame scene, no text, no collage, no split-screen, no grid")
    if safety.lower() in low:
        base = re.sub(re.escape(safety), " ", text, flags=re.IGNORECASE)
        base = re.sub(r"\s{2,}", " ", base).strip(" ,.;")
        joined = _join_parts([base, guide, safety]) if "_join_parts" in globals() else (base.rstrip(" .") + ". " + guide + ". " + safety)
        return joined[:max_len]
    return _join_parts([text, guide])[:max_len] if "_join_parts" in globals() else (text.rstrip(" .") + ". " + guide)[:max_len]


def _character_reference_unsupported_warning(ref: Dict[str, Any]) -> str:
    if not _safe_bool(ref.get("enabled"), False):
        return ""
    if _safe_str(ref.get("global_reference_sheet_path")) and not _as_list(ref.get("selected_reference_sheet_paths")):
        return "Global/group sheet is stored as context only for HiDream multi-reference; use separate one-person character sheets for direct reference input."
    return "HiDream/selected image model reference sheet path saved, but direct reference-image input is not wired for this selected path; text-only identity guidance was used."


def _apply_character_reference_to_bibles(chars: Any, groups: Any, ref: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out_chars = [dict(x) for x in _as_list(chars) if isinstance(x, dict)]
    out_groups = [dict(x) for x in _as_list(groups) if isinstance(x, dict)]
    if not _safe_bool(ref.get("enabled"), False):
        return out_chars, out_groups
    global_path = _safe_str(ref.get("global_reference_sheet_path"))
    sheets = ref.get("character_reference_sheets") if isinstance(ref.get("character_reference_sheets"), dict) else {}
    for idx, ch in enumerate(out_chars, start=1):
        slot_key = f"char_{idx:02d}"
        path = _safe_str(sheets.get(slot_key))
        ch["uses_reference_sheet"] = bool(path)
        ch["reference_sheet_path"] = path
        ch["reference_slot"] = slot_key if path else ""
        if global_path and not path:
            ch["global_reference_context_path"] = global_path
    for grp in out_groups:
        member_ids = [_safe_str(x) for x in _as_list(grp.get("member_ids")) if _safe_str(x)]
        group_paths: List[str] = []
        for cid in member_ids[:5]:
            p = _character_reference_for_character_id(ref, cid)
            if p:
                group_paths.append(p)
        group_paths = _dedupe_texts(group_paths, max_items=5, max_len=4000) if "_dedupe_texts" in globals() else list(dict.fromkeys(group_paths))[:5]
        grp["uses_reference_sheet"] = bool(group_paths)
        grp["reference_sheet_path"] = group_paths[0] if group_paths else ""
        grp["reference_sheet_paths"] = group_paths
        if global_path:
            grp["global_reference_context_path"] = global_path
    return out_chars, out_groups


def _apply_character_reference_to_item(item: Dict[str, Any], brief: Dict[str, str], ref: Dict[str, Any], *, image_prompt_keys: Optional[List[str]] = None, passed_to_model: bool = False) -> Dict[str, Any]:
    out = dict(item)
    enabled = _safe_bool(ref.get("enabled"), False)
    concept = out.get("director_scene_concept") if isinstance(out.get("director_scene_concept"), dict) else {}
    subject_info = _shot_subject_mode(out, brief, concept)
    selected_paths, mode, select_warnings = _select_character_reference_paths_for_item(out, ref)
    available_reference_sheet_paths = _reference_available_paths_from_normalized(ref, limit=5) if "_reference_available_paths_from_normalized" in globals() else [_safe_str(x) for x in _as_list(ref.get("available_reference_sheet_paths")) if _safe_str(x)][:5]
    loaded_reference_count = len(available_reference_sheet_paths)
    wanted_reference_count, selection_reason = _reference_selection_intent_for_subject_shot(out, concept, loaded_reference_count) if "_reference_selection_intent_for_subject_shot" in globals() else (len(selected_paths), mode)
    collapsed_to_single_ref_warning = bool(loaded_reference_count > 1 and wanted_reference_count > 1 and len(selected_paths) == 1)
    model_passed = bool(passed_to_model and selected_paths and subject_info.get("shot_subject_mode") == "subject_present")
    ref_for_prompt = dict(ref)
    ref_for_prompt["selected_reference_sheet_paths"] = selected_paths
    has_global_context = bool(_safe_str(ref.get("global_reference_sheet_path")))
    out["shot_subject_mode"] = _safe_str(subject_info.get("shot_subject_mode"))
    out["visible_subject_detected"] = bool(subject_info.get("visible_subject_detected"))
    out["reference_eligible"] = bool(enabled and subject_info.get("shot_subject_mode") == "subject_present")
    out["reference_routing_reason"] = _safe_str(subject_info.get("reference_routing_reason"))
    out["skipped_reference_reason"] = "" if selected_paths else mode
    out["environment_prompt_omitted_subjects"] = bool(subject_info.get("environment_prompt_omitted_subjects"))
    out["uses_character_reference"] = bool(enabled and selected_paths)
    out["character_reference_path"] = selected_paths[0] if selected_paths else ""
    out["character_reference_sheet_paths"] = selected_paths
    out["selected_reference_sheet_paths"] = selected_paths
    out["available_reference_sheet_paths"] = available_reference_sheet_paths
    out["loaded_reference_count"] = loaded_reference_count
    out["wanted_reference_count"] = wanted_reference_count
    out["selection_reason"] = selection_reason
    out["collapsed_to_single_ref_warning"] = collapsed_to_single_ref_warning
    out["source_had_full_character_reference_sheets"] = bool(ref.get("source_had_full_character_reference_sheets"))
    out["character_reference_passed_to_image_model"] = model_passed
    out["image_model_reference_mode"] = "direct_reference_image" if model_passed else ("environment_only" if subject_info.get("shot_subject_mode") == "environment_only" else mode)
    out["character_reference"] = {
        "enabled": bool(enabled),
        "global_reference_sheet_path": _safe_str(ref.get("global_reference_sheet_path")),
        "global_reference_direct_model_use": False,
        "global_reference_usage": "metadata_text_context_only",
        "reference_sheet_path": selected_paths[0] if selected_paths else "",
        "selected_reference_sheet_paths": selected_paths,
        "selected_reference_sheet_paths_source": _safe_str(ref.get("selected_reference_sheet_paths_source") or "selection_router"),
        "available_reference_sheet_paths": available_reference_sheet_paths,
        "loaded_reference_count": loaded_reference_count,
        "wanted_reference_count": wanted_reference_count,
        "selection_reason": selection_reason,
        "source_had_full_character_reference_sheets": bool(ref.get("source_had_full_character_reference_sheets")),
        "collapsed_to_single_ref_warning": collapsed_to_single_ref_warning,
        "character_reference_sheets": ref.get("character_reference_sheets") if isinstance(ref.get("character_reference_sheets"), dict) else {},
        "max_character_reference_slots": 5,
        "reference_type": _safe_str(ref.get("reference_type") or "per_character_reference_sheets"),
        "source": _safe_str(ref.get("source") or ("user_loaded_image" if enabled else "none")),
        "passed_to_model": model_passed,
        "model_reference_mode": "direct_reference_image" if model_passed else mode,
        "global_reference_passed_to_model": False,
        "hidream_multi_reference_policy": "per_character_sheets_only",
        "mode": "reference_sheet",
        "shot_subject_mode": _safe_str(subject_info.get("shot_subject_mode")),
        "visible_subject_detected": bool(subject_info.get("visible_subject_detected")),
        "reference_eligible": bool(enabled and subject_info.get("shot_subject_mode") == "subject_present"),
        "reference_routing_reason": _safe_str(subject_info.get("reference_routing_reason")),
        "skipped_reference_reason": "" if selected_paths else mode,
        "environment_prompt_omitted_subjects": bool(subject_info.get("environment_prompt_omitted_subjects")),
        "warnings": select_warnings,
    }
    # Only append the compact identity hint when a subject is really visible and
    # direct refs were selected.  Environment shots stay clean: location/style/mood
    # only, no heavy anti-human wording and no reference workflow text.
    if enabled and selected_paths and subject_info.get("shot_subject_mode") == "subject_present":
        keys_to_update = ["image_prompt", "director_image_prompt", "template_image_prompt"] if image_prompt_keys is None else list(image_prompt_keys)
        for key in keys_to_update:
            if key in out and _safe_str(out.get(key)):
                out[key] = _inject_character_reference_prompt(out.get(key), brief, ref_for_prompt, max_len=1800)
    warnings = [_safe_str(x) for x in _as_list(out.get("character_reference_warnings")) if _safe_str(x)]
    for w in select_warnings:
        if w and w not in warnings:
            warnings.append(w)
    warn = _character_reference_unsupported_warning(ref) if enabled and selected_paths and not model_passed else ""
    if warn and warn not in warnings:
        warnings.append(warn)
    if warnings:
        out["character_reference_warnings"] = warnings[:10]
    return out


def _plan_character_reference_warnings(ref: Dict[str, Any], *, passed_to_model: bool = False) -> List[str]:
    warnings = [_safe_str(x) for x in _as_list(ref.get("warnings")) if _safe_str(x)]
    if _safe_bool(ref.get("enabled"), False) and not bool(passed_to_model):
        # Plans only store/reference the sheets; direct model pass-through happens later at start-image generation.
        w = "Character reference sheets are stored in the plan; direct image-model reference use depends on the selected start-image model."
        if w not in warnings:
            warnings.append(w)
    return warnings[:20]

def _overlapping_lyrics(scene_start: float, scene_end: float, lyric_segments: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for seg in lyric_segments:
        if not isinstance(seg, dict):
            continue
        start = _safe_float(seg.get("start"), 0.0)
        end = _safe_float(seg.get("end"), 0.0)
        if end <= start:
            continue
        if max(scene_start, start) < min(scene_end, end):
            text = _safe_str(seg.get("text"))
            if text:
                parts.append(text)
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    if len(text) > 700:
        text = text[:697].rstrip() + "..."
    return text


def _normalize_scene(item: Any, index: int, lyric_segments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(item, dict):
        return None
    idx = _safe_int(item.get("index"), index)
    if idx <= 0:
        idx = index

    start = _safe_float(item.get("start"), 0.0)
    end = _safe_float(item.get("end"), 0.0)
    duration = _safe_float(item.get("duration"), max(0.0, end - start))
    if end <= start and duration > 0.0:
        end = start + duration
    if duration <= 0.0 and end > start:
        duration = end - start
    if end <= start or duration <= 0.0:
        return None

    lyric_text = _safe_str(item.get("lyrics") or item.get("lyric_text"))
    if not lyric_text and lyric_segments:
        lyric_text = _overlapping_lyrics(start, end, lyric_segments)

    scene_id = _safe_str(item.get("id")) or f"S{idx:02d}"
    if not re.match(r"^S\d+", scene_id, flags=re.IGNORECASE):
        scene_id = f"S{idx:02d}"
    else:
        m = re.search(r"\d+", scene_id)
        if m:
            scene_id = f"S{int(m.group(0)):02d}"

    out: Dict[str, Any] = {
        "id": scene_id,
        "index": idx,
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(duration, 3),
        "section": _safe_str(item.get("section") or item.get("section_label"), "unknown") or "unknown",
        "energy": _safe_str(item.get("energy") or item.get("energy_class"), "mid") or "mid",
        "beat_strength": round(_safe_float(item.get("beat_strength"), 0.0), 4),
        "transition_intensity": round(_safe_float(item.get("transition_intensity"), 0.0), 4),
        "effect_intensity": round(_safe_float(item.get("effect_intensity"), 0.0), 4),
        "preferred_clip_role": _safe_str(item.get("preferred_clip_role")),
        "lyrics": lyric_text,
        "lyric_text": lyric_text,
        "notes": _safe_str(item.get("notes")),
    }

    # Preserve a few future-friendly optional timing fields when present.
    for key in ("beat_count", "lyric_start", "lyric_end", "active_lyric_start", "active_lyric_end", "srt_path", "active_vocal_window", "vocal_timing_reason"):
        if key in item:
            if key in ("beat_count",):
                out[key] = _safe_int(item.get(key), 0)
            elif key in ("lyric_start", "lyric_end", "active_lyric_start", "active_lyric_end"):
                out[key] = round(_safe_float(item.get(key), 0.0), 3)
            elif key == "active_vocal_window":
                out[key] = _safe_bool(item.get(key), False)
            else:
                out[key] = _safe_str(item.get(key))
    return out


def _normalize_scenes(scene_items: Any, lyric_items: Any) -> List[Dict[str, Any]]:
    lyric_segments: List[Dict[str, Any]] = [x for x in _as_list(lyric_items) if isinstance(x, dict)]
    scenes: List[Dict[str, Any]] = []
    for i, item in enumerate(_as_list(scene_items), start=1):
        scene = _normalize_scene(item, i, lyric_segments)
        if scene is not None:
            # Re-number sequentially so the exported IDs are clean even if old data was odd.
            scene["index"] = len(scenes) + 1
            scene["id"] = f"S{len(scenes) + 1:02d}"
            scenes.append(scene)
    return scenes


def export_musicclip_scene_plan(payload: dict) -> dict:
    """Write a music-video scene plan JSON and return a small status dict.

    This function never raises UI-facing exceptions. Any failure is returned as
    {"ok": False, "message": "..."} so Music Clip Creator can remain stable.
    """
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Bridge payload was not a dictionary."}

        audio_path = _safe_str(payload.get("audio_path"))
        if not audio_path:
            return {"ok": False, "message": "No audio path was provided."}
        if not os.path.isfile(audio_path):
            return {"ok": False, "message": f"Audio file was not found: {audio_path}"}

        output_dir = _ltx_job_output_dir_from_payload(payload, audio_path)
        plan_path = output_dir / "musicclip_scene_plan.json"

        scenes = _normalize_scenes(
            payload.get("smart_scene_map") or payload.get("scenes") or [],
            payload.get("lyric_segments") or [],
        )
        if not scenes:
            return {"ok": False, "message": "No Smart Scene Map scenes were provided."}

        srt_path = _safe_str(payload.get("srt_path"))
        bridge_generation_settings = _normalize_bridge_generation_settings(payload)
        duration_profile = _duration_profile_from_bridge_settings(bridge_generation_settings)
        microclip_profile = _microclip_profile_from_bridge_settings(bridge_generation_settings, duration_profile)
        effects_profile = _effects_profile_from_bridge_settings(bridge_generation_settings)
        character_reference = _character_reference_from_sources(payload)
        plan = {
            "version": _VERSION,
            "source": _SOURCE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "audio_path": audio_path,
            "srt_path": srt_path,
            "musicclip_output_dir": _safe_str(payload.get("output_dir")),
            "bridge_output_dir": str(output_dir),
            "smart_scene_map_enabled": _safe_bool(payload.get("smart_scene_map_enabled"), True),
            "use_lyrics_srt_timing": _safe_bool(payload.get("use_lyrics_srt_timing"), False),
            "selected_clip_preset": _safe_str(payload.get("selected_clip_preset")),
            "director_preset": _safe_str(payload.get("director_preset")),
            "scene_length": _safe_str(payload.get("scene_length") or payload.get("scene_cut_style")),
            "cut_style": _safe_str(payload.get("cut_style") or payload.get("scene_cut_style")),
            "cut_timing_offset": _safe_float(payload.get("cut_timing_offset"), 0.0),
            "bridge_generation_settings": bridge_generation_settings,
            "duration_profile": duration_profile,
            "microclip_profile": microclip_profile,
            "effects_profile": effects_profile,
            "character_reference": {k: v for k, v in character_reference.items() if k != "warnings"},
            "character_reference_warnings": _plan_character_reference_warnings(character_reference, passed_to_model=False),
            "scene_count": len(scenes),
            "creative_brief": _normalize_creative_brief(payload.get("creative_brief")),
            "lyrics_srt_segments": [x for x in _as_list(payload.get("lyric_segments")) if isinstance(x, dict)],
            "scenes": scenes,
        }

        with plan_path.open("w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, ensure_ascii=False)

        return {
            "ok": True,
            "plan_path": str(plan_path),
            "output_dir": str(output_dir),
            "scene_count": len(scenes),
            "message": f"Scene plan exported: {len(scenes)} scenes.",
        }
    except Exception as exc:
        return {"ok": False, "message": f"Scene plan export failed: {exc}"}

# -----------------------------
# Chunk 4B/4C: scene plan -> prompt plan with vocal/non-vocal role planning
# -----------------------------
_PROMPT_BACKEND_TEMPLATE = "template_rule_based"

# Role/timing guards for SRT-driven lipsync decisions.
# Long SRT segments often cover instrumental space after one short phrase; do not
# treat the whole segment as continuous singing.
LONG_LYRIC_SEGMENT_SECONDS = 8.0
MAX_LIPSYNC_SCENES_PER_REPEATED_LYRIC = 1
MAX_LIPSYNC_SECONDS_PER_LONG_SEGMENT = 3.5
MAX_DELAYED_LIPSYNC_START_SECONDS = 1.25
ACTIVE_LYRIC_NEAR_SHOT_START_SECONDS = 0.65
ACTIVE_LYRIC_MIN_OVERLAP_SECONDS = 0.35
LONG_BLOCK_STRICT_START_GRACE_SECONDS = 0.45
INSTRUMENTAL_TAIL_GUARD_SECONDS = 40.0

# LTX shot-boundary guard. Keep these fixed/private for now; no UI/settings.
LYRIC_START_PAD_SECONDS = 0.20
LYRIC_END_PAD_SECONDS = 0.35
LYRIC_BOUNDARY_SNAP_TOLERANCE_SECONDS = 1.25
LYRIC_BOUNDARY_EPS_SECONDS = 0.08
LTX_SOFT_MAX_SHOT_SECONDS = 10.0
LTX_HARD_MAX_SHOT_SECONDS = 12.0


def _bridge_first_value(*values: Any) -> Any:
    for value in values:
        if isinstance(value, str):
            if value.strip():
                return value
        elif value is not None:
            return value
    return None


def _normalize_bridge_generation_settings(*sources: Any) -> Dict[str, Any]:
    """Normalize existing Music Clip Creator settings for bridge planning only.

    This does not create/persist settings. It only carries already-existing UI
    values through scene/prompt/LTX/director plan JSON so grouping can honor them.
    """
    merged: Dict[str, Any] = {}
    for src in sources:
        if not isinstance(src, dict):
            continue
        nested = src.get("bridge_generation_settings")
        if isinstance(nested, dict):
            for k, v in nested.items():
                if v not in (None, ""):
                    merged[k] = v
        for k, v in src.items():
            if k == "bridge_generation_settings":
                continue
            if k not in merged and v not in (None, ""):
                merged[k] = v

    preset = _clean_text(_bridge_first_value(
        merged.get("preset"), merged.get("director_preset"), merged.get("music_video_preset"), merged.get("selected_clip_preset")
    )) or "Balanced"
    scene_cut_style = _clean_text(_bridge_first_value(
        merged.get("scene_cut_style"), merged.get("cut_style"), merged.get("scene_length"), merged.get("clip_length_mode")
    )) or "Auto"
    max_mode = _clean_text(_bridge_first_value(
        merged.get("smart_max_scene_mode"), merged.get("max_scene_mode"), merged.get("clip_length_mode")
    )) or "Auto"
    beat_style = _clean_text(_bridge_first_value(merged.get("beat_style"), merged.get("scene_cut_style"), merged.get("cut_style"))) or scene_cut_style

    manual = None
    for key in ("manual_max_clip_seconds", "manual_max_scene_seconds", "max_ltx_shot_seconds", "smart_max_scene_len", "manual_clip_seconds"):
        try:
            val = float(merged.get(key))
            if val > 0.05:
                manual = val
                break
        except Exception:
            continue

    mode_low = f"{max_mode} {scene_cut_style}".lower()
    if "manual" in mode_low and manual:
        clip_mode = "manual"
    elif "short" in mode_low or "busy" in mode_low:
        clip_mode = "short"
    elif "long" in mode_low or "cinematic" in preset.lower() or "emotional" in preset.lower():
        clip_mode = "long"
    elif "medium" in mode_low:
        clip_mode = "medium"
    else:
        clip_mode = "auto"

    ltx_backend = _clean_text(_bridge_first_value(
        merged.get("ltx_backend"), merged.get("ltx_generation_backend"), merged.get("generation_backend")
    )).lower()
    ltx_resolution = _clean_text(_bridge_first_value(
        merged.get("ltx_resolution"), merged.get("resolution"), merged.get("output_resolution")
    ))
    output_resolution = _clean_text(merged.get("output_resolution"))
    ltx_aspect_mode = _clean_text(merged.get("ltx_aspect_mode"))

    cfg: Dict[str, Any] = {
        "preset": preset,
        "clip_length_mode": clip_mode,
        "manual_max_clip_seconds": round(float(manual), 3) if manual else None,
        "scene_cut_style": scene_cut_style,
        "smart_max_scene_mode": max_mode,
        "beat_style": beat_style,
        "ltx_backend": ltx_backend,
        "ltx_resolution": ltx_resolution,
        "resolution": ltx_resolution,
        "output_resolution": output_resolution,
        "ltx_aspect_mode": ltx_aspect_mode,
        "ltx_max_generation_seconds": _safe_float(merged.get("ltx_max_generation_seconds"), 0.0) or None,
        "ltx_max_generation_frames": _safe_int(merged.get("ltx_max_generation_frames"), 0) or None,
        "hard_max_ltx_shot_seconds": _safe_float(merged.get("hard_max_ltx_shot_seconds"), 0.0) or None,
        "timestamped_microclips_enabled": _safe_bool(merged.get("timestamped_microclips_enabled"), False),
        "collage_effect_enabled": _safe_bool(merged.get("collage_effect_enabled"), False),
        "avoid_effects_in_first_clip": _safe_bool(merged.get("avoid_effects_in_first_clip"), True),
        "avoid_effects_in_last_clip": _safe_bool(merged.get("avoid_effects_in_last_clip"), True),
        "avoid_short_start_end_clips": _safe_bool(merged.get("avoid_short_start_end_clips"), _safe_bool(merged.get("ltx_avoid_short_start_end"), True)),
    }
    profile = _duration_profile_from_bridge_settings(cfg)
    cfg["target_clip_seconds"] = profile.get("target_clip_seconds")
    cfg["max_ltx_shot_seconds"] = profile.get("max_ltx_shot_seconds")
    return cfg


def _duration_profile_from_bridge_settings(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    settings = settings if isinstance(settings, dict) else {}
    preset = _clean_text(settings.get("preset")) or "Balanced"
    mode = _clean_text(settings.get("clip_length_mode")).lower() or "auto"
    manual = _safe_float(settings.get("manual_max_clip_seconds"), 0.0)
    explicit_target = _safe_float(settings.get("target_clip_seconds"), 0.0)
    explicit_max = _safe_float(settings.get("max_ltx_shot_seconds"), 0.0)
    explicit_hard = _safe_float(settings.get("hard_max_ltx_shot_seconds"), 0.0)
    max_generation_seconds = _safe_float(settings.get("ltx_max_generation_seconds"), 0.0)
    ltx_backend = _safe_str(settings.get("ltx_backend") or settings.get("ltx_generation_backend")).lower().replace("-", "_")
    vramlab_strict_cap = bool(ltx_backend in {"vramlab", "vram_lab", "framevision_vramlab", "own_ltx"} or max_generation_seconds > 0.0 or _safe_int(settings.get("ltx_max_generation_frames"), 0) > 0)
    preset_low = preset.lower()

    # Small profile mapping only; no new UI/settings.
    if "dnb" in preset_low or "drum" in preset_low or "chaos" in preset_low:
        target, max_ltx, min_lip, fast = 4.0, 5.0, 3.0, True
        source = "preset"
    elif "edm" in preset_low or "festival" in preset_low:
        target, max_ltx, min_lip, fast = 4.5, 5.5, 3.0, True
        source = "preset"
    elif "cinematic" in preset_low:
        target, max_ltx, min_lip, fast = 9.0, 12.0, 3.5, False
        source = "preset"
    elif "emotional" in preset_low or "vocal" in preset_low or "pop" in preset_low:
        target, max_ltx, min_lip, fast = 7.0, 10.0, 3.5, False
        source = "preset"
    else:
        target, max_ltx, min_lip, fast = 7.0, float(LTX_SOFT_MAX_SHOT_SECONDS), 3.0, False
        source = "auto"

    if mode == "short":
        target, max_ltx, source = min(target, 4.0), min(max_ltx, 5.0), "preset" if source == "preset" else "mode"
    elif mode == "medium":
        target, max_ltx, source = 7.0, min(max(max_ltx, 7.0), 10.0), "mode"
    elif mode == "long":
        target, max_ltx, source = max(target, 10.0), max(max_ltx, 14.0), "mode"
    elif mode == "manual" and manual > 0.05:
        target, max_ltx, source = manual, manual, "manual"

    if explicit_target > 0.05:
        target = float(explicit_target)
    if explicit_max > 0.05:
        max_ltx = float(explicit_max)
    if vramlab_strict_cap:
        cap_seconds = max_generation_seconds if max_generation_seconds > 0.05 else 10.0
        target = min(float(target), float(cap_seconds))
        max_ltx = min(float(max_ltx), float(cap_seconds))

    hard = max_ltx + (0.6 if source == "manual" else 1.5)
    hard = max(hard, max_ltx)
    # Absolute fallback cap keeps old auto behavior compatible, but manual keeps control.
    if source != "manual":
        hard = max(hard, float(LTX_HARD_MAX_SHOT_SECONDS)) if mode == "auto" else hard
    if explicit_hard > 0.05:
        hard = float(explicit_hard)
    if vramlab_strict_cap:
        cap_seconds = max_generation_seconds if max_generation_seconds > 0.05 else 10.0
        hard = min(float(hard), float(cap_seconds))

    return {
        "source": source,
        "preset": preset,
        "clip_length_mode": mode or "auto",
        "target_clip_seconds": round(float(target), 3),
        "max_ltx_shot_seconds": round(float(max_ltx), 3),
        "hard_max_ltx_shot_seconds": round(float(hard), 3),
        "min_lipsync_seconds": round(float(min_lip), 3),
        "prefer_fast_nonvocal_cuts": bool(fast),
        "strict_generation_frame_cap": bool(vramlab_strict_cap),
        "ltx_max_generation_seconds": round(float(max_generation_seconds), 3) if max_generation_seconds > 0.05 else None,
        "ltx_max_generation_frames": _safe_int(settings.get("ltx_max_generation_frames"), 0) or None,
    }


def _duration_profile_short_copy(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    profile = profile if isinstance(profile, dict) else _duration_profile_from_bridge_settings({})
    keys = ("source", "preset", "clip_length_mode", "target_clip_seconds", "max_ltx_shot_seconds", "hard_max_ltx_shot_seconds", "min_lipsync_seconds", "prefer_fast_nonvocal_cuts")
    return {k: profile.get(k) for k in keys if k in profile}


def _effects_profile_from_bridge_settings(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Bridge-only effects policy from the existing JSON settings system.

    Timestamped microclips and collage are intentionally separate. Microclips are
    sequential full-frame timestamped cuts inside one LTX prompt. Collage is a
    rare optional effect and is never part of normal microclip behavior.
    """
    settings = settings if isinstance(settings, dict) else {}
    return {
        "timestamped_microclips_enabled": _safe_bool(settings.get("timestamped_microclips_enabled"), False),
        "collage_effect_enabled": _safe_bool(settings.get("collage_effect_enabled"), False),
        "avoid_effects_in_first_clip": _safe_bool(settings.get("avoid_effects_in_first_clip"), True),
        "avoid_effects_in_last_clip": _safe_bool(settings.get("avoid_effects_in_last_clip"), True),
        "max_collage_events_per_video": 1,
        "microclips_are_sequential_full_frame": True,
        "collage_is_not_microclips": True,
    }


def _effects_profile_short_copy(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    profile = profile if isinstance(profile, dict) else _effects_profile_from_bridge_settings({})
    keys = (
        "timestamped_microclips_enabled", "collage_effect_enabled",
        "avoid_effects_in_first_clip", "avoid_effects_in_last_clip",
        "max_collage_events_per_video", "microclips_are_sequential_full_frame",
        "collage_is_not_microclips",
    )
    return {k: profile.get(k) for k in keys if k in profile}


def _microclip_profile_from_bridge_settings(settings: Optional[Dict[str, Any]] = None, duration_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a bridge-only microclip profile from existing UI settings.

    This reuses the useful old microclip timing idea (fast beat-hit moments in
    energetic sections), but intentionally does not reuse the old chorus/drop/
    verse/whole-track toggles as authority. Section labels can help explain a
    choice, but they never enable microclips by themselves.
    """
    settings = settings if isinstance(settings, dict) else {}
    profile = duration_profile if isinstance(duration_profile, dict) else _duration_profile_from_bridge_settings(settings)
    preset = _clean_text(settings.get("preset") or settings.get("director_preset")) or _clean_text(profile.get("preset")) or "Balanced"
    cut_style = _clean_text(settings.get("scene_cut_style") or settings.get("cut_style") or settings.get("beat_style")) or "Auto"
    mode = _clean_text(settings.get("clip_length_mode") or profile.get("clip_length_mode")).lower() or "auto"
    manual = _safe_float(settings.get("manual_max_clip_seconds"), 0.0)
    max_ltx = _safe_float(profile.get("max_ltx_shot_seconds"), 0.0)
    blob = f"{preset} {cut_style} {mode}".lower()
    flashy_preset = any(x in blob for x in ("edm", "festival", "dnb", "drum", "chaos", "busy"))
    short_mode = mode == "short" or "short" in blob or "busy" in blob
    manual_short = bool(manual > 0.05 and manual <= 5.25)
    effects_profile = _effects_profile_from_bridge_settings(settings)
    user_enabled = bool(effects_profile.get("timestamped_microclips_enabled"))
    profile_would_allow = bool(flashy_preset or short_mode or manual_short or bool(profile.get("prefer_fast_nonvocal_cuts")))
    enabled = bool(user_enabled and profile_would_allow)
    if enabled:
        if flashy_preset and (short_mode or manual_short):
            source = "preset_short_mode"
        elif flashy_preset:
            source = "preset"
        elif manual_short:
            source = "manual_max_clip_seconds"
        else:
            source = "short_cut_style"
    else:
        source = "disabled_non_flashy_profile"
    return {
        "enabled": bool(enabled),
        "source": source if user_enabled else "disabled_by_effects_toggle",
        "would_enable_from_preset": bool(profile_would_allow),
        "effects_toggle_enabled": bool(user_enabled),
        "preset": preset,
        "cut_style": cut_style,
        "clip_length_mode": mode,
        "manual_max_clip_seconds": round(float(manual), 3) if manual > 0.05 else None,
        "max_ltx_shot_seconds": round(float(max_ltx), 3) if max_ltx > 0.0 else None,
        "microclip_allowed_for_vocals": False,
        "high_energy_seconds": [0.4, 0.9],
        "mid_energy_seconds": [0.6, 1.2],
        "low_energy_seconds": [0.8, 1.5],
        "ltx_strategy": "group 2-4 micro-moments inside one practical 3-5 second LTX shot",
    }


def _microclip_profile_short_copy(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    profile = profile if isinstance(profile, dict) else _microclip_profile_from_bridge_settings({})
    keys = ("enabled", "source", "would_enable_from_preset", "effects_toggle_enabled", "preset", "cut_style", "clip_length_mode", "manual_max_clip_seconds", "max_ltx_shot_seconds", "microclip_allowed_for_vocals", "ltx_strategy")
    return {k: profile.get(k) for k in keys if k in profile}


def _scene_energy_rank(scene: Dict[str, Any]) -> int:
    text = _clean_text(scene.get("energy") or scene.get("energy_class") or scene.get("energy_summary") or "mid").lower()
    if any(x in text for x in ("high", "peak", "drop", "intense", "active", "fast")):
        return 2
    if any(x in text for x in ("low", "calm", "break", "soft", "ambient")):
        return 0
    return 1


def _group_energy_rank(group: List[Dict[str, Any]]) -> int:
    if not group:
        return 1
    return max(_scene_energy_rank(s) for s in group if isinstance(s, dict))


def _group_peak_beat_strength(group: List[Dict[str, Any]]) -> float:
    vals: List[float] = []
    for s in group or []:
        if not isinstance(s, dict):
            continue
        vals.append(_safe_float(s.get("beat_strength"), 0.0))
        vals.append(_safe_float(s.get("beat_score"), 0.0))
        vals.append(_safe_float(s.get("impact_score"), 0.0))
    return max(vals + [0.0])


def _group_role_blob(group: List[Dict[str, Any]]) -> str:
    bits: List[str] = []
    for s in group or []:
        if not isinstance(s, dict):
            continue
        for key in ("preferred_clip_role", "scene_role", "suggested_shot_type", "section", "energy"):
            bits.append(_clean_text(s.get(key)))
    return " ".join(bits).lower()


def _microclip_style_for_group(group: List[Dict[str, Any]], brief: Optional[Dict[str, str]] = None) -> str:
    blob = _group_role_blob(group)
    brief_blob = " ".join([_clean_text(v) for v in (brief or {}).values()]).lower() if isinstance(brief, dict) else ""
    if any(x in blob + " " + brief_blob for x in ("car", "vehicle", "drive", "racing", "wheel", "dashboard", "highway")):
        return "vehicle_detail"
    if any(x in blob for x in ("dance", "dancer", "body", "boots", "hair", "performer")):
        return "dance_detail"
    if any(x in blob for x in ("transition", "whip", "flash", "strobe", "light")):
        return "camera_flash"
    if any(x in blob for x in ("impact", "hit", "drop", "action")):
        return "beat_hit_cutaway"
    return "b_roll_impact"


def _microclip_reason_for_group(group: List[Dict[str, Any]], profile: Dict[str, Any], duration: float) -> str:
    rank = _group_energy_rank(group)
    beat = _group_peak_beat_strength(group)
    blob = _group_role_blob(group)
    reasons = [f"profile={_safe_str(profile.get('source')) or 'enabled'}"]
    if rank >= 2:
        reasons.append("high_energy")
    if beat >= 0.62:
        reasons.append(f"beat_strength={beat:.2f}")
    if any(x in blob for x in ("impact", "transition", "b-roll", "b_roll", "action", "dance", "detail", "hit", "flash")):
        reasons.append("impact_or_broll_role")
    reasons.append(f"duration={duration:.2f}s")
    reasons.append("non_vocal_safe")
    return ", ".join(reasons)


def _microclip_moment_count(duration: float, energy_rank: int) -> int:
    # LTX microclips are an intentional high-energy mode: several fast
    # timestamped full-frame moments inside one short generated clip.  Keep the
    # count practical for LTX, but allow enough moments for 3-4 second chaotic
    # clips without starting a brand-new moment at the very end.
    dur = max(0.1, _safe_float(duration, 0.0))
    if dur < 2.4:
        return 3
    if energy_rank >= 2:
        return max(3, min(5, int(round(dur / 0.8))))
    if energy_rank == 1:
        return max(3, min(5, int(round(dur / 1.0))))
    return max(2, min(4, int(round(dur / 1.25))))


_LTX_MICROCLIP_END_HOLD_SECONDS = 0.9
_LTX_MICROCLIP_ENDING_GUARD = "After the final timestamp, hold and resolve the same chaotic visual energy through the ending; do not introduce a new camera move, action, location, subject change, or last-second transition."


def _ltx_microclip_time_label(seconds: float) -> str:
    try:
        value = max(0.0, float(seconds))
    except Exception:
        value = 0.0
    value = round(value, 1)
    minutes = int(value // 60)
    sec_value = value - (minutes * 60)
    whole = int(sec_value)
    frac = int(round((sec_value - whole) * 10))
    if frac >= 10:
        whole += 1
        frac = 0
    if whole >= 60:
        minutes += whole // 60
        whole = whole % 60
    if frac <= 0:
        return f"{minutes}:{whole:02d}"
    return f"{minutes}:{whole:02d}.{frac}"


def _ltx_microclip_times(duration: float, moment_count: int) -> List[float]:
    dur = max(0.1, _safe_float(duration, 0.0))
    latest_start = max(0.0, dur - _LTX_MICROCLIP_END_HOLD_SECONDS)
    if latest_start <= 0.05:
        return [0.0]

    # Fast enough to feel like real microclips, but never inside the protected
    # ending hold window. For 3s this lands around 0.0/0.7/1.4/2.1; for 4s
    # around 0.0/0.8/1.6/2.3/3.1.
    target_step = 0.7 if dur <= 4.25 else 0.85
    by_step_count = int(math.floor((latest_start + 1e-6) / target_step)) + 1
    wanted = max(2, min(6, int(moment_count or by_step_count)))
    count = max(2, min(6, max(by_step_count, wanted)))
    max_possible = int(math.floor((latest_start + 1e-6) / 0.5)) + 1
    count = min(count, max(2, max_possible))
    if count <= 1:
        return [0.0]

    out: List[float] = []
    for i in range(count):
        value = (latest_start * i) / float(count - 1)
        # Match the label precision so duplicate labels do not appear after
        # rounding.
        value = round(max(0.0, min(latest_start, value)), 1)
        if out and abs(value - out[-1]) < 0.45:
            continue
        out.append(value)
    if not out or out[0] != 0.0:
        out.insert(0, 0.0)
    return out


def _microclip_allowed_for_group(group: List[Dict[str, Any]], *, duration: float, needs_lipsync: bool, clip_lyrics: Optional[List[Dict[str, Any]]], microclip_profile: Optional[Dict[str, Any]], duration_profile: Optional[Dict[str, Any]]) -> tuple:
    profile = microclip_profile if isinstance(microclip_profile, dict) else _microclip_profile_from_bridge_settings({}, duration_profile)
    if not bool(profile.get("enabled")):
        return False, "microclip profile disabled"
    if needs_lipsync:
        return False, "active lipsync shot"
    if clip_lyrics:
        return False, "active lyric window present"
    if any(_scene_has_strong_vocal(s) for s in group if isinstance(s, dict)):
        return False, "source group contains protected vocal scene"
    # LTX microclip mode is for practical short generated clips that contain
    # 2-4 beat details, not very long montage shots.
    max_ltx = _safe_float((duration_profile or {}).get("max_ltx_shot_seconds"), 5.0)
    if duration > max(5.75, max_ltx + 0.75):
        return False, f"duration {duration:.2f}s too long for safe microclip grouping"
    rank = _group_energy_rank(group)
    beat = _group_peak_beat_strength(group)
    blob = _group_role_blob(group)
    role_hit = any(x in blob for x in ("impact", "transition", "b-roll", "b_roll", "action", "dance", "detail", "hit", "flash", "whip", "vehicle", "drive"))
    # Do not trust section labels alone. A word like drop/chorus only helps when
    # the energy/beat/role evidence also says this is a visual hit.
    if rank >= 2 or beat >= 0.62 or role_hit:
        return True, _microclip_reason_for_group(group, profile, duration)
    return False, "not enough energy/beat/role evidence"


def _microclip_timestamped_video_prompt(brief: Dict[str, str], group: List[Dict[str, Any]], duration: float, style: str, moment_count: int) -> str:
    """Create real microclips: sequential full-frame timestamped cuts.

    Important: this is not collage/split-screen. Every timestamp describes one
    full-frame view after the previous one. The first and final timestamps stay
    clean so LTX has a stable opening/ending image flow.
    """
    first_index = _safe_int((group[0] if group else {}).get("index"), 1) if group else 1
    world = _dominant_location_for_index(brief, first_index) or _visual_world(brief)
    subject = _lead_subject(brief)
    style_text = _style_text(brief)
    times = _ltx_microclip_times(duration, int(moment_count or 4))
    count = max(1, len(times))

    def _role_action(n: int, first: bool = False, last: bool = False) -> str:
        if first:
            return f"immediate intense full-frame beat-hit in one coherent {world} setup, flashy strobe-like energy from the first frame"
        if last:
            return "hold the current chaotic full-frame visual energy and resolve it cleanly through the ending, ready for the next cut"
        if style == "vehicle_detail":
            pool = [
                "quick full-frame detail angle of the current movement or vehicle action on the beat",
                "fast full-frame reflection or texture detail from the same scene",
                "quick full-frame side angle that keeps the same location and subject continuity",
            ]
        elif style == "dance_detail":
            pool = [
                "quick full-frame closer angle of the same movement on the beat",
                "fast full-frame body, costume, or motion detail from the same performer setup",
                "sharp full-frame side angle that keeps the same scene and rhythm",
            ]
        elif style == "camera_flash":
            pool = [
                "quick full-frame light or camera-energy beat hit inside the same scene",
                "fast full-frame camera whip that stays in the same location",
                "brief full-frame visual accent that resolves back to the same setup",
            ]
        else:
            pool = [
                "fast full-frame detail shot lands on the beat",
                "quick full-frame camera angle change inside the same scene",
                "short full-frame b-roll punctuation keeps the same world and subject continuity",
            ]
        return pool[n % len(pool)]

    lines: List[str] = []
    for i, t in enumerate(times):
        action = _role_action(i, first=(i == 0), last=(i == count - 1))
        if i == 0:
            line = f"{action}; one location, one subject setup, {style_text}, clean full-frame view"
        elif i == count - 1:
            line = f"{action}; no visible singing, no microphone, clean full-frame view"
        else:
            who = f" with {subject}" if subject and i == 1 else ""
            line = f"sequential full-frame cut{who}: {action}; one view at a time, not simultaneous"
        clean = _strip_non_lipsync_vocal_language(_remove_collage_language(_clean_text(line))).rstrip('.')
        lines.append(f"{_ltx_microclip_time_label(t)} - {clean}.")
    out = _strip_non_lipsync_vocal_language(re.sub(r"\s+", " ", " ".join(lines)).strip())
    if _LTX_MICROCLIP_ENDING_GUARD.lower() not in out.lower():
        out = _sentence(out + " " + _LTX_MICROCLIP_ENDING_GUARD)
    return out


def _read_json_file(path: Any) -> Dict[str, Any]:
    p = Path(_safe_str(path))
    if not p.is_file():
        raise FileNotFoundError(str(p))
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("JSON root was not an object.")
    return data


def _write_json_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _clean_text(value: Any, max_len: int = 0) -> str:
    text = _safe_str(value)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" \t,.;")
    if max_len and len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip(" ,.;") + "..."
    return text


def _sentence(value: Any, fallback: str = "") -> str:
    text = _clean_text(value) or _clean_text(fallback)
    if not text:
        return ""
    text = text.strip()
    if text[-1:] not in ".!?":
        text += "."
    return text


def _join_parts(parts: List[str]) -> str:
    out: List[str] = []
    seen = set()
    for part in parts:
        text = _clean_text(part)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return ", ".join(out)



# -----------------------------
# Character Bible helpers
# -----------------------------
_COUNT_WORDS = {
    "one": 1, "single": 1, "solo": 1,
    "two": 2, "duo": 2, "pair": 2,
    "three": 3, "trio": 3,
    "four": 4, "quartet": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

_GENERIC_IDENTITY_NEGATIVES_SINGLE = [
    "different person",
    "changing face",
    "changing hairstyle",
    "inconsistent outfit",
    "duplicate performer",
    "identity drift",
]
_GENERIC_IDENTITY_NEGATIVES_GROUP = [
    "random extra members",
    "missing members",
    "duplicated member",
    "merging faces",
    "changing people between shots",
    "identity drift",
    "inconsistent outfits",
    "inconsistent hairstyles",
]


def _extract_subject_count(subject: str) -> int:
    text = _clean_text(subject).lower()
    if not text:
        return 1
    m = re.search(r"\b(?:of|with)?\s*(\d{1,2})\b", text)
    if m:
        n = _safe_int(m.group(1), 1)
        if 1 <= n <= 12:
            return n
    for word, value in _COUNT_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", text):
            return value
    if "duo" in text:
        return 2
    if "trio" in text:
        return 3
    if "quartet" in text:
        return 4
    if any(x in text for x in ("band", "group", "choir", "girls", "boys", "women", "men", "singers", "dancers", "performers")):
        return 4
    return 1


def _subject_is_group(subject: str, count: int) -> bool:
    text = _clean_text(subject).lower()
    return bool(count > 1 or any(x in text for x in ("band", "group", "choir", "duo", "trio", "quartet", "crew", "team", "singers", "dancers", "performers")))


def _neutral_role_for_index(subject: str, index: int, count: int) -> str:
    low = _clean_text(subject).lower()
    if index == 1:
        if any(x in low for x in ("rapper", "rap")):
            return "lead_rapper"
        if any(x in low for x in ("singer", "vocal", "choir", "band", "artist", "pop", "song")):
            return "lead_singer"
        return "lead_performer"
    if "choir" in low or "vocal" in low or "singers" in low:
        return f"vocalist_{index}"
    if "dancer" in low or "dance" in low:
        return f"dancer_{index}"
    return f"performer_{index}"


def _neutral_label_for_role(role: str) -> str:
    text = _clean_text(role.replace("_", " "))
    return text[:1].upper() + text[1:] if text else "Performer"


def _character_context_style_phrase(brief: Dict[str, str]) -> str:
    parts = [
        _clean_text(brief.get("characters_subjects"), 220),
        _clean_text(brief.get("style_theme"), 220),
        _clean_text(brief.get("main_idea"), 220),
        _clean_text(brief.get("locations_world"), 220),
    ]
    return _join_parts([p for p in parts if p]) or "the user-provided subject and visual style"


def _identity_negative_prompt(role_label: str, count: int, gender: str = "") -> str:
    base = _GENERIC_IDENTITY_NEGATIVES_GROUP if count > 1 else _GENERIC_IDENTITY_NEGATIVES_SINGLE
    return _join_parts(base)


def _character_fallback_phrases(brief: Dict[str, str], role: str, label: str, count: int) -> Dict[str, str]:
    """Neutral non-creative fallback only.

    Python must not invent fixed clothes, hair, colors, accessories, faces, or
    recurring example looks. The Local LLM is responsible for creative details.
    """
    style_phrase = _character_context_style_phrase(brief)
    if count <= 1:
        identity_anchor = _join_parts([
            "same main performer throughout the whole video",
            "consistent face, hairstyle, body type, and performance identity",
            f"following {style_phrase}",
        ])
        wardrobe_anchor = _join_parts([
            "consistent outfit style for this video",
            f"wardrobe follows {style_phrase}",
        ])
        marker = "the same main performer"
        prompt_phrase = _join_parts(["consistent identity", f"visual style follows {style_phrase}"])
        consistency = "keep the same face, hairstyle, body type, outfit style, and overall performance identity in every shot"
        negative = _join_parts(_GENERIC_IDENTITY_NEGATIVES_SINGLE)
    else:
        identity_anchor = _join_parts([
            f"same {label.lower()} throughout the whole video",
            "visually distinct from the other recurring members",
            "consistent face, hairstyle, body type, and role",
            f"appearance follows {style_phrase}",
        ])
        wardrobe_anchor = _join_parts([
            "consistent outfit style for this member",
            f"wardrobe follows {style_phrase}",
        ])
        marker = f"the same {label.lower()}"
        prompt_phrase = _join_parts(["visually distinct recurring member", f"style follows {style_phrase}"])
        consistency = "keep this member's same face, hairstyle, body type, outfit style, and role in every shot where they appear"
        negative = _join_parts(_GENERIC_IDENTITY_NEGATIVES_GROUP)
    return {
        "identity_anchor": identity_anchor,
        "wardrobe_anchor": wardrobe_anchor,
        "visual_marker_phrase": marker,
        "prompt_phrase": prompt_phrase,
        "consistency_prompt": consistency,
        "negative_prompt": negative,
    }


def _build_character_bibles(brief: Dict[str, str]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Minimal neutral fallback only; never a creative visual preset."""
    subject = _clean_text(brief.get("characters_subjects"), 220) or "main performer"
    count = max(1, min(12, _extract_subject_count(subject)))
    if not _subject_is_group(subject, count):
        count = 1
    chars: List[Dict[str, Any]] = []
    for idx in range(1, count + 1):
        role = _neutral_role_for_index(subject, idx, count)
        label = _neutral_label_for_role(role)
        fb = _character_fallback_phrases(brief, role, label, count)
        identity_prompt = _join_parts([fb["prompt_phrase"], fb["identity_anchor"], fb["wardrobe_anchor"]])
        chars.append({
            "id": f"char_{idx:02d}",
            "role": role,
            "label": label,
            "seed_subject": subject,
            "identity_anchor": fb["identity_anchor"],
            "wardrobe_anchor": fb["wardrobe_anchor"],
            "visual_marker_phrase": fb["visual_marker_phrase"],
            "prompt_phrase": fb["prompt_phrase"],
            "identity_prompt": identity_prompt,
            "consistency_prompt": fb["consistency_prompt"],
            "negative_prompt": fb["negative_prompt"],
        })
    groups: List[Dict[str, Any]] = []
    if count > 1:
        style_phrase = _character_context_style_phrase(brief)
        label = subject or f"Group of {count}"
        groups.append({
            "group_id": "group_01",
            "label": label,
            "member_ids": [c.get("id") for c in chars],
            "group_prompt": _join_parts([
                "same recurring group members throughout the whole video",
                "each member remains visually distinct and consistent",
                f"all outfits and appearance details follow {style_phrase}",
            ]),
            "group_negative_prompt": _join_parts(_GENERIC_IDENTITY_NEGATIVES_GROUP),
        })
    return chars, groups


def _character_compact_identity_phrase(ch: Dict[str, Any]) -> str:
    return _join_parts([
        ch.get("visual_marker_phrase"),
        ch.get("prompt_phrase"),
    ]) or _join_parts([ch.get("identity_anchor"), ch.get("wardrobe_anchor")]) or _safe_str(ch.get("identity_prompt"))


def _character_image_identity_phrase(ch: Dict[str, Any]) -> str:
    return _join_parts([
        ch.get("visual_marker_phrase"),
        ch.get("prompt_phrase"),
        ch.get("identity_anchor"),
        ch.get("wardrobe_anchor"),
        ch.get("consistency_prompt"),
    ]) or _safe_str(ch.get("identity_prompt"))


def _normalize_one_character(ch: Dict[str, Any], idx: int, fallback: Dict[str, Any], brief: Dict[str, str], total_count: int) -> Dict[str, Any]:
    out = dict(ch)
    out.setdefault("id", f"char_{idx:02d}")
    out.setdefault("role", fallback.get("role") or ("lead_performer" if idx == 1 else f"performer_{idx}"))
    out.setdefault("label", fallback.get("label") or _neutral_label_for_role(out.get("role")))
    fb = _character_fallback_phrases(brief, _safe_str(out.get("role")), _safe_str(out.get("label")), total_count)

    old_identity = _safe_str(out.get("identity_prompt"))
    out.setdefault("identity_anchor", old_identity or fb["identity_anchor"])
    out.setdefault("wardrobe_anchor", "")
    out.setdefault("visual_marker_phrase", _safe_str(out.get("prompt_phrase")) or _safe_str(out.get("label")) or fb["visual_marker_phrase"])
    out.setdefault("prompt_phrase", _join_parts([out.get("visual_marker_phrase"), out.get("identity_anchor"), out.get("wardrobe_anchor")]) or fb["prompt_phrase"])
    out.setdefault("consistency_prompt", fb["consistency_prompt"])
    out.setdefault("negative_prompt", fb["negative_prompt"])
    if not old_identity:
        out["identity_prompt"] = _join_parts([out.get("prompt_phrase"), out.get("identity_anchor"), out.get("wardrobe_anchor")])

    for key in ("identity_anchor", "wardrobe_anchor", "visual_marker_phrase", "prompt_phrase", "identity_prompt", "consistency_prompt"):
        out[key] = _sanitize_prompt_no_visible_text(out.get(key), image_prompt=False, max_len=1200 if key in {"identity_anchor", "identity_prompt"} else 700)
    out["negative_prompt"] = _negative_prompt_with_no_visible_text(out.get("negative_prompt"), max_len=900)
    return out


def _normalize_character_bibles(chars: Any, groups: Any, brief: Dict[str, str]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    out_chars = [dict(x) for x in _as_list(chars) if isinstance(x, dict)]
    out_groups = [dict(x) for x in _as_list(groups) if isinstance(x, dict)]
    if not out_chars:
        out_chars, out_groups = _build_character_bibles(brief)
    fallback_chars, fallback_groups = _build_character_bibles(brief)
    desired_count = len(fallback_chars)
    # If the LLM returned too few members for an explicit group count, add only
    # neutral extra slots. Do not invent visual looks in Python.
    while len(out_chars) < desired_count:
        out_chars.append(dict(fallback_chars[len(out_chars)]))
    if desired_count > 1 and len(out_chars) > desired_count:
        out_chars = out_chars[:desired_count]
    fallback_by_index = {i + 1: c for i, c in enumerate(fallback_chars)}
    total_count = max(len(out_chars), desired_count)
    out_chars = [
        _normalize_one_character(ch, idx, fallback_by_index.get(idx, fallback_chars[0] if fallback_chars else {}), brief, total_count)
        for idx, ch in enumerate(out_chars, start=1)
    ]
    valid_ids = {_safe_str(ch.get("id")) for ch in out_chars if _safe_str(ch.get("id"))}
    clean_groups: List[Dict[str, Any]] = []
    for idx, grp in enumerate(out_groups, start=1):
        if not isinstance(grp, dict):
            continue
        g = dict(grp)
        g.setdefault("group_id", f"group_{idx:02d}")
        g.setdefault("label", _clean_text(brief.get("characters_subjects"), 120) or f"Group {idx}")
        member_ids = [_safe_str(x) for x in _as_list(g.get("member_ids")) if _safe_str(x)]
        member_ids = [x for x in member_ids if x in valid_ids]
        if not member_ids and len(out_chars) > 1:
            member_ids = [_safe_str(ch.get("id")) for ch in out_chars if _safe_str(ch.get("id"))]
        g["member_ids"] = member_ids
        if not _safe_str(g.get("group_prompt")):
            g["group_prompt"] = (fallback_groups[0].get("group_prompt") if fallback_groups else "same recurring group members throughout the whole video, visually distinct and consistent")
        if not _safe_str(g.get("group_negative_prompt")):
            g["group_negative_prompt"] = _join_parts(_GENERIC_IDENTITY_NEGATIVES_GROUP)
        g["group_prompt"] = _sanitize_prompt_no_visible_text(g.get("group_prompt"), image_prompt=False, max_len=900)
        g["group_negative_prompt"] = _negative_prompt_with_no_visible_text(g.get("group_negative_prompt"), max_len=900)
        clean_groups.append(g)
    if len(out_chars) > 1 and not clean_groups:
        clean_groups = fallback_groups
    return out_chars, clean_groups


def _find_character_by_role(chars: List[Dict[str, Any]], role_prefixes: List[str]) -> Optional[Dict[str, Any]]:
    prefixes = [x.lower() for x in role_prefixes if x]
    for ch in chars:
        role = _safe_str(ch.get("role")).lower()
        label = _safe_str(ch.get("label")).lower()
        for pref in prefixes:
            if role == pref or role.startswith(pref) or pref in label:
                return ch
    return chars[0] if chars else None



def _has_strong_full_group_evidence(blob: str) -> bool:
    text = f" {_safe_str(blob).lower()} "
    patterns = [
        r"\bfull\s+group\b",
        r"\ball\s+(?:the\s+)?members\b",
        r"\ball\s+(?:the\s+)?performers\s+(?:visible|together|on\s+stage)\b",
        r"\bevery\s+member\b",
        r"\bentire\s+(?:band|group|crew|lineup)\b",
        r"\bwhole\s+(?:band|group|crew|lineup)\b",
        r"\bgroup\s+shot\b",
        r"\bwide\s+shot\s+of\s+(?:the\s+)?(?:band|group|all\s+members|all\s+performers)\b",
        r"\bfull\s+lineup\b",
        r"\bgroup\s+choreography\s+with\s+all\s+members\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _has_duo_evidence(blob: str) -> bool:
    text = f" {_safe_str(blob).lower()} "
    patterns = [
        r"\bduo\b", r"\bduet\b", r"\btwo\s+(?:singers|performers|members|vocalists)\b",
        r"\bdouble\s+vocal\b", r"\bshared\s+vocal\b", r"\bpair\s+of\s+(?:singers|performers|members)\b",
    ]
    return any(re.search(p, text) for p in patterns)


def _has_background_only_evidence(blob: str) -> bool:
    text = f" {_safe_str(blob).lower()} "
    return any(x in text for x in (
        " landscape", " city", " abstract", " environment", " b-roll", " broll",
        " empty room", " no performer", " no character", " crowd only", " location only",
    ))


def _clone_guard_negative_terms(cast_type: Any) -> str:
    cast = _safe_str(cast_type).lower()
    if cast == "full_group":
        terms = [
            "duplicated member", "cloned faces", "repeated same person", "missing member",
            "extra member", "merged faces", "same face on multiple people", "collage",
            "split-screen", "contact sheet", "reference sheet panels",
        ]
    else:
        terms = [
            "duplicate performer", "cloned person", "multiple copies of the same person",
            "repeated face", "duplicated face", "extra copies of the performer",
            "extra singer", "extra band member", "twin duplicate", "identity duplication",
            "collage", "split-screen", "contact sheet", "reference sheet panels",
        ]
    return ", ".join(_dedupe_texts(terms, max_items=20, max_len=1000)) if "_dedupe_texts" in globals() else ", ".join(dict.fromkeys(terms))


def _clone_guard_positive_prompt(cast_type: Any, selected_count: int, *, full_group_evidence: bool = False) -> str:
    cast = _safe_str(cast_type).lower()
    n = max(0, int(selected_count or 0))
    if cast == "full_group":
        if n <= 0:
            return "full-group shot only when explicitly requested; keep people visually distinct; no duplicate faces or cloned copies"
        return f"full-group shot with exactly {n} distinct recurring members; each member is a different person; no extra member, no missing member, no cloned faces, no repeated copies"
    if cast == "duo":
        return "duo shot with exactly 2 distinct recurring performers; no duplicate copies, no cloned faces, no extra band members"
    if cast in {"member_focus", "instrumental_member_focus"}:
        return "member-focus shot with exactly one selected performer as the active subject; no duplicate copies, no cloned faces, no extra band members"
    if cast == "background_only_presence":
        return "background/environment shot; do not add performers unless one is clearly visible"
    return "solo shot with exactly one lead performer as the active subject; no duplicate copies, no cloned faces, no extra singers or band members"


def _clone_guard_payload(cast_type: Any, selected_count: int, *, full_group_evidence: bool) -> Dict[str, Any]:
    cast = _safe_str(cast_type).lower() or "solo_lead"
    count = max(0, int(selected_count or 0))
    out: Dict[str, Any] = {
        "enabled": True,
        "selected_character_count": count,
        "full_group_evidence": bool(full_group_evidence),
        "all_members_injected": bool(cast == "full_group" and full_group_evidence and count > 1),
    }
    if cast == "full_group" and count > 0:
        out["exact_distinct_member_count"] = count
    return out

def _pick_character_members_for_shot(data: Dict[str, Any], brief: Dict[str, str], chars: List[Dict[str, Any]], groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not chars:
        return {
            "character_ids": [],
            "shot_cast_type": "background_only_presence",
            "clone_guard": _clone_guard_payload("background_only_presence", 0, full_group_evidence=False),
            "clone_guard_negative_prompt": _clone_guard_negative_terms("solo_lead"),
        }
    total = len(chars)
    needs_lipsync = _safe_bool(data.get("needs_lipsync"), False)
    blob = " ".join([
        _clean_text(data.get("scene_role")), _clean_text(data.get("scene_role_summary")),
        _clean_text(data.get("section")), _clean_text(data.get("section_summary")),
        _clean_text(data.get("energy")), _clean_text(data.get("energy_summary")),
        _clean_text(data.get("suggested_shot_type")), _clean_text(data.get("vocal_presence_summary")),
        _clean_text(data.get("shot_cast_type")), _clean_text(data.get("director_notes")),
        _clean_text(data.get("image_prompt")), _clean_text(data.get("director_image_prompt")),
        _clean_text(data.get("video_prompt")), _clean_text(data.get("director_video_prompt")),
        _clean_text(data.get("lyrics")), _clean_text(brief.get("characters_subjects")),
    ]).lower()
    idx = max(1, _safe_int(data.get("index"), 1))
    lead = _find_character_by_role(chars, ["lead_singer", "lead_vocalist", "lead_rapper", "lead_performer", "vocalist", "singer"]) or chars[0]
    full_group_evidence = _has_strong_full_group_evidence(blob)
    duo_evidence = _has_duo_evidence(blob)
    instrumental_focus = any(x in blob for x in ("instrumental", "guitar", "bass", "drum", "drummer", "bassist", "guitarist", "keyboard", "solo"))
    background_only = _has_background_only_evidence(blob)

    if total == 1:
        selected = [chars[0]] if not background_only else []
        cast_type = "solo_lead" if selected else "background_only_presence"
    elif full_group_evidence:
        selected = list(chars[:5])
        cast_type = "full_group"
    elif duo_evidence:
        second_pool = [c for c in chars if _safe_str(c.get("id")) != _safe_str(lead.get("id"))]
        selected = [lead] + second_pool[:1]
        selected = selected[:2]
        cast_type = "duo" if len(selected) > 1 else "solo_lead"
    elif needs_lipsync:
        # Chorus/hook/performance/dance are not enough for all members. Default to the lead.
        selected = [lead]
        cast_type = "solo_lead"
    elif instrumental_focus:
        players = [c for c in chars if _safe_str(c.get("id")) != _safe_str(lead.get("id"))] or chars
        selected = [players[(idx - 1) % len(players)]]
        cast_type = "member_focus"
    elif background_only:
        selected = []
        cast_type = "background_only_presence"
    else:
        selected = [lead]
        cast_type = "solo_lead"

    selected_ids = [_safe_str(c.get("id")) for c in selected if _safe_str(c.get("id"))]
    labels = [_safe_str(c.get("label")) for c in selected if _safe_str(c.get("label"))]
    compact_identity = "; ".join(_dedupe_texts([_character_compact_identity_phrase(c) for c in selected], max_items=max(1, len(selected)), max_len=850))
    image_identity = "; ".join(_dedupe_texts([_character_image_identity_phrase(c) for c in selected], max_items=max(1, len(selected)), max_len=1300))
    wardrobe = "; ".join(_dedupe_texts([c.get("wardrobe_anchor") for c in selected], max_items=max(1, len(selected)), max_len=700))
    consistency = "; ".join(_dedupe_texts([c.get("consistency_prompt") for c in selected], max_items=4, max_len=600))
    char_negative = ", ".join(_dedupe_texts([c.get("negative_prompt") for c in selected], max_items=max(1, len(selected)), max_len=800))
    guard_positive = _clone_guard_positive_prompt(cast_type, len(selected_ids), full_group_evidence=full_group_evidence)
    guard_negative = _clone_guard_negative_terms(cast_type)
    group_prompt = ""
    group_negative = ""
    group_ids: List[str] = []
    if groups and cast_type == "full_group":
        grp = groups[0]
        group_ids = [_safe_str(grp.get("group_id"))] if _safe_str(grp.get("group_id")) else []
        group_prompt = _clean_text(grp.get("group_prompt"), 700)
        group_negative = _clean_text(grp.get("group_negative_prompt"), 700)
    return {
        "character_ids": selected_ids,
        "character_labels": labels,
        "group_ids": group_ids,
        "shot_cast_type": cast_type,
        "character_identity_prompt": compact_identity,
        "character_image_identity_prompt": image_identity,
        "character_wardrobe_prompt": wardrobe,
        "group_identity_prompt": group_prompt,
        "character_consistency_prompt": consistency,
        "character_clone_guard_prompt": guard_positive,
        "character_negative_prompt": _join_parts([char_negative, group_negative, guard_negative]),
        "clone_guard_negative_prompt": guard_negative,
        "clone_guard": _clone_guard_payload(cast_type, len(selected_ids), full_group_evidence=full_group_evidence),
        "character_consistency_applied": bool(selected_ids or group_prompt),
    }


def _inject_identity_into_timestamp_prompt(prompt: Any, identity_bits: List[str]) -> str:
    text = _clean_text(prompt, 2400)
    bits = [_clean_text(x, 500) for x in identity_bits if _clean_text(x)]
    identity_text = ". ".join(bits).strip(" .")
    if not text:
        return identity_text + "." if identity_text else ""
    if not identity_text:
        return text
    if identity_text.lower() in text.lower()[:900]:
        return text
    if re.match(r"^0:00\s*-", text):
        return re.sub(r"^0:00\s*-\s*", f"0:00 - {identity_text}. ", text, count=1)
    return f"0:00 - {identity_text}. {text}"


def _apply_character_identity_to_item(data: Dict[str, Any], brief: Dict[str, str], chars: List[Dict[str, Any]], groups: List[Dict[str, Any]], *, include_director_fields: bool = False) -> Dict[str, Any]:
    out = dict(data)
    chars, groups = _normalize_character_bibles(chars, groups, brief)
    assign = _pick_character_members_for_shot(out, brief, chars, groups)
    out.update(assign)
    image_bits = [assign.get("group_identity_prompt"), assign.get("character_image_identity_prompt"), assign.get("character_consistency_prompt"), assign.get("character_clone_guard_prompt")]
    video_bits = [assign.get("group_identity_prompt"), assign.get("character_identity_prompt")]
    timestamp_bits = [assign.get("group_identity_prompt"), assign.get("character_identity_prompt")]
    if _safe_str(out.get("image_prompt")):
        out["image_prompt"] = _sanitize_prompt_no_visible_text(_join_parts(image_bits + [out.get("image_prompt")]), out.get("lyrics"), image_prompt=True, max_len=2400)
    if _safe_str(out.get("video_prompt")):
        out["video_prompt"] = _sanitize_prompt_no_visible_text(_join_parts(video_bits + [out.get("video_prompt")]), out.get("lyrics"), image_prompt=False, max_len=2200)
    if _safe_str(out.get("timestamped_video_prompt")):
        out["timestamped_video_prompt"] = _sanitize_ltx_timestamped_prompt(_normalize_director_timestamp(_inject_identity_into_timestamp_prompt(out.get("timestamped_video_prompt"), timestamp_bits), duration=_safe_float(out.get("duration"), 0.0), needs_lipsync=_safe_bool(out.get("needs_lipsync"), False), fallback=_safe_str(out.get("timestamped_video_prompt"))), _safe_float(out.get("duration"), 0.0))
    if _safe_str(out.get("negative_prompt")) or _safe_str(assign.get("character_negative_prompt")):
        out["negative_prompt"] = _negative_prompt_with_no_visible_text(_join_parts([out.get("negative_prompt"), assign.get("character_negative_prompt")]), max_len=1400)
    if include_director_fields:
        if _safe_str(out.get("director_image_prompt")):
            out["director_image_prompt"] = _sanitize_prompt_no_visible_text(_join_parts(image_bits + [out.get("director_image_prompt")]), out.get("lyrics"), image_prompt=True, max_len=2400)
        if _safe_str(out.get("director_video_prompt")):
            out["director_video_prompt"] = _sanitize_prompt_no_visible_text(_join_parts(video_bits + [out.get("director_video_prompt")]), out.get("lyrics"), image_prompt=False, max_len=2200)
        if _safe_str(out.get("director_timestamped_video_prompt")):
            out["director_timestamped_video_prompt"] = _sanitize_ltx_timestamped_prompt(_normalize_director_timestamp(_inject_identity_into_timestamp_prompt(out.get("director_timestamped_video_prompt"), timestamp_bits), duration=_safe_float(out.get("duration"), 0.0), needs_lipsync=_safe_bool(out.get("needs_lipsync"), False), fallback=_safe_str(out.get("director_timestamped_video_prompt"))), _safe_float(out.get("duration"), 0.0))
        if _safe_str(out.get("director_negative_prompt")) or _safe_str(assign.get("character_negative_prompt")):
            out["director_negative_prompt"] = _negative_prompt_with_no_visible_text(_join_parts([out.get("director_negative_prompt"), assign.get("character_negative_prompt")]), max_len=1400)
        if _safe_str(out.get("template_image_prompt")):
            out["template_image_prompt"] = _sanitize_prompt_no_visible_text(_join_parts(image_bits + [out.get("template_image_prompt")]), out.get("lyrics"), image_prompt=True, max_len=2400)
        if _safe_str(out.get("template_video_prompt")):
            out["template_video_prompt"] = _sanitize_prompt_no_visible_text(_join_parts(video_bits + [out.get("template_video_prompt")]), out.get("lyrics"), image_prompt=False, max_len=2200)
        if _safe_str(out.get("template_timestamped_video_prompt")):
            out["template_timestamped_video_prompt"] = _sanitize_ltx_timestamped_prompt(_normalize_director_timestamp(_inject_identity_into_timestamp_prompt(out.get("template_timestamped_video_prompt"), timestamp_bits), duration=_safe_float(out.get("duration"), 0.0), needs_lipsync=_safe_bool(out.get("needs_lipsync"), False), fallback=_safe_str(out.get("template_timestamped_video_prompt"))), _safe_float(out.get("duration"), 0.0))
    return out


def _character_bible_sidecar_path(plan_path_value: Any) -> Path:
    try:
        raw = _safe_str(plan_path_value)
        if raw:
            plan_path = Path(raw).expanduser().resolve()
            if plan_path.is_dir():
                return (plan_path / "musicclip_character_bible.json").resolve()
            if plan_path.is_file() or plan_path.suffix:
                return (plan_path.parent / "musicclip_character_bible.json").resolve()
    except Exception:
        pass
    return (_project_root() / "output" / "musicclip_character_bible.json").resolve()


def _save_character_bible_sidecar(path: Path, pack: Dict[str, Any], source_plan_path: Any = "") -> None:
    try:
        data = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_plan_path": _safe_str(source_plan_path),
            "character_bible": pack.get("character_bible") or [],
            "group_bibles": pack.get("group_bibles") or [],
            "shot_assignment_guidance": _safe_str(pack.get("shot_assignment_guidance")),
            "character_bible_backend": _safe_str(pack.get("character_bible_backend")),
            "character_bible_warnings": _as_list(pack.get("character_bible_warnings")),
        }
        _write_json_file(path, data)
    except Exception:
        pass


def _load_character_bible_sidecar(path: Path, brief: Dict[str, str]) -> Dict[str, Any]:
    try:
        if path.is_file():
            data = _read_json_file(str(path))
            chars, groups = _normalize_character_bibles(data.get("character_bible"), data.get("group_bibles"), brief)
            return {
                "character_bible": chars,
                "group_bibles": groups,
                "shot_assignment_guidance": _safe_str(data.get("shot_assignment_guidance")),
                "character_bible_backend": _safe_str(data.get("character_bible_backend") or "sidecar_reuse"),
                "character_bible_warnings": _as_list(data.get("character_bible_warnings")),
            }
    except Exception:
        pass
    return {}


def _identity_context_from_plan(shot: Dict[str, Any], plan: Dict[str, Any], payload: Optional[Dict[str, Any]] = None, progress_callback: Optional[Callable[[str], None]] = None) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    brief = _normalize_creative_brief(plan.get("creative_brief"))
    payload = payload if isinstance(payload, dict) else {}
    existing_chars = plan.get("character_bible")
    existing_groups = plan.get("group_bibles")
    pack: Dict[str, Any] = {}
    if existing_chars:
        chars, groups = _normalize_character_bibles(existing_chars, existing_groups, brief)
    else:
        plan_path_value = payload.get("ltx_director_plan_path") or payload.get("source_ltx_shot_plan_path") or payload.get("prompt_plan_path") or plan.get("source_ltx_shot_plan_path") or plan.get("prompt_plan_path")
        sidecar = _character_bible_sidecar_path(plan_path_value)
        pack = _load_character_bible_sidecar(sidecar, brief)
        if not pack and _character_backend_selected(payload, plan):
            pack = _create_llm_character_bible_if_selected(
                payload=payload,
                brief=brief,
                scenes=plan.get("shots") or [shot],
                source_plan=plan,
                progress_callback=progress_callback,
            )
            _save_character_bible_sidecar(sidecar, pack, plan_path_value)
        if not pack:
            chars, groups = _normalize_character_bibles(None, None, brief)
        else:
            chars, groups = _normalize_character_bibles(pack.get("character_bible"), pack.get("group_bibles"), brief)
    effective = _apply_character_identity_to_item(shot, brief, chars, groups, include_director_fields=True)
    return chars, groups, effective


def _character_backend_selected(*sources: Any) -> bool:
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in ("director_backend", "director_brain", "character_bible_backend", "prompt_backend_mode"):
            value = src.get(key)
            if value is not None and _normalize_director_backend(value) == _DIRECTOR_BACKEND_LOCAL_LLM:
                return True
        for key in ("use_local_llm_character_bible", "local_llm_character_bible", "own_llama_enabled"):
            if key in src and _safe_bool(src.get(key), False):
                return True
    return False


def _scene_summary_for_character_bible(scenes: Any, limit: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in _as_list(scenes)[:limit]:
        if not isinstance(item, dict):
            continue
        out.append({
            "id": _safe_str(item.get("id")),
            "start": _safe_float(item.get("song_start", item.get("start", 0.0)), 0.0),
            "end": _safe_float(item.get("song_end", item.get("end", 0.0)), 0.0),
            "section": _clean_text(item.get("section") or item.get("section_summary"), 80),
            "scene_role": _clean_text(item.get("scene_role") or item.get("scene_role_summary"), 120),
            "needs_lipsync": _safe_bool(item.get("needs_lipsync"), False),
            "lyrics": _clean_text(item.get("lyrics") or item.get("lyric_text"), 180),
        })
    return out


def _lyrics_summary_for_character_bible(*sources: Any) -> str:
    texts: List[str] = []
    for src in sources:
        if isinstance(src, dict):
            for key in ("lyrics_srt_segments", "lyric_segments"):
                for item in _as_list(src.get(key))[:12]:
                    if isinstance(item, dict):
                        texts.append(item.get("text") or item.get("lyrics") or item.get("lyric_text"))
            for item in _as_list(src.get("scenes") or src.get("shots"))[:12]:
                if isinstance(item, dict):
                    texts.append(item.get("lyrics") or item.get("lyric_text"))
    return " | ".join(_dedupe_texts(texts, max_items=8, max_len=700))


def _llm_character_bible_system_prompt() -> str:
    return (
        "You create detailed, stable Character Bibles for music-video image/video generation. "
        "Return strict JSON only, no markdown, no commentary. "
        "The Python bridge will not invent creative appearance details; you are responsible for choosing stable visual identities when the user is vague. "
        "If the user provides character, wardrobe, style, era, species, or world details, preserve and expand them. "
        "If the subject is vague, invent tasteful stable visual details that fit the main idea, style/theme, location/world, lyrics mood, and music-video genre. "
        "Avoid repeating one default look across unrelated projects. Do not copy any fixed preset look. "
        "Separate identity_anchor from wardrobe_anchor. Use visual_marker_phrase and prompt_phrase that image/video models can understand without relying on names. "
        "Never include instructions to draw visible lyric text, subtitles, captions, logos, signs, typography, or written words."
    )


def _llm_character_bible_user_prompt(brief: Dict[str, str], scenes: Any, lyric_summary: str) -> str:
    subject = _clean_text(brief.get("characters_subjects"), 220) or "main performer"
    count = _extract_subject_count(subject)
    is_group = _subject_is_group(subject, count)
    payload = {
        "task": "Create a detailed reusable Character Bible for this music video. Return strict JSON object only.",
        "creative_brief": {
            "characters_subjects": _clean_text(brief.get("characters_subjects"), 300),
            "main_idea": _clean_text(brief.get("main_idea"), 400),
            "style_theme": _clean_text(brief.get("style_theme"), 400),
            "locations_world": _clean_text(brief.get("locations_world"), 400),
            "camera_choreography": _clean_text(brief.get("camera_choreography"), 300),
        },
        "derived_subject_info": {
            "approx_subject_count": count,
            "subject_seems_group": is_group,
            "contains_vocal_or_lipsync_moments": any(_safe_bool(s.get("needs_lipsync"), False) for s in _as_list(scenes) if isinstance(s, dict)),
        },
        "lyrics_summary": _clean_text(lyric_summary, 900),
        "scene_summary": _scene_summary_for_character_bible(scenes, limit=12),
        "rules": [
            "Use the user's actual subject/style/world as the source of appearance details.",
            "If clothing/style is described by the user, preserve and expand that clothing/style.",
            "If the subject is vague, you may invent specific stable identity and wardrobe details that fit the context.",
            "Create enough visual detail for image consistency, but do not use a repeated default look across projects.",
            "For single-person subjects, usually create one character entry.",
            "For explicit group counts, create exactly that many member entries plus one group bible.",
            "For group subjects, make members visually distinct using details you infer from the user context, not preset examples.",
            "Use visual markers, not names alone. prompt_phrase should be compact and model-friendly.",
            "identity_anchor should describe stable face/body/hair/species/recognizable visual identity.",
            "wardrobe_anchor should describe the stable outfit or style layer for this video.",
            "visual_marker_phrase should be short and reusable in prompts.",
            "negative prompts should focus on identity drift, duplicates, extra/missing members, and consistency problems.",
        ],
        "expected_json_schema": {
            "character_bible": [
                {
                    "id": "char_01",
                    "role": "lead_singer",
                    "label": "Lead singer",
                    "identity_anchor": "stable face/body/hair visual identity only",
                    "wardrobe_anchor": "stable outfit/style layer for this video",
                    "visual_marker_phrase": "short visual marker phrase, no name required",
                    "prompt_phrase": "compact reusable phrase for prompts",
                    "consistency_prompt": "what must stay identical across shots",
                    "negative_prompt": "identity drift negatives",
                }
            ],
            "group_bibles": [
                {
                    "group_id": "group_01",
                    "label": "group label",
                    "member_ids": ["char_01", "char_02"],
                    "group_prompt": "stable group identity prompt",
                    "group_negative_prompt": "group identity drift negatives",
                }
            ],
            "shot_assignment_guidance": "brief guidance for solo/group/member focus decisions",
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _validate_character_bible_payload(obj: Any, brief: Dict[str, str]) -> Dict[str, Any]:
    data = obj if isinstance(obj, dict) else {}
    chars_raw = data.get("character_bible") if isinstance(data.get("character_bible"), list) else []
    groups_raw = data.get("group_bibles") if isinstance(data.get("group_bibles"), list) else []
    chars, groups = _normalize_character_bibles(chars_raw, groups_raw, brief)
    if not chars or not all((_safe_str(c.get("identity_anchor")) or _safe_str(c.get("identity_prompt"))) and (_safe_str(c.get("visual_marker_phrase")) or _safe_str(c.get("prompt_phrase"))) for c in chars):
        chars, groups = _build_character_bibles(brief)
    valid_ids = {_safe_str(c.get("id")) for c in chars if _safe_str(c.get("id"))}
    for grp in groups:
        grp["member_ids"] = [mid for mid in [_safe_str(x) for x in _as_list(grp.get("member_ids"))] if mid in valid_ids]
        if not grp.get("member_ids") and len(chars) > 1:
            grp["member_ids"] = [_safe_str(c.get("id")) for c in chars if _safe_str(c.get("id"))]
    return {
        "character_bible": chars,
        "group_bibles": groups,
        "shot_assignment_guidance": _clean_text(data.get("shot_assignment_guidance"), 900),
    }


def _create_llm_character_bible_if_selected(*, payload: Dict[str, Any], brief: Dict[str, str], scenes: Any, source_plan: Optional[Dict[str, Any]] = None, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    source_plan = source_plan if isinstance(source_plan, dict) else {}
    warnings: List[str] = []
    use_llm = _character_backend_selected(payload, source_plan)
    if not use_llm:
        chars, groups = _normalize_character_bibles(source_plan.get("character_bible") or payload.get("character_bible"), source_plan.get("group_bibles") or payload.get("group_bibles"), brief)
        return {
            "character_bible": chars,
            "group_bibles": groups,
            "shot_assignment_guidance": _clean_text(source_plan.get("shot_assignment_guidance") or payload.get("shot_assignment_guidance")),
            "character_bible_backend": "neutral_fallback",
            "character_bible_warnings": warnings,
        }
    try:
        if callable(progress_callback):
            progress_callback("Creating Local LLM Character Bible...")
        cfg = _read_planner_llama_settings_for_bridge(payload.get("root_dir") or _project_root(), payload.get("llama_settings") if isinstance(payload.get("llama_settings"), dict) else None)
        if not _safe_bool(cfg.get("enabled"), False):
            raise RuntimeError("Local LLM is selected but own_llama_enabled is off in planner_settings.json")
        client = _DirectorLlamaClient(cfg, log_path=(_project_root() / "logs" / "musicclip_character_bible_llama.log"))
        try:
            raw = client.generate(
                _llm_character_bible_system_prompt(),
                _llm_character_bible_user_prompt(brief, scenes, _lyrics_summary_for_character_bible(source_plan, {"scenes": scenes})),
                temperature=0.72,
                max_tokens=3600,
            )
            parsed = _extract_json_payload_from_text(raw)
            validated = _validate_character_bible_payload(parsed, brief)
            validated["character_bible_backend"] = "local_llm_planner_style"
            validated["character_bible_warnings"] = warnings
            return validated
        finally:
            try:
                client.stop()
            except Exception:
                pass
    except Exception as exc:
        warnings.append(f"Local LLM Character Bible unavailable; neutral fallback used ({exc}).")
        chars, groups = _build_character_bibles(brief)
        return {
            "character_bible": chars,
            "group_bibles": groups,
            "shot_assignment_guidance": "Neutral fallback: keep identities consistent and follow the user-provided subject/style; no hardcoded visual presets were added.",
            "character_bible_backend": "neutral_fallback_after_llm_error",
            "character_bible_warnings": warnings,
        }



# Strong negative terms used anywhere prompts may become image-generation prompts.
# Keep lyrics as mood/performance context only; visible words belong in FrameVision overlays later.
_NO_VISIBLE_TEXT_NEGATIVE_TERMS = [
    "subtitles",
    "captions",
    "on-screen text",
    "lyric text",
    "typography",
    "letters",
    "words",
    "watermark",
    "logo",
]

_TEXT_IN_IMAGE_SAFE_PHRASE = "expressive mood and performance energy matching the sung vocal phrase, without showing any written words"


def _append_unique_prompt_terms(base: Any, terms: List[str], max_len: int = 0) -> str:
    pieces: List[str] = []
    seen = set()
    raw = _clean_text(base)
    for chunk in re.split(r"[,;]+", raw):
        item = _clean_text(chunk)
        if item:
            key = item.lower()
            if key not in seen:
                seen.add(key)
                pieces.append(item)
    for term in terms:
        item = _clean_text(term)
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            pieces.append(item)
    out = ", ".join(pieces)
    if max_len and len(out) > max_len:
        out = out[: max(0, max_len - 3)].rstrip(" ,.;") + "..."
    return out


def _negative_prompt_with_no_visible_text(base: Any, max_len: int = 1200) -> str:
    return _append_unique_prompt_terms(base, _NO_VISIBLE_TEXT_NEGATIVE_TERMS, max_len=max_len)


def _sanitize_prompt_no_visible_text(value: Any, lyrics: Any = "", *, max_len: int = 1800, image_prompt: bool = False) -> str:
    """Remove wording that asks image/video models to draw lyric words or subtitles.

    This keeps lyric meaning as performance energy only. It intentionally does not
    remove lyric metadata stored elsewhere in JSON.
    """
    text = _clean_text(value, max_len=0)
    if not text:
        return ""

    safe = _TEXT_IN_IMAGE_SAFE_PHRASE if image_prompt else "performance energy follows the sung vocal phrase, with no subtitles or on-screen words"

    # Replace the old explicit lyric-quote phrasing before generic quote cleanup.
    text = re.sub(
        r"\bvisual\s+mood\s+inspired\s+by\s+the\s+lyric\s*['\"][^'\"]{1,500}['\"]",
        safe,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:mood|energy|visual\s+mood|performance\s+energy)\s+(?:inspired\s+by|matching|based\s+on)\s+(?:the\s+)?(?:lyric|lyrics|vocal\s+phrase)\s*['\"][^'\"]{1,500}['\"]",
        safe,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(?:inspired\s+by|matching|based\s+on)\s+(?:the\s+)?(?:lyric|lyrics)\s*['\"][^'\"]{1,500}['\"]",
        safe,
        text,
        flags=re.IGNORECASE,
    )

    # If the exact lyric text leaked into the prompt, replace it without touching the JSON metadata.
    lyric_text = _clean_text(lyrics, 0)
    if lyric_text and len(lyric_text) >= 3:
        variants = {lyric_text, lyric_text.replace("'", "’"), lyric_text.replace("’", "'")}
        for variant in sorted(variants, key=len, reverse=True):
            if not variant:
                continue
            text = re.sub(re.escape(variant), "the sung vocal phrase", text, flags=re.IGNORECASE)

    # Remove direct visible-text instructions. These are bad for generated images and LTX motion.
    replacements = [
        (r"\b(?:subtitles?|captions?)\s+(?:appear|appears|show|shows|display|displays|on\s+screen)[^,.;]*", "clean performance motion without on-screen text"),
        (r"\b(?:lyrics?|lyric\s+text|words?)\s+(?:appear|appears|show|shows|display|displays|on\s+screen)[^,.;]*", "performance energy follows the sung vocal phrase without on-screen text"),
        (r"\b(?:text|sign|screen|caption|subtitle)\s+reads\s*['\"][^'\"]{1,300}['\"]", "clean image-only performance framing"),
        (r"\b(?:text|sign|screen|caption|subtitle)\s+reads\b[^,.;]*", "clean image-only performance framing"),
        (r"\bwords?\s+on\s+screen\b", "performance stays image-only without written overlays"),
        (r"\blyrics?\s+on\s+screen\b", "performance follows the lyric mood without written overlays"),
        (r"\blyrics?\s+displayed\b", "performance follows the lyric mood without written overlays"),
        (r"\bvisible\s+(?:lyric\s+)?(?:text|words|letters|typography|subtitles?|captions?)\b", "clean image-only performance framing"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    text = re.sub(r"\bno\s+no\s+visible\b", "no visible", text, flags=re.IGNORECASE)
    text = re.sub(r"\bno\s+no\s+subtitles\b", "no subtitles", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    if max_len and len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip(" ,.;") + "..."
    return text


def _prompt_has_visible_text_request(value: Any, lyrics: Any = "") -> bool:
    text = _clean_text(value).lower()
    if not text:
        return False
    lyric_text = _clean_text(lyrics).lower()
    if lyric_text and len(lyric_text) >= 3 and lyric_text in text:
        return True
    # Ignore safe negative wording; flag only prompts that request or preserve visible text.
    safe_text = re.sub(r"\b(?:no|without|avoid|exclude)\s+(?:visible\s+)?(?:written\s+)?(?:text|words|letters|typography|subtitles?|captions?|lyric\s+text)\b", " ", text, flags=re.IGNORECASE)
    safe_text = re.sub(r"\bwithout\s+showing\s+any\s+written\s+words\b", " ", safe_text, flags=re.IGNORECASE)
    bad_patterns = [
        r"visual\s+mood\s+inspired\s+by\s+the\s+lyric\s*['\"]",
        r"text\s+reads\b",
        r"captions?\s+(?:appear|appears|show|shows|display|displays|on\s+screen)",
        r"subtitles?\s+(?:appear|appears|show|shows|display|displays|on\s+screen)",
        r"lyrics?\s+(?:on\s+screen|displayed|appear|appears|show|shows)",
        r"words?\s+on\s+screen",
        r"visible\s+(?:text|words|letters|typography)",
    ]
    return any(re.search(pat, safe_text, flags=re.IGNORECASE) for pat in bad_patterns)


def _brief_value(brief: Dict[str, str], key: str, fallback: str) -> str:
    return _clean_text(brief.get(key)) or fallback


def _section_kind(scene: Dict[str, Any]) -> str:
    raw = _clean_text(scene.get("section") or scene.get("section_label") or "verse").lower()
    if "chorus" in raw:
        return "chorus"
    if "drop" in raw:
        return "drop"
    if "break" in raw or "bridge" in raw:
        return "break"
    if "intro" in raw:
        return "intro"
    if "outro" in raw or "ending" in raw:
        return "outro"
    if "verse" in raw:
        return "verse"
    return raw or "verse"


def _energy_kind(scene: Dict[str, Any]) -> str:
    raw = _clean_text(scene.get("energy") or scene.get("energy_class") or "mid").lower()
    if any(x in raw for x in ("high", "big", "peak", "intense", "drop", "strong")):
        return "high"
    if any(x in raw for x in ("low", "soft", "calm", "quiet", "slow")):
        return "low"
    return "mid"


def _mood_for_scene(section: str, energy: str, style: str) -> str:
    if section in {"chorus", "drop"} or energy == "high":
        return _join_parts(["bigger chorus energy", "strong movement", "bright visual impact", style])
    if section == "break" or energy == "low":
        return _join_parts(["dreamy atmosphere", "slower emotional movement", "soft surreal tension", style])
    if section == "intro":
        return _join_parts(["establishing mood", "anticipation before the beat opens", style])
    if section == "outro":
        return _join_parts(["resolved closing mood", "final reflective energy", style])
    return _join_parts(["story-focused performance energy", "clear musical rhythm", style])


def _camera_for_scene(section: str, energy: str, camera_hint: str) -> str:
    hint = _clean_text(camera_hint)
    if hint:
        return hint
    if section in {"chorus", "drop"} or energy == "high":
        return "dynamic push-in with a wider reveal on the beat"
    if section == "break" or energy == "low":
        return "slow cinematic drift with gentle parallax"
    if section == "intro":
        return "slow establishing camera move"
    if section == "outro":
        return "smooth pull-back that resolves the scene"
    return "steady music-video camera move following the main action"


def _lyric_hint_text(lyrics: str) -> str:
    text = _clean_text(lyrics, 170)
    if not text:
        return ""
    return "performance energy inspired by the sung vocal phrase, without showing written words"


def _lyric_key(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _annotate_lyric_blocks(scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add contiguous lyric-block metadata so long SRT chunks can vary scene by scene.

    Important: lyric text is metadata/context. It is not proof that every scene
    inside a long unchanged SRT block has active words to mouth.
    """
    out: List[Dict[str, Any]] = [dict(s) for s in scenes if isinstance(s, dict)]
    block_no = 0
    i = 0
    while i < len(out):
        key = _lyric_key(out[i].get("lyrics") or out[i].get("lyric_text"))
        if not key:
            out[i]["lyric_block_id"] = ""
            out[i]["lyric_block_scene_index"] = 0
            out[i]["lyric_block_scene_count"] = 0
            out[i]["lyric_block_start"] = 0.0
            out[i]["lyric_block_end"] = 0.0
            out[i]["lyric_block_duration"] = 0.0
            out[i]["active_vocal_window"] = False
            i += 1
            continue
        j = i + 1
        while j < len(out) and _lyric_key(out[j].get("lyrics") or out[j].get("lyric_text")) == key:
            j += 1
        block_no += 1
        count = j - i
        block_id = f"L{block_no:02d}"

        block_scenes = out[i:j]
        scene_start = min(_safe_float(s.get("start", s.get("song_start", 0.0)), 0.0) for s in block_scenes)
        scene_end = max(_safe_float(s.get("end", s.get("song_end", 0.0)), 0.0) for s in block_scenes)
        lyric_starts = [_safe_float(s.get("lyric_start"), 0.0) for s in block_scenes if "lyric_start" in s]
        lyric_ends = [_safe_float(s.get("lyric_end"), 0.0) for s in block_scenes if "lyric_end" in s]
        block_start = min(lyric_starts) if lyric_starts else scene_start
        block_end = max(lyric_ends) if lyric_ends else scene_end
        if block_end <= block_start:
            block_start, block_end = scene_start, scene_end
        block_duration = max(0.0, block_end - block_start)

        for off in range(count):
            sc = out[i + off]
            sc_start = _safe_float(sc.get("start", sc.get("song_start", 0.0)), 0.0)
            sc["lyric_block_id"] = block_id
            sc["lyric_block_scene_index"] = off + 1
            sc["lyric_block_scene_count"] = count
            sc["lyric_block_start"] = round(block_start, 3)
            sc["lyric_block_end"] = round(block_end, 3)
            sc["lyric_block_duration"] = round(block_duration, 3)
            sc["lyric_block_text_key"] = key
            # Active vocal is only near the beginning of a long/repeated lyric block.
            # Later scenes keep lyric metadata, but should not be asked to mouth words.
            offset_from_block = max(0.0, sc_start - block_start)
            repeated_ok = (off + 1) <= int(MAX_LIPSYNC_SCENES_PER_REPEATED_LYRIC)
            long_ok = (block_duration <= float(LONG_LYRIC_SEGMENT_SECONDS)) or (offset_from_block <= float(MAX_LIPSYNC_SECONDS_PER_LONG_SEGMENT))
            sc["active_vocal_window"] = bool(repeated_ok and long_ok)
            sc["vocal_timing_reason"] = (
                "active_early_lyric_window"
                if sc["active_vocal_window"]
                else "lyric_metadata_only_long_or_repeated_block"
            )
        i = j
    return out

def _brief_blob(brief: Dict[str, str]) -> str:
    return " ".join(_clean_text(v) for v in (brief or {}).values()).lower()


def _brief_has_vehicle_world(brief: Dict[str, str]) -> bool:
    blob = _brief_blob(brief)
    return any(x in blob for x in ("car", "cars", "supercar", "vehicle", "truck", "motorcycle", "cockpit", "highway", "road", "racing", "race", "driver"))


def _brief_has_dance_world(brief: Dict[str, str]) -> bool:
    blob = _brief_blob(brief)
    return any(x in blob for x in ("dance", "dancer", "club", "stage", "performance", "crowd", "disco", "party", "rave"))


def _lead_subject(brief: Dict[str, str]) -> str:
    explicit = _brief_value(brief, "characters_subjects", "")
    if explicit:
        return explicit
    try:
        inferred = _director_infer_subject_from_main_idea(brief)
        if inferred and _director_norm_for_compare(inferred) != _director_norm_for_compare(_DIRECTOR_GENERIC_SUBJECT):
            return inferred
    except Exception:
        pass
    return "the intended visible subject"


def _visual_world(brief: Dict[str, str]) -> str:
    return _brief_value(brief, "locations_world", "the same coherent music-video world")


def _style_text(brief: Dict[str, str]) -> str:
    return _brief_value(brief, "style_theme", "high quality cinematic music video style")



# -----------------------------
# Montage / collage control helpers
# -----------------------------
_COLLAGE_WORD_RE = re.compile(
    r"\b(?:collage|split[-\s]?screen|multi[-\s]?panel|contact[-\s]?sheet|moodboard|"
    r"picture[-\s]?in[-\s]?picture|grid(?:\s+of\s+scenes)?|poster\s+layout|"
    r"multi[-\s]?location\s+composition|multiple\s+locations\s+in\s+one\s+frame|"
    r"several\s+locations\s+at\s+once|locations\s+together|montage\s+layout)\b",
    flags=re.IGNORECASE,
)

_MONTAGE_ROLE_WORD_RE = re.compile(
    r"\b(?:montage_burst|split_screen_impact|transition_montage|microclip_montage|"
    r"chorus_impact_montage|montage burst|split screen impact|transition montage|"
    r"microclip montage|impact montage)\b",
    flags=re.IGNORECASE,
)

_ANTI_COLLAGE_NEGATIVE_BASE = (
    "no collage, no split screen, no multi-panel, no contact sheet, no picture-in-picture, "
    "no moodboard, no multi-location composition, no reference sheet layout, no sheet panels, "
    "no duplicate performer, no duplicate person, no clones, no multiple copies of the same person, "
    "no poster layout, no grid of scenes, no labels, no text, no captions, no watermark"
)


def _director_location_key(value: Any) -> str:
    text = _clean_text(value, 180).lower()
    text = re.sub(r"[^a-z0-9à-ÿ]+", " ", text, flags=re.IGNORECASE).strip()
    return re.sub(r"\s+", " ", text)


def _location_candidates_from_brief(brief: Dict[str, str]) -> List[str]:
    """Treat user locations/worlds as a full candidate pool, not one combined prompt."""
    raw = _clean_text((brief or {}).get("locations_world"), 4000)
    if not raw:
        return []
    text = raw
    text = re.sub(r"\s*(?:\n|\r|;|\|)\s*", " | ", text)
    comma_parts = [p.strip(" .,:;-") for p in text.split(",") if p.strip(" .,:;-")]
    if len(comma_parts) >= 2:
        parts = comma_parts
    else:
        parts = [p.strip(" .,:;-") for p in re.split(r"\s*\|\s*", text) if p.strip(" .,:;-")]
    # As a last resort only, split plain prose location lists on "and/or".
    if len(parts) <= 1:
        parts = [p.strip(" .,:;-") for p in re.split(r"\s+\b(?:and|or)\b\s+", text, flags=re.IGNORECASE) if p.strip(" .,:;-")]
    out: List[str] = []
    seen = set()
    for part in parts:
        part = re.sub(r"^(?:locations?|worlds?|settings?|backgrounds?)\s*[:=-]\s*", "", part, flags=re.IGNORECASE).strip(" .,:;-")
        if not part:
            continue
        if len(part) > 110:
            sub = re.split(r"\b(?:with|where|featuring|including)\b", part, maxsplit=1, flags=re.IGNORECASE)[0].strip(" .,:;-")
            part = sub if 4 <= len(sub) <= 110 else part[:107].rstrip(" ,.;") + "..."
        key = _director_location_key(part)
        if key and key not in seen:
            seen.add(key)
            out.append(part)
        # Keep the full user-provided pool within a sane debug/report bound.
        if len(out) >= 60:
            break
    if len(out) <= 1:
        return []
    return out


def _dominant_location_for_index(brief: Dict[str, str], index: int = 1) -> str:
    pool = _location_candidates_from_brief(brief)
    if pool:
        idx = max(0, _safe_int(index, 1) - 1) % len(pool)
        return pool[idx]
    return _visual_world(brief)


def _single_location_world_for_scene(brief: Dict[str, str], item: Optional[Dict[str, Any]] = None, index: int = 1) -> str:
    if isinstance(item, dict):
        explicit = _clean_text(item.get("dominant_location") or item.get("selected_location") or item.get("main_location"), 160)
        if explicit:
            return explicit
        index = _safe_int(item.get("index"), index)
    return _dominant_location_for_index(brief, index)


def _director_distribute_location_pool(shots: List[Dict[str, Any]], brief: Dict[str, str]) -> List[Dict[str, Any]]:
    pool = _location_candidates_from_brief(brief)
    out: List[Dict[str, Any]] = []
    if not pool:
        return [dict(s) for s in shots if isinstance(s, dict)]
    for pos, shot in enumerate([s for s in shots if isinstance(s, dict)]):
        sc = dict(shot)
        idx = max(0, _safe_int(sc.get("index"), pos + 1) - 1)
        loc = pool[idx % len(pool)]
        sc["dominant_location"] = loc
        sc["selected_location"] = loc
        sc["location_pool_index"] = idx % len(pool)
        sc["location_pool_size"] = len(pool)
        sc["location_source"] = "creative_brief_location_pool_rotation"
        out.append(sc)
    return out


def _director_locations_mentioned_in_text(text: Any, pool: List[str]) -> List[str]:
    norm_text = _director_location_key(text)
    hits: List[str] = []
    for loc in pool:
        key = _director_location_key(loc)
        if key and key in norm_text:
            hits.append(loc)
    return _dedupe_texts(hits, max_items=20, max_len=1200)


def _director_location_debug_summary(shots: List[Dict[str, Any]], brief: Dict[str, str]) -> Dict[str, Any]:
    pool = _location_candidates_from_brief(brief)
    pool_by_key = {_director_location_key(loc): loc for loc in pool}
    counts: Dict[str, int] = {loc: 0 for loc in pool}
    used_keys = set()
    multi_location_shots: List[Dict[str, Any]] = []
    per_shot: List[Dict[str, Any]] = []
    for shot in [s for s in shots if isinstance(s, dict)]:
        concept = shot.get("director_scene_concept") if isinstance(shot.get("director_scene_concept"), dict) else {}
        loc = _clean_text(concept.get("location") or shot.get("dominant_location"), 180)
        key = _director_location_key(loc)
        if key:
            used_keys.add(key)
            pretty = pool_by_key.get(key, loc)
            counts[pretty] = counts.get(pretty, 0) + 1
        prompt_hits = _director_locations_mentioned_in_text(_safe_str(shot.get("director_image_prompt")) + " " + _safe_str(shot.get("director_video_prompt")), pool)
        if len(prompt_hits) > 1:
            multi_location_shots.append({"id": _safe_str(shot.get("id")), "locations_mentioned": prompt_hits})
        per_shot.append({
            "id": _safe_str(shot.get("id")),
            "location": loc,
            "location_key": key,
            "prompt_locations_detected": prompt_hits,
            "one_location_ok": len(prompt_hits) <= 1,
            "llm_director_status": _safe_str(shot.get("llm_director_status") or shot.get("director_source_status")),
        })
    unused = [loc for loc in pool if _director_location_key(loc) not in used_keys]
    repeated = {loc: count for loc, count in counts.items() if count > 1}
    return {
        "full_location_pool": pool,
        "location_pool_count": len(pool),
        "locations_used_count": len([k for k in used_keys if k]),
        "unused_locations": unused,
        "repeated_location_counts": repeated,
        "one_location_per_shot_validation": {
            "ok": len(multi_location_shots) == 0,
            "shots_with_multiple_locations": multi_location_shots,
        },
        "per_shot_locations": per_shot,
    }


def _anti_collage_negative_terms(image_model: str = "") -> str:
    model = _safe_str(image_model).lower()
    extra = ""
    if "z" in model and "image" in model:
        extra = ", extra strict single coherent frame, no cloning, no random letters, no text artifacts"
    return _ANTI_COLLAGE_NEGATIVE_BASE + extra


def _contains_montage_word(value: Any) -> bool:
    text = _safe_str(value)
    if not text:
        return False
    # Do not treat anti-collage guard phrases as requests for collage.
    text = re.sub(
        r"\b(?:no|not|without|avoid)\s+(?:a\s+)?(?:collage|split[-\s]?screen|multi[-\s]?panel|contact[-\s]?sheet|moodboard|picture[-\s]?in[-\s]?picture|grid|poster\s+layout|multi[-\s]?location)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return bool(_COLLAGE_WORD_RE.search(text) or _MONTAGE_ROLE_WORD_RE.search(text))


def _prompt_has_montage_layout_request(*values: Any) -> bool:
    return any(_contains_montage_word(v) for v in values)


def _remove_collage_language(text: Any) -> str:
    out = _clean_text(text, 2600)
    if not out:
        return ""
    replacements = [
        (r"\bsplit[-\s]?screen\b", "single coherent frame"),
        (r"\bcollage\b", "single coherent frame"),
        (r"\bmulti[-\s]?panel\b", "single coherent frame"),
        (r"\bcontact[-\s]?sheet\b", "single coherent frame"),
        (r"\bmoodboard\b", "single coherent frame"),
        (r"\bpicture[-\s]?in[-\s]?picture\b", "single coherent frame"),
        (r"\bposter\s+layout\b", "single coherent frame"),
        (r"\bgrid\s+of\s+scenes\b", "single coherent frame"),
        (r"\bgrid\b", "single coherent frame"),
        (r"\bmulti[-\s]?location\s+composition\b", "single coherent location"),
        (r"\bmultiple\s+locations\s+in\s+one\s+frame\b", "one dominant location"),
        (r"\bseveral\s+locations\s+at\s+once\b", "one dominant location"),
    ]
    for pat, repl in replacements:
        out = re.sub(pat, repl, out, flags=re.IGNORECASE)
    # Repair negative guard phrases that may have contained the forbidden word.
    out = re.sub(r"\bnot\s+a\s+single\s+coherent\s+frame\b", "not a collage", out, flags=re.IGNORECASE)
    out = re.sub(r"\bnot\s+single\s+coherent\s+frame\b", "not a collage", out, flags=re.IGNORECASE)
    out = re.sub(r"\bno\s+a\s+single\s+coherent\s+frame\b", "no collage", out, flags=re.IGNORECASE)
    out = re.sub(r"\bno\s+single\s+coherent\s+frame\b", "no collage", out, flags=re.IGNORECASE)
    out = re.sub(r"\s{2,}", " ", out)
    return out.strip(" ,.;")


def _sanitize_single_frame_prompt(
    prompt: Any,
    *,
    brief: Optional[Dict[str, str]] = None,
    item: Optional[Dict[str, Any]] = None,
    allow_montage: bool = False,
    start_or_end: bool = False,
    image_model: str = "",
    max_len: int = 2200,
) -> str:
    text = _clean_text(prompt, max_len)
    if not text:
        return ""
    dominant = _single_location_world_for_scene(brief or {}, item, _safe_int((item or {}).get("index"), 1)) if isinstance(brief, dict) else ""
    if start_or_end or not allow_montage:
        text = _remove_collage_language(text)
        if isinstance(brief, dict) and dominant:
            # Replace the user's full multi-location list with the selected single
            # location, then remove stray sibling locations that survived from
            # character/identity prompt injection.
            raw_locations = _clean_text((brief or {}).get("locations_world"), 1200)
            if raw_locations and raw_locations in text:
                text = text.replace(raw_locations, dominant)
            for loc in _location_candidates_from_brief(brief):
                if loc and loc.lower() != dominant.lower():
                    text = re.sub(r"(?:,\s*)?" + re.escape(loc) + r"\b", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*,\s*,+", ", ", text)
            text = re.sub(r"\s{2,}", " ", text).strip(" ,.;")
        guard_parts = [
            text,
            "one coherent frame",
            "one dominant subject",
            f"one dominant location: {dominant}" if dominant else "one dominant location",
            "not a collage, not split-screen, not multi-panel, not a contact sheet, not a grid of scenes",
        ]
        if image_model and "z" in image_model.lower() and "image" in image_model.lower():
            guard_parts.append("extra strict Z-Image-safe clean composition, no duplicate performer, no random text")
        text = _join_parts(guard_parts)
    return _clean_text(text, max_len)


def _enforce_single_location_without_negative_words(prompt: Any, brief: Dict[str, str], item: Optional[Dict[str, Any]] = None, max_len: int = 2400) -> str:
    """Remove location-pool overload without adding anti-collage words to video prompts."""
    # Strip safety fragments before generic collage cleanup so existing compact
    # safety lines are not damaged into duplicate partial phrases.
    base_prompt = prompt
    try:
        base_prompt, _removed_safety = _director_collapse_visual_safety(prompt, add_when_needed=False)
    except Exception:
        base_prompt = prompt
    text = _remove_collage_language(base_prompt)
    dominant = _single_location_world_for_scene(brief or {}, item or {}, _safe_int((item or {}).get("index"), 1)) if isinstance(brief, dict) else ""
    if isinstance(brief, dict) and dominant:
        raw_locations = _clean_text((brief or {}).get("locations_world"), 1200)
        if raw_locations and raw_locations in text:
            text = text.replace(raw_locations, dominant)
        for loc in _location_candidates_from_brief(brief):
            if loc and loc.lower() != dominant.lower():
                text = re.sub(r"(?:,\s*)?" + re.escape(loc) + r"\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*,\s*,+", ", ", text)
        text = re.sub(r"\s{2,}", " ", text).strip(" ,.;")
    return _clean_text(text, max_len)


def _montage_policy_from_plan(*sources: Any) -> Dict[str, Any]:
    policy: Dict[str, Any] = {
        "allow_special_montage": False,
        "max_montage_events_per_video": 0,
        "used_montage_events": 0,
        "rule": "location/world lists are a pool; normal shots use one dominant location; collage is optional, rare, and never at the start or end",
    }
    for src in sources:
        if not isinstance(src, dict):
            continue
        for k in ("allow_special_montage", "max_montage_events_per_video", "used_montage_events", "rule"):
            if k in src:
                policy[k] = src.get(k)
        nested = src.get("montage_policy")
        if isinstance(nested, dict):
            for k in ("allow_special_montage", "max_montage_events_per_video", "used_montage_events", "rule"):
                if k in nested:
                    policy[k] = nested.get(k)
        effects = src.get("effects_profile")
        if not isinstance(effects, dict):
            bgs = src.get("bridge_generation_settings")
            if isinstance(bgs, dict):
                effects = _effects_profile_from_bridge_settings(bgs)
        if isinstance(effects, dict):
            collage_on = _safe_bool(effects.get("collage_effect_enabled"), False)
            policy["allow_special_montage"] = bool(collage_on)
            policy["max_montage_events_per_video"] = 1 if collage_on else 0
            policy["collage_control_source"] = "effects_profile"
    policy["allow_special_montage"] = _safe_bool(policy.get("allow_special_montage"), False)
    policy["max_montage_events_per_video"] = max(0, _safe_int(policy.get("max_montage_events_per_video"), 0))
    if not policy["allow_special_montage"]:
        policy["max_montage_events_per_video"] = 0
    policy["used_montage_events"] = max(0, _safe_int(policy.get("used_montage_events"), 0))
    return policy


def _is_intro_or_outro_item(item: Dict[str, Any], index: int, total: int) -> bool:
    if index <= 1 or (total > 0 and index >= total):
        return True
    section = _clean_text(item.get("section") or item.get("section_summary")).lower()
    role = _clean_text(item.get("scene_role") or item.get("scene_role_summary") or item.get("preferred_clip_role")).lower()
    return section in {"intro", "outro"} or "intro" in role or "outro" in role or "closing" in role


def _montage_capable_role(item: Dict[str, Any]) -> bool:
    role_text = " ".join([
        _safe_str(item.get("scene_role")),
        _safe_str(item.get("scene_role_summary")),
        _safe_str(item.get("preferred_clip_role")),
        _safe_str(item.get("microclip_style")),
        _safe_str(item.get("microclip_reason")),
    ]).lower()
    if _safe_bool(item.get("is_microclip"), False):
        # Collage is an optional separate effect now; do not treat timestamped
        # microclips as collage/montage candidates.
        return False
    if _MONTAGE_ROLE_WORD_RE.search(role_text):
        return True
    if any(x in role_text for x in ("impact", "transition", "burst", "beat_hit", "camera_flash", "b_roll_impact", "chorus", "drop")):
        return True
    energy = _safe_str(item.get("energy") or item.get("energy_summary")).lower()
    if any(x in energy for x in ("high", "peak", "intense")) and any(x in role_text for x in ("b_roll", "action", "dance", "visual")):
        return True
    return False


def _intro_outro_safety_defaults() -> Dict[str, Any]:
    return {
        "allow_micro_moments": True,
        "allow_aggressive_micro_moments": False,
        "allow_collage": False,
        "allow_split_screen": False,
        "allow_montage": False,
        "rule": "first and last full-video LTX clips stay coherent, single-location, and stable",
    }


def _stable_intro_outro_timestamped_prompt(shot: Dict[str, Any], brief: Dict[str, str], *, is_intro: bool, is_outro: bool) -> str:
    duration = max(0.1, _safe_float(shot.get("duration"), 0.0))
    needs_lipsync = _safe_bool(shot.get("needs_lipsync"), False)
    subject = _director_subject(brief, needs_lipsync) if "_director_subject" in globals() else (_clean_text(brief.get("characters_subjects"), 160) or "the main subject")
    world = _safe_str(shot.get("dominant_location")) or _single_location_world_for_scene(brief, shot, _safe_int(shot.get("index"), 1))
    end_t = _ltx_time_label(max(0.0, duration - 0.15)) if "_ltx_time_label" in globals() else f"0:{max(0, int(duration)):02d}"
    mid = max(0.75, min(duration * 0.52, max(0.75, duration - 0.75)))
    mid_t = _ltx_time_label(mid) if "_ltx_time_label" in globals() else "0:02"
    opening_role = "establishes the visual world" if is_intro else "opens the final closing moment"
    closing_role = "settles into a readable intro handoff" if is_intro else "holds a calm final closure"
    if needs_lipsync:
        first = f"0:00 - {subject} {opening_role} in one coherent {world} setup, face readable for the active vocal phrase, clean single full-frame composition"
        middle = f"{mid_t} - smooth camera push, angle change, or detail variation stays inside the same location while the performance continues naturally"
    else:
        first = f"0:00 - {subject} {opening_role} in one coherent {world} setup, moving with the music without visible singing, clean single full-frame composition"
        middle = f"{mid_t} - smooth camera push, angle change, or clean detail shot stays inside the same scene with controlled rhythm"
    last = f"{end_t} - one stable single-frame composition {closing_role}, same dominant location, stable full-frame ending"
    return _sentence(". ".join([first, middle, last]))


def _edge_safe_timestamp_prompt(value: Any, shot: Dict[str, Any], brief: Dict[str, str], *, is_intro: bool, is_outro: bool) -> str:
    text = _clean_text(value, 2400)
    # If the edge clip already asks for montage/collage/microclip chaos, replace it with a stable clip-level prompt.
    low = text.lower()
    chaotic = (
        _prompt_has_montage_layout_request(text)
        or bool(_safe_bool(shot.get("is_microclip"), False))
        or any(x in low for x in ("micro-moment", "micro moment", "microclip", "rapid unrelated", "contact sheet", "multi-panel", "split-screen"))
    )
    if chaotic or not text:
        return _stable_intro_outro_timestamped_prompt(shot, brief, is_intro=is_intro, is_outro=is_outro)
    guard = " Opening and final timestamp stay coherent, single-location, stable, and readable with controlled scene flow."
    return _sentence(text + guard)


def _apply_intro_outro_safety_to_items(
    items: List[Dict[str, Any]],
    *,
    brief: Dict[str, str],
    prompt_keys: Optional[List[str]] = None,
    image_model: str = "",
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Hard pass for first/last full-video LTX clips.

    The first clip establishes the music video and the last clip closes it, so
    neither may become an aggressive microclip/montage/collage shot even when
    EDM/DnB/Short profiles are enabled. This is plan-only prompt/metadata safety.
    """
    total = len(items)
    warnings: List[str] = []
    out: List[Dict[str, Any]] = []
    policy = {
        "first_clip_protected": bool(total >= 1),
        "last_clip_protected": bool(total >= 2),
        "allow_mild_internal_variation": True,
        "blocked_on_intro_outro": [
            "collage", "split-screen", "contact-sheet", "multi-panel",
            "chaotic montage burst", "aggressive micro-moment overload",
            "multiple unrelated locations",
        ],
        "rule": "first and last full-video LTX clips are stable coherent single-location clips",
    }
    if prompt_keys is None:
        prompt_keys = [
            "image_prompt", "video_prompt", "timestamped_video_prompt",
            "template_image_prompt", "template_video_prompt", "template_timestamped_video_prompt",
            "director_image_prompt", "director_video_prompt", "director_timestamped_video_prompt",
        ]
    for idx, raw in enumerate(items, start=1):
        item = dict(raw)
        is_intro = idx == 1
        is_outro = total > 1 and idx == total
        item["is_intro_clip"] = bool(is_intro)
        item["is_outro_clip"] = bool(is_outro)
        if is_intro or is_outro:
            item["intro_outro_safety"] = _intro_outro_safety_defaults()
            if _safe_bool(item.get("is_microclip"), False):
                warnings.append(f"{_safe_str(item.get('id')) or idx}: intro/outro safety disabled aggressive microclip behavior.")
            item["is_microclip"] = False
            item["microclip_reason"] = "blocked by intro/outro safety"
            item["microclip_style"] = ""
            item["micro_moment_count"] = 0
            item["is_montage_style"] = False
            item["montage_position_ok"] = False
            item["montage_reason"] = "blocked by intro/outro safety"
            role = _safe_str(item.get("scene_role_summary") or item.get("scene_role") or "")
            if is_intro:
                item["scene_role_summary"] = role or "intro_establishing_shot"
            elif is_outro:
                item["scene_role_summary"] = role or "outro_closing_shot"
            for key in prompt_keys:
                if key not in item:
                    continue
                if "timestamped" in key:
                    item[key] = _edge_safe_timestamp_prompt(item.get(key), item, brief, is_intro=is_intro, is_outro=is_outro)
                else:
                    item[key] = _sanitize_single_frame_prompt(
                        item.get(key),
                        brief=brief,
                        item=item,
                        allow_montage=False,
                        start_or_end=True,
                        image_model=image_model,
                        max_len=2200,
                    )
            neg_key = "director_negative_prompt" if "director_negative_prompt" in item else "negative_prompt"
            if neg_key in item:
                item[neg_key] = _negative_prompt_with_no_visible_text(
                    _join_parts([item.get(neg_key), _anti_collage_negative_terms(image_model)]),
                    max_len=1200,
                )
            item["intro_outro_safety_applied"] = True
        else:
            item.setdefault("intro_outro_safety", {
                "allow_micro_moments": True,
                "allow_aggressive_micro_moments": True,
                "allow_collage": False,
                "allow_split_screen": False,
                "allow_montage": bool(item.get("is_montage_style")),
            })
            item["intro_outro_safety_applied"] = False
        out.append(item)
    return out, policy, warnings


def _apply_effect_policy_metadata(
    items: List[Dict[str, Any]],
    *,
    effects_profile: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Write visible per-shot effect metadata without changing render logic."""
    effects = effects_profile if isinstance(effects_profile, dict) else _effects_profile_from_bridge_settings({})
    total = len(items or [])
    out: List[Dict[str, Any]] = []
    for idx, raw in enumerate(items or [], start=1):
        item = dict(raw)
        first = idx == 1
        last = total > 1 and idx == total
        avoid_first = _safe_bool(effects.get("avoid_effects_in_first_clip"), True)
        avoid_last = _safe_bool(effects.get("avoid_effects_in_last_clip"), True)
        edge_blocked = (first and avoid_first) or (last and avoid_last)
        micro = bool(_safe_bool(item.get("is_microclip"), False) and not edge_blocked)
        collage = bool(_safe_bool(item.get("is_montage_style"), False) and _safe_bool(effects.get("collage_effect_enabled"), False) and not edge_blocked)
        if edge_blocked:
            micro = False
            collage = False
        item["effects_used"] = {
            "timestamped_microclips": bool(micro),
            "collage_effect": bool(collage),
        }
        item["effect_policy"] = {
            "is_first_video_clip": bool(first),
            "is_last_video_clip": bool(last),
            "effects_allowed": bool(not edge_blocked),
            "timestamped_microclips_enabled": bool(effects.get("timestamped_microclips_enabled")),
            "collage_allowed": bool(_safe_bool(effects.get("collage_effect_enabled"), False) and not edge_blocked and not first and not last),
            "avoid_effects_in_first_clip": bool(avoid_first),
            "avoid_effects_in_last_clip": bool(avoid_last),
        }
        if not collage and _safe_str(item.get("collage_effect_reason")):
            item["collage_effect_reason"] = ""
        elif collage and not _safe_str(item.get("collage_effect_reason")):
            item["collage_effect_reason"] = _safe_str(item.get("montage_reason")) or "mid-video energetic impact"
        out.append(item)
    return out


def _apply_montage_policy_to_items(
    items: List[Dict[str, Any]],
    *,
    brief: Dict[str, str],
    policy: Optional[Dict[str, Any]] = None,
    image_model: str = "",
    prompt_keys: Optional[List[str]] = None,
    director_keys: bool = False,
) -> tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    """Final plan pass: one dominant location by default; max one montage event."""
    pol = _montage_policy_from_plan(policy or {})
    used = 0
    max_events = _safe_int(pol.get("max_montage_events_per_video"), 1)
    allow_any = _safe_bool(pol.get("allow_special_montage"), True) and max_events > 0
    warnings: List[str] = []
    out_items: List[Dict[str, Any]] = []
    total = len(items)
    if prompt_keys is None:
        prompt_keys = [
            "image_prompt", "video_prompt", "timestamped_video_prompt",
            "template_image_prompt", "template_video_prompt", "template_timestamped_video_prompt",
            "director_image_prompt", "director_video_prompt", "director_timestamped_video_prompt",
        ]
    for idx, raw in enumerate(items, start=1):
        item = dict(raw)
        dominant = _single_location_world_for_scene(brief, item, idx)
        item["dominant_location"] = dominant
        is_edge = _is_intro_or_outro_item(item, idx, total)
        requested = _prompt_has_montage_layout_request(
            item.get("scene_role"), item.get("scene_role_summary"), item.get("preferred_clip_role"),
            item.get("image_prompt"), item.get("video_prompt"), item.get("timestamped_video_prompt"),
            item.get("director_image_prompt"), item.get("director_video_prompt"), item.get("director_timestamped_video_prompt"),
        )
        role_ok = _montage_capable_role(item)
        middle_ok = not is_edge and idx > 1 and idx < total
        energy_ok = any(x in _safe_str(item.get("energy") or item.get("energy_summary") or item.get("section") or item.get("section_summary")).lower() for x in ("high", "peak", "drop", "chorus", "intense"))
        non_vocal_ok = not _safe_bool(item.get("needs_lipsync"), False) and not _safe_bool(item.get("is_microclip"), False)
        # Collage is now an explicit rare effect toggle. It may be used once in
        # a safe middle non-vocal shot even if the LLM did not already request a
        # montage word; it is still never tied to timestamped microclips.
        allow_this = bool(allow_any and middle_ok and used < max_events and non_vocal_ok and (requested or role_ok or energy_ok))
        if allow_this:
            used += 1
            item["is_montage_style"] = True
            item["montage_reason"] = "allowed rare mid-video montage-capable role"
            item["montage_position_ok"] = True
            item["collage_effect_reason"] = item.get("collage_effect_reason") or "mid-video effect-enabled impact moment"
            if not requested:
                for _tk in ("timestamped_video_prompt", "template_timestamped_video_prompt", "director_timestamped_video_prompt"):
                    if _tk in item and _safe_str(item.get(_tk)):
                        item[_tk] = _sentence(str(item.get(_tk) or "") + " At one middle timestamp only, use a brief collage/split-screen effect burst, then return to one full-frame scene before the final timestamp.")
                        break
            # Even the one allowed montage event must not use a collage as its
            # start image/opening still. Keep image/template image prompts clean;
            # montage may appear only inside the timestamped video prompt.
            for key in prompt_keys:
                if "image" in key and key in item:
                    item[key] = _sanitize_single_frame_prompt(
                        item.get(key),
                        brief=brief,
                        item=item,
                        allow_montage=False,
                        start_or_end=True,
                        image_model=image_model,
                        max_len=2200,
                    )
        else:
            if requested:
                warnings.append(f"{_safe_str(item.get('id')) or idx}: montage/collage wording removed; role/position/one-time policy did not allow it.")
            item["is_montage_style"] = False
            item["montage_reason"] = "normal single-location shot; montage blocked by policy"
            item["montage_position_ok"] = bool(middle_ok)
            for key in prompt_keys:
                if key in item:
                    is_image_key = "image" in key
                    if "timestamped" in key and not is_edge:
                        item[key] = _remove_collage_language(item.get(key))
                    else:
                        item[key] = _sanitize_single_frame_prompt(
                            item.get(key),
                            brief=brief,
                            item=item,
                            allow_montage=False,
                            start_or_end=is_edge or is_image_key,
                            image_model=image_model,
                            max_len=2200 if "negative" not in key else 1200,
                        )
        neg_key = "director_negative_prompt" if director_keys and "director_negative_prompt" in item else "negative_prompt"
        if neg_key in item:
            item[neg_key] = _negative_prompt_with_no_visible_text(
                _join_parts([item.get(neg_key), _anti_collage_negative_terms(image_model)]),
                max_len=1200,
            )
        out_items.append(item)
    pol["used_montage_events"] = used
    return out_items, pol, warnings



def _rotation_index(scene: Dict[str, Any]) -> int:
    base = max(0, _safe_int(scene.get("index"), 1) - 1)
    block_idx = max(0, _safe_int(scene.get("lyric_block_scene_index"), 0) - 1)
    block_count = _safe_int(scene.get("lyric_block_scene_count"), 0)
    if block_count > 1:
        return block_idx
    return base


def _pick(pool: List[str], scene: Dict[str, Any], extra: int = 0) -> str:
    if not pool:
        return "music-video shot"
    return pool[(max(0, _rotation_index(scene)) + int(extra or 0)) % len(pool)]


def _vocal_shot_pool(brief: Dict[str, str]) -> List[str]:
    pool = [
        "close-up singing to camera",
        "medium shot singing while moving",
        "side profile singing shot",
        "performance shot with backup dancers",
        "wide hero shot while continuing lipsync",
        "mirror or reflection singing shot",
        "over-shoulder performance shot",
    ]
    if _brief_has_vehicle_world(brief):
        pool[3:3] = [
            "inside vehicle or cockpit singing shot",
            "singer leaning beside the vehicle while performing",
        ]
    if _brief_has_dance_world(brief):
        pool.insert(4, "dance-performance shot while singing")
    return pool


def _nonvocal_shot_pool(brief: Dict[str, str]) -> List[str]:
    if _brief_has_vehicle_world(brief):
        return [
            "wide racing or action shot",
            "low wheel-level tracking shot",
            "aerial road or city fly-by shot",
            "vehicle detail shot with lights, dashboard or bodywork",
            "fast transition cutaway",
            "environmental spectacle shot",
            "crowd or dancers reacting to the rhythm",
            "chase or cruising shot through the visual world",
        ]
    return [
        "world establishing shot",
        "dance-only performance shot",
        "crowd or environment reaction shot",
        "fast transition cutaway",
        "close-up detail shot of a key object or texture",
        "wide action shot",
        "environmental spectacle shot",
        "rhythmic b-roll shot",
    ]


def _scene_progression_phase(scene: Dict[str, Any]) -> str:
    idx = _safe_int(scene.get("lyric_block_scene_index"), 0)
    count = _safe_int(scene.get("lyric_block_scene_count"), 0)
    if idx > 0 and count > 1:
        frac = idx / max(1, count)
        if frac <= 0.34:
            return "early"
        if frac <= 0.67:
            return "middle"
        return "late"
    scene_index = _safe_int(scene.get("index"), 1)
    mod = (scene_index - 1) % 3
    return ("early", "middle", "late")[mod]


def _scene_has_active_vocal_words(scene: Dict[str, Any], lyrics: str = "") -> bool:
    """True only for strong active lyric evidence in this exact scene.

    Lyric text attached to the scene is context only. It can come from a long
    SRT block or a repeated/stale overlap, so it must not automatically create
    lipsync intent. The phrase must start inside or very near the scene, or be
    part of a short confirmed active vocal window.
    """
    if not _clean_text(lyrics or scene.get("lyrics") or scene.get("lyric_text")):
        return False

    scene_start = _safe_float(scene.get("start", scene.get("song_start", 0.0)), 0.0)
    scene_end = _safe_float(scene.get("end", scene.get("song_end", 0.0)), 0.0)
    if scene_end <= scene_start:
        scene_end = scene_start + max(0.05, _safe_float(scene.get("duration"), 0.05))
    duration = max(0.05, scene_end - scene_start)

    lyric_has_timing = ("lyric_start" in scene) or ("lyric_end" in scene)
    if lyric_has_timing:
        ls = _safe_float(scene.get("lyric_start"), scene_start)
        le = _safe_float(scene.get("lyric_end"), scene_end)
        if le <= ls:
            return False
        overlap = min(scene_end, le) - max(scene_start, ls)
        if overlap < float(ACTIVE_LYRIC_MIN_OVERLAP_SECONDS):
            return False
        # A stale/long SRT block that started several seconds earlier should not
        # keep later instrumental scenes alive as lipsync scenes.
        if (scene_start - ls) > float(MAX_LIPSYNC_SECONDS_PER_LONG_SEGMENT):
            return False
        starts_near_scene = (scene_start - float(ACTIVE_LYRIC_NEAR_SHOT_START_SECONDS)) <= ls <= (scene_end - min(0.05, duration * 0.1))
        short_confirmed_overlap = (le - ls) <= float(LONG_LYRIC_SEGMENT_SECONDS) and overlap >= min(1.0, duration * 0.5) and (scene_start - ls) <= float(MAX_DELAYED_LIPSYNC_START_SECONDS)
        if not (starts_near_scene or short_confirmed_overlap):
            return False

    if "active_vocal_window" in scene and not bool(scene.get("active_vocal_window")):
        return False

    idx = _safe_int(scene.get("lyric_block_scene_index"), 0)
    count = _safe_int(scene.get("lyric_block_scene_count"), 0)
    block_duration = _safe_float(scene.get("lyric_block_duration"), 0.0)
    if count > 1 and idx > int(MAX_LIPSYNC_SCENES_PER_REPEATED_LYRIC):
        return False
    if block_duration > float(LONG_LYRIC_SEGMENT_SECONDS):
        block_start = _safe_float(scene.get("lyric_block_start"), _safe_float(scene.get("lyric_start"), 0.0))
        if max(0.0, scene_start - block_start) > float(MAX_LIPSYNC_SECONDS_PER_LONG_SEGMENT):
            return False
    return True

def _nonvocal_role_for_scene(scene: Dict[str, Any], brief: Dict[str, str], section: str, energy: str) -> str:
    preferred = _clean_text(scene.get("preferred_clip_role")).lower()
    if "transition" in preferred:
        return "transition_cutaway"
    if "impact" in preferred:
        return "impact_moment"
    dance_ok = _brief_has_dance_world(brief)
    if _brief_has_vehicle_world(brief):
        pool = ["vehicle_action", "instrumental_groove", "environment_story_beat", "impact_moment"]
        if dance_ok:
            pool.append("dance_performance")
        return _pick(pool, scene)
    if section in {"chorus", "drop"} or energy == "high":
        pool = ["instrumental_groove", "impact_moment", "wide_story_action", "environment_story_beat"]
        if dance_ok:
            pool.insert(1, "dance_performance")
        return _pick(pool, scene)
    if section in {"intro", "outro"}:
        return _pick(["environment_story_beat", "transition_cutaway", "wide_story_action"], scene)
    if "bridge" in section:
        pool = ["environment_story_beat", "transition_cutaway", "emotional_detail"]
        if dance_ok:
            pool.append("dance_performance")
        return _pick(pool, scene)
    pool = ["instrumental_groove", "environment_story_beat", "wide_story_action", "transition_cutaway"]
    if dance_ok:
        pool.insert(2, "dance_performance")
    return _pick(pool, scene)

def _strip_non_lipsync_vocal_language(value: Any) -> str:
    """Remove wording that tells a non-vocal shot to sing or lipsync."""
    text = _clean_text(value, 2400)
    if not text:
        return ""
    replacements = [
        (r"\bstarts?\s+(?:the\s+)?vocal performance immediately\b", "moves with the beat immediately"),
        (r"\bstarts?\s+singing immediately\b", "moves with the beat immediately"),
        (r"\bkeeps?\s+lipsyncing\b", "keeps moving with the beat"),
        (r"\bcontinues?\s+lipsyncing\b", "continues moving with the beat"),
        (r"\blipsync(?:ing)?\b", "beat-driven movement"),
        (r"\b(?:active\s+)?vocal phrase\b", "beat phrase"),
        (r"\bperforms?\s+the\s+lyrics\b", "moves with the beat"),
        (r"\bsinger\s+continues\b", "performer continues"),
        (r"\bstarts?\s+singing\b", "moves with the beat"),
        (r"\bkeeps?\s+singing\b", "keeps moving with the beat"),
        (r"\bcontinues?\s+singing\b", "continues moving with the beat"),
        (r"\bvocal performance\b", "music-video performance"),
        (r"\bsinging\b", "moving with the beat"),
        (r"\bsinger\b", "performer"),
        (r"\bholding\s+a\s+microphone\b", "moving with the beat"),
        (r"\bsinging\s+into\s+(?:a\s+)?microphone\b", "moving with the beat"),
        (r"\bmicrophone\b", "expressive movement"),
        (r"\bmouth stays visible for [^,.;]+", "face remains naturally framed"),
        (r"\bmouth clearly visible\b", "face naturally visible"),
        (r"\bvisible face and mouth\b", "expressive face and movement"),
        (r"\bface and mouth clearly visible\b", "expressive face and movement"),
        (r"\bface and mouth readable\b", "expressive face-readable movement"),
        (r"\bmouth not visible\b", "unclear expression"),
        (r"\bcovered mouth\b", "covered face"),
        (r"\bhidden mouth\b", "hidden face"),
        (r"\bthe singer/main subject\b", "the main subject"),
        (r"\blead singer/performer\b", "lead performer"),
        (r"\blead singer\b", "lead performer"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:do not|don't)\s+force\s+(?:a\s+)?(?:performer|main subject)\s+beat-driven movement\b", "avoid forced lyric-mouthing", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:do not|don't)\s+force\s+(?:a\s+)?(?:performer|main subject)\s+performing\b", "avoid forced lyric-mouthing", text, flags=re.IGNORECASE)
    text = re.sub(r"\bface\s+visible\s+but\s+not\s+mouthing\s+lyrics\s+yet\b", "natural performance framing without lyric-mouthing", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnot\s+singing\s+yet\b", "not mouthing lyrics", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    return text


def _nonvocal_prompt_keys() -> tuple:
    return (
        "image_prompt", "video_prompt", "timestamped_video_prompt",
        "template_image_prompt", "template_video_prompt", "template_timestamped_video_prompt",
        "director_image_prompt", "director_video_prompt", "director_timestamped_video_prompt",
        "scene_intent", "camera_motion", "character_action", "choreography",
    )


def _nonvocal_scene_role_fallback(item: Dict[str, Any]) -> str:
    blob = " ".join([
        _clean_text(item.get("scene_role")), _clean_text(item.get("scene_role_summary")),
        _clean_text(item.get("suggested_shot_type")), _clean_text(item.get("section")),
        _clean_text(item.get("section_summary")), _clean_text(item.get("energy")),
        _clean_text(item.get("energy_summary")), _clean_text(item.get("image_prompt")),
        _clean_text(item.get("director_image_prompt")),
    ]).lower()
    if any(x in blob for x in ("car", "vehicle", "road", "highway", "racing", "cockpit", "driving")):
        return "vehicle_action"
    if any(x in blob for x in ("dance", "club", "stage", "crowd", "choreography")):
        return "dance_performance"
    if any(x in blob for x in ("outro", "ending", "closing", "final")):
        return "music_video_b_roll"
    return "instrumental_groove"


def _strong_active_lyric_evidence_from_item(item: Dict[str, Any]) -> tuple:
    """Return (has_evidence, reason) for lipsync in this exact LTX/director shot.

    This is intentionally stricter than "any lyric text overlaps". A shot only
    gets lipsync when a real phrase starts inside/near the shot, or a short
    confirmed phrase clearly overlaps near the start. Long/repeated SRT blocks
    are treated as stale after their phrase start.
    """
    shot_start = _safe_float(item.get("song_start"), _safe_float(item.get("start"), 0.0))
    duration = max(0.05, _safe_float(item.get("duration"), _safe_float(item.get("song_end"), 0.0) - shot_start))
    shot_end = shot_start + duration
    best_reason = "no_clip_relative_lyrics"
    for lyr in _as_list(item.get("clip_relative_lyrics")):
        if not isinstance(lyr, dict) or not _clean_text(lyr.get("text")):
            continue
        rel_start = max(0.0, _safe_float(lyr.get("start"), 0.0))
        rel_end = max(0.0, _safe_float(lyr.get("end"), 0.0))
        if rel_end <= rel_start:
            continue
        overlap = rel_end - rel_start
        if overlap < float(ACTIVE_LYRIC_MIN_OVERLAP_SECONDS):
            best_reason = "lyric_overlap_too_short"
            continue
        src_start = _safe_float(lyr.get("source_start"), shot_start + rel_start)
        src_end = _safe_float(lyr.get("source_end"), shot_start + rel_end)
        long_block = bool(lyr.get("long_block_active_window"))
        phrase_starts_inside_or_near = (shot_start - float(ACTIVE_LYRIC_NEAR_SHOT_START_SECONDS)) <= src_start <= (shot_end - min(0.05, duration * 0.1))
        phrase_starts_near_clip_start = rel_start <= float(ACTIVE_LYRIC_NEAR_SHOT_START_SECONDS)

        if long_block:
            # Long SRT lines are the main false-positive source. Only allow lipsync
            # right at the phrase start, not later inside the same long text block.
            if src_start < shot_start - float(LONG_BLOCK_STRICT_START_GRACE_SECONDS):
                best_reason = "long_srt_block_started_before_this_shot"
                continue
            if not (phrase_starts_inside_or_near and phrase_starts_near_clip_start):
                best_reason = "long_srt_block_not_at_active_phrase_start"
                continue
            return True, "active_long_srt_phrase_start_inside_shot"

        if phrase_starts_inside_or_near and phrase_starts_near_clip_start:
            return True, "active_lyric_phrase_start_inside_shot"

        # Short SRT phrases may begin a fraction before a shot after snapping.
        if (src_end - src_start) <= float(LONG_LYRIC_SEGMENT_SECONDS) and overlap >= min(1.0, duration * 0.5) and (shot_start - src_start) <= float(MAX_DELAYED_LIPSYNC_START_SECONDS):
            return True, "short_active_lyric_overlap_near_shot_start"

        best_reason = "lyric_text_overlap_but_no_active_phrase_start"
    return False, best_reason


def _active_lyric_words_from_item(item: Dict[str, Any]) -> bool:
    for lyr in _as_list(item.get("clip_relative_lyrics")):
        if isinstance(lyr, dict) and _clean_text(lyr.get("text")):
            start = _safe_float(lyr.get("start"), 0.0)
            end = _safe_float(lyr.get("end"), 0.0)
            if end > start:
                return True
    return False


def _first_active_lyric_offset(item: Dict[str, Any]) -> float:
    starts = []
    for lyr in _as_list(item.get("clip_relative_lyrics")):
        if isinstance(lyr, dict) and _clean_text(lyr.get("text")):
            starts.append(max(0.0, _safe_float(lyr.get("start"), 0.0)))
    return min(starts) if starts else 9999.0


def _repair_non_lipsync_item(item: Dict[str, Any], reason: str = "no_active_lyric_window") -> Dict[str, Any]:
    out = dict(item)
    role = _nonvocal_scene_role_fallback(out)
    out["needs_lipsync"] = False
    out["active_vocal_window"] = False
    out["active_vocal_scene_count"] = 0
    if "lipsync_windows" in out:
        out["lipsync_windows"] = []
    out["vocal_presence"] = "instrumental" if not _active_lyric_words_from_item(out) else "weak"
    out["vocal_presence_summary"] = out["vocal_presence"]
    out["scene_role"] = role
    out["scene_role_summary"] = role
    out["vocal_timing_reason"] = reason
    out["active_lipsync_confidence"] = "none"
    out["active_lyric_confidence"] = "none"
    out["lipsync_downgrade_reason"] = reason
    for key in _nonvocal_prompt_keys():
        if key in out:
            out[key] = _strip_non_lipsync_vocal_language(out.get(key))
    note = f"Lipsync downgraded: {reason}."
    if _safe_str(out.get("director_notes")):
        if note.lower() not in _safe_str(out.get("director_notes")).lower():
            out["director_notes"] = _clean_text(_join_parts([out.get("director_notes"), note]), 900)
    else:
        out["director_notes"] = note
    return out


def _protect_lipsync_confidence(item: Dict[str, Any], *, song_duration: float = 0.0) -> Dict[str, Any]:
    out = dict(item)
    wants_lipsync = _safe_bool(out.get("needs_lipsync"), False)
    start = _safe_float(out.get("song_start"), _safe_float(out.get("start"), 0.0))
    duration = max(0.0, _safe_float(out.get("duration"), _safe_float(out.get("song_end"), 0.0) - start))
    near_tail = bool(song_duration and (start >= max(0.0, song_duration - float(INSTRUMENTAL_TAIL_GUARD_SECONDS))))
    strong_evidence, evidence_reason = _strong_active_lyric_evidence_from_item(out)
    first_offset = _first_active_lyric_offset(out)

    out["active_lyric_confidence"] = "strong" if strong_evidence else "none"
    out["lipsync_decision_source"] = evidence_reason

    if not strong_evidence:
        reason = "instrumental_tail_no_active_lyrics" if near_tail else evidence_reason
        fixed = _repair_non_lipsync_item(out, reason)
        fixed["active_lyric_confidence"] = "none"
        fixed["lipsync_decision_source"] = evidence_reason
        fixed["lipsync_downgrade_reason"] = reason
        return fixed

    if duration > 0 and wants_lipsync and first_offset > min(float(MAX_DELAYED_LIPSYNC_START_SECONDS), max(0.2, duration * 0.35)):
        fixed = _repair_non_lipsync_item(out, "delayed_or_short_vocal_window_prefers_non_lipsync")
        fixed["active_lyric_confidence"] = "weak"
        fixed["lipsync_decision_source"] = evidence_reason
        fixed["lipsync_downgrade_reason"] = "delayed_or_short_vocal_window_prefers_non_lipsync"
        return fixed

    if wants_lipsync:
        out["active_lipsync_confidence"] = "strong"
        out["active_lyric_confidence"] = "strong"
        out["active_vocal_window"] = True
        out["lipsync_downgrade_reason"] = ""
        if not _safe_str(out.get("vocal_presence")) or _safe_str(out.get("vocal_presence")).lower() in {"weak", "instrumental"}:
            out["vocal_presence"] = "strong"
        if not _safe_str(out.get("vocal_presence_summary")) or _safe_str(out.get("vocal_presence_summary")).lower() in {"weak", "instrumental"}:
            out["vocal_presence_summary"] = "strong"
        return out

    fixed = _repair_non_lipsync_item(out, "active_lyrics_present_but_lipsync_not_confirmed")
    fixed["active_lyric_confidence"] = "weak"
    fixed["lipsync_decision_source"] = evidence_reason
    fixed["lipsync_downgrade_reason"] = "active_lyrics_present_but_lipsync_not_confirmed"
    return fixed

def _nonvocal_vocal_presence(scene: Dict[str, Any], lyrics: str) -> str:
    return "weak" if _clean_text(lyrics) else "instrumental"


def _scene_role_for(scene: Dict[str, Any], brief: Dict[str, str], section: str, energy: str, lyrics: str, use_roles: bool = True) -> Dict[str, Any]:
    has_lyric_metadata = bool(_clean_text(lyrics))
    active_vocal = bool(use_roles) and _scene_has_active_vocal_words(scene, lyrics)
    idx = _rotation_index(scene)

    if not bool(use_roles):
        # Even with role planning disabled, do not turn plain lyric metadata into forced lipsync.
        role = "instrumental_groove"
        active_vocal = False
    elif active_vocal:
        if section in {"chorus", "drop"} or energy == "high":
            role = _pick(["vocal_performance", "mixed_vocal_action", "vocal_closeup"], scene)
        elif section == "break" or energy == "low":
            role = _pick(["vocal_closeup", "vocal_performance", "mixed_vocal_action"], scene)
        elif section in {"intro", "outro"}:
            role = _pick(["vocal_closeup", "vocal_performance", "transition_cutaway"], scene)
        else:
            role = _pick(["vocal_closeup", "vocal_performance", "mixed_vocal_action", "vocal_performance"], scene)
    else:
        role = _nonvocal_role_for_scene(scene, brief, section, energy)

    if active_vocal:
        vocal_presence = "strong" if (section in {"chorus", "drop"} or energy == "high") else "medium"
        needs_lipsync = role != "transition_cutaway"
        identity_priority = "high" if needs_lipsync else "medium"
        variety_priority = "high" if _safe_int(scene.get("lyric_block_scene_count"), 0) >= 3 else "medium"
    else:
        vocal_presence = _nonvocal_vocal_presence(scene, lyrics)
        needs_lipsync = False
        identity_priority = "medium" if has_lyric_metadata else "low"
        variety_priority = "high"

    if section in {"chorus", "drop"} or energy == "high" or role == "dance_performance":
        dance_priority = "high"
    elif has_lyric_metadata or _brief_has_dance_world(brief):
        dance_priority = "medium"
    else:
        dance_priority = "low"

    shot_pool = _vocal_shot_pool(brief) if active_vocal else _nonvocal_shot_pool(brief)
    if role == "transition_cutaway" and not active_vocal:
        shot_type = _pick(["fast transition cutaway", "environmental detail cutaway", "rhythmic match-cut shot"], scene)
    elif role == "impact_moment" and not active_vocal:
        shot_type = _pick(["wide impact shot", "sudden beat-hit spectacle shot", "dynamic action peak shot"], scene)
    elif role == "vehicle_action":
        shot_type = _pick(["wide racing or action shot", "vehicle detail shot with lights, dashboard or bodywork", "chase or cruising shot through the visual world"], scene)
    elif role == "instrumental_groove":
        shot_type = _pick(["dance-only performance shot", "rhythmic b-roll shot", "vehicle detail shot with lights, dashboard or bodywork"], scene)
    elif role == "music_video_b_roll":
        shot_type = _pick(["world establishing shot", "environmental spectacle shot", "close-up detail shot of a key object or texture"], scene)
    else:
        shot_type = _pick(shot_pool, scene)

    return {
        "scene_role": role,
        "needs_lipsync": bool(needs_lipsync),
        "vocal_presence": vocal_presence,
        "identity_priority": identity_priority,
        "variety_priority": variety_priority,
        "dance_priority": dance_priority,
        "suggested_shot_type": shot_type,
        "progression_phase": _scene_progression_phase(scene),
        "active_vocal_window": bool(active_vocal),
        "vocal_timing_reason": _safe_str(scene.get("vocal_timing_reason")) or ("active_vocal_words" if active_vocal else "no_active_vocal_words"),
    }

def _mood_for_scene(section: str, energy: str, style: str, role_plan: Optional[Dict[str, Any]] = None) -> str:
    role = _clean_text((role_plan or {}).get("scene_role"))
    if role in {"vocal_closeup", "vocal_performance"}:
        return _join_parts(["expressive performance mood", "face-readable music-video lighting", style])
    if role == "mixed_vocal_action":
        return _join_parts(["performance mixed with action energy", "clear musical rhythm", style])
    if role == "dance_performance":
        return _join_parts(["dance-heavy performance energy", "strong body movement", style])
    if role in {"world_broll", "music_video_b_roll"}:
        return _join_parts(["world-building atmosphere", "rhythmic visual detail", style])
    if role == "impact_moment":
        return _join_parts(["big impact energy", "strong visual hit on the beat", style])
    if section in {"chorus", "drop"} or energy == "high":
        return _join_parts(["bigger chorus energy", "strong movement", "bright visual impact", style])
    if section == "break" or energy == "low":
        return _join_parts(["dreamy atmosphere", "slower emotional movement", "soft surreal tension", style])
    if section == "intro":
        return _join_parts(["establishing mood", "anticipation before the beat opens", style])
    if section == "outro":
        return _join_parts(["resolved closing mood", "final reflective energy", style])
    return _join_parts(["story-focused performance energy", "clear musical rhythm", style])


def _camera_for_scene(section: str, energy: str, camera_hint: str, role_plan: Optional[Dict[str, Any]] = None) -> str:
    hint = _clean_text(camera_hint)
    role = _clean_text((role_plan or {}).get("scene_role"))
    shot = _clean_text((role_plan or {}).get("suggested_shot_type"))
    needs_lipsync = _safe_bool((role_plan or {}).get("needs_lipsync"), False)
    if needs_lipsync:
        base = hint or "face-aware music-video camera move"
        if "close-up" in shot:
            return f"{base}, keeping the singer's face and mouth clearly visible for lipsync"
        if "side profile" in shot:
            return f"{base}, holding a readable side profile so the lipsync stays visible"
        if "inside" in shot or "cockpit" in shot:
            return f"{base}, tracking the singer inside the vehicle while keeping the face readable"
        return f"{base}, keeping the lead performer's face readable while the shot moves"
    if hint:
        return hint
    if role == "dance_performance":
        return "rhythmic camera move following the choreography"
    if role in {"instrumental_action", "instrumental_groove", "vehicle_action"}:
        return "fast action tracking move with strong beat timing"
    if role in {"world_broll", "music_video_b_roll"}:
        return "smooth cinematic world-building move"
    if role == "transition_cutaway":
        return "quick rhythmic cutaway move that connects to the next shot"
    if role == "impact_moment":
        return "bold push-in or whip-style reveal on the impact beat"
    if section in {"chorus", "drop"} or energy == "high":
        return "dynamic push-in with a wider reveal on the beat"
    if section == "break" or energy == "low":
        return "slow cinematic drift with gentle parallax"
    if section == "intro":
        return "slow establishing camera move"
    if section == "outro":
        return "smooth pull-back that resolves the scene"
    return "steady music-video camera move following the main action"


def _scene_intent(section: str, energy: str, lyrics: str, role: str, role_plan: Optional[Dict[str, Any]] = None) -> str:
    lyric_hint = _lyric_hint_text(lyrics)
    role_text = _clean_text(role)
    scene_role = _clean_text((role_plan or {}).get("scene_role"))
    if _safe_bool((role_plan or {}).get("needs_lipsync"), False):
        if section in {"chorus", "drop"} or energy == "high":
            return _sentence(f"Create a high-energy singing and dance-performance moment, {lyric_hint}")
        if section == "break" or energy == "low":
            return _sentence(f"Create an emotional face-readable vocal performance moment, {lyric_hint}")
        return _sentence(f"Create a clear vocal performance scene with consistent identity, {lyric_hint}")
    if scene_role == "dance_performance":
        return "Use this as a dance-only performance beat with strong musical movement."
    if scene_role in {"world_broll", "music_video_b_roll"}:
        return "Build the visual world with a strong non-vocal b-roll shot."
    if scene_role in {"instrumental_action", "instrumental_groove", "vehicle_action", "performance_no_lipsync"}:
        return "Use this section for beat-driven action, motion and visual variety without lyric-mouthing."
    if scene_role == "transition_cutaway":
        return "Use this as a clean rhythmic cutaway that bridges two moments."
    if scene_role == "impact_moment":
        return "Make this a strong visual hit that lands with the music."
    if lyric_hint:
        return _sentence(f"Move the story forward through a performance moment, {lyric_hint}")
    if role_text:
        return _sentence(f"Use this scene as a {role_text} moment in the music video")
    if section in {"chorus", "drop"} or energy == "high":
        return "Make this a bigger, more energetic music-video highlight."
    if section == "break" or energy == "low":
        return "Use this as a dreamy, slower visual breath before the next section."
    if section == "intro":
        return "Establish the visual world and the main subject."
    if section == "outro":
        return "Resolve the visual idea and leave a clean ending image."
    return "Keep the music-video story moving with one clear performance action."


def _action_for_scene(section: str, energy: str, subject: str, lyrics: str, role: str, role_plan: Optional[Dict[str, Any]] = None) -> str:
    subject_text = _clean_text(subject) or "the main subject"
    lyric_hint = _lyric_hint_text(lyrics)
    role_text = _clean_text(role).lower()
    scene_role = _clean_text((role_plan or {}).get("scene_role"))
    phase = _clean_text((role_plan or {}).get("progression_phase")) or "middle"
    shot = _clean_text((role_plan or {}).get("suggested_shot_type"))

    if _safe_bool((role_plan or {}).get("needs_lipsync"), False):
        if phase == "early":
            return f"{subject_text} starts singing immediately, {shot}, establishing the lyric performance"
        if phase == "middle":
            return f"{subject_text} continues lipsyncing while the performance develops with dancing and movement"
        return f"{subject_text} pushes the vocal performance into a stronger movement before the next cut"

    if scene_role == "dance_performance":
        return f"dancers or the main subject perform choreography in rhythm with the music"
    if scene_role == "instrumental_action":
        if _brief_has_vehicle_world({"x": subject_text}):
            return "vehicles surge through the scene with strong beat-synced action"
        return "the scene uses fast visual action and rhythmic movement without forcing a singer into frame"
    if scene_role == "world_broll":
        return "the camera explores the visual world with music-driven b-roll detail"
    if scene_role == "transition_cutaway" or "transition" in role_text:
        return "a fast rhythmic cutaway links this beat to the next scene"
    if scene_role == "impact_moment" or "impact" in role_text:
        return "a bold visual impact lands directly on the beat"
    if section in {"chorus", "drop"} or energy == "high":
        return f"{subject_text} or the surrounding world bursts into a powerful synchronized music-video movement"
    if section == "break" or energy == "low":
        return f"{subject_text} or the environment moves through a dreamy visual pause"
    if section == "intro":
        return f"the visual world opens as the music begins"
    if section == "outro":
        return f"the final movement resolves into a clean ending image"
    if lyric_hint:
        return f"{subject_text} performs a story-focused action shaped by the lyric"
    return f"one clear music-video action plays in rhythm with the beat"


def _image_prompt_for_scene(brief: Dict[str, str], scene: Dict[str, Any], section: str, energy: str, mood: str, role_plan: Optional[Dict[str, Any]] = None) -> str:
    subject = _lead_subject(brief)
    world = _single_location_world_for_scene(brief, scene, _safe_int(scene.get("index"), 1))
    style = _style_text(brief)
    lyrics = _clean_text(scene.get("lyrics") or scene.get("lyric_text"), 140)
    shot = _clean_text((role_plan or {}).get("suggested_shot_type")) or "music-video starting frame"
    lyric_part = _TEXT_IN_IMAGE_SAFE_PHRASE if lyrics else "clear visual hook for this music section"
    if _safe_bool((role_plan or {}).get("needs_lipsync"), False):
        parts = [
            f"{shot} of {subject}",
            "consistent lead singer identity",
            "visible face and mouth for lipsync",
            "clear performance framing",
            world,
            style,
            mood,
            lyric_part,
            "polished high-quality music video starting frame",
        ]
    else:
        action = _action_for_scene(section, energy, subject, lyrics, _clean_text(scene.get("preferred_clip_role")), role_plan)
        parts = [
            shot,
            action,
            world,
            style,
            mood,
            lyric_part,
            "strong composition, readable silhouettes, detailed lighting, high-quality music video frame",
        ]
    return _sentence(_sanitize_prompt_no_visible_text(_join_parts(parts), lyrics, image_prompt=True))


def _video_prompt_for_scene(brief: Dict[str, str], scene: Dict[str, Any], section: str, energy: str, mood: str, camera_motion: str, role_plan: Optional[Dict[str, Any]] = None) -> str:
    subject = _lead_subject(brief)
    world = _single_location_world_for_scene(brief, scene, _safe_int(scene.get("index"), 1))
    lyrics = _clean_text(scene.get("lyrics") or scene.get("lyric_text"), 140)
    action = _action_for_scene(section, energy, subject, lyrics, _clean_text(scene.get("preferred_clip_role")), role_plan)
    if _safe_bool((role_plan or {}).get("needs_lipsync"), False):
        prompt = f"{action} in {world}; the singer begins performing at once, moves with the beat, and the camera keeps the face visible enough for lipsync, {camera_motion}, {mood}"
    else:
        prompt = f"{action} in {world}, {camera_motion}, strong music-video motion and visual variety, {mood}"
    return _sentence(prompt)


def _format_clip_time(seconds: float) -> str:
    try:
        s = max(0, int(round(float(seconds))))
    except Exception:
        s = 0
    return f"0:{s:02d}"


def _timestamp_points(duration: float) -> List[int]:
    dur = max(0.1, float(duration or 0.0))
    if dur < 3.0:
        raw = [0, 1, min(2, max(1, int(round(dur))))]
    elif dur <= 5.0:
        raw = [0, 2, min(4, max(2, int(round(dur))))]
    else:
        raw = [0, 2, 4, min(6, max(4, int(round(dur))))]
    out: List[int] = []
    for value in raw:
        iv = int(max(0, value))
        if iv not in out:
            out.append(iv)
    while len(out) < 2:
        out.append(len(out))
    return out[:4]


def _timestamped_video_prompt_for_scene(brief: Dict[str, str], scene: Dict[str, Any], section: str, energy: str, camera_motion: str, role_plan: Optional[Dict[str, Any]] = None) -> str:
    subject = _lead_subject(brief)
    world = _single_location_world_for_scene(brief, scene, _safe_int(scene.get("index"), 1))
    style = _style_text(brief)
    lyrics = _clean_text(scene.get("lyrics") or scene.get("lyric_text"), 120)
    shot = _clean_text((role_plan or {}).get("suggested_shot_type")) or "music-video shot"
    action = _action_for_scene(section, energy, subject, lyrics, _clean_text(scene.get("preferred_clip_role")), role_plan)
    points = _timestamp_points(_safe_float(scene.get("duration"), 4.0))
    needs_lipsync = _safe_bool((role_plan or {}).get("needs_lipsync"), False)

    beats: List[str] = []
    for n, t in enumerate(points):
        if needs_lipsync:
            if n == 0:
                line = f"{subject} starts singing immediately, {shot}, visible face and mouth, {world}, {style}"
            elif n == 1:
                line = f"The singer keeps lipsyncing while the body moves with the beat and the lyric mood develops"
            elif n == 2:
                line = f"The performance varies through dancing or a new angle while {camera_motion}"
            else:
                line = "The vocal shot opens wider and carries the performance into the next music-video moment"
        else:
            if n == 0:
                line = f"{action} in {world}, {style}"
            elif n == 1:
                line = f"The non-vocal action builds with stronger movement and rhythm"
            elif n == 2:
                line = f"The shot reveals more detail while {camera_motion}"
            else:
                line = "The scene peaks visually and resolves into the next cut"
        beats.append(f"{_format_clip_time(t)} - {_clean_text(line).rstrip('.')}." )
    return re.sub(r"\s+", " ", " ".join(beats).replace("\n", " ")).strip()


def _negative_prompt_for_role(role_plan: Dict[str, Any]) -> str:
    base = ["blurry", "low quality", "unreadable text", "broken anatomy", "random unrelated scene", "inconsistent character design"]
    if _safe_bool(role_plan.get("needs_lipsync"), False):
        base.extend(["hidden face", "covered mouth", "face out of frame", "chaotic camera during lipsync"])
    return _negative_prompt_with_no_visible_text(", ".join(base), max_len=1200)


def _prompt_scene_from_plan_scene(scene: Dict[str, Any], brief: Dict[str, str], use_vocal_roles: bool = True) -> Dict[str, Any]:
    section = _section_kind(scene)
    energy = _energy_kind(scene)
    style = _style_text(brief)
    lyrics = _clean_text(scene.get("lyrics") or scene.get("lyric_text"), 700)
    role = _clean_text(scene.get("preferred_clip_role"))
    role_plan = _scene_role_for(scene, brief, section, energy, lyrics, use_roles=use_vocal_roles)
    mood = _mood_for_scene(section, energy, style, role_plan)
    camera_motion = _camera_for_scene(section, energy, _brief_value(brief, "camera_choreography", ""), role_plan)
    subject = _lead_subject(brief)
    action = _action_for_scene(section, energy, subject, lyrics, role, role_plan)
    dominant_location = _single_location_world_for_scene(brief, scene, _safe_int(scene.get("index"), 1))

    out = {
        "id": _safe_str(scene.get("id")) or f"S{_safe_int(scene.get('index'), 1):02d}",
        "index": _safe_int(scene.get("index"), 1),
        "song_start": round(_safe_float(scene.get("start"), _safe_float(scene.get("song_start"), 0.0)), 3),
        "song_end": round(_safe_float(scene.get("end"), _safe_float(scene.get("song_end"), 0.0)), 3),
        "duration": round(_safe_float(scene.get("duration"), 0.0), 3),
        "section": _safe_str(scene.get("section") or scene.get("section_label"), section) or section,
        "energy": _safe_str(scene.get("energy") or scene.get("energy_class"), energy) or energy,
        "lyrics": lyrics,
        "lyric_text": lyrics,
        "lyric_block_id": _safe_str(scene.get("lyric_block_id")),
        "lyric_block_scene_index": _safe_int(scene.get("lyric_block_scene_index"), 0),
        "lyric_block_scene_count": _safe_int(scene.get("lyric_block_scene_count"), 0),
        "dominant_location": dominant_location,
        "location_pool_index": max(0, _safe_int(scene.get("index"), 1) - 1),
        "scene_role": _safe_str(role_plan.get("scene_role")),
        "needs_lipsync": _safe_bool(role_plan.get("needs_lipsync"), False),
        "vocal_presence": _safe_str(role_plan.get("vocal_presence")),
        "active_vocal_window": _safe_bool(role_plan.get("active_vocal_window"), False),
        "vocal_timing_reason": _safe_str(role_plan.get("vocal_timing_reason")),
        "active_lyric_confidence": "scene_timed" if _safe_bool(role_plan.get("needs_lipsync"), False) else "none",
        "lipsync_decision_source": _safe_str(role_plan.get("vocal_timing_reason")) or ("active_scene_lyric_timing" if _safe_bool(role_plan.get("needs_lipsync"), False) else "no_active_scene_lyric_timing"),
        "lipsync_downgrade_reason": "" if _safe_bool(role_plan.get("needs_lipsync"), False) else "no_active_scene_lyric_timing",
        "identity_priority": _safe_str(role_plan.get("identity_priority")),
        "variety_priority": _safe_str(role_plan.get("variety_priority")),
        "dance_priority": _safe_str(role_plan.get("dance_priority")),
        "suggested_shot_type": _safe_str(role_plan.get("suggested_shot_type")),
        "scene_intent": _scene_intent(section, energy, lyrics, role, role_plan),
        "image_prompt": _image_prompt_for_scene(brief, scene, section, energy, mood, role_plan),
        "video_prompt": _video_prompt_for_scene(brief, scene, section, energy, mood, camera_motion, role_plan),
        "timestamped_video_prompt": _timestamped_video_prompt_for_scene(brief, scene, section, energy, camera_motion, role_plan),
        "camera_motion": camera_motion,
        "character_action": _sentence(action),
        "choreography": _sentence(action),
        "emotion": _clean_text(mood),
        "mood": _clean_text(mood),
        "negative_prompt": _negative_prompt_for_role(role_plan),
    }
    # Normal prompt-plan scenes should be one coherent setup. Location lists are a pool.
    for _key in ("scene_intent", "image_prompt", "video_prompt", "timestamped_video_prompt"):
        if _key in out:
            out[_key] = _sanitize_single_frame_prompt(out.get(_key), brief=brief, item=out, allow_montage=False, start_or_end=True if _key == "image_prompt" else False)
    if not _safe_bool(out.get("needs_lipsync"), False):
        for _key in ("scene_intent", "image_prompt", "video_prompt", "timestamped_video_prompt", "camera_motion", "character_action", "choreography"):
            out[_key] = _strip_non_lipsync_vocal_language(out.get(_key))
        out["negative_prompt"] = _negative_prompt_for_role(role_plan)
    out["negative_prompt"] = _negative_prompt_with_no_visible_text(_join_parts([out.get("negative_prompt"), _anti_collage_negative_terms()]), max_len=1200)
    return out


def _write_prompt_plan_text(path: Path, prompt_plan: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("FrameVision Music Clip Creator - Prompt Plan")
    lines.append(f"Created: {_safe_str(prompt_plan.get('created_at'))}")
    lines.append(f"Backend: {_safe_str(prompt_plan.get('prompt_backend'))}")
    lines.append("")
    for scene in _as_list(prompt_plan.get("scenes")):
        if not isinstance(scene, dict):
            continue
        lines.append(f"[{_safe_str(scene.get('id'))}] {_safe_float(scene.get('song_start'), 0.0):.2f}s - {_safe_float(scene.get('song_end'), 0.0):.2f}s")
        lines.append(f"Role: {_clean_text(scene.get('scene_role'))} | lipsync={bool(scene.get('needs_lipsync'))} | vocal={_clean_text(scene.get('vocal_presence'))} | shot={_clean_text(scene.get('suggested_shot_type'))}")
        block_id = _clean_text(scene.get("lyric_block_id"))
        if block_id:
            lines.append(f"Lyric block: {block_id} {_safe_int(scene.get('lyric_block_scene_index'), 0)}/{_safe_int(scene.get('lyric_block_scene_count'), 0)}")
        lyr = _clean_text(scene.get("lyrics") or scene.get("lyric_text"))
        if lyr:
            lines.append(f"Lyrics: {lyr}")
        lines.append(f"Image: {_clean_text(scene.get('image_prompt'))}")
        lines.append(f"Video: {_clean_text(scene.get('video_prompt'))}")
        lines.append(f"LTX: {_clean_text(scene.get('timestamped_video_prompt'))}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def create_prompt_plan(payload: dict) -> dict:
    """Create musicclip_prompt_plan.json from a Chunk 4A scene plan.

    This is intentionally rule-based for Chunk 4B/4C. It does not import Planner,
    does not call llama-server, does not run LTX, and never generates media.
    """
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Prompt-plan payload was not a dictionary."}

        scene_plan_path = _safe_str(payload.get("scene_plan_path"))
        scene_plan_obj = payload.get("scene_plan")
        if isinstance(scene_plan_obj, dict):
            scene_plan = scene_plan_obj
        elif scene_plan_path:
            scene_plan = _read_json_file(scene_plan_path)
        else:
            return {"ok": False, "message": "No scene_plan_path was provided."}

        scenes_in = [x for x in _as_list(scene_plan.get("scenes")) if isinstance(x, dict)]
        if not scenes_in:
            return {"ok": False, "message": "Scene plan did not contain any scenes."}

        # Payload brief can override/refresh the brief from the saved scene plan.
        brief = _normalize_creative_brief(scene_plan.get("creative_brief"))
        override_brief = _normalize_creative_brief(payload.get("creative_brief"))
        for key, value in override_brief.items():
            if _clean_text(value):
                brief[key] = value

        if scene_plan_path:
            out_dir = Path(scene_plan_path).resolve().parent
        else:
            out_raw = _safe_str(payload.get("output_dir") or scene_plan.get("bridge_output_dir"))
            out_dir = Path(out_raw).resolve() if out_raw else _default_ltx_videoclip_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        use_vocal_roles = _safe_bool(payload.get("use_vocal_scene_roles", payload.get("use_vocal_nonvocal_scene_roles", True)), True)
        bridge_generation_settings = _normalize_bridge_generation_settings(payload, scene_plan)
        duration_profile = _duration_profile_from_bridge_settings(bridge_generation_settings)
        microclip_profile = _microclip_profile_from_bridge_settings(bridge_generation_settings, duration_profile)
        effects_profile = _effects_profile_from_bridge_settings(bridge_generation_settings)
        character_reference = _character_reference_from_sources(payload, scene_plan)

        cleaned_scenes: List[Dict[str, Any]] = []
        for i, scene in enumerate(scenes_in, start=1):
            sc = dict(scene)
            sc["index"] = _safe_int(sc.get("index"), i) or i
            sc["id"] = _safe_str(sc.get("id")) or f"S{i:02d}"
            # Keep IDs clean and sequential for downstream Planner/LTX testing.
            sc["index"] = i
            sc["id"] = f"S{i:02d}"
            cleaned_scenes.append(sc)
        cleaned_scenes = _annotate_lyric_blocks(cleaned_scenes) if use_vocal_roles else cleaned_scenes

        character_pack = _create_llm_character_bible_if_selected(
            payload=payload,
            brief=brief,
            scenes=cleaned_scenes,
            source_plan=scene_plan,
            progress_callback=payload.get("progress_callback") if callable(payload.get("progress_callback")) else None,
        )
        character_bible = character_pack.get("character_bible") or []
        group_bibles = character_pack.get("group_bibles") or []
        character_bible, group_bibles = _apply_character_reference_to_bibles(character_bible, group_bibles, character_reference)
        shot_assignment_guidance = _safe_str(character_pack.get("shot_assignment_guidance"))
        character_bible_backend = _safe_str(character_pack.get("character_bible_backend"))
        character_bible_warnings = _as_list(character_pack.get("character_bible_warnings"))

        prompt_scenes: List[Dict[str, Any]] = []
        for sc in cleaned_scenes:
            prompt_scene = _prompt_scene_from_plan_scene(sc, brief, use_vocal_roles=use_vocal_roles)
            prompt_scene = _apply_character_identity_to_item(prompt_scene, brief, character_bible, group_bibles, include_director_fields=False)
            prompt_scene = _apply_character_reference_to_item(prompt_scene, brief, character_reference, image_prompt_keys=["image_prompt"], passed_to_model=False)
            prompt_scenes.append(prompt_scene)

        montage_policy = _montage_policy_from_plan(scene_plan, payload, {"effects_profile": effects_profile})
        prompt_scenes, montage_policy, montage_warnings = _apply_montage_policy_to_items(
            prompt_scenes,
            brief=brief,
            policy=montage_policy,
            prompt_keys=["scene_intent", "image_prompt", "video_prompt", "timestamped_video_prompt"],
        )
        prompt_scenes = _apply_effect_policy_metadata(prompt_scenes, effects_profile=effects_profile)

        prompt_plan = {
            "version": _VERSION,
            "source": _SOURCE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "scene_plan_path": scene_plan_path or _safe_str(scene_plan.get("scene_plan_path")),
            "audio_path": _safe_str(scene_plan.get("audio_path")),
            "srt_path": _safe_str(scene_plan.get("srt_path")),
            "creative_brief": brief,
            "character_bible": character_bible,
            "group_bibles": group_bibles,
            "shot_assignment_guidance": shot_assignment_guidance,
            "character_bible_backend": character_bible_backend,
            "character_bible_warnings": character_bible_warnings,
            "character_reference": {k: v for k, v in character_reference.items() if k != "warnings"},
            "character_reference_warnings": _plan_character_reference_warnings(character_reference, passed_to_model=False),
            "prompt_brain": _PROMPT_BACKEND_TEMPLATE,
            "prompt_backend": _PROMPT_BACKEND_TEMPLATE,
            "use_vocal_scene_roles": bool(use_vocal_roles),
            "bridge_generation_settings": bridge_generation_settings,
            "duration_profile": duration_profile,
            "microclip_profile": microclip_profile,
            "effects_profile": effects_profile,
            "montage_policy": montage_policy,
            "montage_warnings": montage_warnings[:40],
            "location_pool": _location_candidates_from_brief(brief),
            "scene_count": len(prompt_scenes),
            "scenes": prompt_scenes,
        }

        prompt_path = out_dir / "musicclip_prompt_plan.json"
        _write_json_file(prompt_path, prompt_plan)

        txt_path = out_dir / "musicclip_prompt_plan.txt"
        try:
            _write_prompt_plan_text(txt_path, prompt_plan)
        except Exception:
            txt_path = Path("")

        result = {
            "ok": True,
            "prompt_plan_path": str(prompt_path),
            "output_dir": str(out_dir),
            "scene_count": len(prompt_scenes),
            "message": "Prompt plan created.",
            "prompt_backend": _PROMPT_BACKEND_TEMPLATE,
        }
        if str(txt_path):
            result["prompt_plan_text_path"] = str(txt_path)
        return result
    except Exception as exc:
        return {"ok": False, "message": f"Prompt plan creation failed: {exc}"}

# -----------------------------
# Chunk 5A: prompt plan -> smaller LTX shot plan + optional audio chunks
# -----------------------------

def _scene_start(scene: Dict[str, Any]) -> float:
    return _safe_float(scene.get("song_start", scene.get("start", 0.0)), 0.0)


def _scene_end(scene: Dict[str, Any]) -> float:
    start = _scene_start(scene)
    end = _safe_float(scene.get("song_end", scene.get("end", 0.0)), 0.0)
    dur = _safe_float(scene.get("duration"), 0.0)
    if end <= start and dur > 0.0:
        end = start + dur
    return end


def _scene_duration(scene: Dict[str, Any]) -> float:
    dur = _safe_float(scene.get("duration"), 0.0)
    if dur <= 0.0:
        dur = max(0.0, _scene_end(scene) - _scene_start(scene))
    return dur


def _scene_id(scene: Dict[str, Any], fallback_index: int = 1) -> str:
    sid = _safe_str(scene.get("id"))
    if sid:
        return sid
    idx = _safe_int(scene.get("index"), fallback_index)
    return f"S{max(1, idx):02d}"


def _dedupe_texts(values: List[Any], *, max_items: int = 8, max_len: int = 900) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= max_items:
            break
    joined_len = 0
    trimmed: List[str] = []
    for item in out:
        if max_len and joined_len + len(item) > max_len:
            remaining = max(0, max_len - joined_len - 3)
            if remaining > 20:
                trimmed.append(item[:remaining].rstrip(" ,.;") + "...")
            break
        trimmed.append(item)
        joined_len += len(item) + 3
    return trimmed


def _summarize_unique(values: List[Any], fallback: str = "") -> str:
    texts = _dedupe_texts(values, max_items=6, max_len=500)
    if not texts:
        return fallback
    return "; ".join(texts)


def _scene_lyric_group_key(scene: Dict[str, Any]) -> str:
    block = _clean_text(scene.get("lyric_block_id"))
    if block:
        return f"block:{block.lower()}"
    lyr = _lyric_key(scene.get("lyrics") or scene.get("lyric_text"))
    if lyr:
        return f"lyrics:{lyr[:120]}"
    return ""


def _scene_has_strong_vocal(scene: Dict[str, Any]) -> bool:
    # For LTX grouping, only explicit active lipsync scenes should make a shot
    # vocal. Lyric metadata or repeated long SRT text is not enough.
    if _safe_bool(scene.get("needs_lipsync"), False):
        return True
    if _safe_bool(scene.get("active_vocal_window"), False):
        vocal = _clean_text(scene.get("vocal_presence")).lower()
        return vocal in {"medium", "strong", "lead", "yes", "true"}
    return False


def _normalize_ltx_lyric_segments(lyric_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return clean lyric timing windows for LTX boundary decisions.

    Long SRT blocks are treated as metadata with a short protected active window,
    matching the existing false-lipsync guard. This prevents one bad 20-30s SRT
    line from becoming one giant uncuttable LTX singing shot.
    """
    out: List[Dict[str, Any]] = []
    for i, raw in enumerate(lyric_segments or []):
        if not isinstance(raw, dict):
            continue
        start = _safe_float(raw.get("start", raw.get("song_start", 0.0)), 0.0)
        end = _safe_float(raw.get("end", raw.get("song_end", 0.0)), 0.0)
        if end <= start:
            continue
        text = _clean_text(raw.get("text") or raw.get("lyrics") or raw.get("lyric_text"), 700)
        if not text:
            continue
        duration = max(0.0, end - start)
        long_block = duration > float(LONG_LYRIC_SEGMENT_SECONDS)
        active_end = end
        if long_block:
            active_end = min(end, start + float(MAX_LIPSYNC_SECONDS_PER_LONG_SEGMENT))
        key = _clean_text(raw.get("block_id") or raw.get("lyric_block_id") or raw.get("id"))
        if not key:
            key = _lyric_key(text)[:120] or f"seg_{i + 1:03d}"
        out.append({
            "index": i + 1,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(duration, 3),
            "active_start": round(start, 3),
            "active_end": round(max(start, active_end), 3),
            "long_block": bool(long_block),
            "key": key,
            "text": text,
        })
    out.sort(key=lambda x: (_safe_float(x.get("start"), 0.0), _safe_float(x.get("end"), 0.0)))
    return out


def _lyric_active_window(seg: Dict[str, Any]) -> tuple:
    return (_safe_float(seg.get("active_start", seg.get("start", 0.0)), 0.0), _safe_float(seg.get("active_end", seg.get("end", 0.0)), 0.0))


def _lyric_boundary_inside_active_segment(boundary: float, lyric_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    t = float(boundary or 0.0)
    eps = float(LYRIC_BOUNDARY_EPS_SECONDS)
    for seg in _normalize_ltx_lyric_segments(lyric_segments):
        a, b = _lyric_active_window(seg)
        if b <= a:
            continue
        if (a + eps) < t < (b - eps):
            return seg
    return {}


def _lyric_segments_overlapping_range(start: float, end: float, lyric_segments: List[Dict[str, Any]], *, active_only: bool = True) -> List[Dict[str, Any]]:
    a = float(start or 0.0)
    b = float(end or 0.0)
    out: List[Dict[str, Any]] = []
    for seg in _normalize_ltx_lyric_segments(lyric_segments):
        s0, s1 = _lyric_active_window(seg) if active_only else (_safe_float(seg.get("start"), 0.0), _safe_float(seg.get("end"), 0.0))
        if s1 <= s0:
            continue
        if max(a, s0) < min(b, s1):
            out.append(seg)
    return out


def _group_boundary_splits_active_lyric(prev_group: List[Dict[str, Any]], next_scene: Dict[str, Any], lyric_segments: List[Dict[str, Any]]) -> bool:
    if not prev_group or not isinstance(next_scene, dict):
        return False
    # Only protect real/active vocal areas. Non-vocal b-roll should not be made
    # into forced lipsync just because stale lyric metadata overlaps it.
    if not (any(_scene_has_strong_vocal(s) for s in prev_group) or _scene_has_strong_vocal(next_scene)):
        return False
    boundary = _scene_start(next_scene)
    seg = _lyric_boundary_inside_active_segment(boundary, lyric_segments)
    if not seg:
        return False
    # If the next scene begins inside the same clear lyric phrase, splitting here
    # makes LTX start in the middle of a word/sentence.
    return True


def _shot_lyric_start_mode(shot: Dict[str, Any]) -> str:
    status = _clean_text(shot.get("lyric_boundary_status")).lower()
    if status == "risky_mid_phrase":
        return "mid_phrase"
    if not _safe_bool(shot.get("needs_lipsync"), False):
        return "non_vocal"
    starts = []
    for item in _as_list(shot.get("clip_relative_lyrics")):
        if isinstance(item, dict):
            starts.append(_safe_float(item.get("start"), 0.0))
    if starts:
        first = min(starts)
        if first > 0.45:
            return "delayed"
        if first <= 0.12:
            return "immediate"
    return "immediate"


def _apply_ltx_lyric_boundary_guard(shots: List[Dict[str, Any]], lyric_segments: List[Dict[str, Any]], fps: float, duration_profile: Optional[Dict[str, Any]] = None) -> List[str]:
    """Mark/safely pad LTX shot boundaries against active lyric phrases.

    This deliberately avoids overlap between shots and keeps the existing hard max.
    If a safe snap is impossible, the shot is kept timeline-safe and marked risky.
    """
    warnings: List[str] = []
    if not shots:
        return warnings
    segs = _normalize_ltx_lyric_segments(lyric_segments)
    if not segs:
        for shot in shots:
            shot["lyric_boundary_status"] = "non_vocal" if not _safe_bool(shot.get("needs_lipsync"), False) else "safe"
            shot["lyric_boundary_warnings"] = []
        return warnings

    target_fps = float(fps or 24.0)
    if target_fps <= 0.0:
        target_fps = 24.0
    song_end = max([_safe_float(x.get("song_end"), 0.0) for x in shots] + [_safe_float(x.get("end"), 0.0) for x in segs] + [0.0])

    for idx, shot in enumerate(shots):
        needs_lipsync = _safe_bool(shot.get("needs_lipsync"), False)
        start = _safe_float(shot.get("song_start"), 0.0)
        end = _safe_float(shot.get("song_end"), start + _safe_float(shot.get("duration"), 0.0))
        if end <= start:
            end = start + max(0.05, _safe_float(shot.get("duration"), 0.05))
        prev_end = _safe_float(shots[idx - 1].get("song_end"), 0.0) if idx > 0 else 0.0
        next_start = _safe_float(shots[idx + 1].get("song_start"), song_end) if idx + 1 < len(shots) else song_end
        shot_warnings: List[str] = []
        status = "non_vocal" if not needs_lipsync else "safe"
        original_start, original_end = start, end

        if needs_lipsync:
            tol = float(LYRIC_BOUNDARY_SNAP_TOLERANCE_SECONDS)
            start_pad = float(LYRIC_START_PAD_SECONDS)
            end_pad = float(LYRIC_END_PAD_SECONDS)
            profile = duration_profile if isinstance(duration_profile, dict) else _duration_profile_from_bridge_settings({})
            hard_max = max(_safe_float(profile.get("hard_max_ltx_shot_seconds"), LTX_HARD_MAX_SHOT_SECONDS), _safe_float(profile.get("max_ltx_shot_seconds"), LTX_SOFT_MAX_SHOT_SECONDS))

            # If the shot starts inside a protected lyric phrase, try to move it
            # to just before the phrase start without overlapping the previous shot.
            start_seg = _lyric_boundary_inside_active_segment(start, segs)
            if start_seg:
                seg_start = _safe_float(start_seg.get("active_start"), _safe_float(start_seg.get("start"), start))
                candidate = max(0.0, seg_start - start_pad)
                if (start - candidate) <= tol and candidate >= prev_end - 0.001 and (end - candidate) <= hard_max + 0.001:
                    start = candidate
                    status = "snapped"
                    shot_warnings.append(f"Start snapped before lyric phrase by {start - original_start:+.3f}s.")
                else:
                    status = "risky_mid_phrase"
                    shot_warnings.append("Shot starts inside an active lyric phrase; safe snap was not possible without overlap or exceeding max shot length.")

            # If the shot ends inside a protected lyric phrase, try to extend it
            # until just after that phrase without overlapping the next shot.
            end_seg = _lyric_boundary_inside_active_segment(end, segs)
            if end_seg:
                seg_end = _safe_float(end_seg.get("active_end"), _safe_float(end_seg.get("end"), end))
                candidate = min(song_end, seg_end + end_pad)
                if (candidate - end) <= tol and candidate <= next_start + 0.001 and (candidate - start) <= hard_max + 0.001:
                    end = candidate
                    status = "snapped" if status != "risky_mid_phrase" else status
                    shot_warnings.append(f"End snapped after lyric phrase by {end - original_end:+.3f}s.")
                else:
                    status = "risky_mid_phrase"
                    shot_warnings.append("Shot ends inside an active lyric phrase; safe snap was not possible without overlap or exceeding max shot length.")

            # Near-boundary padding when there is room. This gives LTX a tiny lead-in
            # before a vocal phrase and a tiny tail after it, without creating drift.
            active = _lyric_segments_overlapping_range(start, end, segs, active_only=True)
            if active:
                first = active[0]
                last = active[-1]
                first_start = _safe_float(first.get("active_start"), _safe_float(first.get("start"), start))
                last_end = _safe_float(last.get("active_end"), _safe_float(last.get("end"), end))
                candidate_start = max(0.0, first_start - start_pad)
                if 0.0 <= (start - candidate_start) <= tol and candidate_start >= prev_end - 0.001 and (end - candidate_start) <= hard_max + 0.001:
                    if candidate_start < start - 0.001:
                        start = candidate_start
                        status = "snapped" if status == "safe" else status
                        shot_warnings.append(f"Added lyric lead-in padding: {original_start - start:+.3f}s earlier than original start.")
                candidate_end = min(song_end, last_end + end_pad)
                if 0.0 <= (candidate_end - end) <= tol and candidate_end <= next_start + 0.001 and (candidate_end - start) <= hard_max + 0.001:
                    if candidate_end > end + 0.001:
                        end = candidate_end
                        status = "snapped" if status == "safe" else status
                        shot_warnings.append(f"Added lyric tail padding: {end - original_end:+.3f}s later than original end.")

            if any(_safe_bool(seg.get("long_block"), False) for seg in active):
                shot_warnings.append("Long SRT block detected; only the early active vocal window was protected, not the full lyric block.")

        duration = max(0.05, end - start)
        target_frames = max(1, int(round(duration * target_fps)))
        duration = target_frames / target_fps
        end = start + duration
        shot["song_start"] = round(start, 3)
        shot["song_end"] = round(end, 3)
        shot["duration"] = round(duration, 3)
        shot["target_fps"] = target_fps
        shot["target_frames"] = target_frames
        shot["clip_relative_lyrics"] = _make_clip_relative_lyrics(segs, start, end)
        shot["lyric_boundary_status"] = status
        shot["lyric_boundary_warnings"] = shot_warnings
        if status == "risky_mid_phrase":
            warnings.append(f"{_safe_str(shot.get('id'))}: risky lyric boundary; " + " ".join(shot_warnings))
        elif shot_warnings:
            warnings.append(f"{_safe_str(shot.get('id'))}: " + " ".join(shot_warnings))
    return warnings


def _scene_is_impact(scene: Dict[str, Any]) -> bool:
    blob = " ".join([
        _clean_text(scene.get("section")),
        _clean_text(scene.get("energy")),
        _clean_text(scene.get("scene_role")),
        _clean_text(scene.get("preferred_clip_role")),
        _clean_text(scene.get("suggested_shot_type")),
    ]).lower()
    return any(x in blob for x in ("drop", "impact", "chorus", "peak", "hit"))


def _normalize_ltx_grouping_mode(value: Any) -> str:
    text = _clean_text(value).lower()
    if text.startswith("no") or "one ltx" in text or "per scene" in text:
        return "No grouping"
    return "Auto"


def _group_prompt_scenes_for_ltx(scenes: List[Dict[str, Any]], grouping_mode: str = "Auto", lyric_segments: Optional[List[Dict[str, Any]]] = None, *, duration_profile: Optional[Dict[str, Any]] = None) -> List[List[Dict[str, Any]]]:
    clean = [dict(s) for s in scenes if isinstance(s, dict)]
    clean.sort(key=lambda s: (_scene_start(s), _safe_int(s.get("index"), 0)))
    if not clean:
        return []
    if _normalize_ltx_grouping_mode(grouping_mode) == "No grouping":
        return [[s] for s in clean]

    profile = duration_profile if isinstance(duration_profile, dict) else _duration_profile_from_bridge_settings({})
    target = max(0.5, _safe_float(profile.get("target_clip_seconds"), 7.0))
    target_min = max(0.75, min(target * 0.72, target - 0.25 if target > 1.0 else target))
    soft_max = max(0.75, _safe_float(profile.get("max_ltx_shot_seconds"), LTX_SOFT_MAX_SHOT_SECONDS))
    hard_max = max(soft_max, _safe_float(profile.get("hard_max_ltx_shot_seconds"), LTX_HARD_MAX_SHOT_SECONDS))
    lyric_segments = lyric_segments or []
    groups: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []

    def _cur_duration(items: List[Dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return max(0.0, _scene_end(items[-1]) - _scene_start(items[0]))

    def _same_vocal_phrase(a: List[Dict[str, Any]], b: Dict[str, Any]) -> bool:
        if not a or not _scene_has_strong_vocal(b):
            return False
        if not any(_scene_has_strong_vocal(x) for x in a):
            return False
        key_b = _scene_lyric_group_key(b)
        if not key_b:
            return False
        return key_b == _scene_lyric_group_key(a[-1])

    for scene in clean:
        if not cur:
            cur = [scene]
            continue

        cur_dur = _cur_duration(cur)
        proposed = max(0.0, _scene_end(scene) - _scene_start(cur[0]))
        same_phrase = _same_vocal_phrase(cur, scene)
        boundary_splits_lyric = _group_boundary_splits_active_lyric(cur, scene, lyric_segments)
        section_changed = _clean_text(cur[-1].get("section")).lower() != _clean_text(scene.get("section")).lower()
        next_impact = _scene_is_impact(scene)

        should_split = False
        if _safe_bool(profile.get("strict_generation_frame_cap"), False):
            lyric_overflow_limit = hard_max
        else:
            lyric_overflow_limit = hard_max + (0.35 if profile.get("source") == "manual" else 0.75)
        if proposed > hard_max:
            if (boundary_splits_lyric or same_phrase) and proposed <= lyric_overflow_limit:
                should_split = False
            else:
                should_split = True
        elif cur_dur < target_min:
            should_split = False
        elif boundary_splits_lyric and proposed <= hard_max:
            # Do not start the next LTX shot in the middle of an active lyric phrase.
            should_split = False
        elif same_phrase and proposed <= hard_max:
            # Do not chop a short vocal phrase too aggressively, but still keep it testable.
            should_split = False
        elif next_impact and cur_dur >= target_min:
            should_split = True
        elif section_changed and cur_dur >= target_min:
            should_split = True
        elif proposed <= soft_max:
            should_split = False
        else:
            should_split = True

        if should_split:
            groups.append(cur)
            cur = [scene]
        else:
            cur.append(scene)

    if cur:
        groups.append(cur)
    return groups


def _make_clip_relative_lyrics(lyric_segments: List[Dict[str, Any]], shot_start: float, shot_end: float, *, active_only: bool = True) -> List[Dict[str, Any]]:
    duration = max(0.0, shot_end - shot_start)
    out: List[Dict[str, Any]] = []
    for raw in lyric_segments:
        if not isinstance(raw, dict):
            continue
        if active_only:
            start = _safe_float(raw.get("active_start", raw.get("start", 0.0)), 0.0)
            end = _safe_float(raw.get("active_end", raw.get("end", 0.0)), 0.0)
        else:
            start = _safe_float(raw.get("start"), 0.0)
            end = _safe_float(raw.get("end"), 0.0)
        if end <= start:
            continue
        if max(shot_start, start) >= min(shot_end, end):
            continue
        text = _clean_text(raw.get("text"), 600)
        if not text:
            continue
        out.append({
            "start": round(max(0.0, start - shot_start), 3),
            "end": round(min(duration, end - shot_start), 3),
            "text": text,
            "source_start": round(start, 3),
            "source_end": round(end, 3),
            "long_block_active_window": bool(raw.get("long_block")),
        })
    out.sort(key=lambda x: (_safe_float(x.get("start"), 0.0), _safe_float(x.get("end"), 0.0)))
    return out


def _ltx_time_label(seconds: float) -> str:
    try:
        total = max(0, int(round(float(seconds))))
    except Exception:
        total = 0
    minutes = total // 60
    sec = total % 60
    return f"{minutes}:{sec:02d}"


def _ltx_beat_times(duration: float) -> List[float]:
    dur = max(0.1, float(duration or 0.0))
    if dur <= 3.0:
        pts = [0.0, min(2.0, dur)]
    elif dur <= 6.0:
        pts = [0.0, 2.0, min(4.0, dur)]
    else:
        pts = [0.0]
        t = 3.0
        while t < dur - 0.75 and len(pts) < 7:
            pts.append(t)
            t += 3.0
        final = max(0.0, dur - 0.4)
        if final - pts[-1] >= 2.0 and len(pts) < 7:
            pts.append(final)
    out: List[float] = []
    for value in pts:
        v = round(max(0.0, min(dur, value)), 2)
        if not out or abs(v - out[-1]) > 0.2:
            out.append(v)
    return out or [0.0]


def _major_scene_block_limit(duration: float) -> int:
    """Fallback cap for major location/setup changes inside one LTX clip.

    This is deliberately about major scene blocks, not normal camera/detail variation.
    A 7s shot should not be forced into an inside/outside/new-location mini montage.
    """
    dur = max(0.0, _safe_float(duration, 0.0))
    if dur < 9.0:
        return 1
    if dur < 15.0:
        return 2
    if dur < 21.0:
        return 3
    return 4


def _scene_block_guidance(duration: float) -> str:
    cap = _major_scene_block_limit(duration)
    if cap <= 1:
        return "Keep one dominant setup/location. Use camera movement, reframing, reflections, details, and performance/action variation instead of major location changes."
    if cap == 2:
        return "Allow up to two major scene blocks only if the story naturally needs it; otherwise keep one dominant setup with camera/detail variation."
    if cap == 3:
        return "Allow up to three major scene blocks only if the story naturally needs progression; do not cram extra locations into the clip."
    return "Allow up to four major scene blocks for long clips only when the story benefits; normal angle/detail changes do not count as major blocks."


def _same_setup_variation_pool(brief: Dict[str, str], needs_lipsync: bool, base_setup: str = "") -> List[str]:
    base = _clean_text(base_setup, 180) or ("face-readable performance setup" if needs_lipsync else "main music-video action setup")
    if needs_lipsync:
        return [
            f"{base}, same setup, face-readable framing",
            "side/profile angle in the same setup, background motion and light changes",
            "closer face-and-hands detail in the same setup, rhythmic body movement",
            "windshield/reflection/light variation while the same setup continues",
            "slight push-in or pull-back without changing location",
            "final same-location angle resolving toward the next cut",
        ]
    return [
        f"{base}, same setup, strong beat-driven motion",
        "side angle in the same setup with passing scenery and light changes",
        "detail close-up in the same setup: reflections, hands, wheels, dashboard, fabric, or environment texture",
        "camera push-in or pull-back while the same location continues",
        "same-location action variation with rhythm, speed, or dance energy",
        "final same-setup hero angle resolving toward the next cut",
    ]


def _limited_scene_pool_for_ltx(brief: Dict[str, str], scene_shots: List[str], duration: float, needs_lipsync: bool) -> List[str]:
    """Build timestamp beat material without forcing too many major scene blocks."""
    clean_shots = [_clean_text(x, 180) for x in (scene_shots or []) if _clean_text(x)]
    cap = _major_scene_block_limit(duration)
    if cap <= 1:
        base = clean_shots[0] if clean_shots else ""
        return _same_setup_variation_pool(brief, needs_lipsync, base)
    allowed_major = clean_shots[:cap] if clean_shots else []
    if not allowed_major:
        allowed_major = _shot_variety_pool_for_ltx(brief, needs_lipsync)[:cap]
    out: List[str] = []
    for idx, item in enumerate(allowed_major):
        if idx == 0:
            out.append(item)
            out.append(f"same setup detail variation around {item}")
        else:
            out.append(f"natural story progression to {item}")
            out.append(f"detail or side angle within {item}")
    if len(out) < 6:
        out.extend(_same_setup_variation_pool(brief, needs_lipsync, allowed_major[0] if allowed_major else ""))
    return out[:8]


def _major_scene_keyword_count(value: Any) -> int:
    """Best-effort guard for LLM outputs that cram in too many major switches.

    This is intentionally conservative. It does not count normal camera movement;
    it looks for language that usually means a new setup/location/block.
    """
    text = _safe_str(value).lower()
    if not text:
        return 0
    patterns = [
        r"\binterior\b",
        r"\bexterior\b",
        r"\bcockpit\b",
        r"\bdashboard\b",
        r"\broadside\b",
        r"\bstage\b",
        r"\bdance floor\b",
        r"\bclub\b",
        r"\bcity street\b",
        r"\bcoastal highway\b",
        r"\bconvoy\b",
        r"\baerial\b",
        r"\bwide exterior\b",
        r"\bcuts? to\b",
        r"\bshifts? to\b",
        r"\bopens? to\b",
        r"\bnew location\b",
        r"\banother location\b",
    ]
    hits = 0
    for pat in patterns:
        hits += len(re.findall(pat, text, flags=re.IGNORECASE))
    return hits


def _apply_scene_block_limiter_to_director_shot(shot: Dict[str, Any], brief: Dict[str, str]) -> Dict[str, Any]:
    """Fallback safety guard after template/LLM director shaping.

    If the timestamped director prompt appears to include too many major blocks for
    the duration, replace the timestamped line with the stable template. Image
    prompts remain focused on a single starting frame.
    """
    out = dict(shot or {})
    duration = _safe_float(out.get("duration"), 0.0)
    cap = _major_scene_block_limit(duration)
    out["major_scene_block_limit"] = cap
    out["major_scene_block_rule"] = _scene_block_guidance(duration)
    ts = _safe_str(out.get("director_timestamped_video_prompt") or out.get("timestamped_video_prompt"))
    hits = _major_scene_keyword_count(ts)
    out["major_scene_block_keyword_hits"] = hits
    # The keyword count is only a fallback. Allow a little slack because one scene
    # can mention both cockpit and dashboard without being a true new location.
    if hits > max(cap + 2, cap * 2):
        out["director_timestamped_video_prompt_before_scene_limiter"] = ts
        out["director_timestamped_video_prompt"] = _director_template_timestamp(out, brief)
        notes = _safe_str(out.get("director_notes"))
        limiter_note = f"Scene-block limiter replaced an overloaded timestamped prompt; limit={cap}."
        out["director_notes"] = _clean_text(_join_parts([notes, limiter_note]), 900)
    return out



def _shot_variety_pool_for_ltx(brief: Dict[str, str], needs_lipsync: bool) -> List[str]:
    if needs_lipsync:
        pool = [
            "close-up singing to camera with the face and mouth clearly visible",
            "side profile singing shot while the background moves past",
            "dance-performance angle with the lead singer still readable",
            "inside vehicle or cockpit singing shot with face-aware framing",
            "wide hero performance angle with dancers and the singer still visible",
            "over-shoulder performance shot that returns to the singer's face",
        ]
        if not _brief_has_vehicle_world(brief):
            pool = [x for x in pool if "vehicle" not in x and "cockpit" not in x]
        return pool
    if _brief_has_vehicle_world(brief):
        return [
            "futuristic supercars launch instantly along the road",
            "wheel-level tracking shot with sunlight flashing across chrome bodywork",
            "cars drift around a wide curve while the world streaks past",
            "dashboard, lights, and bodywork details pulse with the beat",
            "wide aerial reveal of the road leading toward the next location",
            "hero racing angle with strong speed and clean motion",
        ]
    return [
        "the main action starts immediately with a strong visual hook",
        "the camera drops closer to rhythmic details in the scene",
        "dancers or background movement build energy without forcing lipsync",
        "the shot widens to reveal more of the world",
        "a fast action beat lands cleanly with the music",
        "the camera resolves into a clear next-cut position",
    ]


def _timestamped_video_prompt_for_ltx_group(brief: Dict[str, str], group: List[Dict[str, Any]], duration: float, needs_lipsync: bool) -> str:
    subject = _lead_subject(brief)
    world = _single_location_world_for_scene(brief, group[0] if group else {}, _safe_int((group[0] if group else {}).get("index"), 1))
    style = _style_text(brief)
    active_scenes = [s for s in group if _scene_has_strong_vocal(s)]
    scene_shots = _dedupe_texts([s.get("suggested_shot_type") for s in group], max_items=8, max_len=500)
    pool = _limited_scene_pool_for_ltx(brief, scene_shots, duration, needs_lipsync)
    times = _ltx_beat_times(duration)
    beats: List[str] = []
    active_windows: List[tuple] = []
    if active_scenes:
        shot_start = min(_scene_start(s) for s in group)
        for s in active_scenes:
            active_windows.append((max(0.0, _scene_start(s) - shot_start), max(0.0, _scene_end(s) - shot_start)))

    def _is_active_time(t: float) -> bool:
        if not active_windows:
            return False
        for a, b in active_windows:
            if a <= float(t) <= max(a + 0.25, b):
                return True
        return False

    for i, t in enumerate(times):
        shot = pool[i % len(pool)] if pool else "music-video shot"
        active_here = bool(needs_lipsync and _is_active_time(t))
        if active_here:
            if i == 0:
                line = f"{subject} continues the vocal phrase naturally in a {shot}, expressive face-readable movement, {world}, {style}"
            else:
                line = f"The camera catches the active vocal phrase in a {shot}, then keeps the movement natural with the beat"
        else:
            if i == 0:
                line = f"{shot} in {world}, {style}, moving with the beat without mouthing lyrics"
            elif i == 1:
                line = f"The camera pushes into {shot}, stronger speed and music-driven motion"
            elif i == len(times) - 1:
                line = f"The action resolves through {shot}, ready for the next cut"
            else:
                line = f"The non-vocal sequence builds through {shot}, energetic b-roll and world-building"
        if not active_here:
            line = _strip_non_lipsync_vocal_language(line)
        beats.append(f"{_ltx_time_label(t)} - {_clean_text(line).rstrip('.')}.")
    out = re.sub(r"\s+", " ", " ".join(beats).replace("\n", " ")).strip()
    if not needs_lipsync:
        out = _strip_non_lipsync_vocal_language(out)
    return out


_LTX_TIMESTAMP_ENDING_GUARD = "Continue the same action smoothly through the ending without introducing a new location, new action, hard camera reset, or last-second transition."
_LTX_TIMESTAMP_RE = re.compile(
    r"(?<!\w)(?P<label>(?:(?P<minute>\d{1,2})\s*[:.]\s*(?P<second>\d{2})|(?P<onlysec>\d{1,3})\s*s))\s*(?P<sep>[-–—:])\s*",
    re.IGNORECASE,
)
_LTX_TIMESTAMP_PROMPT_KEYS = (
    "timestamped_video_prompt",
    "template_timestamped_video_prompt",
    "director_timestamped_video_prompt",
)


def _ltx_timestamp_seconds_from_match(match: re.Match) -> int:
    try:
        if match.group("onlysec") is not None:
            return max(0, int(match.group("onlysec")))
        return max(0, int(match.group("minute")) * 60 + int(match.group("second")))
    except Exception:
        return 0


def _ltx_allowed_timestamp_seconds(duration: float) -> set:
    dur = max(0.0, _safe_float(duration, 0.0))
    allowed = {0}
    if dur <= 0.0:
        return allowed
    last_allowed = int(max(0.0, dur - 3.0) + 1e-6)
    t = 5
    while t <= last_allowed:
        allowed.add(t)
        t += 5
    return allowed


def _ltx_clean_timestamp_chunk_text(value: Any) -> str:
    text = _clean_text(value, 0)
    text = re.sub(r"\s+", " ", text).strip(" \t,;:-–—")
    return text


def _ltx_merge_timestamp_chunk(existing: str, addition: str) -> str:
    existing = _ltx_clean_timestamp_chunk_text(existing)
    addition = _ltx_clean_timestamp_chunk_text(addition)
    if not addition:
        return existing
    if not existing:
        return addition
    if addition.lower() in existing.lower():
        return existing
    sep = " " if existing[-1:] in ".!?" else ". "
    return (existing + sep + addition).strip()


_LTX_MICROCLIP_TIMESTAMP_RE = re.compile(
    r"(?<!\w)(?P<label>(?:(?P<minute>\d{1,2})\s*:\s*(?P<second>\d{2})(?:\.(?P<fraction>\d{1,3}))?|(?P<onlysec>\d{1,3})(?:\.(?P<onlyfrac>\d{1,3}))?\s*s))\s*(?P<sep>[-–—:])\s*",
    re.IGNORECASE,
)


def _ltx_microclip_timestamp_seconds_from_match(match: re.Match) -> float:
    try:
        if match.group("onlysec") is not None:
            sec = float(match.group("onlysec") or 0)
            frac = _safe_str(match.group("onlyfrac"))
            if frac:
                sec += float("0." + frac)
            return max(0.0, sec)
        sec = float(match.group("second") or 0)
        frac = _safe_str(match.group("fraction"))
        if frac:
            sec += float("0." + frac)
        return max(0.0, float(match.group("minute") or 0) * 60.0 + sec)
    except Exception:
        return 0.0


def _sanitize_ltx_microclip_timestamped_prompt_with_stats(prompt: str, duration: float) -> tuple:
    text = _clean_text(prompt, 0)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    if not text:
        return "", 0

    matches = list(_LTX_MICROCLIP_TIMESTAMP_RE.finditer(text))
    latest_start = max(0.0, _safe_float(duration, 0.0) - _LTX_MICROCLIP_END_HOLD_SECONDS)
    if not matches:
        out = text
        if not re.match(r"^0:00(?:\.0)?\s*-", out):
            out = "0:00 - " + out
        if _LTX_MICROCLIP_ENDING_GUARD.lower() not in out.lower():
            out = _sentence(out + " " + _LTX_MICROCLIP_ENDING_GUARD)
        return out, 0

    kept: List[Dict[str, Any]] = []
    removed_count = 0

    def add_chunk(seconds: float, chunk_text: str) -> None:
        chunk_text = _ltx_clean_timestamp_chunk_text(chunk_text)
        if not chunk_text:
            return
        seconds = max(0.0, round(float(seconds or 0.0), 1))
        if kept and abs(_safe_float(kept[-1].get("seconds"), 0.0) - seconds) < 0.05:
            kept[-1]["text"] = _ltx_merge_timestamp_chunk(_safe_str(kept[-1].get("text")), chunk_text)
            return
        kept.append({"seconds": seconds, "text": chunk_text})

    prefix = _ltx_clean_timestamp_chunk_text(text[: matches[0].start()])
    if prefix:
        add_chunk(0.0, prefix)

    for idx, match in enumerate(matches):
        seconds = _ltx_microclip_timestamp_seconds_from_match(match)
        chunk_start = match.end()
        chunk_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = _ltx_clean_timestamp_chunk_text(text[chunk_start:chunk_end])
        if seconds <= 0.05:
            add_chunk(0.0, chunk)
        elif seconds <= latest_start + 1e-6:
            add_chunk(seconds, chunk)
        else:
            removed_count += 1
            if kept:
                kept[-1]["text"] = _ltx_merge_timestamp_chunk(_safe_str(kept[-1].get("text")), chunk)
            else:
                add_chunk(0.0, chunk)

    if not kept:
        add_chunk(0.0, text)
    if not any(abs(_safe_float(x.get("seconds"), 0.0)) < 0.05 for x in kept):
        kept.insert(0, {"seconds": 0.0, "text": "Immediate intense full-frame microclip opening."})

    ordered: List[Dict[str, Any]] = []
    for item in sorted(kept, key=lambda x: _safe_float(x.get("seconds"), 0.0)):
        seconds = round(_safe_float(item.get("seconds"), 0.0), 1)
        if seconds > latest_start + 1e-6 and seconds > 0.05:
            removed_count += 1
            if ordered:
                ordered[-1]["text"] = _ltx_merge_timestamp_chunk(_safe_str(ordered[-1].get("text")), item.get("text"))
            else:
                ordered.append({"seconds": 0.0, "text": _safe_str(item.get("text"))})
            continue
        if ordered and abs(_safe_float(ordered[-1].get("seconds"), 0.0) - seconds) < 0.05:
            ordered[-1]["text"] = _ltx_merge_timestamp_chunk(_safe_str(ordered[-1].get("text")), item.get("text"))
        else:
            ordered.append({"seconds": seconds, "text": _safe_str(item.get("text"))})

    lines: List[str] = []
    for item in ordered:
        seconds = round(_safe_float(item.get("seconds"), 0.0), 1)
        chunk = _ltx_clean_timestamp_chunk_text(item.get("text"))
        if not chunk:
            continue
        if chunk[-1:] not in ".!?":
            chunk += "."
        lines.append(f"{_ltx_microclip_time_label(seconds)} - {chunk}")
    out = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not out:
        out = "0:00 - Immediate intense full-frame microclip opening."
    if _LTX_MICROCLIP_ENDING_GUARD.lower() not in out.lower():
        out = _sentence(out + " " + _LTX_MICROCLIP_ENDING_GUARD)
    elif out[-1:] not in ".!?":
        out += "."
    return out, removed_count


def _sanitize_ltx_microclip_timestamped_prompt(prompt: str, duration: float) -> str:
    text, _removed_count = _sanitize_ltx_microclip_timestamped_prompt_with_stats(prompt, duration)
    return text


def _sanitize_ltx_timestamped_prompt_with_stats(prompt: str, duration: float) -> tuple:
    text = _clean_text(prompt, 0)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    if not text:
        return "", 0

    matches = list(_LTX_TIMESTAMP_RE.finditer(text))
    if not matches:
        out = text
        if not re.match(r"^0:00\s*-", out):
            out = "0:00 - " + out
        if out[-1:] not in ".!?":
            out += "."
        return out, 0

    allowed = _ltx_allowed_timestamp_seconds(duration)
    kept: List[Dict[str, Any]] = []
    removed_count = 0

    def add_chunk(seconds: int, chunk_text: str) -> None:
        chunk_text = _ltx_clean_timestamp_chunk_text(chunk_text)
        if not chunk_text:
            return
        if kept and _safe_int(kept[-1].get("seconds"), 0) == int(seconds):
            kept[-1]["text"] = _ltx_merge_timestamp_chunk(_safe_str(kept[-1].get("text")), chunk_text)
            return
        kept.append({"seconds": int(seconds), "text": chunk_text})

    prefix = _ltx_clean_timestamp_chunk_text(text[: matches[0].start()])
    if prefix:
        add_chunk(0, prefix)

    for idx, match in enumerate(matches):
        seconds = _ltx_timestamp_seconds_from_match(match)
        chunk_start = match.end()
        chunk_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = _ltx_clean_timestamp_chunk_text(text[chunk_start:chunk_end])
        is_opening = seconds == 0
        is_allowed_interval = bool(seconds in allowed and (seconds == 0 or seconds % 5 == 0))
        if is_opening:
            add_chunk(0, chunk)
        elif is_allowed_interval:
            add_chunk(seconds, chunk)
        else:
            removed_count += 1
            previous_allowed = max([x for x in allowed if x < seconds] or [0])
            if previous_allowed == 0 and not kept:
                add_chunk(0, chunk)
            else:
                add_chunk(previous_allowed, chunk)

    if not kept:
        add_chunk(0, text)

    if not any(_safe_int(x.get("seconds"), 0) == 0 for x in kept):
        kept.insert(0, {"seconds": 0, "text": "The shot starts immediately in the same setup."})

    # Keep readable chronological order, but never create a new timestamp outside the rule.
    ordered: List[Dict[str, Any]] = []
    seen_allowed = set()
    for item in kept:
        seconds = _safe_int(item.get("seconds"), 0)
        if seconds != 0 and seconds not in allowed:
            removed_count += 1
            if ordered:
                ordered[-1]["text"] = _ltx_merge_timestamp_chunk(_safe_str(ordered[-1].get("text")), item.get("text"))
            else:
                ordered.append({"seconds": 0, "text": _safe_str(item.get("text"))})
            continue
        if seconds in seen_allowed:
            if ordered:
                ordered[-1]["text"] = _ltx_merge_timestamp_chunk(_safe_str(ordered[-1].get("text")), item.get("text"))
            continue
        seen_allowed.add(seconds)
        ordered.append({"seconds": seconds, "text": _safe_str(item.get("text"))})

    ordered.sort(key=lambda x: _safe_int(x.get("seconds"), 0))
    lines: List[str] = []
    for item in ordered:
        seconds = _safe_int(item.get("seconds"), 0)
        chunk = _ltx_clean_timestamp_chunk_text(item.get("text"))
        if not chunk:
            continue
        if chunk[-1:] not in ".!?":
            chunk += "."
        lines.append(f"{_ltx_time_label(seconds)} - {chunk}")
    out = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not out:
        out = "0:00 - The shot starts immediately in the same setup."
    if removed_count > 0 and _LTX_TIMESTAMP_ENDING_GUARD.lower() not in out.lower():
        out = _sentence(out + " " + _LTX_TIMESTAMP_ENDING_GUARD)
    elif out[-1:] not in ".!?":
        out += "."
    return out, removed_count


def _sanitize_ltx_timestamped_prompt(prompt: str, duration: float) -> str:
    text, _removed_count = _sanitize_ltx_timestamped_prompt_with_stats(prompt, duration)
    return text


def _sanitize_ltx_timestamp_fields_in_item(item: Dict[str, Any], keys: Optional[List[str]] = None) -> Dict[str, Any]:
    if not isinstance(item, dict):
        return item
    duration = _safe_float(item.get("duration"), 0.0)
    total_removed = 0
    is_microclip = _safe_bool(item.get("is_microclip"), False)
    for key in (keys or list(_LTX_TIMESTAMP_PROMPT_KEYS)):
        if key not in item:
            continue
        value = _safe_str(item.get(key))
        if not value:
            continue
        if is_microclip:
            sanitized, removed = _sanitize_ltx_microclip_timestamped_prompt_with_stats(value, duration)
        else:
            sanitized, removed = _sanitize_ltx_timestamped_prompt_with_stats(value, duration)
        item[key] = sanitized
        total_removed += int(removed or 0)
    if total_removed > 0:
        item["timestamp_sanitizer_applied"] = True
        item["timestamp_sanitizer_removed_count"] = _safe_int(item.get("timestamp_sanitizer_removed_count"), 0) + total_removed
    return item


def _merge_group_image_prompt(brief: Dict[str, str], group: List[Dict[str, Any]], needs_lipsync: bool) -> str:
    first = _clean_text((group[0] if group else {}).get("image_prompt"), 700)
    shots = _summarize_unique([s.get("suggested_shot_type") for s in group], "cinematic still")
    world = _single_location_world_for_scene(brief, group[0] if group else {}, _safe_int((group[0] if group else {}).get("index"), 1))
    if needs_lipsync:
        extra = f"clean opening still with one consistent lead performer identity, expressive face-readable performance pose for the active vocal phrase, {world}, visual variety suggested by {shots}"
    else:
        extra = f"clean opening still focused on one clear non-vocal action or b-roll detail, {world}, visual variety suggested by {shots}, no lyric-mouthing"
    out = _sentence(_sanitize_prompt_no_visible_text(_join_parts([first, extra]), image_prompt=True))
    out, _removed = _director_clean_final_prompt(out, image_prompt=True, max_words=80, add_safety=True) if "_director_clean_final_prompt" in globals() else (out, [])
    return out if needs_lipsync else _strip_non_lipsync_vocal_language(out)


def _merge_group_video_prompt(brief: Dict[str, str], group: List[Dict[str, Any]], needs_lipsync: bool) -> str:
    first = _clean_text((group[0] if group else {}).get("video_prompt"), 700)
    roles = _summarize_unique([s.get("scene_role") for s in group], "motion-focused action")
    shots = _summarize_unique([s.get("suggested_shot_type") for s in group], "varied shot angles")
    if needs_lipsync:
        extra = f"keep the main subject readable during the active vocal window while changing angles and performance energy through {shots}"
    else:
        extra = f"focus on action, b-roll, dancing, and world-building with no lyric-mouthing, moving through {shots}"
    out = _sentence(_join_parts([first, roles, extra]))
    out, _removed = _director_clean_final_prompt(out, image_prompt=False, max_words=70, add_safety=False) if "_director_clean_final_prompt" in globals() else (out, [])
    return out if needs_lipsync else _strip_non_lipsync_vocal_language(out)


def _notes_for_ltx_group(group: List[Dict[str, Any]], grouping_mode: str, audio_warning: str = "") -> str:
    ids = ", ".join(_scene_id(s, i + 1) for i, s in enumerate(group))
    text = f"Chunk 5A grouped source scenes {ids} using {grouping_mode}. Final master audio remains the original full song; audio chunk is only for possible LTX lipsync/audio-guidance testing."
    if audio_warning:
        text += f" Audio chunk warning: {audio_warning}"
    return text


def _find_bridge_ffmpeg(root_dir: Any = "") -> str:
    root = Path(_safe_str(root_dir) or str(_project_root())).resolve()
    candidates = [
        root / "presets" / "bin" / "ffmpeg.exe",
        root / "presets" / "bin" / "ffmpeg",
        Path("ffmpeg.exe"),
        Path("ffmpeg"),
    ]
    for cand in candidates:
        try:
            text = str(cand)
            if cand.is_absolute() and cand.is_file():
                return text
            found = shutil.which(text)
            if found:
                return found
        except Exception:
            continue
    return "ffmpeg"


def _export_ltx_audio_chunk(ffmpeg: str, audio_path: str, out_path: Path, start: float, duration: float) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg or "ffmpeg"),
        "-y",
        "-ss", f"{max(0.0, float(start)):.3f}",
        "-i", str(audio_path),
        "-t", f"{max(0.05, float(duration)):.3f}",
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "48000",
        "-ac", "2",
        str(out_path),
    ]
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creationflags,
            timeout=180,
        )
    except Exception as exc:
        return str(exc)
    if proc.returncode != 0:
        err = _clean_text(proc.stderr or proc.stdout, 500)
        return err or f"ffmpeg exited with code {proc.returncode}"
    if not out_path.is_file():
        return "ffmpeg finished but the WAV file was not created"
    return ""


def _make_ltx_retry_audio_guide(
    *,
    root: Path,
    plan_path: Path,
    director_plan: Dict[str, Any],
    shot: Dict[str, Any],
    out_dir: Path,
    stem: str,
    attempt_index: int,
    desired_generation_duration: float,
    fallback_audio_path: str,
) -> str:
    """Create a longer per-attempt LTX WAV guide for under-delivered shots.

    Some LTX bridge backends obey the audio guide duration more strongly than the
    requested frame count. For old Review jobs this is the missing piece: simply
    asking for more frames can still return the same 2.x-second clip. On retry we
    cut a longer WAV guide from the original master song, while the final assembly
    still uses the full master audio as before.
    """
    fallback_audio_path = _safe_str(fallback_audio_path)
    master_audio = _safe_str(_master_audio_from_plan(director_plan))
    if not master_audio or not os.path.isfile(master_audio):
        master_audio = fallback_audio_path
    if not master_audio or not os.path.isfile(master_audio):
        return fallback_audio_path
    desired = max(0.25, float(desired_generation_duration or 0.0))
    # Give the backend more audio than the timeline slot so it can create real
    # motion and we can trim back to the exact planned duration afterward.
    desired = max(desired, _safe_float(shot.get("duration"), 0.0) + float(LTX_SHORT_SHOT_EXTRA_PAD_SECONDS))
    desired = max(desired, float(LTX_SHORT_SHOT_MIN_REQUEST_SECONDS))
    song_start = _safe_float(shot.get("song_start"), 0.0)
    audio_start = max(0.0, song_start - float(LTX_GENERATION_PRE_PAD_SECONDS))
    try:
        song_duration = _probe_media_duration_seconds(root, Path(master_audio))
    except Exception:
        song_duration = 0.0
    if song_duration > 0.0 and audio_start + desired > song_duration:
        # Last-shot safety: if there is not enough song after the shot start, pull
        # the guide window earlier instead of making a too-short WAV. Final muxing
        # still uses the untouched original song.
        audio_start = max(0.0, song_duration - desired)
        desired = max(0.05, min(desired, song_duration - audio_start))
    retry_dir = Path(out_dir).resolve() / "retry_audio_guides"
    retry_dir.mkdir(parents=True, exist_ok=True)
    out_wav = retry_dir / f"{_safe_stem(stem)}_retry{int(attempt_index)}_{int(round(desired * 1000.0))}ms.wav"
    err = _export_ltx_audio_chunk(_find_bridge_ffmpeg(root), master_audio, out_wav, audio_start, desired)
    if err or not out_wav.is_file():
        return fallback_audio_path
    return str(out_wav)


def _ensure_ltx_audio_guide_covers_generation_duration(
    *,
    root: Path,
    director_plan: Dict[str, Any],
    shot: Dict[str, Any],
    out_dir: Path,
    stem: str,
    attempt_index: int,
    desired_generation_duration: float,
    fallback_audio_path: str,
) -> tuple[str, Dict[str, Any]]:
    """Return an LTX audio guide that is at least as long as the requested frames.

    The raw LTX clip intentionally gets extra generation frames so the Music Clip
    Creator can trim/sync later.  The audio guide must follow that longer raw
    generation duration; otherwise A2V can build a shorter audio latent and crash
    with a latent-shape mismatch before denoise starts.  This function never
    shortens the requested generation.  It only re-cuts/pads the guide WAV when
    the existing guide is too short.
    """
    fallback_audio_path = _safe_str(fallback_audio_path)
    desired = max(0.05, float(desired_generation_duration or 0.0))
    info: Dict[str, Any] = {
        "enabled": bool(fallback_audio_path),
        "desired_generation_duration": round(float(desired), 6),
        "original_audio_guide": fallback_audio_path,
        "original_audio_guide_duration": 0.0,
        "padded": False,
        "reason": "",
        "audio_start": None,
        "audio_duration": None,
        "final_audio_guide": fallback_audio_path,
        "error": "",
    }
    if not fallback_audio_path or not os.path.isfile(fallback_audio_path):
        info["enabled"] = False
        info["reason"] = "no existing audio guide"
        return fallback_audio_path, info

    try:
        current_duration = _probe_media_duration_seconds(root, Path(fallback_audio_path))
    except Exception:
        current_duration = 0.0
    info["original_audio_guide_duration"] = round(float(current_duration), 6)

    # Small tolerance avoids rebuilding for tiny encoder/probe rounding drift.
    if current_duration > 0.0 and current_duration + max(float(DURATION_TOLERANCE_SECONDS), 0.04) >= desired:
        info["reason"] = "existing guide already covers requested generation duration"
        return fallback_audio_path, info

    master_audio = _safe_str(_master_audio_from_plan(director_plan))
    if not master_audio or not os.path.isfile(master_audio):
        master_audio = fallback_audio_path
    if not master_audio or not os.path.isfile(master_audio):
        info["reason"] = "no master audio available for padded guide"
        return fallback_audio_path, info

    song_start = _safe_float(shot.get("song_start"), 0.0)
    audio_start = max(0.0, song_start - float(LTX_GENERATION_PRE_PAD_SECONDS))
    audio_duration = desired
    try:
        song_duration = _probe_media_duration_seconds(root, Path(master_audio))
    except Exception:
        song_duration = 0.0
    if song_duration > 0.0 and audio_start + audio_duration > song_duration:
        # Keep the guide long enough by pulling the window earlier near the end
        # of the song.  Do not shorten the raw LTX generation request.
        audio_start = max(0.0, song_duration - audio_duration)
        audio_duration = max(0.05, min(audio_duration, song_duration - audio_start))

    guide_dir = Path(out_dir).resolve() / "ltx_audio_guides_padded"
    guide_dir.mkdir(parents=True, exist_ok=True)
    out_wav = guide_dir / f"{_safe_stem(stem)}_attempt{int(attempt_index)}_{int(round(audio_duration * 1000.0))}ms.wav"
    err = _export_ltx_audio_chunk(_find_bridge_ffmpeg(root), master_audio, out_wav, audio_start, audio_duration)
    if err or not out_wav.is_file():
        info["reason"] = "failed to create padded guide; using original guide"
        info["error"] = _clean_text(err, 500) if err else "padded guide file was not created"
        return fallback_audio_path, info

    info["padded"] = True
    info["reason"] = "existing guide was shorter than requested generation duration"
    info["audio_start"] = round(float(audio_start), 6)
    info["audio_duration"] = round(float(audio_duration), 6)
    info["final_audio_guide"] = str(out_wav)
    return str(out_wav), info




LTX_EDGE_FIRST_MIN_SECONDS = 3.0
LTX_EDGE_LAST_TARGET_SECONDS = 5.0
LTX_EDGE_DONOR_MIN_SECONDS = 0.25


def _payload_ltx_edge_duration_safety_enabled(payload: Optional[Dict[str, Any]], director_plan: Optional[Dict[str, Any]] = None) -> bool:
    payload = payload if isinstance(payload, dict) else {}
    for key in ("ltx_avoid_short_start_end", "avoid_short_start_end_clips", "start_end_duration_safety", "edge_duration_safety"):
        if key in payload:
            return _safe_bool(payload.get(key), True)
    plan = director_plan if isinstance(director_plan, dict) else {}
    cfg = plan.get("bridge_generation_settings") if isinstance(plan.get("bridge_generation_settings"), dict) else {}
    for key in ("avoid_short_start_end_clips", "ltx_avoid_short_start_end"):
        if key in cfg:
            return _safe_bool(cfg.get(key), True)
    meta = plan.get("ltx_start_end_duration_safety") if isinstance(plan.get("ltx_start_end_duration_safety"), dict) else {}
    if "enabled" in meta:
        return _safe_bool(meta.get("enabled"), True)
    return True


def _ltx_shot_start_end(shot: Dict[str, Any]) -> tuple:
    start = _safe_float(shot.get("song_start"), 0.0)
    end = _safe_float(shot.get("song_end"), 0.0)
    duration = _safe_float(shot.get("duration"), 0.0)
    if end <= start and duration > 0.0:
        end = start + duration
    if duration <= 0.0 and end > start:
        duration = end - start
    return start, end, max(0.0, duration)


def _set_ltx_shot_timing_fields(shot: Dict[str, Any], start: float, end: float, fps: float) -> None:
    target_fps = max(1.0, float(fps or _safe_float(shot.get("target_fps"), 24.0) or 24.0))
    start = max(0.0, float(start))
    end = max(start + 0.05, float(end))
    duration = max(0.05, end - start)
    frames = max(1, int(round(duration * target_fps)))
    shot["song_start"] = round(start, 3)
    shot["song_end"] = round(end, 3)
    shot["duration"] = round(duration, 3)
    shot["target_fps"] = int(round(target_fps))
    shot["target_frames"] = int(frames)
    # Mark duration-sensitive generated assets as stale when a caller changes the
    # timeline after an older/shorter clip may already exist.
    shot["duration_contract_seconds"] = round(duration, 3)


def _ltx_vramlab_frame_cap_for_fps(fps: Any) -> tuple[int, float]:
    target_fps = max(1, int(round(_safe_float(fps, 24.0) or 24.0)))
    max_frames = min(241, max(1, int(round(10.0 * float(target_fps)))))
    return int(max_frames), float(max_frames) / float(target_fps)


def _cap_ltx_shot_timing_to_generation_limit(shot: Dict[str, Any], fps: Any, *, reason: str = "vramlab") -> Dict[str, Any]:
    """Clamp one shot's planned duration/frames to the own LTX-VRAMLab cap.

    The previous code capped only the command frames. That could leave plan/debug
    metadata saying 300+ planned frames while the command generated ~241 frames,
    which later made sync/assembly think the clip was too short.
    """
    if not isinstance(shot, dict):
        return {"changed": False, "max_frames": 241, "max_seconds": 10.0}
    target_fps = max(1, int(round(_safe_float(fps, _safe_float(shot.get("target_fps"), 24.0)) or 24.0)))
    max_frames, max_seconds = _ltx_vramlab_frame_cap_for_fps(target_fps)
    start, end, duration = _ltx_shot_start_end(shot)
    current_frames = max(1, int(round(max(0.001, duration) * float(target_fps))))
    if current_frames <= max_frames and duration <= max_seconds + 0.002:
        shot["ltx_generation_frame_cap"] = int(max_frames)
        shot["ltx_generation_seconds_cap"] = round(float(max_seconds), 6)
        return {"changed": False, "max_frames": int(max_frames), "max_seconds": float(max_seconds)}
    old = {
        "song_start": round(float(start), 6),
        "song_end": round(float(end), 6),
        "duration": round(float(duration), 6),
        "target_frames": int(current_frames),
    }
    new_end = float(start) + float(max_seconds)
    _set_ltx_shot_timing_fields(shot, start, new_end, target_fps)
    shot["ltx_generation_frame_cap_applied"] = True
    shot["ltx_generation_frame_cap_reason"] = str(reason or "vramlab")
    shot["ltx_generation_frame_cap"] = int(max_frames)
    shot["ltx_generation_seconds_cap"] = round(float(max_seconds), 6)
    shot["ltx_generation_frame_cap_previous"] = old
    return {"changed": True, "max_frames": int(max_frames), "max_seconds": float(max_seconds), "previous": old}


def _reduce_duration_pool(durations: List[float], amount: float, indexes: List[int], minimums: List[float]) -> float:
    remaining = max(0.0, float(amount))
    if remaining <= 0.0:
        return 0.0
    # Largest donors first keeps tiny preset/microclip shots from being crushed
    # before larger neighboring shots have contributed.
    for idx in sorted(indexes, key=lambda i: durations[i] - minimums[i], reverse=True):
        if remaining <= 0.0001:
            break
        reducible = max(0.0, durations[idx] - minimums[idx])
        if reducible <= 0.0:
            continue
        take = min(reducible, remaining)
        durations[idx] = max(minimums[idx], durations[idx] - take)
        remaining -= take
    return remaining


def _refresh_ltx_audio_chunk_for_shot(root: Path, plan_path: Path, director_plan: Dict[str, Any], shot: Dict[str, Any], warnings: List[str]) -> None:
    """Regenerate the LTX WAV guide after edge duration timing changes.

    The final assembly still uses the full original song as master audio. This is
    only the per-shot guide used by LTX during image-to-video generation/recreate.
    """
    audio_path = _safe_str(director_plan.get("final_master_audio") or director_plan.get("audio_path") or director_plan.get("music_path") or director_plan.get("source_audio_path"))
    if not audio_path or not os.path.isfile(audio_path):
        return
    sid = _safe_str(shot.get("id")) or "LTX"
    chunks_dir = _safe_str(director_plan.get("audio_chunks_dir"))
    if chunks_dir:
        audio_chunks_path = Path(chunks_dir).expanduser().resolve()
    else:
        audio_chunks_path = (Path(plan_path).resolve().parent / "audio_chunks").resolve()
        director_plan["audio_chunks_dir"] = str(audio_chunks_path)
    out_wav = audio_chunks_path / f"{_safe_stem(sid)}.wav"
    planned_start = _safe_float(shot.get("song_start"), 0.0)
    planned_end = _safe_float(shot.get("song_end"), planned_start + _safe_float(shot.get("duration"), 0.0))
    audio_start = max(0.0, planned_start - float(LTX_GENERATION_PRE_PAD_SECONDS))
    audio_end = planned_end + float(LTX_GENERATION_TAIL_PAD_SECONDS)
    try:
        song_duration = _probe_media_duration_seconds(root, Path(audio_path))
        if song_duration > 0.0:
            audio_end = min(song_duration, audio_end)
    except Exception:
        pass
    audio_duration = max(0.05, audio_end - audio_start)
    shot["audio_guide_start"] = round(float(audio_start), 6)
    shot["audio_guide_end"] = round(float(audio_end), 6)
    shot["audio_guide_duration"] = round(float(audio_duration), 6)
    shot["generation_pre_pad_seconds"] = LTX_GENERATION_PRE_PAD_SECONDS
    shot["generation_tail_pad_seconds"] = LTX_GENERATION_TAIL_PAD_SECONDS
    err = _export_ltx_audio_chunk(_find_bridge_ffmpeg(root), audio_path, out_wav, audio_start, audio_duration)
    if err:
        warnings.append(f"{sid}: could not refresh LTX audio guide after duration safety update: {err}")
    else:
        shot["audio_clip_path"] = str(out_wav)
        director_plan["audio_chunks_enabled"] = True


def _enforce_ltx_start_end_duration_contract(
    director_plan: Dict[str, Any],
    *,
    plan_path: Optional[Path] = None,
    root: Optional[Path] = None,
    enabled: bool = True,
    refresh_audio_chunks: bool = False,
) -> Dict[str, Any]:
    """Make the first/last LTX timing contract real in the plan JSON.

    This is intentionally stronger than the older neighboring-shot-only UI pass:
    if a short preset creates tiny shots, the first shot still becomes at least
    3s and the last shot targets 5s by redistributing time across the middle of
    the plan. The shot count is preserved; only timing fields are rewritten.
    """
    result = {
        "ok": True,
        "enabled": bool(enabled),
        "changed": False,
        "changed_shot_ids": [],
        "warnings": [],
        "first_min_seconds": float(LTX_EDGE_FIRST_MIN_SECONDS),
        "last_target_seconds": float(LTX_EDGE_LAST_TARGET_SECONDS),
    }
    if not enabled or not isinstance(director_plan, dict):
        return result
    shots = _director_plan_shots(director_plan)
    if len(shots) < 2:
        return result

    fps = _safe_float(director_plan.get("fps"), 24.0)
    if fps <= 0.0:
        fps = 24.0
    starts: List[float] = []
    ends: List[float] = []
    original_durations: List[float] = []
    for idx, shot in enumerate(shots):
        start, end, duration = _ltx_shot_start_end(shot)
        if idx > 0 and start < ends[-1] - 0.001:
            start = ends[-1]
            end = max(start + max(0.05, duration), end)
        if end <= start:
            end = start + max(0.05, duration)
        starts.append(start)
        ends.append(end)
        original_durations.append(max(0.05, end - start))

    timeline_start = starts[0]
    timeline_end = max(ends[-1], timeline_start + sum(original_durations))
    total_duration = max(0.05, timeline_end - timeline_start)
    first_min = float(LTX_EDGE_FIRST_MIN_SECONDS)
    last_min = float(LTX_EDGE_LAST_TARGET_SECONDS)
    donor_min = float(LTX_EDGE_DONOR_MIN_SECONDS)

    if original_durations[0] >= first_min - 0.001 and original_durations[-1] >= last_min - 0.001:
        director_plan.setdefault("ltx_start_end_duration_safety", {
            "enabled": True,
            "first_min_seconds": first_min,
            "last_target_seconds": last_min,
            "unchanged": True,
        })
        return result

    durations = list(original_durations)
    durations[0] = max(durations[0], min(first_min, total_duration))
    if len(durations) >= 2:
        durations[-1] = max(durations[-1], min(last_min, total_duration - max(0.05, durations[0])))

    minimums = [0.05 for _ in durations]
    if durations:
        minimums[0] = min(first_min, total_duration)
    if len(durations) >= 2:
        minimums[-1] = min(last_min, max(0.05, total_duration - minimums[0]))
    for i in range(1, max(1, len(durations) - 1)):
        minimums[i] = min(donor_min, max(0.05, durations[i]))

    overflow = sum(durations) - total_duration
    if overflow > 0.0001:
        middle = list(range(1, len(durations) - 1))
        overflow = _reduce_duration_pool(durations, overflow, middle, minimums)
    if overflow > 0.0001:
        # If first/last were already longer than their required minimum, allow that
        # extra edge time to help satisfy the opposite edge minimum.
        overflow = _reduce_duration_pool(durations, overflow, [0, len(durations) - 1], minimums)
    if overflow > 0.0001:
        # Extremely short songs/plans cannot physically hold 3s + 5s + middle shots.
        # Keep the plan valid instead of creating negative/overlapping shots.
        result["warnings"].append("Track segment is too short to fully satisfy first=3s and last=5s while preserving all shots; applied the closest valid timing.")
        flexible = list(range(len(durations)))
        hard_mins = [0.05 for _ in durations]
        overflow = _reduce_duration_pool(durations, overflow, flexible, hard_mins)

    underflow = total_duration - sum(durations)
    if underflow > 0.0001:
        # Put leftover time back into middle shots first so first/last remain clean
        # minimum/target slots instead of becoming unexpectedly long.
        receivers = list(range(1, len(durations) - 1)) or [0]
        weights = [max(0.05, original_durations[i]) for i in receivers]
        weight_total = sum(weights) or float(len(receivers))
        for idx, weight in zip(receivers, weights):
            durations[idx] += underflow * (weight / weight_total)

    cursor = timeline_start
    changed_ids: List[str] = []
    for idx, shot in enumerate(shots):
        new_start = cursor
        new_end = cursor + max(0.05, durations[idx])
        old_start, old_end, old_duration = _ltx_shot_start_end(shot)
        if abs(new_start - old_start) > 0.002 or abs(new_end - old_end) > 0.002 or abs((new_end - new_start) - old_duration) > 0.002:
            sid = _safe_str(shot.get("id")) or f"LTX{idx+1:02d}"
            changed_ids.append(sid)
            shot["duration_safety_changed"] = True
            shot["duration_safety_previous"] = {
                "song_start": round(float(old_start), 6),
                "song_end": round(float(old_end), 6),
                "duration": round(float(old_duration), 6),
            }
        _set_ltx_shot_timing_fields(shot, new_start, new_end, fps)
        # Clip-relative lyrics and timestamp prompts are allowed to remain as-is;
        # the LTX sanitizer/recreate path clamps unsafe timestamps to the new duration.
        cursor = new_end

    if changed_ids:
        result["changed"] = True
        result["changed_shot_ids"] = changed_ids
        result["warnings"].append(
            f"Applied LTX edge duration safety: first >= {first_min:.1f}s, last target >= {last_min:.1f}s."
        )
        if refresh_audio_chunks and plan_path is not None:
            root_path = Path(root or _project_root()).resolve()
            for shot in shots:
                sid = _safe_str(shot.get("id"))
                if sid in changed_ids:
                    try:
                        _refresh_ltx_audio_chunk_for_shot(root_path, Path(plan_path), director_plan, shot, result["warnings"])
                    except Exception as exc:
                        result["warnings"].append(f"{sid}: could not refresh audio guide after duration update: {exc}")

    director_plan["ltx_start_end_duration_safety"] = {
        "enabled": True,
        "first_min_seconds": first_min,
        "last_target_seconds": last_min,
        "middle_donor_min_seconds": donor_min,
        "changed": bool(result["changed"]),
        "changed_shot_ids": result["changed_shot_ids"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "warnings": result["warnings"],
    }
    if isinstance(director_plan.get("bridge_generation_settings"), dict):
        director_plan["bridge_generation_settings"]["avoid_short_start_end_clips"] = True
    return result


def apply_ltx_start_end_duration_safety_to_plan(payload: dict) -> dict:
    """Public bridge helper used by the UI and review loader for old jobs."""
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Duration safety payload was not a dictionary."}
        plan_path = Path(_safe_str(payload.get("ltx_director_plan_path") or payload.get("plan_path"))).expanduser().resolve()
        if not plan_path.is_file():
            return {"ok": False, "message": f"LTX director plan was not found: {plan_path}"}
        director_plan = _read_json_file(str(plan_path))
        enabled = _payload_ltx_edge_duration_safety_enabled(payload, director_plan)
        root_raw = _safe_str(payload.get("root_dir"))
        root = Path(root_raw).resolve() if root_raw else _project_root()
        result = _enforce_ltx_start_end_duration_contract(
            director_plan,
            plan_path=plan_path,
            root=root,
            enabled=enabled,
            refresh_audio_chunks=_safe_bool(payload.get("refresh_audio_chunks"), True),
        )
        if bool(result.get("changed")):
            existing_warnings = _as_list(director_plan.get("warnings"))
            merged = existing_warnings + [w for w in _as_list(result.get("warnings")) if w]
            if merged:
                director_plan["warnings"] = merged[:80]
            _write_json_file(plan_path, director_plan)
        return {"ok": True, "message": "LTX start/end duration safety applied." if bool(result.get("changed")) else "LTX start/end duration safety already OK.", "plan_path": str(plan_path), **result}
    except Exception as exc:
        return {"ok": False, "message": f"Could not apply LTX start/end duration safety: {exc}"}

def _build_ltx_shot_item(
    *,
    group: List[Dict[str, Any]],
    index: int,
    brief: Dict[str, str],
    fps: float,
    grouping_mode: str,
    lyric_segments: List[Dict[str, Any]],
    character_bible: Optional[List[Dict[str, Any]]] = None,
    group_bibles: Optional[List[Dict[str, Any]]] = None,
    duration_profile: Optional[Dict[str, Any]] = None,
    microclip_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    shot_id = f"LTX{index:02d}"
    start = min(_scene_start(s) for s in group)
    end = max(_scene_end(s) for s in group)
    duration = max(0.05, end - start)
    profile = duration_profile if isinstance(duration_profile, dict) else _duration_profile_from_bridge_settings({})
    micro_profile = microclip_profile if isinstance(microclip_profile, dict) else _microclip_profile_from_bridge_settings({}, profile)
    active_vocal_scenes = [s for s in group if _scene_has_strong_vocal(s)]
    non_vocal_scenes = [s for s in group if not _scene_has_strong_vocal(s)]
    norm_lyrics = _normalize_ltx_lyric_segments(lyric_segments)
    clip_lyrics = _make_clip_relative_lyrics(norm_lyrics, start, end, active_only=True)
    needs_lipsync = bool(active_vocal_scenes and clip_lyrics)
    min_lipsync = max(0.0, _safe_float(profile.get("min_lipsync_seconds"), 3.0))
    if needs_lipsync and duration < min_lipsync:
        needs_lipsync = False
    is_microclip, microclip_reason = _microclip_allowed_for_group(
        group,
        duration=duration,
        needs_lipsync=needs_lipsync,
        clip_lyrics=clip_lyrics,
        microclip_profile=micro_profile,
        duration_profile=profile,
    )
    microclip_style = _microclip_style_for_group(group, brief) if is_microclip else ""
    micro_moment_count = _microclip_moment_count(duration, _group_energy_rank(group)) if is_microclip else 0
    if is_microclip:
        needs_lipsync = False
        active_vocal_scenes = []
        non_vocal_scenes = list(group)
        clip_lyrics = []
    lyrics_texts = [x.get("text") for x in clip_lyrics] if clip_lyrics else []
    lyrics = " ".join(_dedupe_texts(lyrics_texts, max_items=10, max_len=1200))
    lipsync_windows = [
        {
            "source_scene_id": _scene_id(s, _safe_int(s.get("index"), 1)),
            "start": round(max(0.0, _scene_start(s) - start), 3),
            "end": round(max(0.0, _scene_end(s) - start), 3),
            "lyrics": _clean_text(s.get("lyrics") or s.get("lyric_text"), 300),
        }
        for s in active_vocal_scenes
    ]
    target_fps = float(fps or 24.0)
    target_frames = int(round(duration * target_fps))
    source_ids = [_scene_id(s, i + 1) for i, s in enumerate(group)]
    source_indexes = [_safe_int(s.get("index"), i + 1) for i, s in enumerate(group)]
    dominant_location = _single_location_world_for_scene(brief, group[0] if group else {}, source_indexes[0] if source_indexes else index)
    timestamped = _microclip_timestamped_video_prompt(brief, group, duration, microclip_style, micro_moment_count) if is_microclip else _timestamped_video_prompt_for_ltx_group(brief, group, duration, needs_lipsync)
    timestamped = _sanitize_ltx_microclip_timestamped_prompt(timestamped, duration) if is_microclip else _sanitize_ltx_timestamped_prompt(timestamped, duration)
    shot = {
        "id": shot_id,
        "index": index,
        "source_scene_ids": source_ids,
        "source_scene_indexes": source_indexes,
        "song_start": round(start, 3),
        "song_end": round(end, 3),
        "duration": round(duration, 3),
        "dominant_location": dominant_location,
        "duration_profile_used": _duration_profile_short_copy(profile),
        "microclip_profile_used": _microclip_profile_short_copy(micro_profile),
        "grouping_reason": (("microclip-safe grouped beat-hit moments" if is_microclip else "single source scene") if len(group) == 1 else f"grouped {len(group)} scenes by {grouping_mode} using max { _safe_float(profile.get('max_ltx_shot_seconds'), LTX_SOFT_MAX_SHOT_SECONDS):.1f}s"),
        "is_microclip": bool(is_microclip),
        "microclip_reason": microclip_reason if is_microclip else "",
        "microclip_style": microclip_style,
        "micro_moment_count": int(micro_moment_count),
        "major_scene_block_limit": _major_scene_block_limit(duration),
        "major_scene_block_rule": _scene_block_guidance(duration),
        "target_fps": target_fps,
        "target_frames": target_frames,
        "section_summary": _summarize_unique([s.get("section") for s in group], "unknown"),
        "energy_summary": _summarize_unique([s.get("energy") for s in group], "mid"),
        "vocal_presence_summary": ("instrumental" if is_microclip else ("mixed" if active_vocal_scenes and non_vocal_scenes else _summarize_unique([s.get("vocal_presence") for s in group], "instrumental" if not needs_lipsync else "vocal"))),
        "needs_lipsync": bool(False if is_microclip else needs_lipsync),
        "active_vocal_scene_count": len(active_vocal_scenes),
        "non_vocal_scene_count": len(non_vocal_scenes),
        "lipsync_windows": lipsync_windows,
        "scene_role_summary": (microclip_style or _summarize_unique([s.get("scene_role") for s in group], "music-video shot")),
        "lyrics": lyrics,
        "clip_relative_lyrics": clip_lyrics,
        "image_prompt": _merge_group_image_prompt(brief, group, needs_lipsync),
        "video_prompt": _merge_group_video_prompt(brief, group, needs_lipsync),
        "timestamped_video_prompt": timestamped,
        "negative_prompt": _negative_prompt_with_no_visible_text(_summarize_unique([s.get("negative_prompt") for s in group], "blurry, low quality, broken anatomy"), max_len=1200),
        "audio_clip_path": "",
        "notes": _notes_for_ltx_group(group, grouping_mode),
    }
    # LTX start-image prompts must stay clean and single-location; timestamped
    # prompts can contain motion, but the first/end frame still cannot be a montage.
    for _key in ("image_prompt", "video_prompt", "timestamped_video_prompt"):
        if _key == "timestamped_video_prompt":
            shot[_key] = _remove_collage_language(shot.get(_key))
        else:
            shot[_key] = _sanitize_single_frame_prompt(shot.get(_key), brief=brief, item=shot, allow_montage=False, start_or_end=(_key == "image_prompt"))
    shot["negative_prompt"] = _negative_prompt_with_no_visible_text(_join_parts([shot.get("negative_prompt"), _anti_collage_negative_terms()]), max_len=1200)

    if is_microclip:
        shot["vocal_presence_summary"] = "instrumental"
        shot["scene_role_summary"] = microclip_style
        shot["image_prompt"] = _strip_non_lipsync_vocal_language(_sentence(str(shot.get("image_prompt") or "") + f" Clean microclip still for {microclip_style}: one coherent full-frame view, same world, same subject continuity, action/detail/b-roll framing, no lyric-mouthing and no staged vocal props."))
        shot["video_prompt"] = _strip_non_lipsync_vocal_language(_sentence(str(shot.get("video_prompt") or "") + f" Use sequential full-frame beat cuts ({microclip_style}); one view at a time, same scene/world, no lipsync."))
        shot["notes"] = _sentence(str(shot.get("notes") or "") + " Timestamped microclip mode: sequential full-frame beat cuts inside one short LTX generation instead of sub-second LTX jobs; multi-panel effects are separate and optional.")

    max_ltx_for_warning = _safe_float(profile.get("max_ltx_shot_seconds"), 0.0)
    hard_ltx_for_warning = _safe_float(profile.get("hard_max_ltx_shot_seconds"), max_ltx_for_warning)
    if max_ltx_for_warning > 0 and duration > max_ltx_for_warning + 0.15:
        overflow = duration - max_ltx_for_warning
        shot["duration_warning"] = f"Shot is {overflow:.2f}s over the requested/profile max to preserve grouping or lyric-boundary safety."
    if hard_ltx_for_warning > 0 and duration > hard_ltx_for_warning + 0.15:
        shot["duration_warning"] = f"Shot exceeds hard max by {duration - hard_ltx_for_warning:.2f}s; inspect grouping."
    if bool(profile.get("prefer_fast_nonvocal_cuts")) and not needs_lipsync:
        shot["video_prompt"] = _sentence(str(shot.get("video_prompt") or "") + " Fast energetic music-video movement, dance/action/b-roll timing, camera reacts to the beat without lyric-mouthing.")
        if not _safe_bool(shot.get("is_microclip"), False):
            shot["timestamped_video_prompt"] = _sentence(str(shot.get("timestamped_video_prompt") or "") + " Keep it energetic with movement, action, lights, or b-roll; avoid lyric-mouthing.")
    shot = _protect_lipsync_confidence(shot)
    shot = _apply_character_identity_to_item(shot, brief, character_bible or [], group_bibles or [], include_director_fields=False)
    if _safe_bool(shot.get("is_microclip"), False):
        shot["timestamped_video_prompt"] = _enforce_single_location_without_negative_words(shot.get("timestamped_video_prompt"), brief, shot)
        shot["video_prompt"] = _enforce_single_location_without_negative_words(shot.get("video_prompt"), brief, shot)
    if not _safe_bool(shot.get("needs_lipsync"), False):
        for _key in _nonvocal_prompt_keys():
            if _key in shot:
                shot[_key] = _strip_non_lipsync_vocal_language(shot.get(_key))
    shot = _sanitize_ltx_timestamp_fields_in_item(shot, keys=["timestamped_video_prompt"])
    return shot


def create_ltx_shot_plan(payload: dict) -> dict:
    """Create a smaller LTX shot plan from musicclip_prompt_plan.json.

    This does not run Planner, does not run LTX and does not generate images/video.
    Optional WAV chunks are only exported for private LTX lipsync/audio-guidance tests;
    final assembled audio should still use the original full song.
    """
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "LTX shot-plan payload was not a dictionary."}

        prompt_plan_path = _safe_str(payload.get("prompt_plan_path"))
        prompt_plan_obj = payload.get("prompt_plan")
        if isinstance(prompt_plan_obj, dict):
            prompt_plan = prompt_plan_obj
        elif prompt_plan_path:
            prompt_plan = _read_json_file(prompt_plan_path)
        else:
            return {"ok": False, "message": "No prompt_plan_path was provided."}

        scenes = [x for x in _as_list(prompt_plan.get("scenes")) if isinstance(x, dict)]
        if not scenes:
            return {"ok": False, "message": "Prompt plan did not contain any scenes."}

        scene_plan_path = _safe_str(payload.get("scene_plan_path") or prompt_plan.get("scene_plan_path"))
        scene_plan: Dict[str, Any] = {}
        if scene_plan_path:
            try:
                scene_plan = _read_json_file(scene_plan_path)
            except Exception:
                scene_plan = {}

        out_raw = _safe_str(payload.get("output_dir"))
        if out_raw:
            out_dir = Path(out_raw).resolve()
        elif prompt_plan_path:
            out_dir = Path(prompt_plan_path).resolve().parent
        else:
            out_dir = _default_ltx_videoclip_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        brief = _normalize_creative_brief(prompt_plan.get("creative_brief") or scene_plan.get("creative_brief") or payload.get("creative_brief"))
        character_bible, group_bibles = _normalize_character_bibles(
            prompt_plan.get("character_bible") or scene_plan.get("character_bible") or payload.get("character_bible"),
            prompt_plan.get("group_bibles") or scene_plan.get("group_bibles") or payload.get("group_bibles"),
            brief,
        )
        shot_assignment_guidance = _safe_str(prompt_plan.get("shot_assignment_guidance") or scene_plan.get("shot_assignment_guidance") or payload.get("shot_assignment_guidance"))
        character_bible_backend = _safe_str(prompt_plan.get("character_bible_backend") or "inherited_or_neutral_fallback")
        character_bible_warnings = _as_list(prompt_plan.get("character_bible_warnings"))
        fps = _safe_float(payload.get("fps"), 24.0)
        if fps <= 0.0:
            fps = 24.0
        grouping_mode = _normalize_ltx_grouping_mode(payload.get("grouping_mode") or payload.get("shot_grouping_mode") or "Auto")
        export_audio = _safe_bool(payload.get("export_audio_chunks", payload.get("audio_chunks_enabled", False)), False)
        audio_path = _safe_str(payload.get("audio_path") or prompt_plan.get("audio_path") or scene_plan.get("audio_path"))
        srt_path = _safe_str(payload.get("srt_path") or prompt_plan.get("srt_path") or scene_plan.get("srt_path"))
        lyric_segments = [x for x in _as_list(payload.get("lyric_segments") or scene_plan.get("lyrics_srt_segments") or prompt_plan.get("lyrics_srt_segments")) if isinstance(x, dict)]
        bridge_generation_settings = _normalize_bridge_generation_settings(payload, prompt_plan, scene_plan)
        duration_profile = _duration_profile_from_bridge_settings(bridge_generation_settings)
        microclip_profile = _microclip_profile_from_bridge_settings(bridge_generation_settings, duration_profile)
        effects_profile = _effects_profile_from_bridge_settings(bridge_generation_settings)
        character_reference = _character_reference_from_sources(payload, prompt_plan, scene_plan)
        character_bible, group_bibles = _apply_character_reference_to_bibles(character_bible, group_bibles, character_reference)

        groups = _group_prompt_scenes_for_ltx(scenes, grouping_mode, lyric_segments, duration_profile=duration_profile)
        if not groups:
            return {"ok": False, "message": "No LTX shot groups could be created."}

        warnings: List[str] = []
        shots: List[Dict[str, Any]] = []
        for i, group in enumerate(groups, start=1):
            shots.append(_build_ltx_shot_item(
                group=group,
                index=i,
                brief=brief,
                fps=fps,
                grouping_mode=grouping_mode,
                lyric_segments=lyric_segments,
                character_bible=character_bible,
                group_bibles=group_bibles,
                duration_profile=duration_profile,
                microclip_profile=microclip_profile,
            ))

        boundary_warnings = _apply_ltx_lyric_boundary_guard(shots, lyric_segments, fps, duration_profile=duration_profile)
        warnings.extend(boundary_warnings)
        max_ltx_profile = _safe_float(duration_profile.get("max_ltx_shot_seconds"), 0.0)
        hard_ltx_profile = _safe_float(duration_profile.get("hard_max_ltx_shot_seconds"), max_ltx_profile)
        for shot in shots:
            dur_now = _safe_float(shot.get("duration"), 0.0)
            if max_ltx_profile > 0 and dur_now > max_ltx_profile + 0.15 and not _safe_str(shot.get("duration_warning")):
                shot["duration_warning"] = f"Shot is {dur_now - max_ltx_profile:.2f}s over the requested/profile max after lyric-boundary protection."
            if hard_ltx_profile > 0 and dur_now > hard_ltx_profile + 0.15:
                shot["duration_warning"] = f"Shot exceeds hard max by {dur_now - hard_ltx_profile:.2f}s after lyric-boundary protection; inspect grouping."
                warnings.append(f"{_safe_str(shot.get('id'))}: {shot['duration_warning']}")
        song_duration = max([_safe_float(x.get("song_end"), 0.0) for x in shots] + [0.0])
        protected_shots: List[Dict[str, Any]] = []
        for shot in shots:
            before = _safe_bool(shot.get("needs_lipsync"), False)
            fixed_shot = _protect_lipsync_confidence(shot, song_duration=song_duration)
            fixed_shot = _apply_character_reference_to_item(fixed_shot, brief, character_reference, image_prompt_keys=["image_prompt"], passed_to_model=False)
            if _safe_bool(fixed_shot.get("is_microclip"), False):
                fixed_shot["needs_lipsync"] = False
                fixed_shot["vocal_presence_summary"] = "instrumental"
                fixed_shot["clip_relative_lyrics"] = []
                for _key in _nonvocal_prompt_keys():
                    if _key in fixed_shot:
                        fixed_shot[_key] = _strip_non_lipsync_vocal_language(fixed_shot.get(_key))
            if before and not _safe_bool(fixed_shot.get("needs_lipsync"), False):
                warnings.append(f"{_safe_str(shot.get('id'))}: downgraded to non-vocal because no active clip-relative lyric window was found.")
            protected_shots.append(fixed_shot)
        shots = protected_shots

        montage_policy = _montage_policy_from_plan(prompt_plan, scene_plan, payload, {"effects_profile": effects_profile})
        shots, montage_policy, montage_warnings = _apply_montage_policy_to_items(
            shots,
            brief=brief,
            policy=montage_policy,
            prompt_keys=["image_prompt", "video_prompt", "timestamped_video_prompt"],
        )
        warnings.extend(montage_warnings)

        shots, intro_outro_policy, intro_outro_warnings = _apply_intro_outro_safety_to_items(
            shots,
            brief=brief,
            prompt_keys=["image_prompt", "video_prompt", "timestamped_video_prompt"],
        )
        warnings.extend(intro_outro_warnings)
        shots = _apply_effect_policy_metadata(shots, effects_profile=effects_profile)
        shots = [_sanitize_ltx_timestamp_fields_in_item(shot, keys=["timestamped_video_prompt"]) for shot in shots]

        # Make the visible "avoid short start/end clips" option part of the real
        # director plan before clips/audio chunks are generated. This prevents a
        # 2.x second first shot from being rendered and then failing assembly later.
        edge_safety = _enforce_ltx_start_end_duration_contract(
            {"fps": fps, "shots": shots, "bridge_generation_settings": bridge_generation_settings},
            enabled=_safe_bool(bridge_generation_settings.get("avoid_short_start_end_clips"), True),
            refresh_audio_chunks=False,
        )
        if edge_safety.get("changed"):
            warnings.extend(_as_list(edge_safety.get("warnings")))

        audio_chunks_dir = ""
        if export_audio:
            audio_chunks_path = out_dir / "audio_chunks"
            audio_chunks_dir = str(audio_chunks_path)
            if not audio_path or not os.path.isfile(audio_path):
                audio_missing_warning = "Audio chunks were requested but the original audio file was not found."
                warnings.append(audio_missing_warning)
                for shot in shots:
                    shot["notes"] = _clean_text(str(shot.get("notes") or "") + " " + audio_missing_warning)
            else:
                ffmpeg = _find_bridge_ffmpeg(payload.get("root_dir") or _project_root())
                for shot in shots:
                    out_wav = audio_chunks_path / f"{_safe_str(shot.get('id'))}.wav"
                    planned_start = _safe_float(shot.get("song_start"), 0.0)
                    planned_end = _safe_float(shot.get("song_end"), planned_start + _safe_float(shot.get("duration"), 0.0))
                    audio_start = max(0.0, planned_start - float(LTX_GENERATION_PRE_PAD_SECONDS))
                    audio_end = planned_end + float(LTX_GENERATION_TAIL_PAD_SECONDS)
                    song_duration = _probe_media_duration_seconds(Path(payload.get("root_dir") or _project_root()), Path(audio_path))
                    if song_duration > 0.0:
                        audio_end = min(song_duration, audio_end)
                    audio_duration = max(0.05, audio_end - audio_start)
                    shot["audio_guide_start"] = round(float(audio_start), 6)
                    shot["audio_guide_end"] = round(float(audio_end), 6)
                    shot["audio_guide_duration"] = round(float(audio_duration), 6)
                    shot["generation_pre_pad_seconds"] = LTX_GENERATION_PRE_PAD_SECONDS
                    shot["generation_tail_pad_seconds"] = LTX_GENERATION_TAIL_PAD_SECONDS
                    err = _export_ltx_audio_chunk(
                        ffmpeg,
                        audio_path,
                        out_wav,
                        audio_start,
                        audio_duration,
                    )
                    if err:
                        warnings.append(f"{_safe_str(shot.get('id'))}: {err}")
                        shot["audio_clip_path"] = ""
                        shot["notes"] = _notes_for_ltx_group([s for s in groups[_safe_int(shot.get('index'), 1)-1]], grouping_mode, err)
                    else:
                        shot["audio_clip_path"] = str(out_wav)

        plan_path = out_dir / "musicclip_ltx_shot_plan.json"
        ltx_plan = {
            "version": _VERSION,
            "source": _SOURCE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "prompt_plan_path": prompt_plan_path,
            "scene_plan_path": scene_plan_path,
            "audio_path": audio_path,
            "srt_path": srt_path,
            "creative_brief": brief,
            "character_bible": character_bible,
            "group_bibles": group_bibles,
            "shot_assignment_guidance": shot_assignment_guidance,
            "character_bible_backend": character_bible_backend,
            "character_bible_warnings": character_bible_warnings,
            "character_reference": {k: v for k, v in character_reference.items() if k != "warnings"},
            "character_reference_warnings": _plan_character_reference_warnings(character_reference, passed_to_model=False),
            "fps": fps,
            "resolution": _safe_str(bridge_generation_settings.get("ltx_resolution") or bridge_generation_settings.get("resolution"), "1280x704"),
            "ltx_resolution": _safe_str(bridge_generation_settings.get("ltx_resolution") or bridge_generation_settings.get("resolution"), "1280x704"),
            "ltx_aspect_mode": _safe_str(bridge_generation_settings.get("ltx_aspect_mode")),
            "ltx_shot_count": len(shots),
            "source_scene_count": len(scenes),
            "shot_grouping_mode": grouping_mode,
            "bridge_generation_settings": bridge_generation_settings,
            "duration_profile": duration_profile,
            "microclip_profile": microclip_profile,
            "effects_profile": effects_profile,
            "montage_policy": montage_policy,
            "intro_outro_policy": intro_outro_policy,
            "location_pool": _location_candidates_from_brief(brief),
            "lyric_boundary_guard": {
                "enabled": True,
                "start_pad_seconds": LYRIC_START_PAD_SECONDS,
                "end_pad_seconds": LYRIC_END_PAD_SECONDS,
                "snap_tolerance_seconds": LYRIC_BOUNDARY_SNAP_TOLERANCE_SECONDS,
                "long_lyric_segment_seconds": LONG_LYRIC_SEGMENT_SECONDS,
                "max_lipsync_seconds_per_long_segment": MAX_LIPSYNC_SECONDS_PER_LONG_SEGMENT,
            },
            "audio_chunks_enabled": bool(export_audio),
            "audio_chunks_dir": audio_chunks_dir,
            "final_master_audio": audio_path,
            "shots": shots,
        }
        if warnings:
            ltx_plan["warnings"] = warnings[:50]

        _write_json_file(plan_path, ltx_plan)
        msg = "LTX shot plan created."
        if warnings:
            msg += f" Warnings: {len(warnings)} planning issue(s)."
        return {
            "ok": True,
            "ltx_shot_plan_path": str(plan_path),
            "output_dir": str(out_dir),
            "ltx_shot_count": len(shots),
            "source_scene_count": len(scenes),
            "audio_chunks_dir": audio_chunks_dir,
            "message": msg,
            "warnings": warnings[:20],
        }
    except Exception as exc:
        return {"ok": False, "message": f"LTX shot plan creation failed: {exc}"}

# -----------------------------
# Chunk 5B: LTX shot plan -> Director rewrite plan
# -----------------------------
_DIRECTOR_BACKEND_TEMPLATE = "template_cleanup"
_DIRECTOR_BACKEND_LOCAL_LLM = "local_llm_planner_style"


def _director_backend_label(value: Any) -> str:
    text = _safe_str(value) or "Template cleanup"
    low = text.lower()
    if "llm" in low or "llama" in low or "planner" in low:
        return "Local LLM / Planner-style"
    return "Template cleanup"


def _normalize_director_backend(value: Any) -> str:
    return _DIRECTOR_BACKEND_LOCAL_LLM if _director_backend_label(value).startswith("Local LLM") else _DIRECTOR_BACKEND_TEMPLATE


def _director_clean_prompt_text(value: Any, max_len: int = 1800) -> str:
    text = _clean_text(value, max_len)
    replacements = [
        (r"\bextra\s+terrestial\b", "extraterrestrial"),
        (r"\bextraterrestial\b", "extraterrestrial"),
        (r"\bailen\b", "alien"),
        (r"\baliens\s+starts\b", "the alien performer starts"),
        (r"\bthe singer begins performing at once,\s*", ""),
        (r"\bstarts singing immediately,\s*starts singing immediately\b", "starts singing immediately"),
        (r"\bof characters\b", "with the characters"),
        (r"\bof aliens,\s*consistent lead singer identity\b", "with a consistent alien lead singer identity"),
        (r"\s+([,.;:!?])", r"\1"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    return text[:max_len].rstrip(" ,.;") if max_len and len(text) > max_len else text


def _director_subject(brief: Dict[str, str], needs_lipsync: bool) -> str:
    subject = _clean_text(brief.get("characters_subjects"), 160)
    if not subject:
        subject = "the lead performer" if needs_lipsync else "the main subjects"
    low = subject.lower()
    if needs_lipsync and not any(x in low for x in ("singer", "performer", "vocal", "artist")):
        if "alien" in low:
            return "the alien lead singer"
        return f"the lead singer/performer ({subject})"
    return subject


def _director_world(brief: Dict[str, str]) -> str:
    # Important: the main idea/story is context only.  It must not be copied into
    # every generated still-image prompt as if it were a location.
    loc = _clean_text(brief.get("locations_world"), 180)
    style = _clean_text(brief.get("style_theme"), 180)
    parts = []
    if loc:
        parts.append(loc)
    if style:
        parts.append(style)
    return _join_parts(parts) or "a cinematic visual world"


def _director_action_theme(brief: Dict[str, str]) -> str:
    # Keep the user idea as source context only.  Per-shot actions are generated
    # through director_scene_concept, not by pasting main_idea into every prompt.
    subject = _clean_text(brief.get("characters_subjects"), 160)
    if subject:
        return f"{subject} performs one clear scene-specific action"
    return "the intended visible subject performs one clear scene-specific action"




_DIRECTOR_VISUAL_SAFETY_LINE = "single coherent full-frame scene, no text, no collage, no split-screen, no grid"
_DIRECTOR_FORBIDDEN_IMAGE_TERMS = (
    "ltx", "grouped ltx", "shot id", "scene id", "previous shot", "next shot", "continue",
    "final video", "grouped", "workflow", "director plan", "metadata", "internal timing",
)
_DIRECTOR_GENERIC_SUBJECT = "the intended visible subject"


# Human/reference routing helpers -------------------------------------------------
# These helpers intentionally avoid hardcoded story examples.  They only decide
# whether a finished shot prompt/concept expects visible human subjects.  Reference
# images are routed from that decision, not from broad labels like b-roll/background.
_HUMAN_SUBJECT_RE = re.compile(
    r"\b(?:person|people|human|humans|character|characters|man|men|woman|women|boy|boys|girl|girls|"
    r"male|female|husband|wife|couple|lovers|partners|girlfriend|boyfriend|performer|performers|"
    r"singer|singers|vocalist|vocalists|dancer|dancers|band|group|duo|trio|crew|artist|artists|"
    r"actor|actors|model|models|member|members|dj|rapper|rappers)\b",
    flags=re.IGNORECASE,
)
_HUMAN_NEGATION_RE = re.compile(
    r"\b(?:no|without|omit|omits|empty of|do not add|don't add|avoid|exclude|excluding)\s+"
    r"(?:visible\s+)?(?:people|persons|humans|human|characters|performers|singers|dancers|men|women|man|woman|person|band|group)\b",
    flags=re.IGNORECASE,
)
_ENVIRONMENT_ONLY_RE = re.compile(
    r"\b(?:environment[-_\s]?only|location[-_\s]?only|background[-_\s]?only(?:[-_\s]?presence)?|"
    r"background_only_presence|b[-_\s]?roll[-_\s]?no[-_\s]?character|no[-_\s]?character|"
    r"no\s+visible\s+(?:subject|people|person|human|humans|character|characters)|"
    r"empty\s+(?:room|street|hallway|cockpit|bridge|corridor|kitchen|bedroom|dining\s+room|tv\s+room|engine\s+room|promenade|park|space|landscape)|"
    r"landscape\s+only|object[-_\s]?only|world\s+only)\b",
    flags=re.IGNORECASE,
)
_MULTI_HUMAN_RE = re.compile(
    r"\b(?:all\s+(?:the\s+)?(?:performers|members|people|characters|dancers|singers|artists)|"
    r"(?:the\s+)?performers|performers\s+(?:dancing|walking|posing|moving|together)|"
    r"group(?:\s+(?:shot|of\s+(?:performers|dancers|people|members|characters|artists)))?|"
    r"full\s+group|whole\s+group|band|trio|dancers|crew|ensemble|"
    r"three\s+(?:people|performers|members|characters|singers|dancers|artists)|3\s+(?:people|performers|members|characters|singers|dancers|artists)|"
    r"multiple\s+(?:people|performers|members|characters|singers|dancers|artists)|several\s+(?:people|performers|members|characters|singers|dancers|artists))\b",
    flags=re.IGNORECASE,
)
_DUO_HUMAN_RE = re.compile(
    r"\b(?:couple|duo|duet|two\s+(?:people|performers|members|characters|singers|dancers|artists)|2\s+(?:people|performers|members|characters|singers|dancers|artists)|both\s+(?:people|performers|members|characters|singers|dancers|artists)|both\s+people|husband\s+and\s+wife|man\s+and\s+(?:his\s+)?wife|man\s+and\s+woman|woman\s+and\s+man)\b",
    flags=re.IGNORECASE,
)
_SOLO_HUMAN_RE = re.compile(
    r"\b(?:solo|one\s+(?:person|performer|member|character|singer|dancer|man|woman|artist)|1\s+(?:person|performer|member|character|singer|dancer|artist)|single\s+(?:person|performer|member|character|artist)|lead\s+(?:performer|singer|character|artist)\s*(?:only)?|portrait\s+of\s+one|close[-\s]?up\s+of\s+one|one\s+performer\s+close[-\s]?up)\b",
    flags=re.IGNORECASE,
)


def _reference_valid_slot_paths(ref: Dict[str, Any]) -> tuple[List[tuple[str, str]], List[str]]:
    warnings: List[str] = []
    sheets = ref.get("character_reference_sheets") if isinstance(ref.get("character_reference_sheets"), dict) else {}
    valid_slots: List[tuple[str, str]] = []
    for key in _character_reference_slot_keys():
        path = _safe_str(sheets.get(key))
        if not path:
            continue
        try:
            if os.path.isfile(path):
                valid_slots.append((key, path))
            else:
                warnings.append(f"Reference sheet path for {key} is missing: {path}")
        except Exception:
            warnings.append(f"Reference sheet path for {key} could not be checked: {path}")
    return valid_slots, warnings


def _reference_routing_text(shot: Dict[str, Any], concept: Optional[Dict[str, Any]] = None, *, include_old_templates: bool = False) -> str:
    concept = concept if isinstance(concept, dict) else (shot.get("director_scene_concept") if isinstance(shot.get("director_scene_concept"), dict) else {})
    fields: List[Any] = [
        concept.get("visible_subject"), concept.get("main_action"), concept.get("shot_type"), concept.get("notes"),
        shot.get("visible_subject"), shot.get("main_subject"), shot.get("shot_cast_type"),
        shot.get("director_image_prompt"), shot.get("director_video_prompt"), shot.get("director_timestamped_video_prompt"),
        shot.get("character_labels"), shot.get("character_identity_prompt"), shot.get("group_identity_prompt"),
        shot.get("scene_role"), shot.get("scene_role_summary"), shot.get("role"), shot.get("preferred_clip_role"),
        shot.get("notes"), shot.get("lyrics"), shot.get("lyric_text"),
    ]
    if include_old_templates:
        fields.extend([shot.get("image_prompt"), shot.get("template_image_prompt"), shot.get("prompt"), shot.get("video_prompt")])
    parts: List[str] = []
    for value in fields:
        if isinstance(value, list):
            value = " ".join(_safe_str(x) for x in value if _safe_str(x))
        if _safe_str(value):
            parts.append(_safe_str(value))
    return " ".join(parts)


def _has_visible_human_subject_text(value: Any) -> bool:
    text = _safe_str(value)
    if not text:
        return False
    # Remove negative instructions before checking for positive human evidence, so
    # "no people" does not become a false reference trigger.
    cleaned = _HUMAN_NEGATION_RE.sub(" ", text)
    cleaned = re.sub(r"\b(?:no|without)\s+(?:visible\s+)?(?:subject|subjects)\b", " ", cleaned, flags=re.IGNORECASE)
    return bool(_HUMAN_SUBJECT_RE.search(cleaned))


def _has_environment_only_signal(shot: Dict[str, Any], concept: Optional[Dict[str, Any]] = None) -> bool:
    text = _reference_routing_text(shot, concept, include_old_templates=False)
    if _ENVIRONMENT_ONLY_RE.search(text):
        return True
    concept = concept if isinstance(concept, dict) else (shot.get("director_scene_concept") if isinstance(shot.get("director_scene_concept"), dict) else {})
    subject = _clean_text(concept.get("visible_subject") or shot.get("visible_subject"), 160).lower()
    if subject in {"", "none", "no subject", "no visible subject", "no people", "no humans", "environment only", "location only"}:
        role = _safe_str(shot.get("scene_role") or shot.get("scene_role_summary") or shot.get("preferred_clip_role")).lower()
        if any(x in role for x in ("background", "environment", "b_roll", "b-roll", "world", "establish")):
            return True
    return False


def _shot_subject_mode(shot: Dict[str, Any], brief: Optional[Dict[str, str]] = None, concept: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    concept = concept if isinstance(concept, dict) else (shot.get("director_scene_concept") if isinstance(shot.get("director_scene_concept"), dict) else {})
    routing_text = _reference_routing_text(shot, concept, include_old_templates=False)
    visible = _has_visible_human_subject_text(routing_text)
    environment_signal = _has_environment_only_signal(shot, concept)
    if visible:
        mode = "subject_present"
        reason = "visible human subject detected in final shot concept/prompt"
    elif environment_signal:
        mode = "environment_only"
        reason = "environment-only signal with no visible human subject evidence"
    else:
        mode = "environment_only"
        reason = "no visible human subject evidence in final shot concept/prompt"
    return {
        "shot_subject_mode": mode,
        "visible_subject_detected": bool(visible),
        "environment_prompt_omitted_subjects": bool(mode == "environment_only"),
        "reference_routing_reason": reason,
        "routing_text_excerpt": _clean_text(routing_text, 360),
    }


def _reference_selection_intent_for_subject_shot(shot: Dict[str, Any], concept: Optional[Dict[str, Any]], loaded_count: int) -> tuple[int, str]:
    text = _reference_routing_text(shot, concept, include_old_templates=False)
    if loaded_count <= 0:
        return 0, "no loaded references available"
    if loaded_count == 1:
        return 1, "only one loaded reference available"

    solo = bool(_SOLO_HUMAN_RE.search(text))
    duo = bool(_DUO_HUMAN_RE.search(text))
    multi = bool(_MULTI_HUMAN_RE.search(text))

    # Duo/multi wins over broad solo wording if both appear in generated text.
    if duo:
        return min(2, loaded_count), "couple/duo refs selected"
    if multi:
        return min(5, loaded_count), "group/multi refs selected"
    if solo:
        return 1, "explicit solo ref selected"

    char_ids = [_safe_str(x) for x in _as_list(shot.get("character_ids")) if _safe_str(x)]
    if len(char_ids) >= 2:
        return min(len(char_ids), loaded_count, 5), "multiple character ids selected"
    if len(char_ids) == 1:
        return 1, "single character id selected"

    # New default: subject shots with more than one loaded reference should stay
    # group-safe unless the director explicitly asks for a solo/portrait shot.
    return min(5, loaded_count), "group/multi refs selected by default for non-solo subject"


def _reference_count_for_subject_shot(shot: Dict[str, Any], concept: Optional[Dict[str, Any]], loaded_count: int) -> int:
    wanted_count, _reason = _reference_selection_intent_for_subject_shot(shot, concept, loaded_count)
    return wanted_count

def _director_word_count(value: Any) -> int:
    return len(re.findall(r"\b[\w'’-]+\b", _safe_str(value)))


def _director_norm_for_compare(value: Any) -> str:
    text = _safe_str(value).lower()
    text = re.sub(r"\b(?:the|a|an|and|with|in|on|at|of|to|for)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()




def _director_is_full_main_idea_copy(value: Any, brief: Dict[str, str]) -> bool:
    idea = _clean_text((brief or {}).get("main_idea"), 240)
    text = _clean_text(value, 280)
    if not idea or not text or _director_word_count(idea) < 3:
        return False
    idea_norm = _director_norm_for_compare(idea)
    text_norm = _director_norm_for_compare(text)
    if not idea_norm or not text_norm:
        return False
    # Reject exact/near direct story copies, but allow shorter derived subjects
    # such as "the couple" when the full idea was "couple walking dog".
    return text_norm == idea_norm or (idea_norm in text_norm and _director_word_count(text) <= _director_word_count(idea) + 4)

def _director_split_prompt_phrases(value: Any) -> List[str]:
    text = _clean_text(value, 0)
    if not text:
        return []
    raw = re.split(r"[,;]+|(?<=[.!?])\s+", text)
    return [_clean_text(x, 500).strip(" ,.;") for x in raw if _safe_str(x).strip(" ,.;")]


def _director_strip_workflow_meta(value: Any) -> tuple[str, List[str]]:
    """Remove app/workflow language from model-facing prompts and report what changed."""
    text = _clean_text(value, 0)
    removed: List[str] = []
    if not text:
        return "", removed
    patterns = [
        r"\bGrouped\s+LTX\s+starting\s+frame\s+for\s+[^,.;:]+[:;]?\s*",
        r"\bThis\s+single\s+LTX\s+shot\s+covers\s+[^,.;]+[,.;]?\s*",
        r"\bLTX\b",
        r"\bsource\s+scenes?\s*[:#]?\s*[^,.;]+[,.;]?\s*",
        r"\bscene\s+ids?\s*[:#]?\s*[^,.;]+[,.;]?\s*",
        r"\bshot\s+ids?\s*[:#]?\s*[^,.;]+[,.;]?\s*",
        r"\bS\d{1,3}\b",
        r"\bLTX\d{1,3}\b",
        r"\b(?:previous|next)\s+shot\b[^,.;]*[,.;]?\s*",
        r"\bcontinue(?:s|d|ing)?\s+(?:from|into|the)?\s*[^,.;]*[,.;]?\s*",
        r"\bfinal\s+video\b[^,.;]*[,.;]?\s*",
        r"\bthis\s+is\s+the\s+end\b[^,.;]*[,.;]?\s*",
        r"\bgrouped\b[^,.;]*[,.;]?\s*",
        r"\bworkflow\b[^,.;]*[,.;]?\s*",
        r"\bdirector\s+plan\b[^,.;]*[,.;]?\s*",
        r"\bmetadata\b[^,.;]*[,.;]?\s*",
        r"\binternal\s+timing\b[^,.;]*[,.;]?\s*",
        r"\b(?:use|using|with|from)\s+(?:the\s+)?(?:stored\s+)?(?:group|global|subject|character)?\s*reference(?:\s+sheet|\s+sheets|\s+images)?\b[^,.;]*[,.;]?\s*",
        r"\b(?:subject|character|grouped|global)\s+reference\s+sheets?\b[^,.;]*[,.;]?\s*",
        r"\breference\s+sheets?\b[^,.;]*[,.;]?\s*",
        r"\bcontact\s+sheet\b[^,.;]*[,.;]?\s*",
        r"\bcloned\s+lineup\b[^,.;]*[,.;]?\s*",
    ]
    for pat in patterns:
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            removed.extend(_clean_text(x, 160) for x in found if _safe_str(x))
            text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmusic[-\s]?video\s+starting\s+frame\b", "cinematic still frame", text, flags=re.IGNORECASE)
    text = re.sub(r"\bstarting\s+frame\b", "still frame", text, flags=re.IGNORECASE)
    text = re.sub(r"\bimage[-\s]?to[-\s]?video\b", "motion-ready", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;")
    return text, removed


def _director_remove_duplicate_phrases(value: Any) -> tuple[str, List[str]]:
    phrases = _director_split_prompt_phrases(value)
    out: List[str] = []
    seen: List[str] = []
    removed: List[str] = []
    for phrase in phrases:
        norm = _director_norm_for_compare(phrase)
        if not norm:
            continue
        duplicate = False
        for old in seen:
            if norm == old or (len(norm) > 18 and (norm in old or old in norm)):
                duplicate = True
                break
        if duplicate:
            removed.append(phrase)
            continue
        seen.append(norm)
        out.append(phrase)
    text = ", ".join(out)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;")
    return text, removed


def _director_collapse_visual_safety(value: Any, *, add_when_needed: bool = False) -> tuple[str, List[str]]:
    text = _clean_text(value, 0)
    removed: List[str] = []
    if not text:
        return "", removed
    safety_patterns = [
        r"\bsingle coherent full[-\s]?frame scene,?\s*no text,?\s*no collage,?\s*no split[-\s]?screen,?\s*no grid\b",
        r"\bnot a collage,?\s*not split[-\s]?screen,?\s*not multi[-\s]?panel,?\s*not a contact sheet,?\s*not a grid of scenes\b",
        r"\bno text,?\s*no collage,?\s*no split[-\s]?screen,?\s*no grid\b",
        r"\bno text,?\s*no collage\b",
        r"\bno collage,?\s*no split[-\s]?screen,?\s*no multi[-\s]?panel,?\s*no contact sheet,?\s*no grid\b",
        r"\bone coherent frame,?\s*one dominant subject,?\s*one dominant location[^,.;]*\b",
    ]
    hits = 0
    for pat in safety_patterns:
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            hits += len(found)
            removed.extend(_clean_text(x, 160) for x in found if _safe_str(x))
            text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,+", ", ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;")
    # Only add the compact safety line when the caller explicitly asks for it.
    # Otherwise this helper is used as a removal/collapse pass and must not create
    # partial duplicated safety fragments.
    need_line = bool(add_when_needed)
    if need_line and _DIRECTOR_VISUAL_SAFETY_LINE.lower() not in text.lower():
        text = _join_parts([text, _DIRECTOR_VISUAL_SAFETY_LINE])
    return text, removed


def _director_limit_words_by_phrase(value: Any, max_words: int) -> str:
    text = _clean_text(value, 0)
    if max_words <= 0 or _director_word_count(text) <= max_words:
        return text
    kept: List[str] = []
    count = 0
    for phrase in _director_split_prompt_phrases(text):
        wc = _director_word_count(phrase)
        if kept and count + wc > max_words:
            break
        kept.append(phrase)
        count += wc
    if not kept:
        words = re.findall(r"\S+", text)
        return " ".join(words[:max_words]).strip(" ,.;")
    return ", ".join(kept).strip(" ,.;")


def _director_clean_final_prompt(value: Any, *, image_prompt: bool, max_words: int, add_safety: bool = False) -> tuple[str, List[str]]:
    removed_all: List[str] = []
    text, removed = _director_strip_workflow_meta(value)
    removed_all.extend(removed)
    if image_prompt:
        text = re.sub(r"\bvideo\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmusic[-\s]?video\b", "cinematic", text, flags=re.IGNORECASE)
        # Remove existing safety fragments before generic collage cleanup; otherwise
        # "no collage" can be transformed into damaged duplicate safety wording.
        text, removed = _director_collapse_visual_safety(text, add_when_needed=False)
        removed_all.extend(removed)
    text = _remove_collage_language(text)
    text, removed = _director_collapse_visual_safety(text, add_when_needed=add_safety if image_prompt else False)
    removed_all.extend(removed)
    text, removed = _director_remove_duplicate_phrases(text)
    removed_all.extend(removed)
    safety_words = _director_word_count(_DIRECTOR_VISUAL_SAFETY_LINE)
    if image_prompt and add_safety:
        base_limit = max(20, int(max_words) - safety_words)
        text = _director_limit_words_by_phrase(text, base_limit)
        if _DIRECTOR_VISUAL_SAFETY_LINE.lower() not in text.lower():
            text = _join_parts([text, _DIRECTOR_VISUAL_SAFETY_LINE])
        text, removed = _director_collapse_visual_safety(text, add_when_needed=False)
        removed_all.extend(removed)
        if _DIRECTOR_VISUAL_SAFETY_LINE.lower() not in text.lower():
            text = _join_parts([_director_limit_words_by_phrase(text, base_limit), _DIRECTOR_VISUAL_SAFETY_LINE])
    else:
        text = _director_limit_words_by_phrase(text, max_words)
    text = _clean_text(text, 0).strip(" ,.;")
    return _sentence(text), _dedupe_texts(removed_all, max_items=30, max_len=1800)


def _director_location_for_concept(shot: Dict[str, Any], brief: Dict[str, str], llm_item: Optional[Dict[str, Any]] = None) -> str:
    concept = (llm_item or {}).get("scene_concept") if isinstance((llm_item or {}).get("scene_concept"), dict) else {}
    for value in (
        concept.get("location") if isinstance(concept, dict) else "",
        (llm_item or {}).get("location") if isinstance(llm_item, dict) else "",
        shot.get("dominant_location"), shot.get("selected_location"), shot.get("main_location"),
    ):
        loc = _clean_text(value, 160)
        if loc:
            return loc
    return _single_location_world_for_scene(brief, shot, _safe_int(shot.get("index"), 1))


def _director_visible_subject_for_concept(shot: Dict[str, Any], brief: Dict[str, str], llm_item: Optional[Dict[str, Any]] = None) -> str:
    concept = (llm_item or {}).get("scene_concept") if isinstance((llm_item or {}).get("scene_concept"), dict) else {}
    candidates = [
        concept.get("visible_subject") if isinstance(concept, dict) else "",
        (llm_item or {}).get("visible_subject") if isinstance(llm_item, dict) else "",
        shot.get("visible_subject"),
        shot.get("character_labels"),
        shot.get("character_identity_prompt"),
        brief.get("characters_subjects"),
    ]
    explicit_subject_text = " ".join(
        _safe_str(", ".join(_safe_str(x) for x in v if _safe_str(x)) if isinstance(v, list) else v)
        for v in candidates[:-1]
        if _safe_str(v)
    )
    # If the director explicitly makes this an environment/location-only shot,
    # do not backfill the main idea as a fake subject.  This keeps environment
    # prompts simple instead of forcing them into anti-human negative prompt spam.
    if _has_environment_only_signal(shot, concept) and not _has_visible_human_subject_text(explicit_subject_text):
        return ""
    for value in candidates:
        if isinstance(value, list):
            value = ", ".join(_safe_str(x) for x in value if _safe_str(x))
        subject = _clean_text(value, 220)
        if subject:
            if _ENVIRONMENT_ONLY_RE.search(subject) and not _has_visible_human_subject_text(subject):
                return ""
            if not _clean_text(brief.get("characters_subjects"), 160) and _director_is_full_main_idea_copy(subject, brief):
                continue
            return subject
    inferred = _director_infer_subject_from_main_idea(brief)
    if inferred and not _director_is_full_main_idea_copy(inferred, brief):
        return inferred
    return _DIRECTOR_GENERIC_SUBJECT


def _director_infer_subject_from_main_idea(brief: Dict[str, str]) -> str:
    """Infer a short subject hint without copying the full idea sentence.

    This is intentionally conservative. It extracts only a compact visible-subject
    phrase from the user's own words and falls back to a neutral subject when the
    idea is too abstract.
    """
    idea_raw = _clean_text((brief or {}).get("main_idea"), 260)
    if not idea_raw:
        return _DIRECTOR_GENERIC_SUBJECT
    idea, _removed = _director_strip_workflow_meta(idea_raw)
    low = idea.lower()

    performer_terms = ("performer", "singer", "rapper", "dj", "band", "dancer", "artist")
    relationship_terms = ("couple", "husband", "wife", "girlfriend", "boyfriend", "lovers", "partners")
    family_terms = ("family", "mother", "father", "parents", "child", "children")
    pet_match = re.search(r"\b(dog|puppy|cat|kitten|pet|horse|bird|parrot)\b", low, flags=re.IGNORECASE)

    if any(t in low for t in relationship_terms) or re.search(r"\b(?:man|guy|boy)\b.*\b(?:woman|wife|girl)\b|\b(?:woman|wife|girl)\b.*\b(?:man|guy|boy)\b", low):
        base = "a loving couple" if any(t in low for t in ("romantic", "love", "lovers", "wife", "husband")) else "a couple"
        if pet_match:
            pet = pet_match.group(1).lower()
            if pet == "puppy":
                pet = "dog"
            elif pet == "kitten":
                pet = "cat"
            elif pet == "pet":
                pet = "pet"
            return f"{base} and their {pet}"
        return base
    if any(t in low for t in family_terms):
        if pet_match:
            return f"a family and their {pet_match.group(1).lower()}"
        return "a family group"
    if any(t in low for t in performer_terms):
        if "band" in low:
            m = re.search(r"\bband\s+of\s+(\d+)\b", low)
            return f"a pop band of {m.group(1)} performers" if m else "the band members"
        if "dj" in low:
            return "the DJ performer"
        if "dancer" in low or "dance" in low:
            return "the dancers"
        return "the performer"

    words = re.findall(r"[A-Za-zÀ-ÿ0-9'’-]+", idea)
    if not words:
        return _DIRECTOR_GENERIC_SUBJECT
    stop = {
        "a", "an", "the", "and", "or", "with", "without", "in", "on", "at", "to", "for", "from", "of", "by",
        "music", "video", "clip", "scene", "story", "about", "showing", "featuring", "where", "that", "this",
        "romantic", "romance", "dreamy", "cinematic", "pop", "deeply", "beautiful", "pretty", "cool", "flashy",
        "fast", "slow", "pace", "paced", "style", "theme", "mood", "vibe", "very", "really", "lots", "lot",
    }
    action_like = {
        "walk", "walking", "stroll", "strolling", "run", "running", "dance", "dancing", "sing", "singing",
        "perform", "performing", "drive", "driving", "ride", "riding", "fly", "flying", "travel", "travelling",
        "traveling", "go", "going", "sit", "sitting", "stand", "standing", "look", "looking", "move", "moving",
        "explore", "exploring", "meet", "meeting", "kiss", "kissing", "hug", "hugging", "play", "playing",
    }
    kept: List[str] = []
    for w in words:
        lw = w.lower().strip("'’-")
        if not lw or lw in stop or lw in action_like or lw.endswith("ing"):
            continue
        if len(lw) <= 2 and not lw.isdigit():
            continue
        kept.append(w)
        if len(kept) >= 4:
            break
    if not kept:
        return _DIRECTOR_GENERIC_SUBJECT
    subject = _clean_text(" ".join(kept), 120)
    if _director_is_full_main_idea_copy(subject, brief) or _director_word_count(subject) > 6:
        return _DIRECTOR_GENERIC_SUBJECT
    return subject or _DIRECTOR_GENERIC_SUBJECT


def _director_action_from_main_idea(brief: Dict[str, str], subject: str = "") -> str:
    idea = _clean_text((brief or {}).get("main_idea"), 220).lower()
    subj = _clean_text(subject, 120) or _DIRECTOR_GENERIC_SUBJECT
    if not idea:
        return "moves naturally through the scene with readable emotion"
    movement_map = [
        (("walk", "walking", "stroll", "strolling"), "walks through the location with natural relaxed movement"),
        (("run", "running"), "moves quickly through the location with clear forward energy"),
        (("dance", "dancing"), "performs a rhythm-focused movement beat"),
        (("sing", "singing", "vocal"), "performs with expressive face-readable energy"),
        (("drive", "driving", "ride", "riding"), "travels through the location with steady motion"),
        (("fly", "flying"), "moves through the scene with floating cinematic motion"),
        (("sit", "sitting"), "shares a quiet seated moment with readable emotion"),
        (("stand", "standing"), "holds a clear expressive pose inside the location"),
        (("explore", "exploring"), "explores the location with curious movement"),
    ]
    for keys, action in movement_map:
        if any(k in idea for k in keys):
            return action
    return f"{subj} performs one clear scene-specific action shaped by the music"


def _director_safe_role_for_brief(role_value: Any, brief: Dict[str, str]) -> str:
    role = _clean_text(role_value).lower()
    if not role:
        return "story_action"
    dance_ok = _brief_has_dance_world(brief)
    vehicle_ok = _brief_has_vehicle_world(brief)
    if "vehicle" in role and not vehicle_ok:
        return "moving_story_action"
    if "dance" in role and not dance_ok:
        return "rhythmic_story_action"
    if "performer" in role and not _brief_has_performance_world(brief):
        return "visible_subject_story_action"
    if role in {"music_video_b_roll", "b_roll", "world_broll"}:
        return "environmental_story_detail"
    return role



def _director_human_shot_type(role_value: Any, brief: Dict[str, str], *, needs_lipsync: bool = False) -> str:
    role = _director_safe_role_for_brief(role_value, brief)
    low = role.lower().replace("_", " ").strip()
    mapping = [
        ("intro", "wide establishing still"),
        ("outro", "quiet closing still"),
        ("transition", "cinematic detail cutaway"),
        ("emotional", "emotional close-medium still"),
        ("environment", "atmospheric environmental story still"),
        ("world", "atmospheric environmental story still"),
        ("rhythmic", "medium-wide rhythmic story still"),
        ("wide story", "wide story-action still"),
        ("impact", "dynamic cinematic impact still"),
        ("instrumental", "cinematic movement still"),
        ("moving story", "cinematic movement still"),
    ]
    if needs_lipsync:
        return "face-readable expressive still"
    for needle, label in mapping:
        if needle in low:
            return label
    low = re.sub(r"\s+", " ", low).strip()
    if low and low not in {"story action", "visible subject story action"}:
        return f"cinematic {low} still"
    return "cinematic medium-wide still"

def _director_fallback_action_for_shot(shot: Dict[str, Any], brief: Optional[Dict[str, str]] = None, subject: str = "") -> str:
    brief = brief or {}
    role = _director_safe_role_for_brief(shot.get("scene_role_summary") or shot.get("scene_role") or shot.get("preferred_clip_role"), brief)
    section = _clean_text(shot.get("section_summary") or shot.get("section")).lower()
    energy = _clean_text(shot.get("energy_summary") or shot.get("energy")).lower()
    idx = max(0, _safe_int(shot.get("index"), 1) - 1)
    non_vehicle_world = not _brief_has_vehicle_world(brief)
    if "vehicle_action" in role and non_vehicle_world:
        role = "instrumental_groove"
    if _safe_bool(shot.get("needs_lipsync"), False):
        pool = [
            "performs the active vocal phrase with readable face and natural body movement",
            "holds an expressive vocal moment while moving gently with the rhythm",
            "delivers the lyric with a clear face-forward performance pose",
            "sings through the beat while the surrounding scene adds motion",
        ]
    elif "transition" in role:
        pool = [
            "uses a clean rhythmic cutaway action that connects the surrounding moments",
            "shows a quick environmental detail moving with the beat",
            "captures a passing visual accent that changes the energy without changing the story",
        ]
    elif "b_roll" in role or "world" in role or "environment" in role:
        pool = [
            "reveals a strong location detail with atmosphere and motion",
            "shows the world reacting to the music through light and movement",
            "frames a cinematic environmental beat with clear depth and texture",
        ]
    elif "dance" in role and _brief_has_dance_world(brief):
        pool = [
            "performs a clean rhythm-focused dance pose with expressive movement",
            "hits a choreographed beat with clear body language and strong framing",
            "moves through a dance beat while the location adds depth",
        ]
    elif "impact" in role or "high" in energy or "chorus" in section or "drop" in section:
        pool = [
            "hits a bold visual accent with stronger motion and lighting",
            "pushes into a bigger energetic action beat",
            "creates a bright performance peak with dynamic body language",
        ]
    elif "intro" in section:
        pool = [
            "establishes the location and mood with a calm opening action",
            "introduces the visible subject through a simple readable pose",
        ]
    elif "outro" in section:
        pool = [
            "settles into a clean closing pose with calm atmosphere",
            "resolves the visual idea with a gentle final movement",
        ]
    else:
        main_action = _director_action_from_main_idea(brief, subject)
        if main_action and _director_norm_for_compare(main_action) != _director_norm_for_compare((brief or {}).get("main_idea")):
            pool = [
                main_action,
                "moves naturally through the scene with readable emotion",
                "creates a simple expressive beat with strong composition",
            ]
        else:
            pool = [
                "performs one clear story action shaped by the music",
                "moves naturally through the scene with readable emotion",
                "creates a simple expressive beat with strong composition",
            ]
    return pool[idx % len(pool)]


def _director_scene_concept_from_shot(shot: Dict[str, Any], brief: Dict[str, str], llm_item: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    llm_item = llm_item if isinstance(llm_item, dict) else {}
    raw_concept = llm_item.get("scene_concept") if isinstance(llm_item.get("scene_concept"), dict) else {}
    location = _director_location_for_concept(shot, brief, llm_item)
    subject = _director_visible_subject_for_concept(shot, brief, llm_item)
    action = _clean_text(raw_concept.get("main_action") or llm_item.get("main_action") or shot.get("main_action"), 220)
    if not action or _director_is_full_main_idea_copy(action, brief):
        action = _director_fallback_action_for_shot(shot, brief, subject)
    emotional = _clean_text(raw_concept.get("emotional_beat") or llm_item.get("emotional_beat") or shot.get("emotional_beat") or shot.get("energy_summary") or shot.get("section_summary"), 180)
    if not emotional:
        emotional = _clean_text(brief.get("style_theme"), 120) or "clear cinematic mood"
    raw_shot_type = raw_concept.get("shot_type") or llm_item.get("shot_type") or shot.get("suggested_shot_type") or shot.get("scene_role_summary")
    shot_type = _clean_text(raw_shot_type, 160)
    if not shot_type or "_" in shot_type or shot_type.lower() in {"dance_performance", "music_video_b_roll", "vehicle_action"}:
        shot_type = _director_human_shot_type(raw_shot_type, brief, needs_lipsync=_safe_bool(shot.get("needs_lipsync"), False))
    if "vehicle" in shot_type.lower() and not _brief_has_vehicle_world(brief):
        shot_type = "cinematic medium-wide still" if not _safe_bool(shot.get("needs_lipsync"), False) else "face-readable performance still"
    if "dance" in shot_type.lower() and not _brief_has_dance_world(brief):
        shot_type = "cinematic medium-wide still" if not _safe_bool(shot.get("needs_lipsync"), False) else "face-readable expressive still"
    if "performer" in shot_type.lower() and not _brief_has_performance_world(brief):
        shot_type = "cinematic medium-wide still" if not _safe_bool(shot.get("needs_lipsync"), False) else "face-readable expressive still"
    composition = _clean_text(raw_concept.get("still_image_composition") or llm_item.get("still_image_composition"), 220)
    if not composition:
        idx = max(0, _safe_int(shot.get("index"), 1) - 1)
        pool = [
            "medium-wide framing with clear foreground, subject, and background separation",
            "low-angle three-quarter framing with strong depth and readable silhouettes",
            "close medium framing with expressive face or body language and detailed lighting",
            "wide establishing composition with the subject placed clearly inside the location",
            "side-angle composition with motion lines, reflections, or environmental detail",
        ]
        composition = pool[idx % len(pool)]
    camera_motion = _clean_text(raw_concept.get("camera_motion") or llm_item.get("camera_motion") or shot.get("camera_motion"), 180)
    if not camera_motion:
        camera_motion = "smooth tracking move" if not _safe_bool(shot.get("needs_lipsync"), False) else "gentle face-readable push-in"
    video_motion = _clean_text(raw_concept.get("video_motion") or llm_item.get("video_motion"), 220)
    if not video_motion or _director_prompt_contains_meta(video_motion) or _director_is_full_main_idea_copy(video_motion, brief):
        # Fallback concepts must not inherit the old LTX/template video prompt,
        # because that is where copy-pasted main ideas and workflow words leaked
        # into the actual LTX command. Build clean motion from concept fields.
        video_motion = f"{action}, {camera_motion} through {location}"
    notes = _clean_text(raw_concept.get("notes") or llm_item.get("notes") or llm_item.get("director_notes") or shot.get("director_notes"), 300)
    concept = {
        "id": _safe_str(shot.get("id")),
        "location": location,
        "visible_subject": subject,
        "main_action": action,
        "emotional_beat": emotional,
        "shot_type": shot_type,
        "still_image_composition": composition,
        "camera_motion": camera_motion,
        "video_motion": video_motion,
        "notes": notes,
    }
    clean: Dict[str, Any] = {}
    removed_all: List[str] = []
    for key, value in concept.items():
        if key == "id":
            clean[key] = _safe_str(value)
            continue
        cleaned, removed = _director_strip_workflow_meta(value)
        removed_all.extend(removed)
        clean[key] = _clean_text(cleaned, 500)
    if removed_all:
        clean["concept_removed_meta_phrases"] = _dedupe_texts(removed_all, max_items=20, max_len=1200)
    return clean


def _director_style_summary(brief: Dict[str, str]) -> str:
    style = _clean_text(brief.get("style_theme"), 220)
    if style:
        return style
    return "cinematic polished lighting"


def _director_lighting_for_concept(concept: Dict[str, Any], brief: Dict[str, str]) -> str:
    mood = _clean_text(concept.get("emotional_beat"), 160) or _director_style_summary(brief)
    if any(x in mood.lower() for x in ("night", "neon", "dark")):
        return "controlled contrast lighting with readable silhouettes"
    if any(x in mood.lower() for x in ("dream", "romantic", "soft", "emotional")):
        return "soft cinematic light with warm depth and gentle atmosphere"
    if any(x in mood.lower() for x in ("high", "drop", "impact", "fast", "energy")):
        return "bright dynamic lighting with strong depth and crisp subject separation"
    return "cinematic lighting with clear depth and polished detail"


def _director_identity_hint_for_prompt(shot: Dict[str, Any], *, image_prompt: bool) -> str:
    bits = []
    if image_prompt:
        bits.extend([shot.get("group_identity_prompt"), shot.get("character_image_identity_prompt"), shot.get("character_consistency_prompt"), shot.get("character_clone_guard_prompt")])
    else:
        bits.extend([shot.get("group_identity_prompt"), shot.get("character_identity_prompt")])
    text = "; ".join(_dedupe_texts(bits, max_items=3, max_len=260 if image_prompt else 180))
    return _clean_text(text, 260 if image_prompt else 180)


def _director_compile_image_prompt(shot: Dict[str, Any], brief: Dict[str, str], concept: Dict[str, Any]) -> tuple[str, List[str]]:
    subject_info = _shot_subject_mode(shot, brief, concept)
    environment_only = subject_info.get("shot_subject_mode") == "environment_only"
    subject = _clean_text(concept.get("visible_subject"), 220)
    if not environment_only and not subject:
        subject = _DIRECTOR_GENERIC_SUBJECT
    location = _clean_text(concept.get("location"), 180) or _single_location_world_for_scene(brief, shot, _safe_int(shot.get("index"), 1))
    action = _clean_text(concept.get("main_action"), 220) or _director_fallback_action_for_shot(shot, brief, subject)
    shot_type = _clean_text(concept.get("shot_type"), 140) or "cinematic still frame"
    composition = _clean_text(concept.get("still_image_composition"), 220) or "clear full-frame composition with readable subject and location"
    style = _director_style_summary(brief)
    lighting = _director_lighting_for_concept(concept, brief)
    identity = "" if environment_only else _director_identity_hint_for_prompt(shot, image_prompt=True)
    if "background/environment shot" in identity.lower() and _director_norm_for_compare(subject) != _director_norm_for_compare(_DIRECTOR_GENERIC_SUBJECT):
        identity = ""
    if environment_only:
        composition_env = re.sub(r"wide establishing composition with the subject placed clearly inside the location", "wide establishing composition with clear depth and foreground/background separation", composition, flags=re.IGNORECASE)
        composition_env = re.sub(r"medium-wide framing with clear foreground, subject, and background separation", "medium-wide framing with clear foreground and background separation", composition_env, flags=re.IGNORECASE)
        composition_env = re.sub(r"close medium framing with expressive face or body language and detailed lighting", "close environmental detail framing with textured lighting", composition_env, flags=re.IGNORECASE)
        composition_env = re.sub(r"\bsubject\b", "location detail", composition_env, flags=re.IGNORECASE)
        composition_env = re.sub(r"expressive\s+face\s+or\s+body\s+language", "architectural and environmental detail", composition_env, flags=re.IGNORECASE)
        composition_env = re.sub(r"readable\s+faces?", "readable environmental detail", composition_env, flags=re.IGNORECASE)
        parts = [
            f"{shot_type} inside {location}" if location else shot_type,
            composition_env,
            style,
            lighting,
            _clean_text(concept.get("emotional_beat"), 160),
            _DIRECTOR_VISUAL_SAFETY_LINE,
        ]
    else:
        parts = [
            f"{shot_type} of {subject}",
            action,
            f"inside {location}" if location else "inside the selected visual world",
            composition,
            style,
            lighting,
            identity,
            _DIRECTOR_VISUAL_SAFETY_LINE,
        ]
    return _director_clean_final_prompt(_join_parts(parts), image_prompt=True, max_words=80, add_safety=True)


def _director_compile_video_prompt(shot: Dict[str, Any], brief: Dict[str, str], concept: Dict[str, Any]) -> tuple[str, List[str]]:
    subject_info = _shot_subject_mode(shot, brief, concept)
    environment_only = subject_info.get("shot_subject_mode") == "environment_only"
    subject = _clean_text(concept.get("visible_subject"), 180)
    if not environment_only and not subject:
        subject = _DIRECTOR_GENERIC_SUBJECT
    location = _clean_text(concept.get("location"), 160) or _single_location_world_for_scene(brief, shot, _safe_int(shot.get("index"), 1))
    action = _clean_text(concept.get("main_action"), 220) or _director_fallback_action_for_shot(shot, brief, subject)
    camera = _clean_text(concept.get("camera_motion"), 180) or "smooth camera movement"
    mood = _clean_text(concept.get("emotional_beat"), 120) or _director_style_summary(brief)
    identity = "" if environment_only else _director_identity_hint_for_prompt(shot, image_prompt=False)
    if "background/environment shot" in identity.lower() and _director_norm_for_compare(subject) != _director_norm_for_compare(_DIRECTOR_GENERIC_SUBJECT):
        identity = ""
    if environment_only:
        parts = [
            f"{camera} through {location}" if location else f"{camera} through the selected visual world",
            "light, reflections, and environmental details evolve with the rhythm",
            mood,
        ]
    else:
        action_line = action if _director_norm_for_compare(subject) and _director_norm_for_compare(subject) in _director_norm_for_compare(action) else f"{subject} {action}"
        lipsync = "face and mouth stay readable during the active vocal phrase" if _safe_bool(shot.get("needs_lipsync"), False) else "motion follows the beat without lyric-mouthing"
        parts = [
            f"{camera} as {action_line}",
            f"through {location}" if location else "through the selected visual world",
            "background light and environmental details evolve with the rhythm",
            mood,
            identity,
            lipsync,
        ]
    return _director_clean_final_prompt(_join_parts(parts), image_prompt=False, max_words=70, add_safety=False)


def _director_compile_timestamped_prompt(shot: Dict[str, Any], brief: Dict[str, str], concept: Dict[str, Any], video_prompt: str) -> tuple[str, List[str]]:
    duration = max(0.1, _safe_float(shot.get("duration"), 0.0))
    needs_lipsync = _safe_bool(shot.get("needs_lipsync"), False)
    existing = _safe_str(shot.get("director_timestamped_video_prompt") or shot.get("timestamped_video_prompt"))
    removed_all: List[str] = []
    if _safe_bool(shot.get("is_microclip"), False) and existing:
        text, removed = _director_clean_final_prompt(existing, image_prompt=False, max_words=140, add_safety=False)
        removed_all.extend(removed)
        return _normalize_director_timestamp(text, duration=duration, needs_lipsync=needs_lipsync, fallback=text), removed_all
    location = _clean_text(concept.get("location"), 160)
    action = _clean_text(concept.get("main_action"), 180) or _director_fallback_action_for_shot(shot, brief, _clean_text(concept.get("visible_subject"), 160))
    camera = _clean_text(concept.get("camera_motion"), 140) or "smooth camera movement"
    first = _director_limit_words_by_phrase(video_prompt or f"{action}, {camera} in {location}", 38)
    beats = [f"0:00 - {first}"]
    safe_times = [t for t in _ltx_beat_times(duration) if t >= 4.75 and duration - t >= 3.0]
    if safe_times:
        second = _join_parts([action, camera, "same location continuity", _clean_text(concept.get("emotional_beat"), 100)])
        second, removed = _director_clean_final_prompt(second, image_prompt=False, max_words=34, add_safety=False)
        removed_all.extend(removed)
        beats.append(f"{_ltx_time_label(safe_times[0])} - {second}")
    text = ". ".join(beats)
    text, removed = _director_clean_final_prompt(text, image_prompt=False, max_words=120, add_safety=False)
    removed_all.extend(removed)
    return _normalize_director_timestamp(text, duration=duration, needs_lipsync=needs_lipsync, fallback=text), _dedupe_texts(removed_all, max_items=30, max_len=1600)


def _director_prompt_repeated_phrases(value: Any) -> List[str]:
    phrases = _director_split_prompt_phrases(value)
    counts: Dict[str, int] = {}
    pretty: Dict[str, str] = {}
    for phrase in phrases:
        norm = _director_norm_for_compare(phrase)
        if len(norm) < 12:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        pretty.setdefault(norm, phrase)
    return [pretty[k] for k, v in counts.items() if v > 1]


def _director_prompt_contains_meta(value: Any) -> bool:
    low = _safe_str(value).lower()
    if not low:
        return False
    if re.search(r"\b(?:ltx|ltx\d{1,3}|s\d{1,3})\b", low, flags=re.IGNORECASE):
        return True
    return any(term in low for term in _DIRECTOR_FORBIDDEN_IMAGE_TERMS)


def _director_image_repeats_main_idea(img: str, brief: Dict[str, str]) -> bool:
    idea = _clean_text(brief.get("main_idea"), 240)
    if not idea or _director_word_count(idea) < 3:
        return False
    return _director_norm_for_compare(idea) in _director_norm_for_compare(img)


def _director_remove_full_main_idea_text(value: Any, brief: Dict[str, str]) -> tuple[str, List[str]]:
    text = _clean_text(value, 0)
    idea = _clean_text((brief or {}).get("main_idea"), 240)
    if not text or not idea or _director_word_count(idea) < 3:
        return text, []
    removed: List[str] = []
    pattern = re.compile(re.escape(idea), flags=re.IGNORECASE)
    if pattern.search(text):
        text = pattern.sub("scene-specific visual action", text)
        removed.append("full main idea sentence")
    # Catch normalized exact phrase copies that differ only by punctuation.
    if _director_norm_for_compare(idea) in _director_norm_for_compare(text):
        words = re.findall(r"\S+", text)
        idea_words = set(_director_norm_for_compare(idea).split())
        kept = [w for w in words if _director_norm_for_compare(w) not in idea_words]
        if len(kept) >= max(4, len(words) // 3):
            text = " ".join(kept)
            removed.append("normalized full main idea words")
    return _clean_text(text, 0), _dedupe_texts(removed, max_items=5, max_len=300)


def _director_strip_visual_safety_for_video(value: Any) -> tuple[str, List[str]]:
    text = _clean_text(value, 0)
    removed: List[str] = []
    patterns = [
        r"\bsingle coherent full[-\s]?frame scene,?\s*no text,?\s*no collage,?\s*no split[-\s]?screen,?\s*no grid\b",
        r"\bone coherent frame\b",
        r"\bone dominant subject\b",
        r"\bone dominant location(?::\s*[^,.;]+)?\b",
        r"\bnot a collage\b",
        r"\bno collage\b",
        r"\bnot split[-\s]?screen\b",
        r"\bno split[-\s]?screen\b",
        r"\bnot multi[-\s]?panel\b",
        r"\bno multi[-\s]?panel\b",
        r"\bnot a contact sheet\b",
        r"\bno contact sheet\b",
        r"\bnot a grid of scenes\b",
        r"\bno grid\b",
        r"\bno text\b",
    ]
    for pat in patterns:
        found = re.findall(pat, text, flags=re.IGNORECASE)
        if found:
            removed.extend(_clean_text(x, 160) for x in found if _safe_str(x))
            text = re.sub(pat, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*,+", ", ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.;")
    return text, _dedupe_texts(removed, max_items=20, max_len=1000)


def _select_start_image_prompt_source(shot: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> tuple[str, str]:
    payload = payload if isinstance(payload, dict) else {}
    override = _safe_str(payload.get("image_prompt_override"))
    if override:
        return override, "image_prompt_override"
    for key in ("director_image_prompt", "image_prompt", "template_image_prompt"):
        value = _safe_str(shot.get(key))
        if value:
            return value, key
    return "", ""


def _sanitize_final_start_image_prompt_for_model(raw_prompt: Any, *, brief: Dict[str, str], shot: Dict[str, Any], lyrics: Any = "", image_model: str = "") -> tuple[str, List[str]]:
    removed_all: List[str] = []
    text = _sanitize_prompt_no_visible_text(raw_prompt, lyrics, image_prompt=True, max_len=1800)
    text, removed = _director_strip_workflow_meta(text)
    removed_all.extend(removed)
    text, removed = _director_remove_full_main_idea_text(text, brief)
    removed_all.extend(removed)
    text = _enforce_single_location_without_negative_words(text, brief or {}, shot or {}, max_len=1800)
    if not _safe_bool((shot or {}).get("needs_lipsync"), False):
        text = _strip_non_lipsync_vocal_language(text)
    # Keep only one compact visual safety line, after all older guards/reference text were cleaned.
    text, removed = _director_clean_final_prompt(text, image_prompt=True, max_words=80, add_safety=True)
    removed_all.extend(removed)
    # One more pass catches safety words added by older helpers before the final compiler.
    text, removed = _director_remove_full_main_idea_text(text, brief)
    removed_all.extend(removed)
    text, removed = _director_clean_final_prompt(text, image_prompt=True, max_words=80, add_safety=True)
    removed_all.extend(removed)
    text = re.sub(r"\b(?:in|on|at|inside|through)\s+([,.;])", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;])", r"\1", text).strip(" ,.;")
    return _sentence(text), _dedupe_texts(removed_all, max_items=40, max_len=2200)


def _select_ltx_video_prompt_source(shot: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> tuple[str, str, bool]:
    payload = payload if isinstance(payload, dict) else {}
    override = _safe_str(
        payload.get("clip_prompt_override")
        or payload.get("video_prompt_override")
        or payload.get("director_timestamped_video_prompt_override")
        or payload.get("timestamped_video_prompt_override")
        or payload.get("clip_prompt")
    )
    if override:
        return override, "clip_prompt_override", bool(_LTX_TIMESTAMP_RE.search(override))
    for key in ("director_timestamped_video_prompt", "director_video_prompt", "video_prompt", "template_timestamped_video_prompt", "template_video_prompt"):
        value = _safe_str(shot.get(key))
        if value:
            return value, key, "timestamped" in key or bool(_LTX_TIMESTAMP_RE.search(value))
    return "", "", False


def _sanitize_final_ltx_prompt_for_model(raw_prompt: Any, *, brief: Dict[str, str], shot: Dict[str, Any], prompt_is_timestamped: bool = False) -> tuple[str, List[str]]:
    removed_all: List[str] = []
    text = _clean_text(raw_prompt, 2400)
    text, removed = _director_strip_workflow_meta(text)
    removed_all.extend(removed)
    text, removed = _director_remove_full_main_idea_text(text, brief)
    removed_all.extend(removed)
    text, removed = _director_strip_visual_safety_for_video(text)
    removed_all.extend(removed)
    text = _sanitize_prompt_no_visible_text(text, (shot or {}).get("lyrics"), image_prompt=False, max_len=2200)
    if not _safe_bool((shot or {}).get("needs_lipsync"), False):
        text = _strip_non_lipsync_vocal_language(text)
    max_words = 120 if prompt_is_timestamped or bool(_LTX_TIMESTAMP_RE.search(text)) else 70
    text, removed = _director_clean_final_prompt(text, image_prompt=False, max_words=max_words, add_safety=False)
    removed_all.extend(removed)
    text, removed = _director_strip_visual_safety_for_video(text)
    removed_all.extend(removed)
    text, removed = _director_remove_full_main_idea_text(text, brief)
    removed_all.extend(removed)
    if prompt_is_timestamped or bool(_LTX_TIMESTAMP_RE.search(text)):
        text = _normalize_director_timestamp(text, duration=_safe_float((shot or {}).get("duration"), 0.0), needs_lipsync=_safe_bool((shot or {}).get("needs_lipsync"), False), fallback=text)
        text = _sanitize_ltx_timestamped_prompt(text, _safe_float((shot or {}).get("duration"), 0.0))
    text = re.sub(r"\b(?:in|on|at|inside|through)\s+([,.;])", r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([,.;])", r"\1", text).strip(" ,.;")
    return _sentence(_clean_text(text, 2200)), _dedupe_texts(removed_all, max_items=40, max_len=2200)


def _director_final_validation_warnings(shot: Dict[str, Any], brief: Dict[str, str]) -> List[str]:
    warn = _director_validation_warnings(shot)
    img = _safe_str(shot.get("director_image_prompt"))
    low = img.lower()
    if _director_prompt_contains_meta(img):
        warn.append("director_image_prompt contains workflow/meta wording")
    if _director_word_count(img) > 90:
        warn.append(f"director_image_prompt over 90 words ({_director_word_count(img)})")
    if _director_image_repeats_main_idea(img, brief):
        warn.append("director_image_prompt repeats the main idea sentence")
    reps = _director_prompt_repeated_phrases(img)
    if reps:
        warn.append("director_image_prompt repeats phrase(s): " + "; ".join(reps[:3]))
    collage_mentions = len(re.findall(r"\bnot\s+a\s+collage\b|\bno\s+collage\b", low, flags=re.IGNORECASE))
    if collage_mentions > 1:
        warn.append("collage safety wording appears more than once")
    return warn


def _director_neighbor_key(shot: Dict[str, Any]) -> str:
    concept = shot.get("director_scene_concept") if isinstance(shot.get("director_scene_concept"), dict) else {}
    parts = [concept.get("location"), concept.get("visible_subject"), concept.get("main_action"), concept.get("shot_type")]
    return " | ".join(_director_norm_for_compare(x) for x in parts if _safe_str(x))


def _director_add_neighbor_similarity_warnings(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [dict(s) for s in shots]
    for i in range(1, len(out)):
        prev_key = _director_neighbor_key(out[i - 1])
        cur_key = _director_neighbor_key(out[i])
        if not prev_key or not cur_key:
            continue
        prev_parts = set(prev_key.split())
        cur_parts = set(cur_key.split())
        if not prev_parts or not cur_parts:
            continue
        overlap = len(prev_parts & cur_parts) / max(1, min(len(prev_parts), len(cur_parts)))
        exact = prev_key == cur_key
        if exact or overlap >= 0.86:
            msg = "neighboring shot is nearly the same location + subject + action + shot_type"
            for idx in (i - 1, i):
                warnings = [_safe_str(x) for x in _as_list(out[idx].get("director_warnings")) if _safe_str(x)]
                if msg not in warnings:
                    warnings.append(msg)
                out[idx]["director_warnings"] = warnings
                out[idx]["director_validation_status"] = "repaired_or_fallback"
    return out


def _director_mark_final_validation(shot: Dict[str, Any], brief: Dict[str, str], extra_warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    out = dict(shot)
    warn = _director_final_validation_warnings(out, brief)
    if extra_warnings:
        warn.extend([_clean_text(w, 400) for w in extra_warnings if _safe_str(w)])
    seen = set()
    clean_warn: List[str] = []
    for w in warn:
        ww = _clean_text(w, 500)
        if ww and ww not in seen:
            seen.add(ww)
            clean_warn.append(ww)
    out["director_warnings"] = clean_warn
    out["director_validation_status"] = "ok" if not clean_warn else "repaired_or_fallback"
    return out


def _director_prompt_is_clean_for_preserve(value: Any, *, brief: Dict[str, str], image_prompt: bool) -> bool:
    text = _safe_str(value)
    if not text:
        return False
    if _director_prompt_contains_meta(text):
        return False
    if _director_is_full_main_idea_copy(text, brief):
        return False
    if image_prompt and _prompt_has_visible_text_request(text, ""):
        return False
    if image_prompt and "background/environment shot" in text.lower():
        return False
    if image_prompt and _director_prompt_repeated_phrases(text):
        return False
    if image_prompt and len(re.findall(r"\bnot\s+a\s+collage\b|\bno\s+collage\b", text, flags=re.IGNORECASE)) > 1:
        return False
    limit = 90 if image_prompt else 95
    if _director_word_count(text) > limit:
        return False
    return True


def _director_compile_shot_prompts(shot: Dict[str, Any], brief: Dict[str, str]) -> Dict[str, Any]:
    out = dict(shot)
    raw_llm = out.get("director_raw_llm_output") if isinstance(out.get("director_raw_llm_output"), dict) else {}
    concept = out.get("director_scene_concept") if isinstance(out.get("director_scene_concept"), dict) else {}
    if not concept:
        concept = _director_scene_concept_from_shot(out, brief, raw_llm)
    else:
        concept = _director_scene_concept_from_shot({**out, **concept}, brief, raw_llm)
    out["director_scene_concept"] = concept
    if _safe_str(concept.get("location")):
        out["dominant_location"] = _safe_str(concept.get("location"))
        out["selected_location"] = _safe_str(concept.get("location"))
    removed_all: List[str] = []

    compiled_img, removed = _director_compile_image_prompt(out, brief, concept)
    removed_all.extend(removed)
    existing_img = _safe_str(out.get("director_llm_image_prompt_original") or out.get("director_image_prompt"))
    if _safe_str(out.get("llm_director_status", "")).startswith("success") and _director_prompt_is_clean_for_preserve(existing_img, brief=brief, image_prompt=True):
        img, removed = _sanitize_final_start_image_prompt_for_model(existing_img, brief=brief, shot=out, lyrics=out.get("lyrics"))
        if _director_prompt_is_clean_for_preserve(img, brief=brief, image_prompt=True):
            out["director_image_prompt_source"] = "local_llm_preserved"
        else:
            img = compiled_img
            out["director_image_prompt_source"] = "compiled_scene_concept"
        removed_all.extend(removed)
    else:
        img = compiled_img
        out["director_image_prompt_source"] = "compiled_scene_concept"

    compiled_vid, removed = _director_compile_video_prompt(out, brief, concept)
    removed_all.extend(removed)
    existing_vid = _safe_str(out.get("director_llm_video_prompt_original") or out.get("director_video_prompt"))
    if _safe_str(out.get("llm_director_status", "")).startswith("success") and _director_prompt_is_clean_for_preserve(existing_vid, brief=brief, image_prompt=False):
        vid, removed = _sanitize_final_ltx_prompt_for_model(existing_vid, brief=brief, shot=out, prompt_is_timestamped=False)
        if not _safe_str(vid):
            vid = compiled_vid
            out["director_video_prompt_source"] = "compiled_scene_concept"
        else:
            out["director_video_prompt_source"] = "local_llm_preserved"
        removed_all.extend(removed)
    else:
        vid = compiled_vid
        out["director_video_prompt_source"] = "compiled_scene_concept"

    existing_ts = _safe_str(out.get("director_llm_timestamped_video_prompt_original") or out.get("director_timestamped_video_prompt"))
    if _safe_str(out.get("llm_director_status", "")).startswith("success") and existing_ts and _director_prompt_is_clean_for_preserve(existing_ts, brief=brief, image_prompt=False):
        ts, removed = _sanitize_final_ltx_prompt_for_model(existing_ts, brief=brief, shot=out, prompt_is_timestamped=True)
        removed_all.extend(removed)
        if not _safe_str(ts):
            ts, removed = _director_compile_timestamped_prompt(out, brief, concept, vid)
            removed_all.extend(removed)
            out["director_timestamped_prompt_source"] = "compiled_scene_concept"
        else:
            out["director_timestamped_prompt_source"] = "local_llm_preserved"
    else:
        ts, removed = _director_compile_timestamped_prompt(out, brief, concept, vid)
        removed_all.extend(removed)
        out["director_timestamped_prompt_source"] = "compiled_scene_concept"

    out["director_image_prompt"] = img
    out["director_video_prompt"] = vid
    out["director_timestamped_video_prompt"] = _sanitize_ltx_timestamped_prompt(ts, _safe_float(out.get("duration"), 0.0))
    out["prompt_sanitizer_removed_phrases"] = _dedupe_texts(_as_list(out.get("prompt_sanitizer_removed_phrases")) + removed_all + _as_list(concept.get("concept_removed_meta_phrases")), max_items=40, max_len=2200)
    out = _director_mark_final_validation(out, brief)
    return out

def _director_compile_all_shot_prompts(shots: List[Dict[str, Any]], brief: Dict[str, str]) -> List[Dict[str, Any]]:
    compiled = [_director_compile_shot_prompts(shot, brief) for shot in shots]
    compiled = _director_add_neighbor_similarity_warnings(compiled)
    return compiled


def _director_prompt_debug_records(shots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for shot in shots:
        if not isinstance(shot, dict):
            continue
        records.append({
            "id": _safe_str(shot.get("id")),
            "source_scene_ids": _as_list(shot.get("source_scene_ids")),
            "raw_scene_concept": shot.get("director_scene_concept") if isinstance(shot.get("director_scene_concept"), dict) else {},
            "raw_llm_output": shot.get("director_raw_llm_output") if isinstance(shot.get("director_raw_llm_output"), dict) else {},
            "raw_failed_llm_output": _safe_str(shot.get("raw_failed_llm_output")),
            "parse_error": _safe_str(shot.get("parse_error")),
            "retry_attempts": _as_list(shot.get("llm_retry_attempts") or shot.get("retry_attempts")),
            "final_reason_for_fallback": _safe_str(shot.get("final_reason_for_fallback") or shot.get("fallback_reason")),
            "llm_director_status": _safe_str(shot.get("llm_director_status") or shot.get("director_source_status")),
            "final_image_prompt": _safe_str(shot.get("director_image_prompt")),
            "final_video_prompt": _safe_str(shot.get("director_video_prompt")),
            "final_timestamped_video_prompt": _safe_str(shot.get("director_timestamped_video_prompt")),
            "image_word_count": _director_word_count(shot.get("director_image_prompt")),
            "video_word_count": _director_word_count(shot.get("director_video_prompt")),
            "removed_duplicate_meta_phrases": _as_list(shot.get("prompt_sanitizer_removed_phrases")),
            "validation_warnings": _as_list(shot.get("director_warnings")),
        })
    return records


def _director_llm_quality_summary(shots: List[Dict[str, Any]], *, local_requested: bool) -> Dict[str, Any]:
    total = len([s for s in shots if isinstance(s, dict)])
    success = 0
    fallback = 0
    per_shot: List[Dict[str, Any]] = []
    for shot in [s for s in shots if isinstance(s, dict)]:
        status = _safe_str(shot.get("llm_director_status") or shot.get("director_source_status"))
        is_success = status.startswith("success")
        is_fallback = ("fallback" in status) or (local_requested and not is_success)
        success += 1 if is_success else 0
        fallback += 1 if is_fallback else 0
        per_shot.append({
            "id": _safe_str(shot.get("id")),
            "llm_director_status": status or ("not_requested" if not local_requested else "unknown"),
            "fallback_reason": _safe_str(shot.get("final_reason_for_fallback") or shot.get("fallback_reason")),
            "parse_error": _safe_str(shot.get("parse_error")),
        })
    rate = (success / total) if total else 0.0
    if not local_requested:
        quality = "template_only"
    elif total == 0:
        quality = "fail_no_shots"
    elif fallback / max(1, total) > 0.25:
        quality = "warning_too_many_fallbacks"
    elif fallback:
        quality = "warning_some_fallbacks"
    else:
        quality = "ok"
    return {
        "llm_success_count": success,
        "llm_fallback_count": fallback,
        "llm_success_rate": round(rate, 4),
        "director_quality_status": quality,
        "llm_success_fallback_per_shot": per_shot,
    }


def _write_director_prompt_debug_reports(out_dir: Path, director_plan: Dict[str, Any]) -> tuple[str, str]:
    records = _director_prompt_debug_records([s for s in _as_list(director_plan.get("shots")) if isinstance(s, dict)])
    payload = {
        "version": _VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_ltx_shot_plan_path": _safe_str(director_plan.get("source_ltx_shot_plan_path")),
        "director_backend": _safe_str(director_plan.get("director_backend")),
        "requested_director_backend": _safe_str(director_plan.get("requested_director_backend")),
        "creative_brief": director_plan.get("creative_brief") if isinstance(director_plan.get("creative_brief"), dict) else {},
        "shot_count": len(records),
        "acceptance_summary": _director_prompt_acceptance_summary(records, director_plan.get("creative_brief") if isinstance(director_plan.get("creative_brief"), dict) else {}),
        "llm_summary": director_plan.get("llm_summary") if isinstance(director_plan.get("llm_summary"), dict) else {},
        "location_debug": director_plan.get("location_debug") if isinstance(director_plan.get("location_debug"), dict) else _director_location_debug_summary([s for s in _as_list(director_plan.get("shots")) if isinstance(s, dict)], director_plan.get("creative_brief") if isinstance(director_plan.get("creative_brief"), dict) else {}),
        "shots": records,
    }
    json_path = out_dir / "musicclip_ltx_prompt_debug_report.json"
    txt_path = out_dir / "musicclip_ltx_prompt_debug_report.txt"
    _write_json_file(json_path, payload)
    lines = [
        "FrameVision Music Clip Creator - LTX Prompt Debug Report",
        f"Created: {payload['created_at']}",
        f"Director backend: {payload['director_backend']}",
        f"Shots: {payload['shot_count']}",
        "",
        "LLM summary:",
    ]
    for key, value in (payload.get("llm_summary") or {}).items():
        lines.append(f"- {key}: {value}")
    loc_debug = payload.get("location_debug") or {}
    if loc_debug:
        lines.extend([
            "",
            "Location summary:",
            f"- full_location_pool: {loc_debug.get('full_location_pool')}",
            f"- locations_used_count: {loc_debug.get('locations_used_count')}",
            f"- unused_locations: {loc_debug.get('unused_locations')}",
            f"- repeated_location_counts: {loc_debug.get('repeated_location_counts')}",
            f"- one_location_per_shot_validation: {loc_debug.get('one_location_per_shot_validation')}",
        ])
    lines.extend(["", "Acceptance summary:"])
    for key, value in payload["acceptance_summary"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    for rec in records:
        lines.append(f"[{rec['id']}]")
        lines.append("Scene concept: " + json.dumps(rec.get("raw_scene_concept") or {}, ensure_ascii=False))
        lines.append(f"LLM status: {rec.get('llm_director_status') or ''}")
        if rec.get("retry_attempts"):
            lines.append("Retry attempts: " + json.dumps(rec.get("retry_attempts") or [], ensure_ascii=False)[:2000])
        if rec.get("parse_error"):
            lines.append("Parse error: " + _safe_str(rec.get("parse_error"))[:1000])
        if rec.get("raw_failed_llm_output"):
            lines.append("Raw failed LLM output: " + _safe_str(rec.get("raw_failed_llm_output"))[:2000])
        if rec.get("final_reason_for_fallback"):
            lines.append("Final fallback reason: " + _safe_str(rec.get("final_reason_for_fallback"))[:1000])
        if rec.get("raw_llm_output"):
            lines.append("Raw LLM output: " + json.dumps(rec.get("raw_llm_output") or {}, ensure_ascii=False)[:2000])
        lines.append(f"Image ({rec['image_word_count']} words): {rec['final_image_prompt']}")
        lines.append(f"Video ({rec['video_word_count']} words): {rec['final_video_prompt']}")
        lines.append(f"Timestamped: {rec['final_timestamped_video_prompt']}")
        if rec.get("removed_duplicate_meta_phrases"):
            lines.append("Removed: " + "; ".join(_safe_str(x) for x in rec.get("removed_duplicate_meta_phrases")[:12]))
        if rec.get("validation_warnings"):
            lines.append("Warnings: " + "; ".join(_safe_str(x) for x in rec.get("validation_warnings")[:12]))
        lines.append("")
    txt_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(txt_path), str(json_path)


def _director_prompt_acceptance_summary(records: List[Dict[str, Any]], brief: Dict[str, str]) -> Dict[str, Any]:
    imgs = [_safe_str(r.get("final_image_prompt")) for r in records]
    normalized_imgs = [_director_norm_for_compare(x) for x in imgs if x]
    full_idea = _director_norm_for_compare(brief.get("main_idea"))
    return {
        "shots_checked": len(records),
        "unique_image_prompts": len(set(normalized_imgs)),
        "all_image_prompts_unique": len(set(normalized_imgs)) == len(normalized_imgs),
        "image_prompts_with_ltx": sum(1 for x in imgs if re.search(r"\bltx\b", x, flags=re.IGNORECASE)),
        "image_prompts_with_scene_ids": sum(1 for x in imgs if re.search(r"\bS\d{1,3}\b", x, flags=re.IGNORECASE)),
        "image_prompts_with_workflow_words": sum(1 for x in imgs if _director_prompt_contains_meta(x)),
        "image_prompts_repeating_full_main_idea": sum(1 for x in imgs if full_idea and full_idea in _director_norm_for_compare(x)),
        "image_prompts_with_repeated_not_a_collage": sum(1 for x in imgs if len(re.findall(r"\bnot\s+a\s+collage\b", x, flags=re.IGNORECASE)) > 1),
        "locations_used": sorted({ _clean_text((r.get("raw_scene_concept") or {}).get("location"), 120) for r in records if isinstance(r.get("raw_scene_concept"), dict) and _safe_str((r.get("raw_scene_concept") or {}).get("location")) }),
    }


def demo_ltx_prompt_director_debug_report(output_dir: Any = "") -> Dict[str, Any]:
    """Small internal prompt-pipeline demo for acceptance testing.

    It creates a temporary director plan from sample shot data and writes the same
    debug reports as the real bridge, without generating images/video or touching
    FrameVision UI code.
    """
    out_dir = Path(_safe_str(output_dir) or (_project_root() / "output" / "musicclip_prompt_debug_demo")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    brief = _normalize_creative_brief({
        "main_idea": "romantic couple walking dog",
        "style_theme": "romantic dreamy cinematic pop",
        "characters_subjects": "",
        "locations_world": "park, bridge, café street, lakeside, promenade, old town alley, flower market, beach path, rooftop terrace, lantern street, quiet garden, riverside stairs",
        "camera_choreography": "gentle tracking, close emotional details, wide romantic movement",
    })
    locations = _location_candidates_from_brief(brief) or ["park", "bridge", "café street", "lakeside", "promenade"]
    roles = ["intro_establishing_shot", "romantic_detail", "environment_story_beat", "rhythmic_story_action", "transition_cutaway", "emotional_closeup", "wide_story_action", "outro_closing_shot"]
    shots: List[Dict[str, Any]] = []
    for idx in range(12):
        loc = locations[idx % len(locations)]
        shots.append({
            "id": f"LTX{idx + 1:02d}",
            "index": idx + 1,
            "duration": 3.0,
            "song_start": idx * 3.0,
            "song_end": (idx + 1) * 3.0,
            "dominant_location": loc,
            "scene_role_summary": roles[idx % len(roles)],
            "energy_summary": "soft" if idx in (0, 5, 7) else "medium",
            "needs_lipsync": False,
            "image_prompt": f"Grouped LTX starting frame for S{idx+1:02d}; romantic couple walking dog, {loc}, dreamy cinematic pop, not a collage, not a collage",
            "video_prompt": f"This single LTX shot covers S{idx+1:02d}; continue from previous shot with romantic couple walking dog in {loc}",
            "timestamped_video_prompt": f"0:00 - Continue from previous shot, grouped LTX motion in {loc}.",
            "negative_prompt": "text, logos, low quality",
        })
    compiled = _director_compile_all_shot_prompts([_director_template_shot(s, brief) for s in shots], brief)
    plan = {
        "version": _VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_ltx_shot_plan_path": "demo_only",
        "creative_brief": brief,
        "director_backend": "demo_template_cleanup",
        "requested_director_backend": "demo",
        "shots": compiled,
    }
    txt_path, json_path = _write_director_prompt_debug_reports(out_dir, plan)
    return {"ok": True, "debug_report_text_path": txt_path, "debug_report_json_path": json_path, "acceptance_summary": _director_prompt_acceptance_summary(_director_prompt_debug_records(compiled), brief)}

def _normalize_director_timestamp(value: Any, *, duration: float, needs_lipsync: bool, fallback: str = "") -> str:
    text = _director_clean_prompt_text(value, 2400)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip(" ,.;")
    if not text:
        text = _director_clean_prompt_text(fallback, 2400)
    if not text:
        return ""
    text = re.sub(r"(?:^|\s)(\d{1,2})\s*[:.]\s*(\d{2})\s*[-–—:]", lambda m: f" {int(m.group(1))}:{m.group(2)} -", text).strip()
    if not re.match(r"^0:00\s*-", text):
        # Keep the model output but force clip-relative first timestamp.
        text = re.sub(r"^\d{1,2}:\d{2}\s*-\s*", "", text).strip()
        text = "0:00 - " + text
    # Hard guard: no bullets/markdown/list line leftovers.
    text = re.sub(r"\s*[-*•]\s+(?=\d{1,2}:\d{2})", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text[-1:] not in ".!?":
        text += "."
    return text


def _director_template_timestamp(shot: Dict[str, Any], brief: Dict[str, str]) -> str:
    duration = max(0.1, _safe_float(shot.get("duration"), 0.0))
    needs_lipsync = _safe_bool(shot.get("needs_lipsync"), False)
    subject = _director_subject(brief, needs_lipsync)
    world = _single_location_world_for_scene(brief, shot, _safe_int(shot.get("index"), 1))
    action = _director_action_theme(brief)
    times = _ltx_beat_times(duration)
    # Keep 2-3 second rhythm; trim absurdly long prompt plans.
    if len(times) > 7:
        times = times[:7]
    cap = _major_scene_block_limit(duration)
    shot_hint = _safe_str(shot.get("scene_role_summary") or shot.get("section_summary") or action)
    same_setup = _same_setup_variation_pool(brief, needs_lipsync, shot_hint)
    lyric_start_mode = _shot_lyric_start_mode(shot)
    first_lyric_offset = 0.0
    lyric_items = [x for x in _as_list(shot.get("clip_relative_lyrics")) if isinstance(x, dict)]
    if lyric_items:
        first_lyric_offset = max(0.0, min(_safe_float(x.get("start"), 0.0) for x in lyric_items))
    if cap <= 1:
        if needs_lipsync:
            if lyric_start_mode == "delayed":
                variants = [
                    f"{subject} sets up the performance in one clear setup, face visible but not mouthing lyrics yet in {world}",
                    f"around {_ltx_time_label(first_lyric_offset)} the vocal phrase begins and {subject} starts singing with readable face and mouth",
                    "side/profile angle in the same setup while the performer moves with the beat",
                    "closer face, hands, reflection, or light-detail angle keeps the same setup alive",
                    "final same-setup angle resolves toward the next cut",
                ]
            elif lyric_start_mode == "mid_phrase":
                variants = [
                    f"{subject} continues the vocal phrase naturally mid-phrase in one clear setup, face readable in {world}",
                    "side/profile angle in the same setup while the performer continues the performance",
                    "closer face, hands, reflection, or light-detail angle keeps the same setup alive",
                    "gentle push-in or pull-back keeps continuity without jumping to a new location",
                    "final same-setup angle resolves toward the next cut",
                ]
            else:
                variants = [
                    f"{subject} performs the active vocal phrase in one clear setup, face readable in {world}",
                    "side/profile angle in the same setup while the performer moves with the beat",
                    "closer face, hands, reflection, or light-detail angle keeps the same setup alive",
                    "gentle push-in or pull-back keeps continuity without jumping to a new location",
                    "final same-setup angle resolves toward the next cut",
                ]
        else:
            variants = [
                f"{action} starts immediately in one dominant setup in {world}",
                "side angle and passing details keep the same setup moving with the beat",
                "close detail variation adds rhythm without switching location",
                "camera movement, reflections, and environmental motion keep the shot alive",
                "final same-setup hero angle resolves toward the next cut",
            ]
    elif needs_lipsync:
        mixed = _safe_int(shot.get("non_vocal_scene_count"), 0) > 0
        if mixed:
            if lyric_start_mode == "delayed":
                variants = [
                    f"{subject} moves with the beat in {world}, face visible but not mouthing lyrics yet",
                    f"around {_ltx_time_label(first_lyric_offset)} the vocal phrase begins and the performer starts singing with readable face and mouth",
                    "the vocal moment ends and the subject moves with the beat without mouthing lyrics",
                    "the camera uses a connected action or detail angle rather than a hard new location",
                    "the shot resolves with music-video action and natural expression",
                ]
            elif lyric_start_mode == "mid_phrase":
                variants = [
                    f"{subject} continues the vocal phrase naturally mid-phrase, face readable in {world}",
                    "the camera stays connected to the performance rather than acting like a fresh lyric start",
                    "the vocal moment ends and the subject moves with the beat without mouthing lyrics",
                    "the camera uses a connected action or detail angle rather than a hard new location",
                    "the shot resolves with music-video action and natural expression",
                ]
            else:
                variants = [
                    f"{subject} performs the short active vocal phrase at the start, face readable in {world}",
                    "the vocal moment ends and the subject moves with the beat without mouthing lyrics",
                    "the camera uses a connected action or detail angle rather than a hard new location",
                    "one natural story progression may open up if the clip has room",
                    "the shot resolves with music-video action and natural expression",
                ]
        else:
            if lyric_start_mode == "delayed":
                variants = [
                    f"{subject} prepares the performance in {world}, moving with the beat but not singing yet",
                    f"around {_ltx_time_label(first_lyric_offset)} the vocal phrase begins and {subject} starts singing with readable face and mouth",
                    "detail, reflection, or body movement variation keeps continuity around the performer",
                    "one wider performance angle may open up without cramming multiple locations",
                    "the shot resolves toward the next cut while keeping the same visual world",
                ]
            elif lyric_start_mode == "mid_phrase":
                variants = [
                    f"{subject} continues the vocal phrase naturally mid-phrase, face and mouth clearly visible in {world}",
                    f"the camera shifts to a readable side/profile angle in the same setup while {subject} continues performing",
                    "detail, reflection, or body movement variation keeps continuity around the performer",
                    "one wider performance angle may open up without cramming multiple locations",
                    "the shot resolves toward the next cut while keeping the same visual world",
                ]
            else:
                variants = [
                    f"{subject} starts the vocal performance immediately, face and mouth clearly visible in {world}",
                    f"the camera shifts to a readable side/profile angle in the same setup while {subject} keeps performing",
                    "detail, reflection, or body movement variation keeps continuity around the performer",
                    "one wider performance angle may open up without cramming multiple locations",
                    "the shot resolves toward the next cut while keeping the same visual world",
                ]
    else:
        variants = [
            f"{action} starts immediately in {world}",
            "the camera drops into a dynamic detail angle with strong motion and beat-driven energy",
            "the action expands only when the story needs it, otherwise staying in the same setup",
            "a connected wider angle may reveal more of the world without jumping locations",
            "movement intensifies through a curve, pass-by, dance hit, or visual impact moment",
            "the scene drives forward toward the next cut without forced lyric-mouthing",
        ]
    if len(variants) < len(times):
        variants.extend(same_setup)
    beats = []
    for i, t in enumerate(times):
        beats.append(f"{_ltx_time_label(t)} - {variants[min(i, len(variants) - 1)]}.")
    return _normalize_director_timestamp(" ".join(beats), duration=duration, needs_lipsync=needs_lipsync)


def _director_template_image_prompt(shot: Dict[str, Any], brief: Dict[str, str]) -> str:
    concept = _director_scene_concept_from_shot(shot, brief)
    prompt, _removed = _director_compile_image_prompt(shot, brief, concept)
    return prompt


def _director_template_video_prompt(shot: Dict[str, Any], brief: Dict[str, str]) -> str:
    concept = _director_scene_concept_from_shot(shot, brief)
    prompt, _removed = _director_compile_video_prompt(shot, brief, concept)
    return prompt


def _director_template_negative_prompt(shot: Dict[str, Any]) -> str:
    base = _director_clean_prompt_text(shot.get("negative_prompt"), 900)
    extras = ["blurry", "low quality", "broken anatomy", "unreadable text", "random unrelated scene", "inconsistent character design", _anti_collage_negative_terms()]
    if _safe_bool(shot.get("needs_lipsync"), False):
        extras += ["hidden face", "covered mouth", "face out of frame", "chaotic camera during lipsync", "mouth not visible"]
    text = _join_parts([base] + extras)
    return _director_clean_prompt_text(_negative_prompt_with_no_visible_text(text, max_len=1200), 1200)


def _director_template_shot(shot: Dict[str, Any], brief: Dict[str, str], notes: str = "Template cleanup fallback.") -> Dict[str, Any]:
    out = dict(shot)
    out["template_image_prompt"] = _sanitize_prompt_no_visible_text(_director_clean_prompt_text(shot.get("image_prompt"), 1800), shot.get("lyrics"), image_prompt=True)
    out["template_video_prompt"] = _director_clean_prompt_text(shot.get("video_prompt"), 1800)
    out["template_timestamped_video_prompt"] = _normalize_director_timestamp(
        shot.get("timestamped_video_prompt"),
        duration=_safe_float(shot.get("duration"), 0.0),
        needs_lipsync=_safe_bool(shot.get("needs_lipsync"), False),
    )
    out["director_image_prompt"] = _director_template_image_prompt(shot, brief)
    out["director_video_prompt"] = _director_template_video_prompt(shot, brief)
    fallback_ts = _director_template_timestamp(shot, brief)
    # Use a regenerated timestamped line because the old one is often repetitive.
    out["director_timestamped_video_prompt"] = _normalize_director_timestamp(
        fallback_ts,
        duration=_safe_float(shot.get("duration"), 0.0),
        needs_lipsync=_safe_bool(shot.get("needs_lipsync"), False),
    )
    out["director_negative_prompt"] = _director_template_negative_prompt(shot)
    boundary_warnings = [str(x) for x in _as_list(out.get("lyric_boundary_warnings")) if str(x).strip()]
    boundary_note = " ".join(boundary_warnings[:3])
    out["director_notes"] = _clean_text(_join_parts([notes, boundary_note]), 900)
    if not _safe_bool(out.get("needs_lipsync"), False):
        for _key in ("template_image_prompt", "template_video_prompt", "template_timestamped_video_prompt", "director_image_prompt", "director_video_prompt", "director_timestamped_video_prompt"):
            out[_key] = _strip_non_lipsync_vocal_language(out.get(_key))
    out = _apply_scene_block_limiter_to_director_shot(out, brief)
    out["director_scene_concept"] = _director_scene_concept_from_shot(out, brief)
    out["director_raw_llm_output"] = {}
    out.setdefault("llm_director_status", "template_cleanup")
    out.setdefault("fallback_reason", notes)
    out = _sanitize_ltx_timestamp_fields_in_item(out, keys=["template_timestamped_video_prompt", "director_timestamped_video_prompt"])
    return out


def _planner_settings_path_for_bridge(root: Any = None) -> Path:
    try:
        base = Path(root).resolve() if root else _project_root()
    except Exception:
        base = _project_root()
    return (base / "presets" / "setsave" / "planner_settings.json").resolve()


def _read_planner_llama_settings_for_bridge(root: Any = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    try:
        path = _planner_settings_path_for_bridge(root)
        if path.is_file():
            data = _read_json_file(path)
    except Exception:
        data = {}
    if isinstance(overrides, dict):
        data.update({k: v for k, v in overrides.items() if v is not None})
    runner = _safe_str(data.get("own_llama_runner_path"))
    if not runner:
        try:
            bin_dir = (_project_root() / "presets" / "bin").resolve()
            for name in ("llama-server.exe", "llama-server", "server.exe", "server"):
                cand = bin_dir / name
                if cand.is_file():
                    runner = str(cand)
                    break
        except Exception:
            pass
    return {
        "enabled": _safe_bool(data.get("own_llama_enabled"), False),
        "runner_path": runner,
        "model_path": _safe_str(data.get("own_llama_model_path")),
        "template_kind": _safe_str(data.get("own_llama_template_kind"), "smart") or "smart",
        "template_value": _safe_str(data.get("own_llama_template_value")),
        "ctx_size": _safe_int(data.get("own_llama_ctx_size"), 8192) or 8192,
        "top_p": _safe_float(data.get("own_llama_top_p"), 0.9) or 0.9,
    }


def _resolve_llama_server_executable(path: str) -> str:
    raw = os.path.abspath(_safe_str(path)) if path else ""
    if not raw:
        return ""
    base = os.path.basename(raw).lower()
    if "server" in base:
        return raw
    folder = os.path.dirname(raw)
    for name in ("llama-server.exe", "llama-server", "server.exe", "server"):
        cand = os.path.join(folder, name)
        if os.path.isfile(cand):
            return cand
    return raw


def _director_llama_template_args(model_path: str, kind: str, value: str) -> List[str]:
    kind = _safe_str(kind).lower() or "smart"
    value = _safe_str(value)
    if kind == "jinja":
        return ["--jinja"]
    if kind == "builtin" and value:
        return ["--chat-template", value]
    if kind == "smart":
        hay = f"{os.path.basename(model_path).lower()} {str(model_path).lower()}"
        guesses = [
            (("qwen",), "chatml"),
            (("llama-3", "llama3", "meta-llama-3"), "llama3"),
            (("llama-4", "llama4"), "llama4"),
            (("mistral", "mixtral"), "mistral-v7"),
            (("gemma",), "gemma"),
            (("phi-4", "phi4"), "phi4"),
            (("deepseek",), "deepseek"),
        ]
        for needles, template in guesses:
            if any(n in hay for n in needles):
                return ["--chat-template", template]
    return []


class _DirectorLlamaClient:
    def __init__(self, cfg: Dict[str, Any], *, log_path: Optional[Path] = None):
        self.cfg = dict(cfg or {})
        self.runner_path = _resolve_llama_server_executable(_safe_str(self.cfg.get("runner_path")))
        self.model_path = os.path.abspath(_safe_str(self.cfg.get("model_path"))) if self.cfg.get("model_path") else ""
        self.ctx_size = max(1024, _safe_int(self.cfg.get("ctx_size"), 8192))
        self.top_p = _safe_float(self.cfg.get("top_p"), 0.9) or 0.9
        self.proc = None
        self.port = 0
        self.base_url = ""
        self.log_path = log_path or (_project_root() / "logs" / "musicclip_ltx_director_llama.log")

    @staticmethod
    def _pick_free_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
        sock.close()
        return port

    @staticmethod
    def _http_get_json(url: str, timeout: float = 8.0) -> tuple:
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
    def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 300.0) -> tuple:
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
            return "\n".join(p for p in parts if p.strip()).strip()
        if isinstance(content, dict):
            return _safe_str(content.get("text", content.get("content", "")))
        return _safe_str(content)

    def start(self) -> None:
        if not self.runner_path or not os.path.isfile(self.runner_path):
            raise RuntimeError(f"llama-server executable not found: {self.runner_path or '[empty]'}")
        if not self.model_path or not os.path.isfile(self.model_path):
            raise RuntimeError(f"GGUF model not found: {self.model_path or '[empty]'}")
        if self.proc is not None and self.proc.poll() is None:
            return
        self.port = self._pick_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        base_args = [
            "-m", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(self.ctx_size),
        ]
        template_args = _director_llama_template_args(self.model_path, self.cfg.get("template_kind"), self.cfg.get("template_value"))
        attempts = [base_args + ["--reasoning-budget", "0"] + template_args, base_args + template_args]
        if template_args:
            attempts.append(base_args + ["--reasoning-budget", "0"])
            attempts.append(base_args)
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        last_error = ""
        for args in attempts:
            try:
                if self.proc is not None and self.proc.poll() is None:
                    self.stop()
            except Exception:
                pass
            with self.log_path.open("w", encoding="utf-8", errors="replace") as f:
                f.write("[FrameVision Music Clip Creator LTX Director llama-server]\n")
                f.write(f"runner={self.runner_path}\nmodel={self.model_path}\nargs={json.dumps(args, ensure_ascii=False)}\n\n")
            log_handle = self.log_path.open("a", encoding="utf-8", errors="replace")
            try:
                self.proc = subprocess.Popen(
                    [self.runner_path] + args,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(self.runner_path),
                    creationflags=creationflags,
                )
            finally:
                try:
                    log_handle.close()
                except Exception:
                    pass
            started = time.time()
            while time.time() - started <= 240:
                if self.proc.poll() is not None:
                    last_error = f"llama-server exited before ready; see {self.log_path}"
                    break
                try:
                    code, payload = self._http_get_json(f"{self.base_url}/health", timeout=4.0)
                    if code == 200:
                        return
                    if code == 503:
                        last_error = _safe_str(((payload or {}).get("error") or {}).get("message")) or "Loading model..."
                except Exception:
                    pass
                time.sleep(1.0)
        raise RuntimeError(last_error or "Timed out waiting for llama-server.")

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

    def generate(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.55, max_tokens: int = 4096) -> str:
        if not self.base_url or self.proc is None or self.proc.poll() is not None:
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
        code, data = self._http_post_json(f"{self.base_url}/v1/chat/completions", payload, timeout=420.0)
        if code >= 400:
            msg = ((data or {}).get("error") or {}).get("message") or f"HTTP {code}"
            raise RuntimeError(str(msg))
        choices = (data or {}).get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned by llama-server.")
        return self._extract_message_text((choices[0] or {}).get("message") or {}).strip()


def _extract_json_payload_from_text(text: str) -> Any:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty model response")
    try:
        return json.loads(raw)
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except Exception:
            pass
    starts = [i for i in (raw.find("{"), raw.find("[")) if i >= 0]
    if not starts:
        raise ValueError("no JSON object or array found")
    start = min(starts)
    for end in range(len(raw), start, -1):
        chunk = raw[start:end].strip()
        if not chunk:
            continue
        if chunk[-1] not in "}]":
            continue
        try:
            return json.loads(chunk)
        except Exception:
            continue
    raise ValueError("could not parse JSON from model response")


def _llm_director_system_prompt() -> str:
    return (
        "You are a practical music-video director creating structured scene concepts for still-image and motion prompts. "
        "Do not rewrite bulky template prompts. Use the creative brief only as context. "
        "The user's main idea/story must never be copied as a full sentence into every image prompt. "
        "Return strict JSON only. No markdown. No commentary. Preserve shot IDs, order, timing, durations, needs_lipsync flags, and source scene IDs. "
        "For each shot, create a scene_concept object with: id, location, visible_subject, main_action, emotional_beat, shot_type, still_image_composition, camera_motion, video_motion, notes. "
        "Image prompt content must be visual-only: visible subject, one dominant location, action/pose, style, lighting, composition, mood. "
        "Do not include app/workflow words in image prompts: LTX, Grouped LTX, shot IDs, scene IDs, previous shot, next shot, continue, final video, grouped starting frame, workflow, director plan, metadata, or internal timing. "
        "For motion prompts describe only motion and camera movement, not app workflow. "
        "Lyrics may guide mood, facial expression, dance energy, story intent, and performance intensity only; they must not become visible text or overlays. "
        "Never put lyric words, subtitles, captions, signs, logos, typography, letters, or written words into image prompts. "
        "Normal shots must stay one coherent full-frame scene with one dominant location; no collage, split-screen, contact sheet, grid, or multi-location layout unless explicitly marked montage."
    )


def _llm_director_user_prompt(batch: List[Dict[str, Any]], brief: Dict[str, str]) -> str:
    small_shots = []
    for shot in batch:
        small_shots.append({
            "id": _safe_str(shot.get("id")),
            "duration": _safe_float(shot.get("duration"), 0.0),
            "needs_lipsync": _safe_bool(shot.get("needs_lipsync"), False),
            "source_scene_ids": _as_list(shot.get("source_scene_ids")),
            "section_summary": _safe_str(shot.get("section_summary")),
            "energy_summary": _safe_str(shot.get("energy_summary")),
            "scene_role_summary": _director_safe_role_for_brief(shot.get("scene_role_summary"), brief),
            "dominant_location": _safe_str(shot.get("dominant_location")),
            "is_montage_style": _safe_bool(shot.get("is_montage_style"), False),
            "lyrics_context_only": _clean_text(shot.get("lyrics"), 500),
            "major_scene_block_limit": _safe_int(shot.get("major_scene_block_limit"), _major_scene_block_limit(_safe_float(shot.get("duration"), 0.0))),
            "major_scene_block_rule": _safe_str(shot.get("major_scene_block_rule") or _scene_block_guidance(_safe_float(shot.get("duration"), 0.0))),
        })
    brief_context = {
        "main_idea_source_context_only": _clean_text(brief.get("main_idea"), 500),
        "style_theme": _clean_text(brief.get("style_theme"), 500),
        "characters_subjects": _clean_text(brief.get("characters_subjects"), 500),
        "locations_world_pool": _clean_text(brief.get("locations_world"), 1600),
        "location_pool_list": _location_candidates_from_brief(brief),
        "camera_choreography": _clean_text(brief.get("camera_choreography"), 500),
        "negative_prompt": _clean_text(brief.get("negative_prompt"), 500),
    }
    payload = {
        "task": "Create structured scene concepts first. Optional prompt strings may be returned, but FrameVision will compile final prompts from scene_concept.",
        "creative_brief_context_only": brief_context,
        "rules": [
            "Do not copy main_idea_source_context_only as a full sentence into any image prompt or visible_subject field.",
            "If characters_subjects is empty, infer visible_subject from the shot concept; otherwise use 'the intended visible subject', never the whole story sentence.",
            "Each scene_concept must pick one dominant location from locations_world_pool or the shot dominant_location.",
            "Vary location, action, framing and shot_type between neighboring shots.",
            "For image prompts use only visual still-image content. Do not mention LTX, video, shot IDs, scene IDs, previous, next, continue, grouped, final video, workflow, director plan, metadata, or internal timing.",
            "For motion prompts describe motion and camera movement only, with no app/workflow jargon.",
            "No visible text, subtitles, captions, signs, logos, typography, lyric words, or on-screen words.",
            "Normal shots are one coherent full-frame scene, one dominant location, no collage/split-screen/grid/contact sheet.",
            "Timestamped prompts, if returned, must be single-line, clip-relative, start with 0:00 -, and use safe 5-second intervals only when at least 3 seconds remain after the timestamp.",
        ],
        "shots": small_shots,
        "output_schema": {
            "shots": [
                {
                    "id": "LTX01",
                    "scene_concept": {
                        "id": "LTX01",
                        "location": "one dominant location",
                        "visible_subject": "who or what is visible",
                        "main_action": "one clear action or pose",
                        "emotional_beat": "mood/emotional role",
                        "shot_type": "framing/shot type",
                        "still_image_composition": "still-image composition only",
                        "camera_motion": "camera movement for video",
                        "video_motion": "subject/environment motion for video",
                        "notes": "short notes"
                    },
                    "director_image_prompt": "optional short visual-only still image prompt",
                    "director_video_prompt": "optional motion prompt",
                    "director_timestamped_video_prompt": "optional single-line timestamped motion prompt",
                    "director_negative_prompt": "string",
                    "director_notes": "short string"
                }
            ]
        },
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)



def _merge_llm_director_fields(base: Dict[str, Any], llm_item: Dict[str, Any], brief: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    if not isinstance(llm_item, dict):
        return base
    out = dict(base)
    brief = brief if isinstance(brief, dict) else {}
    lyrics = base.get("lyrics")
    visible_llm_item = {k: v for k, v in llm_item.items() if not str(k).startswith("_")}
    out["director_raw_llm_output"] = visible_llm_item
    out["llm_director_status"] = _safe_str(llm_item.get("_llm_director_status") or "success")
    out["llm_retry_attempts"] = _as_list(llm_item.get("_llm_retry_attempts"))
    out["fallback_reason"] = ""
    out["director_scene_concept"] = _director_scene_concept_from_shot(base, brief, llm_item)
    for key in ("director_image_prompt", "director_video_prompt", "director_negative_prompt", "director_notes"):
        value = _director_clean_prompt_text(llm_item.get(key), 2200 if key != "director_negative_prompt" else 1200)
        if value:
            if key == "director_image_prompt":
                value = _sanitize_prompt_no_visible_text(value, lyrics, image_prompt=True, max_len=2200)
                value, removed = _director_clean_final_prompt(value, image_prompt=True, max_words=90, add_safety=True)
                out["prompt_sanitizer_removed_phrases"] = _dedupe_texts(_as_list(out.get("prompt_sanitizer_removed_phrases")) + removed, max_items=40, max_len=2200)
                if len(value) < 30 or re.match(r"^(?:clean image-only|no visible|without written)", value.lower()):
                    value = _safe_str(base.get("director_image_prompt")) or value
                out["director_llm_image_prompt_original"] = value
            elif key == "director_video_prompt":
                value = _sanitize_prompt_no_visible_text(value, lyrics, image_prompt=False, max_len=2200)
                value, removed = _director_clean_final_prompt(value, image_prompt=False, max_words=80, add_safety=False)
                out["prompt_sanitizer_removed_phrases"] = _dedupe_texts(_as_list(out.get("prompt_sanitizer_removed_phrases")) + removed, max_items=40, max_len=2200)
                if len(value) < 25:
                    value = _safe_str(base.get("director_video_prompt")) or value
                out["director_llm_video_prompt_original"] = value
            elif key == "director_negative_prompt":
                value = _negative_prompt_with_no_visible_text(value, max_len=1200)
            out[key] = value
    ts = _normalize_director_timestamp(
        llm_item.get("director_timestamped_video_prompt"),
        duration=_safe_float(base.get("duration"), 0.0),
        needs_lipsync=_safe_bool(base.get("needs_lipsync"), False),
        fallback=base.get("director_timestamped_video_prompt"),
    )
    if ts:
        ts = _sanitize_prompt_no_visible_text(ts, base.get("lyrics"), image_prompt=False, max_len=2400)
        ts, removed = _director_clean_final_prompt(ts, image_prompt=False, max_words=140, add_safety=False)
        out["prompt_sanitizer_removed_phrases"] = _dedupe_texts(_as_list(out.get("prompt_sanitizer_removed_phrases")) + removed, max_items=40, max_len=2200)
        out["director_llm_timestamped_video_prompt_original"] = ts
        out["director_timestamped_video_prompt"] = ts
    if not _safe_str(out.get("director_notes")):
        out["director_notes"] = "Local LLM scene-concept director output."
    if not _safe_bool(out.get("needs_lipsync"), False):
        for _key in ("director_image_prompt", "director_video_prompt", "director_timestamped_video_prompt"):
            out[_key] = _strip_non_lipsync_vocal_language(out.get(_key))
    out = _apply_scene_block_limiter_to_director_shot(out, base if isinstance(base, dict) else {})
    out = _sanitize_ltx_timestamp_fields_in_item(out, keys=["director_timestamped_video_prompt"])
    return out

def _director_validation_warnings(shot: Dict[str, Any]) -> List[str]:
    warn: List[str] = []
    img = _safe_str(shot.get("director_image_prompt"))
    vid = _safe_str(shot.get("director_video_prompt"))
    ts = _safe_str(shot.get("director_timestamped_video_prompt"))
    for key, value in (("director_image_prompt", img), ("director_video_prompt", vid), ("director_timestamped_video_prompt", ts)):
        if not value:
            warn.append(f"{key} is empty")
        low = value.lower()
        if "```" in value:
            warn.append(f"{key} contained markdown/code fences")
        if "here is" in low[:80] or "here are" in low[:80]:
            warn.append(f"{key} contained assistant preamble text")
        if "chain of thought" in low or "reasoning:" in low:
            warn.append(f"{key} contained reasoning text")
        if "//" in value:
            warn.append(f"{key} may contain JSON-style comments")
    if "\n" in ts or "\r" in ts:
        warn.append("director_timestamped_video_prompt was not single-line")
    if "0:00" not in ts:
        warn.append("director_timestamped_video_prompt did not include clip-relative 0:00")
    # Absolute song timestamps are hard to prove, but anything starting beyond 0 can be suspicious.
    if re.match(r"^\s*(?:[1-9]|\d{2,}):\d{2}\s*-", ts):
        warn.append("director_timestamped_video_prompt appeared to start with an absolute/non-zero timestamp")
    if _prompt_has_visible_text_request(img, shot.get("lyrics")):
        warn.append("director_image_prompt contained visible lyric/text wording and should be repaired")
    return warn


def _director_mark_validation(shot: Dict[str, Any], extra_warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    out = dict(shot)
    warn = _director_validation_warnings(out)
    if extra_warnings:
        warn.extend([_clean_text(w, 400) for w in extra_warnings if _safe_str(w)])
    # De-duplicate while preserving order.
    seen = set()
    clean_warn: List[str] = []
    for w in warn:
        ww = _clean_text(w, 500)
        if ww and ww not in seen:
            seen.add(ww)
            clean_warn.append(ww)
    out["director_warnings"] = clean_warn
    out["director_validation_status"] = "ok" if not clean_warn else "repaired_or_fallback"
    return out


def _llm_director_repair_prompt(shot: Dict[str, Any], bad_item: Dict[str, Any], validation_warnings: List[str]) -> str:
    payload = {
        "repair_task": "Repair this one LTX director rewrite. Return strict JSON object only, no markdown, no commentary.",
        "validation_warnings": validation_warnings,
        "rules": [
            "Return only the fields: id, scene_concept, director_image_prompt, director_video_prompt, director_timestamped_video_prompt, director_negative_prompt, director_notes.",
            "scene_concept must include: location, visible_subject, main_action, emotional_beat, shot_type, still_image_composition, camera_motion, video_motion, notes.",
            "Do not copy the full main idea/story sentence into director_image_prompt or visible_subject.",
            "director_timestamped_video_prompt must be one single-line string.",
            "director_timestamped_video_prompt must start with 0:00 - and use clip-relative timestamps only.",
            "No bullets, no code fences, no preamble, no reasoning text.",
            "Do not include quoted lyric words or visible text/subtitles/captions/signs/logos/typography in director_image_prompt.",
            "Lyrics can influence mood and performance only; written lyric text must not appear in the image.",
        ],
        "trusted_shot_context": {
            "id": _safe_str(shot.get("id")),
            "duration": _safe_float(shot.get("duration"), 0.0),
            "needs_lipsync": _safe_bool(shot.get("needs_lipsync"), False),
            "lyrics": _clean_text(shot.get("lyrics"), 700),
            "template_image_prompt": _clean_text(shot.get("template_image_prompt") or shot.get("image_prompt"), 900),
            "template_video_prompt": _clean_text(shot.get("template_video_prompt") or shot.get("video_prompt"), 900),
            "template_timestamped_video_prompt": _clean_text(shot.get("template_timestamped_video_prompt") or shot.get("timestamped_video_prompt"), 1000),
        },
        "bad_output": bad_item if isinstance(bad_item, dict) else {},
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)

def _llm_director_json_repair_prompt(batch: List[Dict[str, Any]], raw_text: str, parse_error: str) -> str:
    payload = {
        "repair_task": "Repair the malformed local-LLM director JSON. Return strict JSON only, no markdown, no commentary.",
        "parse_error": _clean_text(parse_error, 600),
        "expected_ids": [_safe_str(s.get("id")) for s in batch if isinstance(s, dict)],
        "rules": [
            "Return a JSON object with one key: shots.",
            "shots must be a list of objects, one per expected id when possible.",
            "Preserve ids exactly.",
            "Do not invent app/workflow wording.",
            "Do not copy full main idea sentences.",
        ],
        "malformed_text": _clean_text(raw_text, 6000),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _llm_director_parse_items(raw: str) -> Dict[str, Dict[str, Any]]:
    parsed = _extract_json_payload_from_text(raw)
    items = parsed.get("shots") if isinstance(parsed, dict) else parsed
    if isinstance(items, dict):
        items = list(items.values())
    if not isinstance(items, list):
        raise ValueError("LLM response did not contain a shots list.")
    by_id: Dict[str, Dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and _safe_str(item.get("id")):
            by_id[_safe_str(item.get("id"))] = item
    if not by_id:
        raise ValueError("LLM response contained no shot objects with ids.")
    return by_id


def _llm_director_new_diag(shot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": _safe_str(shot.get("id")),
        "llm_director_status": "pending",
        "retry_attempts": [],
        "raw_failed_llm_output": "",
        "parse_error": "",
        "final_reason_for_fallback": "",
    }


def _llm_director_record_attempt(diag: Dict[str, Any], *, stage: str, ok: bool, raw: str = "", error: str = "") -> None:
    attempts = _as_list(diag.get("retry_attempts"))
    attempts.append({
        "stage": _clean_text(stage, 120),
        "ok": bool(ok),
        "error": _clean_text(error, 600),
        "raw_excerpt": _clean_text(raw, 1200) if raw and not ok else "",
    })
    diag["retry_attempts"] = attempts[-8:]
    if raw and not ok:
        diag["raw_failed_llm_output"] = _clean_text(raw, 4000)
    if error and not ok:
        diag["parse_error"] = _clean_text(error, 1000)


def _run_llm_director_batches(shots: List[Dict[str, Any]], brief: Dict[str, str], cfg: Dict[str, Any], warnings: List[str], progress_callback: Optional[Callable[[str], None]] = None) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    if not _safe_bool(cfg.get("enabled"), False):
        raise RuntimeError("Local LLM is not enabled in planner_settings.json.")
    client = _DirectorLlamaClient(cfg)
    results: Dict[str, Dict[str, Any]] = {}
    diagnostics: Dict[str, Dict[str, Any]] = {_safe_str(s.get("id")): _llm_director_new_diag(s) for s in shots if isinstance(s, dict) and _safe_str(s.get("id"))}

    def _emit_progress(message: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(str(message or ""))
            except Exception:
                pass

    def _mark_success(sid: str, item: Dict[str, Any], stage: str) -> None:
        diag = diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []})
        diag["llm_director_status"] = "success" if stage == "batch_4" else f"success_after_{stage}"
        diag["final_reason_for_fallback"] = ""
        item["_llm_director_status"] = diag["llm_director_status"]
        item["_llm_retry_attempts"] = _as_list(diag.get("retry_attempts"))

    def _mark_fallback(sid: str, reason: str) -> None:
        diag = diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []})
        diag["llm_director_status"] = "fallback"
        diag["final_reason_for_fallback"] = _clean_text(reason, 1000) or "local LLM returned no valid rewrite"

    def _generate_and_parse(batch: List[Dict[str, Any]], stage: str, *, temperature: float, max_tokens: int) -> Dict[str, Dict[str, Any]]:
        raw = ""
        ids = [_safe_str(s.get("id")) for s in batch if isinstance(s, dict)]
        try:
            raw = client.generate(
                _llm_director_system_prompt(),
                _llm_director_user_prompt(batch, brief),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            by_id = _llm_director_parse_items(raw)
            for sid in ids:
                _llm_director_record_attempt(diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []}), stage=stage, ok=True)
            return by_id
        except Exception as exc:
            first_error = str(exc)
            for sid in ids:
                _llm_director_record_attempt(diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []}), stage=stage, ok=False, raw=raw, error=first_error)
            # One repair attempt keeps a malformed but useful model answer from discarding a whole batch.
            if raw:
                try:
                    repair_raw = client.generate(
                        _llm_director_system_prompt(),
                        _llm_director_json_repair_prompt(batch, raw, first_error),
                        temperature=0.1,
                        max_tokens=max(1200, min(max_tokens, 3600)),
                    )
                    repaired = _llm_director_parse_items(repair_raw)
                    for sid in ids:
                        _llm_director_record_attempt(diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []}), stage=f"{stage}_json_repair", ok=True)
                    return repaired
                except Exception as repair_exc:
                    for sid in ids:
                        _llm_director_record_attempt(diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []}), stage=f"{stage}_json_repair", ok=False, raw=raw, error=str(repair_exc))
            return {}

    def _accept_items(batch: List[Dict[str, Any]], by_id: Dict[str, Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
        failed: List[Dict[str, Any]] = []
        for shot in batch:
            sid = _safe_str(shot.get("id"))
            if not sid or sid in results:
                continue
            item = by_id.get(sid)
            if not isinstance(item, dict):
                _llm_director_record_attempt(diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []}), stage=f"{stage}_id_match", ok=False, error="no item returned for this shot id")
                failed.append(shot)
                continue
            merged = _merge_llm_director_fields(shot, item, brief)
            validation = _director_validation_warnings(merged)
            if validation:
                try:
                    repair_raw = client.generate(
                        _llm_director_system_prompt(),
                        _llm_director_repair_prompt(shot, item, validation),
                        temperature=0.25,
                        max_tokens=1800,
                    )
                    repaired = _extract_json_payload_from_text(repair_raw)
                    if isinstance(repaired, dict) and "shots" in repaired and isinstance(repaired.get("shots"), list) and repaired.get("shots"):
                        repaired = repaired.get("shots")[0]
                    if not isinstance(repaired, dict):
                        raise ValueError("repair response was not a JSON object")
                    repaired["id"] = sid
                    merged2 = _merge_llm_director_fields(shot, repaired, brief)
                    validation2 = _director_validation_warnings(merged2)
                    if validation2:
                        msg = "; ".join(validation2[:3])
                        _llm_director_record_attempt(diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []}), stage=f"{stage}_field_repair", ok=False, raw=repair_raw, error=msg)
                        failed.append(shot)
                        continue
                    item = repaired
                    _llm_director_record_attempt(diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []}), stage=f"{stage}_field_repair", ok=True)
                except Exception as repair_exc:
                    _llm_director_record_attempt(diagnostics.setdefault(sid, {"id": sid, "retry_attempts": []}), stage=f"{stage}_field_repair", ok=False, error=str(repair_exc))
                    failed.append(shot)
                    continue
            _mark_success(sid, item, stage)
            results[sid] = item
        return failed

    try:
        _emit_progress("Loading director backend...")
        client.start()
        stages = [(4, "batch_4", 0.55, 3600), (2, "batch_2_retry", 0.45, 2600), (1, "single_shot_retry", 0.35, 1600)]
        pending = [s for s in shots if isinstance(s, dict)]
        for batch_size, stage, temperature, max_tokens in stages:
            if not pending:
                break
            next_pending: List[Dict[str, Any]] = []
            total_batches = max(1, math.ceil(len(pending) / float(batch_size)))
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                batch_idx = int(start // batch_size) + 1
                _emit_progress(f"Processing director {stage.replace('_', ' ')} {batch_idx} / {total_batches}...")
                by_id = _generate_and_parse(batch, stage, temperature=temperature, max_tokens=max_tokens)
                failed = _accept_items(batch, by_id, stage) if by_id else batch
                next_pending.extend([s for s in failed if _safe_str(s.get("id")) not in results])
            pending = next_pending
        for shot in pending:
            sid = _safe_str(shot.get("id"))
            if sid and sid not in results:
                _mark_fallback(sid, diagnostics.get(sid, {}).get("parse_error") or "all local LLM retry attempts failed")
                warnings.append(f"{sid}: local LLM failed after batch, smaller-batch, and single-shot retry; clean fallback used.")
        for sid, diag in diagnostics.items():
            if sid not in results and diag.get("llm_director_status") != "fallback":
                _mark_fallback(sid, diag.get("parse_error") or "local LLM returned no usable rewrite")
    finally:
        try:
            client.stop()
        except Exception:
            pass
    return results, diagnostics

def _write_director_plan_text(path: Path, plan: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("FrameVision Music Clip Creator - LTX Director Plan")
    lines.append(f"Created: {_safe_str(plan.get('created_at'))}")
    lines.append(f"Director backend: {_safe_str(plan.get('director_backend'))}")
    lines.append(f"Source LTX shot plan: {_safe_str(plan.get('source_ltx_shot_plan_path'))}")
    lines.append("")
    for shot in _as_list(plan.get("shots")):
        if not isinstance(shot, dict):
            continue
        sid = _safe_str(shot.get("id"))
        start = _safe_float(shot.get("song_start"), 0.0)
        end = _safe_float(shot.get("song_end"), 0.0)
        lines.append(f"[{sid}] {start:.2f}s - {end:.2f}s")
        lines.append(f"Source scenes: {', '.join(_as_list(shot.get('source_scene_ids')))}")
        lines.append(f"Needs lipsync: {bool(shot.get('needs_lipsync'))}")
        lyr = _clean_text(shot.get("lyrics"), 1000)
        if lyr:
            lines.append(f"Lyrics: {lyr}")
        lines.append(f"Image: {_clean_text(shot.get('director_image_prompt'))}")
        lines.append(f"Video: {_clean_text(shot.get('director_video_prompt'))}")
        lines.append(f"LTX: {_clean_text(shot.get('director_timestamped_video_prompt'))}")
        audio = _safe_str(shot.get("audio_clip_path"))
        if audio:
            lines.append(f"Audio chunk: {audio}")
        notes = _clean_text(shot.get("director_notes"), 600)
        if notes:
            lines.append(f"Notes: {notes}")
        shot_warnings = [str(x) for x in _as_list(shot.get("director_warnings")) if _safe_str(x)]
        if shot_warnings:
            lines.append("Warnings: " + "; ".join(shot_warnings[:8]))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def create_ltx_director_plan(payload: dict) -> dict:
    """Create musicclip_ltx_director_plan.json from musicclip_ltx_shot_plan.json.

    This is a planning/rewrite layer only. It does not run Planner, does not run LTX,
    does not generate images/videos and does not touch the normal render pipeline.
    """
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Director-plan payload was not a dictionary."}

        progress_callback = payload.get("progress_callback")

        def _emit_progress(message: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(str(message or ""))
                except Exception:
                    pass

        _emit_progress("Creating director plan...")
        ltx_path = _safe_str(payload.get("ltx_shot_plan_path") or payload.get("source_ltx_shot_plan_path"))
        ltx_obj = payload.get("ltx_shot_plan")
        if isinstance(ltx_obj, dict):
            ltx_plan = ltx_obj
        elif ltx_path:
            ltx_plan = _read_json_file(ltx_path)
        else:
            return {"ok": False, "message": "Create an LTX shot plan first."}

        shots_src = [s for s in _as_list(ltx_plan.get("shots")) if isinstance(s, dict)]
        if not shots_src:
            return {"ok": False, "message": "LTX shot plan did not contain any shots."}

        out_raw = _safe_str(payload.get("output_dir"))
        if out_raw:
            out_dir = Path(out_raw).resolve()
        elif ltx_path:
            out_dir = Path(ltx_path).resolve().parent
        else:
            out_dir = _default_ltx_videoclip_output_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        brief = _normalize_creative_brief(payload.get("creative_brief") or ltx_plan.get("creative_brief"))
        backend = _normalize_director_backend(payload.get("director_backend") or payload.get("director_brain") or "Template cleanup")
        backend_label = _director_backend_label(payload.get("director_backend") or payload.get("director_brain") or "Template cleanup")
        bridge_generation_settings = _normalize_bridge_generation_settings(payload, ltx_plan)
        duration_profile = _duration_profile_from_bridge_settings(bridge_generation_settings)
        microclip_profile = _microclip_profile_from_bridge_settings(bridge_generation_settings, duration_profile)
        effects_profile = _effects_profile_from_bridge_settings(bridge_generation_settings)
        character_reference = _character_reference_from_sources(payload, ltx_plan)
        if ltx_plan.get("character_bible") or payload.get("character_bible"):
            character_bible, group_bibles = _normalize_character_bibles(
                ltx_plan.get("character_bible") or payload.get("character_bible"),
                ltx_plan.get("group_bibles") or payload.get("group_bibles"),
                brief,
            )
            shot_assignment_guidance = _safe_str(ltx_plan.get("shot_assignment_guidance") or payload.get("shot_assignment_guidance"))
            character_bible_backend = _safe_str(ltx_plan.get("character_bible_backend") or "inherited_or_neutral_fallback")
            character_bible_warnings = _as_list(ltx_plan.get("character_bible_warnings"))
        else:
            character_pack = _create_llm_character_bible_if_selected(
                payload={**payload, "director_backend": backend_label},
                brief=brief,
                scenes=shots_src,
                source_plan=ltx_plan,
                progress_callback=_emit_progress,
            )
            character_bible = character_pack.get("character_bible") or []
            group_bibles = character_pack.get("group_bibles") or []
            shot_assignment_guidance = _safe_str(character_pack.get("shot_assignment_guidance"))
            character_bible_backend = _safe_str(character_pack.get("character_bible_backend"))
            character_bible_warnings = _as_list(character_pack.get("character_bible_warnings"))
        character_bible, group_bibles = _apply_character_reference_to_bibles(character_bible, group_bibles, character_reference)
        warnings: List[str] = []
        warnings.extend([_clean_text(w, 500) for w in character_bible_warnings if _safe_str(w)])
        warnings.extend(_plan_character_reference_warnings(character_reference, passed_to_model=False))

        shots_src = _director_distribute_location_pool(shots_src, brief)
        director_shots = [_director_template_shot(shot, brief) for shot in shots_src]
        actual_backend = "template_cleanup"
        llm_diagnostics: Dict[str, Dict[str, Any]] = {}

        if backend == _DIRECTOR_BACKEND_LOCAL_LLM:
            try:
                _emit_progress("Loading director backend...")
                cfg = _read_planner_llama_settings_for_bridge(payload.get("root_dir") or _project_root(), payload.get("llama_settings") if isinstance(payload.get("llama_settings"), dict) else None)
                llm_items, llm_diagnostics = _run_llm_director_batches(director_shots, brief, cfg, warnings, progress_callback=_emit_progress)
                if llm_items or llm_diagnostics:
                    rewritten = []
                    for shot in director_shots:
                        sid = _safe_str(shot.get("id"))
                        diag = llm_diagnostics.get(sid, {}) if isinstance(llm_diagnostics, dict) else {}
                        if sid in llm_items:
                            merged = _merge_llm_director_fields(shot, llm_items[sid])
                            merged["llm_retry_attempts"] = _as_list(diag.get("retry_attempts") or merged.get("llm_retry_attempts"))
                            rewritten.append(merged)
                        else:
                            fallback = dict(shot)
                            fallback["llm_director_status"] = "fallback"
                            fallback["raw_failed_llm_output"] = _safe_str(diag.get("raw_failed_llm_output"))
                            fallback["parse_error"] = _safe_str(diag.get("parse_error"))
                            fallback["llm_retry_attempts"] = _as_list(diag.get("retry_attempts"))
                            fallback["final_reason_for_fallback"] = _safe_str(diag.get("final_reason_for_fallback") or "local LLM returned no valid rewrite after retries")
                            fallback["fallback_reason"] = fallback["final_reason_for_fallback"]
                            rewritten.append(fallback)
                    director_shots = rewritten
                    actual_backend = "local_llm_planner_style"
                else:
                    warnings.append("Local LLM produced no usable director rewrites; template cleanup kept.")
            except Exception as exc:
                warnings.append(f"Local LLM director rewrite unavailable; template cleanup kept ({exc}).")
                actual_backend = "template_cleanup_fallback"

        # Hard validation: same count, same IDs/order, single-line timestamped prompts.
        fixed: List[Dict[str, Any]] = []
        for src, shot in zip(shots_src, director_shots):
            out = dict(shot)
            out["id"] = _safe_str(src.get("id"))
            out["index"] = _safe_int(src.get("index"), _safe_int(out.get("index"), 0))
            for key in (
                "source_scene_ids", "source_scene_indexes", "song_start", "song_end", "duration",
                "target_fps", "target_frames", "section_summary", "energy_summary",
                "vocal_presence_summary", "needs_lipsync", "scene_role_summary", "lyrics",
                "clip_relative_lyrics", "audio_clip_path", "duration_profile_used", "grouping_reason", "duration_warning",
                "is_microclip", "microclip_reason", "microclip_style", "micro_moment_count", "microclip_profile_used",
                "dominant_location", "is_montage_style", "montage_reason", "montage_position_ok",
                "effects_used", "effect_policy", "collage_effect_reason",
            ):
                if key in src:
                    out[key] = src.get(key)
            out["scene_role_summary"] = _director_safe_role_for_brief(out.get("scene_role_summary"), brief)
            out["template_image_prompt"] = _sanitize_prompt_no_visible_text(_director_clean_prompt_text(src.get("image_prompt"), 1800), src.get("lyrics"), image_prompt=True)
            out["template_video_prompt"] = _director_clean_prompt_text(src.get("video_prompt"), 1800)
            out["template_timestamped_video_prompt"] = _normalize_director_timestamp(src.get("timestamped_video_prompt"), duration=_safe_float(src.get("duration"), 0.0), needs_lipsync=_safe_bool(src.get("needs_lipsync"), False))
            out["director_timestamped_video_prompt"] = _normalize_director_timestamp(
                out.get("director_timestamped_video_prompt"),
                duration=_safe_float(src.get("duration"), 0.0),
                needs_lipsync=_safe_bool(src.get("needs_lipsync"), False),
                fallback=_director_template_timestamp(src, brief),
            )
            out["director_image_prompt"] = _sanitize_prompt_no_visible_text(out.get("director_image_prompt") or _director_template_image_prompt(src, brief), src.get("lyrics"), image_prompt=True)
            out["director_video_prompt"] = _sanitize_prompt_no_visible_text(out.get("director_video_prompt") or _director_template_video_prompt(src, brief), src.get("lyrics"), image_prompt=False)
            out["director_negative_prompt"] = _negative_prompt_with_no_visible_text(out.get("director_negative_prompt") or _director_template_negative_prompt(src), max_len=1200)
            if not _safe_str(out.get("director_notes")):
                out["director_notes"] = "Template cleanup fallback."
            out = _apply_character_identity_to_item(out, brief, character_bible, group_bibles, include_director_fields=True)
            out = _apply_character_reference_to_item(out, brief, character_reference, image_prompt_keys=["template_image_prompt", "director_image_prompt"], passed_to_model=False)
            before_lipsync = _safe_bool(out.get("needs_lipsync"), False)
            out = _protect_lipsync_confidence(out, song_duration=max([_safe_float(x.get("song_end"), 0.0) for x in shots_src] + [0.0]))
            if _safe_bool(out.get("is_microclip"), False):
                out["needs_lipsync"] = False
                out["vocal_presence_summary"] = "instrumental"
                out["clip_relative_lyrics"] = []
                out["scene_role_summary"] = _safe_str(out.get("microclip_style")) or "beat_hit_cutaway"
                for _key in _nonvocal_prompt_keys():
                    if _key in out:
                        out[_key] = _strip_non_lipsync_vocal_language(out.get(_key))
            extra_validation = []
            if before_lipsync and not _safe_bool(out.get("needs_lipsync"), False):
                extra_validation.append("director plan downgraded lipsync: no active lyric window survived validation")
                warnings.append(f"{_safe_str(out.get('id'))}: director lipsync downgraded; no active lyric window.")
            # Per-shot validation metadata is written into every final director shot.
            # This keeps bad LLM output local to the shot and makes fallback/repair visible.
            out = _director_mark_validation(out, extra_validation)
            out = _sanitize_ltx_timestamp_fields_in_item(out, keys=["template_timestamped_video_prompt", "director_timestamped_video_prompt"])
            fixed.append(out)
        director_shots = fixed

        montage_policy = _montage_policy_from_plan(ltx_plan, payload, {"effects_profile": effects_profile})
        director_shots, montage_policy, montage_warnings = _apply_montage_policy_to_items(
            director_shots,
            brief=brief,
            policy=montage_policy,
            prompt_keys=[
                "template_image_prompt", "template_video_prompt", "template_timestamped_video_prompt",
                "director_image_prompt", "director_video_prompt", "director_timestamped_video_prompt",
            ],
            director_keys=True,
        )
        warnings.extend(montage_warnings)

        director_shots, intro_outro_policy, intro_outro_warnings = _apply_intro_outro_safety_to_items(
            director_shots,
            brief=brief,
            prompt_keys=[
                "template_image_prompt", "template_video_prompt", "template_timestamped_video_prompt",
                "director_image_prompt", "director_video_prompt", "director_timestamped_video_prompt",
            ],
        )
        warnings.extend(intro_outro_warnings)
        director_shots = _apply_effect_policy_metadata(director_shots, effects_profile=effects_profile)
        director_shots = [_sanitize_ltx_timestamp_fields_in_item(shot, keys=["template_timestamped_video_prompt", "director_timestamped_video_prompt"]) for shot in director_shots]

        # Final compiler gate: model-facing prompts are rebuilt from scene_concept
        # just before saving, so LLM/template/meta leakage cannot reach image generation.
        director_shots = _director_compile_all_shot_prompts(director_shots, brief)
        local_llm_requested = bool(backend_label.startswith("Local LLM") or backend == _DIRECTOR_BACKEND_LOCAL_LLM)
        llm_summary = _director_llm_quality_summary(director_shots, local_requested=local_llm_requested)
        location_debug = _director_location_debug_summary(director_shots, brief)
        if local_llm_requested and llm_summary.get("director_quality_status") in {"warning_too_many_fallbacks", "fail_no_shots"}:
            warnings.append(
                f"Director quality warning: {llm_summary.get('llm_fallback_count')} of {len(director_shots)} shots used fallback after local LLM retries."
            )
        if not (location_debug.get("one_location_per_shot_validation") or {}).get("ok", True):
            warnings.append("Location validation warning: at least one final prompt mentions multiple locations.")

        plan_path = out_dir / "musicclip_ltx_director_plan.json"
        txt_path = out_dir / "musicclip_ltx_director_plan.txt"
        director_plan = {
            "version": _VERSION,
            "source": _SOURCE,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_ltx_shot_plan_path": ltx_path,
            "audio_path": _safe_str(ltx_plan.get("audio_path")),
            "srt_path": _safe_str(ltx_plan.get("srt_path")),
            "creative_brief": brief,
            "character_bible": character_bible,
            "group_bibles": group_bibles,
            "shot_assignment_guidance": shot_assignment_guidance,
            "character_bible_backend": character_bible_backend,
            "character_bible_warnings": character_bible_warnings,
            "character_reference": {k: v for k, v in character_reference.items() if k != "warnings"},
            "character_reference_warnings": _plan_character_reference_warnings(character_reference, passed_to_model=False),
            "fps": _safe_float(ltx_plan.get("fps"), 24.0),
            "resolution": _safe_str(bridge_generation_settings.get("ltx_resolution") or ltx_plan.get("ltx_resolution") or ltx_plan.get("resolution"), "1280x704"),
            "ltx_resolution": _safe_str(bridge_generation_settings.get("ltx_resolution") or ltx_plan.get("ltx_resolution") or ltx_plan.get("resolution"), "1280x704"),
            "ltx_aspect_mode": _safe_str(bridge_generation_settings.get("ltx_aspect_mode") or ltx_plan.get("ltx_aspect_mode")),
            "bridge_generation_settings": bridge_generation_settings,
            "duration_profile": duration_profile,
            "microclip_profile": microclip_profile,
            "effects_profile": effects_profile,
            "montage_policy": montage_policy,
            "intro_outro_policy": intro_outro_policy,
            "location_pool": _location_candidates_from_brief(brief),
            "location_debug": location_debug,
            "llm_summary": llm_summary,
            "llm_success_count": llm_summary.get("llm_success_count"),
            "llm_fallback_count": llm_summary.get("llm_fallback_count"),
            "llm_success_rate": llm_summary.get("llm_success_rate"),
            "director_quality_status": llm_summary.get("director_quality_status"),
            "ltx_shot_count": len(director_shots),
            "prompt_backend": _safe_str(ltx_plan.get("prompt_backend")) or "ltx_shot_plan_template",
            "director_backend": actual_backend if backend_label.startswith("Local LLM") else "template_cleanup",
            "requested_director_backend": backend_label,
            "final_master_audio": _safe_str(ltx_plan.get("final_master_audio") or ltx_plan.get("audio_path")),
            "audio_chunks_dir": _safe_str(ltx_plan.get("audio_chunks_dir")),
            "warnings": warnings[:80],
            "shots": director_shots,
        }

        _emit_progress("Writing director plan files...")
        try:
            debug_txt_path, debug_json_path = _write_director_prompt_debug_reports(out_dir, director_plan)
            director_plan["prompt_debug_report_text_path"] = debug_txt_path
            director_plan["prompt_debug_report_json_path"] = debug_json_path
        except Exception as exc:
            warnings.append(f"Prompt debug report could not be written: {exc}")
            director_plan["warnings"] = warnings[:80]
        _write_json_file(plan_path, director_plan)
        try:
            _write_director_plan_text(txt_path, director_plan)
        except Exception as exc:
            warnings.append(f"Text director plan could not be written: {exc}")
            director_plan["warnings"] = warnings[:80]
            _write_json_file(plan_path, director_plan)

        _emit_progress(f"Director plan saved: {plan_path}")
        msg = "LTX director plan created."
        if warnings:
            msg += f" Warnings: {len(warnings)} fallback/warning item(s)."
        return {
            "ok": True,
            "ltx_director_plan_path": str(plan_path),
            "ltx_director_plan_text_path": str(txt_path) if txt_path.is_file() else "",
            "prompt_debug_report_text_path": director_plan.get("prompt_debug_report_text_path", ""),
            "prompt_debug_report_json_path": director_plan.get("prompt_debug_report_json_path", ""),
            "output_dir": str(out_dir),
            "ltx_shot_count": len(director_shots),
            "director_backend": director_plan.get("director_backend"),
            "requested_director_backend": backend_label,
            "message": msg,
            "warnings": warnings[:30],
        }
    except Exception as exc:
        return {"ok": False, "message": f"LTX director plan creation failed: {exc}"}





def _parse_resolution_value(value: Any, default: str = "1280x720") -> tuple[int, int, str]:
    raw = _safe_str(value, default) or default
    m = re.search(r"(\d{3,5})\s*[xX]\s*(\d{3,5})", raw)
    if not m:
        raw = default
        m = re.search(r"(\d{3,5})\s*[xX]\s*(\d{3,5})", raw)
    width = _safe_int(m.group(1), 1280) if m else 1280
    height = _safe_int(m.group(2), 720) if m else 720
    width = max(256, min(width, 4096))
    height = max(256, min(height, 4096))
    return width, height, f"{width}x{height}"


def _normalize_ltx23_vramlab_resolution(value: Any, default: str = "1280x704") -> tuple[int, int, str]:
    """Map normal video/image sizes to the tuned LTX 2.3 VRAMLab buckets."""
    width, height, _ = _parse_resolution_value(value, default)
    portrait = bool(height > width)
    long_side = max(width, height)
    short_side = min(width, height)
    if portrait:
        if long_side >= 1500 or short_side >= 1000:
            return 1088, 1920, "1088x1920"
        if long_side >= 1000 or short_side >= 700:
            return 704, 1280, "704x1280"
        return 512, 832, "512x832"
    if long_side >= 1500 or short_side >= 1000:
        return 1920, 1088, "1920x1088"
    if long_side >= 1000 or short_side >= 700:
        return 1280, 704, "1280x704"
    return 832, 512, "832x512"


def _make_ltx_test_dir(plan_path: Path, shot_id: str) -> Path:
    base = (plan_path.parent / "ltx_tests").resolve()
    base.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(shot_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = base / f"{stem}_{timestamp}"
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate.resolve()
    for idx in range(2, 1000):
        candidate = base / f"{stem}_{timestamp}_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate.resolve()
    raise RuntimeError("Could not create a unique LTX test output folder.")


def _pick_flux_klein_models_highest(root: Path, force_klein_b: int = 0) -> Dict[str, str]:
    """Pick Flux Klein GGUF files without importing Planner or touching Planner state."""
    root = Path(root).resolve()
    sdcli = (root / "presets" / "bin" / ("sd-cli.exe" if os.name == "nt" else "sd-cli")).resolve()
    models_root = (root / "models").resolve()
    preferred_dirs: List[Path] = []
    for dname in ("klein9b_gguf", "klein4b_gguf"):
        d = (models_root / dname).resolve()
        if d.exists():
            preferred_dirs.append(d)
    try:
        if models_root.exists():
            for d in models_root.iterdir():
                if not d.is_dir():
                    continue
                dn = d.name.lower()
                if ("klein" in dn) and ("gguf" in dn) and (d.resolve() not in preferred_dirs):
                    preferred_dirs.append(d.resolve())
    except Exception:
        pass

    def _parse_qnum(name: str) -> int:
        m = re.search(r"(?i)\bQ(\d+)", name)
        return int(m.group(1)) if m else -1

    def _parse_klein_b(name: str) -> int:
        m = re.search(r"(?i)klein[-_ ]?(\d+)b", name)
        return int(m.group(1)) if m else -1

    def _parse_qwen_b(name: str) -> int:
        m = re.search(r"(?i)qwen\s*3[-_ ]?(\d+)b", name)
        return int(m.group(1)) if m else -1

    def _safe_size(pp: Path) -> int:
        try:
            return int(pp.stat().st_size)
        except Exception:
            return 0

    diffusion = ""
    llm = ""
    vae = ""
    chosen_model_dir = ""
    diffusion_cands: List[tuple] = []
    for d in preferred_dirs:
        try:
            for pp in d.rglob("*.gguf"):
                if not pp.is_file():
                    continue
                n = pp.name.lower()
                if ("flux" in n) and ("klein" in n) and ("qwen" not in n) and ("llm" not in n):
                    kb = _parse_klein_b(pp.name)
                    if int(force_klein_b or 0) > 0 and kb != int(force_klein_b):
                        continue
                    diffusion_cands.append((kb, _parse_qnum(pp.name), _safe_size(pp), str(pp), d))
        except Exception:
            continue
    preferred_qwen_b = 8 if int(force_klein_b or 0) == 9 else 4
    if diffusion_cands:
        diffusion_cands.sort(key=lambda t: (t[0], t[1], t[2], t[3]), reverse=True)
        kb, _qn, _sz, best_path, best_dir = diffusion_cands[0]
        diffusion = best_path
        chosen_model_dir = str(best_dir)
        preferred_qwen_b = 8 if kb >= 9 else 4

    llm_cands: List[tuple] = []
    for d in preferred_dirs:
        try:
            for pp in d.rglob("*.gguf"):
                if not pp.is_file():
                    continue
                n = pp.name.lower()
                if ("qwen" in n) and ("3" in n):
                    qb = _parse_qwen_b(pp.name)
                    pref = 1 if qb == preferred_qwen_b else 0
                    llm_cands.append((pref, qb, _parse_qnum(pp.name), _safe_size(pp), str(pp)))
        except Exception:
            continue
    if llm_cands:
        llm_cands.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]), reverse=True)
        llm = llm_cands[0][4]

    vae_dirs: List[Path] = []
    if diffusion:
        try:
            vae_dirs.append(Path(diffusion).resolve().parent)
        except Exception:
            pass
    for d in preferred_dirs:
        if d not in vae_dirs:
            vae_dirs.append(d)
    vae_cands: List[tuple] = []
    for d in vae_dirs:
        try:
            for pp in d.rglob("*.safetensors"):
                if not pp.is_file():
                    continue
                n = pp.name.lower()
                is_flux_vae = 1 if (("flux" in n) and (("ae" in n) or ("vae" in n))) else 0
                vae_cands.append((is_flux_vae, _safe_size(pp), str(pp)))
        except Exception:
            continue
    if vae_cands:
        vae_cands.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        vae = vae_cands[0][2]
    if not chosen_model_dir and preferred_dirs:
        chosen_model_dir = str(preferred_dirs[0])
    return {
        "sd_cli": str(sdcli),
        "model_dir": str(chosen_model_dir),
        "diffusion": diffusion,
        "llm": llm,
        "vae": vae,
        "searched_dirs": ";".join([str(d) for d in preferred_dirs]),
    }


def _run_flux_klein_start_image(root: Path, *, prompt: str, negative: str, output_path: Path, width: int, height: int, seed: int, log_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    def _emit(text: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(str(text or ""))
            except Exception:
                pass

    files = _pick_flux_klein_models_highest(root, force_klein_b=9)
    sdcli_path = _safe_str(files.get("sd_cli"))
    diffusion = _safe_str(files.get("diffusion"))
    llm = _safe_str(files.get("llm"))
    vae = _safe_str(files.get("vae"))
    searched = _safe_str(files.get("searched_dirs"))
    if not sdcli_path or not os.path.isfile(sdcli_path):
        raise RuntimeError("Flux Klein 9B model files were not found. sd-cli.exe was not found at presets/bin/sd-cli.exe.")
    if not diffusion or not os.path.isfile(diffusion):
        raise RuntimeError(f"Flux Klein 9B model files were not found. No Klein 9B diffusion GGUF was found. Searched: {searched or '[no Klein GGUF folders]'}")
    if not llm or not os.path.isfile(llm):
        raise RuntimeError(f"Flux Klein 9B model files were not found. No Qwen3 LLM GGUF was found. Searched: {searched or '[no Klein GGUF folders]'}")
    if not vae or not os.path.isfile(vae):
        raise RuntimeError(f"Flux Klein 9B model files were not found. No VAE safetensors was found. Searched: {searched or '[no Klein GGUF folders]'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.exists():
            output_path.unlink()
    except Exception:
        pass
    cmd = [sdcli_path]
    cmd += ["--diffusion-model", diffusion]
    cmd += ["--vae", vae]
    cmd += ["--llm", llm]
    cmd += ["-p", prompt]
    if negative:
        cmd += ["-n", negative]
    cmd += ["-W", str(int(width)), "-H", str(int(height))]
    cmd += ["--steps", "4"]
    cmd += ["--cfg-scale", "1.0"]
    cmd += ["--sampling-method", "euler"]
    cmd += ["-s", str(int(seed))]
    cmd += ["--diffusion-fa"]
    cmd += ["-o", str(output_path), "-v"]

    _emit("Creating start image with Flux Klein 9B...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
        lf.write("[musicclip bridge] LTX single-shot start image generation\n")
        lf.write("Image model: Flux Klein 9B\n")
        lf.write(f"Output: {output_path}\n")
        lf.write("Command:\n")
        lf.write(" ".join([str(x) for x in cmd]) + "\n\n")
        lf.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if proc.stdout:
            for raw in proc.stdout:
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip() if isinstance(raw, bytes) else str(raw).rstrip()
                except Exception:
                    line = str(raw).rstrip()
                if line:
                    try:
                        lf.write(line + "\n")
                        lf.flush()
                    except Exception:
                        pass
                    _emit(line[:240])
        rc = int(proc.wait() or 0)
        lf.write(f"\nExit code: {rc}\n")
    if rc != 0:
        raise RuntimeError(f"Flux Klein 9B sd-cli failed (exit code {rc}). See log: {log_path}")
    if not output_path.is_file() or output_path.stat().st_size < 1024:
        raise RuntimeError(f"Flux Klein 9B finished but no valid output image was created: {output_path}")
    return {"command": cmd, "model_files": files, "rc": rc, "out_file": str(output_path)}



_HIDREAM_MODEL_INFO = {
    "dev_2604_bf16": {
        "label": "Dev 2604 BF16",
        "folder": "HiDream-O1-Image-Dev-2604-BF16",
        "steps": 28,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
        "timesteps": "dev",
    },
    "dev_2604_fp8": {
        "label": "Dev 2604 FP8",
        "folder": "HiDream-O1-Image-Dev-2604-FP8",
        "steps": 28,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
        "timesteps": "dev",
    },
    "dev_fp8": {
        "label": "Dev FP8",
        "folder": "HiDream-O1-Image-Dev-FP8",
        "steps": 28,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
        "timesteps": "dev",
    },
    "dev": {
        "label": "Dev BF16",
        "folder": "HiDream-O1-Image-Dev-BF16",
        "steps": 28,
        "guidance_scale": 0.0,
        "shift": 1.0,
        "scheduler_name": "flash",
        "timesteps": "dev",
    },
    "base_fp8": {
        "label": "Base FP8",
        "folder": "HiDream-O1-Image-FP8",
        "steps": 50,
        "guidance_scale": 5.0,
        "shift": 3.0,
        "scheduler_name": "flash",
        "timesteps": "none",
    },
    "base": {
        "label": "Base BF16",
        "folder": "HiDream-O1-Image-BF16",
        "steps": 50,
        "guidance_scale": 5.0,
        "shift": 3.0,
        "scheduler_name": "flash",
        "timesteps": "none",
    },
}


def _bridge_txt2img_settings_path(root: Path) -> Path:
    return (Path(root).resolve() / "presets" / "setsave" / "txt2img.json").resolve()


def _bridge_read_txt2img_base_job(root: Path) -> Dict[str, Any]:
    path = _bridge_txt2img_settings_path(root)
    try:
        if path.is_file():
            data = _read_json_file(str(path))
            if isinstance(data, dict):
                return dict(data)
    except Exception:
        pass
    return {}


def _bridge_pick_zimage_gguf(root: Path, quality: str = "medium") -> str:
    """Planner-style Z-Image Turbo GGUF picker.

    Low picks the lowest quant, medium picks Q5-or-nearest-above, high picks the
    highest quant. Unknown quant names are used only as a last resort.
    """
    try:
        mode = _safe_str(quality, "medium").lower()
        if "high" in mode:
            mode = "high"
        elif "low" in mode:
            mode = "low"
        else:
            mode = "medium"
        gguf_dir = (Path(root).resolve() / "models" / "Z-Image-Turbo GGUF").resolve()
        if not gguf_dir.exists():
            return ""
        cands: List[tuple] = []
        unknown: List[tuple] = []
        for pp in gguf_dir.glob("**/*.gguf"):
            try:
                if not pp.is_file():
                    continue
                name = pp.name.lower()
                if not ("z_image_turbo" in name or "z-image-turbo" in name or ("zimage" in name and "turbo" in name)):
                    continue
                m = re.search(r"\bQ(\d+)\b", pp.name, flags=re.IGNORECASE)
                qn = int(m.group(1)) if m else -1
                try:
                    sz = pp.stat().st_size
                except Exception:
                    sz = 0
                if qn < 0:
                    unknown.append((sz, str(pp.resolve())))
                else:
                    cands.append((qn, sz, str(pp.resolve())))
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
        elif mode == "low":
            pick_q = min(qs)
        else:
            above = [q for q in qs if q >= 5]
            pick_q = min(above) if above else (max([q for q in qs if q < 5]) if [q for q in qs if q < 5] else max(qs))
        best = [t for t in cands if t[0] == pick_q]
        best.sort(key=lambda t: (-t[1], t[2]))
        return best[0][2]
    except Exception:
        return ""


def _bridge_zimage_fp16_available(root: Path) -> bool:
    """Best-effort check for the normal/FP16 Z-Image Turbo pack.

    The real generator still does final validation. This check only helps choose
    between the normal zimage backend and the GGUF backend without importing Planner.
    """
    try:
        models = (Path(root).resolve() / "models").resolve()
        candidates = [models / "Z-Image-Turbo", models / "z-image-turbo", models / "Z-Image Turbo"]
        for d in candidates:
            if d.exists() and d.is_dir():
                return True
        if models.exists():
            for d in models.iterdir():
                if not d.is_dir():
                    continue
                name = d.name.lower()
                if "z-image" in name and "turbo" in name and "gguf" not in name:
                    return True
    except Exception:
        pass
    return False


def _bridge_import_txt2img(root: Path):
    try:
        return __import__("helpers.txt2img", fromlist=["*"])
    except Exception:
        helpers_dir = str((Path(root).resolve() / "helpers").resolve())
        if helpers_dir not in sys.path:
            sys.path.insert(0, helpers_dir)
        try:
            return __import__("txt2img")
        except Exception as exc:
            raise RuntimeError(f"helpers/txt2img.py could not be imported for Z-Image generation: {exc}")


def _copy_generated_image_to_start(src: str, dst: Path) -> str:
    src_path = Path(_safe_str(src)).expanduser().resolve()
    if not src_path.is_file() or src_path.stat().st_size < 1024:
        raise RuntimeError(f"Generator returned no valid output image: {src or '[empty]'}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src_path != dst.resolve():
        shutil.copy2(str(src_path), str(dst))
    if not dst.is_file() or dst.stat().st_size < 1024:
        raise RuntimeError(f"Could not copy generated image to: {dst}")
    return str(dst.resolve())


def _run_txt2img_job_with_retry(t2i_mod: Any, job: Dict[str, Any], images_dir: str, log_func: Callable[[str], None]) -> Dict[str, Any]:
    if not hasattr(t2i_mod, "generate_one_from_job"):
        raise RuntimeError("txt2img generator entrypoint not found (generate_one_from_job)")
    work = dict(job)
    # Older backends may not accept deterministic filename fields. We copy/rename
    # the returned image to LTXxx_start.png after generation instead.
    for key in ("filename_template", "format", "cancel_event"):
        work.pop(key, None)
    stripped: List[str] = []
    while True:
        try:
            return t2i_mod.generate_one_from_job(work, images_dir)
        except TypeError as exc:
            msg = str(exc)
            m = re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]", msg)
            if m:
                bad = m.group(1)
                if bad in work and len(stripped) < 64:
                    work.pop(bad, None)
                    stripped.append(bad)
                    try:
                        log_func(f"[IMG] backend stripped unsupported key: {bad}")
                    except Exception:
                        pass
                    continue
            raise


def _latest_image_in_folder(folder: Path, since_ts: float) -> str:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    best = (0.0, "")
    try:
        for pp in folder.glob("**/*"):
            if not pp.is_file() or pp.suffix.lower() not in exts:
                continue
            try:
                mt = pp.stat().st_mtime
                sz = pp.stat().st_size
            except Exception:
                continue
            if sz < 1024 or mt < since_ts - 2.0:
                continue
            if mt > best[0]:
                best = (mt, str(pp.resolve()))
    except Exception:
        pass
    return best[1]


def _run_zimage_start_image(root: Path, *, prompt: str, negative: str, output_path: Path, width: int, height: int, seed: int, log_path: Path, progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    def _emit(text: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(str(text or ""))
            except Exception:
                pass

    root = Path(root).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.exists():
            output_path.unlink()
    except Exception:
        pass

    base_job = _bridge_read_txt2img_base_job(root)
    use_fp16 = _bridge_zimage_fp16_available(root)
    gguf_path = ""
    engine = "zimage" if use_fp16 else "zimage_gguf"
    if engine == "zimage_gguf":
        gguf_path = _bridge_pick_zimage_gguf(root, "medium")
        if not gguf_path:
            searched = str((root / "models" / "Z-Image-Turbo GGUF").resolve())
            raise RuntimeError(f"Z-Image Turbo model files were not found. No FP16 Z-Image Turbo folder was found and no diffusion GGUF was found in: {searched}")

    t2i_job = dict(base_job)
    t2i_job.update({
        "engine": engine,
        "prompt": prompt,
        "negative_prompt": negative,
        "negative": negative,
        "neg_prompt": negative,
        "seed": int(seed),
        "batch": 1,
        "output": str(output_path.parent),
        "width": int(width),
        "height": int(height),
        "steps": 10,
        "cfg_scale": 0.0,
        "cfg": 0.0,
    })
    if gguf_path:
        # Same convention used by Planner/txt2img: zimage_gguf repurposes lora_path
        # as the selected diffusion GGUF override.
        t2i_job["lora_path"] = gguf_path

    _emit("Creating start image with Z-Image Turbo...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_summary = {
        "backend": "helpers.txt2img.generate_one_from_job",
        "engine": engine,
        "width": int(width),
        "height": int(height),
        "steps": 10,
        "cfg": 0.0,
        "seed": int(seed),
        "gguf_path": gguf_path,
        "base_settings": str(_bridge_txt2img_settings_path(root)),
    }
    start_ts = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
        def _log(line: str) -> None:
            try:
                lf.write(str(line) + "\n")
                lf.flush()
            except Exception:
                pass
            _emit(str(line)[:240])

        lf.write("[musicclip bridge] LTX single-shot start image generation\n")
        lf.write("Image model: Z-Image Turbo\n")
        lf.write(f"Output: {output_path}\n")
        lf.write("Command summary:\n")
        lf.write(json.dumps(command_summary, indent=2, ensure_ascii=False) + "\n\n")
        lf.flush()
        mod = _bridge_import_txt2img(root)
        res = _run_txt2img_job_with_retry(mod, t2i_job, str(output_path.parent), _log)
        lf.write("\nResult:\n")
        try:
            lf.write(json.dumps(res, indent=2, ensure_ascii=False, default=str) + "\n")
        except Exception:
            lf.write(str(res) + "\n")

    out_file = ""
    try:
        if isinstance(res, dict) and res.get("files"):
            out_file = str((res.get("files") or [""])[0] or "")
        if isinstance(res, dict) and not out_file:
            out_file = str(res.get("out_file") or res.get("file") or "")
    except Exception:
        out_file = ""
    if not out_file:
        out_file = _latest_image_in_folder(output_path.parent, start_ts)
    final_path = _copy_generated_image_to_start(out_file, output_path)
    return {"command": command_summary, "model_files": {"zimage_gguf": gguf_path, "engine": engine}, "out_file": final_path}


def _hidream_root(root: Path) -> Path:
    return (Path(root).resolve() / "models" / "hidream_bf16").resolve()


def _hidream_model_dir(root: Path, model_key: str) -> Path:
    info = _HIDREAM_MODEL_INFO.get(_safe_str(model_key), _HIDREAM_MODEL_INFO["base"])
    return (_hidream_root(root) / _safe_str(info.get("folder"))).resolve()


def _hidream_model_installed(root: Path, model_key: str) -> bool:
    try:
        d = _hidream_model_dir(root, model_key)
        return bool(d.exists() and (d / "config.json").exists())
    except Exception:
        return False


def _hidream_cli_supports_model_key(root: Path, model_key: str) -> bool:
    """Avoid selecting a model key that the installed helpers/hidream_cli.py cannot parse."""
    key = _safe_str(model_key)
    if not key:
        return False
    cli = _hidream_cli_path(root)
    if not cli:
        # If the CLI cannot be inspected yet, keep legacy behavior and let the later path check fail clearly.
        return True
    try:
        raw = Path(cli).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return True
    # Modern helpers expose MODEL_MAP / MODEL_DEFAULTS keys literally. This also
    # keeps older helpers safe if they do not yet include dev_2604_* choices.
    return bool(re.search(rf"[\"']{re.escape(key)}[\"']\s*:", raw) or re.search(rf"\b{re.escape(key)}\b", raw))


def _pick_hidream_model_key(root: Path) -> str:
    """Pick the HiDream model for Character Sheet image creation/edit helpers.

    Character-sheet workflows should prefer Dev models only.
    Preference order requested by the user:
    1) Dev FP8
    2) Dev BF16
    3) Dev 2604 BF16
    4) Dev 2604 FP8

    Base models are intentionally excluded from this picker.
    """
    for key in ("dev_fp8", "dev", "dev_2604_bf16", "dev_2604_fp8"):
        if _hidream_model_installed(root, key) and _hidream_cli_supports_model_key(root, key):
            return key
    return ""


def _hidream_cli_path(root: Path) -> str:
    root = Path(root).resolve()
    candidates = [
        (root / "helpers" / "hidream_cli.py").resolve(),
        (root / "hidream_cli.py").resolve(),
        (root / "models" / "hidream_bf16" / "run_hidream.py").resolve(),
        (root / "models" / "hidream_bf16" / "run_hidream_bf16.py").resolve(),
    ]
    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand)
        except Exception:
            continue
    return ""


def _hidream_batch_cli_path(root: Path) -> str:
    root = Path(root).resolve()
    candidates = [
        (root / "helpers" / "hidream_batch_cli.py").resolve(),
        (root / "hidream_batch_cli.py").resolve(),
    ]
    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand)
        except Exception:
            continue
    return ""


def _hidream_python_path(root: Path) -> str:
    root = Path(root).resolve()
    candidates = [
        (root / "environments" / ".hidream_dev" / "python.exe").resolve(),
        (root / "environments" / ".hidream_bf16" / "python.exe").resolve(),
        (root / ".hidream_dev" / "python.exe").resolve(),
        (root / ".hidream_bf16" / "python.exe").resolve(),
    ]
    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand)
        except Exception:
            continue
    return sys.executable or "python"


def _hidream_defaults_for_key(model_key: str) -> Dict[str, Any]:
    info = dict(_HIDREAM_MODEL_INFO.get(_safe_str(model_key), _HIDREAM_MODEL_INFO["dev"]))
    return {
        "steps": int(info.get("steps") or 50),
        "guidance_scale": float(info.get("guidance_scale") or 0.0),
        "shift": float(info.get("shift") or 1.0),
        "scheduler_name": _safe_str(info.get("scheduler_name"), "flash") or "flash",
        "timesteps": _safe_str(info.get("timesteps"), "none") or "none",
        "label": _safe_str(info.get("label") or model_key),
    }


def _hidream_missing_message(root: Path) -> str:
    searched = [str(_hidream_model_dir(root, key)) for key in ("dev_2604_bf16", "dev_2604_fp8", "dev", "dev_fp8")]
    return (
        "HiDream character-sheet workflow needs a supported Dev model, but none was found. "
        "Preferred order is Dev 2604 BF16, Dev 2604 FP8, older Dev BF16, older Dev FP8. "
        "Expected one of:\n" + "\n".join(searched)
    )


def _existing_unique_reference_paths(paths: Any, *, limit: int = 5) -> List[str]:
    out: List[str] = []
    for raw in _as_list(paths):
        path = _safe_str(raw).strip().strip('"')
        if not path:
            continue
        try:
            if os.path.isfile(path) and path not in out:
                out.append(path)
        except Exception:
            continue
        if len(out) >= max(1, int(limit or 5)):
            break
    return out


def _reference_available_paths_from_normalized(ref: Dict[str, Any], *, limit: int = 5) -> List[str]:
    """Return valid direct per-character reference paths from normalized metadata."""
    out: List[str] = []
    hard_limit = max(1, int(limit or 5))
    global_path = _safe_str(ref.get("global_reference_sheet_path"))

    # Prefer explicit char slots.  The global/group sheet is metadata context and
    # should not become a direct HiDream reference unless it is also present as a
    # real per-character slot/selected path.
    sheets = ref.get("character_reference_sheets") if isinstance(ref.get("character_reference_sheets"), dict) else {}
    for key in _character_reference_slot_keys():
        path = _character_reference_existing_path(sheets.get(key))
        if path and path not in out:
            out.append(path)
        if len(out) >= hard_limit:
            return out[:hard_limit]

    for raw_list in (ref.get("selected_reference_sheet_paths"), ref.get("available_reference_sheet_paths")):
        for path in _existing_unique_reference_paths(raw_list, limit=hard_limit):
            if global_path and os.path.normcase(path) == os.path.normcase(global_path):
                continue
            if path not in out:
                out.append(path)
            if len(out) >= hard_limit:
                return out[:hard_limit]
    return out[:hard_limit]



def _selected_reference_paths_for_start_image_handoff(
    shot: Dict[str, Any],
    character_reference: Dict[str, Any],
    payload: Optional[Dict[str, Any]] = None,
    director_plan: Optional[Dict[str, Any]] = None,
    *,
    limit: int = 5,
) -> tuple[List[str], str]:
    """Return reference paths for the actual image-model call.

    A one-ref selected list from review/payload is allowed for explicit solo
    shots, but it must not collapse a group/multi shot when the normalized
    character reference still has more loaded sheets available.
    """
    payload = payload if isinstance(payload, dict) else {}
    director_plan = director_plan if isinstance(director_plan, dict) else {}
    hard_limit = max(1, min(5, int(limit or 5)))

    available_paths = _reference_available_paths_from_normalized(character_reference, limit=hard_limit)
    concept = shot.get("director_scene_concept") if isinstance(shot.get("director_scene_concept"), dict) else {}
    wanted_count, selection_reason = _reference_selection_intent_for_subject_shot(shot, concept, len(available_paths))
    wanted_count = max(0, min(hard_limit, wanted_count))

    candidates: List[tuple[str, Any]] = []
    candidates.append(("shot.selected_reference_sheet_paths", shot.get("selected_reference_sheet_paths")))
    shot_ref = shot.get("character_reference") if isinstance(shot.get("character_reference"), dict) else {}
    candidates.append(("shot.character_reference.selected_reference_sheet_paths", shot_ref.get("selected_reference_sheet_paths")))
    candidates.append(("shot.character_reference.actual_image_model_reference_paths_passed", shot_ref.get("actual_image_model_reference_paths_passed")))
    candidates.append(("normalized_character_reference.selected_reference_sheet_paths", character_reference.get("selected_reference_sheet_paths")))

    payload_ref = payload.get("character_reference") if isinstance(payload.get("character_reference"), dict) else {}
    candidates.append(("payload.character_reference.selected_reference_sheet_paths", payload_ref.get("selected_reference_sheet_paths")))
    candidates.append(("payload.character_reference.actual_image_model_reference_paths_passed", payload_ref.get("actual_image_model_reference_paths_passed")))

    plan_ref = director_plan.get("character_reference") if isinstance(director_plan.get("character_reference"), dict) else {}
    candidates.append(("director_plan.character_reference.selected_reference_sheet_paths", plan_ref.get("selected_reference_sheet_paths")))
    candidates.append(("director_plan.character_reference.actual_image_model_reference_paths_passed", plan_ref.get("actual_image_model_reference_paths_passed")))
    candidates.append(("director_plan.character_reference.available_reference_sheet_paths", plan_ref.get("available_reference_sheet_paths")))

    first_paths: List[str] = []
    first_source = ""
    for source, raw_paths in candidates:
        paths = _existing_unique_reference_paths(raw_paths, limit=hard_limit)
        if paths:
            first_paths = paths
            first_source = source
            break

    if first_paths:
        if wanted_count > len(first_paths) and len(available_paths) > len(first_paths):
            expanded = list(first_paths)
            for path in available_paths:
                if path not in expanded:
                    expanded.append(path)
                if len(expanded) >= wanted_count:
                    break
            if len(expanded) > len(first_paths):
                return expanded[:wanted_count], first_source + "+expanded_from_available_reference_sheet_paths"
        return first_paths[:hard_limit], first_source

    if wanted_count > 0 and available_paths:
        return available_paths[:wanted_count], "normalized_character_reference.available_reference_sheet_paths"
    return [], "no_selected_reference_sheet_paths"



def _hidream_cli_reference_arg(cli_path: Any) -> tuple[str, str]:
    """Return the reference-image flag supported by the installed HiDream CLI.

    FrameVision installations have used more than one helper/CLI variant.  A
    fixed flag can make the bridge look like it passed references while the
    actual helper runs text-only.  This detector keeps the bridge source-only:
    it inspects the local helper file at runtime and picks the first known
    direct-reference flag that is really present.  If no known flag is visible,
    the legacy --ref_images flag is still used so older helpers remain working.
    """
    candidates = [
        "--ref_images",
        "--reference_images",
        "--reference_image",
        "--ref_image",
        "--subject_reference_images",
        "--subject_refs",
        "--input_images",
        "--input_image",
    ]
    text = ""
    try:
        text = Path(_safe_str(cli_path)).read_text(encoding="utf-8", errors="ignore")[:500000]
    except Exception:
        text = ""
    for arg in candidates:
        if arg in text:
            return arg, "detected_in_hidream_cli"
    return "--ref_images", "legacy_default_not_detected"


def _append_hidream_reference_args(cmd: List[str], cli_path: Any, reference_paths: Any) -> Dict[str, Any]:
    ref_paths = _existing_unique_reference_paths(reference_paths, limit=5)
    arg_name, arg_source = _hidream_cli_reference_arg(cli_path)
    if not ref_paths:
        return {
            "reference_paths_requested": [_safe_str(x) for x in _as_list(reference_paths) if _safe_str(x)],
            "reference_paths_passed": [],
            "reference_arg_name": arg_name,
            "reference_arg_source": arg_source,
            "reference_handoff_supported": False,
            "reference_handoff_reason": "no_valid_reference_paths_to_pass",
        }
    # Most FrameVision helpers use a plural nargs-style flag.  If a singular
    # flag is detected, repeat it per image for safer argparse compatibility.
    if arg_name.endswith("_image") or arg_name in {"--ref_image", "--input_image", "--reference_image"}:
        for rp in ref_paths:
            cmd += [arg_name, rp]
    else:
        cmd += [arg_name] + ref_paths
    return {
        "reference_paths_requested": [_safe_str(x) for x in _as_list(reference_paths) if _safe_str(x)],
        "reference_paths_passed": ref_paths,
        "reference_arg_name": arg_name,
        "reference_arg_source": arg_source,
        "reference_handoff_supported": True,
        "reference_handoff_reason": "direct_reference_paths_added_to_hidream_command",
    }


def _run_hidream_start_image(root: Path, *, prompt: str, negative: str, output_path: Path, width: int, height: int, seed: int, log_path: Path, progress_callback: Optional[Callable[[str], None]] = None, reference_paths: Optional[List[str]] = None) -> Dict[str, Any]:
    def _emit(text: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(str(text or ""))
            except Exception:
                pass

    root = Path(root).resolve()
    model_key = _pick_hidream_model_key(root)
    if not model_key:
        raise RuntimeError(_hidream_missing_message(root))
    if not _hidream_model_installed(root, model_key):
        raise RuntimeError(f"HiDream model '{model_key}' is not installed at: {_hidream_model_dir(root, model_key)}")
    cli = _hidream_cli_path(root)
    if not cli or not os.path.isfile(cli):
        raise RuntimeError("HiDream selected but helpers/hidream_cli.py was not found.")
    py = _hidream_python_path(root)
    defaults = _hidream_defaults_for_key(model_key)
    _emit("HiDream priority order: Dev FP8 -> Dev BF16 -> Dev 2604 BF16 -> Dev 2604 FP8")
    _emit(f"HiDream model selected: {defaults.get('label', model_key)} ({model_key})")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.exists():
            output_path.unlink()
    except Exception:
        pass

    cmd = [
        py,
        cli,
        "--model_key", model_key,
        "--prompt", prompt,
        "--output_image", str(output_path),
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--seed", str(int(seed)),
        "--steps", str(int(defaults["steps"])),
        "--guidance_scale", str(float(defaults["guidance_scale"])),
        "--shift", str(float(defaults["shift"])),
        "--scheduler_name", str(defaults["scheduler_name"]),
        "--timesteps", str(defaults["timesteps"]),
        "--noise_scale_start", "7.5",
        "--noise_scale_end", "7.5",
        "--noise_clip_std", "2.5",
        "--device_map", "cuda",
        "--resolution_mode", "framevision",
    ]
    if negative:
        cmd += ["--negative_prompt", negative]
    reference_handoff = _append_hidream_reference_args(cmd, cli, reference_paths)
    ref_paths = list(reference_handoff.get("reference_paths_passed") or [])

    _emit(f"Creating start image with HiDream {defaults['label']}...")
    if ref_paths:
        _emit(f"Passing {len(ref_paths)} subject reference image(s) to HiDream.")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as lf:
        lf.write("[musicclip bridge] LTX single-shot start image generation\n")
        lf.write("HiDream priority order: Dev FP8 -> Dev BF16 -> Dev 2604 BF16 -> Dev 2604 FP8\n")
        lf.write(f"Image model: HiDream {defaults['label']}\n")
        lf.write(f"Output: {output_path}\n")
        lf.write("Command:\n")
        lf.write(" ".join([str(x) for x in cmd]) + "\n\n")
        lf.flush()
        env = os.environ.copy()
        if ref_paths:
            # Harmless compatibility side-channel for newer/alternate HiDream
            # helpers that read references from environment instead of argv.
            try:
                env["FRAMEVISION_HIDREAM_REFERENCE_IMAGES"] = json.dumps(ref_paths, ensure_ascii=False)
                env["FRAMEVISION_HIDREAM_REFERENCE_IMAGE_PATHS"] = os.pathsep.join(ref_paths)
            except Exception:
                pass
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            env=env,
        )
        if proc.stdout:
            for raw in proc.stdout:
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip() if isinstance(raw, bytes) else str(raw).rstrip()
                except Exception:
                    line = str(raw).rstrip()
                if line:
                    try:
                        lf.write(line + "\n")
                        lf.flush()
                    except Exception:
                        pass
                    _emit(line[:240])
        rc = int(proc.wait() or 0)
        lf.write(f"\nExit code: {rc}\n")
    if rc != 0:
        raise RuntimeError(f"HiDream CLI failed (exit code {rc}). See log: {log_path}")
    if not output_path.is_file() or output_path.stat().st_size < 1024:
        raise RuntimeError(f"HiDream finished but no valid output image was created: {output_path}")
    return {
        "command": cmd,
        "model_files": {
            "hidream_model_key": model_key,
            "hidream_model_dir": str(_hidream_model_dir(root, model_key)),
            "hidream_cli": cli,
            "python": py,
            "defaults": defaults,
        },
        "reference_handoff": reference_handoff,
        "actual_image_model_reference_paths_passed": ref_paths,
        "rc": rc,
        "out_file": str(output_path),
    }


def _run_hidream_batch_start_images(root: Path, *, jobs: List[Dict[str, Any]], progress_callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    def _emit(text: str) -> None:
        if callable(progress_callback):
            try:
                progress_callback(str(text or ""))
            except Exception:
                pass

    if not jobs:
        return {"ok": True, "jobs": [], "job_count": 0}
    root = Path(root).resolve()
    model_key = _pick_hidream_model_key(root)
    if not model_key:
        raise RuntimeError(_hidream_missing_message(root))
    cli = _hidream_batch_cli_path(root)
    if not cli or not os.path.isfile(cli):
        raise RuntimeError("HiDream batch selected but helpers/hidream_batch_cli.py was not found.")
    py = _hidream_python_path(root)

    first_job = jobs[0] if isinstance(jobs[0], dict) else {}
    batch_root = Path(_safe_str(first_job.get("output_dir") or Path(_safe_str(first_job.get("start_image_path"))).parent or root)).expanduser().resolve()
    batch_dir = (batch_root / "_hidream_batch").resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = batch_dir / "hidream_batch_manifest.json"
    results_path = batch_dir / "hidream_batch_results.json"

    manifest_jobs: List[Dict[str, Any]] = []
    for job in jobs:
        manifest_jobs.append({
            "shot_id": _safe_str(job.get("shot_id")),
            "prompt": _safe_str(job.get("prompt")),
            "negative_prompt": _safe_str(job.get("negative_prompt")),
            "output_image": _safe_str(job.get("start_image_path")),
            "payload_json_path": _safe_str(job.get("payload_path")),
            "log_path": _safe_str(job.get("log_path")),
            "width": _safe_int(job.get("width"), 1280),
            "height": _safe_int(job.get("height"), 720),
            "seed": _safe_int(job.get("seed"), -1),
            "steps": _safe_int(job.get("steps"), 28),
            "guidance_scale": _safe_float(job.get("guidance_scale"), 0.0),
            "shift": _safe_float(job.get("shift"), 1.0),
            "scheduler_name": _safe_str(job.get("scheduler_name"), "flash"),
            "timesteps": _safe_str(job.get("timesteps"), "none"),
            "noise_scale_start": _safe_float(job.get("noise_scale_start"), 7.5),
            "noise_scale_end": _safe_float(job.get("noise_scale_end"), 7.5),
            "noise_clip_std": _safe_float(job.get("noise_clip_std"), 2.5),
            "keep_original_aspect": _safe_bool(job.get("keep_original_aspect"), False),
            "reference_paths": list(job.get("reference_paths") or []),
            "debug_payload_base": dict(job.get("debug_payload_base") or {}),
        })
    _write_json_file(manifest_path, {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_key": model_key,
        "jobs": manifest_jobs,
    })

    cmd = [
        py,
        cli,
        "--manifest_json", str(manifest_path),
        "--results_json", str(results_path),
        "--resolution_mode", "framevision",
    ]
    _emit(f"Loading HiDream once for a warm batch of {len(manifest_jobs)} image(s)...")
    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        env=env,
    )
    if proc.stdout:
        for raw in proc.stdout:
            try:
                line = raw.decode("utf-8", errors="replace").rstrip() if isinstance(raw, bytes) else str(raw).rstrip()
            except Exception:
                line = str(raw).rstrip()
            if line:
                _emit(line[:240])
    rc = int(proc.wait() or 0)
    results: Dict[str, Any] = {}
    if results_path.is_file():
        try:
            results = _read_json_file(str(results_path))
        except Exception:
            results = {}
    if rc != 0 and not isinstance(results, dict):
        raise RuntimeError(f"HiDream batch CLI failed (exit code {rc}).")
    if not isinstance(results, dict):
        raise RuntimeError("HiDream batch CLI did not return a readable results JSON.")
    results.setdefault("rc", rc)
    results.setdefault("manifest_json", str(manifest_path))
    results.setdefault("results_json", str(results_path))
    return results




def generate_ltx_start_image_for_shot(payload: dict) -> dict:
    """Generate one start image for one selected LTX director shot.

    This stays separate from the LTX test runner: it only writes LTXxx_start.png
    and debug JSON/log files inside a timestamped ltx_tests folder.
    """
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Start-image payload was not a dictionary."}
        progress_callback = payload.get("progress_callback")

        def _emit(message: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(str(message or ""))
                except Exception:
                    pass

        _emit("Loading LTX director plan...")
        root_raw = _safe_str(payload.get("root_dir"))
        root = Path(root_raw).resolve() if root_raw else _project_root()
        plan_path = Path(_safe_str(payload.get("ltx_director_plan_path"))).expanduser().resolve()
        if not plan_path.is_file():
            return {"ok": False, "message": f"LTX director plan was not found: {plan_path}"}
        director_plan = _read_json_file(str(plan_path))
        safety = _enforce_ltx_start_end_duration_contract(
            director_plan,
            plan_path=plan_path,
            root=root,
            enabled=_payload_ltx_edge_duration_safety_enabled(payload, director_plan),
            refresh_audio_chunks=True,
        )
        if bool(safety.get("changed")):
            try:
                existing_warnings = _as_list(director_plan.get("warnings"))
                merged = existing_warnings + [w for w in _as_list(safety.get("warnings")) if w]
                if merged:
                    director_plan["warnings"] = merged[:80]
                _write_json_file(plan_path, director_plan)
                _emit("Updated LTX plan timing: first clip is at least 3 seconds and final clip targets 5 seconds.")
            except Exception as exc:
                _emit(f"Warning: could not save LTX duration safety update: {exc}")
        character_reference = _character_reference_from_sources(payload, director_plan)
        shot_id = _safe_str(payload.get("shot_id"))
        if not shot_id:
            return {"ok": False, "message": "No LTX shot was selected."}
        shot = _find_director_shot(director_plan, shot_id)
        if not shot:
            return {"ok": False, "message": f"Selected shot was not found in the director plan: {shot_id}"}
        _chars, _groups, shot = _identity_context_from_plan(shot, director_plan, payload, progress_callback)
        brief = _normalize_creative_brief(director_plan.get("creative_brief"))
        # Protect old director plans too: start-image prompts for non-vocal shots
        # must not ask for microphones, singing, lipsync, or mouth-readable framing.
        shot = _protect_lipsync_confidence(shot, song_duration=max([_safe_float(x.get("song_end"), 0.0) for x in _director_plan_shots(director_plan)] + [0.0]))
        # Recompile prompt fields before reference routing.  The final concept/prompt,
        # not stale b-roll/background labels, decides whether human refs are needed.
        shot = _director_compile_shot_prompts(shot, brief)
        image_model = _safe_str(payload.get("image_model") or payload.get("image_mode"), "flux_klein_9b").lower() or "flux_klein_9b"
        # Re-select per-character reference sheets after final prompt compilation and
        # after knowing whether the selected image model can pass direct references.
        shot = _apply_character_reference_to_item(
            shot,
            brief,
            character_reference,
            image_prompt_keys=["director_image_prompt"],
            passed_to_model=(image_model == "hidream"),
        )
        character_reference = dict(shot.get("character_reference") or character_reference)
        shot_id = _safe_str(shot.get("id")) or shot_id
        _emit(f"Selected {shot_id}")

        if image_model in {"existing", "use existing start image", "existing_start_image"}:
            return {"ok": False, "message": "Use existing start image does not need generation. Pick Flux Klein 9B, Z-Image Turbo, or HiDream."}

        out_dir_raw = _safe_str(payload.get("output_dir"))
        if out_dir_raw:
            test_dir = Path(out_dir_raw).expanduser().resolve()
            test_dir.mkdir(parents=True, exist_ok=True)
        else:
            test_dir = _make_ltx_test_dir(plan_path, shot_id)
        stem = _safe_stem(shot_id)
        start_path = _safe_child_file_path(test_dir, payload.get("start_image_name"), f"{stem}_start.png")
        payload_path = _safe_child_file_path(test_dir, payload.get("start_image_payload_name"), f"{stem}_start_image_payload.json")
        log_path = _safe_child_file_path(test_dir, payload.get("start_image_log_name"), f"{stem}_imagegen.log.txt")

        raw_prompt, selected_prompt_source = _select_start_image_prompt_source(shot, payload)
        if not raw_prompt:
            return {"ok": False, "message": f"{shot_id} has no director image prompt."}
        lyrics = shot.get("lyrics") or shot.get("clip_relative_lyrics") or ""
        prompt, prompt_removed_phrases = _sanitize_final_start_image_prompt_for_model(
            raw_prompt,
            brief=brief,
            shot=shot,
            lyrics=lyrics,
            image_model=image_model,
        )
        negative = _negative_prompt_with_no_visible_text(
            _join_parts([shot.get("director_negative_prompt") or shot.get("negative_prompt"), _anti_collage_negative_terms(image_model)]),
            max_len=1200,
        )
        if not prompt:
            return {"ok": False, "message": f"{shot_id} image prompt became empty after safety cleanup."}
        requested_resolution_source = payload.get("resolution") or shot.get("resolution") or director_plan.get("resolution")
        _req_w, _req_h, _req_res = _parse_resolution_value(requested_resolution_source, "1280x704")
        requested_portrait = bool(_req_h > _req_w)
        hidream_reference_edit_resolution = "896x1600" if requested_portrait else "1600x896"
        default_resolution = "1280x704"
        resolution_source = requested_resolution_source
        if image_model == "z_image":
            default_resolution = "768x1344" if requested_portrait else "1344x768"
        elif image_model == "hidream":
            # Keep normal HiDream start-image generation high quality, but obey
            # the Music Clip Creator landscape/portrait choice. Reference/edit
            # shots are lowered after reference routing below.
            default_resolution = "1088x1920" if requested_portrait else "1920x1088"
        width, height, resolution = _parse_resolution_value(resolution_source, default_resolution)
        seed_raw = payload.get("seed", None)
        if seed_raw in (None, ""):
            seed = int(time.time() * 1000) % 2147483647
        else:
            seed = _safe_int(seed_raw, int(time.time() * 1000) % 2147483647)

        prepare_only = _safe_bool(payload.get("prepare_only"), False)

        selected_reference_paths, selected_reference_paths_source = _selected_reference_paths_for_start_image_handoff(shot, character_reference, payload, director_plan, limit=5)
        available_reference_sheet_paths = _reference_available_paths_from_normalized(character_reference, limit=5)
        loaded_reference_count = len(available_reference_sheet_paths)
        wanted_reference_count, selection_reason = _reference_selection_intent_for_subject_shot(shot, shot.get("director_scene_concept") if isinstance(shot.get("director_scene_concept"), dict) else {}, loaded_reference_count)
        wanted_reference_count = max(0, min(5, wanted_reference_count))
        source_had_full_character_reference_sheets = bool(character_reference.get("source_had_full_character_reference_sheets"))
        collapsed_to_single_ref_warning = bool(loaded_reference_count > 1 and wanted_reference_count > 1 and len(selected_reference_paths) == 1)
        shot_subject_mode = _safe_str(shot.get("shot_subject_mode"), "environment_only") or "environment_only"
        image_model_forced_to_hidream_for_reference = False
        if selected_reference_paths and image_model == "hidream":
            # HiDream reference/edit workflow test size: keep plain HiDream
            # generation at its existing default, but use 1376x768 whenever
            # direct reference images are actually selected for the shot.
            width, height, resolution = _parse_resolution_value(hidream_reference_edit_resolution, hidream_reference_edit_resolution)
            shot["image_model_reference_resolution_override"] = hidream_reference_edit_resolution
            shot["reference_routing_reason"] = _join_parts([shot.get("reference_routing_reason"), f"HiDream reference/edit resolution override: {hidream_reference_edit_resolution}"])
        if selected_reference_paths and image_model != "hidream":
            # At this point references have already been selected by the prompt
            # director / character-reference router.  Do not let a text-only image
            # model generate random people when direct reference paths exist.
            image_model_forced_to_hidream_for_reference = True
            image_model = "hidream"
            # Match the HiDream reference/edit workflow size when refs force a
            # text-only selection over to HiDream. Plain HiDream generation still
            # keeps its existing default above.
            width, height, resolution = _parse_resolution_value(hidream_reference_edit_resolution, hidream_reference_edit_resolution)
            shot["image_model_reference_resolution_override"] = hidream_reference_edit_resolution
            shot["image_model_reference_mode"] = "direct_reference_image"
            shot["reference_routing_reason"] = _join_parts([shot.get("reference_routing_reason"), "start-image model forced to HiDream because selected reference paths are loaded"])
        reference_paths_to_pass = selected_reference_paths if image_model == "hidream" else []
        character_reference_passed = False
        image_model_reference_mode = "direct_reference_image_pending" if reference_paths_to_pass else ("environment_only" if shot_subject_mode == "environment_only" else (_safe_str(shot.get("image_model_reference_mode")) or "text_only"))

        debug_payload_base: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "shot_id": shot_id,
            "source_director_plan_path": str(plan_path),
            "selected_shot": shot,
            "image_model": image_model,
            "image_model_forced_to_hidream_for_reference": bool(image_model_forced_to_hidream_for_reference),
            "image_model_reference_resolution_override": _safe_str(shot.get("image_model_reference_resolution_override")),
            "selected_prompt_source": selected_prompt_source,
            "raw_selected_image_prompt": raw_prompt,
            "sanitized_image_prompt_used": prompt,
            "image_prompt_word_count": _director_word_count(prompt),
            "removed_duplicate_phrases": prompt_removed_phrases,
            "director_raw_llm_output": shot.get("director_raw_llm_output") if isinstance(shot.get("director_raw_llm_output"), dict) and shot.get("director_raw_llm_output") else {},
            "fallback_reason": _safe_str(shot.get("fallback_reason") or ("director_raw_llm_output_empty" if not shot.get("director_raw_llm_output") else "")),
            "negative_prompt_used": negative,
            "seed": seed,
            "resolution": resolution,
            "width": width,
            "height": height,
            "output_path": str(start_path),
            "command_summary": {"type": "pending_prepare_only"} if prepare_only else {},
            "model_files_used": {},
            "shot_subject_mode": shot_subject_mode,
            "visible_subject_detected": bool(shot.get("visible_subject_detected")),
            "reference_eligible": bool(shot.get("reference_eligible")),
            "reference_routing_reason": _safe_str(shot.get("reference_routing_reason")),
            "available_reference_sheet_paths": available_reference_sheet_paths,
            "loaded_reference_count": loaded_reference_count,
            "wanted_reference_count": wanted_reference_count,
            "selected_reference_sheet_paths": selected_reference_paths,
            "selected_reference_sheet_paths_source": selected_reference_paths_source,
            "selection_reason": selection_reason,
            "source_had_full_character_reference_sheets": source_had_full_character_reference_sheets,
            "collapsed_to_single_ref_warning": collapsed_to_single_ref_warning,
            "reference_paths_requested_for_handoff": reference_paths_to_pass,
            "character_reference_passed_to_image_model": False,
            "image_model_reference_mode": image_model_reference_mode,
            "actual_image_model_reference_paths_passed": [],
            "reference_handoff": {
                "reference_paths_requested": reference_paths_to_pass,
                "reference_paths_passed": [],
                "reference_arg_name": "pending",
                "reference_arg_source": "prepare_only",
                "reference_handoff_supported": bool(reference_paths_to_pass),
                "reference_handoff_reason": "prepare_only_pending_generation",
            },
            "skipped_reference_reason": _safe_str("selected_reference_paths_not_added_to_hidream_command" if selected_reference_paths and image_model == "hidream" else "image_model_has_no_direct_reference_input" if selected_reference_paths and image_model != "hidream" else shot.get("skipped_reference_reason") or ""),
            "environment_prompt_omitted_subjects": bool(shot.get("environment_prompt_omitted_subjects")),
            "character_reference": {
                "enabled": bool(_safe_bool(character_reference.get("enabled"), False)),
                "global_reference_sheet_path": _safe_str(character_reference.get("global_reference_sheet_path")),
                "global_reference_direct_model_use": False,
                "global_reference_usage": "metadata_text_context_only",
                "reference_sheet_path": selected_reference_paths[0] if selected_reference_paths else "",
                "character_reference_sheets": character_reference.get("character_reference_sheets") if isinstance(character_reference.get("character_reference_sheets"), dict) else {},
                "available_reference_sheet_paths": available_reference_sheet_paths,
                "loaded_reference_count": loaded_reference_count,
                "wanted_reference_count": wanted_reference_count,
                "selected_reference_sheet_paths": selected_reference_paths,
                "selected_reference_sheet_paths_source": selected_reference_paths_source,
                "selection_reason": selection_reason,
                "source_had_full_character_reference_sheets": source_had_full_character_reference_sheets,
                "collapsed_to_single_ref_warning": collapsed_to_single_ref_warning,
                "actual_image_model_reference_paths_passed": [],
                "passed_to_model": False,
                "global_reference_passed_to_model": False,
                "model_reference_mode": image_model_reference_mode,
                "hidream_multi_reference_policy": "per_character_sheets_only",
                "mode": "reference_sheet",
                "shot_subject_mode": shot_subject_mode,
                "visible_subject_detected": bool(shot.get("visible_subject_detected")),
                "reference_eligible": bool(shot.get("reference_eligible")),
                "reference_routing_reason": _safe_str(shot.get("reference_routing_reason")),
                "skipped_reference_reason": _safe_str(shot.get("skipped_reference_reason")),
                "environment_prompt_omitted_subjects": bool(shot.get("environment_prompt_omitted_subjects")),
            },
            "character_reference_warnings": _plan_character_reference_warnings(character_reference, passed_to_model=False),
        }
        if prepare_only:
            return {
                "ok": True,
                "prepared": True,
                "shot_id": shot_id,
                "image_model": image_model,
                "output_dir": str(test_dir),
                "start_image_path": str(start_path),
                "payload_path": str(payload_path),
                "log_path": str(log_path),
                "prompt": prompt,
                "negative_prompt": negative,
                "seed": seed,
                "resolution": resolution,
                "width": width,
                "height": height,
                "steps": _safe_int(_hidream_defaults_for_key(_pick_hidream_model_key(root) or "dev").get("steps"), 28) if image_model == "hidream" else 0,
                "guidance_scale": _safe_float(_hidream_defaults_for_key(_pick_hidream_model_key(root) or "dev").get("guidance_scale"), 0.0) if image_model == "hidream" else 0.0,
                "shift": _safe_float(_hidream_defaults_for_key(_pick_hidream_model_key(root) or "dev").get("shift"), 1.0) if image_model == "hidream" else 1.0,
                "scheduler_name": _safe_str(_hidream_defaults_for_key(_pick_hidream_model_key(root) or "dev").get("scheduler_name"), "flash") if image_model == "hidream" else "",
                "timesteps": _safe_str(_hidream_defaults_for_key(_pick_hidream_model_key(root) or "dev").get("timesteps"), "none") if image_model == "hidream" else "",
                "noise_scale_start": 7.5,
                "noise_scale_end": 7.5,
                "noise_clip_std": 2.5,
                "keep_original_aspect": False,
                "reference_paths": list(reference_paths_to_pass),
                "debug_payload_base": debug_payload_base,
                "message": "Start image prepared.",
            }

        command_summary: Dict[str, Any] = {}
        model_files: Dict[str, Any] = {}
        actual_image_model_reference_paths_passed: List[str] = []
        reference_handoff: Dict[str, Any] = {}
        if image_model == "flux_klein_9b":
            result = _run_flux_klein_start_image(
                root,
                prompt=prompt,
                negative=negative,
                output_path=start_path,
                width=width,
                height=height,
                seed=seed,
                log_path=log_path,
                progress_callback=progress_callback,
            )
            command_summary = {"argv": result.get("command") or []}
            model_files = result.get("model_files") if isinstance(result.get("model_files"), dict) else {}
        elif image_model == "z_image":
            result = _run_zimage_start_image(
                root,
                prompt=prompt,
                negative=negative,
                output_path=start_path,
                width=width,
                height=height,
                seed=seed,
                log_path=log_path,
                progress_callback=progress_callback,
            )
            command_summary = result.get("command") if isinstance(result.get("command"), dict) else {"summary": result.get("command") or []}
            model_files = result.get("model_files") if isinstance(result.get("model_files"), dict) else {}
        elif image_model == "hidream":
            result = _run_hidream_start_image(
                root,
                prompt=prompt,
                negative=negative,
                output_path=start_path,
                width=width,
                height=height,
                seed=seed,
                log_path=log_path,
                progress_callback=progress_callback,
                reference_paths=reference_paths_to_pass,
            )
            command_summary = {"argv": result.get("command") or []}
            model_files = result.get("model_files") if isinstance(result.get("model_files"), dict) else {}
            actual_image_model_reference_paths_passed = _existing_unique_reference_paths(result.get("actual_image_model_reference_paths_passed") or ((result.get("reference_handoff") or {}).get("reference_paths_passed") if isinstance(result.get("reference_handoff"), dict) else []), limit=5)
            reference_handoff = result.get("reference_handoff") if isinstance(result.get("reference_handoff"), dict) else {}
        else:
            return {"ok": False, "message": f"Unknown start-image model: {image_model}"}

        character_reference_passed = bool(image_model == "hidream" and actual_image_model_reference_paths_passed)
        image_model_reference_mode = "direct_reference_image" if character_reference_passed else ("environment_only" if not selected_reference_paths and shot_subject_mode == "environment_only" else "text_only_reference_not_passed")
        shot["available_reference_sheet_paths"] = available_reference_sheet_paths
        shot["loaded_reference_count"] = loaded_reference_count
        shot["wanted_reference_count"] = wanted_reference_count
        shot["selected_reference_sheet_paths"] = selected_reference_paths
        shot["selected_reference_sheet_paths_source"] = selected_reference_paths_source
        shot["selection_reason"] = selection_reason
        shot["source_had_full_character_reference_sheets"] = source_had_full_character_reference_sheets
        shot["collapsed_to_single_ref_warning"] = collapsed_to_single_ref_warning
        shot["actual_image_model_reference_paths_passed"] = actual_image_model_reference_paths_passed
        shot["character_reference_passed_to_image_model"] = character_reference_passed
        shot["image_model_reference_mode"] = image_model_reference_mode
        shot["image_model_forced_to_hidream_for_reference"] = bool(image_model_forced_to_hidream_for_reference)
        if isinstance(shot.get("character_reference"), dict):
            shot_ref = dict(shot.get("character_reference") or {})
            shot_ref["available_reference_sheet_paths"] = available_reference_sheet_paths
            shot_ref["loaded_reference_count"] = loaded_reference_count
            shot_ref["wanted_reference_count"] = wanted_reference_count
            shot_ref["selected_reference_sheet_paths"] = selected_reference_paths
            shot_ref["selected_reference_sheet_paths_source"] = selected_reference_paths_source
            shot_ref["selection_reason"] = selection_reason
            shot_ref["source_had_full_character_reference_sheets"] = source_had_full_character_reference_sheets
            shot_ref["collapsed_to_single_ref_warning"] = collapsed_to_single_ref_warning
            shot_ref["actual_image_model_reference_paths_passed"] = actual_image_model_reference_paths_passed
            shot_ref["passed_to_model"] = character_reference_passed
            shot_ref["model_reference_mode"] = image_model_reference_mode
            shot_ref["skipped_reference_reason"] = "" if character_reference_passed else _safe_str(shot.get("skipped_reference_reason"))
            shot["character_reference"] = shot_ref

        debug_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "shot_id": shot_id,
            "source_director_plan_path": str(plan_path),
            "selected_shot": shot,
            "image_model": image_model,
            "image_model_forced_to_hidream_for_reference": bool(image_model_forced_to_hidream_for_reference),
            "image_model_reference_resolution_override": _safe_str(shot.get("image_model_reference_resolution_override")),
            "selected_prompt_source": selected_prompt_source,
            "raw_selected_image_prompt": raw_prompt,
            "sanitized_image_prompt_used": prompt,
            "image_prompt_word_count": _director_word_count(prompt),
            "removed_duplicate_phrases": prompt_removed_phrases,
            "director_raw_llm_output": shot.get("director_raw_llm_output") if isinstance(shot.get("director_raw_llm_output"), dict) and shot.get("director_raw_llm_output") else {},
            "fallback_reason": _safe_str(shot.get("fallback_reason") or ("director_raw_llm_output_empty" if not shot.get("director_raw_llm_output") else "")),
            "negative_prompt_used": negative,
            "seed": seed,
            "resolution": resolution,
            "width": width,
            "height": height,
            "output_path": str(start_path),
            "command_summary": command_summary,
            "model_files_used": model_files,
            "shot_subject_mode": shot_subject_mode,
            "visible_subject_detected": bool(shot.get("visible_subject_detected")),
            "reference_eligible": bool(shot.get("reference_eligible")),
            "reference_routing_reason": _safe_str(shot.get("reference_routing_reason")),
            "available_reference_sheet_paths": available_reference_sheet_paths,
            "loaded_reference_count": loaded_reference_count,
            "wanted_reference_count": wanted_reference_count,
            "selected_reference_sheet_paths": selected_reference_paths,
            "selected_reference_sheet_paths_source": selected_reference_paths_source,
            "selection_reason": selection_reason,
            "source_had_full_character_reference_sheets": source_had_full_character_reference_sheets,
            "collapsed_to_single_ref_warning": collapsed_to_single_ref_warning,
            "reference_paths_requested_for_handoff": reference_paths_to_pass,
            "character_reference_passed_to_image_model": character_reference_passed,
            "image_model_reference_mode": image_model_reference_mode,
            "actual_image_model_reference_paths_passed": actual_image_model_reference_paths_passed,
            "reference_handoff": reference_handoff,
            "skipped_reference_reason": "" if character_reference_passed else _safe_str("selected_reference_paths_not_added_to_hidream_command" if selected_reference_paths and image_model == "hidream" else "image_model_has_no_direct_reference_input" if selected_reference_paths and image_model != "hidream" else shot.get("skipped_reference_reason") or ""),
            "environment_prompt_omitted_subjects": bool(shot.get("environment_prompt_omitted_subjects")),
            "character_reference": {
                "enabled": bool(_safe_bool(character_reference.get("enabled"), False)),
                "global_reference_sheet_path": _safe_str(character_reference.get("global_reference_sheet_path")),
                "global_reference_direct_model_use": False,
                "global_reference_usage": "metadata_text_context_only",
                "reference_sheet_path": selected_reference_paths[0] if selected_reference_paths else "",
                "character_reference_sheets": character_reference.get("character_reference_sheets") if isinstance(character_reference.get("character_reference_sheets"), dict) else {},
                "available_reference_sheet_paths": available_reference_sheet_paths,
                "loaded_reference_count": loaded_reference_count,
                "wanted_reference_count": wanted_reference_count,
                "selected_reference_sheet_paths": selected_reference_paths,
                "selected_reference_sheet_paths_source": selected_reference_paths_source,
                "selection_reason": selection_reason,
                "source_had_full_character_reference_sheets": source_had_full_character_reference_sheets,
                "collapsed_to_single_ref_warning": collapsed_to_single_ref_warning,
                "actual_image_model_reference_paths_passed": actual_image_model_reference_paths_passed,
                "passed_to_model": character_reference_passed,
                "global_reference_passed_to_model": False,
                "model_reference_mode": image_model_reference_mode,
                "hidream_multi_reference_policy": "per_character_sheets_only",
                "mode": "reference_sheet",
                "shot_subject_mode": shot_subject_mode,
                "visible_subject_detected": bool(shot.get("visible_subject_detected")),
                "reference_eligible": bool(shot.get("reference_eligible")),
                "reference_routing_reason": _safe_str(shot.get("reference_routing_reason")),
                "skipped_reference_reason": "" if character_reference_passed else _safe_str(shot.get("skipped_reference_reason")),
                "environment_prompt_omitted_subjects": bool(shot.get("environment_prompt_omitted_subjects")),
            },
            "character_reference_warnings": _plan_character_reference_warnings(character_reference, passed_to_model=character_reference_passed),
        }
        _write_json_file(payload_path, debug_payload)
        _emit(f"Start image saved: {start_path}")
        return {
            "ok": True,
            "shot_id": shot_id,
            "image_model": image_model,
            "output_dir": str(test_dir),
            "start_image_path": str(start_path),
            "payload_path": str(payload_path),
            "log_path": str(log_path),
            "character_reference": debug_payload.get("character_reference"),
            "shot_subject_mode": shot_subject_mode,
            "reference_eligible": bool(shot.get("reference_eligible")),
            "available_reference_sheet_paths": available_reference_sheet_paths,
            "loaded_reference_count": loaded_reference_count,
            "wanted_reference_count": wanted_reference_count,
            "selected_reference_sheet_paths": selected_reference_paths,
            "selected_reference_sheet_paths_source": selected_reference_paths_source,
            "selection_reason": selection_reason,
            "source_had_full_character_reference_sheets": source_had_full_character_reference_sheets,
            "collapsed_to_single_ref_warning": collapsed_to_single_ref_warning,
            "character_reference_passed_to_image_model": character_reference_passed,
            "image_model_reference_mode": image_model_reference_mode,
            "actual_image_model_reference_paths_passed": actual_image_model_reference_paths_passed,
            "reference_handoff": reference_handoff,
            "image_model_forced_to_hidream_for_reference": bool(image_model_forced_to_hidream_for_reference),
            "character_reference_warnings": debug_payload.get("character_reference_warnings") or [],
            "message": "Start image generated.",
        }
    except Exception as exc:
        return {"ok": False, "message": f"Start image generation failed: {exc}"}


# -----------------------------
# Chunk 6A: single LTX image-to-video shot test runner
# -----------------------------

def _ltx23_guess_wangp_root(root: Path) -> str:
    guesses: List[str] = []
    try:
        env_root = _safe_str(os.environ.get("FRAMEVISION_LTX23_WANGP_ROOT"))
    except Exception:
        env_root = ""
    if env_root:
        guesses.append(env_root)
    if os.name == "nt":
        guesses += [r"C:\\WanGP\\Wan2GP", r"C:\WanGP"]
    try:
        guesses += [
            str((root.parent / "Wan2GP").resolve()),
            str((root.parent / "WanGP" / "Wan2GP").resolve()),
            str((root / "Wan2GP").resolve()),
            str((root / "WanGP" / "Wan2GP").resolve()),
        ]
    except Exception:
        pass
    for raw in guesses:
        s = _safe_str(raw)
        if not s:
            continue
        try:
            pp = Path(s)
            if pp.is_dir() and (pp / "wgp.py").exists():
                return str(pp.resolve())
            if pp.is_dir() and (pp / "Wan2GP" / "wgp.py").exists():
                return str((pp / "Wan2GP").resolve())
        except Exception:
            continue
    return ""


def _ltx23_bridge_config(root: Path) -> Dict[str, str]:
    root_dir = _ltx23_guess_wangp_root(root)
    try:
        env_lora = _safe_str(os.environ.get("FRAMEVISION_LTX23_LORA_FILE"))
    except Exception:
        env_lora = ""
    try:
        env_json = _safe_str(os.environ.get("FRAMEVISION_LTX23_LORA_JSON"))
    except Exception:
        env_json = ""
    try:
        env_wgp = _safe_str(os.environ.get("FRAMEVISION_LTX23_WGP_PY"))
    except Exception:
        env_wgp = ""
    wgp_py = env_wgp or (str((Path(root_dir) / "wgp.py").resolve()) if root_dir else "")
    lora_file = env_lora or (str((Path(root_dir) / "loras" / "ltx2" / "ltx2.3-transition.safetensors").resolve()) if root_dir else "")
    lora_json = env_json or (str((Path(root_dir) / "loras" / "ltx2" / "LTX-2.3.json").resolve()) if root_dir else "")
    return {
        "wangp_root": str(root_dir or ""),
        "wgp_py": str(wgp_py or ""),
        "lora_file": str(lora_file or ""),
        "lora_json": str(lora_json or ""),
    }


def _ltx23_vramlab_ui_settings(root: Path) -> Dict[str, Any]:
    path = Path(root).resolve() / "presets" / "setsave" / "ltx23_ui.json"
    data = _read_json_file(path)
    if not isinstance(data, dict):
        data = {}
    data["__settings_path"] = str(path)
    return data


def _ltx23_path_from_settings(root: Path, raw: Any, fallback: Path) -> str:
    text = _safe_str(raw)
    if not text:
        return str(fallback)
    p = Path(text).expanduser()
    if not p.is_absolute():
        p = Path(root).resolve() / p
    return str(p.resolve())


def _ltx23_default_python(root: Path, settings: Dict[str, Any]) -> str:
    raw = _safe_str(settings.get("python_exe"))
    if raw and Path(raw).expanduser().exists():
        return str(Path(raw).expanduser().resolve())
    for cand in (
        Path(root) / "environments" / ".ltx23" / "python.exe",
        Path(root) / "environments" / ".ltx23" / "Scripts" / "python.exe",
        Path(root) / "environments" / ".ltx23_native" / "Scripts" / "python.exe",
    ):
        if cand.is_file():
            return str(cand.resolve())
    return sys.executable


def _ltx23_cache_mode(value: Any) -> str:
    text = _safe_str(value, "read").lower()
    if "rebuild" in text:
        return "rebuild"
    if "off" in text or "official" in text:
        return "off"
    if "auto" in text:
        return "auto"
    return "read"


def _ltx23_append_existing(cmd: List[str], flag: str, path: str) -> None:
    try:
        p = Path(path).expanduser()
        if path and p.exists():
            cmd.extend([flag, str(p.resolve())])
    except Exception:
        pass


def _ltx23_manual_vram_overrides_enabled(settings: Dict[str, Any]) -> bool:
    """Only forward advanced VRAM override flags when explicitly enabled.

    The native LTX/VRAM Lab CLI owns the modern profile defaults. Saved UI
    values in presets/setsave/ltx23_ui.json are not treated as overrides unless
    a future UI/test patch deliberately enables one of these flags.
    """
    try:
        env_value = _safe_str(os.environ.get("FRAMEVISION_MUSICCLIP_LTX_MANUAL_VRAM_OVERRIDES")).lower()
        if env_value in {"1", "true", "yes", "on"}:
            return True
    except Exception:
        pass
    for key in (
        "manual_vram_overrides",
        "use_manual_vram_overrides",
        "ltx_manual_vram_overrides",
        "enable_manual_vram_overrides",
        "advanced_vram_overrides_enabled",
    ):
        if _safe_bool(settings.get(key), False):
            return True
    return False


def _ltx23_build_vramlab_direct_args(
    *,
    root: Path,
    prompt: str,
    start_image_path: Path,
    out_path: Path,
    fps: int,
    frame_count: int,
    steps: int,
    resolution: str,
    audio_path: str,
    seed: Optional[int],
    lora_file: str = "",
) -> List[str]:
    """Build the real own-workflow command. No tiny wrapper CLI, no Wan2GP CLI dependency."""
    root = Path(root).resolve()
    settings = _ltx23_vramlab_ui_settings(root)
    vram_cli = root / "helpers" / "ltx23_vram_lab_cli.py"
    if not vram_cli.is_file():
        raise FileNotFoundError(f"Missing own LTX VRAMLab CLI: {vram_cli}")
    python_exe = _ltx23_default_python(root, settings)
    width, height, normalized_resolution = _normalize_ltx23_vramlab_resolution(resolution, "1280x704")

    checkpoint = _ltx23_path_from_settings(root, settings.get("checkpoint_path"), root / "models" / "ltx23" / "distilled-1.1" / "ltx-2.3-22b-distilled-1.1.safetensors")
    gemma = _ltx23_path_from_settings(root, settings.get("gemma_root"), root / "models" / "ltx23" / "text_encoder" / "lightricks_gemma_original")
    report = _ltx23_path_from_settings(root, settings.get("report_path"), root / "tools" / "vram_lab" / "ltx_vram_lab_integration_report.txt")
    deep_log = _ltx23_path_from_settings(root, settings.get("deep_log_path"), root / "tools" / "vram_lab" / "ltx_deep_lifecycle_latest.txt")

    profile = _safe_str(settings.get("vram_profile"), "24")
    if profile not in {"auto", "24", "16", "12"}:
        profile = "24"
    vram_lab = _safe_str(settings.get("vram_lab"), "safe").lower()
    if vram_lab not in {"off", "safe", "edge", "balanced", "aggressive"}:
        vram_lab = "safe"

    use_seed = int(seed if seed is not None else _safe_int(settings.get("seed"), 12345))
    cmd: List[str] = [
        python_exe,
        str(vram_cli),
        "--pipeline", "a2vid_two_stage",
        "--vram-lab", vram_lab,
        "--vram-profile", profile,
        "--checkpoint-path", checkpoint,
        "--gemma-root", gemma,
        "--prompt", _safe_str(prompt),
        "--output-path", str(out_path),
        "--height", str(int(height)),
        "--width", str(int(width)),
        "--num-frames", str(int(frame_count)),
        "--frame-rate", str(int(fps)),
        "--num-inference-steps", str(int(steps)),
        "--seed", str(int(use_seed)),
        "--shift", str(_safe_float(settings.get("scheduler_shift"), 5.0)),
        "--ltx-root", str(root),
        "--report-path", report,
        "--deep-log-interval", str(_safe_float(settings.get("deep_log_interval"), 2.0)),
        "--deep-log-max-events", str(_safe_int(settings.get("deep_log_max_events"), 4100)),
        "--deep-log-path", deep_log,
        "--audio-path", str(audio_path),
        "--audio-start-time", str(_safe_float(settings.get("audio_start_time"), 0.0)),
    ]

    if _ltx23_manual_vram_overrides_enabled(settings):
        main_hot = _safe_float(settings.get("main_hot_window_gb"), 0.0)
        if main_hot > 0:
            cmd.extend(["--main-hot-window-gb", str(main_hot)])
        stage2 = _safe_float(settings.get("stage2_block_size_limit_gb"), 0.0)
        if stage2 > 0:
            cmd.extend(["--stage2-block-size-limit-gb", str(stage2)])
        emergency_floor = _safe_float(settings.get("emergency_free_vram_floor_gb"), 0.0)
        if emergency_floor > 0:
            cmd.extend(["--emergency-free-vram-floor-gb", str(emergency_floor)])
        stage1_hotset = _safe_float(settings.get("stage1_stable_hotset_fraction"), 0.0)
        if stage1_hotset > 0:
            cmd.extend(["--stage1-stable-hotset-fraction", str(stage1_hotset)])
        stage2_hotset = _safe_float(settings.get("stage2_stable_hotset_fraction"), 0.0)
        if stage2_hotset > 0:
            cmd.extend(["--stage2-stable-hotset-fraction", str(stage2_hotset)])

    cmd.extend(["--attention-backend", "auto", "--no-boundary-echo"])

    try:
        image = Path(start_image_path).expanduser()
        if image.is_file():
            cmd.extend([
                "--i2v-image", str(image.resolve()),
                "--i2v-image-frame", "0",
                "--i2v-image-strength", str(_safe_float(settings.get("start_image_strength"), 1.0)),
                "--i2v-image-crf", "0",
            ])
    except Exception:
        pass

    spatial = _ltx23_path_from_settings(root, settings.get("spatial_upsampler_path"), root / "models" / "ltx23" / "spatial_upsampler" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
    _ltx23_append_existing(cmd, "--spatial-upsampler-path", spatial)

    cmd.extend([
        "--extra",
        "--video-cfg-guidance-scale", str(_safe_float(settings.get("video_cfg_guidance_scale"), 2.0)),
        "--video-stg-guidance-scale", str(_safe_float(settings.get("video_stg_guidance_scale"), 0.0)),
        "--video-rescale-scale", str(_safe_float(settings.get("video_rescale_scale"), 0.7)),
        "--audio-cfg-guidance-scale", str(_safe_float(settings.get("audio_cfg_guidance_scale"), 1.0)),
        "--audio-stg-guidance-scale", str(_safe_float(settings.get("audio_stg_guidance_scale"), 0.0)),
        "--audio-rescale-scale", str(_safe_float(settings.get("audio_rescale_scale"), 0.0)),
        "--a2v-guidance-scale", str(_safe_float(settings.get("a2v_guidance_scale"), 1.0)),
        "--v2a-guidance-scale", str(_safe_float(settings.get("v2a_guidance_scale"), 1.0)),
        "--video-skip-step", str(_safe_int(settings.get("video_skip_step"), 0)),
        "--audio-skip-step", str(_safe_int(settings.get("audio_skip_step"), 0)),
        "--max-batch-size", str(_safe_int(settings.get("max_batch_size"), 2)),
    ])
    custom_extra = _safe_str(settings.get("custom_extra_args"))
    if custom_extra:
        try:
            cmd.extend(shlex.split(custom_extra))
        except Exception:
            cmd.extend(custom_extra.split())
    return cmd


def _normalize_ltx_generation_backend(value: Any, root: Optional[Path] = None) -> str:
    text = _safe_str(value).lower().replace("_", "-").strip()
    r = Path(root).resolve() if root else _project_root()
    helpers = r / "helpers"
    if text in {"ltx-vramlab", "vramlab", "vram-lab", "own", "own-ltx", "framevision-vramlab"}:
        return "vramlab"
    if text in {"wan2gp", "wangp", "wan-gp", "wgp"}:
        return "wan2gp"
    try:
        env_backend = _safe_str(os.environ.get("FRAMEVISION_MUSICCLIP_LTX_BACKEND")).lower().replace("_", "-")
        if env_backend in {"ltx-vramlab", "vramlab", "vram-lab", "own"}:
            return "vramlab"
        if env_backend in {"wan2gp", "wangp", "wan-gp", "wgp"}:
            return "wan2gp"
    except Exception:
        pass
    # Default to the own FrameVision LTX workflow whenever its real bridge and raw VRAMLab CLI are present.
    # Do not silently force Wan2GP just because helpers/ltx23_cli.py exists; that file can be offline-only.
    if (helpers / "clip2ltx_cli.py").is_file() and (helpers / "ltx23_vram_lab_cli.py").is_file():
        return "vramlab"
    if (helpers / "musicclip_planner_bridge.py").is_file() and _ltx23_guess_wangp_root(r):
        return "wan2gp"
    return "vramlab"


def _ltx23_musicclip_vramlab_cli(root: Path) -> str:
    candidate = (Path(root).resolve() / "helpers" / "ltx23_vram_lab_cli.py").resolve()
    return str(candidate) if candidate.is_file() else ""


def _ltx23_wan2gp_musicclip_cli(root: Path) -> str:
    try:
        env_cli = _safe_str(os.environ.get("FRAMEVISION_LTX23_WAN2GP_CLI"))
        if env_cli and Path(env_cli).expanduser().is_file():
            return str(Path(env_cli).expanduser().resolve())
    except Exception:
        pass
    candidate = (Path(root).resolve() / "helpers" / "ltx23_wan2gp_musicclip_cli.py").resolve()
    return str(candidate) if candidate.is_file() else ""


def _find_director_shot(plan: Dict[str, Any], shot_id: str) -> Optional[Dict[str, Any]]:
    want = _safe_str(shot_id).lower()
    for shot in _as_list(plan.get("shots")):
        if isinstance(shot, dict) and _safe_str(shot.get("id")).lower() == want:
            return dict(shot)
    return None


def _safe_child_file_path(base_dir: Path, raw_value: Any, default_name: str) -> Path:
    """Return a real file path inside base_dir unless raw_value is a valid explicit file path.

    Guard against missing payload names: Path(base_dir) / "" resolves to base_dir,
    and opening that directory as JSON/log/image causes Windows Errno 13.
    """
    base = Path(base_dir).expanduser().resolve()
    default_name = _safe_stem(default_name) + Path(_safe_str(default_name)).suffix if _safe_str(default_name) else "output_file"
    raw = _safe_str(raw_value)
    if not raw:
        return base / default_name
    try:
        candidate = Path(raw).expanduser()
        # Payload names are normally simple filenames. If a full file path is supplied,
        # allow it only when it clearly points to a file, not a folder.
        if candidate.is_absolute():
            resolved = candidate.resolve()
            if resolved.exists() and resolved.is_dir():
                return base / default_name
            if _safe_str(resolved.suffix):
                return resolved
            return base / default_name
        # Reject accidental nested folder/directory values for filename fields.
        if len(candidate.parts) > 1:
            if _safe_str(candidate.suffix):
                return (base / candidate.name).resolve()
            return base / default_name
        child = (base / candidate.name).resolve()
        if child.exists() and child.is_dir():
            return base / default_name
        if not _safe_str(child.suffix) and _safe_str(Path(default_name).suffix):
            return base / default_name
        return child
    except Exception:
        return base / default_name


def _copy_existing_start_image(src: str, dst: Path) -> str:
    src_path = Path(src).expanduser().resolve()
    if not src_path.is_file():
        raise FileNotFoundError(f"Start image was not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src_path.resolve() != dst.resolve():
        shutil.copy2(str(src_path), str(dst))
    return str(dst.resolve())


def _write_ltx_test_payload(path: Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    if p.exists() and p.is_dir():
        p = p / "ltx_test_payload.json"
    elif not _safe_str(p.suffix):
        p = p.with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


DURATION_TOLERANCE_SECONDS = 0.04
LTX_GENERATION_PRE_PAD_SECONDS = 0.00
LTX_GENERATION_TAIL_PAD_SECONDS = 0.50
# LTX can under-deliver very short clips, especially when frame-rate handling differs
# between the bridge and the backend. For short shots, request extra raw material
# up front and then trim back to the planned timeline duration. Never fix this by
# slowing down the finished raw clip.
LTX_SHORT_SHOT_PAD_THRESHOLD_SECONDS = 4.00
LTX_SHORT_SHOT_MIN_REQUEST_SECONDS = 7.00
LTX_SHORT_SHOT_EXTRA_PAD_SECONDS = 3.00
LTX_SHORT_REGEN_MAX_ATTEMPTS = 5
LTX_SHORT_REGEN_SCALE_SAFETY = 1.55
LTX_SHORT_RAW_REGEN_MESSAGE = "Raw LTX output is shorter than planned duration; regenerate this shot instead of slowing it down."


def _find_media_binary(root: Path, env_name: str, binary_name: str) -> str:
    try:
        env_value = _safe_str(os.environ.get(env_name))
        if env_value and (os.path.isfile(env_value) or shutil.which(env_value)):
            return env_value
    except Exception:
        pass
    names = [binary_name]
    if os.name == "nt" and not binary_name.lower().endswith(".exe"):
        names.insert(0, binary_name + ".exe")
    try:
        for name in names:
            cand = (Path(root).resolve() / "presets" / "bin" / name).resolve()
            if cand.is_file():
                return str(cand)
    except Exception:
        pass
    for name in names:
        try:
            found = shutil.which(name)
            if found:
                return found
        except Exception:
            pass
    return binary_name


def _probe_media_duration_seconds(root: Path, media_path: Path) -> float:
    src = Path(media_path)
    if not src.is_file():
        return 0.0
    ffprobe = _find_media_binary(root, "FV_FFPROBE", "ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(src),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if proc.returncode == 0:
            return max(0.0, float(_safe_str(proc.stdout) or 0.0))
    except Exception:
        return 0.0
    return 0.0


def _ltx_generation_timing_plan(shot: Dict[str, Any], fps: int) -> Dict[str, Any]:
    target_fps = max(1, int(fps or _safe_int(shot.get("target_fps"), 24) or 24))
    start = _safe_float(shot.get("song_start"), 0.0)
    end = _safe_float(shot.get("song_end"), 0.0)
    planned_duration = 0.0
    planned_source = "unknown"
    if end > start:
        planned_duration = max(0.001, end - start)
        planned_source = "song_end-song_start"
    else:
        duration = _safe_float(shot.get("duration"), 0.0)
        if duration > 0.0:
            planned_duration = max(0.001, duration)
            planned_source = "duration"
        else:
            target_frames = _safe_int(shot.get("target_frames"), 0)
            if target_frames > 0:
                planned_duration = max(0.001, target_frames / float(target_fps))
                planned_source = "target_frames/target_fps"
    planned_duration_json = max(0.001, float(planned_duration)) if planned_duration > 0 else 0.0
    planned_frames = max(1, int(round(planned_duration_json * float(target_fps)))) if planned_duration_json > 0 else max(1, _safe_int(shot.get("target_frames"), target_fps))
    final_timeline_duration = planned_frames / float(target_fps)
    pre_pad = max(0.0, float(LTX_GENERATION_PRE_PAD_SECONDS))
    tail_pad = max(0.0, float(LTX_GENERATION_TAIL_PAD_SECONDS))
    base_requested_duration = max(0.001, planned_duration_json + tail_pad + pre_pad)
    short_pad_applied = False
    short_pad_seconds = 0.0
    short_min_request = max(0.0, float(LTX_SHORT_SHOT_MIN_REQUEST_SECONDS))
    short_extra_pad = max(0.0, float(LTX_SHORT_SHOT_EXTRA_PAD_SECONDS))
    if planned_duration_json > 0.0 and planned_duration_json <= float(LTX_SHORT_SHOT_PAD_THRESHOLD_SECONDS):
        padded_duration = max(base_requested_duration, planned_duration_json + short_extra_pad, short_min_request)
        short_pad_seconds = max(0.0, padded_duration - base_requested_duration)
        if short_pad_seconds > 0.0001:
            short_pad_applied = True
            base_requested_duration = padded_duration
    requested_generation_duration = base_requested_duration
    requested_generation_frames = max(planned_frames, int(round(requested_generation_duration * float(target_fps))))
    requested_generation_duration = requested_generation_frames / float(target_fps)
    return {
        "planned_duration": round(float(planned_duration_json or final_timeline_duration), 6),
        "planned_duration_source": planned_source,
        "planned_target_frames": int(planned_frames),
        "final_timeline_duration": round(float(final_timeline_duration), 6),
        "generation_pre_pad_seconds": round(float(pre_pad), 6),
        "generation_tail_pad_seconds": round(float(tail_pad), 6),
        "short_generation_padding_applied": bool(short_pad_applied),
        "short_generation_extra_pad_seconds": round(float(short_pad_seconds), 6),
        "short_generation_min_request_seconds": round(float(short_min_request), 6),
        "requested_generation_duration": round(float(requested_generation_duration), 6),
        "requested_generation_frames": int(requested_generation_frames),
    }


def _planned_ltx_duration_seconds(shot: Dict[str, Any], fps: int, frames: int) -> tuple:
    start = _safe_float(shot.get("song_start"), 0.0)
    end = _safe_float(shot.get("song_end"), 0.0)
    if end > start:
        return max(0.001, end - start), "song_end-song_start"
    duration = _safe_float(shot.get("duration"), 0.0)
    if duration > 0:
        return max(0.001, duration), "duration"
    try:
        f = float(frames)
        r = float(fps)
        if f > 0 and r > 0:
            return max(0.001, f / r), "target_frames/target_fps"
    except Exception:
        pass
    return 0.0, "unknown"


def _copy_video_for_sync(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Path(src).resolve() == Path(dst).resolve():
        return
    shutil.copy2(str(src), str(dst))


def _valid_corrected_clip(path: Path) -> bool:
    try:
        return bool(path.is_file() and path.stat().st_size > 1024)
    except Exception:
        return False


def _media_has_audio_stream(root: Path, media_path: Path) -> bool:
    ffprobe = _find_media_binary(root, "FV_FFPROBE", "ffprobe")
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(media_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        return bool(proc.returncode == 0 and _safe_str(proc.stdout))
    except Exception:
        return False


def _run_ffmpeg_logged(cmd: List[Any], log_path: Path, header: str, timeout: int = 600) -> Dict[str, Any]:
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as lf:
            lf.write(f"\n{header}\n")
            lf.write("Command:\n")
            lf.write(" ".join(str(x) for x in cmd) + "\n\n")
            lf.flush()
            proc = subprocess.run(
                [str(x) for x in cmd],
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
            lf.write(f"\n[sync guard] ffmpeg exit code: {proc.returncode}\n")
        return {"returncode": int(proc.returncode), "cmd": [str(x) for x in cmd]}
    except Exception as exc:
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as lf:
                lf.write(f"\n[sync guard] ffmpeg command failed: {exc}\n")
        except Exception:
            pass
        return {"returncode": -1, "cmd": [str(x) for x in cmd], "error": str(exc)}


def _trim_video_to_duration(root: Path, src: Path, dst: Path, duration: float, log_path: Path) -> Dict[str, Any]:
    """Trim extra LTX tail while preserving the raw output quality first.

    Default path is stream copy: no re-encode, no quality loss, audio kept.
    Re-encode is a last-resort fallback only.
    """
    ffmpeg = _find_media_binary(root, "FV_FFMPEG", "ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass

    has_audio = _media_has_audio_stream(root, src)
    stream_cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-i", str(src),
        "-t", f"{float(duration):.6f}",
        "-map", "0",
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ]
    stream_result = _run_ffmpeg_logged(stream_cmd, log_path, "[sync guard] Trimming raw LTX output with stream copy (quality-preserving)")
    if int(stream_result.get("returncode", -1)) == 0 and _valid_corrected_clip(dst):
        return {
            "ok": True,
            "cmd": stream_cmd,
            "returncode": int(stream_result.get("returncode", 0)),
            "attempts": [stream_result],
            "correction_quality_mode": "stream_copy",
            "audio_preserved": bool(has_audio and _media_has_audio_stream(root, dst)),
            "video_reencoded": False,
            "ffmpeg_command_used": [str(x) for x in stream_cmd],
            "fallback_used": False,
            "fallback_reason": "",
        }

    fallback_reason = "stream-copy trim failed or produced no valid output"
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass

    # High-quality fallback. This can re-encode video, but uses a visually-safe CRF
    # and keeps audio instead of silently dropping it.
    fallback_cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-i", str(src),
        "-t", f"{float(duration):.6f}",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(dst),
    ]
    fallback_result = _run_ffmpeg_logged(fallback_cmd, log_path, "[sync guard] Stream-copy trim fallback: high-quality re-encode with audio preserved when possible")
    if int(fallback_result.get("returncode", -1)) != 0 or not _valid_corrected_clip(dst):
        # Some containers/codecs dislike copied audio after a trim; retry with AAC.
        try:
            if dst.exists():
                dst.unlink()
        except Exception:
            pass
        fallback_reason += "; audio-copy fallback failed, retried with AAC audio"
        fallback_cmd_aac = list(fallback_cmd)
        try:
            ai = fallback_cmd_aac.index("copy")
            # The first "copy" here should be the audio codec value because video uses libx264.
            fallback_cmd_aac[ai] = "aac"
            fallback_cmd_aac[ai:ai+1] = ["aac", "-b:a", "192k"]
        except Exception:
            fallback_cmd_aac = [
                ffmpeg, "-y", "-hide_banner", "-i", str(src), "-t", f"{float(duration):.6f}",
                "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)
            ]
        fallback_result_aac = _run_ffmpeg_logged(fallback_cmd_aac, log_path, "[sync guard] High-quality re-encode retry with AAC audio")
        attempts = [stream_result, fallback_result, fallback_result_aac]
        ok = bool(int(fallback_result_aac.get("returncode", -1)) == 0 and _valid_corrected_clip(dst))
        return {
            "ok": ok,
            "cmd": fallback_cmd_aac,
            "returncode": int(fallback_result_aac.get("returncode", -1)),
            "attempts": attempts,
            "correction_quality_mode": "high_quality_reencode" if ok else "failed",
            "audio_preserved": bool(has_audio and _media_has_audio_stream(root, dst)),
            "video_reencoded": bool(ok),
            "ffmpeg_command_used": [str(x) for x in fallback_cmd_aac],
            "fallback_used": True,
            "fallback_reason": fallback_reason,
        }

    attempts = [stream_result, fallback_result]
    ok = bool(_valid_corrected_clip(dst))
    return {
        "ok": ok,
        "cmd": fallback_cmd,
        "returncode": int(fallback_result.get("returncode", -1)),
        "attempts": attempts,
        "correction_quality_mode": "high_quality_reencode" if ok else "failed",
        "audio_preserved": bool(has_audio and _media_has_audio_stream(root, dst)),
        "video_reencoded": bool(ok),
        "ffmpeg_command_used": [str(x) for x in fallback_cmd],
        "fallback_used": True,
        "fallback_reason": fallback_reason,
    }


def _pad_short_ltx_clip_to_duration(
    root: Path,
    src: Path,
    dst: Path,
    planned_duration: float,
    planned_frames: int,
    fps: int,
    log_path: Path,
    *,
    header: str = "[sync guard] Padding short LTX output by holding the last frame",
) -> Dict[str, Any]:
    """Extend a too-short LTX clip without changing playback speed.

    This holds the final video frame and pads audio with silence when audio
    exists. It never uses setpts/atempo to slow the clip down.
    """
    actual = _probe_media_duration_seconds(root, src)
    planned = max(0.001, float(planned_duration or 0.0))
    fps_i = max(1, int(fps or 24))
    frames_i = max(1, int(planned_frames or round(planned * float(fps_i))))
    pad_seconds = max(0.0, planned - max(0.0, actual))
    if actual <= 0.0 or planned <= 0.0 or pad_seconds <= float(DURATION_TOLERANCE_SECONDS):
        return {
            "ok": False,
            "actual_duration": round(float(actual), 6) if actual else 0.0,
            "planned_duration": round(float(planned), 6),
            "pad_seconds": round(float(pad_seconds), 6),
            "message": "Invalid or unnecessary last-frame padding request.",
            "raw_output_shorter_than_planned": bool(actual > 0.0 and actual < planned - float(DURATION_TOLERANCE_SECONDS)),
            "short_raw_padded_to_planned": False,
            "retime_skipped_to_avoid_slow_motion": True,
        }
    ffmpeg = _find_media_binary(root, "FV_FFMPEG", "ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    has_audio = _media_has_audio_stream(root, src)
    if has_audio:
        filter_complex = (
            f"[0:v]fps={fps_i},tpad=stop_mode=clone:stop_duration={pad_seconds:.6f},"
            f"trim=duration={planned:.6f},setpts=PTS-STARTPTS,format=yuv420p[v];"
            f"[0:a]apad,atrim=duration={planned:.6f},asetpts=PTS-STARTPTS[a]"
        )
        cmd = [ffmpeg, "-y", "-hide_banner", "-i", str(src), "-filter_complex", filter_complex,
               "-map", "[v]", "-map", "[a]", "-frames:v", str(frames_i), "-r", str(fps_i),
               "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(dst)]
    else:
        vf = (f"fps={fps_i},tpad=stop_mode=clone:stop_duration={pad_seconds:.6f},"
              f"trim=duration={planned:.6f},setpts=PTS-STARTPTS,format=yuv420p")
        cmd = [ffmpeg, "-y", "-hide_banner", "-i", str(src), "-map", "0:v:0", "-vf", vf,
               "-frames:v", str(frames_i), "-r", str(fps_i), "-an", "-c:v", "libx264",
               "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    result = _run_ffmpeg_logged(cmd, log_path, header, timeout=1800)
    ok = bool(int(result.get("returncode", -1)) == 0 and _valid_corrected_clip(dst))
    out_duration = _probe_media_duration_seconds(root, dst) if ok else 0.0
    duration_ok = bool(ok and out_duration >= planned - max(float(DURATION_TOLERANCE_SECONDS), 1.0 / float(fps_i)))
    return {
        "ok": bool(duration_ok),
        "cmd": [str(x) for x in cmd],
        "returncode": int(result.get("returncode", -1)),
        "actual_duration": round(float(actual), 6),
        "planned_duration": round(float(planned), 6),
        "padded_duration": round(float(out_duration), 6) if out_duration else 0.0,
        "retimed_duration": round(float(out_duration), 6) if out_duration else 0.0,
        "pad_seconds": round(float(pad_seconds), 6),
        "speed_factor": 1.0,
        "natural_duration_factor": round(float(actual / planned), 8) if planned > 0 else 0.0,
        "correction_quality_mode": "freeze_last_frame_pad_no_slowdown",
        "audio_preserved": bool(has_audio and _media_has_audio_stream(root, dst)),
        "video_reencoded": True,
        "ffmpeg_command_used": [str(x) for x in cmd],
        "fallback_used": False,
        "fallback_reason": "",
        "raw_output_shorter_than_planned": True,
        "short_raw_padded_to_planned": bool(duration_ok),
        "freeze_last_frame_padding_applied": bool(duration_ok),
        "retime_skipped_to_avoid_slow_motion": True,
        "recommended_action": "ok - padded final frame hold" if duration_ok else "regenerate shot",
        "message": "Short LTX output was padded by holding the last frame; no slow-motion retime was used." if duration_ok else "Last-frame padding failed; regenerate shot.",
    }


def _retime_video_to_duration(root: Path, src: Path, dst: Path, planned_duration: float, planned_frames: int, fps: int, log_path: Path) -> Dict[str, Any]:
    ffmpeg = _find_media_binary(root, "FV_FFMPEG", "ffmpeg")
    actual = _probe_media_duration_seconds(root, src)
    has_audio = _media_has_audio_stream(root, src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    if actual <= 0.0 or planned_duration <= 0.0:
        return {
            "ok": False,
            "returncode": -1,
            "error": "Invalid duration for retime.",
            "actual_duration": actual,
            "correction_quality_mode": "failed",
            "audio_preserved": False,
            "video_reencoded": False,
            "ffmpeg_command_used": [],
            "fallback_used": False,
            "fallback_reason": "invalid duration for retime",
        }
    if actual < planned_duration - float(DURATION_TOLERANCE_SECONDS):
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as lf:
                lf.write("\n[sync guard] Short raw LTX output was not stretched. Regenerate the shot instead.\n")
                lf.write(f"actual={actual:.6f} planned={planned_duration:.6f}\n")
        except Exception:
            pass
        return {
            "ok": False,
            "returncode": -1,
            "error": LTX_SHORT_RAW_REGEN_MESSAGE,
            "actual_duration": round(float(actual), 6),
            "speed_factor": round(float(actual / max(0.001, planned_duration)), 8),
            "correction_quality_mode": "short_raw_not_retimed",
            "audio_preserved": bool(has_audio),
            "video_reencoded": False,
            "ffmpeg_command_used": [],
            "fallback_used": False,
            "fallback_reason": LTX_SHORT_RAW_REGEN_MESSAGE,
            "raw_output_shorter_than_planned": True,
            "retime_skipped_to_avoid_slow_motion": True,
            "recommended_action": "regenerate shot",
        }
    speed_factor = actual / planned_duration
    vf = f"setpts=(PTS-STARTPTS)/{speed_factor:.10f},fps={int(fps)}"
    if has_audio and 0.5 <= speed_factor <= 2.0:
        cmd = [
            ffmpeg, "-y", "-hide_banner",
            "-i", str(src),
            "-filter_complex", f"[0:v]{vf}[v];[0:a]atempo={speed_factor:.10f}[a]",
            "-map", "[v]", "-map", "[a]",
            "-frames:v", str(max(1, int(planned_frames))),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-movflags", "+faststart",
            str(dst),
        ]
        audio_mode = "audio_retimed_with_atempo"
    else:
        cmd = [
            ffmpeg, "-y", "-hide_banner",
            "-i", str(src),
            "-vf", vf,
            "-frames:v", str(max(1, int(planned_frames))),
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(dst),
        ]
        audio_mode = "audio_copied_unretimed" if has_audio else "no_audio_source"
    result = _run_ffmpeg_logged(cmd, log_path, f"[sync guard] Retiming short raw LTX output to planned timeline slot ({audio_mode})")
    ok = bool(int(result.get("returncode", -1)) == 0 and _valid_corrected_clip(dst))
    return {
        "ok": ok,
        "cmd": cmd,
        "returncode": int(result.get("returncode", -1)),
        "actual_duration": round(float(actual), 6),
        "speed_factor": round(float(speed_factor), 8),
        "correction_quality_mode": "retime_reencode" if ok else "failed",
        "audio_preserved": bool(has_audio and _media_has_audio_stream(root, dst)),
        "audio_retime_mode": audio_mode,
        "video_reencoded": bool(ok),
        "ffmpeg_command_used": [str(x) for x in cmd],
        "fallback_used": False,
        "fallback_reason": "",
    }


def _create_ltx_sync_clip(
    *,
    root: Path,
    raw_clip_path: Path,
    sync_clip_path: Path,
    shot: Dict[str, Any],
    fps: int,
    frames: int,
    log_path: Path,
    duration_report_path: Path,
) -> Dict[str, Any]:
    timing = _ltx_generation_timing_plan(shot, fps)
    planned_duration = _safe_float(timing.get("planned_duration"), 0.0)
    planned_source = _safe_str(timing.get("planned_duration_source"))
    planned_frames = _safe_int(timing.get("planned_target_frames"), max(1, int(round(planned_duration * max(1, fps)))))
    actual_duration = _probe_media_duration_seconds(root, raw_clip_path)
    delta = actual_duration - planned_duration if actual_duration > 0 and planned_duration > 0 else 0.0
    tolerance = float(DURATION_TOLERANCE_SECONDS)
    warning = ""
    applied = False
    action = "none"
    trim_result: Dict[str, Any] = {}
    retime_result: Dict[str, Any] = {}

    if planned_duration <= 0:
        warning = "Planned duration could not be calculated; sync clip was copied without correction."
        action = "failed"
        _copy_video_for_sync(raw_clip_path, sync_clip_path)
    elif actual_duration <= 0:
        warning = "Raw clip duration could not be probed; sync clip was copied without correction."
        action = "failed"
        _copy_video_for_sync(raw_clip_path, sync_clip_path)
    elif actual_duration > planned_duration + tolerance:
        action = "trim_extra_tail"
        applied = True
        trim_result = _trim_video_to_duration(root, raw_clip_path, sync_clip_path, planned_duration, log_path)
        if not bool(trim_result.get("ok")):
            applied = False
            warning = "ffmpeg trim failed; sync clip was copied from raw clip instead."
            action = "failed"
            _copy_video_for_sync(raw_clip_path, sync_clip_path)
    elif actual_duration < planned_duration - tolerance:
        action = "raw_shorter_than_planned"
        applied = False
        warning = LTX_SHORT_RAW_REGEN_MESSAGE
        _copy_video_for_sync(raw_clip_path, sync_clip_path)
        retime_result.update({
            "ok": False,
            "correction_quality_mode": "short_raw_copied_no_retime",
            "audio_preserved": bool(_media_has_audio_stream(root, raw_clip_path) == _media_has_audio_stream(root, sync_clip_path)),
            "video_reencoded": False,
            "ffmpeg_command_used": [],
            "fallback_used": True,
            "fallback_reason": warning,
            "raw_output_shorter_than_planned": True,
            "short_raw_padded_to_planned": False,
            "freeze_last_frame_padding_applied": False,
            "retime_skipped_to_avoid_slow_motion": True,
            "recommended_action": "regenerate shot",
        })
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as lf:
                lf.write("\n[sync guard] Short raw LTX output detected. No freeze-hold and no slow-motion retime was applied.\n")
                lf.write(f"actual={actual_duration:.6f} planned={planned_duration:.6f} delta={delta:.6f} action={action}\n")
        except Exception:
            pass
    else:
        action = "copy_exact"
        _copy_video_for_sync(raw_clip_path, sync_clip_path)

    sync_duration = _probe_media_duration_seconds(root, sync_clip_path)
    correction_result = trim_result if trim_result else (retime_result if retime_result else {})
    if action == "copy_exact":
        correction_quality_mode = "copy_exact"
        audio_preserved = bool(_media_has_audio_stream(root, raw_clip_path) == _media_has_audio_stream(root, sync_clip_path))
        video_reencoded = False
        ffmpeg_command_used: List[str] = []
        fallback_used = False
        fallback_reason = ""
    elif action == "failed":
        correction_quality_mode = _safe_str(correction_result.get("correction_quality_mode"), "failed") or "failed"
        audio_preserved = bool(_media_has_audio_stream(root, sync_clip_path))
        video_reencoded = bool(correction_result.get("video_reencoded", False))
        ffmpeg_command_used = [str(x) for x in _as_list(correction_result.get("ffmpeg_command_used"))]
        fallback_used = bool(correction_result.get("fallback_used", False))
        fallback_reason = _safe_str(correction_result.get("fallback_reason") or warning)
    elif action == "raw_shorter_than_planned":
        correction_quality_mode = _safe_str(correction_result.get("correction_quality_mode"), "short_raw_copied_no_retime") or "short_raw_copied_no_retime"
        audio_preserved = bool(correction_result.get("audio_preserved", _media_has_audio_stream(root, sync_clip_path)))
        video_reencoded = False
        ffmpeg_command_used = []
        fallback_used = False
        fallback_reason = _safe_str(correction_result.get("fallback_reason") or warning)
    else:
        correction_quality_mode = _safe_str(correction_result.get("correction_quality_mode"), "") or ("stream_copy" if action == "trim_extra_tail" else "retime_reencode")
        audio_preserved = bool(correction_result.get("audio_preserved", _media_has_audio_stream(root, sync_clip_path)))
        video_reencoded = bool(correction_result.get("video_reencoded", action == "retime_to_planned"))
        ffmpeg_command_used = [str(x) for x in _as_list(correction_result.get("ffmpeg_command_used") or correction_result.get("cmd"))]
        fallback_used = bool(correction_result.get("fallback_used", False))
        fallback_reason = _safe_str(correction_result.get("fallback_reason"))

    report = {
        **timing,
        "planned_duration": round(float(planned_duration), 6),
        "planned_duration_source": planned_source,
        "actual_raw_duration": round(float(actual_duration), 6) if actual_duration else 0.0,
        "raw_output_duration": round(float(actual_duration), 6) if actual_duration else 0.0,
        "corrected_output_duration": round(float(sync_duration), 6) if sync_duration else 0.0,
        "duration_delta": round(float(delta), 6),
        "duration_tolerance": tolerance,
        "sync_correction_applied": bool(applied),
        "sync_action": action,
        "duration_correction_method": action,
        "correction_quality_mode": correction_quality_mode,
        "audio_preserved": bool(audio_preserved),
        "video_reencoded": bool(video_reencoded),
        "ffmpeg_command_used": ffmpeg_command_used,
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "sync_clip_path": str(sync_clip_path),
        "sync_clip_duration": round(float(sync_duration), 6) if sync_duration else 0.0,
        "sync_warning": warning,
        "raw_output_shorter_than_planned": bool(action in {"raw_shorter_than_planned", "pad_last_frame_to_planned"}),
        "short_raw_padded_to_planned": bool(correction_result.get("short_raw_padded_to_planned", action == "pad_last_frame_to_planned")),
        "freeze_last_frame_padding_applied": bool(correction_result.get("freeze_last_frame_padding_applied", action == "pad_last_frame_to_planned")),
        "short_raw_padding_seconds": _safe_float(correction_result.get("pad_seconds"), 0.0),
        "retime_skipped_to_avoid_slow_motion": bool(correction_result.get("retime_skipped_to_avoid_slow_motion", action in {"raw_shorter_than_planned", "pad_last_frame_to_planned"})),
        "recommended_action": _safe_str(correction_result.get("recommended_action")) or ("regenerate shot" if action == "raw_shorter_than_planned" else ""),
        "final_master_audio_rule": "Split WAV chunks are only for LTX lipsync/audio guidance. Final assembly should concatenate sync-safe video clips and use the original full song/final_master_audio as the master audio, not raw LTX clip audio.",
        "trim_result": trim_result,
        "retime_result": retime_result,
    }
    try:
        _write_ltx_test_payload(duration_report_path, report)
    except Exception:
        pass
    return report


def run_single_ltx_shot_test(payload: dict) -> dict:
    """Run one private Music Clip Creator -> LTX image-to-video test shot.

    This function is intentionally isolated from the normal Music Clip Creator renderer,
    queue, settings, and Planner pipeline. It reads an existing
    musicclip_ltx_director_plan.json, prepares one start image, then calls
    a backend-specific Music Clip LTX CLI for exactly one selected shot.
    """
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "LTX test payload was not a dictionary."}

        progress_callback = payload.get("progress_callback")

        def _emit(message: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(str(message or ""))
                except Exception:
                    pass

        _emit("Loading LTX director plan...")
        root_raw = _safe_str(payload.get("root_dir"))
        root = Path(root_raw).resolve() if root_raw else _project_root()
        plan_path = Path(_safe_str(payload.get("ltx_director_plan_path"))).expanduser().resolve()
        if not plan_path.is_file():
            return {"ok": False, "message": f"LTX director plan was not found: {plan_path}"}
        director_plan = _read_json_file(str(plan_path))
        safety = _enforce_ltx_start_end_duration_contract(
            director_plan,
            plan_path=plan_path,
            root=root,
            enabled=_payload_ltx_edge_duration_safety_enabled(payload, director_plan),
            refresh_audio_chunks=True,
        )
        if bool(safety.get("changed")):
            try:
                existing_warnings = _as_list(director_plan.get("warnings"))
                merged = existing_warnings + [w for w in _as_list(safety.get("warnings")) if w]
                if merged:
                    director_plan["warnings"] = merged[:80]
                _write_json_file(plan_path, director_plan)
                _emit("Updated LTX plan timing: first clip is at least 3 seconds and final clip targets 5 seconds.")
            except Exception as exc:
                _emit(f"Warning: could not save LTX duration safety update: {exc}")
        character_reference = _character_reference_from_sources(payload, director_plan)
        shot_id = _safe_str(payload.get("shot_id"))
        if not shot_id:
            return {"ok": False, "message": "No LTX shot was selected."}
        shot = _find_director_shot(director_plan, shot_id)
        if not shot:
            return {"ok": False, "message": f"Selected shot was not found in the director plan: {shot_id}"}

        brief = _normalize_creative_brief(director_plan.get("creative_brief"))
        _chars, _groups, shot = _identity_context_from_plan(shot, director_plan, payload, progress_callback)
        shot = _protect_lipsync_confidence(shot, song_duration=max([_safe_float(x.get("song_end"), 0.0) for x in _director_plan_shots(director_plan)] + [0.0]))
        shot = _director_compile_shot_prompts(shot, brief)
        # Review/recreate may pass an edited prompt without rewriting the source
        # director plan. Apply that override after identity/lipsync processing so
        # those helpers cannot silently restore the original prompt.
        review_clip_prompt_override = _safe_str(
            payload.get("clip_prompt_override")
            or payload.get("video_prompt_override")
            or payload.get("director_timestamped_video_prompt_override")
            or payload.get("timestamped_video_prompt_override")
            or payload.get("clip_prompt")
        )
        if review_clip_prompt_override:
            shot = dict(shot)
            shot["director_timestamped_video_prompt"] = review_clip_prompt_override
            shot["director_video_prompt"] = review_clip_prompt_override
            shot["video_prompt"] = review_clip_prompt_override
            shot["review_clip_prompt_override_applied"] = True
        shot_id = _safe_str(shot.get("id")) or shot_id
        _emit(f"Selected {shot_id}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir_raw = _safe_str(payload.get("output_dir"))
        if out_dir_raw:
            test_dir = Path(out_dir_raw).expanduser().resolve()
            test_dir.mkdir(parents=True, exist_ok=True)
        else:
            test_dir = _make_ltx_test_dir(plan_path, shot_id)
        stem = _safe_stem(shot_id)
        start_path = _safe_child_file_path(test_dir, payload.get("start_image_name"), f"{stem}_start.png")
        clip_path = _safe_child_file_path(test_dir, payload.get("raw_clip_name"), f"{stem}_ltx.mp4")
        log_path = _safe_child_file_path(test_dir, payload.get("log_name"), f"{stem}_ltx23.log.txt")
        test_payload_path = _safe_child_file_path(test_dir, payload.get("test_payload_name"), f"{stem}_test_payload.json")
        sync_clip_path = _safe_child_file_path(test_dir, payload.get("sync_clip_name"), f"{stem}_ltx_sync.mp4")
        duration_report_path = _safe_child_file_path(test_dir, payload.get("duration_report_name"), f"{stem}_duration_report.json")

        image_mode = _safe_str(payload.get("image_mode"), "existing").lower() or "existing"
        if image_mode in {"use existing start image", "existing_start_image"}:
            image_mode = "existing"
        if image_mode == "existing":
            existing = _safe_str(payload.get("existing_start_image_path"))
            if not existing:
                return {"ok": False, "message": "Use existing start image is selected, but no start image path was provided."}
            _emit("Using existing start image...")
            start_image_path = _copy_existing_start_image(existing, start_path)
            image_generation_result: Dict[str, Any] = {}
        elif image_mode in {"flux_klein_9b", "z_image", "hidream"}:
            _emit(f"Preparing start image with {image_mode.replace('_', ' ')}...")
            start_payload: Dict[str, Any] = {
                "root_dir": str(root),
                "ltx_director_plan_path": str(plan_path),
                "shot_id": shot_id,
                "image_model": image_mode,
                "seed": payload.get("seed"),
                "resolution": payload.get("resolution") or shot.get("resolution") or "1280x720",
                "output_dir": str(test_dir),
                "progress_callback": progress_callback,
                "character_reference": payload.get("character_reference") or director_plan.get("character_reference") or shot.get("character_reference") or {},
            }
            gen_result = generate_ltx_start_image_for_shot(start_payload)
            if not isinstance(gen_result, dict) or not bool(gen_result.get("ok")):
                msg = "Start image generation failed before the LTX test could start."
                if isinstance(gen_result, dict):
                    msg = _safe_str(gen_result.get("message"), msg) or msg
                return {"ok": False, "message": msg, "output_dir": str(test_dir)}
            start_image_path = _safe_str(gen_result.get("start_image_path"))
            if not start_image_path or not os.path.isfile(start_image_path):
                return {"ok": False, "message": "Start image generation reported success, but no start image file was found.", "output_dir": str(test_dir)}
            image_generation_result = dict(gen_result)
            _emit(f"Using generated start image: {start_image_path}")
        else:
            return {"ok": False, "message": f"Unknown start image mode: {image_mode}"}

        raw_ltx_prompt, final_ltx_prompt_source, prompt_is_timestamped = _select_ltx_video_prompt_source(shot, payload)
        prompt, ltx_prompt_removed_phrases = _sanitize_final_ltx_prompt_for_model(
            raw_ltx_prompt,
            brief=brief,
            shot=shot,
            prompt_is_timestamped=prompt_is_timestamped,
        )
        negative = _safe_str(shot.get("director_negative_prompt"))
        if not prompt:
            return {"ok": False, "message": f"{shot_id} has no director timestamped video prompt."}

        fps = max(1, _safe_int(shot.get("target_fps"), _safe_int(director_plan.get("fps"), 24)))
        backend = _normalize_ltx_generation_backend(payload.get("ltx_backend") or payload.get("ltx_generation_backend"), root)
        max_generation_frames = 0
        if backend == "vramlab":
            cap_info = _cap_ltx_shot_timing_to_generation_limit(shot, fps, reason="musicclip_ltx_vramlab")
            max_generation_frames = _safe_int(cap_info.get("max_frames"), 241)
            if bool(cap_info.get("changed")):
                _emit(f"LTX-VRAMLab cap active: clamped {shot_id} plan to {cap_info.get('max_seconds'):.2f}s / {max_generation_frames} frames before building the command.")
        timing_plan = _ltx_generation_timing_plan(shot, fps)
        planned_frames = _safe_int(timing_plan.get("planned_target_frames"), max(1, _safe_int(shot.get("target_frames"), int(round(max(0.1, _safe_float(shot.get("duration"), 5.0)) * fps)))))
        frames = max(1, _safe_int(timing_plan.get("requested_generation_frames"), planned_frames))
        if backend == "vramlab" and max_generation_frames > 0:
            frames = min(int(frames), int(max_generation_frames))
        steps = max(1, _safe_int(payload.get("steps"), 8))
        resolution = _safe_str(payload.get("resolution") or shot.get("resolution") or director_plan.get("resolution"), "1280x704") or "1280x704"
        seed_raw = payload.get("seed", None)
        seed = None
        if seed_raw not in (None, ""):
            seed = _safe_int(seed_raw, -1)

        audio_path = _safe_str(shot.get("audio_clip_path"))
        audio_ok = bool(audio_path and os.path.isfile(audio_path))
        needs_lipsync = _safe_bool(shot.get("needs_lipsync"), False)
        allow_no_audio = _safe_bool(payload.get("allow_no_audio"), False)
        if not audio_ok:
            if needs_lipsync and not allow_no_audio:
                return {"ok": False, "message": f"{shot_id} is a lipsync shot, but its WAV audio chunk was not found: {audio_path or '[empty]'}"}
            _emit(f"Warning: no WAV audio guide found for {shot_id}; running without audio guide.")
            audio_path = ""

        bridge_cfg = _ltx23_bridge_config(root)
        wangp_root = _safe_str(payload.get("wangp_root") or bridge_cfg.get("wangp_root"))
        wgp_py = _safe_str(payload.get("wgp_py") or bridge_cfg.get("wgp_py"))
        if backend == "vramlab":
            cli = Path(_ltx23_musicclip_vramlab_cli(root)).resolve()
            if not cli.is_file():
                return {"ok": False, "message": f"Missing own LTX VRAMLab CLI: {cli}"}
        else:
            cli_text = _ltx23_wan2gp_musicclip_cli(root)
            cli = Path(cli_text).resolve() if cli_text else (root / "helpers" / "ltx23_wan2gp_musicclip_cli.py").resolve()
            if not cli.is_file():
                return {"ok": False, "message": f"Missing dedicated Wan2GP Music Clip CLI: {cli}. Copy/rename your offline Wan2GP CLI to this filename if you want this backend visible."}
            if not wangp_root or not os.path.isdir(wangp_root):
                return {
                    "ok": False,
                    "message": "WanGP root for LTX 2.3 was not found. Expected FRAMEVISION_LTX23_WANGP_ROOT, C:\\WanGP\\Wan2GP, or a nearby Wan2GP folder.",
                }

        lora_file = _safe_str(payload.get("ltx_lora_file"))
        lora_json = _safe_str(payload.get("ltx_lora_json"))
        lora_multiplier = float(_safe_float(payload.get("ltx_lora_multiplier"), 1.0) or 1.0)
        if backend != "vramlab" and lora_file and not os.path.isfile(lora_file):
            return {"ok": False, "message": f"LTX LoRA file was not found: {lora_file}"}
        if backend != "vramlab" and lora_json and not os.path.isfile(lora_json):
            return {"ok": False, "message": f"LTX LoRA JSON was not found: {lora_json}"}

        def _build_ltx_args(out_path: Path, frame_count: int, audio_override_path: str = "") -> List[str]:
            if backend == "vramlab":
                audio_for_args = _safe_str(audio_override_path) or audio_path
                if not audio_for_args:
                    raise ValueError("LTX-VRAMLab two-stage workflow needs an audio guide file.")
                return _ltx23_build_vramlab_direct_args(
                    root=root,
                    prompt=prompt,
                    start_image_path=start_image_path,
                    out_path=out_path,
                    fps=int(fps),
                    frame_count=int(frame_count),
                    steps=int(steps),
                    resolution=str(resolution),
                    audio_path=str(audio_for_args),
                    seed=seed,
                    lora_file=lora_file,
                )

            built = [
                str(sys.executable), str(cli), "generate",
                "--wangp-root", str(wangp_root),
                "--prompt", prompt,
                "--negative", negative,
                "--image", str(start_image_path),
                "--output", str(out_path),
                "--fps", str(int(fps)),
                "--frames", str(int(frame_count)),
                "--steps", str(int(steps)),
                "--resolution", str(resolution),
            ]
            audio_for_args = _safe_str(audio_override_path) or audio_path
            if audio_for_args:
                built += ["--audio", str(audio_for_args)]
            if seed is not None:
                built += ["--seed", str(int(seed))]
            if wgp_py and os.path.isfile(wgp_py):
                built += ["--wgp-py", wgp_py]
            if lora_file:
                built += ["--lora-file", lora_file]
            if lora_json:
                built += ["--lora-json", lora_json]
            if lora_file or lora_json:
                built += ["--lora-multiplier", str(float(lora_multiplier))]
            return built

        planned_duration_for_retry = _safe_float(timing_plan.get("planned_duration"), 0.0)
        planned_ok_threshold = max(0.0, planned_duration_for_retry - float(DURATION_TOLERANCE_SECONDS))
        max_attempts = max(1, int(LTX_SHORT_REGEN_MAX_ATTEMPTS))
        attempt_records: List[Dict[str, Any]] = []
        rc = -1
        args = _build_ltx_args(clip_path, frames)
        final_frames_used = int(frames)
        final_clip_path = Path(clip_path)
        generation_duration_ok = False
        debug_payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "shot_id": shot_id,
            "selected_shot": shot,
            "image_mode": image_mode,
            "start_image_path": str(start_image_path),
            "start_image_generation": image_generation_result,
            "audio_clip_path": audio_path,
            "audio_guide_role": "LTX lipsync/audio guidance only; not final master audio.",
            "final_master_audio_rule": "Final music-video assembly should use the original full song/final_master_audio as master audio, not raw LTX clip audio.",
            "needs_lipsync": needs_lipsync,
            "target_fps": fps,
            "target_frames": planned_frames,
            "planned_target_frames": planned_frames,
            "generation_pre_pad_seconds": timing_plan.get("generation_pre_pad_seconds"),
            "generation_tail_pad_seconds": timing_plan.get("generation_tail_pad_seconds"),
            "short_generation_padding_applied": timing_plan.get("short_generation_padding_applied"),
            "short_generation_extra_pad_seconds": timing_plan.get("short_generation_extra_pad_seconds"),
            "short_generation_min_request_seconds": timing_plan.get("short_generation_min_request_seconds"),
            "planned_duration": timing_plan.get("planned_duration"),
            "final_timeline_duration": timing_plan.get("final_timeline_duration"),
            "requested_generation_duration": timing_plan.get("requested_generation_duration"),
            "requested_generation_frames": frames,
            "resolution": resolution,
            "ltx_generation_backend": backend,
            "steps": steps,
            "seed": seed,
            "ltx_lora_file": lora_file,
            "ltx_lora_json": lora_json,
            "ltx_lora_multiplier": lora_multiplier,
            "output_clip_path": str(clip_path),
            "final_ltx_prompt_source": final_ltx_prompt_source,
            "raw_selected_ltx_prompt": raw_ltx_prompt,
            "final_ltx_prompt_word_count": _director_word_count(prompt),
            "ltx_prompt_removed_duplicate_phrases": ltx_prompt_removed_phrases,
            "ltx_command_args": args,
            "ltx_command_summary": {
                "prompt": prompt,
                "negative": negative,
                "image": str(start_image_path),
                "audio": audio_path,
                "fps": fps,
                "planned_frames": planned_frames,
                "frames": frames,
                "requested_generation_frames": frames,
                "output": str(clip_path),
            },
            "short_output_retry_enabled": True,
            "short_output_retry_max_attempts": max_attempts,
        }
        _write_ltx_test_payload(test_payload_path, debug_payload)

        current_frames = int(frames)
        last_short_duration = 0.0
        for attempt_index in range(1, max_attempts + 1):
            attempt_clip_path = Path(clip_path) if attempt_index == 1 else _safe_child_file_path(test_dir, f"{stem}_ltx_retry{attempt_index}.mp4", f"{stem}_ltx_retry{attempt_index}.mp4")
            try:
                if attempt_clip_path.exists():
                    attempt_clip_path.unlink()
            except Exception:
                pass
            attempt_audio_path = audio_path
            audio_sync_info: Dict[str, Any] = {}
            desired_attempt_audio_duration = max(float(current_frames) / float(max(1, fps)), planned_duration_for_retry)
            if audio_path:
                if attempt_index > 1:
                    attempt_audio_path = _make_ltx_retry_audio_guide(
                        root=root,
                        plan_path=plan_path,
                        director_plan=director_plan,
                        shot=shot,
                        out_dir=test_dir,
                        stem=stem,
                        attempt_index=attempt_index,
                        desired_generation_duration=max(desired_attempt_audio_duration, planned_duration_for_retry + float(LTX_SHORT_SHOT_EXTRA_PAD_SECONDS)),
                        fallback_audio_path=audio_path,
                    )
                # Attempt 1 also needs the audio guide to cover the extra raw
                # generation frames.  Do not shorten the LTX clip to the WAV;
                # pad/re-cut the WAV so later sync trimming still has headroom.
                attempt_audio_path, audio_sync_info = _ensure_ltx_audio_guide_covers_generation_duration(
                    root=root,
                    director_plan=director_plan,
                    shot=shot,
                    out_dir=test_dir,
                    stem=stem,
                    attempt_index=attempt_index,
                    desired_generation_duration=desired_attempt_audio_duration,
                    fallback_audio_path=attempt_audio_path,
                )
            attempt_args = _build_ltx_args(attempt_clip_path, current_frames, attempt_audio_path)
            args = attempt_args
            final_frames_used = int(current_frames)
            _emit("Running LTX image-to-video..." if attempt_index == 1 else f"Retrying LTX image-to-video for {shot_id}: attempt {attempt_index}/{max_attempts} with {current_frames} frames and a longer audio guide...")
            log_mode = "w" if attempt_index == 1 else "a"
            with log_path.open(log_mode, encoding="utf-8", errors="replace") as lf:
                if attempt_index == 1:
                    lf.write("[musicclip bridge] single LTX shot test\n")
                lf.write(f"\n[LTX generation attempt {attempt_index}/{max_attempts}]\n")
                lf.write("Command:\n")
                lf.write(" ".join([str(x) for x in attempt_args]) + "\n\n")
                lf.write(f"Planned frames: {planned_frames}\n")
                lf.write(f"Requested generation frames: {current_frames}\n")
                lf.write(f"Attempt audio guide: {attempt_audio_path or 'disabled'}\n")
                if audio_sync_info:
                    lf.write(f"Audio guide original duration: {_safe_float(audio_sync_info.get('original_audio_guide_duration'), 0.0):.6f}s\n")
                    lf.write(f"Audio guide desired generation duration: {_safe_float(audio_sync_info.get('desired_generation_duration'), 0.0):.6f}s\n")
                    lf.write(f"Audio guide padded for requested frames: {bool(audio_sync_info.get('padded'))}\n")
                    lf.write(f"Audio guide padding reason: {_safe_str(audio_sync_info.get('reason'))}\n")
                    if _safe_str(audio_sync_info.get('error')):
                        lf.write(f"Audio guide padding error: {_safe_str(audio_sync_info.get('error'))}\n")
                lf.write(f"Planned duration: {planned_duration_for_retry:.6f}s\n")
                lf.write(f"Tail pad seconds: {LTX_GENERATION_TAIL_PAD_SECONDS:.3f}\n")
                lf.write(f"Short-shot generation padding applied: {bool(timing_plan.get('short_generation_padding_applied'))}\n")
                lf.write(f"Short-shot extra pad seconds: {_safe_float(timing_plan.get('short_generation_extra_pad_seconds'), 0.0):.3f}\n")
                lf.write(f"Requested fps: {fps}\n")
                lf.write(f"Audio guide: {audio_path or 'disabled'}\n\n")
                lf.flush()
                proc = subprocess.Popen(
                    attempt_args,
                    cwd=str(root),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    encoding="utf-8",
                    errors="replace",
                )
                assert proc.stdout is not None
                for line in proc.stdout:
                    try:
                        lf.write(line)
                        lf.flush()
                    except Exception:
                        pass
                    txt = str(line or "").strip()
                    if txt:
                        _emit(txt[:240])
                rc = int(proc.wait() or 0)
                lf.write(f"\nExit code: {rc}\n")

            actual_attempt_duration = _probe_media_duration_seconds(root, attempt_clip_path) if attempt_clip_path.is_file() else 0.0
            attempt_record = {
                "attempt": attempt_index,
                "returncode": int(rc),
                "requested_frames": int(current_frames),
                "requested_duration": round(float(current_frames) / float(max(1, fps)), 6),
                "output_path": str(attempt_clip_path),
                "audio_guide_path": str(attempt_audio_path or ""),
                "audio_guide_duration_sync": dict(audio_sync_info) if isinstance(audio_sync_info, dict) else {},
                "actual_raw_duration": round(float(actual_attempt_duration), 6) if actual_attempt_duration else 0.0,
            }
            attempt_records.append(attempt_record)
            debug_payload["ltx_generation_attempts"] = attempt_records
            debug_payload["ltx_command_args"] = attempt_args
            debug_payload["ltx_command_summary"]["frames"] = int(current_frames)
            debug_payload["ltx_command_summary"]["requested_generation_frames"] = int(current_frames)
            debug_payload["ltx_command_summary"]["output"] = str(attempt_clip_path)
            _write_ltx_test_payload(test_payload_path, debug_payload)

            if rc != 0:
                break
            if not attempt_clip_path.is_file():
                break
            if planned_duration_for_retry <= 0.0 or actual_attempt_duration >= planned_ok_threshold:
                generation_duration_ok = True
                final_clip_path = attempt_clip_path
                if final_clip_path.resolve() != Path(clip_path).resolve():
                    try:
                        shutil.copy2(str(final_clip_path), str(clip_path))
                        final_clip_path = Path(clip_path)
                    except Exception:
                        pass
                break

            last_short_duration = actual_attempt_duration
            if attempt_index >= max_attempts:
                final_clip_path = attempt_clip_path
                break

            # LTX sometimes under-delivers short outputs. Do not pad or slow down.
            # Ask the backend for a genuinely longer new clip, based on the observed
            # short result, then trim naturally after a successful generation.
            ratio = planned_duration_for_retry / max(0.001, actual_attempt_duration)
            next_frames = int(math.ceil(float(current_frames) * ratio * float(LTX_SHORT_REGEN_SCALE_SAFETY)))
            # Make the next retry a real jump. A few LTX wrappers ignore small frame
            # increases, so add a full planned slot plus the short-shot pad.
            jump_frames = int(math.ceil((planned_duration_for_retry + float(LTX_SHORT_SHOT_EXTRA_PAD_SECONDS)) * float(max(1, fps))))
            next_frames = max(next_frames, current_frames + jump_frames, int(math.ceil(float(LTX_SHORT_SHOT_MIN_REQUEST_SECONDS) * float(max(1, fps)))))
            if backend == "vramlab" and max_generation_frames > 0:
                if int(current_frames) >= int(max_generation_frames):
                    _emit(f"LTX returned {actual_attempt_duration:.2f}s for planned {planned_duration_for_retry:.2f}s, but the LTX-VRAMLab frame cap is already reached ({max_generation_frames} frames). Not requesting more than the allowed maximum.")
                    break
                next_frames = min(int(next_frames), int(max_generation_frames))
            _emit(f"LTX returned {actual_attempt_duration:.2f}s for planned {planned_duration_for_retry:.2f}s. Creating a new longer raw clip with more frames and a longer audio guide; no freeze or slow-motion padding.")
            current_frames = int(next_frames)

        frames = int(final_frames_used)
        if rc != 0:

            return {"ok": False, "message": f"LTX image-to-video failed for {shot_id} (exit code {rc}). See log: {log_path}", "output_dir": str(test_dir), "log_path": str(log_path)}
        if not generation_duration_ok and planned_duration_for_retry > 0.0:
            msg = f"LTX returned a short clip again for {shot_id}: planned {planned_duration_for_retry:.2f}s, latest raw {last_short_duration:.2f}s. No freeze-hold or slow-motion padding was used. Recreate the shot again or increase the shot duration."
            try:
                debug_payload["short_output_retry_failed"] = True
                debug_payload["short_output_retry_message"] = msg
                _write_ltx_test_payload(test_payload_path, debug_payload)
            except Exception:
                pass
            _emit(msg)
            return {
                "ok": False,
                "status": "needs_regeneration",
                "message": msg,
                "output_dir": str(test_dir),
                "log_path": str(log_path),
                "test_payload_path": str(test_payload_path),
                "ltx_generation_attempts": attempt_records,
            }
        if not clip_path.is_file():
            return {"ok": False, "message": f"LTX finished but the expected clip was not found: {clip_path}", "output_dir": str(test_dir), "log_path": str(log_path)}

        sync_report = _create_ltx_sync_clip(
            root=root,
            raw_clip_path=clip_path,
            sync_clip_path=sync_clip_path,
            shot=shot,
            fps=fps,
            frames=frames,
            log_path=log_path,
            duration_report_path=duration_report_path,
        )

        try:
            debug_payload["output_exists"] = True
            debug_payload["output_size_bytes"] = int(clip_path.stat().st_size)
            debug_payload.update(sync_report)
            debug_payload["duration_report_path"] = str(duration_report_path)
            _write_ltx_test_payload(test_payload_path, debug_payload)
        except Exception:
            pass

        delta = _safe_float(sync_report.get("duration_delta"), 0.0)
        warning = _safe_str(sync_report.get("sync_warning"))
        if bool(sync_report.get("sync_correction_applied")):
            method = _safe_str(sync_report.get("duration_correction_method") or sync_report.get("sync_action"))
            if method == "retime_to_planned":
                msg = f"LTX test finished. Raw: {clip_path.name} | Sync-safe: {sync_clip_path.name}. Warning: old retime mode was requested but short-clip stretching is disabled."
            else:
                msg = f"LTX test finished. Raw: {clip_path.name} | Sync-safe: {sync_clip_path.name}. Trimmed extra {max(0.0, delta):.2f}s to preserve song sync."
        elif warning:
            if _safe_bool(sync_report.get("short_raw_padded_to_planned"), False):
                msg = f"LTX test finished. Raw: {clip_path.name} | Sync-safe: {sync_clip_path.name}. Short raw output was finished by holding the last frame; no slow-motion retime was used."
            else:
                msg = f"LTX test finished. Raw: {clip_path.name} | Sync-safe: {sync_clip_path.name}. Warning: {warning}"
        else:
            msg = f"LTX test finished. Raw: {clip_path.name} | Sync-safe: {sync_clip_path.name}."
        _emit(msg)
        return {
            "ok": True,
            "shot_id": shot_id,
            "output_dir": str(test_dir),
            "start_image_path": str(start_image_path),
            "ltx_clip_path": str(clip_path),
            "sync_clip_path": str(sync_clip_path),
            "duration_report_path": str(duration_report_path),
            "duration_report": sync_report,
            "log_path": str(log_path),
            "test_payload_path": str(test_payload_path),
            "message": msg,
        }
    except Exception as exc:
        return {"ok": False, "message": f"LTX single-shot test failed: {exc}"}



# -----------------------------
# Chunk 7A: full LTX director-shot generation
# -----------------------------

def _director_plan_shots(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    shots = obj.get("shots")
    if isinstance(shots, list):
        return [s for s in shots if isinstance(s, dict)]
    data = obj.get("director_plan")
    if isinstance(data, dict) and isinstance(data.get("shots"), list):
        return [s for s in data.get("shots") if isinstance(s, dict)]
    return []


def _make_ltx_full_run_dir(plan_path: Path) -> Path:
    base = (Path(plan_path).resolve().parent / "ltx_full_run").resolve()
    base.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cand = base / stamp
    if not cand.exists():
        cand.mkdir(parents=True, exist_ok=False)
        return cand.resolve()
    for idx in range(2, 1000):
        cand = base / f"{stamp}_{idx:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return cand.resolve()
    raise RuntimeError("Could not create a unique LTX full-run output folder.")


def _valid_nonempty_file(path: Any, min_bytes: int = 1024) -> bool:
    try:
        p = Path(_safe_str(path))
        return bool(p.is_file() and p.stat().st_size >= int(min_bytes))
    except Exception:
        return False


def _shot_existing_start_image_path(shot: Dict[str, Any]) -> str:
    keys = (
        "existing_start_image_path",
        "start_image_path",
        "director_start_image_path",
        "image_path",
        "start_image",
        "image",
    )
    for key in keys:
        val = _safe_str(shot.get(key))
        if val and os.path.isfile(val):
            return val
    return ""


def _write_full_run_text_report(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("FrameVision Music Clip Creator - LTX full run report")
    lines.append(f"Timestamp: {_safe_str(report.get('timestamp'))}")
    lines.append(f"Director plan: {_safe_str(report.get('ltx_director_plan_path'))}")
    lines.append(f"Output folder: {_safe_str(report.get('output_dir'))}")
    lines.append("")
    lines.append(f"Total shots: {_safe_int(report.get('total_shots'), 0)}")
    lines.append(f"Finished: {_safe_int(report.get('finished_count'), 0)}")
    lines.append(f"Failed: {_safe_int(report.get('failed_count'), 0)}")
    lines.append(f"Skipped: {_safe_int(report.get('skipped_count'), 0)}")
    lines.append(f"Cancelled: {_safe_bool(report.get('cancelled'), False)}")
    lines.append("")
    for item in report.get("shots", []) if isinstance(report.get("shots"), list) else []:
        if not isinstance(item, dict):
            continue
        lines.append(f"{_safe_str(item.get('shot_id'))}: {_safe_str(item.get('status'))}")
        msg = _safe_str(item.get("message") or item.get("error"))
        if msg:
            lines.append(f"  {msg}")
        for key in ("start_image_path", "raw_clip_path", "trimmed_clip_path"):
            val = _safe_str(item.get(key))
            if val:
                lines.append(f"  {key}: {val}")
        if "planned_duration" in item or "raw_duration" in item or "final_duration" in item:
            lines.append(
                "  durations: planned={:.3f}, raw={:.3f}, final={:.3f}, drift={:.3f}".format(
                    _safe_float(item.get("planned_duration"), 0.0),
                    _safe_float(item.get("raw_duration"), 0.0),
                    _safe_float(item.get("final_duration"), 0.0),
                    _safe_float(item.get("duration_delta"), 0.0),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_all_ltx_director_shots(payload: dict) -> dict:
    """Generate a complete LTX director run in two strict phases.

    Phase 1 prepares every required start image (including HiDream reference
    workflows). Phase 2 converts those prepared images to LTX clips.
    """
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Full LTX payload was not a dictionary."}
        progress_callback = payload.get("progress_callback")

        def _emit(message: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(str(message or ""))
                except Exception:
                    pass

        root_raw = _safe_str(payload.get("root_dir"))
        root = Path(root_raw).resolve() if root_raw else _project_root()
        plan_path = Path(_safe_str(payload.get("ltx_director_plan_path"))).expanduser().resolve()
        if not plan_path.is_file():
            return {"ok": False, "message": f"LTX director plan was not found: {plan_path}"}
        director_plan = _read_json_file(str(plan_path))
        safety = _enforce_ltx_start_end_duration_contract(
            director_plan,
            plan_path=plan_path,
            root=root,
            enabled=_payload_ltx_edge_duration_safety_enabled(payload, director_plan),
            refresh_audio_chunks=True,
        )
        if bool(safety.get("changed")):
            try:
                existing_warnings = _as_list(director_plan.get("warnings"))
                merged = existing_warnings + [w for w in _as_list(safety.get("warnings")) if w]
                if merged:
                    director_plan["warnings"] = merged[:80]
                _write_json_file(plan_path, director_plan)
                _emit("Updated LTX plan timing: first clip is at least 3 seconds and final clip targets 5 seconds.")
            except Exception as exc:
                _emit(f"Warning: could not save LTX duration safety update: {exc}")
        shots = _director_plan_shots(director_plan)
        if not shots:
            return {"ok": False, "message": "No shots were found in the LTX director plan."}

        out_dir_raw = _safe_str(payload.get("output_dir"))
        run_dir = Path(out_dir_raw).expanduser().resolve() if out_dir_raw else _make_ltx_full_run_dir(plan_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        report_json = run_dir / "ltx_full_run_report.json"
        report_txt = run_dir / "ltx_full_run_report.txt"

        image_mode = _safe_str(payload.get("image_mode"), "flux_klein_9b").lower() or "flux_klein_9b"
        if image_mode in {"use existing start image", "existing_start_image"}:
            image_mode = "existing"
        skip_completed = _safe_bool(payload.get("skip_completed"), True)
        trim_clips = _safe_bool(payload.get("trim_clips"), True)
        # Retry a failed LTX clip twice by default: total attempts = 3.
        # This handles occasional backend crashes without aborting the whole music-video run.
        ltx_retry_attempts = max(1, min(10, _safe_int(payload.get("ltx_retry_attempts"), _safe_int(payload.get("retry_attempts"), 3))))
        steps = max(1, _safe_int(payload.get("steps"), 8))
        resolution = _safe_str(payload.get("resolution") or director_plan.get("resolution") or director_plan.get("ltx_resolution"), "1280x704") or "1280x704"
        review_continue_missing_only = _safe_bool(payload.get("review_continue_missing_only"), False)
        use_review_current_images = _safe_bool(payload.get("use_review_current_images"), False)
        review_state = _read_ltx_review_state(plan_path) if (review_continue_missing_only or use_review_current_images) else {}
        review_dir = _ltx_review_dir(plan_path) if (review_continue_missing_only or use_review_current_images) else run_dir

        report: Dict[str, Any] = {
            "ok": True,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ltx_director_plan_path": str(plan_path),
            "output_dir": str(run_dir),
            "image_mode": image_mode,
            "skip_completed": skip_completed,
            "trim_clips": trim_clips,
            "ltx_retry_attempts_per_shot": ltx_retry_attempts,
            "total_shots": len(shots),
            "finished_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "cancelled": False,
            "master_audio_rule": "Split WAV chunks are only for LTX lipsync/audio guidance. Final assembly should use the original full song/final_master_audio as master audio, not raw LTX clip audio.",
            "shots": [],
        }

        def _save_report() -> None:
            try:
                _write_json_file(report_json, report)
            except Exception:
                pass
            try:
                _write_full_run_text_report(report_txt, report)
            except Exception:
                pass

        # Full music-clip runs are intentionally split into two strict phases.
        # This keeps large image models loaded long enough to finish the complete
        # start-image batch and prevents reference/HiDream workflows from falling
        # back to image 1 -> video 1 -> image 2 -> video 2 scheduling.
        prepared_start_images: Dict[str, str] = {}
        image_phase_failures: Dict[str, str] = {}
        image_phase_generated = 0
        image_phase_reused = 0
        image_phase_skipped_for_ready_clip = 0
        report["execution_order"] = "all_start_images_then_all_ltx_videos"
        hidream_batch_prepared_jobs: List[Dict[str, Any]] = []
        report["image_phase"] = {
            "status": "running",
            "generated_count": 0,
            "reused_count": 0,
            "skipped_for_ready_clip_count": 0,
            "failed_count": 0,
            "prepared_start_images": {},
            "failures": {},
        }
        _save_report()

        _emit(f"Phase 1/2 - preparing all start images: {len(shots)} shots")
        for image_idx, image_shot in enumerate(shots, start=1):
            image_sid = _safe_str(image_shot.get("id")) or f"LTX{image_idx:02d}"
            image_stem = _safe_stem(image_sid)
            try:
                if callable(payload.get("cancel_check")) and bool(payload.get("cancel_check")()):
                    report["cancelled"] = True
                    report["image_phase"]["status"] = "cancelled"
                    _emit("Cancelled while preparing start images.")
                    break

                image_review_item = ((review_state.get("shots") or {}).get(image_sid) if isinstance(review_state.get("shots"), dict) else {}) or {}
                image_review_start = _safe_str(image_review_item.get("current_start_image_path"))
                image_review_clip = _safe_str(image_review_item.get("current_clip_path"))
                image_review_clip_ready = bool(image_review_clip and os.path.isfile(image_review_clip))
                image_force_review_clip = bool(
                    use_review_current_images
                    and image_review_start
                    and os.path.isfile(image_review_start)
                    and not image_review_clip_ready
                    and _safe_bool(image_review_item.get("clip_invalidated_by_new_image"), False)
                )
                image_trimmed = run_dir / f"{image_stem}_ltx_trimmed.mp4"

                # A ready clip needs no new image. This mirrors the video phase's
                # skip rules so resuming a run does not waste image generations.
                if review_continue_missing_only and image_review_clip_ready:
                    image_phase_skipped_for_ready_clip += 1
                    continue
                if skip_completed and (not image_force_review_clip) and _valid_nonempty_file(image_trimmed):
                    image_phase_skipped_for_ready_clip += 1
                    continue

                if image_force_review_clip:
                    prepared_start_images[image_sid] = image_review_start
                    image_phase_reused += 1
                    _emit(f"Phase 1/2 image {image_idx}/{len(shots)}: using reviewed start image for {image_sid}")
                    continue

                if image_mode == "existing":
                    existing_image = _shot_existing_start_image_path(image_shot)
                    if not existing_image or not os.path.isfile(existing_image):
                        raise RuntimeError("Use existing start image is selected, but this shot has no valid existing start image path assigned.")
                    prepared_start_images[image_sid] = existing_image
                    image_phase_reused += 1
                    _emit(f"Phase 1/2 image {image_idx}/{len(shots)}: registered existing start image for {image_sid}")
                    continue

                generated_image_path = (run_dir / f"{image_stem}_start.png").resolve()
                if skip_completed and _valid_nonempty_file(generated_image_path):
                    prepared_start_images[image_sid] = str(generated_image_path)
                    image_phase_reused += 1
                    _emit(f"Phase 1/2 image {image_idx}/{len(shots)}: reusing completed start image for {image_sid}")
                    continue

                _emit(f"Phase 1/2 image {image_idx}/{len(shots)}: preparing {image_sid} with {image_mode.replace('_', ' ')}")
                image_payload: Dict[str, Any] = {
                    "root_dir": str(root),
                    "ltx_director_plan_path": str(plan_path),
                    "shot_id": image_sid,
                    "image_model": image_mode,
                    "seed": payload.get("seed"),
                    "resolution": resolution,
                    "output_dir": str(run_dir),
                    "start_image_name": f"{image_stem}_start.png",
                    "start_image_payload_name": f"{image_stem}_start_image_payload.json",
                    "start_image_log_name": f"{image_stem}_imagegen.log.txt",
                    "progress_callback": lambda msg, _sid=image_sid: _emit(f"{_sid}: {msg}"),
                }
                if "character_reference" in payload:
                    image_payload["character_reference"] = payload.get("character_reference")
                prepared = generate_ltx_start_image_for_shot(dict(image_payload, prepare_only=True))
                if not isinstance(prepared, dict) or not bool(prepared.get("ok")):
                    raise RuntimeError(_safe_str(prepared.get("message") if isinstance(prepared, dict) else "", "Start image preparation failed.") or "Start image preparation failed.")
                prepared_model = _safe_str(prepared.get("image_model"), image_mode)
                if prepared_model == "hidream":
                    hidream_batch_prepared_jobs.append(prepared)
                    _emit(f"Phase 1/2 image {image_idx}/{len(shots)}: queued {image_sid} for warm HiDream batch")
                else:
                    image_result = generate_ltx_start_image_for_shot(image_payload)
                    if not isinstance(image_result, dict) or not bool(image_result.get("ok")):
                        raise RuntimeError(_safe_str(image_result.get("message") if isinstance(image_result, dict) else "", "Start image generation failed.") or "Start image generation failed.")
                    final_image_path = _safe_str(image_result.get("start_image_path"))
                    if not final_image_path or not os.path.isfile(final_image_path):
                        raise RuntimeError("Start image generation reported success, but no start image file was found.")
                    prepared_start_images[image_sid] = final_image_path
                    image_phase_generated += 1
            except Exception as image_exc:
                image_phase_failures[image_sid] = str(image_exc)
                _emit(f"Start image failed for {image_sid}: {image_exc}")

            report["image_phase"].update({
                "generated_count": image_phase_generated,
                "reused_count": image_phase_reused,
                "skipped_for_ready_clip_count": image_phase_skipped_for_ready_clip,
                "failed_count": len(image_phase_failures),
                "prepared_start_images": dict(prepared_start_images),
                "failures": dict(image_phase_failures),
            })
            _save_report()

        if hidream_batch_prepared_jobs and not _safe_bool(report.get("cancelled"), False):
            try:
                _emit(f"Phase 1/2 - running warm HiDream batch for {len(hidream_batch_prepared_jobs)} queued image(s)")
                batch_results = _run_hidream_batch_start_images(
                    root,
                    jobs=hidream_batch_prepared_jobs,
                    progress_callback=lambda msg: _emit(f"HiDream batch: {msg}"),
                )
                result_map = {}
                if isinstance(batch_results.get("jobs"), list):
                    for item in batch_results.get("jobs") or []:
                        if isinstance(item, dict):
                            result_map[_safe_str(item.get("shot_id"))] = item
                for prepared in hidream_batch_prepared_jobs:
                    sid = _safe_str(prepared.get("shot_id"))
                    res = result_map.get(sid) or {}
                    final_image_path = _safe_str(res.get("output_image") or prepared.get("start_image_path"))
                    if bool(res.get("ok")) and final_image_path and os.path.isfile(final_image_path):
                        prepared_start_images[sid] = final_image_path
                        image_phase_generated += 1
                    else:
                        image_phase_failures[sid] = _safe_str(res.get("error") or batch_results.get("message") or "HiDream batch generation failed.")
                        _emit(f"Start image failed for {sid}: {image_phase_failures[sid]}")
                    report["image_phase"].update({
                        "generated_count": image_phase_generated,
                        "reused_count": image_phase_reused,
                        "skipped_for_ready_clip_count": image_phase_skipped_for_ready_clip,
                        "failed_count": len(image_phase_failures),
                        "prepared_start_images": dict(prepared_start_images),
                        "failures": dict(image_phase_failures),
                    })
                    _save_report()
            except Exception as batch_exc:
                batch_msg = str(batch_exc)
                for prepared in hidream_batch_prepared_jobs:
                    sid = _safe_str(prepared.get("shot_id"))
                    if sid not in prepared_start_images and sid not in image_phase_failures:
                        image_phase_failures[sid] = batch_msg
                _emit(f"HiDream warm batch failed: {batch_msg}")
                report["image_phase"].update({
                    "generated_count": image_phase_generated,
                    "reused_count": image_phase_reused,
                    "skipped_for_ready_clip_count": image_phase_skipped_for_ready_clip,
                    "failed_count": len(image_phase_failures),
                    "prepared_start_images": dict(prepared_start_images),
                    "failures": dict(image_phase_failures),
                })
                _save_report()

        report["image_phase"].update({
            "status": "cancelled" if _safe_bool(report.get("cancelled"), False) else "finished",
            "generated_count": image_phase_generated,
            "reused_count": image_phase_reused,
            "skipped_for_ready_clip_count": image_phase_skipped_for_ready_clip,
            "failed_count": len(image_phase_failures),
            "prepared_start_images": dict(prepared_start_images),
            "failures": dict(image_phase_failures),
        })
        _save_report()

        if _safe_bool(report.get("cancelled"), False):
            msg = (
                f"Full LTX run cancelled during start-image preparation. "
                f"Images ready: {len(prepared_start_images)} | Image failures: {len(image_phase_failures)} | Output: {run_dir}"
            )
            report["message"] = msg
            report["report_json"] = str(report_json)
            report["report_txt"] = str(report_txt)
            _save_report()
            return report

        _emit(
            f"Phase 1/2 complete - start images ready: {len(prepared_start_images)} | "
            f"failed: {len(image_phase_failures)}. Phase 2/2 - generating all LTX videos."
        )

        _emit(f"Starting Phase 2/2 LTX video run: {len(shots)} shots")
        for idx, shot in enumerate(shots, start=1):
            sid = _safe_str(shot.get("id")) or f"LTX{idx:02d}"
            stem = _safe_stem(sid)
            item: Dict[str, Any] = {"shot_id": sid, "index": idx, "status": "pending"}
            prepared_start = _safe_str(prepared_start_images.get(sid))
            review_item = ((review_state.get("shots") or {}).get(sid) if isinstance(review_state.get("shots"), dict) else {}) or {}
            review_start = _safe_str(review_item.get("current_start_image_path"))
            review_clip = _safe_str(review_item.get("current_clip_path"))
            review_clip_ready = bool(review_clip and os.path.isfile(review_clip))
            force_review_clip_from_image = bool(
                use_review_current_images
                and review_start
                and os.path.isfile(review_start)
                and not review_clip_ready
                and _safe_bool(review_item.get("clip_invalidated_by_new_image"), False)
            )
            try:
                if callable(payload.get("cancel_check")) and bool(payload.get("cancel_check")()):
                    report["cancelled"] = True
                    item["status"] = "cancelled"
                    item["message"] = "Cancelled before this shot started."
                    report["shots"].append(item)
                    _save_report()
                    break

                if sid in image_phase_failures:
                    item.update({
                        "status": "failed",
                        "stage": "start_image",
                        "error": image_phase_failures[sid],
                        "message": "Video generation was skipped because the start image failed in Phase 1/2.",
                        "audio_path": _safe_str(shot.get("audio_clip_path")),
                        "can_recreate_later": True,
                    })
                    report["failed_count"] = _safe_int(report.get("failed_count"), 0) + 1
                    report["shots"].append(item)
                    _emit(f"Skipped video for {sid}: start image failed in Phase 1/2")
                    _save_report()
                    continue

                trimmed = run_dir / f"{stem}_ltx_trimmed.mp4"
                if review_continue_missing_only and review_clip_ready:
                    item.update({
                        "status": "skipped",
                        "message": "Reviewed clip already exists.",
                        "trimmed_clip_path": review_clip,
                    })
                    report["skipped_count"] = _safe_int(report.get("skipped_count"), 0) + 1
                    report["shots"].append(item)
                    _emit(f"Skipped {sid}: reviewed clip already exists")
                    _save_report()
                    continue
                if skip_completed and (not force_review_clip_from_image) and _valid_nonempty_file(trimmed):
                    item.update({
                        "status": "skipped",
                        "message": "Trimmed clip already exists.",
                        "trimmed_clip_path": str(trimmed),
                    })
                    report["skipped_count"] = _safe_int(report.get("skipped_count"), 0) + 1
                    report["shots"].append(item)
                    _emit(f"Skipped {sid}: already completed")
                    _save_report()
                    continue

                _emit(f"Generating shot {idx} / {len(shots)}: {sid}")

                run_image_mode = image_mode
                existing_start = ""
                shot_output_dir = run_dir
                start_image_name = f"{stem}_start.png"
                raw_clip_name = f"{stem}_ltx_raw.mp4"
                sync_clip_name = f"{stem}_ltx_trimmed.mp4" if trim_clips else f"{stem}_ltx_sync.mp4"
                test_payload_name = f"{stem}_payload.json"
                log_name = f"{stem}_ltx23.log.txt"
                duration_report_name = f"{stem}_duration_report.json"
                if force_review_clip_from_image:
                    run_image_mode = "existing"
                    existing_start = review_start
                    shot_output_dir = review_dir
                    start_image_name = f"{stem}_review_start_used.png"
                    raw_clip_name = f"{stem}_review_ltx_raw.mp4"
                    sync_clip_name = f"{stem}_review_ltx_trimmed.mp4" if trim_clips else f"{stem}_review_ltx_sync.mp4"
                    test_payload_name = f"{stem}_review_payload.json"
                    log_name = f"{stem}_review_ltx23.log.txt"
                    duration_report_name = f"{stem}_review_duration_report.json"
                elif prepared_start:
                    # Phase 1 already generated or registered this image. Force the
                    # single-shot runner into clip-only mode so no image model can
                    # re-enter between LTX video generations.
                    run_image_mode = "existing"
                    existing_start = prepared_start
                elif run_image_mode == "existing":
                    existing_start = _shot_existing_start_image_path(shot)
                    if not existing_start:
                        raise RuntimeError("Use existing start image is selected, but this shot has no existing start image path assigned.")

                single_payload: Dict[str, Any] = {
                    "root_dir": str(root),
                    "ltx_director_plan_path": str(plan_path),
                    "shot_id": sid,
                    "image_mode": run_image_mode,
                    "existing_start_image_path": existing_start,
                    "steps": steps,
                    "resolution": resolution,
                    "allow_no_audio": False,
                    "output_dir": str(shot_output_dir),
                    "start_image_name": start_image_name,
                    "raw_clip_name": raw_clip_name,
                    "sync_clip_name": sync_clip_name,
                    "test_payload_name": test_payload_name,
                    "log_name": log_name,
                    "duration_report_name": duration_report_name,
                    "progress_callback": lambda msg, _sid=sid: _emit(f"{_sid}: {msg}"),
                }
                for key in ("ltx_backend", "ltx_generation_backend", "ltx_lora_file", "ltx_lora_json", "ltx_lora_multiplier", "wangp_root", "wgp_py", "character_reference"):
                    if key in payload:
                        single_payload[key] = payload.get(key)

                result: Dict[str, Any] = {}
                attempt_summaries: List[Dict[str, Any]] = []
                last_error = ""
                for attempt_no in range(1, ltx_retry_attempts + 1):
                    retry_payload = dict(single_payload)
                    if attempt_no > 1:
                        existing_retry_start = str((shot_output_dir / start_image_name).resolve())
                        if os.path.isfile(existing_retry_start):
                            retry_payload["image_mode"] = "existing"
                            retry_payload["existing_start_image_path"] = existing_retry_start
                        retry_payload["progress_callback"] = lambda msg, _sid=sid, _attempt=attempt_no: _emit(f"{_sid}: retry {_attempt}/{ltx_retry_attempts}: {msg}")
                        _emit(f"{sid}: retrying failed LTX clip, attempt {attempt_no}/{ltx_retry_attempts}...")
                    result_raw = run_single_ltx_shot_test(retry_payload)
                    result = result_raw if isinstance(result_raw, dict) else {}
                    ok_attempt = bool(isinstance(result_raw, dict) and result_raw.get("ok"))
                    last_error = _safe_str(result.get("message"), "LTX shot failed.") or "LTX shot failed."
                    attempt_summaries.append({
                        "attempt": attempt_no,
                        "ok": ok_attempt,
                        "message": last_error,
                        "output_dir": _safe_str(result.get("output_dir")),
                        "log_path": _safe_str(result.get("log_path")),
                    })
                    if ok_attempt:
                        if attempt_no > 1:
                            _emit(f"{sid}: retry succeeded on attempt {attempt_no}/{ltx_retry_attempts}")
                        break
                    if attempt_no < ltx_retry_attempts:
                        _emit(f"{sid}: attempt {attempt_no}/{ltx_retry_attempts} failed: {last_error}")
                if not isinstance(result, dict) or not bool(result.get("ok")):
                    item["attempts"] = attempt_summaries
                    raise RuntimeError(f"LTX shot failed after {ltx_retry_attempts} attempt(s): {last_error}")

                dur = result.get("duration_report") if isinstance(result.get("duration_report"), dict) else {}
                item.update({
                    "status": "ok",
                    "message": _safe_str(result.get("message"), "Finished."),
                    "attempts": attempt_summaries,
                    "succeeded_attempt": len(attempt_summaries),
                    "retry_count": max(0, len(attempt_summaries) - 1),
                    "start_image_path": _safe_str(result.get("start_image_path")),
                    "raw_clip_path": _safe_str(result.get("ltx_clip_path")),
                    "trimmed_clip_path": _safe_str(result.get("sync_clip_path")),
                    "payload_path": str(shot_output_dir / test_payload_name),
                    "log_path": _safe_str(result.get("log_path")),
                    "duration_report_path": _safe_str(result.get("duration_report_path")),
                    "audio_path": _safe_str(shot.get("audio_clip_path")),
                    "planned_duration": _safe_float(dur.get("planned_duration"), 0.0),
                    "planned_target_frames": _safe_int(dur.get("planned_target_frames"), _safe_int(shot.get("target_frames"), 0)),
                    "generation_tail_pad_seconds": _safe_float(dur.get("generation_tail_pad_seconds"), 0.0),
                    "generation_pre_pad_seconds": _safe_float(dur.get("generation_pre_pad_seconds"), 0.0),
                    "short_generation_padding_applied": _safe_bool(dur.get("short_generation_padding_applied"), False),
                    "short_generation_extra_pad_seconds": _safe_float(dur.get("short_generation_extra_pad_seconds"), 0.0),
                    "short_generation_min_request_seconds": _safe_float(dur.get("short_generation_min_request_seconds"), 0.0),
                    "requested_generation_duration": _safe_float(dur.get("requested_generation_duration"), 0.0),
                    "requested_generation_frames": _safe_int(dur.get("requested_generation_frames"), 0),
                    "audio_guide_start": _safe_float(shot.get("audio_guide_start"), 0.0),
                    "audio_guide_end": _safe_float(shot.get("audio_guide_end"), 0.0),
                    "audio_guide_duration": _safe_float(shot.get("audio_guide_duration"), 0.0),
                    "final_timeline_duration": _safe_float(dur.get("final_timeline_duration"), _safe_float(dur.get("planned_duration"), 0.0)),
                    "raw_output_duration": _safe_float(dur.get("raw_output_duration"), _safe_float(dur.get("actual_raw_duration"), 0.0)),
                    "raw_duration": _safe_float(dur.get("actual_raw_duration"), 0.0),
                    "corrected_output_duration": _safe_float(dur.get("corrected_output_duration"), _safe_float(dur.get("sync_clip_duration"), 0.0)),
                    "final_duration": _safe_float(dur.get("sync_clip_duration"), 0.0),
                    "duration_delta": _safe_float(dur.get("duration_delta"), 0.0),
                    "duration_correction_method": _safe_str(dur.get("duration_correction_method") or dur.get("sync_action")),
                    "correction_quality_mode": _safe_str(dur.get("correction_quality_mode")),
                    "audio_preserved": _safe_bool(dur.get("audio_preserved"), False),
                    "video_reencoded": _safe_bool(dur.get("video_reencoded"), False),
                    "fallback_used": _safe_bool(dur.get("fallback_used"), False),
                    "fallback_reason": _safe_str(dur.get("fallback_reason")),
                    "raw_output_shorter_than_planned": _safe_bool(dur.get("raw_output_shorter_than_planned"), False),
                    "short_raw_padded_to_planned": _safe_bool(dur.get("short_raw_padded_to_planned"), False),
                    "freeze_last_frame_padding_applied": _safe_bool(dur.get("freeze_last_frame_padding_applied"), False),
                    "short_raw_padding_seconds": _safe_float(dur.get("short_raw_padding_seconds"), _safe_float(dur.get("pad_seconds"), 0.0)),
                    "retime_skipped_to_avoid_slow_motion": _safe_bool(dur.get("retime_skipped_to_avoid_slow_motion"), False),
                    "recommended_action": _safe_str(dur.get("recommended_action")),
                    "sync_warning": _safe_str(dur.get("sync_warning")),
                    "command_summary": {
                        "prompt": _safe_str(shot.get("director_timestamped_video_prompt")),
                        "negative": _safe_str(shot.get("director_negative_prompt")),
                        "fps": _safe_int(shot.get("target_fps"), 24),
                        "planned_frames": _safe_int(dur.get("planned_target_frames"), _safe_int(shot.get("target_frames"), 0)),
                        "frames": _safe_int(dur.get("requested_generation_frames"), 0),
                        "requested_generation_frames": _safe_int(dur.get("requested_generation_frames"), 0),
                    },
                })
                if _safe_bool(dur.get("raw_output_shorter_than_planned"), False) and not _safe_bool(dur.get("short_raw_padded_to_planned"), False):
                    item["status"] = "needs_regeneration"
                    item["message"] = LTX_SHORT_RAW_REGEN_MESSAGE
                    report["failed_count"] = _safe_int(report.get("failed_count"), 0) + 1
                    _emit(f"{sid}: raw output shorter than planned; regenerate this shot")
                else:
                    if _safe_bool(dur.get("short_raw_padded_to_planned"), False):
                        item["status"] = "ok_padded_hold"
                        item["message"] = "Finished with last-frame hold padding; no slow-motion retime was used."
                    report["finished_count"] = _safe_int(report.get("finished_count"), 0) + 1
                    _emit(f"Finished {sid}")
                if force_review_clip_from_image:
                    try:
                        review_item.update({
                            "status": item.get("status") or "Updated",
                            "current_start_image_path": review_start,
                            "current_raw_clip_path": _safe_str(result.get("ltx_clip_path")),
                            "current_clip_path": _safe_str(result.get("sync_clip_path") or result.get("ltx_clip_path")),
                            "last_clip_result": result,
                            "clip_invalidated_by_new_image": False,
                            "review_clip_prompt_override_pending": False,
                        })
                        _write_ltx_review_state(plan_path, review_state)
                    except Exception as state_exc:
                        item.setdefault("warnings", []).append(f"Could not update review clip pointer: {state_exc}")
                report["shots"].append(item)
                _save_report()
            except Exception as exc:
                item.update({
                    "status": "failed",
                    "error": str(exc),
                    "audio_path": _safe_str(shot.get("audio_clip_path")),
                    "can_recreate_later": True,
                })
                report["failed_count"] = _safe_int(report.get("failed_count"), 0) + 1
                report["shots"].append(item)
                _emit(f"Failed {sid}: {exc}")
                _save_report()
                continue

        _save_report()
        if _safe_bool(report.get("cancelled"), False):
            msg = (
                f"Full LTX run cancelled. Finished: {report['finished_count']} | "
                f"Failed: {report['failed_count']} | Skipped: {report['skipped_count']} | "
                f"Output: {run_dir}"
            )
        else:
            msg = (
                f"Full LTX run finished. Finished: {report['finished_count']} | "
                f"Failed: {report['failed_count']} | Skipped: {report['skipped_count']} | "
                f"Output: {run_dir}"
            )
        _emit(msg)
        report["message"] = msg
        report["report_json"] = str(report_json)
        report["report_txt"] = str(report_txt)
        return report
    except Exception as exc:
        return {"ok": False, "message": f"Full LTX run failed: {exc}"}



# -----------------------------
# LTX review / recreate helpers
# -----------------------------

def _ltx_review_dir(plan_path: Path) -> Path:
    d = (plan_path.parent / "ltx_review_current").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _ltx_review_state_path(plan_path: Path) -> Path:
    return (plan_path.parent / "musicclip_ltx_review_state.json").resolve()


def _read_ltx_review_state(plan_path: Path) -> Dict[str, Any]:
    path = _ltx_review_state_path(plan_path)
    if path.is_file():
        data = _read_json_file(str(path))
        if isinstance(data, dict):
            data.setdefault("shots", {})
            return data
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "director_plan_path": str(plan_path),
        "review_dir": str(_ltx_review_dir(plan_path)),
        "shots": {},
    }


def _write_ltx_review_state(plan_path: Path, state: Dict[str, Any]) -> None:
    if not isinstance(state, dict):
        state = {}
    state["updated_at"] = datetime.now().isoformat(timespec="seconds")
    state["director_plan_path"] = str(plan_path)
    state.setdefault("review_dir", str(_ltx_review_dir(plan_path)))
    state.setdefault("shots", {})
    _write_json_file(_ltx_review_state_path(plan_path), state)


def _ltx_review_shot_state(state: Dict[str, Any], shot_id: str) -> Dict[str, Any]:
    shots = state.setdefault("shots", {}) if isinstance(state, dict) else {}
    if not isinstance(shots, dict):
        state["shots"] = shots = {}
    item = shots.get(shot_id)
    if not isinstance(item, dict):
        item = {}
        shots[shot_id] = item
    return item


def _latest_ltx_review_or_fullrun_paths(plan_path: Path, shot: Dict[str, Any], full_run_dir: Optional[Path] = None) -> Dict[str, str]:
    sid = _safe_str(shot.get("id"))
    stem = _safe_stem(sid)
    state = _read_ltx_review_state(plan_path)
    item = ((state.get("shots") or {}).get(sid) if isinstance(state.get("shots"), dict) else {}) or {}
    start = _safe_str(item.get("current_start_image_path"))
    raw = _safe_str(item.get("current_raw_clip_path"))
    clip = _safe_str(item.get("current_clip_path")) or raw
    review_dir = Path(_safe_str(state.get("review_dir")) or str(_ltx_review_dir(plan_path)))
    if not start:
        for cand in (review_dir / f"{stem}_review_start.png", review_dir / f"{stem}_start.png"):
            if cand.is_file():
                start = str(cand); break
    if not clip:
        for cand in (review_dir / f"{stem}_review_ltx_trimmed.mp4", review_dir / f"{stem}_review_ltx_sync.mp4", review_dir / f"{stem}_review_ltx_raw.mp4"):
            if cand.is_file():
                clip = str(cand); break
    if not full_run_dir:
        try:
            full_run_dir = _latest_ltx_full_run_dir(plan_path)
        except Exception:
            full_run_dir = None
    if full_run_dir and full_run_dir.is_dir():
        if not start:
            cand = full_run_dir / f"{stem}_start.png"
            if cand.is_file():
                start = str(cand)
        if not clip:
            for cand in (full_run_dir / f"{stem}_ltx_trimmed.mp4", full_run_dir / f"{stem}_ltx_sync.mp4", full_run_dir / f"{stem}_ltx_raw.mp4"):
                if cand.is_file():
                    clip = str(cand); break
    return {"start_image_path": start, "clip_path": clip, "raw_clip_path": raw}


def load_ltx_review_state(payload: dict) -> dict:
    """Return simple per-shot review rows for the LTX workflow UI."""
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Review payload was not a dictionary."}
        plan_path = Path(_safe_str(payload.get("ltx_director_plan_path"))).expanduser().resolve()
        if not plan_path.is_file():
            return {"ok": False, "message": f"LTX director plan was not found: {plan_path}"}
        director_plan = _read_json_file(str(plan_path))
        safety = _enforce_ltx_start_end_duration_contract(
            director_plan,
            plan_path=plan_path,
            root=Path(_safe_str(payload.get("root_dir")) or str(_project_root())).resolve(),
            enabled=_payload_ltx_edge_duration_safety_enabled(payload, director_plan),
            refresh_audio_chunks=True,
        )
        if bool(safety.get("changed")):
            existing_warnings = _as_list(director_plan.get("warnings"))
            merged = existing_warnings + [w for w in _as_list(safety.get("warnings")) if w]
            if merged:
                director_plan["warnings"] = merged[:80]
            _write_json_file(plan_path, director_plan)
        shots = _director_plan_shots(director_plan)
        state = _read_ltx_review_state(plan_path)
        full_run_dir = None
        try:
            full_run_dir = _latest_ltx_full_run_dir(plan_path)
        except Exception:
            full_run_dir = None
        rows: List[Dict[str, Any]] = []
        for idx, shot in enumerate(shots, start=1):
            sid = _safe_str(shot.get("id")) or f"LTX{idx:02d}"
            paths = _latest_ltx_review_or_fullrun_paths(plan_path, shot, full_run_dir)
            item = ((state.get("shots") or {}).get(sid) if isinstance(state.get("shots"), dict) else {}) or {}
            prompt = _safe_str(item.get("image_prompt")) or _safe_str(shot.get("director_image_prompt") or shot.get("image_prompt") or shot.get("template_image_prompt"))
            seed = item.get("image_seed")
            if seed in (None, ""):
                seed = ""
            clip_prompt = _safe_str(item.get("clip_prompt")) or _safe_str(shot.get("director_timestamped_video_prompt") or shot.get("director_video_prompt") or shot.get("video_prompt") or shot.get("template_timestamped_video_prompt") or shot.get("template_video_prompt"))
            clip_seed = item.get("clip_seed")
            if clip_seed in (None, ""):
                clip_seed = ""
            dur = _safe_float(shot.get("duration"), max(0.0, _safe_float(shot.get("song_end"), 0.0) - _safe_float(shot.get("song_start"), 0.0)))
            image_ready = bool(paths.get("start_image_path") and os.path.isfile(paths.get("start_image_path")))
            clip_ready = bool(paths.get("clip_path") and os.path.isfile(paths.get("clip_path")))
            clip_duration = _probe_media_duration_seconds(Path(_safe_str(payload.get("root_dir")) or str(_project_root())).resolve(), Path(paths.get("clip_path"))) if clip_ready else 0.0
            duration_status = "ok"
            if clip_ready and clip_duration > 0.0 and dur > 0.0 and clip_duration < dur - float(DURATION_TOLERANCE_SECONDS):
                duration_status = "needs_recreate"
            status = "Clip ready" if clip_ready else ("Needs clip" if image_ready else "Needs image")
            if duration_status == "needs_recreate":
                status = "Needs clip recreate"
            elif _safe_str(item.get("status")):
                status = _safe_str(item.get("status"))
            rows.append({
                "index": idx,
                "shot_id": sid,
                "title": _safe_str(shot.get("title") or shot.get("scene_role_summary") or shot.get("microclip_style"))[:80],
                "duration": dur,
                "image_ready": image_ready,
                "clip_ready": clip_ready,
                "clip_duration": round(float(clip_duration), 6) if clip_duration else 0.0,
                "duration_status": duration_status,
                "status": status,
                "image_prompt": prompt,
                "image_seed": seed,
                "clip_prompt": clip_prompt,
                "clip_seed": clip_seed,
                "original_clip_prompt": _safe_str(shot.get("director_timestamped_video_prompt") or shot.get("director_video_prompt") or shot.get("video_prompt") or shot.get("template_timestamped_video_prompt") or shot.get("template_video_prompt")),
                "start_image_path": paths.get("start_image_path") or "",
                "clip_path": paths.get("clip_path") or "",
                "summary": clip_prompt[:240],
            })
        return {"ok": True, "message": f"Loaded {len(rows)} review shots.", "review_state_path": str(_ltx_review_state_path(plan_path)), "review_dir": str(_ltx_review_dir(plan_path)), "shots": rows}
    except Exception as exc:
        return {"ok": False, "message": f"Could not load LTX review state: {exc}"}


def recreate_ltx_review_shot(payload: dict) -> dict:
    """Recreate one review shot image+clip or clip-only, then update current pointers."""
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "Review recreate payload was not a dictionary."}
        progress_callback = payload.get("progress_callback")
        def _emit(message: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(str(message or ""))
                except Exception:
                    pass
        plan_path = Path(_safe_str(payload.get("ltx_director_plan_path"))).expanduser().resolve()
        if not plan_path.is_file():
            return {"ok": False, "message": f"LTX director plan was not found: {plan_path}"}
        director_plan = _read_json_file(str(plan_path))
        safety = _enforce_ltx_start_end_duration_contract(
            director_plan,
            plan_path=plan_path,
            root=Path(_safe_str(payload.get("root_dir"))).resolve() if _safe_str(payload.get("root_dir")) else _project_root(),
            enabled=_payload_ltx_edge_duration_safety_enabled(payload, director_plan),
            refresh_audio_chunks=True,
        )
        if bool(safety.get("changed")):
            try:
                existing_warnings = _as_list(director_plan.get("warnings"))
                merged = existing_warnings + [w for w in _as_list(safety.get("warnings")) if w]
                if merged:
                    director_plan["warnings"] = merged[:80]
                _write_json_file(plan_path, director_plan)
                _emit("Updated LTX plan timing: first clip is at least 3 seconds and final clip targets 5 seconds.")
            except Exception as exc:
                _emit(f"Warning: could not save LTX duration safety update: {exc}")
        shot_id = _safe_str(payload.get("shot_id"))
        shot = _find_director_shot(director_plan, shot_id)
        if not shot:
            return {"ok": False, "message": f"Selected shot was not found in the director plan: {shot_id}"}
        shot_id = _safe_str(shot.get("id")) or shot_id
        stem = _safe_stem(shot_id)
        action = _safe_str(payload.get("action"), "image_and_clip").lower()
        review_dir = _ltx_review_dir(plan_path)
        state = _read_ltx_review_state(plan_path)
        item = _ltx_review_shot_state(state, shot_id)
        prompt_override = _safe_str(payload.get("image_prompt"))
        seed_raw = payload.get("image_seed")
        clip_prompt_override = _safe_str(payload.get("clip_prompt_override") or payload.get("video_prompt_override") or payload.get("clip_prompt"))
        clip_seed_raw = payload.get("clip_seed")
        image_model = _safe_str(payload.get("image_model") or payload.get("image_mode"), "flux_klein_9b") or "flux_klein_9b"
        start_path = _safe_str(item.get("current_start_image_path"))
        image_result: Dict[str, Any] = {}
        clip_result: Dict[str, Any] = {}
        if action in {"image", "image_and_clip", "both", "image_then_clip"}:
            _emit(f"Re-creating start image for {shot_id}...")
            image_payload = dict(payload)
            image_payload.update({
                "ltx_director_plan_path": str(plan_path),
                "shot_id": shot_id,
                "image_model": image_model,
                "output_dir": str(review_dir),
                "start_image_name": f"{stem}_review_start.png",
                "start_image_payload_name": f"{stem}_review_start_image_payload.json",
                "start_image_log_name": f"{stem}_review_imagegen.log.txt",
                "image_prompt_override": prompt_override,
                "progress_callback": progress_callback,
            })
            if seed_raw not in (None, ""):
                image_payload["seed"] = seed_raw
            image_result = generate_ltx_start_image_for_shot(image_payload)
            if not isinstance(image_result, dict) or not bool(image_result.get("ok")):
                raise RuntimeError(_safe_str(image_result.get("message") if isinstance(image_result, dict) else "", "Start-image recreate failed."))
            start_path = _safe_str(image_result.get("start_image_path"))
            item.update({
                "status": "Image ready - clip needs recreate",
                "image_prompt": prompt_override,
                "image_seed": seed_raw if seed_raw not in (None, "") else image_result.get("seed", seed_raw),
                "current_start_image_path": start_path,
                "current_raw_clip_path": "",
                "current_clip_path": "",
                "last_image_result": image_result,
                "clip_invalidated_by_new_image": True,
            })
            _write_ltx_review_state(plan_path, state)
            if action == "image":
                return {
                    "ok": True,
                    "message": f"Review image recreated: {shot_id}. Review the image, then recreate the clip or Continue/Reassemble.",
                    "shot_id": shot_id,
                    "review_state_path": str(_ltx_review_state_path(plan_path)),
                    "review_dir": str(review_dir),
                    "start_image_path": _safe_str(item.get("current_start_image_path")),
                    "clip_path": "",
                    "image_result": image_result,
                    "clip_result": {},
                    "image_only": True,
                }
            action = "clip"
        if action in {"clip", "clip_only", "image_and_clip", "both", "image_then_clip"}:
            if not start_path:
                paths = _latest_ltx_review_or_fullrun_paths(plan_path, shot)
                start_path = paths.get("start_image_path") or ""
            if not start_path or not os.path.isfile(start_path):
                return {"ok": False, "message": f"No current start image exists for {shot_id}. Re-create the image first."}
            _emit(f"Re-creating LTX clip for {shot_id}...")
            single_payload = dict(payload)
            if clip_prompt_override or clip_seed_raw not in (None, ""):
                item.update({
                    "clip_prompt": clip_prompt_override or item.get("clip_prompt") or _safe_str(shot.get("director_timestamped_video_prompt") or shot.get("director_video_prompt") or shot.get("video_prompt") or shot.get("template_timestamped_video_prompt") or shot.get("template_video_prompt")),
                    "clip_seed": clip_seed_raw if clip_seed_raw not in (None, "") else item.get("clip_seed", ""),
                    "review_clip_prompt_override_pending": bool(clip_prompt_override),
                })
                _write_ltx_review_state(plan_path, state)
            single_payload.update({
                "ltx_director_plan_path": str(plan_path),
                "shot_id": shot_id,
                "image_mode": "existing",
                "existing_start_image_path": start_path,
                "clip_prompt_override": clip_prompt_override,
                "output_dir": str(review_dir),
                "start_image_name": f"{stem}_review_start_used.png",
                "raw_clip_name": f"{stem}_review_ltx_raw.mp4",
                "sync_clip_name": f"{stem}_review_ltx_trimmed.mp4",
                "test_payload_name": f"{stem}_review_payload.json",
                "log_name": f"{stem}_review_ltx23.log.txt",
                "duration_report_name": f"{stem}_review_duration_report.json",
                "allow_no_audio": False,
                "progress_callback": progress_callback,
            })
            clip_result = run_single_ltx_shot_test(single_payload)
            if not isinstance(clip_result, dict) or not bool(clip_result.get("ok")):
                raise RuntimeError(_safe_str(clip_result.get("message") if isinstance(clip_result, dict) else "", "Clip recreate failed."))
            current_clip = _safe_str(clip_result.get("sync_clip_path") or clip_result.get("ltx_clip_path"))
            duration_report = clip_result.get("duration_report") if isinstance(clip_result.get("duration_report"), dict) else {}
            short_raw = _safe_bool(duration_report.get("raw_output_shorter_than_planned"), False)
            short_padded = _safe_bool(duration_report.get("short_raw_padded_to_planned"), False)
            item.update({
                "status": "Updated (padded hold)" if short_padded else ("Needs regeneration" if short_raw else "Updated"),
                "image_prompt": prompt_override or item.get("image_prompt") or _safe_str(shot.get("director_image_prompt") or shot.get("image_prompt") or shot.get("template_image_prompt")),
                "image_seed": seed_raw if seed_raw not in (None, "") else item.get("image_seed", ""),
                "clip_prompt": clip_prompt_override or item.get("clip_prompt") or _safe_str(shot.get("director_timestamped_video_prompt") or shot.get("director_video_prompt") or shot.get("video_prompt") or shot.get("template_timestamped_video_prompt") or shot.get("template_video_prompt")),
                "clip_seed": clip_seed_raw if clip_seed_raw not in (None, "") else item.get("clip_seed", ""),
                "review_clip_prompt_override_pending": False,
                "review_clip_prompt_override_applied": bool(clip_prompt_override),
                "current_start_image_path": start_path,
                "current_raw_clip_path": _safe_str(clip_result.get("ltx_clip_path")),
                "current_clip_path": current_clip,
                "last_clip_result": clip_result,
                "raw_output_shorter_than_planned": short_raw,
                "short_raw_padded_to_planned": short_padded,
                "freeze_last_frame_padding_applied": _safe_bool(duration_report.get("freeze_last_frame_padding_applied"), False),
                "short_raw_padding_seconds": _safe_float(duration_report.get("short_raw_padding_seconds"), _safe_float(duration_report.get("pad_seconds"), 0.0)),
                "retime_skipped_to_avoid_slow_motion": _safe_bool(duration_report.get("retime_skipped_to_avoid_slow_motion"), False),
                "recommended_action": _safe_str(duration_report.get("recommended_action")),
                "sync_warning": _safe_str(duration_report.get("sync_warning")),
            })
            _write_ltx_review_state(plan_path, state)
        return {
            "ok": True,
            "message": f"Review shot updated: {shot_id}",
            "shot_id": shot_id,
            "review_state_path": str(_ltx_review_state_path(plan_path)),
            "review_dir": str(review_dir),
            "start_image_path": _safe_str(item.get("current_start_image_path")),
            "clip_path": _safe_str(item.get("current_clip_path")),
            "image_result": image_result,
            "clip_result": clip_result,
        }
    except Exception as exc:
        return {"ok": False, "message": f"LTX review recreate failed: {exc}"}


# -----------------------------
# Chunk 7B: timeline-locked LTX final assembly
# -----------------------------

def _latest_ltx_full_run_dir(plan_path: Path) -> Path:
    base = (Path(plan_path).resolve().parent / "ltx_full_run").resolve()
    if not base.is_dir():
        raise RuntimeError(f"No ltx_full_run folder was found next to the director plan: {base}")
    candidates: List[Path] = []
    try:
        for child in base.iterdir():
            if child.is_dir():
                candidates.append(child.resolve())
    except Exception:
        pass
    if not candidates:
        raise RuntimeError(f"No LTX full-run folders were found in: {base}")
    candidates.sort(key=lambda pp: pp.stat().st_mtime if pp.exists() else 0.0, reverse=True)
    return candidates[0]


def _first_existing_file(paths: List[Path], min_bytes: int = 1024) -> str:
    for pp in paths:
        try:
            if pp.is_file() and pp.stat().st_size >= min_bytes:
                return str(pp.resolve())
        except Exception:
            continue
    return ""




def _latest_existing_retimed_clip(full_run_dir: Path, stem: str) -> str:
    candidates: List[Path] = []
    try:
        direct = Path(full_run_dir) / f"{stem}_ltx_retimed.mp4"
        if direct.is_file():
            candidates.append(direct.resolve())
    except Exception:
        pass
    try:
        for child in Path(full_run_dir).iterdir():
            if child.is_dir() and child.name.startswith("assembly_"):
                pp = child / f"{stem}_ltx_retimed.mp4"
                if pp.is_file():
                    candidates.append(pp.resolve())
    except Exception:
        pass
    candidates = [pp for pp in candidates if _valid_nonempty_file(pp)]
    if not candidates:
        return ""
    candidates.sort(key=lambda pp: pp.stat().st_mtime if pp.exists() else 0.0, reverse=True)
    return str(candidates[0])

def _full_run_report_clip_for_shot(full_run_dir: Path, shot_id: str) -> str:
    report_path = Path(full_run_dir) / "ltx_full_run_report.json"
    if not report_path.is_file():
        return ""
    try:
        report = _read_json_file(str(report_path))
    except Exception:
        return ""
    want = _safe_str(shot_id).lower()
    for item in report.get("shots", []) if isinstance(report.get("shots"), list) else []:
        if not isinstance(item, dict):
            continue
        if _safe_str(item.get("shot_id")).lower() != want:
            continue
        # Do not reuse old retimed clips here. Older runs may contain short clips that
        # were stretched with setpts/atempo before the no-slowdown guard existed.
        for key in ("trimmed_clip_path", "sync_clip_path", "raw_clip_path", "ltx_clip_path"):
            val = _safe_str(item.get(key))
            if val and _valid_nonempty_file(val):
                return str(Path(val).resolve())
    return ""


def _assembly_plan_fps(director_plan: Dict[str, Any], shots: List[Dict[str, Any]], payload: Dict[str, Any]) -> int:
    fps = _safe_int(payload.get("fps"), 0)
    if fps <= 0:
        fps = _safe_int(director_plan.get("fps"), 0)
    if fps <= 0:
        for shot in shots:
            fps = _safe_int(shot.get("target_fps"), 0)
            if fps > 0:
                break
    return max(1, fps or 24)




def _auto_regenerate_ltx_review_clip_for_assembly(
    *,
    root: Path,
    plan_path: Path,
    director_plan: Dict[str, Any],
    shot: Dict[str, Any],
    full_run_dir: Optional[Path],
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Recreate a stale/too-short review clip using the current start image.

    Used by assembly for legacy jobs where the first/last timing contract was
    applied after a shorter clip had already been generated. It updates the review
    state so a later reload/reassemble sees the new clip pointer.
    """
    sid = _safe_str(shot.get("id"))
    if not sid:
        return {"ok": False, "message": "Cannot regenerate unnamed LTX shot."}
    stem = _safe_stem(sid)
    paths = _latest_ltx_review_or_fullrun_paths(plan_path, shot, full_run_dir)
    start_path = _safe_str(paths.get("start_image_path"))
    if not start_path or not os.path.isfile(start_path):
        return {"ok": False, "message": f"{sid} needs a longer clip, but no start image was found for automatic recreate."}
    state = _read_ltx_review_state(plan_path)
    item_state = _ltx_review_shot_state(state, sid)
    clip_prompt = _safe_str(item_state.get("clip_prompt") or shot.get("director_timestamped_video_prompt") or shot.get("director_video_prompt") or shot.get("video_prompt"))
    clip_seed = item_state.get("clip_seed", "")
    review_dir = _ltx_review_dir(plan_path)
    if callable(progress_callback):
        try:
            progress_callback(f"{sid} is shorter than the updated plan. Recreating a real longer LTX clip now...")
        except Exception:
            pass
    result = run_single_ltx_shot_test({
        "root_dir": str(root),
        "ltx_director_plan_path": str(plan_path),
        "shot_id": sid,
        "image_mode": "existing",
        "existing_start_image_path": start_path,
        "seed": clip_seed,
        "clip_prompt_override": clip_prompt,
        "video_prompt_override": clip_prompt,
        "steps": 8,
        "resolution": _safe_str(shot.get("resolution"), "1280x720") or "1280x720",
        "allow_no_audio": False,
        "character_reference": director_plan.get("character_reference") if isinstance(director_plan.get("character_reference"), dict) else {},
        "output_dir": str(review_dir),
        "start_image_name": f"{stem}_review_start_used.png",
        "raw_clip_name": f"{stem}_review_ltx_raw.mp4",
        "sync_clip_name": f"{stem}_review_ltx_trimmed.mp4",
        "test_payload_name": f"{stem}_review_payload.json",
        "log_name": f"{stem}_review_ltx23.log.txt",
        "duration_report_name": f"{stem}_review_duration_report.json",
        "progress_callback": progress_callback,
    })
    if not isinstance(result, dict) or not bool(result.get("ok")):
        return {"ok": False, "message": _safe_str(result.get("message") if isinstance(result, dict) else "", "Automatic LTX clip recreate failed."), "result": result if isinstance(result, dict) else {}}
    current_clip = _safe_str(result.get("sync_clip_path") or result.get("ltx_clip_path"))
    item_state.update({
        "status": "Updated",
        "current_start_image_path": start_path,
        "current_raw_clip_path": _safe_str(result.get("ltx_clip_path")),
        "current_clip_path": current_clip,
        "clip_prompt": clip_prompt,
        "clip_seed": clip_seed,
        "review_clip_prompt_override_pending": False,
        "review_clip_prompt_override_applied": bool(clip_prompt),
        "auto_recreated_for_duration_safety": True,
        "last_clip_result": result,
    })
    _write_ltx_review_state(plan_path, state)
    return {"ok": True, "message": f"Recreated {sid} for updated duration safety.", "clip_path": current_clip, "result": result}

def _master_audio_from_plan(director_plan: Dict[str, Any]) -> str:
    keys = ("final_master_audio", "audio_path", "music_path", "source_audio_path", "master_audio_path")
    for key in keys:
        val = _safe_str(director_plan.get(key))
        if val and os.path.isfile(val):
            return str(Path(val).resolve())
    src = director_plan.get("source")
    if isinstance(src, dict):
        for key in keys:
            val = _safe_str(src.get(key))
            if val and os.path.isfile(val):
                return str(Path(val).resolve())
    return ""


def _planned_frame_grid_for_shot(shot: Dict[str, Any], fps: int) -> Dict[str, Any]:
    song_start = _safe_float(shot.get("song_start"), 0.0)
    song_end = _safe_float(shot.get("song_end"), 0.0)
    duration_json = max(0.0, song_end - song_start)
    if duration_json <= 0:
        duration_json = _safe_float(shot.get("duration"), 0.0)
        song_end = song_start + duration_json
    start_frame = int(round(song_start * float(fps)))
    end_frame = int(round(song_end * float(fps)))
    planned_frames = max(1, end_frame - start_frame)
    planned_duration = planned_frames / float(fps)
    return {
        "song_start": song_start,
        "song_end": song_end,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "planned_frames": planned_frames,
        "planned_duration_from_json": duration_json,
        "planned_duration_frame_grid": planned_duration,
    }


def _retime_ltx_clip_to_duration(
    *,
    root: Path,
    src: Path,
    dst: Path,
    fps: int,
    planned_duration: float,
    planned_frames: int,
    log_path: Path,
) -> Dict[str, Any]:
    ffmpeg = _find_media_binary(root, "FV_FFMPEG", "ffmpeg")
    actual = _probe_media_duration_seconds(root, src)
    if actual <= 0:
        return {"ok": False, "actual_duration": 0.0, "speed_factor": 0.0, "message": "Could not probe input clip duration."}
    planned = max(0.001, float(planned_duration))
    speed = max(0.001, float(actual) / planned)
    if actual < planned - float(DURATION_TOLERANCE_SECONDS):
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as lf:
                lf.write("\n[retime] Short input clip needs regeneration. No freeze-hold and no slow-motion retime is used.\n")
                lf.write(f"Input: {src}\nactual={actual:.6f} planned={planned:.6f} natural_duration_factor={speed:.8f}\n")
        except Exception:
            pass
        return {
            "ok": False,
            "status": "needs_regeneration",
            "actual_duration": round(float(actual), 6),
            "planned_duration": round(float(planned), 6),
            "speed_factor": round(float(speed), 8),
            "returncode": 0,
            "message": LTX_SHORT_RAW_REGEN_MESSAGE,
            "raw_output_shorter_than_planned": True,
            "short_raw_padded_to_planned": False,
            "freeze_last_frame_padding_applied": False,
            "retime_skipped_to_avoid_slow_motion": True,
            "recommended_action": "regenerate shot",
            "assembly_should_stop_cleanly": True,
        }
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    vf = (
        f"setpts=(PTS-STARTPTS)/{speed:.10f},"
        f"fps={int(fps)},"
        f"trim=duration={planned:.10f},"
        "setpts=PTS-STARTPTS,format=yuv420p"
    )
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-i", str(src),
        "-map", "0:v:0",
        "-vf", vf,
        "-frames:v", str(max(1, int(planned_frames))),
        "-r", str(int(fps)),
        "-an",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as lf:
            lf.write("\n[retime] Timeline-locked retime\n")
            lf.write(f"Input: {src}\nOutput: {dst}\n")
            lf.write(f"actual={actual:.6f} planned={planned:.6f} fps={fps} frames={planned_frames} speed_factor={speed:.8f}\n")
            lf.write("Command:\n" + " ".join(str(x) for x in cmd) + "\n\n")
            lf.flush()
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=1800,
            )
            lf.write(f"\n[retime] ffmpeg exit code: {proc.returncode}\n")
    except Exception as exc:
        try:
            with log_path.open("a", encoding="utf-8", errors="replace") as lf:
                lf.write(f"\n[retime] failed: {exc}\n")
        except Exception:
            pass
        return {"ok": False, "actual_duration": actual, "speed_factor": speed, "message": str(exc)}
    ok = bool(proc.returncode == 0 and dst.is_file() and dst.stat().st_size >= 1024)
    retimed = _probe_media_duration_seconds(root, dst) if ok else 0.0
    return {
        "ok": ok,
        "actual_duration": actual,
        "speed_factor": speed,
        "retimed_duration": retimed,
        "returncode": int(proc.returncode),
        "message": "ok" if ok else "ffmpeg retime failed or produced no output.",
    }


def _concat_retimed_clips(root: Path, clips: List[Path], concat_list: Path, output_path: Path, fps: int, log_path: Path) -> Dict[str, Any]:
    ffmpeg = _find_media_binary(root, "FV_FFMPEG", "ffmpeg")
    concat_list.parent.mkdir(parents=True, exist_ok=True)
    def _escape_concat_path(pp: Path) -> str:
        s = pp.resolve().as_posix().replace("'", "'\\''")
        return f"file '{s}'"
    concat_list.write_text("\n".join(_escape_concat_path(pp) for pp in clips) + "\n", encoding="utf-8")
    try:
        if output_path.exists():
            output_path.unlink()
    except Exception:
        pass
    copy_cmd = [ffmpeg, "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", "-an", str(output_path)]
    fallback_cmd = [
        ffmpeg, "-y", "-hide_banner", "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-an", "-vf", f"fps={int(fps)},format=yuv420p", "-r", str(int(fps)),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)
    ]
    used = "copy"
    rc = -1
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as lf:
            lf.write("\n[concat] Trying concat demuxer with stream copy\n")
            lf.write("Command:\n" + " ".join(str(x) for x in copy_cmd) + "\n\n")
            lf.flush()
            proc = subprocess.run(copy_cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=1800)
            rc = int(proc.returncode)
            lf.write(f"\n[concat] copy exit code: {rc}\n")
            if not (rc == 0 and output_path.is_file() and output_path.stat().st_size >= 1024):
                used = "reencode_fallback"
                try:
                    if output_path.exists():
                        output_path.unlink()
                except Exception:
                    pass
                lf.write("\n[concat] Copy concat failed; trying re-encode fallback\n")
                lf.write("Command:\n" + " ".join(str(x) for x in fallback_cmd) + "\n\n")
                lf.flush()
                proc = subprocess.run(fallback_cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=2400)
                rc = int(proc.returncode)
                lf.write(f"\n[concat] fallback exit code: {rc}\n")
    except Exception as exc:
        return {"ok": False, "method": used, "returncode": rc, "message": str(exc)}
    ok = bool(rc == 0 and output_path.is_file() and output_path.stat().st_size >= 1024)
    return {"ok": ok, "method": used, "returncode": rc, "duration": _probe_media_duration_seconds(root, output_path) if ok else 0.0}


def _mux_original_audio(root: Path, video_path: Path, audio_path: Path, output_path: Path, log_path: Path) -> Dict[str, Any]:
    ffmpeg = _find_media_binary(root, "FV_FFMPEG", "ffmpeg")
    try:
        if output_path.exists():
            output_path.unlink()
    except Exception:
        pass
    cmd = [
        ffmpeg, "-y", "-hide_banner",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]
    rc = -1
    try:
        with log_path.open("a", encoding="utf-8", errors="replace") as lf:
            lf.write("\n[mux] Adding original full music track as master audio\n")
            lf.write("Command:\n" + " ".join(str(x) for x in cmd) + "\n\n")
            lf.flush()
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", timeout=1800)
            rc = int(proc.returncode)
            lf.write(f"\n[mux] ffmpeg exit code: {rc}\n")
    except Exception as exc:
        return {"ok": False, "returncode": rc, "message": str(exc)}
    ok = bool(rc == 0 and output_path.is_file() and output_path.stat().st_size >= 1024)
    return {"ok": ok, "returncode": rc, "duration": _probe_media_duration_seconds(root, output_path) if ok else 0.0}


def _write_ltx_assembly_text_report(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("FrameVision Music Clip Creator - LTX final assembly report")
    lines.append(f"Created: {_safe_str(report.get('created_at'))}")
    lines.append(f"Director plan: {_safe_str(report.get('director_plan_path'))}")
    lines.append(f"Full run: {_safe_str(report.get('full_run_dir'))}")
    lines.append(f"Output folder: {_safe_str(report.get('output_dir'))}")
    lines.append(f"Final output: {_safe_str(report.get('final_output_path'))}")
    lines.append(f"Master audio: {_safe_str(report.get('final_master_audio'))}")
    lines.append(f"FPS: {_safe_int(report.get('fps'), 0)}")
    lines.append(f"Shots: {_safe_int(report.get('shot_count'), 0)}")
    lines.append(f"Expected total duration: {_safe_float(report.get('expected_total_duration'), 0.0):.6f}s")
    lines.append(f"Assembled video duration: {_safe_float(report.get('assembled_video_duration'), 0.0):.6f}s")
    lines.append(f"Final audio duration: {_safe_float(report.get('final_audio_duration'), 0.0):.6f}s")
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    for shot in report.get("shots", []) if isinstance(report.get("shots"), list) else []:
        if not isinstance(shot, dict):
            continue
        lines.append(f"{_safe_str(shot.get('shot_id'))}: {_safe_str(shot.get('status'))}")
        lines.append(f"  timeline: {_safe_float(shot.get('song_start'), 0.0):.3f} -> {_safe_float(shot.get('song_end'), 0.0):.3f} | frames {_safe_int(shot.get('start_frame'), 0)}->{_safe_int(shot.get('end_frame'), 0)} ({_safe_int(shot.get('planned_frames'), 0)} frames)")
        lines.append(f"  planned json={_safe_float(shot.get('planned_duration_from_json'), 0.0):.6f}s | grid={_safe_float(shot.get('planned_duration_frame_grid'), 0.0):.6f}s")
        status = _safe_str(shot.get("status"))
        if status == "needs_regeneration":
            lines.append(f"  input={_safe_float(shot.get('input_clip_duration'), 0.0):.6f}s | planned={_safe_float(shot.get('planned_duration_frame_grid'), 0.0):.6f}s")
            if _safe_str(shot.get("recommended_action")):
                lines.append(f"  recommended action: {_safe_str(shot.get('recommended_action'))}")
            if _safe_bool(shot.get("retime_skipped_to_avoid_slow_motion"), False):
                lines.append("  retime skipped: short clip was not stretched/slowed down")
        else:
            lines.append(f"  input={_safe_float(shot.get('input_clip_duration'), 0.0):.6f}s | speed={_safe_float(shot.get('speed_factor'), 0.0):.8f} | retimed={_safe_float(shot.get('retimed_clip_duration'), 0.0):.6f}s | error={_safe_float(shot.get('duration_error_after_retime'), 0.0):.6f}s")
        if _safe_str(shot.get("input_clip_path")):
            lines.append(f"  input clip: {_safe_str(shot.get('input_clip_path'))}")
        if status != "needs_regeneration" and _safe_str(shot.get("retimed_clip_path")):
            lines.append(f"  retimed clip: {_safe_str(shot.get('retimed_clip_path'))}")
        sw = shot.get("warnings") if isinstance(shot.get("warnings"), list) else []
        for w in sw:
            lines.append(f"  warning: {w}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_ltx_resolution_pair(value: Any, default: tuple[int, int] = (1280, 720)) -> tuple[int, int]:
    try:
        txt = _safe_str(value).lower().replace("×", "x")
        m = re.search(r"(\d{3,5})\s*x\s*(\d{3,5})", txt)
        if m:
            w = max(2, int(m.group(1)))
            h = max(2, int(m.group(2)))
        else:
            w, h = int(default[0]), int(default[1])
    except Exception:
        w, h = int(default[0]), int(default[1])
    if w % 2:
        w += 1
    if h % 2:
        h += 1
    return max(2, w), max(2, h)


def _make_ltx_missing_placeholder_clip(
    *,
    root: Path,
    dst: Path,
    source_image: str = "",
    planned_duration: float,
    planned_frames: int,
    fps: int,
    resolution: Any = "1280x720",
    log_path: Path,
) -> Dict[str, Any]:
    """Create a timeline-safe placeholder for a failed/missing LTX shot.

    This lets the full music video assembly finish even when one LTX clip failed
    after all retry attempts. If a start image exists, the placeholder is a still
    video of that image; otherwise it is a black frame. The review system can
    still recreate the failed shot later.
    """
    ffmpeg = _find_media_binary(root, "FV_FFMPEG", "ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
    except Exception:
        pass
    planned = max(0.05, float(planned_duration or 0.0))
    frames = max(1, int(planned_frames or round(planned * max(1, int(fps or 24)))))
    fps_i = max(1, int(fps or 24))
    w, h = _parse_ltx_resolution_pair(resolution, (1280, 720))
    src = _safe_str(source_image)
    use_image = bool(src and os.path.isfile(src))
    vf_common = f"fps={fps_i},scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    if use_image:
        cmd = [
            ffmpeg, "-y", "-hide_banner",
            "-loop", "1", "-i", src,
            "-t", f"{planned:.6f}",
            "-vf", vf_common,
            "-frames:v", str(frames),
            "-r", str(fps_i),
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst),
        ]
        header = "[assembly] Creating still-image placeholder for missing LTX clip"
    else:
        cmd = [
            ffmpeg, "-y", "-hide_banner",
            "-f", "lavfi", "-i", f"color=c=black:s={w}x{h}:r={fps_i}",
            "-t", f"{planned:.6f}",
            "-frames:v", str(frames),
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(dst),
        ]
        header = "[assembly] Creating black placeholder for missing LTX clip"
    result = _run_ffmpeg_logged(cmd, log_path, header, timeout=1800)
    ok = bool(int(result.get("returncode", -1)) == 0 and dst.is_file() and dst.stat().st_size >= 1024)
    duration = _probe_media_duration_seconds(root, dst) if ok else 0.0
    return {
        "ok": ok,
        "placeholder_clip_path": str(dst),
        "placeholder_source_image": src if use_image else "",
        "placeholder_used_image": use_image,
        "placeholder_duration": round(float(duration), 6) if duration else 0.0,
        "message": "placeholder created" if ok else "placeholder ffmpeg failed or produced no output",
        "ffmpeg_result": result,
    }


def assemble_ltx_music_video(payload: dict) -> dict:
    """Assemble generated LTX shots into one music video locked to the song timeline.

    This does not generate prompts, images, LTX clips, or settings. It only reads an
    existing musicclip_ltx_director_plan.json plus an ltx_full_run folder, retimes
    each video-only shot to the frame-grid duration, concatenates them, and muxes
    the original full song as master audio.
    """
    try:
        if not isinstance(payload, dict):
            return {"ok": False, "message": "LTX assembly payload was not a dictionary."}
        progress_callback = payload.get("progress_callback")
        def _emit(message: str) -> None:
            if callable(progress_callback):
                try:
                    progress_callback(str(message or ""))
                except Exception:
                    pass

        root_raw = _safe_str(payload.get("root_dir"))
        root = Path(root_raw).resolve() if root_raw else _project_root()
        plan_path = Path(_safe_str(payload.get("ltx_director_plan_path"))).expanduser().resolve()
        if not plan_path.is_file():
            return {"ok": False, "message": f"LTX director plan was not found: {plan_path}"}
        director_plan = _read_json_file(str(plan_path))
        safety = _enforce_ltx_start_end_duration_contract(
            director_plan,
            plan_path=plan_path,
            root=root,
            enabled=_payload_ltx_edge_duration_safety_enabled(payload, director_plan),
            refresh_audio_chunks=True,
        )
        if bool(safety.get("changed")):
            existing_warnings = _as_list(director_plan.get("warnings"))
            merged = existing_warnings + [w for w in _as_list(safety.get("warnings")) if w]
            if merged:
                director_plan["warnings"] = merged[:80]
            _write_json_file(plan_path, director_plan)
        shots = _director_plan_shots(director_plan)
        if not shots:
            return {"ok": False, "message": "No shots were found in the LTX director plan."}
        full_run_raw = _safe_str(payload.get("full_run_dir"))
        full_run_dir = Path(full_run_raw).expanduser().resolve() if full_run_raw else _latest_ltx_full_run_dir(plan_path)
        if not full_run_dir.is_dir():
            return {"ok": False, "message": f"LTX full-run folder was not found: {full_run_dir}"}
        final_master_audio = _safe_str(payload.get("final_master_audio")) or _master_audio_from_plan(director_plan)
        if not final_master_audio or not os.path.isfile(final_master_audio):
            return {"ok": False, "message": "Original master audio was not found in the director plan. Expected final_master_audio or audio_path."}
        final_master_audio = str(Path(final_master_audio).resolve())

        fps = _assembly_plan_fps(director_plan, shots, payload)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (full_run_dir / f"assembly_{stamp}").resolve()
        out_dir.mkdir(parents=True, exist_ok=False)
        log_path = out_dir / "assembly_ffmpeg.log.txt"
        concat_list = out_dir / "concat_list.txt"
        video_only = out_dir / "assembled_video_only.mp4"
        final_output = out_dir / "assembled_with_original_audio.mp4"
        report_json = out_dir / "assembly_report.json"
        report_txt = out_dir / "assembly_report.txt"

        warnings: List[str] = []
        report: Dict[str, Any] = {
            "ok": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "director_plan_path": str(plan_path),
            "full_run_dir": str(full_run_dir),
            "output_dir": str(out_dir),
            "final_master_audio": final_master_audio,
            "fps": fps,
            "shot_count": len(shots),
            "expected_total_duration": 0.0,
            "assembled_video_duration": 0.0,
            "final_audio_duration": 0.0,
            "final_output_path": str(final_output),
            "warnings": warnings,
            "shots": [],
        }

        retimed_clips: List[Path] = []
        review_reassemble_mode = bool(
            _safe_bool(payload.get("review_continue_reassemble"), False)
            or _safe_bool(payload.get("use_review_current_clips"), False)
        )
        # Never let Review / Continue / Reassemble silently reuse old assembly
        # timing files.  Those files can be from the pre-review bad video and have
        # the correct duration, so duration checks pass while the wrong visuals are
        # assembled again.  In review mode we always retime the current selected
        # source clip instead.
        reuse_existing_retimed = bool(
            _safe_bool(payload.get("reuse_existing_retimed_clips"), False)
            and not review_reassemble_mode
        )
        review_state_for_assembly: Dict[str, Any] = {}
        if review_reassemble_mode:
            try:
                review_state_for_assembly = _read_ltx_review_state(plan_path)
            except Exception:
                review_state_for_assembly = {}
        total_frames = 0
        try:
            for _shot in shots:
                total_frames += int(_planned_frame_grid_for_shot(_shot, fps).get("planned_frames", 0) or 0)
            report["expected_total_duration"] = round(float(total_frames) / float(fps), 6) if total_frames > 0 else 0.0
            _write_json_file(report_json, report)
            _write_ltx_assembly_text_report(report_txt, report)
        except Exception:
            total_frames = 0
        processed_frames = 0
        frame_tolerance = 1.0 / float(fps)
        _emit("Preparing LTX assembly...")
        for idx, shot in enumerate(shots, start=1):
            sid = _safe_str(shot.get("id")) or f"LTX{idx:02d}"
            stem = _safe_stem(sid)
            grid = _planned_frame_grid_for_shot(shot, fps)
            planned_duration = float(grid["planned_duration_frame_grid"])
            processed_frames += int(grid["planned_frames"])
            shot_warnings: List[str] = []
            preferred = []
            explicit_review_clip = ""
            explicit_review_item = {}
            if review_reassemble_mode and isinstance(review_state_for_assembly.get("shots"), dict):
                explicit_review_item = (review_state_for_assembly.get("shots") or {}).get(sid) or {}
                if isinstance(explicit_review_item, dict):
                    explicit_review_clip = _safe_str(explicit_review_item.get("current_clip_path") or explicit_review_item.get("current_raw_clip_path"))
            if explicit_review_clip and os.path.isfile(explicit_review_clip):
                preferred.append(Path(explicit_review_clip))
                shot_warnings.append("Using explicit reviewed clip from review state.")
            else:
                try:
                    review_paths = _latest_ltx_review_or_fullrun_paths(plan_path, shot, full_run_dir)
                    review_clip = _safe_str(review_paths.get("clip_path"))
                    if review_clip and os.path.isfile(review_clip):
                        preferred.append(Path(review_clip))
                        shot_warnings.append("Using reviewed/current clip pointer.")
                except Exception:
                    pass
            # Avoid reusing old *_ltx_retimed.mp4 files. They may have been created
            # by the previous stretch-to-planned behavior. Assembly will create fresh
            # timing outputs and refuse to stretch any short source clip.
            preferred += [
                full_run_dir / f"{stem}_ltx_trimmed.mp4",
                full_run_dir / f"{stem}_ltx_sync.mp4",
                full_run_dir / f"{stem}_ltx_raw.mp4",
            ]
            report_clip = _full_run_report_clip_for_shot(full_run_dir, sid)
            if report_clip:
                preferred.append(Path(report_clip))
            input_clip = _first_existing_file(preferred)
            retimed_path = out_dir / f"{stem}_ltx_retimed.mp4"
            item: Dict[str, Any] = {
                "shot_id": sid,
                **grid,
                "input_clip_path": input_clip,
                "input_clip_duration": 0.0,
                "speed_factor": 0.0,
                "status": "pending",
                "warnings": shot_warnings,
            }
            if reuse_existing_retimed:
                try:
                    existing_retimed = _latest_existing_retimed_clip(full_run_dir, stem)
                    if existing_retimed:
                        existing_retimed_path = Path(existing_retimed)
                        existing_retimed_duration = _probe_media_duration_seconds(root, existing_retimed_path)
                        existing_error = abs(float(existing_retimed_duration or 0.0) - planned_duration)
                        if existing_retimed_duration > 0.0 and existing_error <= frame_tolerance + 0.01:
                            reused_path = existing_retimed_path
                            try:
                                if existing_retimed_path.resolve() != retimed_path.resolve():
                                    shutil.copy2(str(existing_retimed_path), str(retimed_path))
                                    if retimed_path.is_file() and retimed_path.stat().st_size >= 1024:
                                        reused_path = retimed_path
                            except Exception as copy_exc:
                                shot_warnings.append(f"Could not copy reused retimed clip into the new assembly folder; using it in place: {copy_exc}")
                            reused_duration = _probe_media_duration_seconds(root, reused_path)
                            reused_error = abs(float(reused_duration or existing_retimed_duration or 0.0) - planned_duration)
                            item.update({
                                "input_clip_path": str(existing_retimed_path),
                                "input_clip_duration": round(float(existing_retimed_duration), 6),
                                "speed_factor": 1.0,
                                "status": "ok",
                                "planned_duration": round(float(planned_duration), 6),
                                "raw_output_shorter_than_planned": False,
                                "short_raw_padded_to_planned": False,
                                "freeze_last_frame_padding_applied": False,
                                "short_raw_padding_seconds": 0.0,
                                "retime_skipped_to_avoid_slow_motion": False,
                                "recommended_action": "",
                                "reused_existing_retimed_clip": True,
                                "retimed_clip_path": str(reused_path),
                                "retimed_clip_duration": round(float(reused_duration or existing_retimed_duration), 6),
                                "duration_error_after_retime": round(float(reused_error), 6),
                            })
                            shot_warnings.append("Reused existing retimed clip from a previous assembly attempt.")
                            retimed_clips.append(reused_path)
                            report["shots"].append(item)
                            _write_json_file(report_json, report)
                            _write_ltx_assembly_text_report(report_txt, report)
                            continue
                        shot_warnings.append("Existing retimed clip found, but its duration does not match the current plan; rebuilding this shot timing.")
                except Exception as reuse_exc:
                    shot_warnings.append(f"Could not check existing retimed clip; rebuilding this shot timing: {reuse_exc}")

            if not input_clip:
                placeholder_source = ""
                for cand in (
                    _safe_str(explicit_review_item.get("current_start_image_path")) if isinstance(explicit_review_item, dict) else "",
                    str(full_run_dir / f"{stem}_start.png"),
                    str(full_run_dir / f"{stem}_start.jpg"),
                    str(full_run_dir / f"{stem}_start.jpeg"),
                    str(full_run_dir / f"{stem}_review_start_used.png"),
                    str(full_run_dir / f"{stem}_review_start.png"),
                ):
                    if cand and os.path.isfile(cand):
                        placeholder_source = cand
                        break
                item["status"] = "placeholder"
                item["missing_original_clip"] = True
                item["can_recreate_later"] = True
                shot_warnings.append("Missing generated LTX clip; assembly inserted a placeholder so the full video can finish. Recreate this shot later from Review.")
                warnings.append(f"{sid}: missing generated clip; placeholder inserted")
                placeholder = _make_ltx_missing_placeholder_clip(
                    root=root,
                    dst=retimed_path,
                    source_image=placeholder_source,
                    planned_duration=planned_duration,
                    planned_frames=int(grid["planned_frames"]),
                    fps=fps,
                    resolution=payload.get("resolution") or shot.get("resolution") or director_plan.get("resolution") or "1280x720",
                    log_path=log_path,
                )
                item["placeholder_clip_path"] = _safe_str(placeholder.get("placeholder_clip_path"))
                item["placeholder_source_image"] = _safe_str(placeholder.get("placeholder_source_image"))
                item["placeholder_used_image"] = _safe_bool(placeholder.get("placeholder_used_image"), False)
                item["input_clip_path"] = _safe_str(placeholder.get("placeholder_clip_path"))
                item["retimed_clip_path"] = _safe_str(placeholder.get("placeholder_clip_path"))
                item["input_clip_duration"] = _safe_float(placeholder.get("placeholder_duration"), planned_duration)
                item["retimed_clip_duration"] = _safe_float(placeholder.get("placeholder_duration"), planned_duration)
                item["speed_factor"] = 1.0
                item["duration_error_after_retime"] = abs(float(item["retimed_clip_duration"] or 0.0) - planned_duration)
                if not _safe_bool(placeholder.get("ok"), False):
                    item["status"] = "missing"
                    item["placeholder_error"] = _safe_str(placeholder.get("message"), "placeholder failed")
                    shot_warnings.append("Placeholder creation failed; assembly cannot continue for this shot.")
                    report["shots"].append(item)
                    _write_json_file(report_json, report)
                    _write_ltx_assembly_text_report(report_txt, report)
                    return {"ok": False, "message": f"Missing generated clip for {sid} and placeholder creation failed. See report: {report_json}", "output_dir": str(out_dir), "report_json": str(report_json)}
                retimed_clips.append(Path(_safe_str(placeholder.get("placeholder_clip_path"))))
                report["shots"].append(item)
                _write_json_file(report_json, report)
                _write_ltx_assembly_text_report(report_txt, report)
                continue

            actual_input_duration = _probe_media_duration_seconds(root, Path(input_clip))
            item["input_clip_duration"] = round(float(actual_input_duration), 6) if actual_input_duration else 0.0
            item["planned_duration"] = round(float(planned_duration), 6)
            if actual_input_duration > 0.0 and actual_input_duration < planned_duration - float(DURATION_TOLERANCE_SECONDS):
                speed_factor = max(0.001, float(actual_input_duration) / max(0.001, float(planned_duration)))
                item.update({
                    "speed_factor": 1.0,
                    "natural_duration_factor": round(float(speed_factor), 8),
                    "raw_output_shorter_than_planned": True,
                    "retime_skipped_to_avoid_slow_motion": True,
                    "recommended_action": "regenerate shot",
                    "duration_error_before_regeneration": round(float(planned_duration - actual_input_duration), 6),
                })
                edge_shot = bool(idx == 1 or idx == len(shots))
                safety_enabled = _payload_ltx_edge_duration_safety_enabled(payload, director_plan)
                if edge_shot and safety_enabled:
                    _emit(f"{sid} is shorter than the updated LTX edge-duration plan. Recreating it instead of blocking assembly...")
                    regen = _auto_regenerate_ltx_review_clip_for_assembly(
                        root=root,
                        plan_path=plan_path,
                        director_plan=director_plan,
                        shot=shot,
                        full_run_dir=full_run_dir,
                        progress_callback=progress_callback,
                    )
                    if bool(regen.get("ok")) and _safe_str(regen.get("clip_path")) and os.path.isfile(_safe_str(regen.get("clip_path"))):
                        input_clip = _safe_str(regen.get("clip_path"))
                        item["input_clip_path"] = input_clip
                        actual_input_duration = _probe_media_duration_seconds(root, Path(input_clip))
                        item["input_clip_duration"] = round(float(actual_input_duration), 6) if actual_input_duration else 0.0
                        item["auto_recreated_for_duration_safety"] = True
                        shot_warnings.append("Automatically recreated because the old clip was shorter than the first/last duration safety contract.")
                        warnings.append(f"{sid}: auto-recreated short edge clip")
                    else:
                        shot_warnings.append(_safe_str(regen.get("message"), f"{sid} is shorter than planned and automatic recreate failed."))
                        warnings.append(f"{sid}: auto-regeneration failed for short edge clip")
                else:
                    shot_warnings.append(f"{sid} is shorter than planned; regenerate this shot instead of padding or slowing it down.")
                    warnings.append(f"{sid}: short input needs regeneration")

            _emit(f"Checking clip length {idx} / {len(shots)}: {sid}")
            retime = _retime_ltx_clip_to_duration(
                root=root,
                src=Path(input_clip),
                dst=retimed_path,
                fps=fps,
                planned_duration=planned_duration,
                planned_frames=int(grid["planned_frames"]),
                log_path=log_path,
            )
            item["input_clip_duration"] = round(float(retime.get("actual_duration") or actual_input_duration or 0.0), 6)
            item["speed_factor"] = round(float(retime.get("speed_factor") or 0.0), 8)
            item["raw_output_shorter_than_planned"] = _safe_bool(retime.get("raw_output_shorter_than_planned"), False)
            item["short_raw_padded_to_planned"] = _safe_bool(retime.get("short_raw_padded_to_planned"), False)
            item["freeze_last_frame_padding_applied"] = _safe_bool(retime.get("freeze_last_frame_padding_applied"), False)
            item["short_raw_padding_seconds"] = _safe_float(retime.get("pad_seconds"), 0.0)
            item["retime_skipped_to_avoid_slow_motion"] = _safe_bool(retime.get("retime_skipped_to_avoid_slow_motion"), False)
            item["recommended_action"] = _safe_str(retime.get("recommended_action"))
            if _safe_bool(retime.get("raw_output_shorter_than_planned"), False) and not _safe_bool(retime.get("short_raw_padded_to_planned"), False):
                item.update({
                    "status": "needs_regeneration",
                    "assembly_blocked": True,
                    "recommended_action": _safe_str(retime.get("recommended_action"), "regenerate shot") or "regenerate shot",
                })
                shot_warnings.append(f"{sid} is shorter than planned. Regenerate this shot from the Review tab.")
                warnings.append(f"{sid}: needs regeneration")
                report["blocked_by_needs_regeneration"] = True
                report["needs_regeneration_count"] = _safe_int(report.get("needs_regeneration_count"), 0) + 1
                placeholder_source = ""
                for cand in (
                    _safe_str(explicit_review_item.get("current_start_image_path")) if isinstance(explicit_review_item, dict) else "",
                    str(full_run_dir / f"{stem}_start.png"),
                    str(full_run_dir / f"{stem}_start.jpg"),
                    str(full_run_dir / f"{stem}_start.jpeg"),
                ):
                    if cand and os.path.isfile(cand):
                        placeholder_source = cand
                        break
                placeholder = _make_ltx_missing_placeholder_clip(
                    root=root,
                    dst=retimed_path,
                    source_image=placeholder_source,
                    planned_duration=planned_duration,
                    planned_frames=int(grid["planned_frames"]),
                    fps=fps,
                    resolution=payload.get("resolution") or shot.get("resolution") or director_plan.get("resolution") or "1280x720",
                    log_path=log_path,
                )
                if _safe_bool(placeholder.get("ok"), False):
                    item["status"] = "placeholder_needs_regeneration"
                    item["assembly_blocked"] = False
                    item["can_recreate_later"] = True
                    item["placeholder_clip_path"] = _safe_str(placeholder.get("placeholder_clip_path"))
                    item["placeholder_source_image"] = _safe_str(placeholder.get("placeholder_source_image"))
                    item["retimed_clip_path"] = _safe_str(placeholder.get("placeholder_clip_path"))
                    item["retimed_clip_duration"] = _safe_float(placeholder.get("placeholder_duration"), planned_duration)
                    retimed_clips.append(Path(_safe_str(placeholder.get("placeholder_clip_path"))))
                    shot_warnings.append("Inserted placeholder instead of stopping assembly; recreate this shot later.")
                    report["warnings"] = warnings
                    report["shots"].append(item)
                    _write_json_file(report_json, report)
                    _write_ltx_assembly_text_report(report_txt, report)
                    continue
                report["warnings"] = warnings
                report["shots"].append(item)
                _write_json_file(report_json, report)
                _write_ltx_assembly_text_report(report_txt, report)
                message = f"Assembly stopped: {sid} needs regeneration and placeholder creation failed. See report: {report_json}"
                _emit(message)
                return {"ok": False, "status": "needs_regeneration", "message": message, "output_dir": str(out_dir), "report_json": str(report_json), "needs_regeneration_shots": [sid]}
            if _safe_bool(retime.get("short_raw_padded_to_planned"), False):
                item["short_raw_padded_to_planned"] = True
                item["freeze_last_frame_padding_applied"] = True
                item["short_raw_padding_seconds"] = _safe_float(retime.get("pad_seconds"), 0.0)
                item["recommended_action"] = _safe_str(retime.get("recommended_action"), "regenerate shot")
                shot_warnings.append("Short clip still needs regeneration; no padding or slow-motion retime was used.")
            if not _safe_bool(retime.get("short_raw_padded_to_planned"), False) and abs(float(item["speed_factor"] or 1.0) - 1.0) > 0.15:
                shot_warnings.append("Severe retime: speed factor differs by more than 15%.")
                warnings.append(f"{sid}: severe retime speed factor {item['speed_factor']}")
            elif not _safe_bool(retime.get("short_raw_padded_to_planned"), False) and abs(float(item["speed_factor"] or 1.0) - 1.0) > 0.08:
                shot_warnings.append("Large retime: speed factor differs by more than 8%.")
                warnings.append(f"{sid}: large retime speed factor {item['speed_factor']}")
            if not bool(retime.get("ok")):
                item["status"] = "failed"
                item["retimed_clip_path"] = str(retimed_path)
                item["retimed_clip_duration"] = 0.0
                item["duration_error_after_retime"] = 0.0
                shot_warnings.append(_safe_str(retime.get("message"), "Retiming failed."))
                warnings.append(f"{sid}: retiming failed")
                report["warnings"] = warnings
                report["shots"].append(item)
                _write_json_file(report_json, report)
                _write_ltx_assembly_text_report(report_txt, report)
                return {"ok": False, "message": f"Retiming failed for {sid}. See report: {report_json}", "output_dir": str(out_dir), "report_json": str(report_json)}
            item["retimed_clip_path"] = str(retimed_path)
            retimed_duration = _probe_media_duration_seconds(root, retimed_path)
            item["retimed_clip_duration"] = round(float(retimed_duration), 6)
            err = abs(float(retimed_duration) - planned_duration)
            item["duration_error_after_retime"] = round(float(err), 6)
            if err > frame_tolerance + 0.002:
                shot_warnings.append("Retimed duration is still off by more than one frame.")
                warnings.append(f"{sid}: retimed duration error {err:.6f}s")
            item["status"] = "ok"
            retimed_clips.append(retimed_path)
            report["shots"].append(item)
            _write_json_file(report_json, report)
            _write_ltx_assembly_text_report(report_txt, report)

        if total_frames <= 0:
            total_frames = int(processed_frames)
        expected_total_duration = total_frames / float(fps)
        report["expected_total_duration"] = round(float(expected_total_duration), 6)
        _emit("Concatenating retimed video clips...")
        concat_result = _concat_retimed_clips(root, retimed_clips, concat_list, video_only, fps, log_path)
        if not bool(concat_result.get("ok")):
            warnings.append("Video-only concat failed.")
            report["warnings"] = warnings
            _write_json_file(report_json, report)
            _write_ltx_assembly_text_report(report_txt, report)
            return {"ok": False, "message": f"Video concat failed. See log: {log_path}", "output_dir": str(out_dir), "report_json": str(report_json)}
        assembled_duration = _probe_media_duration_seconds(root, video_only)
        report["assembled_video_duration"] = round(float(assembled_duration), 6)
        if abs(float(assembled_duration) - expected_total_duration) > frame_tolerance + 0.01:
            warnings.append(f"Assembled video differs from planned total by {abs(float(assembled_duration) - expected_total_duration):.6f}s.")

        audio_duration = _probe_media_duration_seconds(root, Path(final_master_audio))
        report["final_audio_duration"] = round(float(audio_duration), 6)
        if audio_duration > 0 and abs(float(audio_duration) - float(assembled_duration)) > max(frame_tolerance + 0.01, 0.05):
            warnings.append(f"Original audio duration differs from assembled video by {abs(float(audio_duration) - float(assembled_duration)):.6f}s.")

        _emit("Adding original master audio...")
        mux_result = _mux_original_audio(root, video_only, Path(final_master_audio), final_output, log_path)
        if not bool(mux_result.get("ok")):
            warnings.append("Muxing original master audio failed.")
            report["warnings"] = warnings
            _write_json_file(report_json, report)
            _write_ltx_assembly_text_report(report_txt, report)
            return {"ok": False, "message": f"Audio mux failed. See log: {log_path}", "output_dir": str(out_dir), "report_json": str(report_json)}

        report["ok"] = True
        report["warnings"] = warnings
        report["final_output_path"] = str(final_output)
        _write_json_file(report_json, report)
        _write_ltx_assembly_text_report(report_txt, report)
        msg = f"Final music video saved: {final_output}"
        _emit(msg)
        report["message"] = msg
        report["report_json"] = str(report_json)
        report["report_txt"] = str(report_txt)
        report["video_only_path"] = str(video_only)
        report["ffmpeg_log_path"] = str(log_path)
        return report
    except Exception as exc:
        return {"ok": False, "message": f"LTX assembly failed: {exc}"}


# --------------------------- queued/headless runner ---------------------------
def run_ltx_queue_payload(payload_path: str) -> int:
    """Headless entrypoint for FrameVision Queue jobs.

    The Music Clip Creator UI writes a small JSON payload, then Queue/worker runs
    this function in a separate Python process through the normal tools_ffmpeg
    runner.  Keep this stdlib-only so it stays safe to call from worker.py.
    """
    try:
        payload_file = Path(_safe_str(payload_path)).expanduser().resolve()
        if not payload_file.is_file():
            print(f"ERROR: LTX queue payload not found: {payload_file}", flush=True)
            return 2
        payload = _read_json_file(str(payload_file))
        if not isinstance(payload, dict):
            print(f"ERROR: LTX queue payload is not a dictionary: {payload_file}", flush=True)
            return 2
    except Exception as exc:
        print(f"ERROR: failed to read LTX queue payload: {exc}", flush=True)
        return 2

    queue_report_path = None
    try:
        queue_report_raw = _safe_str(payload.get("queue_report_path"))
        if queue_report_raw:
            queue_report_path = Path(queue_report_raw).expanduser().resolve()
    except Exception:
        queue_report_path = None

    progress_state = {"last_pct": -1}

    def _queue_write_report(data: Dict[str, Any]) -> None:
        if queue_report_path is None:
            return
        try:
            queue_report_path.parent.mkdir(parents=True, exist_ok=True)
            _write_json_file(queue_report_path, data)
        except Exception:
            pass

    def _emit_progress(message: str) -> None:
        msg = _safe_str(message)
        if msg:
            print(msg, flush=True)
        # The worker already parses "N / total" and "N%" from stdout, but emit
        # a coarse percent too so long image-generation/LTX stages visibly move.
        try:
            m = re.search(r"(?i)generating\s+shot\s+(\d+)\s*/\s*(\d+)", msg)
            if not m:
                m = re.search(r"(?i)(?:finished|skipped|failed)\s+LTX(\d+)\b", msg)
                # This fallback intentionally stays conservative because not all
                # custom shot IDs end in a number.
            if m and m.lastindex and m.lastindex >= 2:
                cur = max(0, _safe_int(m.group(1), 0) - 1)
                total = max(1, _safe_int(m.group(2), 1))
                pct = int(max(0, min(95, (float(cur) / float(total)) * 95.0)))
                if pct != progress_state.get("last_pct"):
                    progress_state["last_pct"] = pct
                    print(f"{pct}%", flush=True)
        except Exception:
            pass

    try:
        payload = dict(payload)
        payload["progress_callback"] = _emit_progress
        print("0%", flush=True)
        print("Music Clip LTX queue job: starting full LTX run", flush=True)
        result = run_all_ltx_director_shots(payload)
        result = result if isinstance(result, dict) else {"ok": False, "message": "LTX run returned an invalid result."}
        _queue_write_report({"stage": "generation", "result": result})
        if not bool(result.get("ok")):
            print(_safe_str(result.get("message"), "LTX generation failed."), flush=True)
            return 1

        assemble_after = _safe_bool(payload.get("assemble_after"), False)
        if assemble_after:
            print("96%", flush=True)
            print("Music Clip LTX queue job: assembling final video", flush=True)
            assembly_payload = dict(payload)
            assembly_payload["progress_callback"] = _emit_progress
            if _safe_str(result.get("output_dir")):
                assembly_payload["full_run_dir"] = _safe_str(result.get("output_dir"))
            assembly_result = assemble_ltx_music_video(assembly_payload)
            assembly_result = assembly_result if isinstance(assembly_result, dict) else {"ok": False, "message": "LTX assembly returned an invalid result."}
            final_output = _safe_str(assembly_result.get("final_output_path"))
            queue_final_output = _safe_str(payload.get("queue_final_output"))
            if bool(assembly_result.get("ok")) and final_output and queue_final_output:
                try:
                    src = Path(final_output).expanduser().resolve()
                    dst = Path(queue_final_output).expanduser().resolve()
                    if src.is_file():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        if src.resolve() != dst.resolve():
                            shutil.copy2(str(src), str(dst))
                        assembly_result["queue_final_output"] = str(dst)
                        print(f"Queue final output copied: {dst}", flush=True)
                except Exception as copy_exc:
                    assembly_result.setdefault("warnings", [])
                    try:
                        assembly_result["warnings"].append(f"Could not copy queue final output: {copy_exc}")
                    except Exception:
                        pass
                    print(f"Warning: could not copy queue final output: {copy_exc}", flush=True)
            _queue_write_report({"stage": "assembly", "generation_result": result, "assembly_result": assembly_result})
            if not bool(assembly_result.get("ok")):
                print(_safe_str(assembly_result.get("message"), "LTX assembly failed."), flush=True)
                return 1
            print("100%", flush=True)
            print(_safe_str(assembly_result.get("message"), "Music Clip LTX queue job finished."), flush=True)
            return 0

        print("100%", flush=True)
        print(_safe_str(result.get("message"), "Music Clip LTX queue job finished."), flush=True)
        return 0
    except KeyboardInterrupt:
        print("Music Clip LTX queue job cancelled.", flush=True)
        return 130
    except Exception as exc:
        print(f"ERROR: Music Clip LTX queue job failed: {exc}", flush=True)
        return 1

