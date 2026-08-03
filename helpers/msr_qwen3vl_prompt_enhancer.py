from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError("Enhancer input must be a JSON object.")
    return data


def _clean(text: Any) -> str:
    return " ".join(str(text or "").replace("\x00", " ").split()).strip()


def _strip_wrappers(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_descriptions(raw: str) -> Tuple[str, str]:
    """Parse two factual descriptions without ever accepting schema/JSON dumps."""
    text = _strip_wrappers(raw)
    if not text:
        return "", ""
    # Reject the schema-shaped failure mode from the old implementation.
    low = text.lower()
    if "$schema" in low or '"$id"' in low or "raw.githubusercontent.com" in low:
        return "", ""
    bg = ""
    refs = ""
    m_bg = re.search(r"(?:^|\n)\s*BACKGROUND\s*:\s*(.+?)(?=\n\s*REFERENCES?\s*:|\Z)", text, flags=re.I | re.S)
    m_ref = re.search(r"(?:^|\n)\s*REFERENCES?\s*:\s*(.+?)\s*\Z", text, flags=re.I | re.S)
    if m_bg:
        bg = _clean(m_bg.group(1))
    if m_ref:
        refs = _clean(m_ref.group(1))
    return bg[:900], refs[:1400]


def _clean_enhancement(raw: str) -> str:
    text = _strip_wrappers(raw)
    if not text:
        return ""
    low = text.lower()
    if "$schema" in low or '"$id"' in low or "raw.githubusercontent.com" in low:
        return ""
    # Remove accidental labels; the caller adds the exact #!PROMPT! header itself.
    text = re.sub(r"^\s*#!PROMPT!\s*:\s*.*?(?:\n+|$)", "", text, flags=re.I | re.S)
    text = re.sub(r"^\s*(?:ENHANCED PROMPT|FINAL PROMPT|OUTPUT)\s*:\s*", "", text, flags=re.I)
    text = _clean(text)
    # The enhanced paragraph must use concrete visual facts, not vague continuity placeholders.
    vague = (
        r"\bthe same coherent (?:music[- ]video )?world\b",
        r"\bthe same (?:main )?(?:performer|person|woman|man|character|subject|singer|location|background|scene)\b",
        r"\brecurring (?:member|character|performer)\b",
        r"\bconsistent identity\b",
    )
    for pattern in vague:
        text = re.sub(pattern, "", text, flags=re.I)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,;:-")
    return text[:2600]


def _generate(processor, model, device: str, images, instruction: str, max_new_tokens: int) -> str:
    content: List[Dict[str, Any]] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": instruction})
    messages = [{"role": "user", "content": content}]
    chat_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    kwargs: Dict[str, Any] = {"text": [chat_text], "return_tensors": "pt"}
    if images:
        kwargs["images"] = images
    inputs = processor(**kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.06,
        no_repeat_ngram_size=4,
        return_dict_in_generate=True,
    )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    new_ids = sequences[:, prompt_len:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: msr_qwen3vl_prompt_enhancer.py INPUT.json OUTPUT.json", file=sys.stderr)
        return 2
    input_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    payload = _load_json(input_path)
    root = Path(payload.get("root_dir") or ".").resolve()
    model_dir = Path(payload.get("model_dir") or (root / "models" / "describe" / "default" / "qwen3vl2b")).resolve()
    jobs = payload.get("jobs") or []
    if not model_dir.is_dir():
        raise RuntimeError(f"Qwen3-VL model folder not found: {model_dir}")
    if not isinstance(jobs, list):
        raise RuntimeError("jobs must be a list")

    import torch
    from PIL import Image
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except Exception:
        from transformers import AutoModelForVision2Seq as ModelClass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, use_fast=True)
    model = ModelClass.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True, torch_dtype=dtype).to(device)
    model.eval()

    results: List[Dict[str, Any]] = []
    previous_enhancements: List[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        shot_id = _clean(job.get("shot_id")) or f"shot_{len(results)+1}"
        image_paths = [Path(x).resolve() for x in (job.get("image_paths") or []) if x and Path(x).is_file()]
        images = [Image.open(str(path)).convert("RGB") for path in image_paths]
        original_prompt = str(job.get("original_prompt") or "").strip()
        vocals_active = bool(job.get("vocals_active"))

        # PASS 1: only inspect the images. Never rewrite or enhance the user's prompt here.
        pass1_instruction = """Inspect the supplied images and save two factual descriptions for a later prompt-writing pass.

IMAGE ORDER
Image 1 is the exact background plate. Any following images are reference images for people, characters, creatures, objects, vehicles, clothing, instruments, or visual style.

Return exactly two plain-text sections and nothing else:
BACKGROUND: a concrete factual description of the visible location, composition, lighting, colors, architecture and atmosphere.
REFERENCES: a concrete factual description of every important visible subject/object, including appearance, clothing, role, held objects and distinguishing visual details.

Do not write a video prompt. Do not invent action, camera movement, cuts, story, names or facts not visible in the images. Do not output JSON, markdown, schema, commentary or analysis."""
        with torch.inference_mode():
            pass1_raw = _generate(processor, model, device, images, pass1_instruction, 260)
        background_description, reference_description = _parse_descriptions(pass1_raw)

        previous_note = ""
        if previous_enhancements:
            previous_note = "\n\nPREVIOUS SHOT PROMPTS — do not copy their camera sequence or opening pattern:\n" + "\n".join(
                f"- {item[:320]}" for item in previous_enhancements[-3:]
            )

        # PASS 2: text-only composition from exactly the three saved inputs.
        pass2_instruction = f"""Write one rich chronological cinematic video prompt using only these three saved inputs.

ORIGINAL USER PROMPT — preserve its intent and facts:
{original_prompt or '[empty]'}

BACKGROUND DESCRIPTION:
{background_description or '[description unavailable]'}

REFERENCE DESCRIPTION(S):
{reference_description or '[description unavailable]'}

Create a mini visual storyline across the clip rather than one repeated move. Ground every subject, object, outfit, location and lighting detail in the three inputs. Describe a clear opening composition, meaningful visible action for the important subjects or objects, natural expression/body motion when relevant, suitable camera movement, one or more later framing changes or cuts when they improve the scene, environmental and lighting behavior, and a clear later development toward the end of the clip. Camera choices must fit this specific scene; never default to starting at the feet and pushing up to the face. Avoid repeating the camera grammar used in previous shots.

If vocals are active ({'yes' if vocals_active else 'no'}) and the supplied inputs establish a visible singer, describe active face-readable singing and natural performance. Otherwise do not force singing. Do not invent unsupported people, roles, props, vehicles, creatures, clothing, locations or weather.

Never use vague phrases such as 'the same performer', 'the same character', 'the same location', 'the same coherent world', 'recurring member' or 'consistent identity'. State the concrete visible description instead.

Return only the enhanced cinematic paragraph. Do not repeat the original prompt. Do not add #!PROMPT!:. Do not output labels, JSON, schema, markdown, notes or explanations.{previous_note}"""
        with torch.inference_mode():
            pass2_raw = _generate(processor, model, device, [], pass2_instruction, 430)
        enhanced = _clean_enhancement(pass2_raw)
        if enhanced:
            previous_enhancements.append(enhanced)

        combined = f"#!PROMPT!: {original_prompt}\n{enhanced}".strip()
        results.append({
            "shot_id": shot_id,
            "original_prompt": original_prompt,
            "background_description": background_description,
            "reference_description": reference_description,
            "enhanced_script": enhanced,
            "combined_prompt": combined,
            "pass1_raw_output": pass1_raw,
            "pass2_raw_output": pass2_raw,
            "image_paths": [str(p) for p in image_paths],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"ok": True, "pipeline": "two_pass_grounded", "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
