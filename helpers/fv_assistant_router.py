# -*- coding: utf-8 -*-
"""FrameVision Assistant router: text chat -> deterministic FrameVision jobs.

Current scope in this file:
  * text-to-image routing for supported FrameVision image models
  * attached-image edit routing
  * guided edit flow:
      - ask what should be changed first when the edit instruction is missing
      - then ask for Flux Klein or HiDream when the model is not specified
  * one-line smart edit flow such as:
      - "edit this image with flux klein: add sunglasses"
      - "edit this image with hidream: replace the background with a snowy forest"
  * model-specific edit queueing for:
      - Flux Klein (small/local edits)
      - HiDream (bigger edits / reference-image style edits)
  * prompt cleanup for edit commands, including stripping model words and size
    tokens out of the actual edit prompt before queueing
  * HiDream safe-size handling for known-good output resolutions
  * queue-only output: this router does not generate images itself, it only
    creates safe queue jobs for FrameVision to run
  * JSON-backed defaults / model/tool metadata loaded from
    scripts/fv_assistant_image_models.json
    (historical filename; it is now the assistant registry for image models,
    upscalers, music tools, and future assistant-callable FrameVision tools)
  * returns structured AssistantRouteResult data so the chat UI can:
      - show follow-up questions
      - track queued jobs
      - show the finished result in chat later


Notes:
  * This router is deliberately deterministic.
  * Normal conversation can still go to the local LLM.
  * Image requests are routed here first so the assistant does not answer
    "I cannot create images" when FrameVision can queue the job.
  * SeedVR2 upscale, Ace-Step music, and chat UI context-menu behavior are
    handled mostly in llama_chat_ui.py for now unless they are moved here in a
    later cleanup.
"""
from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple


@dataclass
class AssistantRouteResult:
    handled: bool
    message: str = ""
    queued: bool = False
    model: str = ""
    model_label: str = ""
    prompt: str = ""
    width: int = 0
    height: int = 0
    queued_at: float = 0.0
    mode: str = ""
    input_images: Tuple[str, ...] = ()
    output_path: str = ""


def _find_root() -> Path:
    env_root = os.environ.get("FRAMEVISION_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    here = Path(__file__).resolve()
    # helpers/fv_assistant_router.py -> app root
    if here.parent.name.lower() == "helpers":
        return here.parent.parent.resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "presets").exists() or (parent / "models").exists():
            return parent.resolve()
    return Path.cwd().resolve()


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8") or "")
    except Exception:
        pass
    return default


def _save_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _default_registry() -> Dict[str, Any]:
    return {
        "version": 1,
        "default_image_model": "zimage_gguf",
        "default_width": 1376,
        "default_height": 768,
        "max_custom_width": 2048,
        "max_custom_height": 2048,
        "max_custom_area": 2048 * 2048,
        "snap_multiple": 64,
        "models": {
            "zimage_gguf": {
                "label": "Z-Image GGUF",
                "aliases": ["z-image", "zimage", "z image", "z-image gguf", "zimage gguf", "z image gguf"],
                "folder": "models/Z-Image-Turbo GGUF",
                "queue_type": "txt2img",
                "engine": "zimage_gguf",
                "output_dir": "output/photo/txt2img",
                "default_steps": 8,
                "default_cfg": 0.0,
                "default_sampler": "euler",
                "filename_template": "z_img_{seed}_{idx:03d}.png",
            },
            "lens": {
                "label": "Lens Turbo U4",
                "aliases": ["lens", "lens turbo", "lens turbo u4"],
                "folder": "models/lens",
                "queue_type": "lens_turbo_u4",
                "output_dir": "output/lens_turbo_u4",
                "repo_id": "WaveCut/Lens-Turbo-SDNQ-uint4-static",
                "default_steps": 4,
                "default_cfg": 0.0,
            },
            "chroma": {
                "label": "Chroma",
                "aliases": ["chroma", "spark chroma", "spark.chroma"],
                "folder": "models/chroma/SPARK.Chroma_v1",
                "queue_type": "chroma",
                "output_dir": "output/images/chroma",
                "default_steps": 30,
                "default_cfg": 3.0,
            },
            "flux_klein": {
                "label": "Flux Klein",
                "aliases": ["flux klein", "flux-klein", "flux klein gguf", "klein", "klein 4b", "flux 2 klein"],
                "folder": "models/klein4b_gguf",
                "queue_type": "flux_klein",
                "output_dir": "output/edits/flux_klein",
                "default_steps": 4,
                "default_cfg": 1.0,
            },
            "hidream": {
                "label": "HiDream",
                "aliases": ["hidream", "hi dream", "hidream dev", "hidream image", "hidream studio"],
                "folder": "models/hidream_bf16",
                "queue_type": "hidream",
                "output_dir": "output/hidream",
                "default_steps": 28,
                "default_cfg": 0.0,
            },
        },
        "video_models": {
            "ltx23": {
                "label": "LTX 2.3",
                "aliases": ["ltx", "ltx 2.3", "ltx23", "ltx distilled", "distilled 1.1", "normal ltx"],
                "folder": "models/ltx23/distilled-1.1",
                "default_checkpoint": "models/ltx23/distilled-1.1/ltx-2.3-22b-distilled-1.1.safetensors",
                "spatial_upsampler": "models/ltx23/spatial_upsampler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
                "gemma_root": "models/ltx23/text_encoder/lightricks_gemma_original",
                "cli": "helpers/ltx23_vram_lab_cli.py",
                "env_win": "environments/.ltx23/python.exe",
                "env_posix": "environments/.ltx23/bin/python",
                "output_dir": "output/video/ltx23/chat",
                "defaults": {
                    "pipeline": "two_stages",
                    "vram_lab": "safe",
                    "vram_profile": "auto",
                    "steps": 8,
                    "cfg": 2.0,
                    "stg": 0.0,
                    "rescale": 0.7,
                    "shift": 5.0,
                    "fps": 24,
                    "frames": 121,
                    "resolution": "704p",
                    "aspect": "16:9"
                }
            }
        },
    }


class FrameVisionAssistantRouter:
    def __init__(self, root: Optional[Path | str] = None):
        self.root = Path(root).resolve() if root else _find_root()
        self.registry_path = self.root / "scripts" / "fv_assistant_image_models.json"
        self.state_path = self.root / "temp" / "fv_assistant_state.json"
        self.registry = _load_json(self.registry_path, _default_registry())

    # ------------------------- wizard undo / trigger words -------------------------
    def _is_cancel_command(self, text: str) -> bool:
        return str(text or "").lower().strip() in {"cancel", "cancel it", "stop", "never mind", "nevermind"}

    def _is_undo_command(self, text: str) -> bool:
        low = str(text or "").lower().strip()
        return low in {"undo", "go back", "back", "one step back", "previous", "previous step"}

    def _state_without_undo(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in dict(state or {}).items() if k not in {"_undo_stack", "_skip_undo"}}

    def _pending_question_for_state(self, state: Dict[str, Any]) -> str:
        intent = str(state.get("pending_intent") or "")
        waiting = str(state.get("waiting_for") or "")
        if intent == "ltx_video":
            if waiting == "model_choice":
                return "Which LTX model should I use? Use `normal LTX`, `distilled 1.1`, or mention the exact `.safetensors` filename."
            if waiting == "mode_choice":
                return "Text to video, image to video, or continue a video?"
            if waiting == "input_media":
                mode = str(state.get("video_mode") or "")
                if mode == "image":
                    return "Please upload the image you want to animate."
                if mode == "continue":
                    return "Please upload the video you want to continue."
                return "Text to video, image to video, or continue a video?"
            if waiting == "prompt":
                return self._ltx_prompt_question()
            if waiting == "resolution_aspect":
                return "What size do you want? Example: `704p 16:9`, `480p portrait`, or `1088p square`."
            if waiting == "duration_fps":
                warning = str(state.get("shape_warning") or "").strip()
                msg = "How many frames and FPS? You can also say something like `10 seconds at 24fps`."
                return (warning + "\n\n" + msg) if warning else msg
            if waiting == "output_name":
                return "Name for the output file? Optional. Type `default` for a timestamped name."
            if waiting == "confirm":
                return self._ltx_summary_message(state)
            return "Which LTX model should I use?"
        if intent == "image_edit":
            if waiting == "edit_instruction":
                model_id = str(state.get("model_id") or "")
                model_note = " I will use Flux Klein after you give the edit." if model_id == "flux_klein" else " I will use HiDream after you give the edit." if model_id == "hidream" else ""
                return "What should I change in the attached image?" + model_note + "\n\nExample: `replace the background with a beach festival scene`."
            if waiting == "edit_model_choice":
                return "Use Flux Klein or HiDream?\n\n- Flux Klein = small edits / add or remove something in the existing image\n- HiDream = bigger edits / replace backgrounds / reference-image rebuilds."
            return "Please choose `Flux Klein` or `HiDream`."
        if intent == "text_to_image":
            if waiting == "image_prompt":
                return "What should be in the image? Example: `a bee on a flower`."
            if waiting == "workflow_choice":
                return "Default or custom?\n\nDefault uses Z-Image GGUF at 1376×768. Custom can use Z-Image GGUF, Lens, Chroma, Flux Klein, or HiDream."
            if waiting == "custom_model_and_size":
                return "Which model and size? Example: `z-image 1280x704`, `lens 1024x1024`, `chroma 1024x1024`, `flux klein 1024x1024`, or `hidream 1024x1024`."
            return "Please answer `default` or `custom`."
        return "Restored the previous wizard step."

    def _undo_pending_state(self, state: Dict[str, Any]) -> AssistantRouteResult:
        stack = state.get("_undo_stack") if isinstance(state.get("_undo_stack"), list) else []
        if not stack:
            return AssistantRouteResult(True, "Nothing to undo in this wizard yet. Type `cancel` to stop it.")
        prev = dict(stack.pop() or {})
        prev["_undo_stack"] = stack
        prev["_skip_undo"] = True
        self._save_state(prev)
        return AssistantRouteResult(True, "Undone one step.\n\n" + self._pending_question_for_state(prev))

    # ------------------------- public API -------------------------
    def handle_user_text(self, text: str, attachments: Optional[list[dict]] = None) -> AssistantRouteResult:
        text = (text or "").strip()
        image_paths = self._image_paths_from_attachments(attachments or [])
        video_paths = self._media_paths_from_attachments(attachments or [], "video")
        if not text and not image_paths and not video_paths:
            return AssistantRouteResult(False)

        state = self._load_state()
        if state.get("pending_intent") == "ltx_video":
            return self._handle_pending_ltx_video(text, state, attachments or [])
        if state.get("pending_intent") == "image_edit":
            return self._handle_pending_image_edit(text, state)
        if state.get("pending_intent") == "text_to_image":
            return self._handle_pending_image(text, state)

        if text and self._is_ltx_video_request(text):
            return self._start_ltx_video_flow(text, attachments or [])

        if image_paths and text:
            edit_route = self._handle_attached_image_edit_request(text, image_paths)
            if edit_route is not None:
                return edit_route

        try:
            direct = self._parse_direct_image_request(text)
        except ValueError as e:
            return AssistantRouteResult(True, str(e))
        if direct is not None:
            prompt, model_id, width, height = direct
            return self._queue_image(prompt, model_id, width, height)

        prompt = self._extract_image_prompt(text)
        if prompt:
            self._save_state({
                "pending_intent": "text_to_image",
                "prompt": prompt,
                "waiting_for": "workflow_choice",
                "created_at": time.time(),
            })
            return AssistantRouteResult(
                True,
                "Default or custom?\n\nDefault uses Z-Image GGUF at 1376×768. Custom can use Z-Image GGUF, Lens, Chroma, Flux Klein, or HiDream up to 2048×2048, or a similar 16:9 / 9:16 size.",
            )

        # Catch bare commands like "create an image" before the LLM can answer
        # with a generic text-only refusal.
        if self._is_bare_image_request(text):
            self._save_state({
                "pending_intent": "text_to_image",
                "prompt": "",
                "waiting_for": "image_prompt",
                "created_at": time.time(),
            })
            return AssistantRouteResult(
                True,
                "What should be in the image?\n\nExample: `a bee on a flower`.",
            )

        return AssistantRouteResult(False)


    # ------------------------- LTX 2.3 video routing -------------------------
    def _assistant_chat_only_results_enabled(self) -> bool:
        try:
            p = self.root / "presets" / "setsave" / "llama_chat_ui.json"
            data = _load_json(p, {})
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

    def _ltx23_config(self) -> Dict[str, Any]:
        videos = self.registry.get("video_models") if isinstance(self.registry.get("video_models"), dict) else {}
        cfg = videos.get("ltx23") if isinstance(videos.get("ltx23"), dict) else {}
        if cfg:
            return dict(cfg)
        # Fallback keeps older registries usable.
        return {
            "label": "LTX 2.3",
            "folder": "models/ltx23/distilled-1.1",
            "default_checkpoint": "models/ltx23/distilled-1.1/ltx-2.3-22b-distilled-1.1.safetensors",
            "spatial_upsampler": "models/ltx23/spatial_upsampler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            "gemma_root": "models/ltx23/text_encoder/lightricks_gemma_original",
            "cli": "helpers/ltx23_vram_lab_cli.py",
            "env_win": "environments/.ltx23/python.exe",
            "env_posix": "environments/.ltx23/bin/python",
            "output_dir": "output/video/ltx23/chat",
            "defaults": {"pipeline": "two_stages", "vram_lab": "safe", "vram_profile": "auto", "steps": 8, "cfg": 2.0, "stg": 0.0, "rescale": 0.7, "shift": 5.0, "fps": 24, "frames": 121, "resolution": "704p", "aspect": "16:9"},
        }

    def _is_ltx_video_request(self, text: str) -> bool:
        low = " " + re.sub(r"\s+", " ", str(text or "").lower().strip()) + " "
        if not low.strip():
            return False
        if any(x in low for x in (" create a video ", " make a video ", " generate a video ", " text to video ", " image to video ", " continue a video ", " extend a video ", " animate this image ", " animate the image ")):
            return True
        return bool(re.search(r"\b(?:ltx|ltx\s*2\.3|ltx23)\b.*\b(video|animate|continue|extend)\b", low))

    def _media_paths_from_attachments(self, attachments: list[dict], want: str) -> list[str]:
        out: list[str] = []
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
        for att in attachments or []:
            try:
                path = str(att.get("path") or "").strip()
                if not path or not Path(path).exists():
                    continue
                ext = Path(path).suffix.lower()
                kind = str(att.get("kind") or "").lower()
                if want == "image" and (kind == "image" or ext in image_exts):
                    out.append(str(Path(path).resolve()))
                elif want == "video" and (kind == "video" or ext in video_exts):
                    out.append(str(Path(path).resolve()))
            except Exception:
                continue
        return out

    def _start_ltx_video_flow(self, text: str, attachments: list[dict]) -> AssistantRouteResult:
        state = {"pending_intent": "ltx_video", "waiting_for": "model_choice", "created_at": time.time(), "requested_text": text}
        # If user already attached media, keep it for the guided flow.
        imgs = self._media_paths_from_attachments(attachments, "image")
        vids = self._media_paths_from_attachments(attachments, "video")
        if imgs:
            state["attached_image_paths"] = imgs
        if vids:
            state["attached_video_paths"] = vids
        self._save_state(state)
        return AssistantRouteResult(
            True,
            "Which LTX model should I use?\n\nUse `normal LTX`, `distilled 1.1`, or mention the exact `.safetensors` filename for a finetune inside `models/ltx23/distilled-1.1/`. FP16 is preferred; FP8 may not work.\n\nType `cancel` anytime to stop and return to normal chat.",
        )

    def _handle_pending_ltx_video(self, text: str, state: Dict[str, Any], attachments: list[dict]) -> AssistantRouteResult:
        raw = str(text or "").strip()
        low = raw.lower()
        if self._is_undo_command(raw):
            return self._undo_pending_state(state)
        if self._is_cancel_command(raw):
            self._clear_state()
            return AssistantRouteResult(True, "Cancelled the pending LTX video request.")
        # Refresh attachment memory for this turn.
        imgs = self._media_paths_from_attachments(attachments, "image")
        vids = self._media_paths_from_attachments(attachments, "video")
        if imgs:
            state["attached_image_paths"] = imgs
        if vids:
            state["attached_video_paths"] = vids
        waiting = str(state.get("waiting_for") or "model_choice")

        if waiting == "model_choice":
            checkpoint, label_or_msg, ok = self._resolve_ltx_checkpoint(raw)
            if not ok:
                return AssistantRouteResult(True, label_or_msg)
            state["checkpoint_path"] = checkpoint
            state["checkpoint_label"] = label_or_msg
            state["waiting_for"] = "mode_choice"
            self._save_state(state)
            return AssistantRouteResult(True, "Text to video, image to video, or continue a video?")

        if waiting == "mode_choice":
            mode = self._parse_ltx_mode(raw)
            if not mode:
                return AssistantRouteResult(True, "Choose one: `text to video`, `image to video`, or `continue video`.")
            state["video_mode"] = mode
            if mode == "text":
                state["waiting_for"] = "prompt"
                self._save_state(state)
                return AssistantRouteResult(True, self._ltx_prompt_question())
            if mode == "image":
                paths = list(state.get("attached_image_paths") or [])
                if not paths:
                    state["waiting_for"] = "input_media"
                    self._save_state(state)
                    return AssistantRouteResult(True, "Please upload the image you want to animate.")
                return self._accept_ltx_input_image(state, paths[0])
            if mode == "continue":
                paths = list(state.get("attached_video_paths") or [])
                if not paths:
                    state["waiting_for"] = "input_media"
                    self._save_state(state)
                    return AssistantRouteResult(True, "Please upload the video you want to continue.")
                return self._accept_ltx_input_video(state, paths[0])

        if waiting == "input_media":
            mode = str(state.get("video_mode") or "")
            if mode == "image":
                paths = list(state.get("attached_image_paths") or [])
                if not paths:
                    return AssistantRouteResult(True, "I still need the image attachment first.")
                return self._accept_ltx_input_image(state, paths[0])
            if mode == "continue":
                paths = list(state.get("attached_video_paths") or [])
                if not paths:
                    return AssistantRouteResult(True, "I still need the video attachment first.")
                return self._accept_ltx_input_video(state, paths[0])
            state["waiting_for"] = "mode_choice"
            self._save_state(state)
            return AssistantRouteResult(True, "Text to video, image to video, or continue a video?")

        if waiting == "prompt":
            prompt = self._clean_prompt(raw)
            if not prompt:
                return AssistantRouteResult(True, self._ltx_prompt_question())
            state["prompt"] = prompt
            state["waiting_for"] = "resolution_aspect"
            self._save_state(state)
            return AssistantRouteResult(True, "What size do you want? Choose quality and aspect: `480p`, `704p`, or `1088p` plus `16:9`, `9:16`, or `1:1`.\n\nExamples: `704p 16:9`, `480p portrait`, `1088p square`.")

        if waiting == "resolution_aspect":
            parsed = self._parse_ltx_resolution_aspect(raw)
            if not parsed:
                return AssistantRouteResult(True, "Please choose a size like `704p 16:9`, `480p portrait`, or `1088p square`.")
            res_key, aspect_key, width, height = parsed
            state.update({"resolution_key": res_key, "aspect_key": aspect_key, "width": width, "height": height})
            warning = self._ltx_shape_warning(state, width, height)
            state["shape_warning"] = warning
            state["waiting_for"] = "duration_fps"
            self._save_state(state)
            msg = "How many frames and FPS? You can also say something like `10 seconds at 24fps`."
            if warning:
                msg = warning + "\n\n" + msg
            return AssistantRouteResult(True, msg)

        if waiting == "duration_fps":
            parsed = self._parse_ltx_duration_fps(raw, state)
            if not parsed:
                return AssistantRouteResult(True, "Please say the duration/FPS, for example `10 seconds at 24fps` or `240 frames at 24fps`.")
            frames, fps, duration = parsed
            if str(state.get("video_mode") or "") == "continue":
                source_fps = float(state.get("source_fps") or 0.0)
                if source_fps > 0 and abs(float(fps) - source_fps) > 0.05:
                    return AssistantRouteResult(True, f"Sorry, this does not work for now. Continue-video needs the requested FPS to match the uploaded video FPS, because the continuation will be merged with the source video.\n\nYour uploaded video is `{self._fmt_fps(source_fps)}fps`, so please use `{self._fmt_fps(source_fps)}fps`.")
            cap = self._ltx_frame_cap_for_current_gpu(str(state.get("resolution_key") or "704p"))
            if int(frames) > int(cap):
                return AssistantRouteResult(True, f"That is too high. At {state.get('resolution_key', '704p')} your current LTX Auto profile supports up to {int(cap)} frames.\n\nPlease lower the duration/FPS or choose a lower resolution.")
            state.update({"frames": int(frames), "fps": int(fps), "duration_sec": float(duration)})
            state["waiting_for"] = "output_name"
            self._save_state(state)
            return AssistantRouteResult(True, "Name for the output file? Optional. Type `default` for a timestamped name.")

        if waiting == "output_name":
            name = self._sanitize_ltx_filename(raw)
            state["output_name"] = name
            state["waiting_for"] = "confirm"
            self._save_state(state)
            return AssistantRouteResult(True, self._ltx_summary_message(state))

        if waiting == "confirm":
            changed = self._apply_ltx_confirm_change(raw, state)
            if changed:
                self._save_state(state)
                return AssistantRouteResult(True, "Ok done. Any other changes? Type `yes` or `queue it` when it looks right.\n\n" + self._ltx_summary_message(state))
            if low in {"yes", "y", "ok", "okay", "correct", "queue", "queue it", "add to queue", "no", "no changes"}:
                return self._queue_ltx_video_from_state(state)
            return AssistantRouteResult(True, "Type `yes` or `queue it` to add it to the queue, or say what to change. Example: `change resolution to 480p`.")

        state["waiting_for"] = "model_choice"
        self._save_state(state)
        return AssistantRouteResult(True, "Which LTX model should I use?")

    def _ltx_prompt_question(self) -> str:
        return "What is the prompt?\n\nLTX can include sound/speech-style prompt details. If you want dialogue or sound effects, timestamps help. Example:\n`0s: a woman dances under the moonlight 3s: wind blows through the trees 6s: the woman sings \"what a beautiful night\"`\n\nLeave enough time between events so the model can follow them."

    def _resolve_ltx_checkpoint(self, text: str) -> tuple[str, str, bool]:
        cfg = self._ltx23_config()
        folder = (self.root / str(cfg.get("folder") or "models/ltx23/distilled-1.1")).resolve()
        default_path = (self.root / str(cfg.get("default_checkpoint") or "models/ltx23/distilled-1.1/ltx-2.3-22b-distilled-1.1.safetensors")).resolve()
        low = str(text or "").lower().strip()
        default_words = {"", "ltx", "normal", "normal ltx", "default", "default ltx", "distilled", "distilled 1.1", "normal distilled", "ltx 2.3", "ltx23"}
        if low in default_words or any(x in low for x in ("normal ltx", "distilled 1.1", "default")):
            if not default_path.exists():
                return "", f"I could not find the default LTX checkpoint:\n`{default_path}`", False
            return str(default_path), "LTX 2.3 Distilled 1.1", True
        # Find an explicit safetensors filename or strong partial match.
        m = re.search(r"([^\\/\s]+\.safetensors)\b", text or "", flags=re.IGNORECASE)
        wanted = m.group(1).strip() if m else str(text or "").strip()
        if not wanted:
            return "", "Use `normal LTX`, `distilled 1.1`, or mention a `.safetensors` filename.", False
        matches = []
        try:
            for p in folder.glob("*.safetensors"):
                if p.name.lower() == wanted.lower() or wanted.lower() in p.name.lower():
                    matches.append(p)
        except Exception:
            matches = []
        if not matches:
            return "", f"I could not find `{wanted}` in `models/ltx23/distilled-1.1/`. Please check the filename or use `normal LTX`.", False
        chosen = sorted(matches, key=lambda x: (len(x.name), x.name.lower()))[0]
        label = chosen.name
        if "fp8" in chosen.name.lower():
            label += " (FP8 may not work)"
        return str(chosen.resolve()), label, True

    def _parse_ltx_mode(self, text: str) -> str:
        low = str(text or "").lower()
        if any(x in low for x in ("continue", "extend", "resume", "append")):
            return "continue"
        if any(x in low for x in ("image", "animate", "i2v", "img2video", "img to video")):
            return "image"
        if any(x in low for x in ("text", "t2v", "prompt")):
            return "text"
        return ""

    def _accept_ltx_input_image(self, state: Dict[str, Any], image_path: str) -> AssistantRouteResult:
        state["input_image_path"] = str(Path(image_path).resolve())
        state["input_kind"] = "image"
        w, h = self._probe_image_size(image_path)
        if w and h:
            state["input_width"] = w
            state["input_height"] = h
        state["waiting_for"] = "prompt"
        self._save_state(state)
        return AssistantRouteResult(True, "I found the image.\n\n" + self._ltx_prompt_question())

    def _accept_ltx_input_video(self, state: Dict[str, Any], video_path: str) -> AssistantRouteResult:
        video_path = str(Path(video_path).resolve())
        info = self._probe_video_info(video_path)
        frame_path, err = self._extract_ltx_last_frame(video_path)
        if err:
            return AssistantRouteResult(True, err)
        state["source_video_path"] = video_path
        state["input_image_path"] = frame_path
        state["input_kind"] = "video"
        state["input_width"] = int(info.get("width") or 0)
        state["input_height"] = int(info.get("height") or 0)
        state["source_fps"] = float(info.get("fps") or 0.0)
        state["waiting_for"] = "prompt"
        self._save_state(state)
        fps_txt = self._fmt_fps(float(info.get("fps") or 0.0))
        return AssistantRouteResult(True, f"I found the video and extracted the last frame. Source FPS: `{fps_txt}fps`.\n\n" + self._ltx_prompt_question())

    def _parse_ltx_resolution_aspect(self, text: str) -> Optional[tuple[str, str, int, int]]:
        low = str(text or "").lower().replace("×", "x")
        res = "704p"
        if re.search(r"\b(480|512|832x512|512x832|640x640)\b", low):
            res = "480p"
        elif re.search(r"\b(1080|1088|1920x1088|1088x1920|1440x1440)\b", low):
            res = "1088p"
        elif re.search(r"\b(704|720|1280x704|704x1280|1024x1024)\b", low):
            res = "704p"
        aspect = "16:9"
        if any(x in low for x in ("9:16", "portrait", "vertical")):
            aspect = "9:16"
        elif any(x in low for x in ("1:1", "square")):
            aspect = "1:1"
        elif any(x in low for x in ("16:9", "landscape", "wide")):
            aspect = "16:9"
        # If user typed exact dimensions, infer aspect.
        m = re.search(r"(?<!\d)(\d{3,5})\s*x\s*(\d{3,5})(?!\d)", low)
        if m:
            try:
                ww, hh = int(m.group(1)), int(m.group(2))
                if abs((ww / max(1, hh)) - 1.0) < 0.08:
                    aspect = "1:1"
                elif ww > hh:
                    aspect = "16:9"
                else:
                    aspect = "9:16"
            except Exception:
                pass
        table = {
            ("480p", "16:9"): (832, 512), ("480p", "9:16"): (512, 832), ("480p", "1:1"): (640, 640),
            ("704p", "16:9"): (1280, 704), ("704p", "9:16"): (704, 1280), ("704p", "1:1"): (1024, 1024),
            ("1088p", "16:9"): (1920, 1088), ("1088p", "9:16"): (1088, 1920), ("1088p", "1:1"): (1440, 1440),
        }
        w, h = table[(res, aspect)]
        return res, aspect, w, h

    def _parse_ltx_duration_fps(self, text: str, state: Dict[str, Any]) -> Optional[tuple[int, int, float]]:
        low = str(text or "").lower().strip()
        defaults = (self._ltx23_config().get("defaults") or {}) if isinstance(self._ltx23_config().get("defaults"), dict) else {}
        fps = int(defaults.get("fps", 24) or 24)
        frames = int(defaults.get("frames", 121) or 121)

        # IMPORTANT: explicit frame-count shorthand must win over seconds parsing.
        # Examples the chat accepts as FRAMES, not seconds:
        #   "61 @ 20fps", "61@20fps", "61 frames @ 20fps", "61f 20fps"
        # The previous parser only saw "20fps" and silently kept the default
        # 121 frames, which made the confirmation show 6.05s / 121f.
        m_at = re.search(r"(?<!\d)(\d{1,5})\s*@\s*(\d{1,3})\s*(?:fps|frames\s*per\s*second)?\b", low)
        if m_at:
            frames = max(1, int(m_at.group(1)))
            fps = max(1, min(120, int(m_at.group(2))))
            return frames, fps, float(frames) / float(max(1, fps))

        m_fps = re.search(r"(?<!\d)(\d{1,3})\s*(?:fps|frames\s*per\s*second)\b", low)
        if m_fps:
            fps = max(1, min(120, int(m_fps.group(1))))

        # Explicit frame wording, including compact forms like "61f 20fps".
        m_frames = re.search(r"(?<!\d)(\d{1,5})\s*(?:frames|frame|f)\b", low)
        if m_frames:
            frames = int(m_frames.group(1))
            return max(1, frames), fps, float(max(1, frames)) / float(max(1, fps))

        # Bare "61 20fps" is a common answer to "frames + FPS". Treat the
        # first number as frames only when an FPS token is present and there is
        # no seconds word.
        if m_fps:
            prefix = low[:m_fps.start()].strip()
            m_bare_frames = re.search(r"(?<!\d)(\d{1,5})\s*$", prefix)
            if m_bare_frames and not re.search(r"(?:s|sec|secs|second|seconds)\b", prefix):
                frames = int(m_bare_frames.group(1))
                return max(1, frames), fps, float(max(1, frames)) / float(max(1, fps))

        # Seconds are used only when the user explicitly writes a seconds unit.
        m_sec = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:sec|secs|second|seconds|s)\b", low)
        if m_sec:
            duration = float(m_sec.group(1))
            frames = int(round(duration * fps))
            return max(1, frames), fps, duration

        # Accept bare "24fps" by using default frame count.
        if m_fps:
            return max(1, int(frames)), fps, float(frames) / float(max(1, fps))

        return None

    def _ltx_shape_warning(self, state: Dict[str, Any], width: int, height: int) -> str:
        try:
            iw, ih = int(state.get("input_width") or 0), int(state.get("input_height") or 0)
            if iw <= 0 or ih <= 0:
                return ""
            src = self._aspect_label(iw, ih)
            dst = self._aspect_label(width, height)
            if src == dst:
                return ""
            subject = "source video" if str(state.get("input_kind") or "") == "video" else "uploaded input"
            return f"Friendly warning: the {subject} is {src}, but you selected `{dst}`. LTX may crop, stretch, reframe, or change parts of the composition. The result may not match what you expected."
        except Exception:
            return ""

    def _ltx_frame_cap_for_current_gpu(self, resolution_key: str) -> int:
        caps = {
            "480p": {24: 1201, 16: 901, 12: 601},
            "704p": {24: 577, 16: 433, 12: 289},
            "1088p": {24: 265, 16: 199, 12: 133},
        }
        profile = 24
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, text=True, timeout=2)
            gb = float(str(out).splitlines()[0].strip()) / 1024.0
            profile = 24 if gb >= 23.0 else 16 if gb >= 16.0 else 12
        except Exception:
            profile = 24
        return int(caps.get(str(resolution_key), caps["704p"]).get(profile, caps["704p"][24]))

    def _probe_image_size(self, path: str) -> tuple[int, int]:
        try:
            from PIL import Image  # type: ignore
            with Image.open(path) as im:
                return int(im.width), int(im.height)
        except Exception:
            return 0, 0

    def _ffprobe_path(self) -> Path:
        exe = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        cand = self.root / "presets" / "bin" / exe
        return cand if cand.exists() else Path(exe)

    def _ffmpeg_path(self) -> Path:
        exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
        cand = self.root / "presets" / "bin" / exe
        return cand if cand.exists() else Path(exe)

    def _probe_video_info(self, path: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        try:
            out = subprocess.check_output([
                str(self._ffprobe_path()), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate", "-of", "json", str(path)
            ], stderr=subprocess.STDOUT, text=True, timeout=10)
            data = json.loads(out or "{}")
            st = (data.get("streams") or [{}])[0]
            info["width"] = int(st.get("width") or 0)
            info["height"] = int(st.get("height") or 0)
            fps_s = str(st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/0")
            info["fps"] = self._fps_to_float(fps_s)
        except Exception:
            pass
        return info

    def _fps_to_float(self, value: str) -> float:
        try:
            s = str(value or "").strip()
            if "/" in s:
                a, b = s.split("/", 1)
                b_f = float(b)
                return float(a) / b_f if b_f else 0.0
            return float(s)
        except Exception:
            return 0.0

    def _fmt_fps(self, fps: float) -> str:
        try:
            f = float(fps)
            if abs(f - round(f)) < 0.03:
                return str(int(round(f)))
            return f"{f:.3f}".rstrip("0").rstrip(".")
        except Exception:
            return str(fps or "")

    def _extract_ltx_last_frame(self, video_path: str) -> tuple[str, str]:
        ffmpeg = self._ffmpeg_path()
        out_dir = self.root / "temp" / "ltx23_video_frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(video_path).stem).strip("_") or "video"
        out = out_dir / f"assistant_last_{stem}_{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        cmd = [str(ffmpeg), "-y", "-sseof", "-0.1", "-i", str(video_path), "-frames:v", "1", "-update", "1", str(out)]
        try:
            pr = subprocess.run(cmd, cwd=str(self.root), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
            if int(pr.returncode) == 0 and out.exists() and out.stat().st_size > 1024:
                return str(out), ""
            err = (pr.stderr or "").strip()[-800:]
        except Exception as exc:
            err = str(exc)
        return "", "I could not extract the last frame from that video." + (f"\n\nError: {err}" if err else "")

    def _sanitize_ltx_filename(self, text: str) -> str:
        raw = str(text or "").strip()
        if raw.lower() in {"", "default", "skip", "none", "no", "timestamp", "timestamped"}:
            return ""
        s = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw).strip("_").lower()
        return s[:80]

    def _ltx_summary_message(self, state: Dict[str, Any]) -> str:
        mode_label = {"text": "text to video", "image": "image to video", "continue": "continue video"}.get(str(state.get("video_mode") or "text"), "text to video")
        name = str(state.get("output_name") or "default timestamped name")
        warning = str(state.get("shape_warning") or "").strip()
        summary = (
            "OK, we have:\n"
            f"Model: {state.get('checkpoint_label') or 'LTX 2.3'}\n"
            f"Mode: {mode_label}\n"
            f"Resolution: {state.get('resolution_key', '704p')} / {int(state.get('width') or 1280)}×{int(state.get('height') or 704)} / {state.get('aspect_key', '16:9')}\n"
            f"Duration: {float(state.get('duration_sec') or 0.0):.2f}s at {int(state.get('fps') or 24)}fps / {int(state.get('frames') or 121)} frames\n"
            f"Output name: `{name}`\n\n"
            f"Prompt:\n{state.get('prompt') or ''}\n\n"
            "Is this correct, or do you want to make changes?"
        )
        return (warning + "\n\n" + summary) if warning else summary

    def _apply_ltx_confirm_change(self, text: str, state: Dict[str, Any]) -> bool:
        low = str(text or "").lower().strip()
        if not any(x in low for x in ("change", "set ", "resolution", "fps", "frames", "seconds", "prompt", "name", "output")):
            return False
        if "prompt" in low:
            parts = re.split(r"\bprompt\b\s*(?:to|:)?", text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1 and parts[1].strip():
                state["prompt"] = self._clean_prompt(parts[1])
                return True
        if "name" in low or "output" in low:
            parts = re.split(r"\b(?:name|output)\b\s*(?:to|:)?", text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) > 1:
                state["output_name"] = self._sanitize_ltx_filename(parts[1])
                return True
        parsed_res = self._parse_ltx_resolution_aspect(text)
        if parsed_res and any(x in low for x in ("resolution", "size", "480", "704", "720", "1080", "1088", "portrait", "square", "16:9", "9:16", "1:1")):
            res_key, aspect_key, width, height = parsed_res
            state.update({"resolution_key": res_key, "aspect_key": aspect_key, "width": width, "height": height, "shape_warning": self._ltx_shape_warning(state, width, height)})
            return True
        parsed_dur = self._parse_ltx_duration_fps(text, state)
        if parsed_dur and any(x in low for x in ("fps", "frames", "second", "seconds", "duration")):
            frames, fps, duration = parsed_dur
            state.update({"frames": int(frames), "fps": int(fps), "duration_sec": float(duration)})
            return True
        return False

    def _queue_ltx_video_from_state(self, state: Dict[str, Any]) -> AssistantRouteResult:
        cfg = self._ltx23_config()
        defaults = cfg.get("defaults") if isinstance(cfg.get("defaults"), dict) else {}
        prompt = str(state.get("prompt") or "").strip()
        width = int(state.get("width") or 1280)
        height = int(state.get("height") or 704)
        frames = int(state.get("frames") or int(defaults.get("frames", 121) or 121))
        fps = int(state.get("fps") or int(defaults.get("fps", 24) or 24))
        seed = random.randint(1, 2_147_483_647)
        out_dir = (self.root / str(cfg.get("output_dir") or "output/video/ltx23/chat")).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        name = str(state.get("output_name") or "").strip() or "ltx23_video"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        outfile = out_dir / f"{name}_seed{seed}_{stamp}.mp4"
        cli = (self.root / str(cfg.get("cli") or "helpers/ltx23_vram_lab_cli.py")).resolve()
        env_key = "env_win" if os.name == "nt" else "env_posix"
        py = (self.root / str(cfg.get(env_key) or ("environments/.ltx23/python.exe" if os.name == "nt" else "environments/.ltx23/bin/python"))).resolve()
        checkpoint = Path(str(state.get("checkpoint_path") or self.root / str(cfg.get("default_checkpoint") or ""))).resolve()
        gemma = (self.root / str(cfg.get("gemma_root") or "models/ltx23/text_encoder/lightricks_gemma_original")).resolve()
        upsampler = (self.root / str(cfg.get("spatial_upsampler") or "models/ltx23/spatial_upsampler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")).resolve()
        missing = []
        for label, path in (("LTX python", py), ("LTX CLI", cli), ("checkpoint", checkpoint), ("Gemma root", gemma), ("spatial upsampler", upsampler)):
            if not path.exists():
                missing.append(f"{label}: `{path}`")
        if missing:
            return AssistantRouteResult(True, "I cannot queue LTX yet because these required files are missing:\n" + "\n".join(missing))
        cmd = [
            str(py), "-X", "utf8", str(cli),
            "--pipeline", str(defaults.get("pipeline") or "two_stages"),
            "--vram-lab", str(defaults.get("vram_lab") or "safe"),
            "--vram-profile", str(defaults.get("vram_profile") or "auto"),
            "--checkpoint-path", str(checkpoint),
            "--gemma-root", str(gemma),
            "--prompt", prompt,
            "--output-path", str(outfile),
            "--height", str(height),
            "--width", str(width),
            "--num-frames", str(frames),
            "--frame-rate", str(fps),
            "--num-inference-steps", str(int(defaults.get("steps", 8) or 8)),
            "--seed", str(seed),
            "--shift", str(float(defaults.get("shift", 5.0) or 5.0)),
            "--ltx-video-cfg", str(float(defaults.get("cfg", 2.0) or 2.0)),
            "--ltx-video-stg", str(float(defaults.get("stg", 0.0) or 0.0)),
            "--ltx-video-rescale", str(float(defaults.get("rescale", 0.7) or 0.7)),
            "--spatial-upsampler-path", str(upsampler),
        ]
        input_image = str(state.get("input_image_path") or "").strip()
        if input_image:
            cmd += ["--i2v-image", input_image, "--i2v-image-frame", "0", "--i2v-image-strength", "1.0", "--i2v-image-crf", "0"]
        mode = str(state.get("video_mode") or "text")
        glue = mode == "continue" and bool(state.get("source_video_path"))
        label = f"LTX 2.3: {prompt.replace(chr(10), ' ')[:82].strip() or outfile.stem}"
        job = {
            "cmd": cmd,
            "cwd": str(self.root),
            "out_dir": str(out_dir),
            "outfile": str(outfile),
            "scan_dir": str(out_dir),
            "scan_ext": ".mp4",
            "label": label,
            "title": label,
            "engine": "ltx23",
            "prompt": prompt,
            "ffmpeg_path": str(self._ffmpeg_path()),
            "ltx_glue_input_videos": bool(glue),
            "ltx_start_video_path": str(state.get("source_video_path") or ""),
            "ltx_end_video_path": "",
            "ltx_temp_video_frame_paths": [input_image] if glue and input_image else [],
            "assistant_mode": mode,
            "width": width,
            "height": height,
            "frames": frames,
            "fps": fps,
            "seed": seed,
        }
        try:
            job.update(self._assistant_chat_result_flags())
        except Exception:
            pass
        ok = False
        err = ""
        try:
            from helpers.queue_adapter import enqueue_ltx23_generate  # type: ignore
            ok = bool(enqueue_ltx23_generate(job))
        except Exception as exc:
            try:
                from queue_adapter import enqueue_ltx23_generate  # type: ignore
                ok = bool(enqueue_ltx23_generate(job))
            except Exception as exc2:
                err = str(exc2 or exc)
        self._clear_state()
        if ok:
            mode_label = {"text": "text-to-video", "image": "image-to-video", "continue": "continue-video"}.get(mode, "video")
            return AssistantRouteResult(True, f"Added to queue: `{outfile.name}` ({mode_label}, {width}×{height}, {frames} frames at {fps}fps).", True, "ltx23", "LTX 2.3", prompt, width, height, time.time(), "video", tuple([input_image] if input_image else []), str(outfile))
        return AssistantRouteResult(True, "I could not add the LTX job to the FrameVision queue." + (f"\n\nError: {err}" if err else ""), False, "ltx23", "LTX 2.3", prompt, width, height, time.time(), "video", tuple([input_image] if input_image else []), str(outfile))

    # ------------------------- image edit routing -------------------------
    def _image_paths_from_attachments(self, attachments: list[dict]) -> list[str]:
        out: list[str] = []
        for att in attachments or []:
            try:
                kind = str(att.get("kind") or "").lower()
                path = str(att.get("path") or "").strip()
                if kind == "image" and path and Path(path).exists():
                    out.append(str(Path(path).resolve()))
            except Exception:
                continue
        return out

    def _is_image_edit_text(self, text: str) -> bool:
        low = " " + re.sub(r"\s+", " ", (text or "").lower().strip()) + " "
        if not low.strip():
            return False
        if any(x in low for x in [
            " edit ", " change ", " add ", " remove ", " replace ", " repaint ",
            " inpaint ", " retouch ", " make it ", " turn this ", " turn it ",
            " use this image ", " use the image ", " reference image ", " as reference ",
            " new version ", " make a new version ", " redesign ", " restyle ", " background ",
        ]):
            return True
        return bool(re.match(r"^\s*(?:edit|change|add|remove|replace|turn|make|use)\b", text or "", flags=re.IGNORECASE))

    def _strip_edit_command(self, text: str) -> str:
        raw = str(text or "")
        for sep in (":", " -> ", " — ", " - "):
            if sep in raw:
                right = raw.split(sep, 1)[1]
                return self._strip_edit_command(right)
        s = raw
        s = re.sub(r"^\s*(?:please\s+)?(?:edit|change|modify|repaint|retouch|inpaint)\s+(?:this|the)?\s*(?:attached\s+)?(?:image|picture|photo)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*(?:use\s+)?(?:this|the)?\s*(?:attached\s+)?(?:image|picture|photo)\s+(?:as\s+)?(?:a\s+)?(?:reference\s+)?", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*(?:with|using)\s+", "", s, flags=re.IGNORECASE)
        # Remove one explicit supported edit model alias.
        for model_id in ("flux_klein", "hidream"):
            cfg = self._model_cfg(model_id)
            aliases = list(cfg.get("aliases") or []) + [model_id]
            aliases.sort(key=len, reverse=True)
            for alias in aliases:
                ns = re.sub(r"\b" + re.escape(str(alias)) + r"\b", " ", s, count=1, flags=re.IGNORECASE)
                if ns != s:
                    s = ns
                    break
        # Remove one explicit size mention, including wrappers such as
        # "at 1600x896" or "resolution 1920 x 1088".
        s = re.sub(
            r"(?:\b(?:at|in|size|resolution|output\s+size|output\s+resolution)\b\s*)?(?<!\d)\d{3,5}\s*(?:x|×|by)\s*\d{3,5}(?!\d)",
            " ",
            s,
            count=1,
            flags=re.IGNORECASE,
        )
        s = re.sub(r"\b(?:with|using|at|in)\b\s*(?=$)", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"\b(?:with|using|at|in)\b\s+(?:with|using|at|in)\b", " ", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*(?:with|using|to|and|:|-)+\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s+", " ", s).strip(" .,-–—")
        return self._clean_prompt(s)

    def _suggest_edit_model_id(self, prompt: str) -> str:
        low = (prompt or "").lower()
        hidream_words = ["background", "whole scene", "cinematic", "night scene", "reference", "new version", "redesign", "restyle", "style", "combine", "jungle", "snowy forest", "replace the whole"]
        flux_words = ["add ", "remove ", "hat", "sunglasses", "shirt color", "corner", "bottom left", "bottom right", "small edit", "put "]
        if any(w in low for w in hidream_words):
            return "hidream"
        if any(w in low for w in flux_words):
            return "flux_klein"
        return ""

    def _edit_prompt_is_missing_or_bare(self, prompt: str) -> bool:
        """Return True for commands that say to edit, but not what the edit is.

        This prevents the guided flow from asking for Flux Klein/HiDream before
        it knows the actual edit instruction. Otherwise a later answer like
        "hidream" can accidentally become the edit prompt and the model just
        rebuilds the source image.
        """
        p = self._clean_prompt(prompt)
        if not p:
            return True
        low = re.sub(r"[^a-z0-9]+", " ", p.lower()).strip()
        bare = {
            "edit", "edit image", "edit the image", "edit this image",
            "edit picture", "edit the picture", "edit this picture",
            "edit photo", "edit the photo", "edit this photo",
            "change image", "change the image", "change this image",
            "modify image", "modify the image", "modify this image",
            "repaint image", "repaint the image", "repaint this image",
            "retouch image", "retouch the image", "retouch this image",
            "inpaint image", "inpaint the image", "inpaint this image",
            "use this image", "use the image", "use image as reference",
            "use this image as reference", "use the image as reference",
        }
        return low in bare or low in {"with hidream", "hidream", "flux", "flux klein", "klein"}

    def _extract_requested_edit_size(self, text: str) -> tuple[int, int]:
        parsed = self._parse_size(text)
        if not parsed:
            return 0, 0
        try:
            return int(parsed[0]), int(parsed[1])
        except Exception:
            return 0, 0

    def _ask_for_edit_instruction(self, image_paths: list[str], model_id: str = "", requested_size: tuple[int, int] = (0, 0)) -> AssistantRouteResult:
        state = {
            "pending_intent": "image_edit",
            "prompt": "",
            "image_paths": image_paths,
            "waiting_for": "edit_instruction",
            "created_at": time.time(),
        }
        if model_id in ("flux_klein", "hidream"):
            state["model_id"] = model_id
        try:
            req_w, req_h = int(requested_size[0]), int(requested_size[1])
        except Exception:
            req_w, req_h = 0, 0
        if req_w > 0 and req_h > 0:
            state["requested_width"] = req_w
            state["requested_height"] = req_h
        self._save_state(state)
        model_note = ""
        if model_id == "flux_klein":
            model_note = " I will use Flux Klein after you give the edit."
        elif model_id == "hidream":
            model_note = " I will use HiDream after you give the edit."
        return AssistantRouteResult(
            True,
            "What should I change in the attached image?" + model_note + "\n\nExample: `replace the background with a beach festival scene`.",
        )

    def _ask_for_edit_model(self, prompt: str, image_paths: list[str], requested_size: tuple[int, int] = (0, 0)) -> AssistantRouteResult:
        suggested = self._suggest_edit_model_id(prompt)
        state = {
            "pending_intent": "image_edit",
            "prompt": prompt,
            "image_paths": image_paths,
            "waiting_for": "edit_model_choice",
            "created_at": time.time(),
        }
        try:
            req_w, req_h = int(requested_size[0]), int(requested_size[1])
        except Exception:
            req_w, req_h = 0, 0
        if req_w > 0 and req_h > 0:
            state["requested_width"] = req_w
            state["requested_height"] = req_h
        self._save_state(state)
        first = "HiDream" if suggested == "hidream" else "Flux Klein" if suggested == "flux_klein" else "Flux Klein or HiDream"
        return AssistantRouteResult(
            True,
            f"Use Flux Klein or HiDream?\n\n- Flux Klein = small edits / add or remove something in the existing image\n- HiDream = bigger edits / replace backgrounds / reference-image rebuilds\n\nSuggested: {first}.",
        )

    def _handle_attached_image_edit_request(self, text: str, image_paths: list[str]) -> Optional[AssistantRouteResult]:
        if not self._is_image_edit_text(text):
            return None
        model_id = self._parse_model_id(text)
        if model_id and model_id not in ("flux_klein", "hidream"):
            return AssistantRouteResult(True, "For attached-image edits I can use Flux Klein or HiDream. Use Flux Klein for small edits, or HiDream for bigger/reference-image edits.")
        requested_size = self._extract_requested_edit_size(text)
        prompt = self._strip_edit_command(text)
        if not prompt:
            prompt = self._clean_prompt(text)

        # Guided mode must ask for the actual edit first. Do not ask for model
        # choice yet when the user only says "edit this image".
        if self._edit_prompt_is_missing_or_bare(prompt):
            return self._ask_for_edit_instruction(image_paths, model_id=model_id or "", requested_size=requested_size)

        if model_id in ("flux_klein", "hidream"):
            return self._queue_image_edit(prompt, model_id, image_paths, requested_width=requested_size[0], requested_height=requested_size[1])

        return self._ask_for_edit_model(prompt, image_paths, requested_size=requested_size)

    def _handle_pending_image_edit(self, text: str, state: Dict[str, Any]) -> AssistantRouteResult:
        low = (text or "").lower().strip()
        if self._is_undo_command(text):
            return self._undo_pending_state(state)
        if self._is_cancel_command(text):
            self._clear_state()
            return AssistantRouteResult(True, "Cancelled the pending image edit.")
        waiting_for = str(state.get("waiting_for") or "")
        image_paths = [str(p) for p in (state.get("image_paths") or []) if str(p).strip() and Path(str(p)).exists()]
        if not image_paths:
            self._clear_state()
            return AssistantRouteResult(True, "I lost the attached image. Please attach it again and say the edit again.")
        req_w = int(state.get("requested_width") or 0)
        req_h = int(state.get("requested_height") or 0)

        if waiting_for == "edit_instruction":
            prompt = self._strip_edit_command(text)
            if self._edit_prompt_is_missing_or_bare(prompt):
                return AssistantRouteResult(True, "What should I change in the attached image? Example: `add sunglasses` or `replace the background with a beach festival scene`.")
            model_id = str(state.get("model_id") or "")
            parsed_model = self._parse_model_id(text)
            parsed_size = self._extract_requested_edit_size(text)
            if parsed_size[0] > 0 and parsed_size[1] > 0:
                req_w, req_h = parsed_size
            if parsed_model in ("flux_klein", "hidream"):
                model_id = parsed_model
                prompt = self._strip_edit_command(text)
            if model_id in ("flux_klein", "hidream"):
                return self._queue_image_edit(prompt, model_id, image_paths, requested_width=req_w, requested_height=req_h)
            return self._ask_for_edit_model(prompt, image_paths, requested_size=(req_w, req_h))

        prompt = self._clean_prompt(str(state.get("prompt") or ""))
        if not prompt:
            self._clear_state()
            return AssistantRouteResult(True, "I lost the edit instruction. Please attach the image and say the edit again.")
        model_id = self._parse_model_id(text)
        parsed_size = self._extract_requested_edit_size(text)
        if parsed_size[0] > 0 and parsed_size[1] > 0:
            req_w, req_h = parsed_size
        if not model_id:
            # Allow very short answers.
            if "flux" in low or "klein" in low:
                model_id = "flux_klein"
            elif "hidream" in low or "hi dream" in low:
                model_id = "hidream"
        if model_id not in ("flux_klein", "hidream"):
            return AssistantRouteResult(True, "Please choose `Flux Klein` or `HiDream`.")
        return self._queue_image_edit(prompt, model_id, image_paths, requested_width=req_w, requested_height=req_h)

    def _first_image_size(self, image_paths: list[str]) -> tuple[int, int]:
        try:
            from PIL import Image  # type: ignore
            with Image.open(image_paths[0]) as img:
                return int(img.width), int(img.height)
        except Exception:
            return 1024, 1024

    def _hidream_allowed_edit_sizes(self) -> list[tuple[int, int]]:
        """Hard HiDream edit/reference buckets known to behave well.

        HiDream reference/edit mode is picky: many arbitrary sizes can produce
        pale, washed-out or weak-reference results. Keep chat-triggered HiDream
        edits on the same small set of proven output pixel budgets. Portrait
        flips and square equivalents are included so the user can still edit
        portrait/square sources without leaving the safe buckets.
        """
        return [
            # Chat-driven HiDream edit/reference jobs are kept on the same
            # proven buckets as the HiDream tab. Smaller buckets such as
            # 1024x576 can still run, but they have produced pale/weak edits
            # in real testing, so the chat router does not choose them.
            (1600, 896), (896, 1600), (1024, 1024),
            (1920, 1088), (1088, 1920), (1536, 1536),
        ]

    def _hidream_reference_safe_size(self, image_paths: list[str], requested_width: int = 0, requested_height: int = 0) -> tuple[int, int]:
        """Pick a proven HiDream edit/reference output size.

        Chat edits should stay on the same stable buckets the user tested in the
        HiDream UI. For landscape/portrait, keep the choice simple and stable:
        1600×896 / 896×1600 for normal requests and 1920×1088 / 1088×1920 for
        clearly larger requests. Square requests keep the square-safe buckets.
        """
        w, h = self._first_image_size(image_paths)
        if requested_width and requested_height:
            try:
                rw, rh = int(requested_width), int(requested_height)
                if rw > 0 and rh > 0:
                    w, h = rw, rh
            except Exception:
                pass

        requested_ratio = float(w) / max(1.0, float(h))
        requested_area = max(1.0, float(w) * float(h))
        if 0.90 <= requested_ratio <= 1.10:
            square_sizes = [(1024, 1024), (1536, 1536)]
            return square_sizes[1] if requested_area > (1024 * 1024) else square_sizes[0]
        if requested_ratio > 1.10:
            if w >= 1920 or h >= 1088 or requested_area > float(1920 * 1088):
                return (1920, 1088)
            return (1600, 896)
        if h >= 1920 or w >= 1088 or requested_area > float(1920 * 1088):
            return (1088, 1920)
        return (896, 1600)

    def _default_edit_size(self, model_id: str, image_paths: list[str]) -> tuple[int, int]:
        w, h = self._first_image_size(image_paths)
        if model_id == "hidream":
            return self._hidream_reference_safe_size(image_paths)
        # Flux can stay closer to the original image, but snap to registry multiple.
        max_area = int(self.registry.get("max_custom_area") or (2048 * 2048))
        if w * h > max_area:
            scale = (max_area / float(max(1, w * h))) ** 0.5
            w, h = int(w * scale), int(h * scale)
        mult = int(self.registry.get("snap_multiple") or 64)
        w = max(mult, int(round(w / mult) * mult))
        h = max(mult, int(round(h / mult) * mult))
        ok, _msg, w, h = self._validate_size(w, h)
        if ok:
            return w, h
        return 1024, 1024

    # ------------------------- state -------------------------
    def _load_state(self) -> Dict[str, Any]:
        state = _load_json(self.state_path, {})
        if not isinstance(state, dict):
            return {}
        try:
            if time.time() - float(state.get("created_at") or 0) > 60 * 60 * 8:
                return {}
        except Exception:
            pass
        return state

    def _save_state(self, state: Dict[str, Any]) -> None:
        new_state = dict(state or {})
        skip_undo = bool(new_state.pop("_skip_undo", False))
        if not skip_undo and new_state.get("pending_intent"):
            old_state = _load_json(self.state_path, {})
            if isinstance(old_state, dict) and old_state.get("pending_intent") == new_state.get("pending_intent"):
                old_clean = self._state_without_undo(old_state)
                new_clean = self._state_without_undo(new_state)
                if old_clean and old_clean != new_clean:
                    stack = old_state.get("_undo_stack") if isinstance(old_state.get("_undo_stack"), list) else []
                    if not stack or stack[-1] != old_clean:
                        stack = list(stack) + [old_clean]
                    new_state["_undo_stack"] = stack[-20:]
        _save_json_atomic(self.state_path, new_state)

    def _clear_state(self) -> None:
        try:
            if self.state_path.exists():
                self.state_path.unlink()
        except Exception:
            self._save_state({})

    def _clean_prompt(self, prompt: str) -> str:
        """Normalize short user prompt fragments safely.

        This mainly removes separators accidentally captured from chat text, e.g.
        ": an alien walking in the park" -> "an alien walking in the park".
        """
        s = str(prompt or "").strip()
        s = re.sub(r"^[\s:;,.\-–—>]+", "", s).strip()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    # ------------------------- parsing -------------------------
    def _is_image_command(self, text: str) -> bool:
        low = (text or "").lower().strip()
        if not low:
            return False
        if low.startswith(("can i ", "can we ", "how ", "why ", "what ")):
            return False
        return bool(re.match(r"^\s*(?:create|make|generate|render|queue)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo|render)\b", text or "", flags=re.IGNORECASE))

    def _extract_direct_request_prompt(self, text: str) -> str:
        raw = str(text or "")
        # Best/clearest form: everything after a separator becomes the actual prompt.
        for sep in (":", " -> ", " — ", " - "):
            if sep in raw:
                right = raw.split(sep, 1)[1]
                return self._clean_prompt(right)

        # Fallback: if user wrote model+size in one shot without a colon, strip the
        # command prefix and known model/size parts as best as we can.
        s = raw
        s = re.sub(r"^\s*(?:create|make|generate|render|queue)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo|render)\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^\s*(?:with|using)\s+", "", s, flags=re.IGNORECASE)

        # Remove first explicit size mention.
        s = re.sub(r"(?<!\d)\d{3,5}\s*(?:x|×|by)\s*\d{3,5}(?!\d)", " ", s, count=1, flags=re.IGNORECASE)

        # Remove one known model alias near the front if present.
        alias_hits = []
        models = dict(self.registry.get("models") or {})
        for model_id, cfg in models.items():
            aliases = list(cfg.get("aliases") or []) + [model_id]
            for alias in aliases:
                alias_hits.append(str(alias))
        alias_hits.sort(key=len, reverse=True)
        for alias in alias_hits:
            pat = r"\b" + re.escape(alias) + r"\b"
            new_s = re.sub(pat, " ", s, count=1, flags=re.IGNORECASE)
            if new_s != s:
                s = new_s
                break

        s = re.sub(r"^\s*(?:with|using|of)\s+", "", s, flags=re.IGNORECASE)
        s = re.sub(r"^[,;\-–—>\s]+", "", s)
        return self._clean_prompt(s)

    def _parse_direct_image_request(self, text: str) -> Optional[Tuple[str, str, int, int]]:
        if not self._is_image_command(text):
            return None
        parsed_model = self._parse_model_id(text)
        parsed_size = self._parse_size(text)

        # Only auto-queue one-shot requests when the user gave enough structure.
        # This keeps simple prompts like "create an image of a bee" on the friendly
        # default/custom path, while allowing smarter single-line commands such as:
        # "create an image with flux klein 1024x1024: a bee on a flower"
        structured = bool(parsed_size or parsed_model) and any(sep in str(text or "") for sep in (":", " -> ", " — ", " - "))
        if not structured:
            return None

        if parsed_size and not parsed_model:
            unknown = self._looks_like_unknown_custom_model(text)
            if unknown:
                raise ValueError(f"I do not know the image model `{unknown}` yet. Available image models: {self._available_model_names()}.")

        model_id = parsed_model or str(self.registry.get("default_image_model") or "zimage_gguf")
        if not parsed_size:
            raise ValueError("What size should I use? Example: `1280x704`, `1376x768`, or `2048x2048`.")
        width, height = parsed_size
        ok, msg, width, height = self._validate_size(width, height)
        if not ok:
            raise ValueError(msg)

        prompt = self._extract_direct_request_prompt(text)
        if not prompt:
            raise ValueError("What should be in the image? Example: `a bee on a flower`.")
        return prompt, model_id, width, height

    def _extract_image_prompt(self, text: str) -> str:
        low = text.lower().strip()
        # Avoid grabbing questions about image creation instead of commands.
        if low.startswith(("can i ", "can we ", "how ", "why ", "what ")):
            return ""
        patterns = [
            r"^\s*(?:create|make|generate|render|queue)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo|render)\s+(?:of\s+|with\s+)?(.+)$",
            r"^\s*(?:image|picture|photo)\s+(?:of\s+|with\s+)(.+)$",
        ]
        for pat in patterns:
            m = re.match(pat, text, flags=re.IGNORECASE | re.DOTALL)
            if m:
                prompt = self._clean_prompt(m.group(1) or "")
                return prompt
        return ""

    def _is_bare_image_request(self, text: str) -> bool:
        low = re.sub(r"\s+", " ", (text or "").lower().strip())
        if not low:
            return False
        if low.startswith(("can i ", "can we ", "how ", "why ", "what ")):
            return False
        bare_patterns = [
            r"^(?:create|make|generate|render|queue)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo|render)\s*$",
            r"^(?:create|make|generate|render|queue)\s+(?:me\s+)?(?:an?\s+)?(?:image|picture|photo|render)\s+(?:of|with)\s*$",
        ]
        return any(re.match(pat, low, flags=re.IGNORECASE) for pat in bare_patterns)

    def _handle_pending_image(self, text: str, state: Dict[str, Any]) -> AssistantRouteResult:
        low = text.lower().strip()
        if self._is_undo_command(text):
            return self._undo_pending_state(state)
        if self._is_cancel_command(text):
            self._clear_state()
            return AssistantRouteResult(True, "Cancelled the pending image request.")

        prompt = str(state.get("prompt") or "").strip()
        waiting_for = str(state.get("waiting_for") or "").strip()

        if waiting_for == "image_prompt":
            prompt = self._clean_prompt(text)
            if not prompt:
                return AssistantRouteResult(True, "What should be in the image? Example: `a bee on a flower`.")
            state["prompt"] = prompt
            state["waiting_for"] = "workflow_choice"
            state["created_at"] = time.time()
            self._save_state(state)
            return AssistantRouteResult(
                True,
                "Default or custom?\n\nDefault uses Z-Image GGUF at 1376×768. Custom can use Z-Image GGUF, Lens, Chroma, Flux Klein, or HiDream up to 2048×2048, or a similar 16:9 / 9:16 size.",
            )

        if not prompt:
            self._clear_state()
            return AssistantRouteResult(True, "I lost the image prompt. Please say the image idea again.")

        if low in ("default", "use default", "defaults"):
            model_id = str(self.registry.get("default_image_model") or "zimage_gguf")
            width = int(self.registry.get("default_width") or 1376)
            height = int(self.registry.get("default_height") or 768)
            return self._queue_image(prompt, model_id, width, height)

        if low == "custom":
            state["waiting_for"] = "custom_model_and_size"
            self._save_state(state)
            return AssistantRouteResult(True, "Which model and size? Example: `z-image 1280x704`, `lens 1024x1024`, `chroma 1024x1024`, `flux klein 1024x1024`, or `hidream 1024x1024`.")

        parsed_model = self._parse_model_id(text)
        parsed_size = self._parse_size(text)
        if parsed_model or parsed_size:
            if parsed_size and not parsed_model:
                unknown = self._looks_like_unknown_custom_model(text)
                if unknown:
                    return AssistantRouteResult(True, f"I do not know the image model `{unknown}` yet. Available image models: {self._available_model_names()}.")
            model_id = parsed_model or str(self.registry.get("default_image_model") or "zimage_gguf")
            if not parsed_size:
                return AssistantRouteResult(True, "What size should I use? Example: `1280x704`, `1376x768`, or `2048x2048`.")
            width, height = parsed_size
            ok, msg, width, height = self._validate_size(width, height)
            if not ok:
                return AssistantRouteResult(True, msg)
            return self._queue_image(prompt, model_id, width, height)

        return AssistantRouteResult(True, "Please answer `default` or `custom`. For custom, say something like `z-image 1280x704`, `lens 1024x1024`, `chroma 1024x1024`, `flux klein 1024x1024`, or `hidream 1024x1024`.")

    def _parse_model_id(self, text: str) -> str:
        low = " " + re.sub(r"[^a-z0-9]+", " ", text.lower()) + " "
        models = dict(self.registry.get("models") or {})
        # Prefer longer aliases first so "z image gguf" beats "z image".
        alias_hits = []
        for model_id, cfg in models.items():
            aliases = list(cfg.get("aliases") or []) + [model_id]
            for alias in aliases:
                norm = " " + re.sub(r"[^a-z0-9]+", " ", str(alias).lower()).strip() + " "
                if norm.strip() and norm in low:
                    alias_hits.append((len(norm), model_id))
        if not alias_hits:
            return ""
        alias_hits.sort(reverse=True)
        return str(alias_hits[0][1])

    def _parse_size(self, text: str) -> Optional[Tuple[int, int]]:
        m = re.search(r"(?<!\d)(\d{3,5})\s*(?:x|×|by)\s*(\d{3,5})(?!\d)", text, flags=re.IGNORECASE)
        if not m:
            return None
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None

    def _looks_like_unknown_custom_model(self, text: str) -> str:
        """Return a friendly unknown model name when user typed a model-ish word.

        This prevents custom replies like "potato 1024x1024" from silently using
        the default Z-Image model just because a size was found.
        """
        raw = str(text or "")
        # In one-shot requests, only inspect the left side before the actual prompt.
        for sep in (":", " -> ", " — ", " - "):
            if sep in raw:
                raw = raw.split(sep, 1)[0]
                break
        raw = re.sub(r"(?<!\d)\d{3,5}\s*(?:x|×|by)\s*\d{3,5}(?!\d)", " ", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b(create|make|generate|queue|use|using|model|with|at|size|resolution|image|picture|photo|render|default|custom|an|a|of|me)\b", " ", raw, flags=re.IGNORECASE)
        raw = re.sub(r"[^a-zA-Z0-9\-_. ]+", " ", raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        if not raw:
            return ""
        # Ignore bare size/aspect chatter; anything else with letters is likely
        # the user trying to name a model.
        if not re.search(r"[a-zA-Z]", raw):
            return ""
        return raw[:60]

    def _available_model_names(self) -> str:
        return ", ".join(str(v.get("label") or k) for k, v in (self.registry.get("models") or {}).items())

    def _validate_size(self, width: int, height: int) -> Tuple[bool, str, int, int]:
        try:
            width = int(width)
            height = int(height)
        except Exception:
            return False, "That size was not readable. Use something like `1280x704`.", 0, 0
        if width <= 0 or height <= 0:
            return False, "Width and height must be above zero.", width, height
        max_w = int(self.registry.get("max_custom_width") or 2048)
        max_h = int(self.registry.get("max_custom_height") or 2048)
        max_area = int(self.registry.get("max_custom_area") or (2048 * 2048))
        if width > max_w and height > max_h:
            return False, f"That is too large. Custom image size is capped around 2048×2048 or an equivalent 16:9 / 9:16 size.", width, height
        if width * height > max_area:
            return False, f"That is too large. Custom image size is capped at about {max_area:,} pixels, so use 2048×2048, 2688×1536, 1536×2688, 1920×1088, 1376×768, or smaller.", width, height
        mult = int(self.registry.get("snap_multiple") or 64)
        if mult > 1:
            sw = max(mult, int(round(width / mult) * mult))
            sh = max(mult, int(round(height / mult) * mult))
            if (sw, sh) != (width, height):
                # Only silently snap small differences. Big differences should be explicit.
                if abs(sw - width) <= 32 and abs(sh - height) <= 32 and sw * sh <= max_area:
                    width, height = sw, sh
                else:
                    return False, f"Please use a size that is a multiple of {mult}. Closest safe size: {sw}×{sh}.", width, height
        return True, "", width, height

    # ------------------------- queue -------------------------
    def _model_cfg(self, model_id: str) -> Dict[str, Any]:
        return dict((self.registry.get("models") or {}).get(model_id) or {})

    def _folder_exists(self, model_id: str) -> Tuple[bool, str, str]:
        cfg = self._model_cfg(model_id)
        label = str(cfg.get("label") or model_id)
        folder = str(cfg.get("folder") or "")
        path = self.root / folder if folder else self.root
        return path.exists(), label, folder

    def _zimage_quant_score(self, name: str) -> Tuple[int, str]:
        """Return a rough quality score for Z-Image GGUF filenames."""
        low = str(name or "").lower()
        patterns = [
            (r"q8(?:[_-]?0)?", 800),
            (r"q6(?:[_-]?k)?", 650),
            (r"q5[_-]?k[_-]?m", 560),
            (r"q5[_-]?k[_-]?s", 550),
            (r"q5(?:[_-]?1)?", 530),
            (r"q5(?:[_-]?0)?", 520),
            (r"q4[_-]?k[_-]?m", 460),
            (r"q4[_-]?k[_-]?s", 450),
            (r"q4(?:[_-]?1)?", 430),
            (r"q4(?:[_-]?0)?", 420),
            (r"q3[_-]?k[_-]?m", 360),
            (r"q3[_-]?k[_-]?s", 350),
            (r"q3", 330),
            (r"q2[_-]?k", 260),
            (r"q2", 240),
        ]
        for pat, score in patterns:
            if re.search(pat, low, flags=re.IGNORECASE):
                return score, low
        return 0, low

    def _find_best_zimage_diffusion_gguf(self, folder: Path) -> Optional[Path]:
        try:
            candidates = []
            for p in folder.glob("*.gguf"):
                low = p.name.lower()
                if ("z_image" in low or "z-image" in low) and ("instruct" not in low) and ("qwen" not in low):
                    candidates.append(p)
            if not candidates:
                return None
            candidates.sort(key=lambda p: (self._zimage_quant_score(p.name)[0], p.name.lower()))
            return candidates[-1]
        except Exception:
            return None

    def _find_best_zimage_instruct_gguf(self, folder: Path) -> Optional[Path]:
        try:
            candidates = []
            for p in folder.glob("*.gguf"):
                low = p.name.lower()
                if "instruct" in low or "qwen" in low:
                    candidates.append(p)
            if not candidates:
                return None
            candidates.sort(key=lambda p: (("instruct" in p.name.lower()), p.name.lower()))
            return candidates[-1]
        except Exception:
            return None

    def _validate_model_install(self, model_id: str, cfg: Dict[str, Any]) -> Tuple[bool, str]:
        label = str(cfg.get("label") or model_id)
        folder_rel = str(cfg.get("folder") or "")
        folder = self.root / folder_rel if folder_rel else self.root
        if not folder.exists():
            return False, f"{label} is not installed yet. Please install it first from Optional Installs. Expected folder: `{folder_rel}`."

        # Z-Image GGUF needs real assets, not just the folder itself.
        if model_id == "zimage_gguf":
            best_diffusion = self._find_best_zimage_diffusion_gguf(folder)
            best_instruct = self._find_best_zimage_instruct_gguf(folder)
            vae_path = folder / str(cfg.get("required_vae") or "ae.safetensors")
            missing = []
            if best_diffusion is None:
                missing.append("a Z-Image diffusion GGUF (for example z_image_turbo-Q8/Q6/Q5...) in models/Z-Image-Turbo GGUF")
            if best_instruct is None:
                missing.append("an instruct GGUF (for example Qwen Instruct) in models/Z-Image-Turbo GGUF")
            if not vae_path.exists():
                missing.append(str(vae_path))

            # sd-cli may live either in the model folder or presets/bin.
            sd_cli_candidates = [
                folder / "sd-cli.exe",
                folder / "bin" / "sd-cli.exe",
                self.root / "presets" / "bin" / "sd-cli.exe",
                self.root / "presets" / "bin" / "sd.exe",
            ]
            if not any(p.exists() for p in sd_cli_candidates):
                missing.append("sd-cli.exe in presets\bin or models/Z-Image-Turbo GGUF\bin")

            if missing:
                first = missing[0]
                return False, (
                    f"{label} is installed only partly or needs repair. "
                    f"Missing: `{first}`. Please run the Z-Image GGUF Optional Install/Repair first."
                )
        if model_id == "flux_klein":
            if not self._find_best_flux_diffusion_gguf(folder):
                return False, f"{label} is not installed yet or needs repair. No Flux/Klein diffusion GGUF found under `{folder}` (including the unet subfolder). Please install Flux Klein first."
            if not self._find_best_flux_llm_gguf(folder):
                return False, f"{label} is not installed yet or needs repair. No Qwen/LLM GGUF found under `{folder}` (including the text_encoders subfolder). Please install Flux Klein first."
            if not self._find_best_flux_vae(folder):
                return False, f"{label} is not installed yet or needs repair. No VAE/AE safetensors found under `{folder}`. Please install Flux Klein first."
        if model_id == "hidream":
            if not self._find_hidream_model_key(folder):
                return False, f"{label} is not installed yet. Please install HiDream first. Expected one HiDream model folder under `{folder}`."
            env_py = self.root / "environments" / ".hidream_dev" / "python.exe"
            cli_py = self.root / "helpers" / "hidream_cli.py"
            if not env_py.exists():
                return False, f"{label} is installed only partly or needs repair. Missing HiDream environment: `{env_py}`. Please run the HiDream Optional Install/Repair first."
            if not cli_py.exists():
                return False, f"{label} is installed only partly or needs repair. Missing helper CLI: `{cli_py}`. Please update/repair FrameVision so helpers/hidream_cli.py is present."
        return True, ""

    def _queue_image(self, prompt: str, model_id: str, width: int, height: int) -> AssistantRouteResult:
        cfg = self._model_cfg(model_id)
        if not cfg:
            self._clear_state()
            available = ", ".join(str(v.get("label") or k) for k, v in (self.registry.get("models") or {}).items())
            return AssistantRouteResult(True, f"I do not know that image model yet. Available image models: {available}.")

        label = str(cfg.get("label") or model_id)
        ok_install, install_msg = self._validate_model_install(model_id, cfg)
        if not ok_install:
            self._clear_state()
            return AssistantRouteResult(True, install_msg)

        seed = random.randint(1, 2_147_483_647)
        try:
            if model_id == "lens":
                ok = self._enqueue_lens(prompt, cfg, width, height, seed)
            elif model_id == "chroma":
                ok = self._enqueue_chroma(prompt, cfg, width, height, seed)
            elif model_id == "flux_klein":
                ok = self._enqueue_flux_klein(prompt, cfg, width, height, seed)
            elif model_id == "hidream":
                ok = self._enqueue_hidream(prompt, cfg, width, height, seed)
            else:
                ok = self._enqueue_zimage_gguf(prompt, cfg, width, height, seed)
        except Exception as exc:
            ok = False
            err = str(exc)
        else:
            err = ""

        self._clear_state()
        if ok:
            return AssistantRouteResult(True, f"Queued {label}: `{prompt}` at {width}×{height}.", True, model_id, label, prompt, width, height, time.time(), "create", ())
        return AssistantRouteResult(True, f"I could not add the {label} job to the FrameVision queue." + (f"\n\nError: {err}" if err else ""), False, model_id, label, prompt, width, height, time.time(), "create", ())

    def _queue_image_edit(self, prompt: str, model_id: str, image_paths: list[str], requested_width: int = 0, requested_height: int = 0) -> AssistantRouteResult:
        cfg = self._model_cfg(model_id)
        if model_id not in ("flux_klein", "hidream") or not cfg:
            self._clear_state()
            return AssistantRouteResult(True, "Attached-image edits currently support Flux Klein and HiDream.")
        label = str(cfg.get("label") or model_id)
        ok_install, install_msg = self._validate_model_install(model_id, cfg)
        if not ok_install:
            self._clear_state()
            return AssistantRouteResult(True, install_msg)
        image_paths = [str(Path(p).resolve()) for p in image_paths if str(p).strip() and Path(str(p)).exists()]
        if not image_paths:
            self._clear_state()
            return AssistantRouteResult(True, "I could not find the attached image on disk. Please attach it again.")
        try:
            requested_width = int(requested_width or 0)
            requested_height = int(requested_height or 0)
        except Exception:
            requested_width, requested_height = 0, 0
        if model_id == "hidream":
            width, height = self._hidream_reference_safe_size(image_paths, requested_width, requested_height)
        elif requested_width > 0 and requested_height > 0:
            ok_size, _msg, width, height = self._validate_size(requested_width, requested_height)
            if not ok_size:
                width, height = self._default_edit_size(model_id, image_paths)
        else:
            width, height = self._default_edit_size(model_id, image_paths)
        size_note = ""
        if model_id == "hidream" and requested_width > 0 and requested_height > 0:
            if (int(width), int(height)) != (int(requested_width), int(requested_height)):
                size_note = f"\n\nHiDream uses fixed safe edit sizes. Resolution adjusted to {int(width)}×{int(height)}."
        seed = random.randint(1, 2_147_483_647)
        planned_output_path = ""
        try:
            if model_id == "flux_klein":
                ok = self._enqueue_flux_klein(prompt, cfg, width, height, seed, ref_images=image_paths)
            else:
                planned_output_path = self._planned_hidream_output_path(cfg, "edit", seed)
                ok = self._enqueue_hidream(prompt, cfg, width, height, seed, refs=image_paths, mode="edit", forced_output_path=planned_output_path)
        except Exception as exc:
            ok = False
            err = str(exc)
        else:
            err = ""
        self._clear_state()
        if ok:
            return AssistantRouteResult(True, f"Queued {label} image edit: `{prompt}`." + size_note, True, model_id, label, prompt, width, height, time.time(), "edit", tuple(image_paths), planned_output_path)
        return AssistantRouteResult(True, f"I could not add the {label} edit job to the FrameVision queue." + (f"\n\nError: {err}" if err else ""), False, model_id, label, prompt, width, height, time.time(), "edit", tuple(image_paths), planned_output_path)

    def _enqueue_zimage_gguf(self, prompt: str, cfg: Dict[str, Any], width: int, height: int, seed: int) -> bool:
        out_dir = str(self.root / str(cfg.get("output_dir") or "output/photo/txt2img"))
        folder = self.root / str(cfg.get("folder") or "models/Z-Image-Turbo GGUF")
        best_diffusion = self._find_best_zimage_diffusion_gguf(folder)
        best_instruct = self._find_best_zimage_instruct_gguf(folder)
        vae_path = folder / str(cfg.get("required_vae") or "ae.safetensors")
        args = {
            "type": "txt2img",
            "engine": "zimage_gguf",
            "prompt": prompt,
            "negative": "",
            "seed": seed,
            "seed_policy": "fixed",
            "batch": 1,
            "cfg_scale": float(cfg.get("default_cfg", 0.0)),
            "steps": int(cfg.get("default_steps", 8)),
            "width": int(width),
            "height": int(height),
            "output": out_dir,
            "show_in_player": True,
            "use_queue": True,
            "sampler": str(cfg.get("default_sampler") or "euler"),
            "format": "png",
            "filename_template": str(cfg.get("filename_template") or "z_img_{seed}_{idx:03d}.png"),
            "fit_check": True,
            "created_at": time.time(),
            "assistant_origin": "llama_chat",
            "assistant_chat_only": bool(self._assistant_chat_only_results_enabled()),
            "model": "Z-Image-Turbo",
            "model_name": "Z-Image-Turbo",
            "vram_profile": "Auto",
            "gpu_index": 0,
            "threads": 0,
            "attn_slicing": False,
            "vae_device": "Auto",
            "init_image_enabled": False,
            "init_image": "",
            "img2img_strength": 0.35,
        }
        if best_diffusion is not None:
            args["gguf_model_path"] = str(best_diffusion)
        if best_instruct is not None:
            args["gguf_instruct_path"] = str(best_instruct)
        if vae_path.exists():
            args["gguf_vae_path"] = str(vae_path)
        try:
            payload = dict(args, run_now=False)
            from helpers.queue_adapter import enqueue_txt2img  # type: ignore
            return bool(enqueue_txt2img(payload))
        except Exception:
            try:
                from queue_adapter import enqueue_txt2img  # type: ignore
                return bool(enqueue_txt2img(dict(args, run_now=False)))
            except Exception:
                job = {
                    "id": uuid.uuid4().hex,
                    "type": "txt2img",
                    "title": "Z-Image GGUF: " + prompt.replace("\n", " ")[:80],
                    "engine": "zimage_gguf",
                    "args": args,
                    "gguf_model_path": str(best_diffusion) if best_diffusion is not None else "",
                    "gguf_instruct_path": str(best_instruct) if best_instruct is not None else "",
                    "gguf_vae_path": str(vae_path) if vae_path.exists() else "",
                    "out_dir": out_dir,
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "pending",
                }
                return self._write_pending_job(job)

    def _flux_path_blob(self, path: Path) -> str:
        try:
            return "/".join(part.lower() for part in path.parts)
        except Exception:
            return str(path).replace("\\", "/").lower()

    def _find_best_flux_diffusion_gguf(self, folder: Path) -> Optional[Path]:
        try:
            cands = []
            for p in folder.rglob("*.gguf"):
                low_name = p.name.lower()
                low_path = self._flux_path_blob(p)
                is_unet = ("/unet/" in low_path) or ("unet" in low_path)
                is_flux = ("flux" in low_name) or ("klein" in low_name)
                is_text = ("qwen" in low_name) or ("instruct" in low_name) or ("text_encoder" in low_path) or ("text_encoders" in low_path)
                if (is_unet or is_flux) and not is_text:
                    cands.append(p)
            if not cands:
                return None
            cands.sort(key=lambda p: (self._zimage_quant_score(p.name)[0], p.stat().st_size, p.name.lower()))
            return cands[-1]
        except Exception:
            return None

    def _find_best_flux_llm_gguf(self, folder: Path) -> Optional[Path]:
        try:
            cands = []
            for p in folder.rglob("*.gguf"):
                low_name = p.name.lower()
                low_path = self._flux_path_blob(p)
                is_text = ("qwen" in low_name) or ("llm" in low_name) or ("instruct" in low_name) or ("text_encoder" in low_path) or ("text_encoders" in low_path)
                if is_text:
                    cands.append(p)
            if not cands:
                return None
            cands.sort(key=lambda p: (self._zimage_quant_score(p.name)[0], p.stat().st_size, p.name.lower()))
            return cands[-1]
        except Exception:
            return None

    def _find_best_flux_vae(self, folder: Path) -> Optional[Path]:
        try:
            cands = []
            for p in folder.rglob("*.safetensors"):
                low_name = p.name.lower()
                low_path = self._flux_path_blob(p)
                if ("vae" in low_name) or ("ae" in low_name) or ("/vae/" in low_path):
                    cands.append(p)
            if not cands:
                cands = list(folder.rglob("*.safetensors"))
            cands.sort(key=lambda p: (("ae" in p.name.lower() or "vae" in p.name.lower() or "vae" in self._flux_path_blob(p)), p.stat().st_size, p.name.lower()))
            return cands[-1] if cands else None
        except Exception:
            return None

    def _find_hidream_model_key(self, folder: Path) -> str:
        # Prefer Dev BF16, then Dev FP8, matching the FrameVision HiDream default preference.
        choices = [
            ("dev", "HiDream-O1-Image-Dev-BF16"),
            ("dev_fp8", "HiDream-O1-Image-Dev-FP8"),
            ("dev_2604_bf16", "HiDream-O1-Image-Dev-2604-BF16"),
            ("base", "HiDream-O1-Image-BF16"),
            ("base_fp8", "HiDream-O1-Image-FP8"),
        ]
        try:
            for key, name in choices:
                if (folder / name).exists():
                    return key
        except Exception:
            pass
        return ""

    def _hidream_edit_prompt(self, prompt: str, refs: list[str]) -> str:
        """Give HiDream the missing edit/reference context for chat-triggered jobs.

        The HiDream UI is explicit that the image in the edit tab is a reference
        image. The chat router previously passed only the short user edit text,
        so HiDream could treat the image more like loose inspiration and rebuild
        a pale/new scene. This prompt keeps the first attachment as the source
        image while still allowing bigger requested changes such as background
        replacement.
        """
        user_prompt = str(prompt or "").strip()
        ref_names = [Path(p).name for p in (refs or []) if str(p).strip()]
        role_lines = []
        if ref_names:
            role_lines.append(f"Reference image 1 is the source image to edit: {ref_names[0]}.")
            for i, name in enumerate(ref_names[1:], start=2):
                role_lines.append(f"Reference image {i} is additional visual guidance: {name}.")
        guidance = (
            "Edit the source image, do not create an unrelated new image. "
            "Keep the main subject, face/identity, clothing, body shape, camera angle, framing, and overall realism from reference image 1 unless the user specifically asks to change them. "
            "Apply the requested change cleanly and naturally. For background replacement, keep the foreground subject from the source image and replace only the background/scene."
        )
        parts = [guidance, *role_lines]
        if user_prompt:
            parts.append(f"Requested edit: {user_prompt}")
        return "\n".join(parts).strip()

    def _enqueue_chroma(self, prompt: str, cfg: Dict[str, Any], width: int, height: int, seed: int) -> bool:
        data = {
            "prompt": prompt,
            "negative": "",
            "width": int(width),
            "height": int(height),
            "steps": int(cfg.get("default_steps", 30)),
            "guidance": float(cfg.get("default_cfg", 3.0)),
            "seed": int(seed),
            "max_sequence_length": 256,
            "offload_cpu": True,
            "use_queue": True,
            "assistant_origin": "llama_chat",
            "assistant_chat_only": bool(self._assistant_chat_only_results_enabled()),
        }
        try:
            from helpers.queue_adapter import enqueue_chroma_generate  # type: ignore
            return bool(enqueue_chroma_generate(data))
        except Exception:
            try:
                from queue_adapter import enqueue_chroma_generate  # type: ignore
                return bool(enqueue_chroma_generate(data))
            except Exception:
                job = {
                    "id": uuid.uuid4().hex,
                    "type": "chroma",
                    "backend": "chroma",
                    "title": "Chroma: " + prompt.replace("\n", " ")[:80],
                    "prompt": prompt,
                    "args": data,
                    "out_dir": str(self.root / str(cfg.get("output_dir") or "output/images/chroma")),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "pending",
                }
                return self._write_pending_job(job)

    def _find_sd_cli(self) -> Path:
        candidates = [
            self.root / "presets" / "bin" / "sd-cli.exe",
            self.root / "presets" / "bin" / "sd-cli",
            self.root / "presets" / "bin" / "sd.exe",
            self.root / "sd-cli.exe",
            self.root / "sd.exe",
        ]
        for p in candidates:
            try:
                if p.exists() and p.is_file():
                    return p
            except Exception:
                pass
        return candidates[0]

    def _enqueue_flux_klein(self, prompt: str, cfg: Dict[str, Any], width: int, height: int, seed: int, ref_images: Optional[list[str]] = None) -> bool:
        folder = self.root / str(cfg.get("folder") or "models/klein4b_gguf")
        out_dir = self.root / str(cfg.get("output_dir") or "output/edits/flux_klein")
        out_dir.mkdir(parents=True, exist_ok=True)
        diffusion = self._find_best_flux_diffusion_gguf(folder)
        llm = self._find_best_flux_llm_gguf(folder)
        vae = self._find_best_flux_vae(folder)
        out_name = f"flux_klein_{int(time.time())}_{seed}.png"
        ref_images = [str(Path(p).resolve()) for p in (ref_images or []) if str(p).strip() and Path(str(p)).exists()]

        class _Text:
            def __init__(self, value=""):
                self._value = str(value or "")
            def toPlainText(self):
                return self._value
            def text(self):
                return self._value
            def setPlainText(self, value):
                self._value = str(value or "")
            def setText(self, value):
                self._value = str(value or "")

        class _Spin:
            def __init__(self, value=0):
                self._value = value
            def value(self):
                return self._value
            def setValue(self, value):
                self._value = value

        class _Check:
            def __init__(self, checked=False):
                self._checked = bool(checked)
            def isChecked(self):
                return bool(self._checked)
            def setChecked(self, checked):
                self._checked = bool(checked)

        class _Combo:
            def __init__(self, value=""):
                self._value = str(value or "")
            def currentData(self):
                return self._value
            def currentText(self):
                return self._value
            def setCurrentText(self, value):
                self._value = str(value or "")

        class _Label:
            def setText(self, *_args, **_kwargs):
                pass
            def setStyleSheet(self, *_args, **_kwargs):
                pass

        # Prefer the real queue adapter used by the Flux Klein UI itself.
        # The adapter expects a real-ish Flux UI object, not only cfg.prompt.
        try:
            widget_stub = SimpleNamespace(
                paths=SimpleNamespace(
                    root=str(self.root),
                    sd_cli=str(self._find_sd_cli()),
                    model_dir=str(folder),
                    out_dir=str(out_dir),
                ),
                models=SimpleNamespace(
                    diffusion_model=str(diffusion or ""),
                    llm_model=str(llm or ""),
                    vae_file=str(vae or ""),
                    lora_file="",
                ),
                cfg=SimpleNamespace(
                    prompt=str(prompt or ""),
                    negative="",
                    width=int(width),
                    height=int(height),
                    steps=int(cfg.get("default_steps", 4)),
                    cfg_scale=float(cfg.get("default_cfg", 1.0)),
                    seed=int(seed),
                    random_seed=False,
                    sampling_method="euler",
                    diffusion_fa=True,
                    offload_to_cpu=False,
                    vae_tiling=False,
                    out_name=str(out_name),
                    lora_strength=1.0,
                    ref_images=list(ref_images),
                    use_queue=True,
                ),
            )

            # UI-like fields used by flux_klein_editor_ui._collect_state and/or queue_adapter.
            widget_stub.prompt_edit = _Text(prompt)
            widget_stub.neg_edit = _Text("")
            widget_stub.sdcli_edit = _Text(widget_stub.paths.sd_cli)
            widget_stub.modeldir_edit = _Text(str(folder))
            widget_stub.width_spin = _Spin(int(width))
            widget_stub.height_spin = _Spin(int(height))
            widget_stub.steps_spin = _Spin(int(cfg.get("default_steps", 4)))
            widget_stub.cfg_spin = _Spin(float(cfg.get("default_cfg", 1.0)))
            widget_stub.seed_spin = _Spin(int(seed))
            widget_stub.chk_rand_seed = _Check(False)
            widget_stub.sampling_combo = _Combo("euler")
            widget_stub.chk_diffusion_fa = _Check(True)
            widget_stub.chk_offload_cpu = _Check(False)
            widget_stub.chk_vae_tiling = _Check(False)
            widget_stub.out_name_edit = _Text(out_name)
            widget_stub.lora_strength_spin = _Spin(1.0)
            widget_stub.chk_use_queue = _Check(True)
            widget_stub.flux_combo = _Combo(str(diffusion or ""))
            widget_stub.llm_combo = _Combo(str(llm or ""))
            widget_stub.vae_combo = _Combo(str(vae or ""))
            widget_stub.lora_combo = _Combo("")
            widget_stub.lbl_status = _Label()
            try:
                widget_stub.assistant_origin = "llama_chat"
                widget_stub.assistant_chat_only = bool(self._assistant_chat_only_results_enabled())
                widget_stub.cfg.assistant_origin = "llama_chat"
                widget_stub.cfg.assistant_chat_only = bool(self._assistant_chat_only_results_enabled())
            except Exception:
                pass

            def _collect_state_stub():
                widget_stub.paths.sd_cli = str(self._find_sd_cli())
                widget_stub.paths.model_dir = str(folder)
                widget_stub.paths.out_dir = str(out_dir)
                widget_stub.models.diffusion_model = str(diffusion or "")
                widget_stub.models.llm_model = str(llm or "")
                widget_stub.models.vae_file = str(vae or "")
                widget_stub.models.lora_file = ""
                widget_stub.cfg.prompt = str(prompt or "")
                widget_stub.cfg.negative = ""
                widget_stub.cfg.width = int(width)
                widget_stub.cfg.height = int(height)
                widget_stub.cfg.steps = int(cfg.get("default_steps", 4))
                widget_stub.cfg.cfg_scale = float(cfg.get("default_cfg", 1.0))
                widget_stub.cfg.seed = int(seed)
                widget_stub.cfg.random_seed = False
                widget_stub.cfg.sampling_method = "euler"
                widget_stub.cfg.diffusion_fa = True
                widget_stub.cfg.offload_to_cpu = False
                widget_stub.cfg.vae_tiling = False
                widget_stub.cfg.out_name = str(out_name)
                widget_stub.cfg.lora_strength = 1.0
                widget_stub.cfg.ref_images = list(ref_images)
                widget_stub.cfg.use_queue = True

            widget_stub._collect_state = _collect_state_stub
            widget_stub._save_settings = lambda *a, **k: None
            widget_stub._append_log = lambda *a, **k: None
            widget_stub._apply_run_mode_ui = lambda *a, **k: None
            widget_stub._open_queue_tab = lambda *a, **k: None

            try:
                from helpers.queue_adapter import enqueue_flux_klein_from_widget as _enqueue_flux_klein  # type: ignore
            except Exception:
                from queue_adapter import enqueue_flux_klein_from_widget as _enqueue_flux_klein  # type: ignore

            jid = _enqueue_flux_klein(widget_stub)
            self._router_log(
                f"Flux Klein queued via adapter -> {jid} | prompt={prompt!r} | "
                f"model={diffusion} | llm={llm} | vae={vae} | refs={len(ref_images)} | out={out_name}"
            )
            return bool(jid)
        except Exception as e:
            self._router_log(f"Flux Klein adapter enqueue failed: {e}")
            raise RuntimeError(f"Flux Klein could not be queued: {e}")

    def _planned_hidream_output_path(self, cfg: Dict[str, Any], mode: str, seed: int) -> str:
        out_dir = self.root / str(cfg.get("output_dir") or "output/hidream")
        out_dir.mkdir(parents=True, exist_ok=True)
        folder = self.root / str(cfg.get("folder") or "models/hidream_bf16")
        model_key = self._find_hidream_model_key(folder) or "dev"
        safe_mode = "edit" if str(mode).lower().strip() == "edit" else "create"
        return str(out_dir / f"hidream_{model_key}_{safe_mode}_{int(time.time())}_{int(seed)}.png")

    def _enqueue_hidream(self, prompt: str, cfg: Dict[str, Any], width: int, height: int, seed: int, refs: Optional[list[str]] = None, mode: str = "create", forced_output_path: str = "") -> bool:
        """Queue HiDream through the real FrameVision queue adapter.

        The first image-edit patch wrote a generic pending JSON. That can sit in
        pending forever on installs where the main worker only knows the
        HiDream queue-adapter payload. Use the same adapter path as hidream_ui.py
        and provide a small UI-compatible stub.
        """
        folder = self.root / str(cfg.get("folder") or "models/hidream_bf16")
        out_dir = self.root / str(cfg.get("output_dir") or "output/hidream")
        out_dir.mkdir(parents=True, exist_ok=True)
        model_key = self._find_hidream_model_key(folder) or "dev"
        refs = [str(Path(p).resolve()) for p in (refs or []) if str(p).strip() and Path(str(p)).exists()]
        mode = "edit" if refs or str(mode).lower().strip() == "edit" else "create"
        if refs:
            width, height = self._hidream_reference_safe_size(refs, width, height)
            prompt = self._hidream_edit_prompt(prompt, refs)
        output_path = Path(str(forced_output_path or "")).expanduser() if str(forced_output_path or "").strip() else out_dir / f"hidream_{model_key}_{mode}_{int(time.time())}_{seed}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        settings = {
            "width": int(width),
            "height": int(height),
            "steps": 28 if model_key.startswith("dev") else 50,
            "guidance_scale": 0.0 if model_key.startswith("dev") else 5.0,
            "shift": 1.0 if model_key.startswith("dev") else 3.0,
            "seed": int(seed),
            "scheduler_name": "flash",
            "timesteps": "dev" if model_key.startswith("dev") else "none",
            # Match the HiDream UI/CLI defaults. The previous 0/1/0 values were
            # not the normal UI path and could break some reference/edit runs.
            "noise_scale_start": 7.5,
            "noise_scale_end": 7.5,
            "noise_clip_std": 2.5,
            "negative_prompt": "",
            "offload_settings": {"try_auto_cpu_offload": False, "offload_folder": str(self.root / "temp" / "hidream_offload")},
        }

        class _Text:
            def __init__(self, value=""):
                self._value = str(value or "")
            def toPlainText(self):
                return self._value
            def text(self):
                return self._value
            def setPlainText(self, value):
                self._value = str(value or "")
            def setText(self, value):
                self._value = str(value or "")

        class _Check:
            def __init__(self, checked=False):
                self._checked = bool(checked)
            def isChecked(self):
                return bool(self._checked)
            def setChecked(self, checked):
                self._checked = bool(checked)

        class _RefList:
            def __init__(self, paths):
                self._paths = list(paths or [])
            def paths(self):
                return list(self._paths)
            def references(self):
                return [{"path": p, "role": "Main subject" if i == 0 else "General reference"} for i, p in enumerate(self._paths)]

        class _Combo:
            def __init__(self, value=""):
                self._value = str(value or "")
            def currentData(self):
                return self._value
            def currentText(self):
                return self._value

        class _Spin:
            def __init__(self, value=0):
                self._value = value
            def value(self):
                return self._value

        # Prefer the real adapter. This keeps the job shape identical to jobs
        # created from the HiDream tab, so the main worker actually starts it.
        try:
            try:
                from helpers.queue_adapter import enqueue_hidream_from_widget as _enqueue_hidream  # type: ignore
            except Exception:
                from queue_adapter import enqueue_hidream_from_widget as _enqueue_hidream  # type: ignore

            stub = SimpleNamespace()
            stub.assistant_origin = "llama_chat"
            stub.assistant_chat_only = bool(self._assistant_chat_only_results_enabled())
            stub.fv_root = str(self.root)
            stub.output_dir_edit = _Text(str(out_dir))
            stub.prompt_edit = _Text(prompt)
            stub.edit_prompt = _Text(prompt)
            stub.multi_prompt = _Text(prompt)
            stub.negative_prompt_edit = _Text("")
            stub.edit_negative_prompt_edit = _Text("")
            stub.ref_list = _RefList(refs)
            stub.multi_ref_list = _RefList(refs)
            # Match the HiDream tab default: this checkbox is normally off.
            # Keeping it on from chat made some background replacement edits act
            # like weak reference rebuilds instead of a clean edit.
            stub.keep_aspect = _Check(False)
            stub.keep_original_aspect = _Check(False)
            stub.framevision_queue_check = _Check(True)
            stub.model_combo = _Combo(model_key)
            stub.width_spin = _Spin(int(width))
            stub.height_spin = _Spin(int(height))
            stub.steps_spin = _Spin(int(settings["steps"]))
            stub.cfg_spin = _Spin(float(settings["guidance_scale"]))
            stub.shift_spin = _Spin(float(settings["shift"]))
            stub.seed_spin = _Spin(int(seed))

            def _current_model_key():
                return model_key
            def _is_full_variant(key=None):
                key = str(key or model_key)
                return key in ("base", "base_fp8")
            def _current_offload_settings():
                return dict(settings["offload_settings"])
            def _selected_generation_settings(_ui_key="main", negative_prompt=None):
                data = dict(settings)
                # HiDream UI uses reference-safe buckets for edit/multi jobs.
                # Keep that behavior here too, because small/generic sizes can
                # cause pale/blank reference results.
                if str(_ui_key or "").lower() in ("edit", "multi") and refs:
                    safe_w, safe_h = self._hidream_reference_safe_size(refs, data.get("width", 0), data.get("height", 0))
                    data["width"] = int(safe_w)
                    data["height"] = int(safe_h)
                if _is_full_variant(model_key) and negative_prompt:
                    data["negative_prompt"] = str(negative_prompt or "")
                else:
                    data["negative_prompt"] = ""
                return data
            def _reference_safe_settings(_ui_key="edit", incoming=None):
                data = dict(incoming or settings)
                if refs:
                    safe_w, safe_h = self._hidream_reference_safe_size(refs, data.get("width", 0), data.get("height", 0))
                    data["width"] = int(safe_w)
                    data["height"] = int(safe_h)
                return data
            def _generation_settings_from_widgets(_ui_key="edit"):
                return _reference_safe_settings(_ui_key, settings)
            def _output_path(prefix="hidream"):
                return output_path
            def _multi_reference_prompt_with_roles(raw_prompt, references):
                # Same idea as the UI: keep role hints in the prompt for multi-ref jobs.
                lines = [str(raw_prompt or prompt).strip()]
                for ref in references or []:
                    role = str(ref.get("role") or "General reference")
                    path = Path(str(ref.get("path") or "")).name
                    if path:
                        lines.append(f"Reference {role}: {path}")
                return "\n".join([x for x in lines if x])

            stub.current_model_key = _current_model_key
            stub.is_full_variant = _is_full_variant
            stub.current_offload_settings = _current_offload_settings
            stub.selected_generation_settings = _selected_generation_settings
            stub.reference_safe_settings = _reference_safe_settings
            stub.generation_settings_from_widgets = _generation_settings_from_widgets
            stub.output_path = _output_path
            stub.multi_reference_prompt_with_roles = _multi_reference_prompt_with_roles
            stub.log = lambda *a, **k: None
            stub.switch_to_framevision_queue_tab = lambda *a, **k: None

            adapter_mode = "edit" if mode == "edit" else "create"
            jid = _enqueue_hidream(stub, mode=adapter_mode)
            self._router_log(
                f"HiDream queued via adapter -> {jid} | mode={adapter_mode} | model_key={model_key} | "
                f"prompt={prompt!r} | refs={len(refs)} | output={output_path}"
            )
            return bool(jid)
        except Exception as e:
            self._router_log(f"HiDream adapter enqueue failed: {e}")

        # Last-resort fallback for older installs. This includes the CLI args the
        # worker needs if it supports direct command jobs, but the adapter path
        # above is the expected/current path.
        env_py = self.root / "environments" / ".hidream_dev" / "python.exe"
        cli_py = self.root / "helpers" / "hidream_cli.py"
        args = [
            str(env_py), str(cli_py),
            "--model_key", model_key,
            "--width", str(int(width)),
            "--height", str(int(height)),
            "--steps", str(int(settings["steps"])),
            "--guidance_scale", str(float(settings["guidance_scale"])),
            "--shift", str(float(settings["shift"])),
            "--seed", str(int(seed)),
            "--scheduler_name", str(settings["scheduler_name"]),
            "--timesteps", str(settings["timesteps"]),
            "--noise_scale_start", str(float(settings["noise_scale_start"])),
            "--noise_scale_end", str(float(settings["noise_scale_end"])),
            "--noise_clip_std", str(float(settings["noise_clip_std"])),
            "--output_image", str(output_path),
            "--prompt", str(prompt or ""),
            "--resolution_mode", "framevision",
            "--device_map", "cuda",
        ]
        if refs:
            args.extend(["--ref_images", *refs])
        job = {
            "id": uuid.uuid4().hex,
            "type": "hidream",
            "backend": "hidream",
            "mode": mode,
            "title": ("HiDream edit: " if mode == "edit" else "HiDream: ") + prompt.replace("\n", " ")[:80],
            "model_key": model_key,
            "prompt": prompt,
            "refs": refs,
            "keep_original_aspect": False,
            "settings": settings,
            "output_path": str(output_path),
            "out_dir": str(out_dir),
            "cmd": args,
            "args": args,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending",
        }
        return self._write_pending_job(job)

    def _enqueue_lens(self, prompt: str, cfg: Dict[str, Any], width: int, height: int, seed: int) -> bool:
        out_dir = str(self.root / str(cfg.get("output_dir") or "output/lens_turbo_u4"))
        params = {
            "prompt": prompt,
            "negative": "",
            "aspect_ratio": self._aspect_label(width, height),
            "steps": int(cfg.get("default_steps", 4)),
            "cfg": float(cfg.get("default_cfg", 0.0)),
            "images": 1,
            "seed": int(seed),
            "output_dir": out_dir,
            "repo_id": str(cfg.get("repo_id") or "WaveCut/Lens-Turbo-SDNQ-uint4-static"),
            "offload": True,
            "offline": True,
            "keep_model_loaded": False,
            "use_custom_resolution": True,
            "custom_width": int(width),
            "custom_height": int(height),
            "base_resolution": 0,
            "assistant_origin": "llama_chat",
            "assistant_chat_only": bool(self._assistant_chat_only_results_enabled()),
        }
        adapter_job = {"prompt": prompt, "seed": int(seed), "params": params, "output_dir": out_dir, "run_now": False}
        try:
            from helpers.queue_adapter import enqueue_lens_turbo_u4  # type: ignore
            return bool(enqueue_lens_turbo_u4(adapter_job))
        except Exception:
            try:
                from queue_adapter import enqueue_lens_turbo_u4  # type: ignore
                return bool(enqueue_lens_turbo_u4(adapter_job))
            except Exception:
                job = {
                    "id": uuid.uuid4().hex,
                    "type": "lens_turbo_u4",
                    "title": "Lens Turbo U4: " + prompt.replace("\n", " ")[:80],
                    "args": params,
                    "out_dir": out_dir,
                    "model": params["repo_id"],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "pending",
                }
                return self._write_pending_job(job)

    def _router_log(self, message: str) -> None:
        try:
            log_dir = self.root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "fv_assistant_router.log"
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {message}\n")
        except Exception:
            pass

    def _write_pending_job(self, job: Dict[str, Any]) -> bool:
        pending = self.root / "jobs" / "pending"
        pending.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        jid = str(job.get("id") or uuid.uuid4().hex)
        path = pending / f"assistant_{stamp}_{jid[:8]}.json"
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
        return path.exists()

    def _aspect_label(self, width: int, height: int) -> str:
        try:
            ratio = float(width) / float(height)
        except Exception:
            ratio = 1.0
        if abs(ratio - 1.0) < 0.08:
            return "1:1"
        if ratio > 1.0:
            return "16:9"
        return "9:16"
