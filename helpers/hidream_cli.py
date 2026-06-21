#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoProcessor
from PIL import Image


os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


APP_ROOT = Path(__file__).resolve().parents[1]
HIDREAM_ROOT = APP_ROOT / "models" / "hidream_bf16"
REPO_DIR = HIDREAM_ROOT / "HiDream-O1-Image"

MODEL_MAP = {
    "base": {
        "label": "Base / Full BF16",
        "folder": "HiDream-O1-Image-BF16",
        "variant": "full",
        "weight_dtype": "bf16",
        "default_steps": 50,
        "default_guidance": 5.0,
        "default_shift": 3.0,
        "default_scheduler": "flash",
        "default_timesteps": "none",
    },
    "base_fp8": {
        "label": "Base / Full FP8",
        "folder": "HiDream-O1-Image-FP8",
        "variant": "full",
        "weight_dtype": "fp8_e4m3fn",
        "default_steps": 50,
        "default_guidance": 5.0,
        "default_shift": 3.0,
        "default_scheduler": "flash",
        "default_timesteps": "none",
    },
    "dev": {
        "label": "Dev BF16",
        "folder": "HiDream-O1-Image-Dev-BF16",
        "variant": "dev",
        "weight_dtype": "bf16",
        "default_steps": 28,
        "default_guidance": 0.0,
        "default_shift": 1.0,
        "default_scheduler": "flash",
        "default_timesteps": "dev",
    },
    "dev_2604_bf16": {
        "label": "Dev 2604 BF16",
        "folder": "HiDream-O1-Image-Dev-2604-BF16",
        "variant": "dev",
        "weight_dtype": "bf16",
        "default_steps": 28,
        "default_guidance": 0.0,
        "default_shift": 1.0,
        "default_scheduler": "flash",
        "default_timesteps": "dev",
    },
    "dev_fp8": {
        "label": "Dev FP8",
        "folder": "HiDream-O1-Image-Dev-FP8",
        "variant": "dev",
        "weight_dtype": "fp8_e4m3fn",
        "default_steps": 28,
        "default_guidance": 0.0,
        "default_shift": 1.0,
        "default_scheduler": "flash",
        "default_timesteps": "dev",
    },
}

FLOAT8_DTYPE_NAMES = {
    getattr(torch, "float8_e4m3fn", None): "fp8_e4m3fn",
    getattr(torch, "float8_e5m2", None): "fp8_e5m2",
}
FLOAT8_DTYPE_NAMES = {k: v for k, v in FLOAT8_DTYPE_NAMES.items() if k is not None}

SAFETENSORS_DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F8_E4M3": getattr(torch, "float8_e4m3fn", None),
    "F8_E5M2": getattr(torch, "float8_e5m2", None),
}
SAFETENSORS_DTYPE_MAP = {k: v for k, v in SAFETENSORS_DTYPE_MAP.items() if v is not None}

# FrameVision preset buckets. The generated image is saved at the real generation size.
FRAMEVISION_RESOLUTIONS = [
    (640, 480),
    (1024, 768),
    (832, 480),
    (1024, 576),
    (1280, 704),
    (1600, 896),
    (1920, 1088),
    (2560, 1440),
    (480, 640),
    (768, 1024),
    (480, 832),
    (576, 1024),
    (704, 1280),
    (896, 1600),
    (1088, 1920),
    (1440, 2560),
    (512, 512),
    (768, 768),
    (1024, 1024),
    (1536, 1536),
    (2048, 2048),
    (2304, 1728),
    (1728, 2304),
    (2496, 1664),
    (1664, 2496),
    (3104, 1312),
    (1312, 3104),
    (2304, 1792),
    (1792, 2304),
]


# Reference/edit/multi-reference safe buckets.
# The upstream HiDream reference-conditioning path scales/crops references internally.
# Small landscape buckets such as 1024x576 produced grey/pink/blank outputs in testing.
# These buckets are intentionally moderate defaults, not the highest native sizes.
REFERENCE_SAFE_RESOLUTIONS = [
    (1600, 896),   # landscape balanced/default
    (1920, 1088),  # landscape high
    (896, 1600),   # portrait balanced/default
    (1088, 1920),  # portrait high
    (1024, 1024),  # square balanced
    (1536, 1536),  # square high-ish
    (2048, 2048),  # square native/high
]

REFERENCE_SAFE_LANDSCAPE_MIN = (1600, 896)
REFERENCE_SAFE_PORTRAIT_MIN = (896, 1600)
REFERENCE_SAFE_SQUARE_MIN = (1024, 1024)


def _is_reference_workflow(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "ref_images", None))


def _reference_safe_resolution(width: int, height: int, ref_count: int = 1) -> tuple[int, int]:
    """Pick a safer bucket for HiDream edit/reference workflows.

    Text-to-image can still use the full FrameVision bucket list. Reference workflows
    should avoid tiny landscape/portrait buckets because HiDream internally scales
    reference tensors relative to the selected output size.
    """
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return REFERENCE_SAFE_LANDSCAPE_MIN

    # Preserve orientation. Square-ish requests use square buckets.
    ratio = width / max(1, height)
    if 0.90 <= ratio <= 1.10:
        candidates = [p for p in REFERENCE_SAFE_RESOLUTIONS if p[0] == p[1]]
        minimum = REFERENCE_SAFE_SQUARE_MIN
    elif ratio > 1.10:
        candidates = [p for p in REFERENCE_SAFE_RESOLUTIONS if p[0] > p[1]]
        # For 3+ refs the repo halves ref max size, so avoid borderline small buckets.
        minimum = (1920, 1088) if ref_count >= 3 else REFERENCE_SAFE_LANDSCAPE_MIN
    else:
        candidates = [p for p in REFERENCE_SAFE_RESOLUTIONS if p[0] < p[1]]
        minimum = (1088, 1920) if ref_count >= 3 else REFERENCE_SAFE_PORTRAIT_MIN

    min_area = minimum[0] * minimum[1]
    candidates = [p for p in candidates if p[0] * p[1] >= min_area] or [minimum]

    requested_area = max(1, width * height)
    req_ratio = width / max(1, height)

    def score(pair: tuple[int, int]) -> tuple[float, float, int]:
        w, h = pair
        ratio_diff = abs((w / h) - req_ratio)
        # Prefer not to go lower than requested, but keep moderate defaults.
        under = 1 if (w * h) < requested_area else 0
        area_diff = abs((w * h) - requested_area) / requested_area
        return (ratio_diff, under, area_diff)

    return min(candidates, key=score)



FALLBACK_QWEN_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'user' %}<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'system' %}<|im_start|>system
{{ message['content'] }}<|im_end|>
{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""


def _dedupe_resolutions(items: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for w, h in items:
        pair = (int(w), int(h))
        if pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _closest_resolution(width: int, height: int) -> tuple[int, int]:
    width = int(width)
    height = int(height)
    if width <= 0 or height <= 0:
        return 1024, 1024

    requested = (width, height)
    resolutions = _dedupe_resolutions(FRAMEVISION_RESOLUTIONS)
    if requested in resolutions:
        return requested

    req_ratio = width / height
    req_area = width * height

    def score(pair: tuple[int, int]) -> tuple[float, float]:
        w, h = pair
        ratio_diff = abs((w / h) - req_ratio)
        area_diff = abs((w * h) - req_area) / max(1, req_area)
        return (ratio_diff, area_diff)

    return min(resolutions, key=score)


def patch_resolution_picker(mode: str) -> None:
    if mode == "native":
        print("[HiDream CLI] Resolution mode: native repo buckets")
        return

    import models.utils as utils_mod  # type: ignore

    original = list(getattr(utils_mod, "PREDEFINED_RESOLUTIONS", []))
    merged = _dedupe_resolutions(FRAMEVISION_RESOLUTIONS + original)
    utils_mod.PREDEFINED_RESOLUTIONS = merged
    utils_mod.find_closest_resolution = _closest_resolution

    import models.pipeline as pipeline_mod  # type: ignore

    pipeline_mod.find_closest_resolution = _closest_resolution

    print("[HiDream CLI] Resolution mode: FrameVision preset buckets")
    print("[HiDream CLI] FrameVision preset buckets enabled: 640x480, 1024x768, 832x480, 1024x576, 1280x704, 1600x896, 1920x1088, 2560x1440, plus portrait/square variants")


def read_chat_template_from_folder(folder: Path) -> str | None:
    for candidate in [
        folder / "chat_template.jinja",
        folder / "chat_template.json",
        folder / "tokenizer_config.json",
        folder / "processor_config.json",
    ]:
        if not candidate.exists():
            continue
        try:
            raw = candidate.read_text(encoding="utf-8")
            if candidate.suffix.lower() == ".json":
                data = json.loads(raw)
                template = data.get("chat_template") if isinstance(data, dict) else data
            else:
                template = raw
            if isinstance(template, str) and template.strip():
                print(f"[HiDream CLI] Loaded chat template from: {candidate}")
                return template
        except Exception as exc:
            print(f"[HiDream CLI] Could not read chat template {candidate}: {exc}")
    return None


def ensure_processor_chat_template(processor, model_dir: Path, get_tokenizer):
    tokenizer = get_tokenizer(processor)
    template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    if template:
        print("[HiDream CLI] Processor already has a chat template.")
    if not template:
        template = read_chat_template_from_folder(model_dir)
    if not template:
        template = FALLBACK_QWEN_CHAT_TEMPLATE
        print("[HiDream CLI] WARNING: no model chat template found; using built-in fallback template.")
    processor.chat_template = template
    try:
        tokenizer.chat_template = template
    except Exception:
        pass
    return tokenizer



def _resize_pilimage_fit_pad(pil_image, image_size, patch_size=16, resampler=Image.BICUBIC):
    """HiDream reference resize without center-cropping.

    The upstream resize_pilimage() uses resize-to-fill + center crop. That can
    chop faces/bodies in square and portrait reference/edit workflows. For
    FrameVision reference workflows, fit the whole image inside the patch-aligned
    canvas and pad instead of cropping.
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(max(1, x // 2) for x in pil_image.size), resample=Image.BOX
        )

    m = int(patch_size or 16)
    width, height = int(pil_image.width), int(pil_image.height)
    if width <= 0 or height <= 0:
        return pil_image

    s_max = int(image_size) * int(image_size)
    scale = (s_max / max(1, width * height)) ** 0.5

    # Fit inside the target area, never crop. Keep patch alignment.
    fit_w = max(m, int(round(width * scale)) // m * m)
    fit_h = max(m, int(round(height * scale)) // m * m)
    while fit_w * fit_h > s_max and (fit_w > m or fit_h > m):
        if fit_w >= fit_h and fit_w > m:
            fit_w -= m
        elif fit_h > m:
            fit_h -= m
        else:
            break

    resized = pil_image.resize((fit_w, fit_h), resample=resampler)

    # Pad to a patch-aligned canvas close to the fitted image. This preserves the
    # full reference image while avoiding odd dimensions.
    canvas_w = max(m, ((fit_w + m - 1) // m) * m)
    canvas_h = max(m, ((fit_h + m - 1) // m) * m)
    if canvas_w * canvas_h > s_max:
        canvas_w, canvas_h = fit_w, fit_h

    if (canvas_w, canvas_h) == resized.size:
        return resized

    # Use edge-color padding instead of black bars; this is less likely to be
    # treated as an intentional border by the vision/text encoder.
    try:
        pixels = [
            resized.getpixel((0, 0)),
            resized.getpixel((resized.width - 1, 0)),
            resized.getpixel((0, resized.height - 1)),
            resized.getpixel((resized.width - 1, resized.height - 1)),
        ]
        color = tuple(int(sum(p[i] for p in pixels) / len(pixels)) for i in range(3))
    except Exception:
        color = (127, 127, 127)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color)
    left = max(0, (canvas_w - resized.width) // 2)
    top = max(0, (canvas_h - resized.height) // 2)
    canvas.paste(resized, (left, top))
    return canvas


def _patch_hidream_reference_resize_no_crop(pipeline_mod) -> None:
    try:
        pipeline_mod.resize_pilimage = _resize_pilimage_fit_pad
        print("[HiDream CLI] Reference resize mode: fit+pad no-crop")
    except Exception as exc:
        print(f"[HiDream CLI] Warning: could not patch reference resize no-crop mode: {exc}")



def _dtype_from_safetensors(model_dir: Path) -> torch.dtype | None:
    candidates: list[Path] = []
    single = model_dir / "model.safetensors"
    if single.exists():
        candidates.append(single)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            for shard_name in sorted(set(index.get("weight_map", {}).values())):
                shard_path = model_dir / shard_name
                if shard_path.exists():
                    candidates.append(shard_path)
        except Exception:
            pass

    dtype_counts: dict[torch.dtype, int] = {}
    for path in candidates[:2]:
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    dtype_name = handle.get_slice(key).get_dtype()
                    dtype = SAFETENSORS_DTYPE_MAP.get(dtype_name)
                    if dtype is not None and torch.is_floating_point(torch.empty((), dtype=dtype)):
                        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        except Exception:
            continue

    for dtype in FLOAT8_DTYPE_NAMES:
        if dtype_counts.get(dtype, 0) > 0:
            return dtype
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        if dtype_counts.get(dtype, 0) > 0:
            return dtype
    return None


def _config_dtype_name(model_dir: Path) -> str:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return ""
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    config_dtype = str(config.get("dtype") or config.get("torch_dtype") or "")
    text_config = config.get("text_config") or {}
    return str(text_config.get("dtype") or text_config.get("torch_dtype") or config_dtype).lower()


def is_float8_dtype(dtype: torch.dtype) -> bool:
    return dtype in FLOAT8_DTYPE_NAMES


def dtype_label(dtype: torch.dtype) -> str:
    return FLOAT8_DTYPE_NAMES.get(dtype, str(dtype).replace("torch.", ""))


def compute_dtype_from_weight_dtype(weight_dtype: torch.dtype) -> torch.dtype:
    if is_float8_dtype(weight_dtype):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if weight_dtype == torch.float16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return weight_dtype


class _TorchNNProxy:
    def __init__(self, base_nn, operations):
        self._base_nn = base_nn
        self._operations = operations

    def __getattr__(self, name: str):
        if hasattr(self._operations, name):
            return getattr(self._operations, name)
        return getattr(self._base_nn, name)


def build_manual_fp8_operations(compute_dtype: torch.dtype):
    class ManualFp8Ops:
        class Linear(torch.nn.Linear):
            def forward(self, input):
                target_dtype = input.dtype if torch.is_floating_point(input) else compute_dtype
                weight = self.weight if self.weight.dtype == target_dtype else self.weight.to(dtype=target_dtype)
                bias = self.bias
                if bias is not None and bias.dtype != target_dtype:
                    bias = bias.to(dtype=target_dtype)
                return F.linear(input, weight, bias)

        class Embedding(torch.nn.Embedding):
            def forward(self, input):
                weight = self.weight if self.weight.dtype == compute_dtype else self.weight.to(dtype=compute_dtype)
                return F.embedding(
                    input,
                    weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )

        class Conv3d(torch.nn.Conv3d):
            def forward(self, input):
                target_dtype = input.dtype if torch.is_floating_point(input) else compute_dtype
                weight = self.weight if self.weight.dtype == target_dtype else self.weight.to(dtype=target_dtype)
                bias = self.bias
                if bias is not None and bias.dtype != target_dtype:
                    bias = bias.to(dtype=target_dtype)
                source = input if input.dtype == target_dtype else input.to(dtype=target_dtype)
                return F.conv3d(source, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    return ManualFp8Ops


def resolve_weight_dtype(model_dir: Path, configured_name: str) -> torch.dtype:
    if configured_name == "bf16":
        return torch.bfloat16
    if configured_name == "fp16":
        return torch.float16
    if configured_name == "fp32":
        return torch.float32
    if configured_name == "fp8_e4m3fn":
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is None:
            raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")
        return dtype
    if configured_name == "fp8_e5m2":
        dtype = getattr(torch, "float8_e5m2", None)
        if dtype is None:
            raise RuntimeError("This PyTorch build does not expose torch.float8_e5m2.")
        return dtype

    detected = _dtype_from_safetensors(model_dir)
    if detected is not None:
        return detected

    config_dtype = _config_dtype_name(model_dir)
    if "float8_e4m3fn" in config_dtype or "fp8_e4m3fn" in config_dtype:
        dtype = getattr(torch, "float8_e4m3fn", None)
        if dtype is not None:
            return dtype
    if "float8_e5m2" in config_dtype or "fp8_e5m2" in config_dtype:
        dtype = getattr(torch, "float8_e5m2", None)
        if dtype is not None:
            return dtype
    if "float16" in config_dtype or "fp16" in config_dtype:
        return torch.float16
    return torch.bfloat16


def _convert_matrix_params_to_dtype(model: torch.nn.Module, target_dtype: torch.dtype) -> int:
    converted = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not torch.is_floating_point(param) or param.dtype == target_dtype:
                continue
            if param.ndim < 2 or name.endswith(".bias"):
                continue
            param.data = param.data.to(dtype=target_dtype)
            converted += 1
    return converted


def _fp8_safety_recast(model: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    recast = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not is_float8_dtype(param.dtype):
                continue
            if param.ndim >= 2 and not name.endswith(".bias"):
                continue
            param.data = param.data.to(dtype=compute_dtype)
            recast += 1
    return recast


def _recast_all_float8_params(model: torch.nn.Module, compute_dtype: torch.dtype) -> int:
    recast = 0
    with torch.no_grad():
        for param in model.parameters():
            if not is_float8_dtype(param.dtype):
                continue
            param.data = param.data.to(dtype=compute_dtype)
            recast += 1
    return recast


def _recast_float8_prefixes(model: torch.nn.Module, prefixes: list[str], compute_dtype: torch.dtype) -> int:
    recast = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not is_float8_dtype(param.dtype):
                continue
            if not any(name.startswith(prefix) for prefix in prefixes):
                continue
            param.data = param.data.to(dtype=compute_dtype)
            recast += 1
    return recast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FrameVision HiDream CLI")
    parser.add_argument("--model_key", choices=list(MODEL_MAP.keys()), default="base")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--ref_images", nargs="*", default=[])
    parser.add_argument("--output_image", type=str, default=str(HIDREAM_ROOT / "results" / "hidream.png"))
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--scheduler_name", choices=["default", "flash"], default=None)
    parser.add_argument("--timesteps", choices=["none", "dev"], default=None)
    parser.add_argument("--keep_original_aspect", action="store_true")
    parser.add_argument("--noise_scale_start", type=float, default=7.5)
    parser.add_argument("--noise_scale_end", type=float, default=7.5)
    parser.add_argument("--noise_clip_std", type=float, default=2.5)
    parser.add_argument("--device_map", type=str, default="cuda", choices=["cuda", "auto"])
    parser.add_argument("--offload_folder", type=str, default=str(APP_ROOT / "temp" / "hidream_offload"))
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--resolution_mode", choices=["native", "framevision"], default="framevision")
    parser.add_argument("--disable_ref_safe_resolution", action="store_true", help="Disable FrameVision reference/edit safe bucket override.")
    parser.add_argument("--disable_ref_no_crop", action="store_true", help="Disable FrameVision fit+pad reference resize patch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not REPO_DIR.exists():
        raise RuntimeError(f"HiDream repo folder was not found: {REPO_DIR}")

    sys.path.insert(0, str(REPO_DIR))
    patch_resolution_picker(args.resolution_mode)

    import models.qwen3_vl_transformers as qwen3_vl_transformers  # type: ignore
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # type: ignore
    import models.pipeline as pipeline_mod  # type: ignore
    from models.pipeline import DEFAULT_TIMESTEPS, generate_image  # type: ignore
    from inference import add_special_tokens, get_tokenizer  # type: ignore

    def parse_timesteps(mode: str):
        if mode == "dev":
            return DEFAULT_TIMESTEPS
        return None

    info = MODEL_MAP[args.model_key]
    model_dir = HIDREAM_ROOT / info["folder"]

    if args.steps is None:
        args.steps = info["default_steps"]
    if args.guidance_scale is None:
        args.guidance_scale = info["default_guidance"]
    if args.shift is None:
        args.shift = info["default_shift"]
    if args.scheduler_name is None:
        args.scheduler_name = info["default_scheduler"]
    if args.timesteps is None:
        args.timesteps = info["default_timesteps"]

    if not model_dir.exists():
        raise RuntimeError(f"Selected model is not installed: {model_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU mode is not useful for this model.")
    if args.seed < 0:
        args.seed = random.randint(0, 2**31 - 1)

    output_path = Path(args.output_image).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ref_count = len([p for p in (args.ref_images or []) if str(p).strip()])
    reference_safe_active = bool(ref_count and args.resolution_mode == "framevision" and not args.disable_ref_safe_resolution)
    reference_no_crop_active = bool(ref_count and not args.disable_ref_no_crop)
    if reference_no_crop_active:
        _patch_hidream_reference_resize_no_crop(pipeline_mod)
    if args.resolution_mode == "framevision":
        if reference_safe_active:
            selected_w, selected_h = _reference_safe_resolution(args.width, args.height, ref_count)
        else:
            selected_w, selected_h = _closest_resolution(args.width, args.height)
    else:
        selected_w, selected_h = (args.width, args.height)

    print("[HiDream CLI] Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("[HiDream CLI] Repo:", REPO_DIR)
    print("[HiDream CLI] Model:", model_dir)
    print("[HiDream CLI] Model key:", args.model_key, "-", info["label"])
    print("[HiDream CLI] Output:", output_path)
    print("[HiDream CLI] Requested size:", f"{args.width}x{args.height}")
    print("[HiDream CLI] Selected generation size:", f"{selected_w}x{selected_h}")
    if ref_count:
        print(f"[HiDream CLI] Reference workflow: {ref_count} reference image(s)")
        if reference_safe_active:
            print("[HiDream CLI] Reference-safe bucket override: enabled")
        elif args.disable_ref_safe_resolution:
            print("[HiDream CLI] Reference-safe bucket override: disabled by --disable_ref_safe_resolution")
        if reference_no_crop_active:
            print("[HiDream CLI] Reference no-crop resize: enabled")
        elif args.disable_ref_no_crop:
            print("[HiDream CLI] Reference no-crop resize: disabled by --disable_ref_no_crop")
    if (selected_w, selected_h) != (args.width, args.height):
        print(f"[HiDream CLI] Note: requested {args.width}x{args.height} will use closest test bucket {selected_w}x{selected_h}")
    print("[HiDream CLI] Active generation settings:")
    print(f"  size: {selected_w}x{selected_h}")
    print(f"  seed: {args.seed}")
    print(f"  steps: {args.steps}")
    print(f"  guidance_scale: {args.guidance_scale}")
    print(f"  shift: {args.shift}")
    if args.scheduler_name == "flash":
        print("  scheduler_name: flash (FlashFlowMatchEulerDiscreteScheduler / Euler path)")
    else:
        print("  scheduler_name: default (FlowUniPCMultistepScheduler / UniPC path)")
    timesteps_list = parse_timesteps(args.timesteps)
    print(f"  timesteps: {args.timesteps}" + (f" ({len(timesteps_list)})" if timesteps_list else ""))
    print(f"  noise_scale_start: {args.noise_scale_start}")
    print(f"  noise_scale_end: {args.noise_scale_end}")
    print(f"  noise_clip_std: {args.noise_clip_std}")
    if args.negative_prompt and info["variant"] == "full":
        print("  negative_prompt: enabled")
    elif args.negative_prompt:
        print("  negative_prompt: ignored for Dev model")

    file_dtype = _dtype_from_safetensors(model_dir)
    weight_dtype = resolve_weight_dtype(model_dir, str(info.get("weight_dtype", "bf16")))
    compute_dtype = compute_dtype_from_weight_dtype(weight_dtype)
    print(f"[HiDream CLI] Weight dtype: {dtype_label(weight_dtype)}")
    print(f"[HiDream CLI] Compute dtype: {dtype_label(compute_dtype)}")

    print(f"[HiDream CLI] Loading processor and model ({dtype_label(weight_dtype)} weights)...")
    processor = AutoProcessor.from_pretrained(str(model_dir))
    tokenizer = ensure_processor_chat_template(processor, model_dir, get_tokenizer)
    add_special_tokens(tokenizer)

    load_dtype = compute_dtype if is_float8_dtype(weight_dtype) else weight_dtype
    load_kwargs = {"dtype": load_dtype}
    if args.device_map == "auto":
        offload_folder = Path(args.offload_folder).expanduser()
        offload_folder.mkdir(parents=True, exist_ok=True)
        load_kwargs["device_map"] = "auto"
        load_kwargs["offload_folder"] = str(offload_folder)
        if is_float8_dtype(weight_dtype):
            print("[HiDream CLI] Note: FP8 model selected with auto offload; initial load will use the FP8-safe compute dtype, then recast large tensors.")
    else:
        load_kwargs["device_map"] = "cuda"

    fp8_ops = build_manual_fp8_operations(compute_dtype) if is_float8_dtype(weight_dtype) else None
    if fp8_ops is not None:
        print("[HiDream CLI] Using standalone FP8 manual-cast ops.")
        operations = fp8_ops
        nn_proxy = _TorchNNProxy(qwen3_vl_transformers.nn, operations)
        original_nn = qwen3_vl_transformers.nn
        try:
            qwen3_vl_transformers.nn = nn_proxy
            model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), **load_kwargs).eval()
        finally:
            qwen3_vl_transformers.nn = original_nn
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_dir), **load_kwargs).eval()

    if is_float8_dtype(weight_dtype):
        if fp8_ops is not None:
            fp8_params = sum(1 for param in model.parameters() if param.dtype == weight_dtype)
            if file_dtype != weight_dtype or fp8_params == 0:
                converted = _convert_matrix_params_to_dtype(model, weight_dtype)
                print(f"[HiDream CLI] Converted {converted} large tensors to {dtype_label(weight_dtype)} after load.")
            recast = _fp8_safety_recast(model, compute_dtype)
            if recast:
                print(f"[HiDream CLI] Recast {recast} small/bias FP8 tensors to {dtype_label(compute_dtype)} for stable math.")
            visual_recast = _recast_float8_prefixes(model, ["model.visual."], compute_dtype)
            if visual_recast:
                print(f"[HiDream CLI] Recast {visual_recast} vision-path FP8 tensors to {dtype_label(compute_dtype)} for reference-image compatibility.")
        else:
            recast = _recast_all_float8_params(model, compute_dtype)
            if recast:
                print(f"[HiDream CLI] Recast {recast} FP8 tensors to {dtype_label(compute_dtype)} because the FP8 execution path was unavailable.")

    prompt = args.prompt
    if args.negative_prompt and info["variant"] == "full":
        prompt = prompt.strip()
        print("[HiDream CLI] Warning: HiDream pipeline has no separate negative-prompt argument; negative text is logged but not injected.")

    print("[HiDream CLI] Generating...")
    t0 = time.time()
    image = generate_image(
        model=model,
        processor=processor,
        prompt=prompt,
        ref_image_paths=args.ref_images,
        height=selected_h,
        width=selected_w,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        shift=args.shift,
        timesteps_list=timesteps_list,
        scheduler_name=args.scheduler_name,
        seed=args.seed,
        noise_scale_start=args.noise_scale_start,
        noise_scale_end=args.noise_scale_end,
        noise_clip_std=args.noise_clip_std,
        keep_original_aspect=args.keep_original_aspect,
    )

    image.save(output_path)
    dt = time.time() - t0
    try:
        print(f"[HiDream CLI] Saved: {output_path}")
        print(f"[HiDream CLI] Final image size: {image.size[0]}x{image.size[1]}")
        print(f"[HiDream CLI] Generation time: {dt:.1f}s")
    except Exception:
        pass


if __name__ == "__main__":
    main()
