"""FrameVision native LTX 2.5 + Licon MSR pipeline.

This is a separate LTX-2.5 MSR implementation.  It intentionally does not touch
or import the older LTX-2.3 pseudo-video/IC-LoRA helper path.

Licon LTX-2.5 MSR differences implemented here:
- loads the learned reference_slot_embedding tensors from the MSR safetensors;
- keeps the remaining checkpoint weights as a normal LTX LoRA on BOTH stages;
- encodes 1..5 still references independently in stable pic1..pic4/background order;
- adds the learned slot embedding to each reference latent;
- appends reference tokens at consecutive negative temporal positions;
- rebuilds/re-encodes the MSR references for Stage 2 after latent upscaling;
- supports supplied audio as a frozen latent and returns the original waveform.

The implementation uses native ltx-core ConditioningItem primitives rather than
ComfyUI and therefore requires no ComfyUI installation.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.conditioning.mask_utils import extend_keyframes_mask, update_attention_mask
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.video_vae import AUTO_TILING, TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser, resolve_cli_params
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    ensure_tiling_config,
    get_device,
    tiling_scale_factors_for_vae,
)
from ltx_pipelines.utils.blocks import AudioConditioner
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.types import ModalitySpec

try:
    from ltx_core.conditioning import ConditioningItemAttentionStrengthWrapper
except ImportError:  # older package layout
    from ltx_core.conditioning.types.attention_strength_wrapper import ConditioningItemAttentionStrengthWrapper

try:
    from ltx_pipelines.distilled import DistilledPipeline
except ImportError as exc:  # explicit error is much easier to diagnose in FrameVision log
    raise RuntimeError(
        "LTX 2.5 MSR requires the native ltx_pipelines.distilled pipeline. "
        "Install/update the normal FrameVision LTX 2.5 runtime first."
    ) from exc


LOG = logging.getLogger(__name__)
_SLOT_PREFIXES = (
    "diffusion_model.reference_slot_embedding.",
    "reference_slot_embedding.",
)
_REQUIRED_SLOT_KEYS = {
    "frequencies",
    "net.0.weight",
    "net.0.bias",
    "net.2.weight",
    "net.2.bias",
}


class VideoConditionByMSRReferenceLatent(ConditioningItem):
    """Append a learned MSR reference latent at a negative pixel-frame offset.

    Native LTX reference conditioning appends clean tokens.  Licon's 2.5 MSR
    differs mainly in where those tokens live in temporal position space: each
    independently encoded reference receives a consecutive negative frame
    position.  Slot identity itself is already baked into ``latent`` by adding
    the learned Fourier/MLP embedding before this conditioning item is created.
    """

    def __init__(
        self,
        latent: torch.Tensor,
        *,
        frame_offset: int,
        strength: float = 1.0,
        downscale_factor: int = 1,
    ) -> None:
        self.latent = latent
        self.frame_offset = int(frame_offset)
        self.strength = float(strength)
        self.downscale_factor = int(downscale_factor)

    def apply_to(self, latent_state: LatentState, latent_tools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.latent.shape),
            device=self.latent.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            # Licon's append_keyframe path uses causal_fix=True.
            causal_fix=True,
        ).to(dtype=torch.float32)

        # Licon 2.5 MSR uses pic1-based consecutive negative time positions.
        positions[:, 0, ...] += self.frame_offset
        positions[:, 0, ...] /= latent_tools.fps

        if self.downscale_factor != 1:
            positions[:, 1, ...] *= self.downscale_factor
            positions[:, 2, ...] *= self.downscale_factor

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=self.latent.dtype,
        )
        attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=self.latent.device,
            dtype=self.latent.dtype,
        )
        return LatentState(
            latent=torch.cat([latent_state.latent, torch.zeros_like(tokens)], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=attention_mask,
            keyframes_mask=extend_keyframes_mask(latent_state, tokens.shape[1], marked=False),
            generated_keyframe_layout=latent_state.generated_keyframe_layout,
            generated_keyframes=latent_state.generated_keyframes,
            frozen=latent_state.frozen,
        )


def _metadata_int(metadata: dict[str, str], key: str, default: int) -> int:
    try:
        return max(1, round(float(metadata.get(key, default))))
    except (TypeError, ValueError):
        return int(default)


def load_msr_slot_state(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str], int]:
    """Load only Licon's five small slot tensors plus checkpoint metadata.

    ``safe_open`` is lazy, so this does not make another 1.3 GB in-memory copy of
    the complete LoRA. The normal LTX LoRA loader still owns the actual adapter
    weights used by the transformer.
    """

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"LTX 2.5 MSR checkpoint not found: {checkpoint}")

    state: dict[str, torch.Tensor] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            for prefix in _SLOT_PREFIXES:
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = handle.get_tensor(key).detach().cpu()
                    break

    missing = sorted(_REQUIRED_SLOT_KEYS.difference(state))
    if missing:
        raise ValueError(
            "This checkpoint is not a compatible Licon LTX 2.5 MSR checkpoint; "
            "missing reference slot tensors: " + ", ".join(missing)
        )

    if metadata.get("reference_token_order", "prepend") != "prepend":
        raise ValueError("Unsupported MSR metadata: reference_token_order must be 'prepend'.")
    if metadata.get("reference_slot_time_offsets", "pic1_based_negative_time") != "pic1_based_negative_time":
        raise ValueError(
            "Unsupported MSR metadata: reference_slot_time_offsets must be 'pic1_based_negative_time'."
        )

    downscale = _metadata_int(metadata, "reference_downscale_factor", 1)
    return state, metadata, downscale


def _slot_embedding(slot_id: int, state: dict[str, torch.Tensor], device, dtype) -> torch.Tensor:
    frequencies = state["frequencies"].to(device=device, dtype=torch.float32)
    slot_value = torch.tensor(float(slot_id), device=device, dtype=torch.float32)
    scaled = slot_value / 16.0
    phases = scaled * frequencies
    features = torch.cat((scaled.reshape(1), torch.sin(phases), torch.cos(phases)))
    weight0 = state["net.0.weight"].to(device=device, dtype=torch.float32)
    bias0 = state["net.0.bias"].to(device=device, dtype=torch.float32)
    hidden = F.silu(F.linear(features, weight0, bias0))
    weight2 = state["net.2.weight"].to(device=device, dtype=torch.float32)
    bias2 = state["net.2.bias"].to(device=device, dtype=torch.float32)
    return F.linear(hidden, weight2, bias2).to(dtype=dtype)


def _aspect_family(width: int, height: int) -> str:
    ratio = width / height
    if ratio >= 1.25:
        return "landscape"
    if ratio <= 0.8:
        return "portrait"
    return "square"


def _pil_to_bchw(image: Image.Image, *, device, dtype) -> torch.Tensor:
    # Avoid torchvision dependency. Pillow and torch are already present in LTX.
    byte_storage = torch.ByteStorage.from_buffer(image.tobytes())
    tensor = torch.ByteTensor(byte_storage).view(image.height, image.width, 3)
    tensor = tensor.to(device=device, dtype=torch.float32) / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(dtype=dtype)


def _resize_reference_tensor(
    image: torch.Tensor,
    *,
    target_width: int,
    target_height: int,
    is_background: bool,
) -> torch.Tensor:
    """Match Licon's current reference resize policy.

    Backgrounds are center-cropped to fill. Subjects preserve the complete image
    with white padding when aspect families differ or the source is smaller;
    otherwise they use center-crop-to-fill.
    """

    _, _, source_height, source_width = image.shape
    if source_width == target_width and source_height == target_height:
        return image

    if is_background:
        scale = max(target_width / source_width, target_height / source_height)
        new_w = max(target_width, round(source_width * scale))
        new_h = max(target_height, round(source_height * scale))
        resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        left = (new_w - target_width) // 2
        top = (new_h - target_height) // 2
        return resized[:, :, top : top + target_height, left : left + target_width]

    same_family = _aspect_family(source_width, source_height) == _aspect_family(target_width, target_height)
    source_is_smaller = source_width <= target_width and source_height <= target_height
    if same_family and not source_is_smaller:
        scale = max(target_width / source_width, target_height / source_height)
        new_w = max(target_width, round(source_width * scale))
        new_h = max(target_height, round(source_height * scale))
        resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
        left = (new_w - target_width) // 2
        top = (new_h - target_height) // 2
        return resized[:, :, top : top + target_height, left : left + target_width]

    scale = min(target_width / source_width, target_height / source_height)
    new_w = max(1, min(target_width, round(source_width * scale)))
    new_h = max(1, min(target_height, round(source_height * scale)))
    resized = F.interpolate(image, size=(new_h, new_w), mode="bilinear", align_corners=False)
    canvas = torch.ones(
        (image.shape[0], image.shape[1], target_height, target_width),
        dtype=image.dtype,
        device=image.device,
    )
    left = (target_width - new_w) // 2
    top = (target_height - new_h) // 2
    canvas[:, :, top : top + new_h, left : left + new_w] = resized
    return canvas


def _reference_pixels(
    path: str | Path,
    *,
    width: int,
    height: int,
    reference_frames: int,
    is_background: bool,
    device,
    dtype,
) -> torch.Tensor:
    with Image.open(Path(path).expanduser()) as pil:
        pil = pil.convert("RGB")
        image = _pil_to_bchw(pil, device=device, dtype=dtype)
    image = _resize_reference_tensor(
        image,
        target_width=width,
        target_height=height,
        is_background=is_background,
    )
    # Native LTX video VAE consumes [-1, 1], [B,C,F,H,W].
    image = image.mul(2.0).sub(1.0)
    return image.unsqueeze(2).repeat(1, 1, reference_frames, 1, 1)


def _make_reference_tiling(video_encoder: VideoEncoder, width: int, height: int, reference_frames: int, tile_size: int, tile_overlap: int):
    """Best-effort native LTX equivalent of Comfy encode_tiled(tile=256, overlap=64).

    LTX's tiling API changed during the 2.5 cycle, so this helper deliberately
    falls back to the package default if the current revision cannot construct
    the explicit spatial configuration.
    """

    try:
        from ltx_core.model.video_vae import TileSizeConfig
        from ltx_core.tiling import DimensionSizeConfig

        long_side = DimensionSizeConfig(tile_size=int(tile_size), overlap=int(tile_overlap))

        # LTX 2.5's video VAE requires the temporal TILE SIZE itself to be
        # divisible by the temporal compression factor (8). MSR reference
        # sequence lengths remain 25 or 33 frames; only the internal temporal
        # tile window is rounded down to the largest valid multiple of 8.
        temporal_tile = max(8, (int(reference_frames) // 8) * 8)
        temporal_overlap = min(16, max(0, temporal_tile - 8))
        frames = DimensionSizeConfig(
            tile_size=temporal_tile,
            overlap=temporal_overlap,
        )
        return TileSizeConfig.from_long_side(
            long_side=long_side,
            height=height,
            width=width,
            scale_factors=video_encoder.video_scale_factors,
            frames=frames,
        )
    except Exception as exc:
        LOG.warning(
            "[LTX25-MSR] Could not build explicit %d/%d reference tiling (%s); using native default tiling.",
            tile_size,
            tile_overlap,
            exc,
        )
        try:
            from ltx_core.model.video_vae import TileSizeConfig
            return TileSizeConfig.default()
        except Exception:
            return None


def build_msr_conditionings(
    *,
    video_encoder: VideoEncoder,
    references: Sequence[tuple[str, str, bool]],
    slot_state: dict[str, torch.Tensor],
    downscale_factor: int,
    target_width: int,
    target_height: int,
    reference_frames: int,
    strength: float,
    device,
    dtype,
    use_tiled_encode: bool,
    tile_size: int,
    tile_overlap: int,
) -> list[ConditioningItem]:
    if reference_frames not in (25, 33):
        raise ValueError(f"LTX 2.5 MSR reference_frames must be 25 or 33, got {reference_frames}.")
    if not (1 <= len(references) <= 5):
        raise ValueError(f"LTX 2.5 MSR requires 1-5 references, got {len(references)}.")
    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"LTX 2.5 MSR strength must be in [0,1], got {strength}.")
    if target_width % downscale_factor or target_height % downscale_factor:
        raise ValueError(
            f"Target {target_width}x{target_height} is not divisible by MSR reference_downscale_factor={downscale_factor}."
        )

    ref_width = target_width // downscale_factor
    ref_height = target_height // downscale_factor
    conditionings: list[ConditioningItem] = []
    num_slots = len(references)

    for slot_index, (label, image_path, is_background) in enumerate(references):
        slot_id = slot_index + 1
        pixels = _reference_pixels(
            image_path,
            width=ref_width,
            height=ref_height,
            reference_frames=reference_frames,
            is_background=is_background,
            device=device,
            dtype=dtype,
        )
        if use_tiled_encode:
            tiling = _make_reference_tiling(
                video_encoder, ref_width, ref_height, reference_frames, tile_size, tile_overlap
            )
            if tiling is not None and hasattr(video_encoder, "tiled_encode"):
                guide_latent = video_encoder.tiled_encode(pixels, tiling)
            else:
                guide_latent = video_encoder(pixels)
        else:
            guide_latent = video_encoder(pixels)

        embedding = _slot_embedding(slot_id, slot_state, guide_latent.device, guide_latent.dtype)
        if embedding.numel() != guide_latent.shape[1]:
            raise ValueError(
                f"MSR slot embedding has {embedding.numel()} channels but LTX reference latent has "
                f"{guide_latent.shape[1]} channels. Check that the MSR checkpoint matches LTX 2.5."
            )
        guide_latent = guide_latent + embedding.view(1, -1, 1, 1, 1)

        # Same ordering as Licon's current node: with N refs the first is -N,
        # then -N+1 ... and the final connected reference is -1.
        frame_offset = -(num_slots - slot_index)
        cond: ConditioningItem = VideoConditionByMSRReferenceLatent(
            guide_latent,
            frame_offset=frame_offset,
            strength=strength,
            downscale_factor=downscale_factor,
        )
        if strength < 1.0:
            cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=float(strength))
        conditionings.append(cond)
        LOG.info(
            "[LTX25-MSR] %s slot=%d offset=%d ref_frames=%d latent=%s embedding_norm=%.6f",
            label,
            slot_id,
            frame_offset,
            reference_frames,
            tuple(guide_latent.shape),
            embedding.detach().float().norm().item(),
        )
    return conditionings


def collect_reference_paths(
    ref1: str,
    ref2: str = "",
    ref3: str = "",
    ref4: str = "",
    background: str = "",
) -> list[tuple[str, str, bool]]:
    ordered = [
        ("pic1", ref1, False),
        ("pic2", ref2, False),
        ("pic3", ref3, False),
        ("pic4", ref4, False),
        ("background", background, True),
    ]
    result: list[tuple[str, str, bool]] = []
    for label, value, is_background in ordered:
        value = str(value or "").strip()
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"MSR {label} image not found: {path}")
        result.append((label, str(path), is_background))
    if not result:
        raise ValueError("LTX 2.5 MSR requires at least pic1/reference 1.")
    return result


class LTX25MSRAudioPipeline(DistilledPipeline):
    """Native distilled LTX 2.5 two-stage pipeline with Licon MSR and frozen supplied audio.

    The installed FrameVision LTX 2.5 checkpoint is already distilled.  Do not use
    A2VidPipelineTwoStage here: that class is for the full/dev checkpoint plus a
    separate distilled LoRA, and therefore incorrectly requires --distilled-lora.
    """

    def __init__(
        self,
        *args,
        msr_lora_path: str,
        msr_strength_model: float = 1.0,
        **kwargs,
    ) -> None:
        self.msr_lora_path = str(Path(msr_lora_path).expanduser().resolve())
        self.slot_state, self.msr_metadata, self.reference_downscale_factor = load_msr_slot_state(
            self.msr_lora_path
        )
        self.msr_strength_model = float(msr_strength_model)

        # Inject the MSR adapter into BOTH diffusion stages. Licon's special slot
        # tensors are ignored by the normal LTX LoRA mapper and loaded separately above.
        loras = list(kwargs.pop("loras", ()) or ())
        loras.append(
            LoraPathStrengthAndSDOps(
                self.msr_lora_path,
                self.msr_strength_model,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        )
        kwargs["loras"] = tuple(loras)

        # DistilledPipeline is the same native base used by FrameVision's working
        # standalone LTX 2.5 helper.  Capture its model paths before construction
        # so we can add the supplied-audio encoder used by Music Clip Creator.
        model_paths = kwargs.get("model_paths")
        registry = kwargs.get("registry")
        alloc_trim_strategy = kwargs.get("alloc_trim_strategy")
        super().__init__(*args, **kwargs)
        if model_paths is None:
            raise ValueError("LTX 2.5 MSR requires split model_paths.")
        conditioner_kwargs = {"registry": registry}
        if alloc_trim_strategy is not None:
            conditioner_kwargs["alloc_trim_strategy"] = alloc_trim_strategy
        try:
            self.audio_conditioner = AudioConditioner(
                model_paths.audio_vae(), self.dtype, self.device, **conditioner_kwargs
            )
        except TypeError:
            # Compatibility with LTX package revisions that do not expose
            # alloc_trim_strategy on AudioConditioner.
            conditioner_kwargs.pop("alloc_trim_strategy", None)
            self.audio_conditioner = AudioConditioner(
                model_paths.audio_vae(), self.dtype, self.device, **conditioner_kwargs
            )
        LOG.info(
            "[LTX25-MSR] Loaded %s | model_strength=%g | downscale=%d | slot_dim=%d",
            self.msr_lora_path,
            self.msr_strength_model,
            self.reference_downscale_factor,
            int(self.slot_state["net.2.bias"].numel()),
        )

    def _stage_conditionings(
        self,
        *,
        images,
        references,
        width: int,
        height: int,
        reference_frames: int,
        reference_strength: float,
        use_tiled_encode: bool,
        tile_size: int,
        tile_overlap: int,
        color_space=None,
    ):
        def encode(video_encoder):
            normal = combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=video_encoder,
                dtype=self.dtype,
                device=self.device,
                color_space=color_space,
            )
            normal.extend(
                build_msr_conditionings(
                    video_encoder=video_encoder,
                    references=references,
                    slot_state=self.slot_state,
                    downscale_factor=self.reference_downscale_factor,
                    target_width=width,
                    target_height=height,
                    reference_frames=reference_frames,
                    strength=reference_strength,
                    device=self.device,
                    dtype=self.dtype,
                    use_tiled_encode=use_tiled_encode,
                    tile_size=tile_size,
                    tile_overlap=tile_overlap,
                )
            )
            return normal

        return self.image_conditioner(encode)

    def __call__(  # noqa: PLR0913
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images,
        audio_path: str,
        msr_references: Sequence[tuple[str, str, bool]],
        msr_reference_frames: int = 33,
        msr_reference_strength: float = 1.0,
        msr_use_tiled_encode: bool = True,
        msr_tile_size: int = 256,
        msr_tile_overlap: int = 64,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        vae_dtype: torch.dtype | None = None,
        tiling_config=AUTO_TILING,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        color_space=None,
        **_ignored,
    ):
        if max_batch_size != 1:
            raise ValueError("Licon LTX 2.5 MSR currently requires max_batch_size=1.")
        images = self.image_conditioner.resolve_crf(images)
        assert_resolution(height=height, width=width, is_two_stage=True)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        vae_dtype = vae_dtype or self.dtype

        # Distilled LTX 2.5 uses SimpleDenoiser and its fixed distilled sigma
        # schedules.  A negative CFG context / distilled LoRA is intentionally not
        # used here because the transformer checkpoint itself is distilled.
        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding

        scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
        tiling_config = ensure_tiling_config(
            tiling_config,
            scale_factors=scale_factors,
            vae_checkpoint_path=self.video_decoder.checkpoint_path,
            video_shape=VideoPixelShape(
                batch=1, frames=num_frames, height=height, width=width, fps=frame_rate
            ),
            diffvae_optimization=self.video_decoder.diffvae_optimization,
            device=self.device,
        )

        decoded_audio = decode_audio_from_file(
            audio_path,
            self.device,
            audio_start_time,
            audio_max_duration if audio_max_duration is not None else num_frames / frame_rate,
        )
        if decoded_audio is None:
            raise ValueError(f"Failed to decode supplied audio: {audio_path}")
        encoded_audio_latent = self.audio_conditioner(
            lambda enc: vae_encode_audio(decoded_audio, enc, None)
        )
        audio_shape = AudioLatentShape.from_duration(
            batch=1, duration=num_frames / frame_rate, channels=8, mel_bins=16
        )
        encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
        LOG.info(
            "[LTX25-MSR] Supplied audio encoded and frozen for both stages; latent_frames=%d",
            int(encoded_audio_latent.shape[2]),
        )

        stage_1_w, stage_1_h = width // 2, height // 2
        LOG.info("[LTX25-MSR] Encoding Stage 1 references at %dx%d", stage_1_w, stage_1_h)
        stage_1_conditionings = self._stage_conditionings(
            images=images, references=msr_references, width=stage_1_w, height=stage_1_h,
            reference_frames=msr_reference_frames, reference_strength=msr_reference_strength,
            use_tiled_encode=msr_use_tiled_encode, tile_size=msr_tile_size,
            tile_overlap=msr_tile_overlap, color_space=color_space,
        )
        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage1_extra = {}
        sampler_kwargs = getattr(self, "_stage_1_sampler_kwargs", None)
        if callable(sampler_kwargs):
            stage1_extra.update(sampler_kwargs(seed))
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(v_context_p, a_context_p),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_w, height=stage_1_h, frames=num_frames, fps=frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
            audio=ModalitySpec(
                context=a_context_p, frozen=True, noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            ),
            **stage1_extra,
        )

        # Rebuild reference conditions at the Stage-2 resolution, matching Licon's
        # LTX 2.5 workflow rather than carrying low-resolution ref latents forward.
        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        LOG.info("[LTX25-MSR] Re-encoding Stage 2 references at %dx%d", width, height)
        stage_2_conditionings = self._stage_conditionings(
            images=images, references=msr_references, width=width, height=height,
            reference_frames=msr_reference_frames, reference_strength=msr_reference_strength,
            use_tiled_encode=msr_use_tiled_encode, tile_size=msr_tile_size,
            tile_overlap=msr_tile_overlap, color_space=color_space,
        )
        video_state, _ = self.stage(
            denoiser=SimpleDenoiser(v_context_p, a_context_p),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width, height=height, frames=num_frames, fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p, conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(), initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p, frozen=True, noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            ),
        )

        # Stage 2 is finished. Keep only the final latent required by DiffVAE and
        # release MSR/reference/stage tensors before loading the decoder. Normal
        # FP16 already sits near the 24 GB ceiling; the extra MSR adapter/context
        # can otherwise push Windows into shared GPU memory at this exact handoff.
        final_video_latent = video_state.latent
        del video_state
        del stage_1_conditionings
        del stage_2_conditionings
        del upscaled_video_latent
        del stage_1_sigmas
        del stage_2_sigmas
        del encoded_audio_latent
        del v_context_p
        del a_context_p
        del noiser
        del images

        import gc
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            try:
                free_before, total_mem = torch.cuda.mem_get_info()
            except Exception:
                free_before = total_mem = 0
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            try:
                free_after, _ = torch.cuda.mem_get_info()
                LOG.info(
                    "[LTX25-MSR] Pre-decode cleanup complete; GPU free %.2f -> %.2f GiB / %.2f GiB",
                    free_before / (1024 ** 3),
                    free_after / (1024 ** 3),
                    total_mem / (1024 ** 3),
                )
            except Exception:
                LOG.info("[LTX25-MSR] Pre-decode cleanup complete")

        decoded_video = self.video_decoder(
            final_video_latent, tiling_config, generator, dtype=vae_dtype
        )
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )
        return decoded_video, original_audio, tiling_config


def _add_msr_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Compatibility/routing marker used by FrameVision's Music Clip Creator.
    # This pipeline is MSR-only, so the flag does not change inference behavior;
    # it lets the caller verify that an MSR job was not accidentally routed to
    # ordinary I2V before launching the expensive backend.
    parser.add_argument("--msr-enabled", action="store_true", default=False)
    parser.add_argument("--msr-lora-path", required=True)
    parser.add_argument("--msr-strength-model", type=float, default=1.0)
    parser.add_argument("--msr-ref-1", required=True)
    parser.add_argument("--msr-ref-2", default="")
    parser.add_argument("--msr-ref-3", default="")
    parser.add_argument("--msr-ref-4", default="")
    parser.add_argument("--msr-background", default="")
    parser.add_argument("--msr-reference-strength", type=float, default=1.0)
    parser.add_argument("--msr-reference-frames", type=int, choices=(25, 33), default=33)
    parser.add_argument("--msr-use-tiled-encode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--msr-tile-size", type=int, default=256)
    parser.add_argument("--msr-tile-overlap", type=int, default=64)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--audio-start-time", type=float, default=0.0)
    parser.add_argument("--audio-max-duration", type=float, default=None)
    return parser


def _build_parser() -> argparse.ArgumentParser:
    # The FrameVision LTX 2.5 transformer is already distilled.  Using the full
    # two-stage parser would incorrectly require --distilled-lora.
    try:
        parser = default_2_stage_distilled_arg_parser(params=resolve_cli_params(distilled=True))
    except TypeError:
        try:
            parser = default_2_stage_distilled_arg_parser(params=resolve_cli_params())
        except TypeError:
            parser = default_2_stage_distilled_arg_parser()
    return _add_msr_args(parser)


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args()

    refs = collect_reference_paths(
        args.msr_ref_1,
        args.msr_ref_2,
        args.msr_ref_3,
        args.msr_ref_4,
        args.msr_background,
    )
    # Music Clip Creator MSR jobs are short-lived processes, so retaining CUDA
    # allocator blocks between components only increases peak VRAM. Match the
    # proven normal FP16 helper's non-spilling configuration: CPU offload + TRIM.
    pipeline = LTX25MSRAudioPipeline(
        model_paths=args.model_paths,
        spatial_upsampler_path=args.spatial_upsampler_path,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root=getattr(args, "prompt_enhancer_gemma_root", None),
        diffvae_optimization=args.diffvae_optimization,
        msr_lora_path=args.msr_lora_path,
        msr_strength_model=args.msr_strength_model,
    )
    LOG.info("[LTX25-MSR] CUDA allocator strategy: TRIM (hardcoded for Music Clip MSR)")

    from ltx_pipelines.utils.media_io import resolve_hdr_color_space, vae_dtype_for_hdr

    hdr = resolve_hdr_color_space(images=args.images, hdr=args.hdr)
    vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
    video, audio, tiling_config = pipeline(
        prompt=args.prompt,
        negative_prompt=getattr(args, "negative_prompt", ""),
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        audio_path=args.audio_path,
        audio_start_time=args.audio_start_time,
        audio_max_duration=(
            args.audio_max_duration
            if args.audio_max_duration is not None
            else args.num_frames / args.frame_rate
        ),
        msr_references=refs,
        msr_reference_frames=args.msr_reference_frames,
        msr_reference_strength=args.msr_reference_strength,
        msr_use_tiled_encode=args.msr_use_tiled_encode,
        msr_tile_size=args.msr_tile_size,
        msr_tile_overlap=args.msr_tile_overlap,
        vae_dtype=vae_dtype,
        color_space=hdr,
        tiling_config=AUTO_TILING,
        enhance_prompt=args.enhance_prompt,
        enhance_static_cache=getattr(args, "enhance_static_cache", False),
        max_batch_size=1,
    )
    chunks = get_video_chunks_number(args.num_frames, tiling_config)
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=chunks,
        color_space=hdr,
    )


if __name__ == "__main__":
    main()
