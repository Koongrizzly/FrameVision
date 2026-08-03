"""FrameVision native LTX 2.3 MSR + supplied-audio hybrid pipeline.

This keeps the official IC-LoRA visual-conditioning implementation and adds the
same frozen input-audio latent flow used by the official a2vid_two_stage
pipeline. It intentionally returns the original input waveform so the model
cannot replace the supplied song with generated vocals.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.args import VideoConditioningAction, default_2_stage_distilled_arg_parser
from ltx_pipelines.utils.blocks import AudioConditioner
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution
from ltx_pipelines.utils.media_io import decode_audio_from_file, encode_video
from ltx_pipelines.utils.types import ModalitySpec


class MSRAudioICLoraPipeline(ICLoraPipeline):
    """Official IC-LoRA visual conditioning with a frozen supplied audio latent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        checkpoint_path = kwargs.get("distilled_checkpoint_path")
        if checkpoint_path is None and args:
            checkpoint_path = args[0]
        registry = kwargs.get("registry")
        alloc_trim_strategy = kwargs.get("alloc_trim_strategy")

        conditioner_kwargs = {"registry": registry}
        if alloc_trim_strategy is not None:
            conditioner_kwargs["alloc_trim_strategy"] = alloc_trim_strategy
        try:
            self.audio_conditioner = AudioConditioner(
                checkpoint_path,
                self.dtype,
                self.device,
                **conditioner_kwargs,
            )
        except TypeError:
            # Compatibility with earlier LTX-2 package revisions.
            conditioner_kwargs.pop("alloc_trim_strategy", None)
            self.audio_conditioner = AudioConditioner(
                checkpoint_path,
                self.dtype,
                self.device,
                **conditioner_kwargs,
            )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images,
        video_conditioning,
        audio_path: str,
        audio_start_time: float = 0.0,
        audio_max_duration: float | None = None,
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        stage_1_sigmas: torch.Tensor = DISTILLED_SIGMAS,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
        streaming_prefetch_count: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                "conditioning_attention_strength must be in [0.0, 1.0], "
                f"got {conditioning_attention_strength}"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        prompt_kwargs = {
            "enhance_first_prompt": enhance_prompt,
            "enhance_prompt_image": images[0][0] if len(images) > 0 else None,
        }
        # Newer PromptEncoder revisions accept this seed; older ones do not.
        prompt_kwargs["enhance_prompt_seed"] = seed
        try:
            (ctx_p,) = self.prompt_encoder([prompt], **prompt_kwargs)
        except TypeError:
            prompt_kwargs.pop("enhance_prompt_seed", None)
            (ctx_p,) = self.prompt_encoder([prompt], **prompt_kwargs)
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        logging.info("[MSR-Audio] Decoding supplied audio condition: %s", audio_path)
        decoded_audio = decode_audio_from_file(
            audio_path,
            self.device,
            audio_start_time,
            audio_max_duration,
        )
        if decoded_audio is None:
            raise ValueError(f"Failed to decode supplied audio from {audio_path}")

        encoded_audio_latent = self.audio_conditioner(
            lambda enc: vae_encode_audio(decoded_audio, enc, None)
        )
        audio_shape = AudioLatentShape.from_duration(
            batch=1,
            duration=num_frames / frame_rate,
            channels=8,
            mel_bins=16,
        )
        encoded_audio_latent = encoded_audio_latent[:, :, : audio_shape.frames]
        logging.info(
            "[MSR-Audio] Supplied audio encoded and frozen for both stages; latent_frames=%d",
            int(encoded_audio_latent.shape[2]),
        )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
            )
        )

        stage_1_sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
        stage_call_kwargs = {}
        if streaming_prefetch_count is not None:
            stage_call_kwargs["streaming_prefetch_count"] = streaming_prefetch_count
        try:
            video_state, _ = self.stage_1(
                denoiser=SimpleDenoiser(video_context, audio_context),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
                audio=ModalitySpec(
                    context=audio_context,
                    frozen=True,
                    noise_scale=0.0,
                    initial_latent=encoded_audio_latent,
                ),
                **stage_call_kwargs,
            )
        except TypeError as exc:
            if "streaming_prefetch_count" not in str(exc):
                raise
            stage_call_kwargs.clear()
            video_state, _ = self.stage_1(
                denoiser=SimpleDenoiser(video_context, audio_context),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
                audio=ModalitySpec(
                    context=audio_context,
                    frozen=True,
                    noise_scale=0.0,
                    initial_latent=encoded_audio_latent,
                ),
            )

        if skip_stage_2:
            logging.info("[MSR-Audio] Skipping Stage 2")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            original_audio = Audio(
                waveform=decoded_audio.waveform.squeeze(0),
                sampling_rate=decoded_audio.sampling_rate,
            )
            return decoded_video, original_audio

        upscaled_video_latent = self.upsampler(video_state.latent[:1])
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )
        # IC-LoRA reference video is Stage-1 context. Stage 2 keeps normal image/end-frame conditions.
        from ltx_pipelines.utils.helpers import combined_image_conditionings

        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
            )
        )
        video_state, _ = self.stage_2(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=stage_2_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                frozen=True,
                noise_scale=0.0,
                initial_latent=encoded_audio_latent,
            ),
            **stage_call_kwargs,
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        original_audio = Audio(
            waveform=decoded_audio.waveform.squeeze(0),
            sampling_rate=decoded_audio.sampling_rate,
        )
        logging.info("[MSR-Audio] Returning original supplied waveform; generated audio is discarded")
        return decoded_video, original_audio


def _build_parser():
    try:
        from ltx_pipelines.utils.args import resolve_cli_params

        params = resolve_cli_params(distilled=True)
        parser = default_2_stage_distilled_arg_parser(params=params)
    except (ImportError, TypeError):
        parser = default_2_stage_distilled_arg_parser()

    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
    )
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--audio-start-time", type=float, default=0.0)
    parser.add_argument("--audio-max-duration", type=float, default=None)
    parser.add_argument("--skip-stage-2", action="store_true")
    return parser


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = _build_parser()
    args = parser.parse_args()

    pipeline_kwargs = dict(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
    )
    if hasattr(args, "compile"):
        pipeline_kwargs["compilation_config"] = args.compile
    if hasattr(args, "offload_mode"):
        pipeline_kwargs["offload_mode"] = args.offload_mode

    try:
        pipeline = MSRAudioICLoraPipeline(**pipeline_kwargs)
    except TypeError:
        # Compatibility with older installed IC-LoRA signatures.
        pipeline_kwargs.pop("compilation_config", None)
        pipeline_kwargs.pop("offload_mode", None)
        pipeline = MSRAudioICLoraPipeline(**pipeline_kwargs)

    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        video_conditioning=args.video_conditioning,
        audio_path=args.audio_path,
        audio_start_time=args.audio_start_time,
        audio_max_duration=(
            args.audio_max_duration
            if args.audio_max_duration is not None
            else args.num_frames / args.frame_rate
        ),
        tiling_config=tiling_config,
        enhance_prompt=getattr(args, "enhance_prompt", False),
        skip_stage_2=args.skip_stage_2,
        streaming_prefetch_count=getattr(args, "streaming_prefetch_count", None),
    )
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
