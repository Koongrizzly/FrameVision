from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path


def _force_utf8_console() -> None:
    """Force UTF-8 before importing the community backend.

    The SeedVR2 community code prints Unicode/emoji during module import.
    Windows console pipes otherwise default to cp1252 and can crash before
    inference even starts.
    """
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _register_selected_models(repo: Path, dit: Path, vae: Path) -> None:
    """Allow user-selected safetensors/GGUF names without editing the community repo."""
    sys.path.insert(0, str(repo))
    from src.utils import model_registry as mr

    # The CLI imports these names after this function runs, so changing the module
    # here is enough to make arbitrary on-disk safetensors selectable.
    original_get = mr.get_available_dit_models
    selected_name = dit.name

    def get_available_with_selected():
        names = list(original_get())
        if selected_name not in names:
            names.append(selected_name)
        return names

    mr.get_available_dit_models = get_available_with_selected
    mr.DEFAULT_VAE = vae.name

    # Add metadata for custom community-converted files when possible.  The loader
    # can still inspect the file itself; this mainly keeps registry lookups sane.
    if selected_name not in mr.MODEL_REGISTRY:
        precision = "bf16" if "bf16" in selected_name.lower() else ("fp16" if "fp16" in selected_name.lower() else "custom")
        size = "7B" if "7b" in selected_name.lower() else "3B"
        try:
            mr.MODEL_REGISTRY[selected_name] = mr.ModelInfo(size=size, precision=precision)
        except Exception:
            pass
    if vae.name not in mr.MODEL_REGISTRY:
        try:
            mr.MODEL_REGISTRY[vae.name] = mr.ModelInfo(category="vae", precision="fp16")
        except Exception:
            pass


def _find_ffmpeg() -> str | None:
    root = Path(__file__).resolve().parent.parent
    bundled = root / "presets" / "bin" / "ffmpeg.exe"
    if bundled.exists():
        return str(bundled)
    return shutil.which("ffmpeg")


def _finalize_browser_safe_mp4(source: Path, video_only: Path, final: Path) -> None:
    """Create the actual deliverable as H.264/AAC MP4.

    The community CLI currently writes MPEG-4 Part 2 on some Windows setups.
    Desktop players usually accept that, but Chromium/browser video and canvas
    decoding is inconsistent.  FrameVision therefore treats the CLI file as an
    intermediate and always creates a standards-friendly final MP4:

      H.264/AVC (libx264), yuv420p, AAC, fast-start MP4.

    This is deliberately part of the SeedVR2 output pipeline, not an optional
    compare-tool workaround.  The SeedVR2-generated frames are preserved at the
    same resolution and timing; only the delivery codec/container encoding changes.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is required to finalize SeedVR2 output as browser-safe H.264 MP4. "
            "Expected FrameVision/presets/bin/ffmpeg.exe or ffmpeg on PATH."
        )
    if not source.is_file():
        raise FileNotFoundError(f"Source video not found while finalizing SeedVR2 output: {source}")
    if not video_only.exists():
        raise FileNotFoundError(f"SeedVR2 intermediate video not found: {video_only}")

    tmp = final.with_name(final.stem + "_h264_mux" + final.suffix)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass

    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y",
        "-i", str(video_only), "-i", str(source),
        "-map", "0:v:0", "-map", "1:a?",
        # Widely supported high-quality MP4 delivery format.
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "16",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-tag:v", "avc1",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(tmp),
    ]
    print("Final H.264/AAC encode: " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0 or not tmp.exists() or tmp.stat().st_size <= 0:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(
            "SeedVR2 finished, but FFmpeg could not create the final H.264/AAC MP4. "
            "The incompatible intermediate was left in place for diagnosis: " + str(video_only)
        )

    if final.exists():
        final.unlink()
    tmp.replace(final)
    try:
        video_only.unlink()
    except Exception:
        pass


def main() -> int:
    # Must happen before importing anything from the community SeedVR2 repo.
    _force_utf8_console()
    ap = argparse.ArgumentParser(description="FrameVision clean wrapper for the community SeedVR2 standalone CLI")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--resolution", type=int, default=1080)
    ap.add_argument("--max-resolution", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=33)
    ap.add_argument("--chunk-size", type=int, default=330)
    ap.add_argument("--uniform-batch-size", action="store_true")
    ap.add_argument("--temporal-overlap", type=int, default=3)
    ap.add_argument("--prepend-frames", type=int, default=0)
    ap.add_argument("--blocks-to-swap", type=int, default=32)
    ap.add_argument("--attention-mode", default="sageattn_2")
    ap.add_argument("--color-correction", default="lab")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vae-encode-tiled", action="store_true")
    ap.add_argument("--vae-encode-tile-size", type=int, default=1024)
    ap.add_argument("--vae-encode-overlap", type=int, default=128)
    ap.add_argument("--vae-decode-tiled", action="store_true")
    ap.add_argument("--vae-decode-tile-size", type=int, default=768)
    ap.add_argument("--vae-decode-overlap", type=int, default=128)
    ap.add_argument("--cpu-offload", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    cli = repo / "inference_cli.py"
    src = Path(args.input).resolve()
    final = Path(args.output).resolve()
    dit = Path(args.dit).resolve()
    vae = Path(args.vae).resolve()
    for p, label in ((cli, "community inference_cli.py"), (src, "input"), (dit, "DiT"), (vae, "VAE")):
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")
    if dit.parent != vae.parent:
        raise RuntimeError("Community DiT and VAE must be in the same model directory.")

    # Community CLI writes the video stream. Use an intermediate file so we can
    # restore the original audio afterward without re-encoding SeedVR2's video.
    final.parent.mkdir(parents=True, exist_ok=True)
    temp_out = final.with_name(final.stem + "_videoonly" + final.suffix)
    try:
        temp_out.unlink(missing_ok=True)
    except Exception:
        pass

    _register_selected_models(repo, dit, vae)

    cli_args = [
        str(cli), str(src), "--output", str(temp_out),
        "--model_dir", str(dit.parent), "--dit_model", dit.name,
        "--resolution", str(args.resolution), "--max_resolution", str(args.max_resolution),
        "--batch_size", str(args.batch_size), "--chunk_size", str(args.chunk_size),
        "--temporal_overlap", str(args.temporal_overlap), "--prepend_frames", str(args.prepend_frames),
        "--blocks_to_swap", str(args.blocks_to_swap), "--attention_mode", args.attention_mode,
        "--color_correction", args.color_correction, "--seed", str(args.seed),
        "--vae_encode_tile_size", str(args.vae_encode_tile_size), "--vae_encode_tile_overlap", str(args.vae_encode_overlap),
        "--vae_decode_tile_size", str(args.vae_decode_tile_size), "--vae_decode_tile_overlap", str(args.vae_decode_overlap),
        "--cuda_device", "0", "--tensor_offload_device", "cpu",
    ]
    if args.uniform_batch_size:
        cli_args.append("--uniform_batch_size")
    if args.vae_encode_tiled:
        cli_args.append("--vae_encode_tiled")
    if args.vae_decode_tiled:
        cli_args.append("--vae_decode_tiled")
    if args.cpu_offload:
        cli_args += ["--dit_offload_device", "cpu", "--vae_offload_device", "cpu"]

    print("Community SeedVR2 backend", flush=True)
    print(f"DiT: {dit.name}", flush=True)
    print(f"VAE: {vae.name}", flush=True)
    print(f"Model directory: {dit.parent}", flush=True)
    print(f"Batch={args.batch_size} chunk={args.chunk_size} uniform={args.uniform_batch_size} overlap={args.temporal_overlap}", flush=True)
    print(f"BlockSwap={args.blocks_to_swap} attention={args.attention_mode} CPU offload={args.cpu_offload}", flush=True)
    print("RUN CLI: " + " ".join(cli_args), flush=True)

    old_argv = sys.argv[:]
    old_cwd = os.getcwd()
    try:
        os.chdir(str(repo))
        sys.argv = cli_args
        try:
            runpy.run_path(str(cli), run_name="__main__")
        except SystemExit as exc:
            code = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
            if code != 0:
                return code
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    if not temp_out.exists():
        raise RuntimeError(f"Community SeedVR2 finished without creating output: {temp_out}")
    _finalize_browser_safe_mp4(src, temp_out, final)
    print(f"DONE: {final}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
