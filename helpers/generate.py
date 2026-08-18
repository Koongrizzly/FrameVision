from __future__ import annotations
import argparse
import sys
from pathlib import Path

# This launcher lives in helpers/, while runtime/ remains in the standalone root.
_APP_ROOT = Path(__file__).resolve().parents[1]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from runtime.paths import ROOT
from runtime.validate_models import validate
from runtime import vram_manager as _vram_manager_module

_EXPECTED_VRAM_SIGNATURE = "V11_4_REF_VAE_ADMISSION_20260817A"

def _verify_vram_runtime():
    actual = getattr(_vram_manager_module, "VRAM_MANAGER_SIGNATURE", None)
    if actual != _EXPECTED_VRAM_SIGNATURE:
        path = getattr(_vram_manager_module, "__file__", "unknown")
        raise RuntimeError(
            "VRAM Manager patch mismatch: the launcher is V11.4 but the loaded "
            f"runtime/vram_manager.py is not. Loaded: {path} | signature={actual!r}. "
            "Re-extract the patch into the MiniMax app root so both helpers/ and runtime/ are replaced."
        )
    return str(getattr(_vram_manager_module, "__file__", "unknown"))


def main():
    import os, sys, tempfile, subprocess, shutil, time
    from runtime.ffmpeg_tools import ensure_ffmpeg_tools, tool_path

    ap = argparse.ArgumentParser(description="MiniMax-H3 W4A8 ConvRot standalone generator")
    ap.add_argument("--prompt", default="A cinematic red sports car races through rain-soaked neon streets at night, dynamic tracking camera, realistic reflections and natural engine sound.")
    ap.add_argument("--width", type=int, default=832); ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--frames", type=int, default=362); ap.add_argument("--experimental-long-duration", action="store_true", help="Allow H3 native-grid research durations beyond the normal 719-frame range, up to 2385 frames"); ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--cfg", type=float, default=1.0); ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--shift", type=float, default=12.0); ap.add_argument("--audio-shift", type=float, default=3.0)
    ap.add_argument("--sampler", default="euler"); ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--first-frame"); ap.add_argument("--last-frame"); ap.add_argument("--continue-video"); ap.add_argument("--continue-context-frames", type=int, default=39); ap.add_argument("--continue-audio-memory", action="store_true", help="Experimental: use source clip audio as continuation memory/context"); ap.add_argument("--glue-source"); ap.add_argument("--output")
    ap.add_argument("--fl2va-checkpoint"); ap.add_argument("--ref2va-checkpoint"); ap.add_argument("--text-encoder"); ap.add_argument("--video-vae"); ap.add_argument("--audio-vae")
    ap.add_argument("--lora", action="append", default=[]); ap.add_argument("--lora-strength", action="append", type=float, default=[])
    ap.add_argument("--extended-logging", action="store_true")
    ap.add_argument("--tile-debugging", action="store_true")
    ap.add_argument("--video-vae-tile-size", type=int, default=256)
    ap.add_argument("--video-vae-tile-overlap", type=int, default=128)
    ap.add_argument("--vram-manager", action="store_true")
    ap.add_argument("--vram-manager-auto", action="store_true", help="Automatically bypass VRAM Lab when native sampling is estimated to fit the detected GPU")
    ap.add_argument("--spectrum", action="store_true", help="Enable experimental bundled MiniMax H3 Spectrum feature forecasting")
    ap.add_argument("--sage-attention", action="store_true", help="Use SageAttention for the sampling worker")
    ap.add_argument("--disable-comfy-kitchen", action="store_true", help="Disable Comfy Kitchen quantized W4A8 / ConvRot acceleration for worker processes")
    ap.add_argument("--vram-residency-engine", choices=["static", "dynamic"], default="static")
    ap.add_argument("--vram-runtime-free-gb", type=float, default=0.5)
    ap.add_argument("--vram-text-headroom-gb", type=float, default=1.0)
    ap.add_argument("--vram-diffusion-headroom-gb", type=float, default=1.0)
    ap.add_argument("--vram-offload-chunk-mb", type=int, default=512)
    ap.add_argument("--vram-max-resident-weights-gb", type=float, default=0.0)
    ap.add_argument("--vram-block-check-interval", type=int, default=1)
    ap.add_argument("--vram-async-streams", type=int, default=2)
    ap.add_argument("--vram-video-vae-reserve-gb", type=float, default=2.0)
    ap.add_argument("--vram-audio-vae-reserve-gb", type=float, default=1.0)
    ap.add_argument("--vram-residency-fill", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vram-residency-target-free-gb", type=float, default=0.5)
    ap.add_argument("--vram-residency-warmup-blocks", type=int, default=2)
    ap.add_argument("--vram-residency-refill-interval", type=int, default=1)
    ap.add_argument("--vram-keep-text-encoder", action="store_true")
    ns = ap.parse_args()
    # Worker processes import the vendored Comfy stack afresh, so this environment
    # switch cleanly controls whether comfy.quant_ops may activate Comfy Kitchen.
    if ns.disable_comfy_kitchen:
        os.environ["H3_DISABLE_COMFY_KITCHEN"] = "1"
        print("Comfy Kitchen W4A8 acceleration: disabled", flush=True)
    else:
        os.environ.pop("H3_DISABLE_COMFY_KITCHEN", None)
        try:
            import comfy_kitchen as _ck
            _backends = _ck.list_backends()
            _cuda = _backends.get("cuda", {}) if isinstance(_backends, dict) else {}
            _state = "CUDA" if _cuda.get("available") and not _cuda.get("disabled") else "available fallback"
            print(f"Comfy Kitchen W4A8 acceleration: enabled ({_state})", flush=True)
        except Exception as _exc:
            print(f"Comfy Kitchen W4A8 acceleration: requested but unavailable ({type(_exc).__name__}: {_exc})", flush=True)
    if ns.vram_manager:
        runtime_path = _verify_vram_runtime()
        print(f"[VRAM-MGR] V11.4 runtime verified: {runtime_path}", flush=True)
    if ns.video_vae_tile_size < 128: print("ERROR: video VAE tile size must be at least 128 px"); return 2
    if ns.video_vae_tile_overlap < 0 or ns.video_vae_tile_overlap >= ns.video_vae_tile_size: print("ERROR: video VAE tile overlap must be >= 0 and smaller than tile size"); return 2
    if len(ns.lora) != len(ns.lora_strength): print("ERROR: each --lora needs one matching --lora-strength"); return 2
    if len(ns.lora) > 3: print("ERROR: maximum 3 LoRAs are supported"); return 2
    for lp in ns.lora:
        if not Path(lp).is_file(): print(f"ERROR: LoRA not found: {lp}"); return 2
    if ns.continue_video:
        if not Path(ns.continue_video).is_file(): print(f"ERROR: continue video not found: {ns.continue_video}"); return 2
        if ns.first_frame: print("ERROR: Continue Video already supplies the FL2VA first-frame boundary; clear First frame"); return 2
        if ns.continue_context_frames < 1: print("ERROR: continue context frames must be at least 1"); return 2
    if ns.glue_source:
        if not Path(ns.glue_source).is_file(): print(f"ERROR: glue source video not found: {ns.glue_source}"); return 2
        if not ns.continue_video: print("ERROR: Glue results requires a Continue Video source"); return 2

    errors, info, diff, ref, te, vv, av = validate(
        require_vae=True, mode="fl2va",
        fl2va_path=ns.fl2va_checkpoint, ref2va_path=ns.ref2va_checkpoint,
        text_encoder_path=ns.text_encoder, video_vae_path=ns.video_vae, audio_vae_path=ns.audio_vae,
    )
    if errors:
        print("Model validation failed before generation:")
        for e in errors: print(" -", e)
        return 2

    if ns.seed is None or int(ns.seed) < 0: ns.seed = int.from_bytes(os.urandom(4), "little") % 99_000_000
    else: ns.seed = int(ns.seed) % 99_000_000
    max_frames = 2385 if ns.experimental_long_duration else 719
    if int(ns.frames) > max_frames:
        mode_name = "experimental long-duration" if ns.experimental_long_duration else "normal"
        print(f"ERROR: maximum allowed requested frame count in {mode_name} mode is {max_frames} ({max_frames / 24.0:.3f} seconds at 24 FPS)"); return 2
    width = int(ns.width); height = int(ns.height)
    if width % 32 or height % 32:
        print("ERROR: width and height must be valid preset values divisible by 32"); return 2

    print(f"MiniMax-H3 standalone W4A8 | {width}x{height} | requested frames={ns.frames} | steps={ns.steps} | CFG={ns.cfg:g} | shift={ns.shift:g} | audio shift={ns.audio_shift:g} | seed={ns.seed}", flush=True)
    print(f"Using checkpoint: {Path(diff).name}", flush=True)
    print(f"Using text encoder: {Path(te).name}", flush=True)
    print(f"Using video VAE: {Path(vv).name}", flush=True)
    print(f"Using audio VAE: {Path(av).name}", flush=True)
    py = sys.executable

    use_vram_manager = bool(ns.vram_manager)
    stage_plan = None
    managed_sample_stages = ["reference", "text", "diffusion"] if use_vram_manager else []
    if ns.vram_manager_auto:
        from runtime.vram_auto import decide_vram_stages, print_vram_stage_decisions
        reference_needed = bool(ns.first_frame or ns.last_frame or ns.continue_video)
        reference_frames = ns.continue_context_frames if ns.continue_video else (2 if ns.first_frame and ns.last_frame else 1)
        stage_plan = decide_vram_stages(
            width, height, ns.frames, "fl2va",
            text_encoder_path=te, video_vae_path=vv, audio_vae_path=av,
            reference_needed=reference_needed, reference_frames=reference_frames,
            reference_audio=bool(ns.continue_video and ns.continue_audio_memory),
        )
        print_vram_stage_decisions(stage_plan, width, height, ns.frames)
        managed_sample_stages = [x for x in ("reference", "text", "diffusion") if stage_plan["stages"][x]["use_manager"]]
        use_vram_manager = bool(managed_sample_stages)

    out = Path(ns.output) if ns.output else ROOT / "output" / f"minimax_h3_int4_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)

    def _ffprobe_has_audio(ffprobe, path):
        try:
            r = subprocess.run([ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=False)
            return bool((r.stdout or "").strip())
        except Exception:
            return False

    def _ffprobe_duration(ffprobe, path):
        try:
            r = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True, check=False)
            return max(0.0, float((r.stdout or "0").strip() or 0))
        except Exception:
            return 0.0

    def _ffprobe_video_duration(ffprobe, path):
        """Return the video-stream duration, not the container/audio duration."""
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=duration", "-of", "default=nw=1:nk=1", str(path)],
                capture_output=True, text=True, check=False,
            )
            value=(r.stdout or "").strip()
            if value and value.upper() != "N/A":
                return max(0.0, float(value))
        except Exception:
            pass
        return _ffprobe_duration(ffprobe, path)

    def _concat_source_and_segment(ffmpeg, ffprobe, source, segment, destination, width, height, workdir):
        """Join source + continuation with one continuous, frame-locked audio encode.

        Independently encoded AAC streams carry priming/padding at their boundaries.
        Stream-copying those audio packets can make a tiny source tail or new-clip head
        audible twice even when H3 generated the correct waveform.  Video is still
        stream-copied when compatible, but audio is decoded, forced to each clip's exact
        video duration, concatenated as PCM, then encoded once for the final timeline.
        """
        source = Path(source); segment = Path(segment); destination = Path(destination)
        workdir = Path(workdir)
        concat_list = workdir / "glue_concat.txt"
        def esc(path):
            return str(Path(path).resolve()).replace("'", "'\\''")
        concat_list.write_text(f"file '{esc(source)}'\nfile '{esc(segment)}'\n", encoding="utf-8")

        src_d = _ffprobe_video_duration(ffprobe, source)
        seg_d = _ffprobe_video_duration(ffprobe, segment)
        if src_d <= 0 or seg_d <= 0:
            raise RuntimeError(f"Could not determine exact video durations for glue: source={src_d:.6f}s segment={seg_d:.6f}s")
        print(
            f"Glue results: exact AV join | source={source.name} {src_d:.6f}s | "
            f"result={segment.name} {seg_d:.6f}s | audio=32kHz continuous AAC encode",
            flush=True,
        )

        src_audio = _ffprobe_has_audio(ffprobe, source)
        seg_audio = _ffprobe_has_audio(ffprobe, segment)

        def audio_inputs_and_filter():
            cmd = [ffmpeg, "-y", "-i", str(source), "-i", str(segment)]
            extra_inputs = 0
            if not src_audio:
                cmd += ["-f", "lavfi", "-t", f"{src_d:.9f}", "-i", "anullsrc=r=32000:cl=stereo"]
                src_a = f"[{2 + extra_inputs}:a]"; extra_inputs += 1
            else:
                src_a = "[0:a]"
            if not seg_audio:
                cmd += ["-f", "lavfi", "-t", f"{seg_d:.9f}", "-i", "anullsrc=r=32000:cl=stereo"]
                seg_a = f"[{2 + extra_inputs}:a]"; extra_inputs += 1
            else:
                seg_a = "[1:a]"
            af = ";".join([
                f"{src_a}aresample=32000,apad,atrim=duration={src_d:.9f},asetpts=PTS-STARTPTS[a0]",
                f"{seg_a}aresample=32000,apad,atrim=duration={seg_d:.9f},asetpts=PTS-STARTPTS[a1]",
                "[a0][a1]concat=n=2:v=0:a=1[a]",
            ])
            return cmd, af

        # Fast path: preserve video packets exactly, but never stream-copy the two AAC
        # tracks across their internal encoder-delay boundary.
        video_tmp = workdir / "glued_video_only.mp4"
        video_copy_cmd = [
            ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-map", "0:v:0", "-an", "-c:v", "copy", "-avoid_negative_ts", "make_zero", str(video_tmp),
        ]
        copied = subprocess.run(video_copy_cmd, cwd=ROOT, check=False)
        if copied.returncode == 0 and video_tmp.is_file() and video_tmp.stat().st_size > 0:
            audio_tmp = workdir / "glued_audio.m4a"
            acmd, af = audio_inputs_and_filter()
            acmd += ["-filter_complex", af, "-map", "[a]", "-c:a", "aac", "-b:a", "256k", str(audio_tmp)]
            subprocess.check_call(acmd, cwd=ROOT)
            mux_tmp = workdir / "glued_exact_av.mp4"
            subprocess.check_call([
                ffmpeg, "-y", "-i", str(video_tmp), "-i", str(audio_tmp),
                "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-shortest", str(mux_tmp),
            ], cwd=ROOT)
            if destination.exists(): destination.unlink()
            shutil.move(str(mux_tmp), str(destination))
            print("Glue results complete: video stream copied; audio rebuilt as one exact-duration stream (no AAC join padding).", flush=True)
            return

        video_tmp.unlink(missing_ok=True)
        print("Glue results: video streams differ; re-encoding video while keeping the same exact-duration audio join.", flush=True)
        cmd, af = audio_inputs_and_filter()
        vf = ";".join([
            f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={src_d:.9f},setpts=PTS-STARTPTS[v0]",
            f"[1:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,trim=duration={seg_d:.9f},setpts=PTS-STARTPTS[v1]",
            "[v0][v1]concat=n=2:v=1:a=0[v]",
        ])
        re_tmp = workdir / "glued_reencode.mp4"
        cmd += [
            "-filter_complex", vf + ";" + af, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "256k", "-vsync", "0", "-shortest", str(re_tmp),
        ]
        subprocess.check_call(cmd, cwd=ROOT)
        if destination.exists(): destination.unlink()
        shutil.move(str(re_tmp), str(destination))
        print("Glue results complete: source and continuation joined on exact video/audio boundaries.", flush=True)

    with tempfile.TemporaryDirectory(prefix="h3_isolated_", dir=str(ROOT / "output")) as td:
        td = Path(td); lat = td / "latents.pt"; frames_dir = td / "frames"; wav = td / "audio.wav"
        sample_cmd = [py, "-m", "runtime.sample_worker", "--diffusion", str(diff), "--text-encoder", str(te), "--prompt", ns.prompt, "--width", str(width), "--height", str(height), "--frames", str(ns.frames), "--steps", str(ns.steps), "--cfg", str(ns.cfg), "--seed", str(ns.seed), "--shift", str(ns.shift), "--audio-shift", str(ns.audio_shift), "--sampler", ns.sampler, "--scheduler", ns.scheduler, "--out", str(lat)]
        if ns.experimental_long_duration: sample_cmd += ["--experimental-long-duration"]
        if ns.spectrum: sample_cmd += ["--spectrum"]
        sample_env = os.environ.copy()
        comfy_args = []
        if ns.sage_attention:
            comfy_args += ["--use-sage-attention"]
            print("SageAttention: enabled for sampling worker", flush=True)
        else:
            print("SageAttention: disabled", flush=True)
        if use_vram_manager:
            sample_cmd += ["--vram-manager", "--vram-runtime-free-gb", str(ns.vram_runtime_free_gb), "--vram-text-headroom-gb", str(ns.vram_text_headroom_gb), "--vram-diffusion-headroom-gb", str(ns.vram_diffusion_headroom_gb), "--vram-vae-headroom-gb", str(ns.vram_video_vae_reserve_gb), "--vram-offload-chunk-mb", str(ns.vram_offload_chunk_mb), "--vram-max-resident-weights-gb", str(ns.vram_max_resident_weights_gb), "--vram-block-check-interval", str(ns.vram_block_check_interval), "--vram-residency-target-free-gb", str(ns.vram_residency_target_free_gb), "--vram-residency-warmup-blocks", str(ns.vram_residency_warmup_blocks), "--vram-residency-refill-interval", str(ns.vram_residency_refill_interval)]
            if ns.vram_manager_auto:
                for stage in managed_sample_stages:
                    sample_cmd += ["--vram-managed-stage", stage]
            sample_cmd += ["--vram-residency-fill" if ns.vram_residency_fill else "--no-vram-residency-fill"]
            if ns.vram_keep_text_encoder: sample_cmd += ["--vram-keep-text-encoder"]
            # Keep the process-wide Comfy baseline small. VRAMManager.set_stage()
            # raises EXTRA_RESERVED_VRAM only for stages selected as managed and
            # restores this baseline for native stages.
            initial_reserve = max(ns.vram_runtime_free_gb, 0.5)
            comfy_args += ["--reserve-vram", str(initial_reserve)]
            if "diffusion" in managed_sample_stages and ns.vram_residency_engine == "static":
                # Classic/estimate-based ModelPatcher. Unlike DynamicVRAM/VBAR this gives
                # Comfy a concrete resident-weight budget and avoids intentionally mapping
                # most transformer weights through Windows shared GPU memory.
                comfy_args += ["--disable-dynamic-vram"]
            elif "diffusion" in managed_sample_stages:
                comfy_args += ["--enable-dynamic-vram"]
            if ns.vram_async_streams <= 0: comfy_args += ["--disable-async-offload"]
            else: comfy_args += ["--async-offload", str(ns.vram_async_streams)]
            print(f"VRAM Manager V11.4: sample stages={','.join(managed_sample_stages) if ns.vram_manager_auto else 'forced all'} | engine={ns.vram_residency_engine} | runtime free={ns.vram_runtime_free_gb:g} GB | text load headroom={ns.vram_text_headroom_gb:g} GB | diffusion load headroom={ns.vram_diffusion_headroom_gb:g} GB | chunk={ns.vram_offload_chunk_mb} MB | max weights={'auto' if ns.vram_max_resident_weights_gb <= 0 else f'{ns.vram_max_resident_weights_gb:g} GB'} | block check={ns.vram_block_interval if hasattr(ns, 'vram_block_interval') else ns.vram_block_check_interval} | async streams={ns.vram_async_streams}", flush=True)
        if comfy_args:
            sample_env["H3_COMFY_ARGS"] = " ".join(comfy_args)
        for lp, strength in zip(ns.lora, ns.lora_strength): sample_cmd += ["--lora", str(Path(lp).resolve()), "--lora-strength", str(strength)]
        if ns.extended_logging: sample_cmd += ["--extended-logging"]
        if ns.first_frame or ns.last_frame or ns.continue_video:
            sample_cmd += ["--video-vae", str(vv)]
            if ns.first_frame: sample_cmd += ["--first-frame", str(Path(ns.first_frame).resolve())]
            if ns.last_frame: sample_cmd += ["--last-frame", str(Path(ns.last_frame).resolve())]
            if ns.continue_video:
                sample_cmd += ["--continue-video", str(Path(ns.continue_video).resolve()), "--continue-context-frames", str(ns.continue_context_frames)]
                if ns.continue_audio_memory:
                    sample_cmd += ["--continue-audio-memory", "--audio-vae", str(av)]
        subprocess.check_call(sample_cmd, cwd=ROOT, env=sample_env)
        print("Sampling process exited completely. Starting VAE with clean memory...", flush=True)

        video_env = os.environ.copy()
        video_decode_managed = bool(stage_plan and stage_plan["stages"]["video_decode"]["use_manager"])
        if ns.vram_manager_auto and not video_decode_managed:
            video_env["H3_COMFY_ARGS"] = f"--reserve-vram {max(0.1, ns.vram_runtime_free_gb):g} --fp16-vae"
            print("[VRAM-AUTO] video VAE decode launching NATIVE", flush=True)
        else:
            video_env["H3_COMFY_ARGS"] = f"--lowvram --reserve-vram {max(0.1, ns.vram_video_vae_reserve_gb):g} --fp16-vae"
            if ns.vram_manager_auto: print("[VRAM-AUTO] video VAE decode launching MANAGED (--lowvram)", flush=True)
        video_cmd = [py, "-m", "runtime.video_decode_worker", "--latents", str(lat), "--vae", str(vv), "--frames-dir", str(frames_dir), "--tile-size", str(ns.video_vae_tile_size), "--tile-overlap", str(ns.video_vae_tile_overlap)]
        print(f"Video VAE tiling: {ns.video_vae_tile_size}px tile / {ns.video_vae_tile_overlap}px overlap", flush=True)
        if ns.extended_logging: video_cmd += ["--extended-logging"]
        if ns.tile_debugging: video_cmd += ["--tile-debugging"]
        subprocess.check_call(video_cmd, cwd=ROOT, env=video_env)

        if ns.tile_debugging:
            debug_dir = out.parent / (out.stem + "_tile_debug"); debug_dir.mkdir(parents=True, exist_ok=True); copied = 0
            for name in ("tile_debug_plan.txt", "tile_debug_grid.png", "tile_debug_overlay_frame0.png", "frame_000000.png"):
                src = frames_dir / name
                if src.is_file(): shutil.copy2(src, debug_dir / name); copied += 1
            print(f"Persistent tile diagnostics: {debug_dir} ({copied} files copied)", flush=True)

        ok, msg = ensure_ffmpeg_tools(lambda x: print(x, flush=True))
        if not ok:
            print(f"ERROR: FFmpeg tools unavailable: {msg}", flush=True); return 2
        ffmpeg = str(tool_path("ffmpeg.exe")); ffprobe = str(tool_path("ffprobe.exe")); mux_tmp = td / "final_with_audio.mp4"
        segment_out = (td / "generated_segment.mp4") if ns.glue_source else out
        audio_env = os.environ.copy()
        audio_decode_managed = bool(stage_plan and stage_plan["stages"]["audio_decode"]["use_manager"])
        if ns.vram_manager_auto and not audio_decode_managed:
            audio_env["H3_COMFY_ARGS"] = f"--reserve-vram {max(0.1, ns.vram_runtime_free_gb):g}"
            print("[VRAM-AUTO] audio VAE decode launching NATIVE", flush=True)
        else:
            audio_env["H3_COMFY_ARGS"] = f"--novram --reserve-vram {max(0.1, ns.vram_audio_vae_reserve_gb):g}"
            if ns.vram_manager_auto: print("[VRAM-AUTO] audio VAE decode launching MANAGED (--novram)", flush=True)
        audio_cmd = [py, "-m", "runtime.audio_decode_worker", "--latents", str(lat), "--vae", str(av), "--wav", str(wav)]
        if ns.extended_logging: audio_cmd += ["--extended-logging"]
        audio_ok = False
        try:
            subprocess.check_call(audio_cmd, cwd=ROOT, env=audio_env)
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            if not frame_files:
                raise RuntimeError("Video decode produced no frames for mux")
            exact_duration = len(frame_files) / 24.0
            print(f"Muxing video and audio on exact frame duration: {len(frame_files)} frames / {exact_duration:.6f}s", flush=True)
            cmd = [ffmpeg, "-y", "-framerate", "24", "-i", str(frames_dir / "frame_%06d.png"), "-i", str(wav),
                   "-filter:a", f"aresample=32000,apad,atrim=duration={exact_duration:.9f},asetpts=PTS-STARTPTS",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-b:a", "256k",
                   "-t", f"{exact_duration:.9f}", str(mux_tmp)]
            subprocess.check_call(cmd)
            if segment_out.exists(): segment_out.unlink()
            shutil.move(str(mux_tmp), str(segment_out)); audio_ok = True
        except subprocess.CalledProcessError as e:
            print(f"Audio/mux stage failed ({e}). Creating COMPLETE video-only fallback instead of leaving a partial MP4...", flush=True)
            if mux_tmp.exists(): mux_tmp.unlink()
            if segment_out.exists(): segment_out.unlink()
            subprocess.check_call([ffmpeg, "-y", "-framerate", "24", "-i", str(frames_dir / "frame_%06d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(segment_out)])
        if ns.glue_source:
            try:
                _concat_source_and_segment(ffmpeg, ffprobe, ns.glue_source, segment_out, out, width, height, td)
            except Exception as exc:
                print(f"ERROR: Glue results failed: {type(exc).__name__}: {exc}", flush=True)
                return 2
        if audio_ok:
            print("Saved with audio:", out, flush=True); return 0
        print("Saved FULL video-only fallback:", out, flush=True); return 3

if __name__ == "__main__": raise SystemExit(main())
