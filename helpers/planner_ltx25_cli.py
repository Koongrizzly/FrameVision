from __future__ import annotations
import argparse, json, os, random, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name.lower() == "helpers" else HERE

# Normal/full-size LTX 2.5 backend (fallback)
ENV_PY = ROOT / "environments" / "ltx25" / "Scripts" / "python.exe"
HELPER = ROOT / "helpers" / "ltx25_helper.py"
MODELS = ROOT / "models" / "ltx-2.5"

# ConvRot backend (preferred, matching the GUI helper)
CONVROT_ENV_PY = ROOT / "environments" / "ltx25_convrot" / "Scripts" / "python.exe"
CONVROT_WORKER = ROOT / "helpers" / "ltx25_convrot_worker.py"
CONVROT_MODELS = ROOT / "models" / "ltx_2_5_convrot"
CONVROT_CACHE = CONVROT_MODELS / "cache"


def _frames(n: int) -> int:
    n = max(9, int(n))
    return max(9, n - ((n - 1) % 8))


def _valid_output(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= 1024
    except OSError:
        return False


def _convrot_paths(kind: str = "W4A8") -> dict[str, str]:
    if kind == "INT4":
        transformer = CONVROT_MODELS / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-int4_convrot.safetensors"
        text_encoder = CONVROT_MODELS / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-int4_convrot.safetensors"
    else:
        transformer = CONVROT_MODELS / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-w4a8_convrot.safetensors"
        text_encoder = CONVROT_MODELS / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-w4a8_convrot.safetensors"
    return {
        "transformer": str(transformer),
        "text_encoder": str(text_encoder),
        "video_vae": str(CONVROT_MODELS / "vae" / "ltx-2.5-video-vae-bf16.safetensors"),
        "audio_vae": str(CONVROT_MODELS / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"),
        "upsampler": str(CONVROT_MODELS / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    }


def _full_paths() -> dict[str, str]:
    return {
        "transformer": str(MODELS / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        "text_encoder": str(MODELS / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        "video_vae": str(MODELS / "vae" / "ltx-2.5-video-vae-bf16.safetensors"),
        "audio_vae": str(MODELS / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"),
        "upsampler": str(MODELS / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    }


def _paths_exist(paths: dict[str, str]) -> bool:
    return all(Path(p).is_file() for p in paths.values())


def _base_job(ns, seed: int, image: str, out: Path) -> dict:
    return {
        "cmd": "generate",
        "prompt": ns.prompt,
        "workflow": "two_phase",
        "audio_path": "",
        "seed": seed,
        "width": int(ns.width),
        "height": int(ns.height),
        "frames": _frames(ns.frames),
        "fps": float(ns.fps),
        "output": str(out),
        "images": ([{"path": image, "frame_idx": 0, "strength": 1.0}] if image else []),
        "offload": "cpu",
        "max_batch_size": 1,
        "use_sage_attention": False,
        "use_int8_transformer": False,
        "int8_transformer_bundle": "",
        "use_int8_text_encoder": False,
        "int8_text_encoder_bundle": "",
        "enhance_prompt": False,
        "defer_trim": True,
        "cache_prompt_embeddings": True,
    }


def _write_job(job: dict, out: Path, prefix: str) -> str:
    fd, tmp = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=str(out.parent))
    os.close(fd)
    Path(tmp).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return tmp


def _convrot_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    for sub in ("huggingface", "torch_extensions", "triton", "pip"):
        (CONVROT_CACHE / sub).mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(CONVROT_CACHE / "huggingface")
    env["HF_HUB_CACHE"] = str(CONVROT_CACHE / "huggingface" / "hub")
    env["TORCH_EXTENSIONS_DIR"] = str(CONVROT_CACHE / "torch_extensions")
    env["TRITON_CACHE_DIR"] = str(CONVROT_CACHE / "triton")
    env["PIP_CACHE_DIR"] = str(CONVROT_CACHE / "pip")
    return env


def _run_convrot(base_job: dict, out: Path, kind: str) -> int | None:
    paths = _convrot_paths(kind)
    if not (CONVROT_ENV_PY.is_file() and CONVROT_WORKER.is_file() and _paths_exist(paths)):
        return None

    job = dict(base_job)
    job["model_type"] = f"{kind} ConvRot" if kind == "INT4" else "W4A8 ConvRot (recommended)"
    job["paths"] = paths
    job["quantization"] = "none"

    tmp = _write_job(job, out, "planner_ltx25_convrot_")
    print(f"[LTX25 Planner] Using {job['model_type']} backend.", flush=True)
    try:
        return int(subprocess.call(
            [str(CONVROT_ENV_PY), str(CONVROT_WORKER), "--job", tmp],
            cwd=str(ROOT),
            env=_convrot_env(),
        ))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


def _run_full(base_job: dict, out: Path) -> int:
    if not ENV_PY.is_file():
        raise SystemExit(f"LTX 2.5 environment not found: {ENV_PY}")
    if not HELPER.is_file():
        raise SystemExit(f"LTX 2.5 helper not found: {HELPER}")

    paths = _full_paths()
    missing = [p for p in paths.values() if not Path(p).is_file()]
    if missing:
        raise SystemExit("LTX 2.5 full-size model file(s) not found:\n" + "\n".join(missing))

    job = dict(base_job)
    job["model_type"] = "Full FP16 / BF16"
    job["paths"] = paths
    job["quantization"] = "fp8-cast"

    tmp = _write_job(job, out, "planner_ltx25_full_")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    print("[LTX25 Planner] Using full FP16/BF16 backend (FP8-cast fallback).", flush=True)
    try:
        return int(subprocess.call(
            [str(ENV_PY), str(HELPER), "--queue-job", tmp],
            cwd=str(ROOT),
            env=env,
        ))
    finally:
        try:
            Path(tmp).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="FrameVision Planner LTX 2.5 launcher")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--image", default="")
    ap.add_argument("--output", required=True)
    ap.add_argument("--width", type=int, required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=-1)
    ns = ap.parse_args()

    seed = ns.seed if ns.seed >= 0 else random.randint(0, 2147483647)
    image = str(Path(ns.image).resolve()) if ns.image else ""
    if image and not Path(image).is_file():
        raise SystemExit(f"Start image not found: {image}")

    out = Path(ns.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    base_job = _base_job(ns, seed, image, out)

    # Preferred order mirrors the GUI's optimized ConvRot choices:
    # W4A8 ConvRot -> INT4 ConvRot -> normal/full-size LTX 2.5.
    for kind in ("W4A8", "INT4"):
        rc = _run_convrot(base_job, out, kind)
        if rc is None:
            continue
        if rc == 0 and _valid_output(out):
            return 0
        if _valid_output(out):
            return 0
        print(f"[LTX25 Planner] {kind} ConvRot failed (exit code {rc}); trying next fallback.", flush=True)

    rc = _run_full(base_job, out)
    if rc != 0:
        return int(rc)
    return 0 if _valid_output(out) else 3


if __name__ == "__main__":
    raise SystemExit(main())
