import argparse
import os
import sys
from pathlib import Path

def _print(msg: str):
    print(msg, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="App root folder (optional; auto-detected if omitted)")
    ap.add_argument("--transformer_gguf", default="ltx-video-2b-v0.9-q5_k_m.gguf")
    ap.add_argument("--t5_gguf", default="t5xxl_fp16-q4_0.gguf")
    ap.add_argument("--vae_gguf", default="ltxv_vae_fp32-f16.gguf")
    ap.add_argument("--skip_vae", action="store_true", help="Skip downloading optional VAE GGUF")
    args = ap.parse_args()

    # Defensive root sanitization (Windows quoting edge-cases)
    root_str = (args.root or "").strip()
    # If a stray quote caused the rest of the command line to be captured, trim it.
    if '" --' in root_str:
        root_str = root_str.split('" --', 1)[0]
    if ' --' in root_str:
        root_str = root_str.split(' --', 1)[0]
    root_str = root_str.strip().strip('"')

    root = Path(root_str).resolve()
    models_dir = root / "models" / "ltx"
    ltxv_repo_dir = models_dir / "calcuis_ltxv-gguf"
    decoder_dir = models_dir / "callgg_ltxv-decoder"
    base_dir = models_dir / "Lightricks_LTX-Video"

    # Keep all HF caches inside app folders (portable)
    hf_home = models_dir / "_hf_home"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ["TORCH_HOME"] = str(hf_home / "torch")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    ltxv_repo_dir.mkdir(parents=True, exist_ok=True)
    decoder_dir.mkdir(parents=True, exist_ok=True)
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except Exception as e:
        _print(f"[LTX] ERROR: huggingface_hub import failed: {e}")
        return 1

    _print("[LTX] Downloading small metadata files for calcuis/ltxv-gguf ...")
    try:
        snapshot_download(
            repo_id="calcuis/ltxv-gguf",
            local_dir=str(ltxv_repo_dir),
            local_dir_use_symlinks=False,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.md",
                "tokenizer/*",
                "tokenizer/**",
                "text_encoder/*",
                "text_encoder/**",
                "*.model",
                "*.spm",
                "*.sentencepiece",
            ],
        )
    except Exception as e:
        _print(f"[LTX] WARN: snapshot_download(calcuis/ltxv-gguf) metadata failed: {e}")

    _print(f"[LTX] Downloading GGUF transformer: {args.transformer_gguf} ...")
    try:
        hf_hub_download(
            repo_id="calcuis/ltxv-gguf",
            filename=args.transformer_gguf,
            local_dir=str(ltxv_repo_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        _print(f"[LTX] ERROR: failed to download transformer GGUF: {e}")
        return 2

    _print(f"[LTX] Downloading GGUF text encoder: {args.t5_gguf} ...")
    try:
        hf_hub_download(
            repo_id="calcuis/ltxv-gguf",
            filename=args.t5_gguf,
            local_dir=str(ltxv_repo_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        _print(f"[LTX] ERROR: failed to download T5 GGUF: {e}")
        return 3

    if not args.skip_vae:
        _print(f"[LTX] Downloading optional GGUF VAE: {args.vae_gguf} ...")
        try:
            hf_hub_download(
                repo_id="calcuis/ltxv-gguf",
                filename=args.vae_gguf,
                local_dir=str(ltxv_repo_dir),
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            _print(f"[LTX] WARN: failed to download VAE GGUF (optional): {e}")

    _print("[LTX] Downloading decoder repo callgg/ltxv-decoder ...")
    try:
        snapshot_download(
            repo_id="callgg/ltxv-decoder",
            local_dir=str(decoder_dir),
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        _print(f"[LTX] ERROR: failed to download decoder repo: {e}")
        return 4

    _print("[LTX] Downloading base pipeline repo Lightricks/LTX-Video (portable, no large weights)...")
    try:
        # We only need pipeline/config/tokenizer/scheduler/VAE bits; transformer/text-encoder weights come from GGUF.
        snapshot_download(
            repo_id="Lightricks/LTX-Video",
            local_dir=str(base_dir),
            local_dir_use_symlinks=False,
            ignore_patterns=[
                "transformer/*",
                "transformer/**",
                "text_encoder/*",
                "text_encoder/**",
            ],
        )
    except Exception as e:
        _print(f"[LTX] WARN: failed to download base pipeline repo (will fallback to decoder repo): {e}")

    _print("[LTX] Done.")
    _print(f"[LTX] Files are in:\n  {ltxv_repo_dir}\n  {decoder_dir}\n  {base_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
