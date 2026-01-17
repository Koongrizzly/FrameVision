"""
SongGeneration asset downloader (Hugging Face).

Run using the dedicated env:
  .song_g_env\Scripts\python.exe helpers\song_g_download.py tencent_base
  .song_g_env\Scripts\python.exe helpers\song_g_download.py runtime
  .song_g_env\Scripts\python.exe helpers\song_g_download.py ckpt --model large

Targets:
  - Downloads ckpt/** and third_party/** into: models/song_generation/
  - Downloads a chosen checkpoint into: models/song_generation/ckpt/<folder>/
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return root_dir() / "models" / "song_generation"


def enable_hf_transfer() -> bool:
    try:
        import hf_transfer  # noqa: F401
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        return True
    except Exception:
        return False


MODEL_MAP = {
    "large": ("lglg666/SongGeneration-large", "songgeneration_large"),
    "base-full": ("lglg666/SongGeneration-base-full", "songgeneration_base_full"),
    "base-new": ("lglg666/SongGeneration-base-new", "songgeneration_base_new"),
    "base": ("lglg666/SongGeneration-base", "songgeneration_base"),
}


def dl_ckpt_and_third_party(repo_id: str, target_repo_root: Path) -> None:
    target_repo_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_repo_root),
        local_dir_use_symlinks=False,
        allow_patterns=["ckpt/**", "third_party/**"],
        resume_download=True,
    )


def dl_checkpoint(repo_id: str, folder_name: str, target_repo_root: Path) -> Path:
    ckpt_dir = target_repo_root / "ckpt" / folder_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(ckpt_dir),
        local_dir_use_symlinks=False,
        allow_patterns=["config.yaml", "model.pt"],
        resume_download=True,
    )
    return ckpt_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tencent_base", "runtime", "ckpt"])
    ap.add_argument("--model", default="large", help="ckpt model: large | base-full | base-new | base")
    args = ap.parse_args()

    fast = enable_hf_transfer()
    print(f"[DL] hf_transfer: {'ON' if fast else 'OFF'}")

    target = repo_root()
    print(f"[DL] Target repo root: {target}")

    if args.mode == "tencent_base":
        repo_id = "tencent/SongGeneration"
        print(f"[DL] Downloading ckpt + third_party from {repo_id} ...")
        dl_ckpt_and_third_party(repo_id, target)
        print("[DL] Done.")
        return

    if args.mode == "runtime":
        repo_id = "lglg666/SongGeneration-Runtime"
        print(f"[DL] Downloading runtime (ckpt + third_party) from {repo_id} ...")
        dl_ckpt_and_third_party(repo_id, target)
        print("[DL] Done.")
        return

    if args.mode == "ckpt":
        key = args.model.strip().lower()
        if key not in MODEL_MAP:
            raise SystemExit(f"Unknown model '{key}'. Options: {', '.join(MODEL_MAP.keys())}")
        repo_id, folder_name = MODEL_MAP[key]
        print(f"[DL] Downloading checkpoint {key} from {repo_id} ...")
        ckpt_dir = dl_checkpoint(repo_id, folder_name, target)
        print(f"[DL] Done. Checkpoint folder: {ckpt_dir}")
        return


if __name__ == "__main__":
    main()
