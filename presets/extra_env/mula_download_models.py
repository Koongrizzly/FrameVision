import glob
import os
import sys
import traceback

from huggingface_hub import snapshot_download


def dl(repo_id: str, local_dir: str):
    print(f"[HeartMuLa] download: {repo_id} -> {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def _has_any_safetensors(path: str) -> bool:
    return bool(glob.glob(os.path.join(path, "**", "*.safetensors"), recursive=True))


def main():
    if len(sys.argv) < 2:
        print("Usage: python mula_download_models.py <models_root>")
        return 2

    root = os.path.abspath(sys.argv[1])
    os.makedirs(root, exist_ok=True)

    # Helpful diagnostics for Windows/network issues
    print(f"[HeartMuLa] models_root: {root}")
    print(f"[HeartMuLa] HF_HOME: {os.environ.get('HF_HOME', '')}")
    if os.environ.get("HTTP_PROXY"):
        print(f"[HeartMuLa] HTTP_PROXY: {os.environ.get('HTTP_PROXY')}")
    if os.environ.get("HTTPS_PROXY"):
        print(f"[HeartMuLa] HTTPS_PROXY: {os.environ.get('HTTPS_PROXY')}")
    if os.environ.get("HF_HUB_DISABLE_XET"):
        print(f"[HeartMuLa] HF_HUB_DISABLE_XET: {os.environ.get('HF_HUB_DISABLE_XET')}")

    try:
        # Root repo contains configs/tokenizer in addition to subfolders
        dl("HeartMuLa/HeartMuLaGen", root)
        dl("HeartMuLa/HeartMuLa-oss-3B", os.path.join(root, "HeartMuLa-oss-3B"))
        dl("HeartMuLa/HeartCodec-oss", os.path.join(root, "HeartCodec-oss"))

        # Sanity checks (avoid 'it ran but nothing downloaded')
        m1 = os.path.join(root, "HeartMuLa-oss-3B")
        m2 = os.path.join(root, "HeartCodec-oss")
        if not os.path.isdir(m1) or not os.path.isdir(m2):
            print("[HeartMuLa][ERROR] Expected model folders missing.")
            return 3
        if not _has_any_safetensors(m1):
            print("[HeartMuLa][ERROR] No .safetensors found under HeartMuLa-oss-3B.")
            return 4
        if not _has_any_safetensors(m2):
            print("[HeartMuLa][ERROR] No .safetensors found under HeartCodec-oss.")
            return 5

        print("[HeartMuLa] done. Model folder structure should now contain:")
        print("  HeartCodec-oss/")
        print("  HeartMuLa-oss-3B/")
        print("  (plus config/tokenizer files in the root repo)")
        return 0

    except Exception as e:
        print("[HeartMuLa][ERROR] Download crashed:")
        print(str(e))
        traceback.print_exc()
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
