#!/usr/bin/env python3
"""
Download background-removal model weights to a local folder for fully offline use.
- Default: download only a lightweight portrait model (MODNet, ONNX).
- With --pro: also download BiRefNet (ONNX) for high-quality general matting (large file).

Extras:
- After completion, **all downloaded files are moved to the project root folder `models/bg`**,
  regardless of the temporary download destination used. This ensures a consistent final location
  for downstream tools that expect models under `models/bg`.
- Finally, any stray zip files directly in `models/` (not its subfolders) are deleted.

Usage:
    python downloadbg.py [--dest models/bg] [--pro]
"""
import argparse, hashlib, os, sys, urllib.request, pathlib, shutil

MODELS = {
    "modnet_onnx": {
        "url": "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/modnet_photographic_portrait_matting.onnx?download=true",
        "sha256": "07C308CF0FC7E6E8B2065A12ED7FC07E1DE8FEBB7DC7839D7B7F15DD66584DF9".lower(),
        "filename": "modnet_photographic_portrait_matting.onnx",
        "size_hint": "≈ 100 MB",
        "optional": False
    },
    "birefnet_onnx": {
        # Large, high-quality SOD model (can exceed 900 MB). Only fetched with --pro.
        "url": "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-COD-epoch_125.onnx",
        "sha256": None,  # Upstream does not publish a stable checksum for this asset
        "filename": "BiRefNet-COD-epoch_125.onnx",
        "size_hint": "≈ 900 MB",
        "optional": True
    },
}

def sha256sum(path):
    h=hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, dest_path, expected_sha256=None):
    tmp = str(dest_path) + ".part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    os.replace(tmp, dest_path)
    if expected_sha256:
        got = sha256sum(dest_path)
        if got.lower() != expected_sha256.lower():
            raise RuntimeError(f"SHA256 mismatch for {dest_path.name}: {got} != {expected_sha256}")

def move_all_to_root_models_bg(temp_dest: pathlib.Path) -> int:
    """Move *files* from temp_dest into project-root 'models/bg'.
    Returns the number of files moved. If temp_dest already equals the final root, this is a no-op.
    """
    final_root = pathlib.Path("models/bg")
    final_root.mkdir(parents=True, exist_ok=True)

    # Resolve paths to avoid self-move when they point to the same directory
    try:
        same_dir = temp_dest.resolve() == final_root.resolve()
    except Exception:
        same_dir = False

    moved = 0
    # Only move direct children (the script stores files flat)
    for p in temp_dest.glob("*"):
        if not p.is_file():
            continue
        target = final_root / p.name
        try:
            if same_dir and p.resolve() == target.resolve():
                # already in the correct place
                continue
        except Exception:
            pass
        # If target exists, replace it atomically
        if target.exists():
            try:
                target.unlink()
            except Exception:
                # On Windows, ensure it's not read-only; then retry once
                try:
                    os.chmod(target, 0o666)
                    target.unlink()
                except Exception:
                    print(f"[warn] Could not replace existing {target}")
                    continue
        try:
            shutil.move(str(p), str(target))
            moved += 1
        except Exception as e:
            print(f"[warn] Move failed for {p.name}: {e}")
    return moved

def cleanup_model_root_zips() -> int:
    """Delete *.zip files sitting directly in models/ (not subfolders)."""
    root = pathlib.Path("models")
    if not root.exists():
        return 0
    deleted = 0
    for z in root.glob("*.zip"):  # no recursion -> ignores subfolders
        try:
            z.unlink()
            deleted += 1
        except PermissionError:
            # On Windows, clear read-only and retry once
            try:
                os.chmod(z, 0o666)
                z.unlink()
                deleted += 1
            except Exception as e:
                print(f"[warn] Could not delete {z.name}: {e}")
        except Exception as e:
            print(f"[warn] Could not delete {z.name}: {e}")
    return deleted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="models/bg", help="Temporary download folder (files will be moved to project-root models/bg at the end)")
    ap.add_argument("--pro", action="store_true", help="Also download large BiRefNet ONNX (≈900MB)")
    args = ap.parse_args()

    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    to_get = ["modnet_onnx"]
    if args.pro:
        to_get.append("birefnet_onnx")

    any_fail = False
    for key in to_get:
        m = MODELS[key]
        out = dest / m["filename"]
        if out.exists():
            print(f"[skip] {out.name} already exists in {dest}")
            continue
        print(f"[get ] {out.name}  {m['size_hint']}")
        try:
            fetch(m["url"], out, m["sha256"])
        except Exception as e:
            any_fail = True
            print(f"[fail] {out.name}: {e}")
            break
        else:
            print(f"[ ok ] Saved to {out}")
    if any_fail:
        print("[done] Encountered errors; skipping move-to-root step.")
        return 1

    # Finalize: move everything to the project-root models/bg
    moved = move_all_to_root_models_bg(dest)
    final_root = pathlib.Path("models/bg").resolve()
    if moved:
        print(f"[move] Moved {moved} file(s) to {final_root}")
    else:
        print(f"[move] No files moved; they were already in {final_root}")

    # NEW: clean stray .zip files in models/
    removed = cleanup_model_root_zips()
    if removed:
        print(f"[clean] Removed {removed} zip file(s) from {pathlib.Path('models').resolve()}")
    else:
        print(f"[clean] No zip files to remove in {pathlib.Path('models').resolve()}")

    print("[done] Background models ready in", final_root)
    return 0

if __name__ == "__main__":
    sys.exit(main())
