
# -*- coding: utf-8 -*-
"""
model_assets.py — First-run extractor for bundled model packs.

Responsibilities:
- Look for assets/rife_models.pack (zip-compatible).
- On first run (or if models are missing), extract into models/rife/default/.
- If inner .zip files are present inside the pack, auto-extract them and
  remove the .zip files after extraction.
- Returns the resolved models folder path.
"""
from __future__ import annotations
import os, zipfile, shutil
from pathlib import Path

def _is_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, 'r') as z:
            z.namelist()
        return True
    except Exception:
        return False

def ensure_rife_models(root: Path | str) -> Path | None:
    root = Path(root).resolve()
    models_root = root / "models" / "rife" / "default"
    models_root.mkdir(parents=True, exist_ok=True)

    # If any known model files/dirs already exist, nothing to do.
    try:
        if any(models_root.iterdir()):
            return models_root
    except Exception:
        pass

    pack = root / "assets" / "rife_models.pack"
    if not pack.exists():
        # Nothing to extract; caller can decide to download later.
        return models_root if models_root.exists() else None

    # If .pack is actually a .zip, extract it.
    if _is_zip(pack):
        with zipfile.ZipFile(pack, 'r') as z:
            z.extractall(models_root)

        # If the pack contained nested zips, extract them then delete.
        inner_zips = list(models_root.rglob("*.zip"))
        for zpath in inner_zips:
            try:
                with zipfile.ZipFile(zpath, 'r') as zin:
                    target_dir = zpath.with_suffix("")  # remove .zip
                    target_dir.mkdir(parents=True, exist_ok=True)
                    zin.extractall(target_dir)
                zpath.unlink(missing_ok=True)
            except Exception:
                # If an inner zip fails to extract, keep it for manual inspection.
                pass

        # Clean up any leftover empty folders common in release archives.
        for p in list(models_root.glob("**/*"))[::-1]:
            try:
                if p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
            except Exception:
                pass
    else:
        # Unknown format — just copy alongside as-is for manual extraction.
        dest = models_root / pack.name
        try:
            shutil.copy2(pack, dest)
        except Exception:
            pass

    return models_root if models_root.exists() else None
