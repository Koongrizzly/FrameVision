
from __future__ import annotations
import os
from pathlib import Path
from PySide6.QtCore import QSettings

ORG = "FrameVision"
APP = "FrameVision"

IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")

def _app_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _local_intro_dir() -> Path:
    # prefer explicit setting, else presets/startup under root
    s = QSettings(ORG, APP)
    p = s.value("intro_local_dir", "", type=str) or ""
    base = Path(p) if p else (_app_root() / "presets" / "startup")
    return base

def _list_local_images() -> list[str]:
    base = _local_intro_dir()
    if not base.exists():
        return []
    files = []
    for ext in IMG_EXT:
        files += [str(p) for p in base.rglob(f"*{ext}")]
    return files

def _match_theme(paths: list[str], theme_name: str) -> list[str]:
    if not paths or not theme_name:
        return []
    t = theme_name.strip().lower()
    keys = ["day"] if t.startswith("day") else (["even", "eve"] if t.startswith("even") else (["night"] if t.startswith("night") else []))
    out = []
    for p in paths:
        n = os.path.basename(p).lower()
        parts = [x.lower() for x in Path(p).parts]
        if any(k in n for k in keys) or any(k in parts for k in keys):
            out.append(p)
    return out

def get_logo_sources(theme: str | None = None) -> list[str]:
    s = QSettings(ORG, APP)
    follow = bool(s.value("intro_follow_theme", False, type=bool))
    local = _list_local_images()

    # Always prefer local; if we have any local images, never go remote.
    if local:
        if follow and theme:
            themed = _match_theme(local, theme)[:3]
            if themed:
                return themed
        return local

    # No local files -> empty to avoid pulling remote album
    return []
