
from __future__ import annotations
import os, shutil, sys
from pathlib import Path

def _rm(path: Path) -> int:
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
        return 1
    except Exception:
        return 0

def _iter_matches(base: Path, patterns: list[str]):
    for pat in patterns:
        for p in base.rglob(pat):
            yield p

def _huggingface_cache_dirs() -> list[Path]:
    # Common locations on Windows
    env = os.environ
    dirs = []
    if "HF_HOME" in env:
        dirs.append(Path(env["HF_HOME"]).expanduser())
    home = Path.home()
    dirs.append(home / ".cache" / "huggingface")
    # de-dup
    out, seen = [], set()
    for d in dirs:
        d = d.expanduser().resolve()
        if d not in seen:
            out.append(d); seen.add(d)
    return out

def run_cleanup(project_root: str | None = None, *, 
                clean_pyc: bool = True,
                clean_logs: bool = True,
                clean_thumbs: bool = True,
                clean_qt_cache: bool = True,
                clean_hf_cache: bool = False) -> dict:
    """
    Returns dict with counts of removed items by category.
    Safe defaults: remove __pycache__ and *.pyc/*.pyo and simple log/thumbnail caches under the project.
    """
    base = Path(project_root or os.getcwd()).resolve()
    removed = {"pycache":0, "pyc_pyo":0, "logs":0, "thumbs":0, "qt":0, "hf":0}

    if clean_pyc:
        for p in _iter_matches(base, ["__pycache__"]):
            removed["pycache"] += _rm(p)
        for p in _iter_matches(base, ["*.pyc","*.pyo"]):
            removed["pyc_pyo"] += _rm(p)

    if clean_logs:
        for p in _iter_matches(base, ["logs", "log", "*.log"]):
            if p.name.lower().endswith(".log") or p.is_dir():
                removed["logs"] += _rm(p)

    if clean_thumbs:
        # common temp/thumbnail dirs used in many apps; adjust as needed
        for p in _iter_matches(base, ["thumbs","thumbnails",".thumbs",".thumbnails","cache","tmp",".tmp"]):
            # don't nuke venv or models
            low = p.as_posix().lower()
            if any(x in low for x in ("/.venv","\\\.venv","/env","\\env","/models","\\models")):
                continue
            removed["thumbs"] += _rm(p)

    if clean_qt_cache:
        # QtWebEngine cache dirs sometimes appear under user profile; ignore failures
        qt_dirs = []
        try:
            qt_dirs.append(Path.home()/ "AppData/Local" / "QtProject")
            qt_dirs.append(Path.home()/ "AppData/Local" / "framevision")
        except Exception:
            pass
        for d in qt_dirs:
            if d.exists():
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    removed["qt"] += 1
                except Exception:
                    pass

    if clean_hf_cache:
        for d in _huggingface_cache_dirs():
            if d.exists():
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    removed["hf"] += 1
                except Exception:
                    pass

    return removed
