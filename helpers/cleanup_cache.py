from __future__ import annotations
import os, shutil, sys, time
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
            out.append(d)
            seen.add(d)
    return out


def _is_under_any(path: Path, roots: list[Path]) -> bool:
    """Return True if path is equal to or inside any of the given roots."""
    try:
        path_resolved = path.resolve()
    except Exception:
        path_resolved = path
    for root in roots:
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root
        if root_resolved == path_resolved or root_resolved in path_resolved.parents:
            return True
    return False


def _is_older_than(path: Path, cutoff_ts: float) -> bool:
    try:
        return path.stat().st_mtime < cutoff_ts
    except OSError:
        return False


def run_cleanup(
    project_root: str | None = None,
    *,
    clean_pyc: bool = True,
    clean_logs: bool = True,
    clean_thumbs: bool = True,
    clean_qt_cache: bool = True,
    clean_hf_cache: bool = False,
) -> dict:
    """
    Returns dict with counts of removed items by category.
    Safe defaults: remove __pycache__ and *.pyc/*.pyo and simple log/thumbnail caches under the project.
    """
    base = Path(project_root or os.getcwd()).resolve()
    removed = {"pycache": 0, "pyc_pyo": 0, "logs": 0, "thumbs": 0, "qt": 0, "hf": 0}

    if clean_pyc:
        # Do not touch bytecode inside our virtual envs / extra env presets
        exclude_roots = [
            (base / ".env"),
            (base / ".wan_env"),
            (base / "presets" / "extra_env"),
        ]

        for p in _iter_matches(base, ["__pycache__"]):
            if _is_under_any(p, exclude_roots):
                continue
            removed["pycache"] += _rm(p)

        for p in _iter_matches(base, ["*.pyc", "*.pyo"]):
            if _is_under_any(p, exclude_roots):
                continue
            removed["pyc_pyo"] += _rm(p)

    if clean_logs:
        # Remove log files older than 24 hours
        cutoff = time.time() - 24 * 60 * 60

        # 1) root /logs/ directory
        logs_dir = base / "logs"
        if logs_dir.exists() and logs_dir.is_dir():
            for entry in logs_dir.rglob("*"):
                if entry.is_file() and _is_older_than(entry, cutoff):
                    removed["logs"] += _rm(entry)

        # 2) any *.log file inside /helpers/ and its subfolders
        helpers_dir = base / "helpers"
        if helpers_dir.exists() and helpers_dir.is_dir():
            for log_file in helpers_dir.rglob("*.log"):
                if log_file.is_file() and _is_older_than(log_file, cutoff):
                    removed["logs"] += _rm(log_file)

    if clean_thumbs:
        # common temp/thumbnail dirs used in many apps; adjust as needed
        for p in _iter_matches(
            base,
            ["thumbs", "thumbnails", ".thumbs", ".thumbnails", "cache", "tmp", ".tmp"],
        ):
            # don't nuke venv or models
            low = p.as_posix().lower()
            if any(x in low for x in ("/.venv", "\\.venv", "/env", "\\env", "/models", "\\models")):
                continue
            removed["thumbs"] += _rm(p)

        # Also clear a top-level /temp/ folder under the project root, if present
        temp_dir = base / "temp"
        if temp_dir.exists():
            removed["thumbs"] += _rm(temp_dir)

    if clean_qt_cache:
        # QtWebEngine cache dirs sometimes appear under user profile; ignore failures
        qt_dirs = []
        try:
            qt_dirs.append(Path.home() / "AppData/Local" / "QtProject")
            qt_dirs.append(Path.home() / "AppData/Local" / "framevision")
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
