from __future__ import annotations
import os, shutil, sys, time, tempfile
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




def _clear_dir_contents(base_dir: Path) -> int:
    """Remove everything inside base_dir (keeps base_dir itself)."""
    removed = 0
    try:
        items = list(base_dir.rglob("*"))
    except Exception:
        return 0

    # delete deepest paths first so directories can be removed safely
    items.sort(key=lambda p: len(p.parts), reverse=True)
    for p in items:
        removed += _rm(p)

    # prune empty dirs (best-effort)
    try:
        dirs = [p for p in base_dir.rglob("*") if p.is_dir()]
        dirs.sort(key=lambda p: len(p.parts), reverse=True)
        for d in dirs:
            try:
                if not any(d.iterdir()):
                    d.rmdir()
            except Exception:
                pass
    except Exception:
        pass

    return removed

def _clear_dir_older_than(base_dir: Path, cutoff_ts: float) -> int:
    """Remove everything inside base_dir that is older than cutoff_ts (keeps base_dir itself)."""
    removed = 0
    try:
        items = list(base_dir.rglob("*"))
    except Exception:
        return 0

    # delete deepest paths first so directories can be removed safely
    items.sort(key=lambda p: len(p.parts), reverse=True)
    for p in items:
        if _is_older_than(p, cutoff_ts):
            removed += _rm(p)

    # prune empty dirs (best-effort)
    try:
        dirs = [p for p in base_dir.rglob("*") if p.is_dir()]
        dirs.sort(key=lambda p: len(p.parts), reverse=True)
        for d in dirs:
            try:
                if not any(d.iterdir()):
                    d.rmdir()
            except Exception:
                pass
    except Exception:
        pass

    return removed


def run_cleanup(
    project_root: str | None = None,
    *,
    clean_pyc: bool = True,
    clean_logs: bool = True,
    clean_thumbs: bool = True,
    clean_qt_cache: bool = True,
    clean_hf_cache: bool = False,
    clean_temp: bool = False,
) -> dict:
    """
    Returns dict with counts of removed items by category.
    Safe defaults: remove __pycache__ and *.pyc/*.pyo and simple log/thumbnail caches under the project.
    """
    base = Path(project_root or os.getcwd()).resolve()
    removed = {"pycache": 0, "pyc_pyo": 0, "logs": 0, "thumbs": 0, "qt": 0, "hf": 0, "temp": 0}

    if clean_pyc:
        # Do not touch bytecode inside our virtual envs / extra env presets
        exclude_roots = [
            (base / ".venv"),
            (base / ".ace_env"),
            (base / "models"),
            (base / ".comfy_env"),
            (base / ".zimage_env"),
            (base / ".wan_env"),
            (base / ".hunyuan15_env"),
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

        # Also clear items older than 24 hours inside a top-level /temp/ folder under the project root, if present
        temp_dir = base / "temp"
        if temp_dir.exists() and temp_dir.is_dir():
            cutoff = time.time() - 24 * 60 * 60
            removed["thumbs"] += _clear_dir_older_than(temp_dir, cutoff)

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

    if clean_temp:
        # Project temp/work folders (keeps folder itself)
        for d in (base / "output" / "_temp", base / "work"):
            if d.exists() and d.is_dir():
                removed["temp"] += _clear_dir_contents(d)

        # Best-effort: also nuke OS temp children created by us
        try:
            app_tmp = Path(tempfile.gettempdir()) / "framevision_tmp"
            if app_tmp.exists():
                removed["temp"] += _rm(app_tmp)
        except Exception:
            pass

    return removed

if __name__ == "__main__":
    import argparse, json

    ap = argparse.ArgumentParser(description="FrameVision cache cleanup helper")
    ap.add_argument("--project-root", default=None, help="Project root to clean (defaults to CWD)")
    ap.add_argument("--no-pyc", action="store_true", help="Do not remove __pycache__ / *.pyc / *.pyo")
    ap.add_argument("--no-logs", action="store_true", help="Do not remove old logs")
    ap.add_argument("--no-thumbs", action="store_true", help="Do not remove thumbnail/cache/temp folders")
    ap.add_argument("--no-qt", action="store_true", help="Do not remove Qt cache dirs under the user profile")
    ap.add_argument("--clean-hf-cache", action="store_true", help="Also remove HuggingFace cache (can be large)")
    ap.add_argument("--clean-temp", action="store_true", help="Also remove project temp folders (output/_temp, work) and FrameVision temp in OS temp")
    ap.add_argument("--json", action="store_true", help="Print result as JSON")

    args = ap.parse_args()
    result = run_cleanup(
        args.project_root,
        clean_pyc=not args.no_pyc,
        clean_logs=not args.no_logs,
        clean_thumbs=not args.no_thumbs,
        clean_qt_cache=not args.no_qt,
        clean_hf_cache=args.clean_hf_cache,
        clean_temp=args.clean_temp,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)

