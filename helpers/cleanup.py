
from __future__ import annotations
import os, shutil, sys, tempfile

def _rm(path: str) -> int:
    try:
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
        return 0
    except Exception:
        return 1

def _iter_paths(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        # remove __pycache__ dirs
        for d in list(dirnames):
            if d == '__pycache__':
                yield os.path.join(dirpath, d)
        # remove .pyc files
        for f in filenames:
            if f.endswith('.pyc'):
                yield os.path.join(dirpath, f)

def clear_pyc_in_dirs(*dirs: str) -> int:
    rc = 0
    for d in dirs:
        if not d:
            continue
        if not os.path.isdir(d):
            continue
        for p in _iter_paths(d):
            rc |= _rm(p)
    return rc

def clear_temp_paths(*paths: str) -> int:
    rc = 0
    for p in paths:
        if not p:
            continue
        if os.path.exists(p):
            rc |= _rm(p)
    # Best-effort: also nuke OS temp children created by us
    app_tmp = os.path.join(tempfile.gettempdir(), 'framevision_tmp')
    if os.path.exists(app_tmp):
        rc |= _rm(app_tmp)
    return rc
