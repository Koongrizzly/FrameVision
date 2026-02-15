"""remove_hide.py

FrameVision helper: UI tool to remove (delete) model packs/environments and manage hide/unhide flags.

Design goals:
- Safe to iterate: lives separate from Optional Installs.
- Works standalone: `python helpers/remove_hide.py`.
- Uses a registry (single source of truth) for paths + shared-env rules.

Notes:
- Paths in the registry are *app-root relative* (they start with '/'). They are NOT treated as OS root.
  Example: '/models/wan22/' resolves to '<app_root>/models/wan22/'.
- Shared env groups:
  - Qwen 2511 & Qwen 2512 share '/.qwen2512'
  - Z-image Turbo GGUF & FP16 share '/.zimage_env'

This file does NOT yet enforce hiding inside the rest of the app. It only stores hide flags
and provides UI to toggle them (plus a global "Show hidden" toggle for this manager UI).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from PySide6 import QtCore, QtGui, QtWidgets


# -------------------------
# Registry
# -------------------------

@dataclass
class Entry:
    id: str
    title: str
    env_paths: List[str] = field(default_factory=list)
    model_paths: List[str] = field(default_factory=list)
    repo_paths: List[str] = field(default_factory=list)
    helper_paths: List[str] = field(default_factory=list)
    # Some packs are auto-hidden/always-present and shouldn't expose hide/unhide.
    can_hide: bool = True
    shared_env_group: Optional[str] = None
    shared_env_path: Optional[str] = None  # app-relative
    shared_helper_note: Optional[str] = None


def build_registry() -> List[Entry]:
    return [
        Entry(
            id="wan22",
            title="WAN 2.2",
            env_paths=["/.wan_venv/"],
            model_paths=["/models/wan22/"],
            repo_paths=[],
            helper_paths=["/helpers/wan22.py"],
        ),
        Entry(
            id="hunyuan15",
            title="HunyuanVideo 1.5",
            env_paths=["/.hunyuan15_env/"],
            model_paths=["/models/hunyuanvideo-community_HunyuanVideo*.*"],
            helper_paths=["/helpers/hunyuan15.py"],
        ),

Entry(
    id="ace_step_15",
    title="Ace Step 1.5 Music creation",
    env_paths=["/environments/.ace_15/"],
    model_paths=["/models/ace_step_15/"],
    repo_paths=["/models/ace_step_15/repo/ACE-Step-1.5/"],
    helper_paths=[],
),

        Entry(
            id="sdxl_txt2img",
            title="SDXL Text-to-Image",
            model_paths=["/models/sdxl/"],
            helper_paths=["/helpers/txt2img"],
            shared_helper_note="Shared helper with other models , hiding all will also hide the full tab in the app , hiding all will also hide the full tab in the app .",
        ),
        Entry(
            id="sdxl_inpaint",
            title="SDXL Inpaint",
            env_paths=["/.sdxl_inpaint/"],
            model_paths=["/models/inpaint/"],
            helper_paths=["/helpers/sdxl_inpaint.py"],
        ),
        Entry(
            id="qwen_edit_2511",
            title="Qwen Edit 2511",
            env_paths=["/.qwen2512"],
            model_paths=["/models/qwen2511gguf/"],
            helper_paths=["/helpers/qwen2511.py"],
            shared_env_group="qwen2512_shared",
            shared_env_path="/.qwen2512",
        ),
        Entry(
            id="qwen_image_2512",
            title="Qwen Image 2512",
            env_paths=["/.qwen2512"],
            model_paths=["/models/Qwen-Image-2512 GGUF/"],
            helper_paths=["/helpers/qwen2512.py"],
            shared_env_group="qwen2512_shared",
            shared_env_path="/.qwen2512",
        ),
        Entry(
            id="zimage_turbo_gguf",
            title="Z-image Turbo (GGUF)",
            env_paths=["/.zimage_env"],
            model_paths=["/models/Z-Image-Turbo GGUF/"],
            helper_paths=["/helpers/txt2img.py"],
            shared_env_group="zimage_shared",
            shared_env_path="/.zimage_env",
            shared_helper_note="Shared helper with other models , hiding all will also hide the full tab in the app .",
        ),
        Entry(
            id="zimage_turbo_fp16",
            title="Z-image Turbo (FP16)",
            env_paths=["/.zimage_env"],
            model_paths=["/models/Z-Image-Turbo/"],
            helper_paths=["/helpers/txt2img.py"],
            shared_env_group="zimage_shared",
            shared_env_path="/.zimage_env",
            shared_helper_note="Shared helper with other models , hiding all will also hide the full tab in the app .",
        ),

        Entry(
            id="qwentts",
            title="Qwen3 TTS",
            env_paths=["/environments/.qwen3tts/"],
            model_paths=["/models/Qwen3-TTS*"],
            helper_paths=["/helpers/qwentts_ui.py"],
        ),
        Entry(
            id="whisper",
            title="Whisper Lab",
            env_paths=["/environments/.whisper/"],
            model_paths=["/models/faster_whisper/"],
            helper_paths=["/helpers/whisper.py"],
        ),
        Entry(
            id="gfpgan",
            title="GFPGAN (face upscale)",
            # GFPGAN stores its Python env inside the model folder.
            # Keeping these split avoids double-counting size-on-disk and makes status clearer.
            env_paths=["/models/gfpgan/.GFPGAN/"],
            model_paths=["/models/gfpgan/"],
            helper_paths=[],
            can_hide=False,
        ),
    ]


# -------------------------
# State (hide flags)
# -------------------------

DEFAULT_STATE = {
    "hidden_ids": [],
    "show_hidden_in_manager": False,
}


def _safe_mkdir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def get_state_path(app_name: str = "FrameVision") -> str:
    """Where to store state.

    Priority:
    1) FRAMEVISION_STATE_DIR env var
    2) QStandardPaths AppDataLocation
    3) CWD
    """
    env_dir = os.environ.get("FRAMEVISION_STATE_DIR", "").strip()
    if env_dir:
        _safe_mkdir(env_dir)
        return os.path.join(env_dir, "remove_hide_state.json")

    base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.AppDataLocation)
    if base:
        _safe_mkdir(base)
        return os.path.join(base, "remove_hide_state.json")

    return os.path.abspath("remove_hide_state.json")


def load_state(state_path: str) -> Dict:
    if not os.path.exists(state_path):
        return dict(DEFAULT_STATE)
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(DEFAULT_STATE)
        out = dict(DEFAULT_STATE)
        out.update(data)
        if not isinstance(out.get("hidden_ids"), list):
            out["hidden_ids"] = []
        return out
    except Exception:
        return dict(DEFAULT_STATE)


def save_state(state_path: str, state: Dict) -> None:
    try:
        _safe_mkdir(os.path.dirname(state_path) or ".")
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


# -------------------------
# Path resolution + checks
# -------------------------


def resolve_app_path(app_root: str, p: str) -> str:
    """Resolve an app-relative path spec to an absolute filesystem path."""
    if not p:
        return p
    # Treat leading slashes as app-root relative (not OS root)
    if p.startswith("/") or p.startswith("\\"):
        p = p.lstrip("/\\")
    return os.path.normpath(os.path.join(app_root, p))


def expand_specs(app_root: str, spec: str) -> List[str]:
    """Expand globs and return matching absolute paths."""
    abs_spec = resolve_app_path(app_root, spec)
    matches = glob.glob(abs_spec)
    if matches:
        return sorted(set(matches))
    # If no glob matches, return the literal path
    return [abs_spec]


def path_exists_any(paths: List[str]) -> bool:
    return any(os.path.exists(p) for p in paths)


def entry_paths_resolved(app_root: str, e: Entry) -> Dict[str, List[str]]:
    envs = []
    for s in e.env_paths:
        envs.extend(expand_specs(app_root, s))
    models = []
    for s in e.model_paths:
        models.extend(expand_specs(app_root, s))
    repos = []
    for s in e.repo_paths:
        repos.extend(expand_specs(app_root, s))
    helpers = [resolve_app_path(app_root, s) for s in e.helper_paths]
    shared_env = [resolve_app_path(app_root, e.shared_env_path)] if e.shared_env_path else []

    # Deduplicate while preserving order
    def dedup(lst: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in lst:
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    return {
        "envs": dedup(envs),
        "models": dedup(models),
        "repos": dedup(repos),
        "helpers": dedup(helpers),
        "shared_env": dedup(shared_env),
    }


# -------------------------
# Size on disk helpers
# -------------------------


def _format_bytes(num_bytes: int) -> str:
    try:
        n = float(max(0, int(num_bytes)))
    except Exception:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    i = 0
    while n >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)} {units[i]}"
    return f"{n:.1f} {units[i]}"


def _path_size_bytes(p: str) -> int:
    """Return total size on disk for a file/folder path.

    - Includes subfolders.
    - Skips symlinks.
    - Ignores unreadable files.
    """
    if not p or not os.path.exists(p):
        return 0

    # File
    try:
        if os.path.isfile(p) and not os.path.islink(p):
            return int(os.path.getsize(p))
    except Exception:
        return 0

    # Directory
    total = 0

    def walk_dir(root: str) -> None:
        nonlocal total
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            walk_dir(entry.path)
                        else:
                            try:
                                total += int(entry.stat(follow_symlinks=False).st_size)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        if os.path.isdir(p) and not os.path.islink(p):
            walk_dir(p)
    except Exception:
        pass

    return int(total)


def _norm_path(p: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(os.path.normpath(p)))
    except Exception:
        return os.path.normcase(os.path.normpath(p or ""))


def _is_within(child: str, parent: str) -> bool:
    """True if child is parent or inside parent."""
    c = _norm_path(child)
    p = _norm_path(parent)
    if not c or not p:
        return False
    if c == p:
        return True
    return c.startswith(p + os.sep)


def _path_size_bytes_excluding(base_path: str, exclude_paths: List[str]) -> int:
    """Like _path_size_bytes, but skips any subfolders/files under exclude_paths.

    Important: Only excludes paths that are *inside* base_path (never excludes a parent).
    """
    if not base_path or not os.path.exists(base_path):
        return 0

    base_n = _norm_path(base_path)
    excludes = []
    for ep in exclude_paths or []:
        if not ep:
            continue
        en = _norm_path(ep)
        # Only exclude descendants of the base path (not parents).
        if en != base_n and en.startswith(base_n + os.sep):
            excludes.append(en)

    if not excludes:
        return _path_size_bytes(base_path)

    # File
    try:
        if os.path.isfile(base_path) and not os.path.islink(base_path):
            # Only excluded if the file itself is under an excluded prefix
            bn = base_n
            if any(bn == ex or bn.startswith(ex + os.sep) for ex in excludes):
                return 0
            return int(os.path.getsize(base_path))
    except Exception:
        return 0

    # Directory
    total = 0

    def is_excluded(path: str) -> bool:
        pn = _norm_path(path)
        return any(pn == ex or pn.startswith(ex + os.sep) for ex in excludes)

    def walk_dir(root: str) -> None:
        nonlocal total
        if is_excluded(root):
            return
        try:
            with os.scandir(root) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if not is_excluded(entry.path):
                                walk_dir(entry.path)
                        else:
                            if is_excluded(entry.path):
                                continue
                            try:
                                total += int(entry.stat(follow_symlinks=False).st_size)
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass

    try:
        if os.path.isdir(base_path) and not os.path.islink(base_path):
            walk_dir(base_path)
    except Exception:
        pass

    return int(total)



def installed_status(app_root: str, e: Entry) -> Tuple[bool, bool, bool]:
    """Return (env_ok, models_ok, repos_ok)."""
    r = entry_paths_resolved(app_root, e)
    env_ok = path_exists_any(r["envs"]) if r["envs"] else False
    models_ok = path_exists_any(r["models"]) if r["models"] else False
    repos_ok = path_exists_any(r["repos"]) if r["repos"] else False
    return env_ok, models_ok, repos_ok


def is_installed(app_root: str, e: Entry) -> bool:
    env_ok, models_ok, repos_ok = installed_status(app_root, e)
    return env_ok or models_ok or repos_ok


def siblings_in_group(registry: List[Entry], group_id: str, exclude_id: str) -> List[Entry]:
    return [x for x in registry if x.shared_env_group == group_id and x.id != exclude_id]


# -------------------------
# Delete operations
# -------------------------


def remove_path(p: str) -> Tuple[bool, str]:
    """Attempt to delete file/folder at p."""
    if not os.path.exists(p):
        return True, f"Already missing: {p}"
    try:
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=False)
        else:
            os.remove(p)
        return True, f"Deleted: {p}"
    except Exception as ex:
        return False, f"Failed: {p} ({ex})"


def remove_entry(app_root: str, registry: List[Entry], e: Entry) -> List[Tuple[bool, str]]:
    """Delete models/repos/envs for an entry. Honors shared-env groups."""
    results: List[Tuple[bool, str]] = []
    r = entry_paths_resolved(app_root, e)

    # 1) Remove models and repos (always safe)
    for p in r["models"]:
        ok, msg = remove_path(p)
        results.append((ok, msg))
    for p in r["repos"]:
        ok, msg = remove_path(p)
        results.append((ok, msg))

    # 2) Remove envs with shared rules
    # Some entries list env_paths directly; if shared group exists, only remove shared env when no sibling remains installed.
    if e.shared_env_group and e.shared_env_path:
        sibs = siblings_in_group(registry, e.shared_env_group, e.id)
        sib_installed = any(is_installed(app_root, s) for s in sibs)
        shared_env_abs = resolve_app_path(app_root, e.shared_env_path)
        if sib_installed:
            results.append((True, f"Skipped shared env (still needed): {shared_env_abs}"))
        else:
            ok, msg = remove_path(shared_env_abs)
            results.append((ok, msg))

        # If env_paths include the same shared env, skip individual deletions to avoid duplicates
        # (Still okay if we try again; but keep output tidy.)
        env_abs_set = set([shared_env_abs])
        for p in r["envs"]:
            if p in env_abs_set:
                continue
            ok, msg = remove_path(p)
            results.append((ok, msg))
    else:
        for p in r["envs"]:
            ok, msg = remove_path(p)
            results.append((ok, msg))

    return results


# -------------------------
# UI components
# -------------------------


class _SizeSignals(QtCore.QObject):
    done = QtCore.Signal(str)


class _SizeOnDiskJob(QtCore.QRunnable):
    """Background job to compute env+model size for an Entry."""

    def __init__(self, app_root: str, entry: Entry):
        super().__init__()
        self.app_root = app_root
        self.entry = entry
        self.signals = _SizeSignals()

    def run(self) -> None:
        try:
            r = entry_paths_resolved(self.app_root, self.entry)
            env_paths = [p for p in r.get("envs", []) if os.path.exists(p)]
            model_paths = [p for p in r.get("models", []) if os.path.exists(p)]

            # Avoid double counting when envs live inside model folders (e.g., GFPGAN).
            env_bytes = sum(_path_size_bytes_excluding(p, model_paths) for p in env_paths)
            model_bytes = sum(_path_size_bytes_excluding(p, env_paths) for p in model_paths)

            # True union size for Total (dedupe nested roots across env+models)
            all_roots = []
            for p in (env_paths + model_paths):
                if p and os.path.exists(p):
                    all_roots.append(p)
            # Deduplicate and remove roots that are inside another root.
            uniq = []
            seen = set()
            for p in all_roots:
                pn = _norm_path(p)
                if pn not in seen:
                    uniq.append(p)
                    seen.add(pn)
            uniq.sort(key=lambda x: len(_norm_path(x)))
            kept: List[str] = []
            for p in uniq:
                if any(_is_within(p, k) for k in kept):
                    continue
                kept.append(p)
            total = sum(_path_size_bytes(p) for p in kept)

            # Human-friendly summary. Keep it short (UI width varies).
            parts = []
            parts.append(f"Env: {_format_bytes(env_bytes)}" if env_paths else "Env: —")
            parts.append(f"Models: {_format_bytes(model_bytes)}" if model_paths else "Models: —")
            parts.append(f"Total: {_format_bytes(total)}")
            self.signals.done.emit(" | ".join(parts))
        except Exception:
            # Fail silently but show something useful.
            self.signals.done.emit("Size check failed")

class EntryCard(QtWidgets.QFrame):
    def __init__(self, parent: QtWidgets.QWidget, app_root: str, registry: List[Entry], entry: Entry, state: Dict, state_path: str):
        super().__init__(parent)
        self.app_root = app_root
        self.registry = registry
        self.entry = entry
        self.state = state
        self.state_path = state_path

        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setFrameShadow(QtWidgets.QFrame.Raised)

        self.title_lbl = QtWidgets.QLabel(entry.title)
        f = self.title_lbl.font()
        f.setPointSize(max(10, f.pointSize() + 1))
        f.setBold(True)
        self.title_lbl.setFont(f)

        self.badge = QtWidgets.QLabel("")
        self.badge.setAlignment(QtCore.Qt.AlignCenter)
        self.badge.setMinimumWidth(120)
        self.badge.setStyleSheet("QLabel { padding: 4px 8px; border-radius: 8px; }")

        self.paths_lbl = QtWidgets.QLabel("")
        self.paths_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.paths_lbl.setWordWrap(True)

        self.btn_remove = QtWidgets.QPushButton("Delete")
        self.btn_size = QtWidgets.QPushButton("Size on disk")
        self.size_lbl = QtWidgets.QLabel("")
        self.size_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.size_lbl.setMinimumWidth(240)
        self.btn_hide = QtWidgets.QPushButton("Hide")
        self.btn_unhide = QtWidgets.QPushButton("Unhide")

        self.btn_remove.clicked.connect(self.on_remove)
        self.btn_size.clicked.connect(self.on_size)
        self.btn_hide.clicked.connect(self.on_hide)
        self.btn_unhide.clicked.connect(self.on_unhide)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.btn_remove)
        btns.addWidget(self.btn_size)
        btns.addWidget(self.size_lbl)
        btns.addStretch(1)
        btns.addWidget(self.btn_hide)
        btns.addWidget(self.btn_unhide)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.title_lbl)
        top.addStretch(1)
        top.addWidget(self.badge)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.paths_lbl)
        lay.addLayout(btns)

        self.refresh()

    def on_size(self) -> None:
        # Lazy: only compute when clicked.
        self.btn_size.setEnabled(False)
        self.btn_size.setText("working...")
        self.size_lbl.setText("")

        job = _SizeOnDiskJob(self.app_root, self.entry)

        def _done(text: str) -> None:
            self.size_lbl.setText(text)
            self.btn_size.setText("Size on disk")
            self.btn_size.setEnabled(True)

        job.signals.done.connect(_done)
        QtCore.QThreadPool.globalInstance().start(job)

    def is_hidden(self) -> bool:
        if not getattr(self.entry, "can_hide", True):
            return False
        return self.entry.id in set(self.state.get("hidden_ids", []))

    def set_hidden(self, hidden: bool) -> None:
        if not getattr(self.entry, "can_hide", True):
            return
        hidden_ids = set(self.state.get("hidden_ids", []))
        if hidden:
            hidden_ids.add(self.entry.id)
        else:
            hidden_ids.discard(self.entry.id)
        self.state["hidden_ids"] = sorted(hidden_ids)
        save_state(self.state_path, self.state)

    def refresh(self) -> None:
        env_ok, models_ok, repos_ok = installed_status(self.app_root, self.entry)
        hidden = self.is_hidden()

        status_parts = []
        if env_ok:
            status_parts.append("Env")
        if models_ok:
            status_parts.append("Models")
        if repos_ok:
            status_parts.append("Repo")

        installed = bool(status_parts)
        if installed:
            base = "Installed (" + ", ".join(status_parts) + ")"
        else:
            base = "Not installed"
        if hidden:
            base += " • Hidden"

        # Simple color badges (avoid hardcoding theme colors; just use neutral styles)
        if installed:
            self.badge.setStyleSheet("QLabel { padding: 4px 8px; border-radius: 8px; border: 1px solid palette(mid); }")
        else:
            self.badge.setStyleSheet("QLabel { padding: 4px 8px; border-radius: 8px; border: 1px dashed palette(mid); }")
        self.badge.setText(base)

        r = entry_paths_resolved(self.app_root, self.entry)

        def fmt_list(label: str, items: List[str]) -> str:
            if not items:
                return ""
            return label + "\n" + "\n".join(["  - " + x for x in items])

        blocks = []
        if self.entry.env_paths:
            blocks.append(fmt_list("Env paths:", r["envs"]))
        if self.entry.model_paths:
            blocks.append(fmt_list("Model paths:", r["models"]))
        if self.entry.repo_paths:
            blocks.append(fmt_list("Repo paths:", r["repos"]))
        if self.entry.helper_paths:
            blocks.append(fmt_list("Helper refs:", r["helpers"]))
        if self.entry.shared_env_group and self.entry.shared_env_path:
            blocks.append(f"Shared env group: {self.entry.shared_env_group}\n  - {resolve_app_path(self.app_root, self.entry.shared_env_path)}")
        if self.entry.shared_helper_note:
            blocks.append(f"Note: {self.entry.shared_helper_note}")

        self.paths_lbl.setText("\n\n".join([b for b in blocks if b.strip()]))

        # Button states
        can_hide = bool(getattr(self.entry, "can_hide", True))
        if not can_hide:
            self.btn_hide.setVisible(False)
            self.btn_unhide.setVisible(False)
        else:
            self.btn_unhide.setVisible(hidden)
            self.btn_hide.setVisible(not hidden)


    def _request_refresh_all(self) -> None:

        """Ask the owning pane to rebuild cards, without assuming a fixed parent() chain."""

        w = self.parentWidget()

        # Walk up the parent chain until we find something that implements refresh_all()

        while w is not None:

            try:

                fn = getattr(w, "refresh_all", None)

                if callable(fn):

                    fn()

                    return

            except Exception:

                pass

            try:

                w = w.parentWidget()

            except Exception:

                break



    def on_hide(self) -> None:
        self.set_hidden(True)
        self.refresh()
        self._request_refresh_all()

    def on_unhide(self) -> None:
        self.set_hidden(False)
        self.refresh()
        self._request_refresh_all()

    def on_remove(self) -> None:
        # Confirmation
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setWindowTitle("Delete model/env")
        msg.setText(f"Delete files for: {self.entry.title}?")
        msg.setInformativeText("This will delete model folders and (if allowed) the environment folder(s).")
        msg.setStandardButtons(QtWidgets.QMessageBox.Cancel | QtWidgets.QMessageBox.Ok)
        msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        ret = msg.exec()
        if ret != QtWidgets.QMessageBox.Ok:
            return

        results = remove_entry(self.app_root, self.registry, self.entry)

        # Show result dialog
        ok_all = all(ok for ok, _ in results)
        out = "\n".join([("✓ " if ok else "✗ ") + text for ok, text in results])

        res = QtWidgets.QMessageBox(self)
        res.setWindowTitle("Delete results")
        res.setIcon(QtWidgets.QMessageBox.Information if ok_all else QtWidgets.QMessageBox.Warning)
        res.setText("Done." if ok_all else "Completed with some errors.")
        res.setDetailedText(out)
        res.setStandardButtons(QtWidgets.QMessageBox.Ok)
        res.exec()

        self.refresh()
        self._request_refresh_all()


class RemoveHidePane(QtWidgets.QWidget):
    def __init__(self, app_root: str, state_path: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.app_root = app_root
        self.registry = build_registry()
        self.state_path = state_path
        self.state = load_state(state_path)

        # Drop any hide flags for entries that can't be hidden (keeps UI sane if the state file is old).
        try:
            non_hideable = {e.id for e in self.registry if not getattr(e, "can_hide", True)}
            hidden_ids = set(self.state.get("hidden_ids", []))
            cleaned = sorted([x for x in hidden_ids if x not in non_hideable])
            if cleaned != self.state.get("hidden_ids", []):
                self.state["hidden_ids"] = cleaned
                save_state(self.state_path, self.state)
        except Exception:
            pass

        # Header info
        self.header_lbl = QtWidgets.QLabel(
            "<b>FrameVision Optional installs Remove &amp; Hide</b><br>"
            "_________________________________________________<br>"
            "Hide and /or delete what you don't need.<br>"
            "Click 'Restart App' after hiding/unhiding to apply changes."
        )
        self.header_lbl.setTextFormat(QtCore.Qt.RichText)
        self.header_lbl.setWordWrap(True)

        # Controls
        self.chk_show_hidden = QtWidgets.QCheckBox("Show hidden")
        self.chk_show_hidden.setChecked(bool(self.state.get("show_hidden_in_manager", False)))
        self.chk_show_hidden.stateChanged.connect(self.on_show_hidden_changed)

        self.btn_restart = QtWidgets.QPushButton("Restart App")
        self.btn_restart.clicked.connect(self.on_restart_app)

        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_all)

        self.info_lbl = QtWidgets.QLabel("")
        self.info_lbl.setWordWrap(True)

        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(self.chk_show_hidden)
        ctrl.addStretch(1)
        ctrl.addWidget(self.btn_restart)
        ctrl.addWidget(self.btn_refresh)

        # Scroll area with cards
        self.cards_container = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(10)
        self.cards_layout.addStretch(1)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.cards_container)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.header_lbl)
        lay.addLayout(ctrl)
        lay.addWidget(self.info_lbl)
        lay.addWidget(self.scroll)

        self.refresh_all()

    def on_show_hidden_changed(self) -> None:
        self.state["show_hidden_in_manager"] = bool(self.chk_show_hidden.isChecked())
        save_state(self.state_path, self.state)
        self.refresh_all()

    def on_restart_app(self) -> None:
        """Restart the whole app/process so hide/unhide changes apply."""
        try:
            cwd = os.getcwd()
            # Frozen exe: sys.executable is the app itself, argv[1:] are args.
            if getattr(sys, "frozen", False):
                program = sys.executable
                args = sys.argv[1:]
            else:
                # Python script: restart via interpreter + current argv (includes script path).
                program = sys.executable
                args = sys.argv

            ok = QtCore.QProcess.startDetached(program, args, cwd)
            if not ok:
                raise RuntimeError("startDetached returned False")

            QtCore.QCoreApplication.quit()
            # Belt-and-suspenders: ensure full exit even if embedded.
            QtCore.QTimer.singleShot(50, lambda: os._exit(0))
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Restart failed", f"Could not restart the app.\n\n{ex}")

    def refresh_all(self) -> None:
        # Clear cards
        while self.cards_layout.count() > 0:
            item = self.cards_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        show_hidden = bool(self.chk_show_hidden.isChecked())
        hidden_ids = set(self.state.get("hidden_ids", []))

        cards = []
        installed_count = 0
        hidden_count = 0

        for e in self.registry:
            can_hide = bool(getattr(e, "can_hide", True))
            is_hidden = can_hide and (e.id in hidden_ids)
            if is_hidden:
                hidden_count += 1
                if not show_hidden:
                    continue

            if is_installed(self.app_root, e):
                installed_count += 1

            card = EntryCard(self.cards_container, self.app_root, self.registry, e, self.state, self.state_path)
            cards.append(card)
            self.cards_layout.addWidget(card)

        self.cards_layout.addStretch(1)

        self.info_lbl.setText(
            f"App root: {self.app_root}\n"
            f"Installed entries detected: {installed_count} / {len(self.registry)}\n"
            f"Hidden entries: {hidden_count} (toggle 'Show hidden' to reveal)"
        )


class StandaloneWindow(QtWidgets.QMainWindow):
    def __init__(self, app_root: str, state_path: str):
        super().__init__()
        self.setWindowTitle("Remove / Hide (Model Manager) — Standalone")
        self.resize(1000, 750)

        pane = RemoveHidePane(app_root=app_root, state_path=state_path)
        self.setCentralWidget(pane)


# -------------------------
# Entry points
# -------------------------


def guess_app_root() -> str:
    """Guess app root as parent of the folder that contains this script (helpers/...)."""
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir))


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="FrameVision remove/hide manager")
    parser.add_argument("--root", default=os.environ.get("FRAMEVISION_ROOT", ""), help="FrameVision app root (defaults to parent of helpers)")
    args = parser.parse_args(argv)

    app_root = args.root.strip() or guess_app_root()
    app_root = os.path.abspath(app_root)

    # Set organization/application for QStandardPaths consistency
    QtCore.QCoreApplication.setOrganizationName("FrameVision")
    QtCore.QCoreApplication.setApplicationName("FrameVision")

    app = QtWidgets.QApplication(sys.argv)
    state_path = get_state_path("FrameVision")

    win = StandaloneWindow(app_root=app_root, state_path=state_path)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
