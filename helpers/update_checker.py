
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QMessageBox, QCheckBox, QSpacerItem, QSizePolicy
)
from PySide6.QtGui import QDesktopServices
from PySide6.QtCore import QUrl

# ---------- App paths ----------
APP_ROOT = Path(__file__).resolve().parents[1]
INFO_DIR = APP_ROOT / "presets" / "info"
STATE_PATH = INFO_DIR / "update_state.json"

# ---------- Repo config ----------
GITHUB_OWNER = "Koongrizzly"
GITHUB_REPO  = "FrameVision"

# ---------- HTTP helpers (stdlib) ----------
import urllib.parse
import ssl
from urllib.request import Request, urlopen

# Prefer certifi CA bundle if available (helps embedded/portable Python builds on Windows)
try:
    import certifi  # type: ignore
except Exception:
    certifi = None

def _ssl_context():
    try:
        if certifi is not None:
            return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    try:
        ctx = ssl.create_default_context()
        # Some embedded Pythons don't load OS certs automatically.
        try:
            ctx.load_default_certs()
        except Exception:
            pass
        return ctx
    except Exception:
        return None

_SSL_CONTEXT = _ssl_context()

def _http_json(url: str, timeout: int = 25):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FrameVision-Updater"
    }
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as r:
        data = r.read()
    return json.loads(data.decode("utf-8"))

def _http_bytes(url: str, timeout: int = 25):
    headers = {"User-Agent": "FrameVision-Updater"}
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout, context=_SSL_CONTEXT) as r:
        return r.read()

# ---------- GitHub helpers ----------

def _get_default_branch(owner: str, repo: str) -> str:
    info = _http_json(f"https://api.github.com/repos/{owner}/{repo}")
    return info.get("default_branch", "main")

def _latest_release(owner: str, repo: str) -> tuple[Optional[str], Optional[str]]:
    """Return (tag_name, published_at_iso) or (None, None) if no releases."""
    try:
        data = _http_json(f"https://api.github.com/repos/{owner}/{repo}/releases/latest")
        return (data.get("tag_name"), data.get("published_at"))
    except Exception:
        return (None, None)

def _latest_commit_sha(owner: str, repo: str, branch: str) -> Optional[str]:
    """Return latest commit SHA on branch."""
    arr = _http_json(f"https://api.github.com/repos/{owner}/{repo}/commits?sha={branch}&per_page=1")
    if not arr:
        return None
    return arr[0].get("sha")

def _list_tree_paths(owner: str, repo: str, branch: str) -> list[str]:
    """List all files in the repo tree for a branch."""
    tree = _http_json(f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1")
    out = []
    for item in tree.get("tree", []):
        if item.get("type") == "blob":
            out.append(item.get("path"))
    return out

def _latest_commit_date_for_path(owner: str, repo: str, path: str, branch: str) -> Optional[str]:
    """Return ISO datetime of latest commit touching this path, or None."""
    q = urllib.parse.quote(path)
    url = f"https://api.github.com/repos/{owner}/{repo}/commits?path={q}&sha={branch}&per_page=1"
    arr = _http_json(url)
    if not arr:
        return None
    return arr[0]["commit"]["author"]["date"]  # ISO8601

def _download_raw(owner: str, repo: str, branch: str, path: str) -> bytes:
    q = urllib.parse.quote(path)
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{q}"
    return _http_bytes(url)

# ---------- State ----------

def _load_state() -> dict:
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_state(d: dict) -> None:
    INFO_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

def get_auto_check_enabled() -> bool:
    """Auto update checks were removed. Updates are manual only."""
    return False

def set_auto_check_enabled(enabled: bool) -> None:
    """No-op (kept for backward compatibility)."""
    return

def get_ack_release_tag() -> Optional[str]:
    return _load_state().get("release_ack_tag")

def ack_release(tag: Optional[str]) -> None:
    st = _load_state()
    st["release_ack_tag"] = tag
    _save_state(st)

# ---------- Beta check ----------

BETA_INCLUDE_SUFFIXES = (".py",)  # repo-wide
BETA_INCLUDE_PREFIXES = ("helpers/", "presets/viz/")  # include any file types here
BETA_EXCLUDE_TOP = {"tools", "models", "output", "presets", ".venv", ".git"}  # when scanning whole tree

@dataclass
class BetaFile:
    path: str
    remote_iso: Optional[str]
    local_mtime: Optional[float]

    @property
    def remote_dt(self) -> Optional[datetime]:
        if not self.remote_iso:
            return None
        try:
            return datetime.fromisoformat(self.remote_iso.replace("Z", "+00:00"))
        except Exception:
            return None

    @property
    def local_dt(self) -> Optional[datetime]:
        if self.local_mtime is None:
            return None
        try:
            return datetime.fromtimestamp(self.local_mtime)
        except Exception:
            return None

    def is_remote_newer(self) -> bool:
        if self.remote_dt is None:
            return False
        if self.local_dt is None:
            return True
        return self.remote_dt.timestamp() >= (self.local_dt.timestamp() + 2.0)

def _should_consider(path: str) -> bool:
    # Always consider the explicit prefixes
    for pref in BETA_INCLUDE_PREFIXES:
        if path.startswith(pref):
            return True
    # For others, consider only whitelisted suffixes and skip excluded top folders
    parts = path.split("/")
    if parts and parts[0] in BETA_EXCLUDE_TOP:
        return False
    for suf in BETA_INCLUDE_SUFFIXES:
        if path.endswith(suf):
            return True
    return False

def _collect_beta_candidates(owner: str, repo: str, branch: str) -> List[BetaFile]:
    """
    Repo-wide scan: list files that are newer on GitHub than locally according to last commit touching each path.
    """
    all_paths = _list_tree_paths(owner, repo, branch)
    targets = [p for p in all_paths if _should_consider(p)]
    candidates: List[BetaFile] = []
    for p in targets:
        try:
            iso = _latest_commit_date_for_path(owner, repo, p, branch)
        except Exception:
            iso = None
        local_path = APP_ROOT / p
        local_mtime = local_path.stat().st_mtime if local_path.exists() else None
        bf = BetaFile(path=p, remote_iso=iso, local_mtime=local_mtime)
        if bf.is_remote_newer():
            candidates.append(bf)
    return candidates

# ---------- Background workers (non-blocking UI) ----------

class _ProbeWorker(QThread):
    result = Signal(dict)

    def __init__(self, parent_ui, owner: str, repo: str):
        super().__init__(parent_ui)
        self._owner = owner
        self._repo = repo

    def run(self):
        out = {"release_tag": None, "release_published": None, "has_release": False,
               "beta_hint": False, "branch": None, "latest_sha": None}
        # Release
        try:
            rel_tag, rel_pub = _latest_release(self._owner, self._repo)
            out["release_tag"] = rel_tag
            out["release_published"] = rel_pub
            out["has_release"] = bool(rel_tag)
        except Exception:
            pass
        # Beta hint via latest branch SHA diff
        try:
            branch = _get_default_branch(self._owner, self._repo)
        except Exception:
            branch = "main"
        out["branch"] = branch
        try:
            latest_sha = _latest_commit_sha(self._owner, self._repo, branch)
        except Exception:
            latest_sha = None
        out["latest_sha"] = latest_sha
        st = _load_state()
        last_seen = st.get("last_seen_commit_sha")
        if latest_sha and latest_sha != last_seen:
            out["beta_hint"] = True
        self.result.emit(out)

class _BetaListWorker(QThread):
    result = Signal(list)
    error = Signal(str)

    def __init__(self, parent_ui, owner: str, repo: str, branch: str):
        super().__init__(parent_ui)
        self._owner = owner
        self._repo = repo
        self._branch = branch

    def run(self):
        try:
            items = _collect_beta_candidates(self._owner, self._repo, self._branch)
            self.result.emit(items)
        except Exception as e:
            self.error.emit(str(e))

# ---------- UI ----------

class BetaDialog(QDialog):
    def __init__(self, owner: str, repo: str, branch: str, latest_sha: Optional[str], parent=None):
        super().__init__(parent)
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.latest_sha = latest_sha
        self.setWindowTitle("FrameVision — Beta updater (repo-wide .py + helpers/ & presets/viz/)")
        self.resize(760, 480)

        self.info = QLabel("Choose which newer GitHub files to pull. "
                           "Files you deselect will be kept as-is.")
        self.info.setWordWrap(True)

        self.list = QListWidget(self)
        self.list.setSelectionMode(QListWidget.NoSelection)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_select_all = QPushButton("Select all")
        self.btn_select_none = QPushButton("Select none")
        self.btn_apply = QPushButton("Download selected")
        self.btn_close = QPushButton("Close")

        topbar = QHBoxLayout()
        topbar.addWidget(self.btn_refresh)
        topbar.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        topbar.addWidget(self.btn_select_all)
        topbar.addWidget(self.btn_select_none)
        topbar.addWidget(self.btn_apply)
        topbar.addWidget(self.btn_close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.info)
        layout.addLayout(topbar)
        layout.addWidget(self.list, 1)

        self.btn_refresh.clicked.connect(self._load)
        self.btn_select_all.clicked.connect(self._select_all)
        self.btn_select_none.clicked.connect(self._select_none)
        self.btn_apply.clicked.connect(self._apply)
        self.btn_close.clicked.connect(self.reject)

        self._load()

    def _load(self):
        self.list.clear()
        self.btn_apply.setEnabled(False)
        self.list.addItem(QListWidgetItem("Loading newer files from GitHub…"))
        try:
            branch = self.branch or _get_default_branch(self.owner, self.repo)
        except Exception:
            branch = "main"
        self._worker = _BetaListWorker(self, self.owner, self.repo, branch)
        self._worker.result.connect(self._fill_list)
        self._worker.error.connect(self._show_error)
        self._worker.start()

    def _fill_list(self, items):
        self.list.clear()
        if not items:
            self.list.addItem(QListWidgetItem("No newer files detected."))
            self.btn_apply.setEnabled(False)
            return
        self.btn_apply.setEnabled(True)
        for bf in items:
            remote_str = bf.remote_dt.isoformat(timespec="seconds") if bf.remote_dt else "unknown"
            local_str = bf.local_dt.isoformat(timespec="seconds") if bf.local_dt else "missing locally"
            text = f"{bf.path}\n  remote: {remote_str}   |   local: {local_str}"
            it = QListWidgetItem(text)
            it.setData(Qt.UserRole, bf.path)
            it.setCheckState(Qt.Checked if bf.is_remote_newer() else Qt.Unchecked)
            self.list.addItem(it)

    def _show_error(self, msg):
        self.list.clear()
        self.list.addItem(QListWidgetItem(f"Error loading beta files:\n{msg}"))
        self.btn_apply.setEnabled(False)

    def _select_all(self):
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Checked)

    def _select_none(self):
        for i in range(self.list.count()):
            it = self.list.item(i)
            it.setCheckState(Qt.Unchecked)

    def _apply(self):
        selected = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            path = it.data(Qt.UserRole)
            if path and it.checkState() == Qt.Checked:
                selected.append(path)
        if not selected:
            QMessageBox.information(self, "Nothing selected", "No files selected to download.")
            return
        try:
            branch = self.branch or _get_default_branch(self.owner, self.repo)
        except Exception:
            branch = "main"
        try:
            for p in selected:
                data = _download_raw(self.owner, self.repo, branch, p)
                out = APP_ROOT / p
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "wb") as f:
                    f.write(data)
            # Mark latest SHA as seen to avoid repeat popups
            if self.latest_sha:
                st = _load_state()
                st["last_seen_commit_sha"] = self.latest_sha
                _save_state(st)
            QMessageBox.information(self, "Beta update complete",
                                    f"Downloaded {len(selected)} file(s) from GitHub.\n"
                                    "You may need to restart FrameVision.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Beta update failed", f"{e}\n\n{traceback.format_exc()}")

# ---------- Startup check (non-blocking) ----------

def _today_str() -> str:
    return date.today().isoformat()

def _show_no_updates_toast(parent=None, text: str = " No new updates", msec: int = 5000):
    """Show a small non-modal popup that auto-closes after `msec` milliseconds."""
    try:
        box = QMessageBox(parent)
        box.setWindowTitle("Up to date")
        box.setIcon(QMessageBox.Information)
        box.setText(text)
        # No buttons; closes automatically.
        box.setStandardButtons(QMessageBox.NoButton)
        # Non-blocking / non-modal so it doesn't steal focus.
        try:
            box.setWindowModality(Qt.NonModal)
        except Exception:
            pass
        try:
            box.setAttribute(Qt.WA_ShowWithoutActivating, True)
        except Exception:
            pass
        try:
            # Make it a tool window so it floats above without taskbar button.
            box.setWindowFlag(Qt.Tool, True)
        except Exception:
            pass
        box.show()
        QTimer.singleShot(msec, box.close)
    except Exception:
        # Silently ignore toast errors to avoid breaking startup.
        pass


def _show_updates_popup(parent, result: dict):
    has_release = bool(result.get("has_release"))
    tag = result.get("release_tag")
    if has_release and tag and get_ack_release_tag() == tag:
        has_release = False

    beta_hint = bool(result.get("beta_hint"))

    if not (has_release or beta_hint):
        _show_no_updates_toast(parent)
        return

    lines = []
    if has_release:
        lines.append(f"New release available: {tag} (published {result.get('release_published')})")
    if beta_hint:
        lines.append("Beta: the repository has new changes on the default branch.")

    box = QMessageBox(parent)
    box.setWindowTitle("Updates available on GitHub")
    box.setIcon(QMessageBox.Information)
    box.setText("\n".join(lines))

    # "don't remind me about this release" checkbox
    if has_release and tag:
        cb = QCheckBox(f"I've installed {tag}. Don't remind me again.")
        box.setCheckBox(cb)

    btn_release = box.addButton("Open Release Updater…", QMessageBox.AcceptRole)
    btn_beta = box.addButton("Review Beta Files…", QMessageBox.ActionRole)
    box.addButton("Ignore", QMessageBox.RejectRole)
    box.exec()

    if has_release and tag and box.checkBox() and box.checkBox().isChecked():
        ack_release(tag)

    clicked = box.clickedButton()
    if clicked is btn_release:
        _open_release_updater(parent)
    elif clicked is btn_beta:
        dlg = BetaDialog(GITHUB_OWNER, GITHUB_REPO, result.get("branch") or "main", result.get("latest_sha"), parent)
        dlg.exec()

def check_now(parent=None, owner: str = GITHUB_OWNER, repo: str = GITHUB_REPO):
    """Manual check only. Probes GitHub in the background and shows a popup."""
    worker = _ProbeWorker(parent, owner, repo)
    if not hasattr(check_now, "_workers"):
        check_now._workers = []
    check_now._workers.append(worker)

    def _done(out):
        # Persist last result for debugging only; no scheduling/auto behavior.
        try:
            st2 = _load_state()
            st2["last_result"] = out
            _save_state(st2)
        except Exception:
            pass
        _show_updates_popup(parent, out)
        try:
            check_now._workers.remove(worker)
        except Exception:
            pass

    worker.result.connect(_done)
    worker.start()


def check_on_startup(parent=None, owner: str = GITHUB_OWNER, repo: str = GITHUB_REPO):
    """Auto update checks were removed (kept for compatibility)."""
    return

def force_check_now(parent=None, owner: str = GITHUB_OWNER, repo: str = GITHUB_REPO):
    """Manual check alias."""
    return check_now(parent, owner, repo)

# ---------- Release updater bridge / fallback ----------

def _open_release_updater(parent):
    """
    If the app has 'Info → Updates…' (menu_info.UpdateDialog), open that.
    Otherwise, open GitHub releases page as a fallback.
    """
    # Try to import from app module
    try:
        from helpers.menu_info import _open_update_dialog as _open_update_dialog_fn
    except Exception:
        _open_update_dialog_fn = None

    if _open_update_dialog_fn is not None:
        try:
            _open_update_dialog_fn(parent)
            return
        except Exception:
            pass

    QDesktopServices.openUrl(QUrl("https://github.com/Koongrizzly/FrameVision/releases"))

# ---------- Convenience installers ----------

def install_startup_check(main_window, delay_ms: int = 800):
    """Removed: automatic update checks on startup. Manual updates only."""
    return

def schedule_check_no_parent(delay_ms: int = 800):
    """Removed: automatic update checks on startup. Manual updates only."""
    return

# ---------- Menu helpers (optional) ----------

def open_beta_dialog(parent=None, owner: str = GITHUB_OWNER, repo: str = GITHUB_REPO):
    try:
        branch = _get_default_branch(owner, repo)
    except Exception:
        branch = "main"
    dlg = BetaDialog(owner, repo, branch, latest_sha=None, parent=parent)
    dlg.exec()

def add_updates_to_menu(info_menu, parent=None):
    """Optional helper to add manual update actions to an Info menu.
    Auto-update checks/toggles were removed.
    """
    if info_menu is None:
        return None, None, None
    act_check = info_menu.addAction("Check for Updates Now…")
    act_beta  = info_menu.addAction("Review Beta Files…")
    info_menu.addSeparator()

    act_check.triggered.connect(lambda: check_now(parent))
    act_beta.triggered.connect(lambda: open_beta_dialog(parent))

    return act_check, act_beta, None
