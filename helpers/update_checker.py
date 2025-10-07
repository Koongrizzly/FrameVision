
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QMessageBox, QCheckBox, QSpacerItem, QSizePolicy
)

# ---------- App paths ----------
APP_ROOT = Path(__file__).resolve().parents[1]
INFO_DIR = APP_ROOT / "presets" / "info"
STATE_PATH = INFO_DIR / "update_state.json"

# ---------- Repo config (override if needed) ----------
GITHUB_OWNER = "Koongrizzly"
GITHUB_REPO = "FrameVision"

# ---------- HTTP helpers (stdlib only) ----------
import urllib.parse
from urllib.request import Request, urlopen

def _http_json(url: str, timeout: int = 25):
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FrameVision-Updater"
    }
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as r:
        data = r.read()
    return json.loads(data.decode("utf-8"))

def _http_bytes(url: str, timeout: int = 25):
    headers = {"User-Agent": "FrameVision-Updater"}
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as r:
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

# ---------- Beta check ----------

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
        # consider remote newer if >= local + 2 seconds tolerance
        return self.remote_dt.timestamp() >= (self.local_dt.timestamp() + 2.0)


def _collect_beta_candidates(owner: str, repo: str, branch: str) -> List[BetaFile]:
    """
    Find files in helpers/ and presets/viz/ that are newer on GitHub than locally.
    """
    try:
        all_paths = _list_tree_paths(owner, repo, branch)
    except Exception as e:
        raise RuntimeError(f"Failed to list repo tree: {e}")

    targets = [p for p in all_paths if p.startswith("helpers/") or p.startswith("presets/viz/")]
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

# ---------- UI ----------

class BetaDialog(QDialog):
    def __init__(self, owner: str, repo: str, branch: str, parent=None):
        super().__init__(parent)
        self.owner = owner
        self.repo = repo
        self.branch = branch
        self.setWindowTitle("FrameVision — Beta updater (helpers/ & presets/viz/)")
        self.resize(720, 440)

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
        try:
            branch = self.branch or _get_default_branch(self.owner, self.repo)
            items = _collect_beta_candidates(self.owner, self.repo, branch)
            if not items:
                self.list.addItem(QListWidgetItem("No newer files on GitHub for helpers/ or presets/viz/."))
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
        except Exception as e:
            it = QListWidgetItem(f"Error loading beta files:\n{e}")
            self.list.addItem(it)
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
            for p in selected:
                data = _download_raw(self.owner, self.repo, branch, p)
                out = APP_ROOT / p
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, "wb") as f:
                    f.write(data)
            QMessageBox.information(self, "Beta update complete",
                                    f"Downloaded {len(selected)} file(s) from GitHub.\n"
                                    "You may need to restart FrameVision.")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Beta update failed", f"{e}\n\n{traceback.format_exc()}")

# ---------- Startup check ----------

def _today_str() -> str:
    return date.today().isoformat()

def check_on_startup(parent=None, owner: str = GITHUB_OWNER, repo: str = GITHUB_REPO):
    """
    Call this once during app startup (e.g., after main window shows).
    Runs at most once per calendar day. Pops up if a release or beta updates exist.
    """
    st = _load_state()
    last_day = st.get("last_check_day")
    if last_day == _today_str():
        return  # already checked today

    # Mark checked day early to avoid repeat popups if network is slow/fails later
    st["last_check_day"] = _today_str()
    _save_state(st)

    # Do the checks
    release_tag, release_published = _latest_release(owner, repo)
    try:
        branch = _get_default_branch(owner, repo)
    except Exception:
        branch = "main"
    try:
        beta_candidates = _collect_beta_candidates(owner, repo, branch)
    except Exception:
        beta_candidates = []

    has_release = bool(release_tag)
    has_beta = len(beta_candidates) > 0

    if not (has_release or has_beta):
        return

    # Build message
    lines = []
    if has_release:
        rel_dt = release_published or "unknown date"
        lines.append(f"New release available: {release_tag} (published {rel_dt})")
    if has_beta:
        lines.append(f"Beta: {len(beta_candidates)} newer file(s) found in helpers/ or presets/viz/.")

    msg = "\n".join(lines)
    box = QMessageBox(parent)
    box.setWindowTitle("Updates available on GitHub")
    box.setIcon(QMessageBox.Information)
    box.setText(msg)
    # Buttons
    btn_release = box.addButton("Open Release Updater…", QMessageBox.AcceptRole)
    btn_beta = box.addButton("Review Beta Files…", QMessageBox.ActionRole)
    box.addButton("Ignore", QMessageBox.RejectRole)
    box.exec()

    clicked = box.clickedButton()
    if clicked is btn_release:
        _open_release_updater(parent)
    elif clicked is btn_beta:
        dlg = BetaDialog(owner, repo, branch, parent)
        dlg.exec()

def force_check_now(parent=None, owner: str = GITHUB_OWNER, repo: str = GITHUB_REPO):
    """Force-run the startup check regardless of last run day (useful for testing)."""
    st = _load_state()
    st["last_check_day"] = "1970-01-01"
    _save_state(st)
    check_on_startup(parent, owner, repo)

# ---------- Release updater bridge ----------

def _open_release_updater(parent):
    """
    If the app has the 'Info → Updates…' dialog wired (menu_info.UpdateDialog),
    open that. Otherwise, show a minimal message explaining where to update.
    """
    # Try to import from app module (no hard dependency)
    try:
        from menu_info import _open_update_dialog as _open_update_dialog_fn
    except Exception:
        _open_update_dialog_fn = None

    if _open_update_dialog_fn is not None:
        try:
            _open_update_dialog_fn(parent)
            return
        except Exception:
            pass

    QMessageBox.information(parent, "Release updater",
                            "Couldn't locate the built-in updater dialog. "
                            "Open 'Info → Updates…' manually to update from the latest release.")

# ---------- Convenience installer ----------

def install_startup_check(main_window, delay_ms: int = 800):
    """
    Call once after creating your main window (post-show). Example:
        from helpers.update_checker import install_startup_check
        install_startup_check(self)
    """
    QTimer.singleShot(delay_ms, lambda: check_on_startup(main_window))


# ---------- Menu helpers (optional) ----------

def open_beta_dialog(parent=None, owner: str = GITHUB_OWNER, repo: str = GITHUB_REPO):
    """Open the Beta dialog on demand (e.g., from your Info menu)."""
    try:
        branch = _get_default_branch(owner, repo)
    except Exception:
        branch = "main"
    dlg = BetaDialog(owner, repo, branch, parent)
    dlg.exec()

def add_updates_to_menu(info_menu, parent=None):
    """
    Given a QMenu (your 'Info' menu), add:
      - 'Check for Updates Now…' (runs both release+beta checks immediately)
      - 'Review Beta Files…' (opens the checkbox picker)
    """
    if info_menu is None:
        return None, None
    act_check = info_menu.addAction("Check for Updates Now…")
    act_beta  = info_menu.addAction("Review Beta Files…")
    act_check.triggered.connect(lambda: force_check_now(parent))
    act_beta.triggered.connect(lambda: open_beta_dialog(parent))
    return act_check, act_beta

def schedule_check_no_parent(delay_ms: int = 800):
    """
    Schedule the daily check without a parent window.
    Useful if you're calling from modules that don't have access to the main window yet.
    """
    QTimer.singleShot(delay_ms, lambda: check_on_startup(None))

