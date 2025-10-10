
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QTextBrowser, QVBoxLayout, QWidget, QCheckBox
)

APP_ROOT = Path(__file__).resolve().parents[1]
INFO_DIR = APP_ROOT / "presets" / "info"
KB_PATH = INFO_DIR / "framevision_knowledge_base.json"
HTML_GUIDE_PATH = INFO_DIR / "FrameVision_Feature_Guide.html"


def _normalize_kb(raw):
    """Accepts multiple schema shapes and normalizes to a list of sections.
    Each section is a dict with keys: title, description, tips/items, qa (list of {q,a}).
    """
    if raw is None:
        return []
    # Case 1: already a list
    if isinstance(raw, list):
        return raw

    # Case 2: dict with known wrapper key
    if isinstance(raw, dict):
        # e.g., {"FrameVisionKnowledgeBase": { "SectionName": {...}}}
        if "FrameVisionKnowledgeBase" in raw and isinstance(raw["FrameVisionKnowledgeBase"], dict):
            sections = []
            for name, sec in raw["FrameVisionKnowledgeBase"].items():
                section = {
                    "title": name,
                    "description": sec.get("description", ""),
                }
                # Map tips/items
                items = sec.get("hints_tips") or sec.get("tips") or sec.get("items") or []
                section["items"] = items

                # Map QA
                qas = []
                for pair in sec.get("qa_pairs", []) or []:
                    qas.append({"q": pair.get("question", ""), "a": pair.get("answer", "")})
                section["qa"] = qas

                # Include key terms if present
                if sec.get("key_terms"):
                    section["items"] = (section.get("items") or []) + [f"<em>Key term:</em> {t}" for t in sec["key_terms"]]

                sections.append(section)
            return sections

        # Case 3: dict of sections (title -> section dict)
        possible_sections = []
        for k, v in raw.items():
            if isinstance(v, dict) and {"description", "qa_pairs"} & set(v.keys()):
                possible_sections.append((k, v))
        if possible_sections:
            out = []
            for name, sec in possible_sections:
                out.append({
                    "title": name,
                    "description": sec.get("description", ""),
                    "items": sec.get("items") or sec.get("tips") or [],
                    "qa": [{"q": p.get("question",""), "a": p.get("answer","")} for p in sec.get("qa_pairs", [])],
                })
            return out

    return []


def _load_kb():
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return _normalize_kb(raw)
    except FileNotFoundError:
        return {"error": f"Knowledge base not found at {KB_PATH}"}
    except Exception as e:
        return {"error": f"Failed to load knowledge base: {e}"}


class KnowledgeDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("FrameVision — Knowledge & Q&A")
        self.resize(900, 600)

        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Type a question or keywords… (Enter to search)")
        self.search.returnPressed.connect(self.perform_search)

        self.list = QListWidget(self)
        self.list.currentItemChanged.connect(self._on_item_changed)

        self.viewer = QTextBrowser(self)
        self.viewer.setOpenExternalLinks(True)

        # Layouts
        left = QVBoxLayout()
        left.addWidget(QLabel("Sections"))
        left.addWidget(self.list, 1)

        right = QVBoxLayout()
        right.addWidget(self.search)
        right.addWidget(self.viewer, 1)

        main = QHBoxLayout(self)
        left_w = QWidget(self); left_w.setLayout(left)
        right_w = QWidget(self); right_w.setLayout(right)
        main.addWidget(left_w, 1)
        main.addWidget(right_w, 2)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Close, self)
        buttons.rejected.connect(self.reject)
        right.addWidget(buttons)

        # Data
        self.kb = _load_kb()
        self._populate_sections()

    def _populate_sections(self):
        self.list.clear()
        if isinstance(self.kb, dict) and "error" in self.kb:
            self.viewer.setHtml(f"<h3>Error</h3><p>{self.kb['error']}</p>")
            return

        sections = self.kb if isinstance(self.kb, list) else []
        for section in sections:
            title = section.get("title") or section.get("section") or "Untitled"
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, section)
            self.list.addItem(item)

        if self.list.count() > 0:
            self.list.setCurrentRow(0)

    def _on_item_changed(self, current: QListWidgetItem, _prev: QListWidgetItem):
        if not current:
            return
        section = current.data(Qt.UserRole) or {}
        html = self._section_to_html(section)
        self.viewer.setHtml(html)

    def perform_search(self):
        query = self.search.text().strip().lower()
        if not query:
            return
        # Simple search over section titles, descriptions and Q&A text
        results = []
        data = self.kb if isinstance(self.kb, list) else []
        for section in data:
            text_blobs = [
                section.get("title",""),
                section.get("description",""),
                " ".join(section.get("items", []) or []),
                " ".join([f"{qa.get('q','')} {qa.get('a','')}" for qa in section.get("qa", []) or []]),
            ]
            joined = " ".join(text_blobs).lower()
            score = joined.count(query) if query in joined else 0
            if score > 0:
                results.append((score, section))
        if not results:
            QMessageBox.information(self, "No results", "No matches found in the knowledge base.")
            return
        results.sort(reverse=True, key=lambda x: x[0])
        top = results[0][1]
        # Select the matching section
        for i in range(self.list.count()):
            if self.list.item(i).text() == (top.get("title") or "Untitled"):
                self.list.setCurrentRow(i)
                break
        self.viewer.setHtml(self._section_to_html(top, highlight=query))

    @staticmethod
    def _section_to_html(section: dict, highlight: str | None = None) -> str:
        def h(text: str) -> str:
            if not text:
                return ""
            if highlight:
                return text.replace(highlight, f"<mark>{highlight}</mark>")
            return text

        title = h(section.get("title") or "Untitled")
        parts = [f"<h2>{title}</h2>"]

        if "description" in section and section["description"]:
            parts.append(f"<p>{h(section['description'])}</p>")

        # Q&A entries
        qas = section.get("qa") or []
        if qas:
            parts.append("<h3>Q&A</h3><ul>")
            for qa in qas:
                q = h(qa.get("q") or "")
                a = h(qa.get("a") or "")
                parts.append(f"<li><strong>{q}</strong><br>{a}</li>")
            parts.append("</ul>")

        # Tips / items
        items = section.get("items") or []
        if items:
            parts.append("<h3>Notes</h3><ul>")
            for it in items:
                parts.append(f"<li>{h(str(it))}</li>")
            parts.append("</ul>")

        return "\n".join(parts)


def _open_html_guide():
    path = HTML_GUIDE_PATH
    if not path.exists():
        QMessageBox.warning(None, "Feature Guide missing",
                            f"Couldn't find the HTML guide at:\n{path}")
        return
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def _open_kb_dialog(parent):
    dlg = KnowledgeDialog(parent)
    dlg.exec()


def install_info_menu(main_window):
    """Create the Info menu with 'Knowledge & Q&A' and 'Feature Guide (HTML)'.
    Safe to call multiple times.
    """
    if not hasattr(main_window, "menuBar"):
        return

    menubar = main_window.menuBar()
    # Find existing "Info" menu if any
    info_menu = None
    for action in menubar.actions():
        if action.text().replace("&", "").strip().lower() == "info":
            info_menu = action.menu()
            break

    if info_menu is None:
        info_menu = QMenu("&Info", menubar)
        menubar.addMenu(info_menu)

    # Actions
    act_kb = QAction("Knowledge && Q&A…", main_window)
    act_kb.triggered.connect(lambda: _open_kb_dialog(main_window))
    
    act_html = QAction("Feature Guide (HTML)…", main_window)
    act_html.triggered.connect(_open_html_guide)

    act_update = QAction("Updates…", main_window)
    act_update.setToolTip("Update from GitHub (Stable release or Beta branch)")
    act_update.triggered.connect(lambda: _open_update_dialog(main_window))
    
    act_update = QAction("Updates…", main_window)
    act_update.setToolTip("Update from GitHub (helpers/presets or full app)")
    act_update.triggered.connect(lambda: _open_update_dialog(main_window))
    
    existing = {a.text(): a for a in info_menu.actions()}
    if act_kb.text() not in existing:
        info_menu.addAction(act_kb)
    if act_html.text() not in existing:
        info_menu.addAction(act_html)
    if act_update.text() not in existing:
        info_menu.addAction(act_update)
    if act_update.text() not in existing:
        info_menu.addAction(act_update)





# ===== Update Dialog: Stable (Release) vs Beta (Default Branch) =====
import shutil, tempfile, traceback, zipfile as _zipfile
from datetime import datetime
from urllib.request import Request, urlopen
from PySide6.QtWidgets import QPushButton, QFileDialog, QPlainTextEdit, QSpacerItem, QSizePolicy

GITHUB_OWNER = "Koongrizzly"
GITHUB_REPO = "FrameVision"

def _http_get_bytes(url: str, headers: dict | None = None, timeout: int = 30) -> bytes:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as r:
        return r.read()

def _http_get_json(url: str, headers: dict | None = None, timeout: int = 30) -> dict:
    data = _http_get_bytes(url, headers=headers, timeout=timeout)
    return json.loads(data.decode("utf-8"))

def _default_branch(owner: str, repo: str) -> str:
    try:
        info = _http_get_json(f"https://api.github.com/repos/{owner}/{repo}",
                              headers={"Accept": "application/vnd.github+json"})
        return info.get("default_branch") or "main"
    except Exception:
        return "main"

def _download_latest_release_zip(owner: str, repo: str) -> tuple[Path, str]:
    """Latest stable release ZIP (GitHub Releases). Falls back to default branch if no releases exist."""
    try:
        info = _http_get_json(f"https://api.github.com/repos/{owner}/{repo}/releases/latest",
                              headers={"Accept": "application/vnd.github+json"})
        zip_url = info.get("zipball_url")
        tag = (info.get("tag_name") or "latest").lstrip("v")
        if not zip_url:
            raise RuntimeError("No stable release found")
        data = _http_get_bytes(zip_url, headers={"Accept": "application/octet-stream"})
        tmpdir = Path(tempfile.mkdtemp(prefix="framevision_rel_"))
        zpath = tmpdir / f"{repo}_{tag}.zip"
        zpath.write_bytes(data)
        return zpath, tag
    except Exception:
        # Fallback: use default branch ZIP but clearly label as fallback
        branch = _default_branch(owner, repo)
        url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
        data = _http_get_bytes(url, headers={"Accept": "application/octet-stream"})
        tmpdir = Path(tempfile.mkdtemp(prefix="framevision_rel_fb_"))
        zpath = tmpdir / f"{repo}_{branch}.zip"
        zpath.write_bytes(data)
        return zpath, f"{branch} (fallback)"

def _download_latest_branch_zip(owner: str, repo: str) -> tuple[Path, str]:
    """Latest beta ZIP (default branch HEAD)."""
    branch = _default_branch(owner, repo)
    url = f"https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"
    data = _http_get_bytes(url, headers={"Accept": "application/octet-stream"})
    tmpdir = Path(tempfile.mkdtemp(prefix="framevision_beta_"))
    zpath = tmpdir / f"{repo}_{branch}.zip"
    zpath.write_bytes(data)
    return zpath, branch

def _extract_selected(zpath: Path, dest_root: Path, mode: str):
    """mode: 'partial' (helpers + presets/viz) or 'full' (entire repo)."""
    with _zipfile.ZipFile(zpath, "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            parts = Path(member.filename).parts
            if len(parts) < 2:
                continue
            rel = Path(*parts[1:])  # drop the top-level folder GitHub adds
            if mode == "partial":
                if not (
                    (len(rel.parts) >= 1 and rel.parts[0] == "helpers") or
                    (len(rel.parts) >= 2 and rel.parts[0] == "presets" and rel.parts[1] == "viz")
                ):
                    continue
            out_path = dest_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

def _create_backup_zip(app_root: Path, target_dir: Path) -> Path:
    import os, zipfile as _zipfile
    from datetime import datetime as _dt

    EXCLUDE_TOP = {"tools", "models", "output", ".venv"}
    EXCLUDE_PRESETS_SUB = {"bin"}
    EXCLUDE_EXTS = {".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz"}
    app_prefix = (app_root.name + "_backup_").lower()

    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    out = Path(target_dir) / f"{app_root.name}_backup_{ts}.zip"

    with _zipfile.ZipFile(out, "w", compression=_zipfile.ZIP_STORED, allowZip64=True) as zf:
        for root, dirs, files in os.walk(app_root):
            rel_root = Path(root).relative_to(app_root)
            parts = rel_root.parts
            if parts:
                top = parts[0]
                if top in EXCLUDE_TOP:
                    dirs[:] = []
                    continue
                if top == "presets" and len(parts) >= 2 and parts[1] in EXCLUDE_PRESETS_SUB:
                    dirs[:] = []
                    continue
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for fname in files:
                p = Path(root) / fname
                rel = p.relative_to(app_root)
                if p.suffix.lower() in EXCLUDE_EXTS:
                    continue
                if rel.name.lower().startswith(app_prefix) and p.suffix.lower() == ".zip":
                    continue
                zf.write(p, arcname=str(rel))
    return out


class _BackupWorker(QThread):
    prog = Signal(int, int)
    msg  = Signal(str)
    ok   = Signal(str)
    err  = Signal(str)

    def __init__(self, app_root: Path, target_dir: Path):
        super().__init__()
        self._app_root = app_root
        self._target_dir = target_dir

    def _gather_files(self):
        EXCLUDE_TOP = {"tools", "models", "output", ".venv"}
        EXCLUDE_PRESETS_SUB = {"bin"}
        EXCLUDE_EXTS = {".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz"}
        app_prefix = (self._app_root.name + "_backup_").lower()
        files = []
        for p in self._app_root.rglob("*"):
            if "__pycache__" in p.parts:
                continue
            try:
                rel = p.relative_to(self._app_root)
            except ValueError:
                continue
            if rel.parts:
                top = rel.parts[0]
                if top in EXCLUDE_TOP:
                    continue
                if top == "presets" and len(rel.parts) >= 2 and rel.parts[1] in EXCLUDE_PRESETS_SUB:
                    continue
            if p.is_file():
                if p.suffix.lower() in EXCLUDE_EXTS:
                    continue
                if rel.name.lower().startswith(app_prefix) and p.suffix.lower() == ".zip":
                    continue
                files.append((p, rel))
        return files

    def run(self):
        try:
            files = self._gather_files()
            total = len(files)
            if total == 0:
                raise RuntimeError("Nothing to back up.")
            self.msg.emit(f"Found {total} files. Writing ZIP…")
            from datetime import datetime as _dt
            import zipfile as _zipfile
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            out = Path(self._target_dir) / f"{self._app_root.name}_backup_{ts}.zip"
            with _zipfile.ZipFile(out, "w", compression=_zipfile.ZIP_STORED, allowZip64=True) as zf:
                for i, (p, rel) in enumerate(files, start=1):
                    zf.write(p, arcname=str(rel))
                    if i == total or (i % 50 == 0):
                        self.prog.emit(i, total)
            self.ok.emit(str(out))
        except Exception:
            import traceback as _tb
            self.err.emit(_tb.format_exc())
class UpdateDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("FrameVision — Update from GitHub")
        self.resize(640, 420)

        warn = QLabel(
            "<b>Warning:</b> Updating will <b>overwrite</b> local changes (e.g., in <code>helpers/</code> "
            "or <code>presets/viz/</code>).<br>Consider creating a backup ZIP first."
        )
        warn.setWordWrap(True)

        stable_hdr = QLabel("<b>Latest Stable release</b>")
        beta_hdr = QLabel("<b>Beta</b> <span style='color:#888'>(can have test releases and bugs)</span>")

        self.log = QPlainTextEdit(self); self.log.setReadOnly(True)
        self.log.setPlaceholderText("Update log…")

        btn_backup = QPushButton("Create backup ZIP…", self)

        btn_rel_partial = QPushButton("Replace python files only", self)
        btn_rel_full    = QPushButton("Replace with release", self)
        btn_beta_partial= QPushButton("Update python files only", self)
        btn_beta_full   = QPushButton("Update all files", self)
        btn_cancel      = QPushButton("Cancel", self)

        btn_backup.clicked.connect(self._on_backup)
        btn_rel_partial.clicked.connect(lambda: self._on_update('release','partial'))
        btn_rel_full.clicked.connect(lambda: self._on_update('release','full'))
        btn_beta_partial.clicked.connect(lambda: self._on_update('branch','partial'))
        btn_beta_full.clicked.connect(lambda: self._on_update('branch','full'))
        btn_cancel.clicked.connect(self.reject)

        v = QVBoxLayout(self)
        v.addWidget(warn)
        v.addSpacing(6)

        v.addWidget(stable_hdr)
        row1 = QHBoxLayout()
        row1.addWidget(btn_rel_partial)
        row1.addWidget(btn_rel_full)
        v.addLayout(row1)

        v.addSpacing(8)

        # Auto-check toggle (persists via helpers.update_checker)
        try:
            from helpers.update_checker import get_auto_check_enabled, set_auto_check_enabled
            auto_on = get_auto_check_enabled()
        except Exception:
            auto_on = True
            get_auto_check_enabled = None
            set_auto_check_enabled = None

        self.chk_auto = QCheckBox("Auto check for updates at startup", self)
        self.chk_auto.setChecked(bool(auto_on))
        def _on_auto(ticked):
            if set_auto_check_enabled is not None:
                try:
                    set_auto_check_enabled(bool(ticked))
                except Exception:
                    pass
        self.chk_auto.toggled.connect(_on_auto)
        v.addWidget(self.chk_auto)

        v.addWidget(beta_hdr)
        row2 = QHBoxLayout()
        row2.addWidget(btn_beta_partial)
        row2.addWidget(btn_beta_full)
        v.addLayout(row2)

        v.addSpacing(8)

        # Auto-check toggle (persists via helpers.update_checker)
        try:
            from helpers.update_checker import get_auto_check_enabled, set_auto_check_enabled
            auto_on = get_auto_check_enabled()
        except Exception:
            auto_on = True
            get_auto_check_enabled = None
            set_auto_check_enabled = None

        self.chk_auto = QCheckBox("Auto check for updates at startup", self)
        self.chk_auto.setChecked(bool(auto_on))
        def _on_auto(ticked):
            if set_auto_check_enabled is not None:
                try:
                    set_auto_check_enabled(bool(ticked))
                except Exception:
                    pass
        self.chk_auto.toggled.connect(_on_auto)
        v.addWidget(self.chk_auto)

        v.addWidget(self.log, 1)

        row3 = QHBoxLayout()
        row3.addWidget(btn_backup)
        row3.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        row3.addWidget(btn_cancel)
        v.addLayout(row3)

    def _append(self, msg: str):
        self.log.appendPlainText(msg)

    def _on_backup(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to save backup ZIP")
        if not folder:
            return
        self.setEnabled(False)
        self._append("Creating backup (no compression)…")
        self._backup_worker = _BackupWorker(APP_ROOT, Path(folder))
        self._backup_worker.msg.connect(lambda s: self._append(s))
        self._backup_worker.prog.connect(lambda d, t: self._append(f"Zipping {d}/{t} files…"))
        def _ok(path_str: str):
            self._append(f"Backup saved: {path_str}")
            QMessageBox.information(self, "Backup created", f"Saved to:\n{path_str}")
            self.setEnabled(True)
            self._backup_worker = None
        def _err(msg: str):
            self._append("Backup failed:\n" + msg)
            QMessageBox.critical(self, "Backup failed", "See log for details.")
            self.setEnabled(True)
            self._backup_worker = None
        self._backup_worker.ok.connect(_ok)
        self._backup_worker.err.connect(_err)
        self._backup_worker.start()


    def _on_update(self, source: str, mode: str):
        assert source in ("release", "branch")
        assert mode in ("partial", "full")
        self.setEnabled(False)
        try:
            if source == "release":
                self._append("Fetching latest STABLE release ZIP…")
                zpath, label = _download_latest_release_zip(GITHUB_OWNER, GITHUB_REPO)
            else:
                self._append("Fetching latest BETA (default-branch) ZIP…")
                zpath, label = _download_latest_branch_zip(GITHUB_OWNER, GITHUB_REPO)
            self._append(f"Downloaded: {zpath.name} [{label}]")
            self._append(f"Applying update ({'helpers + presets/viz' if mode=='partial' else 'entire app'})…")
            _extract_selected(zpath, APP_ROOT, mode)
            self._append("Update complete.")
            QMessageBox.information(
                self, "Update complete",
                f"{'Stable release' if source=='release' else 'Beta (branch)'} applied: {label}\\n"
                f"Mode: {'Python files only' if mode=='partial' else 'All files'}.\\n\\n"
                "Please restart FrameVision."
            )
        except Exception as e:
            self._append("Update failed:\\n" + traceback.format_exc())
            QMessageBox.critical(self, "Update failed", str(e))
        finally:
            self.setEnabled(True)

def _open_update_dialog(parent: QWidget | None):
    dlg = UpdateDialog(parent)
    dlg.exec()
# ===== End Update Dialog =====
