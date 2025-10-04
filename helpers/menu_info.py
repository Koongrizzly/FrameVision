
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QTextBrowser, QVBoxLayout, QWidget
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
    act_update.setToolTip("Update from GitHub (helpers/presets or full app)")
    act_update.triggered.connect(lambda: _open_update_dialog(main_window))
    
    existing = {a.text(): a for a in info_menu.actions()}
    if act_kb.text() not in existing:
        info_menu.addAction(act_kb)
    if act_html.text() not in existing:
        info_menu.addAction(act_html)
    if act_update.text() not in existing:
        info_menu.addAction(act_update)


# ===== Update Dialog & GitHub updater helpers =====
import shutil
import tempfile
import zipfile as _zipfile
import traceback
from datetime import datetime
from urllib.request import Request, urlopen
from PySide6.QtWidgets import (
    QPushButton, QFileDialog, QPlainTextEdit, QSpacerItem, QSizePolicy
)

GITHUB_OWNER = "Koongrizzly"
GITHUB_REPO = "FrameVision"

def _http_get(url: str, headers: dict | None = None, timeout: int = 25) -> bytes:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as r:
        return r.read()

def _latest_release_zip_url(owner: str, repo: str) -> tuple[str, str]:
    """Return (zipball_url, tag_name) for the newest release. Falls back to default branch zip if no releases."""
    base = f"https://api.github.com/repos/{owner}/{repo}"
    h = {"Accept": "application/vnd.github+json"}
    # Try latest release
    try:
        data = json.loads(_http_get(f"{base}/releases/latest", headers=h).decode("utf-8"))
        tag = (data.get("tag_name") or "").lstrip("v")
        zip_url = data.get("zipball_url")
        if zip_url:
            return zip_url, tag or "latest"
    except Exception:
        pass
    # Fallback: default branch
    try:
        repo_info = json.loads(_http_get(f"{base}", headers=h).decode("utf-8"))
        default_branch = repo_info.get("default_branch", "main")
        return f"{base}/zipball/{default_branch}", default_branch
    except Exception as e:
        raise RuntimeError(f"Unable to determine latest release or branch: {e}")

def _download_latest_zip(owner: str, repo: str) -> tuple[Path, str]:
    zip_url, tag = _latest_release_zip_url(owner, repo)
    data = _http_get(zip_url, headers={"Accept": "application/vnd.github+json"})
    tmpdir = Path(tempfile.mkdtemp(prefix="framevision_update_"))
    zpath = tmpdir / f"{repo}_{tag}.zip"
    zpath.write_bytes(data)
    return zpath, tag

def _extract_selected(zpath: Path, dest_root: Path, mode: str):
    """mode: 'partial' (helpers + presets/viz) or 'full' (entire repo)."""
    with _zipfile.ZipFile(zpath, "r") as zf:
        for m in zf.infolist():
            # Skip directories
            if m.is_dir():
                continue
            # Zipball has root/… path; drop the first component
            parts = Path(m.filename).parts
            if len(parts) < 2:
                continue
            rel = Path(*parts[1:])  # path inside repo
            if mode == "partial":
                # Only helpers/* and presets/viz/*
                if not (
                    (len(rel.parts) >= 1 and rel.parts[0] == "helpers") or
                    (len(rel.parts) >= 2 and rel.parts[0] == "presets" and rel.parts[1] == "viz")
                ):
                    continue
            # Compute output path and ensure parent exists
            out_path = dest_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(m, "r") as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

def _create_backup_zip(app_root: Path, target_dir: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{app_root.name}_backup_{ts}.zip"
    out = target_dir / name
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in app_root.rglob("*"):
            # Skip backup files and __pycache__
            if "__pycache__" in p.parts:
                continue
            if p.is_file():
                arc = str(p.relative_to(app_root))
                zf.write(p, arcname=arc)
    return out

class UpdateDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("FrameVision — Update from GitHub")
        self.resize(560, 320)

        info = QLabel(
            "<b>Warning:</b> Updating will <b>overwrite</b> local files that you've modified "
            "(e.g., in <code>helpers/</code> or <code>presets/viz/</code>).<br>"
            "It's strongly recommended to create a backup ZIP first."
        )
        info.setWordWrap(True)

        self.log = QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Update log will appear here…")
        self.log.setMaximumBlockCount(2000)

        btn_backup = QPushButton("Create backup ZIP…", self)
        btn_partial = QPushButton("Update Python files only (helpers + presets/viz)", self)
        btn_full = QPushButton("Update (entire app)", self)
        btn_cancel = QPushButton("Cancel", self)

        btn_backup.clicked.connect(self._do_backup)
        btn_partial.clicked.connect(lambda: self._do_update("partial"))
        btn_full.clicked.connect(lambda: self._do_update("full"))
        btn_cancel.clicked.connect(self.reject)

        # Layout
        v = QVBoxLayout(self)
        v.addWidget(info)
        v.addSpacing(8)
        v.addWidget(self.log, 1)

        h = QHBoxLayout()
        h.addWidget(btn_backup)
        h.addItem(QSpacerItem(10, 10, QSizePolicy.Expanding, QSizePolicy.Minimum))
        h.addWidget(btn_partial)
        h.addWidget(btn_full)
        h.addWidget(btn_cancel)
        v.addLayout(h)

    def _append(self, text: str):
        self.log.appendPlainText(text)

    def _do_backup(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder to save backup ZIP")
        if not folder:
            return
        try:
            self.setEnabled(False)
            self._append("Creating backup…")
            out = _create_backup_zip(APP_ROOT, Path(folder))
            self._append(f"Backup saved: {out}")
            QMessageBox.information(self, "Backup created", f"Backup ZIP saved to:\n{out}")
        except Exception as e:
            self._append("Backup failed:\n" + traceback.format_exc())
            QMessageBox.critical(self, "Backup failed", str(e))
        finally:
            self.setEnabled(True)

    def _do_update(self, mode: str):
        assert mode in ("partial", "full")
        try:
            self.setEnabled(False)
            self._append("Fetching latest release…")
            zpath, tag = _download_latest_zip(GITHUB_OWNER, GITHUB_REPO)
            self._append(f"Downloaded release/branch zip: {zpath.name} (tag: {tag})")
            self._append("Applying update… (this may take a moment)")
            _extract_selected(zpath, APP_ROOT, "partial" if mode == "partial" else "full")
            self._append("Update complete.")
            QMessageBox.information(self, "Update complete",
                                    f"Updated from GitHub ({tag}).\n"
                                    f"Mode: {'Python files only' if mode=='partial' else 'Full app'}.\n\n"
                                    "You may need to restart FrameVision.")
        except Exception as e:
            self._append("Update failed:\n" + traceback.format_exc())
            QMessageBox.critical(self, "Update failed", str(e))
        finally:
            self.setEnabled(True)

def _open_update_dialog(parent):
    dlg = UpdateDialog(parent)
    dlg.exec()
# ===== End Update Dialog =====