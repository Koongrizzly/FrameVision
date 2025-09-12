
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

    # Clear old duplicates, then add
    existing = {a.text(): a for a in info_menu.actions()}
    if act_kb.text() not in existing:
        info_menu.addAction(act_kb)
    if act_html.text() not in existing:
        info_menu.addAction(act_html)
