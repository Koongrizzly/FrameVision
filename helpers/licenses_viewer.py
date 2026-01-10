"""
Third-party licenses viewer (PySide6)

Loads presets/info/3rd_party_licenses.json and shows a searchable list with clickable links.

Important behavior:
- "Open source" opens a *landing page* (repo/model page), not a direct download URL.
- If a URL looks like a direct file download, we try to convert it to a safe landing page.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets


def _root_from_any(path_hint: Optional[str] = None) -> Path:
    if path_hint:
        return Path(path_hint).resolve()
    try:
        p = Path(__file__).resolve()
        if p.parent.name.lower() == "helpers":
            return p.parent.parent
        return p.parent
    except Exception:
        return Path.cwd().resolve()


# File extensions that are likely to trigger a browser download prompt
_DIRECT_FILE_EXTS = (
    ".zip", ".7z", ".rar",
    ".exe", ".dll",
    ".onnx", ".safetensors", ".pt", ".pth", ".bin", ".gguf",
    ".tar", ".gz", ".tgz",
)


def _looks_like_direct_download(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip().lower()
    if not (u.startswith("http://") or u.startswith("https://")):
        return False
    u_no_q = u.split("?", 1)[0]
    if "/releases/download/" in u_no_q:
        return True
    if "/resolve/" in u_no_q:  # Hugging Face direct file resolve
        return True
    return u_no_q.endswith(_DIRECT_FILE_EXTS)


def _to_landing_page(url: str) -> str:
    """
    Convert common direct-download URLs to a safer landing page.
    If no conversion is possible, returns the input URL unchanged.
    """
    if not url:
        return url

    # GitHub release asset -> release tag page
    m = re.match(r"https://github\.com/([^/]+)/([^/]+)/releases/download/([^/]+)/(.+)", url)
    if m:
        owner, repo, tag, _ = m.groups()
        return f"https://github.com/{owner}/{repo}/releases/tag/{tag}"

    # Hugging Face resolve -> repo tree at revision
    m = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)", url)
    if m:
        repo, rev, _ = m.groups()
        return f"https://huggingface.co/{repo}/tree/{rev}"

    # SourceForge download -> file page (strip trailing /download)
    if "sourceforge.net" in url and url.endswith("/download"):
        return url[:-len("/download")]

    # GitHub raw -> blob page
    m = re.match(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)", url)
    if m:
        owner, repo, branch, path = m.groups()
        return f"https://github.com/{owner}/{repo}/blob/{branch}/{path}"

    return url


def _safe_open_url(parent: QtWidgets.QWidget, url: str) -> None:
    """
    Open URL in the default browser, avoiding direct-download URLs by default.
    """
    if not url:
        return

    landing = _to_landing_page(url) if _looks_like_direct_download(url) else url

    # If we had to transform, prefer the landing page without downloading
    if landing != url:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(landing))
        return

    # If it still looks like a direct download, ask before opening
    if _looks_like_direct_download(url):
        box = QtWidgets.QMessageBox(parent)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("Direct download link")
        box.setText("This link looks like a direct file download.\n"
                    "For safety, you can copy it instead of opening it.")
        btn_open = box.addButton("Open anyway", QtWidgets.QMessageBox.AcceptRole)
        btn_copy = box.addButton("Copy link", QtWidgets.QMessageBox.ActionRole)
        btn_cancel = box.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        box.setDefaultButton(btn_copy)
        box.exec()
        clicked = box.clickedButton()
        if clicked == btn_copy:
            QtWidgets.QApplication.clipboard().setText(url)
        elif clicked == btn_open:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        else:
            return
        return

    QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))


@dataclass(frozen=True)
class LicenseItem:
    name: str
    category: str
    installed_by: str
    source: str
    source_url: str
    download_url: str
    license: str
    license_url: str
    notes: str


def _load_items(json_path: Path) -> List[LicenseItem]:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []

    out: List[LicenseItem] = []
    for it in (data.get("items") or []):
        try:
            out.append(
                LicenseItem(
                    name=str(it.get("name", "")).strip(),
                    category=str(it.get("category", "")).strip(),
                    installed_by=str(it.get("installed_by", "")).strip(),
                    source=str(it.get("source", "")).strip(),
                    source_url=str(it.get("source_url", "")).strip(),
                    download_url=str(it.get("download_url", "")).strip(),
                    license=str(it.get("license", "")).strip(),
                    license_url=str(it.get("license_url", "")).strip(),
                    notes=str(it.get("notes", "")).strip(),
                )
            )
        except Exception:
            continue
    return out


class _ItemRow(QtWidgets.QFrame):
    def __init__(self, item: LicenseItem, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.item = item
        self.setObjectName("LicenseRow")

        name_lbl = QtWidgets.QLabel(item.name)
        f = name_lbl.font()
        f.setBold(True)
        name_lbl.setFont(f)

        meta_lbl = QtWidgets.QLabel(f"{item.category} • {item.installed_by}")
        meta_lbl.setStyleSheet("opacity: 0.75;")

        lic_lbl = QtWidgets.QLabel(f"License: {item.license}")
        lic_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        notes_lbl = QtWidgets.QLabel(item.notes or "")
        notes_lbl.setWordWrap(True)
        notes_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        btn_license = QtWidgets.QPushButton("Open license")
        btn_source = QtWidgets.QPushButton("Open source")
        btn_license.setCursor(QtCore.Qt.PointingHandCursor)
        btn_source.setCursor(QtCore.Qt.PointingHandCursor)

        btn_license.clicked.connect(lambda: _safe_open_url(self, item.license_url))
        # Prefer source_url (landing page). Fall back to converting download_url.
        src = item.source_url or item.download_url
        btn_source.clicked.connect(lambda: _safe_open_url(self, src))

        btns = QtWidgets.QHBoxLayout()
        btns.setContentsMargins(0, 0, 0, 0)
        btns.setSpacing(8)
        btns.addWidget(btn_license)
        btns.addWidget(btn_source)
        btns.addStretch(1)

        top = QtWidgets.QVBoxLayout(self)
        top.setContentsMargins(12, 10, 12, 10)
        top.setSpacing(4)
        top.addWidget(name_lbl)
        top.addWidget(meta_lbl)
        top.addWidget(lic_lbl)
        if item.notes:
            top.addWidget(notes_lbl)
        top.addLayout(btns)

        self.setStyleSheet("""
        QFrame#LicenseRow {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
        }
        QPushButton {
            padding: 5px 10px;
            border-radius: 8px;
        }
        """)


class ThirdPartyLicensesDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, root_dir: Optional[str] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Third-party licenses")
        self.setMinimumSize(820, 560)
        self.setModal(False)
        self.setWindowModality(QtCore.Qt.NonModal)

        self.root_dir = _root_from_any(root_dir)
        self.json_path = (self.root_dir / "presets" / "info" / "3rd_party_licenses.json").resolve()

        title = QtWidgets.QLabel("Third-party licenses")
        tf = title.font()
        tf.setPointSize(tf.pointSize() + 4)
        tf.setBold(True)
        title.setFont(tf)

        subtitle = QtWidgets.QLabel("Open license/source to view upstream terms. Links avoid direct downloads by default.")
        subtitle.setWordWrap(True)

        self.search = QtWidgets.QLineEdit()
        self.search.setPlaceholderText("Search (name, license, source, notes)…")
        self.search.textChanged.connect(self._rebuild)

        self.chk_default = QtWidgets.QCheckBox("Default install")
        self.chk_optional = QtWidgets.QCheckBox("Optional installs")
        self.chk_default.setChecked(True)
        self.chk_optional.setChecked(True)
        self.chk_default.stateChanged.connect(lambda _=None: self._rebuild())
        self.chk_optional.stateChanged.connect(lambda _=None: self._rebuild())

        top_filters = QtWidgets.QHBoxLayout()
        top_filters.addWidget(self.search, 1)
        top_filters.addWidget(self.chk_default)
        top_filters.addWidget(self.chk_optional)

        self._list_box = QtWidgets.QWidget()
        self._list_lay = QtWidgets.QVBoxLayout(self._list_box)
        self._list_lay.setContentsMargins(0, 0, 0, 0)
        self._list_lay.setSpacing(10)
        self._list_lay.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(self._list_box)

        self.open_file_btn = QtWidgets.QPushButton("Open JSON file")
        self.open_file_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.open_file_btn.clicked.connect(self._open_json_file)

        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.close_btn.clicked.connect(self.close)

        btns = QtWidgets.QHBoxLayout()
        btns.addWidget(self.open_file_btn)
        btns.addStretch(1)
        btns.addWidget(self.close_btn)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(10)
        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addLayout(top_filters)
        lay.addWidget(scroll, 1)
        lay.addLayout(btns)

        self._items: List[LicenseItem] = _load_items(self.json_path)
        self._rebuild()

    def _open_json_file(self) -> None:
        if self.json_path.exists():
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.json_path)))

    def _clear_rows(self) -> None:
        while self._list_lay.count() > 1:
            item = self._list_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _rebuild(self) -> None:
        self._clear_rows()

        q = (self.search.text() or "").strip().lower()
        show_default = self.chk_default.isChecked()
        show_optional = self.chk_optional.isChecked()

        for it in self._items:
            if it.installed_by.lower().startswith("default") and not show_default:
                continue
            if it.installed_by.lower().startswith("optional") and not show_optional:
                continue

            hay = " ".join([it.name, it.category, it.installed_by, it.source, it.license, it.license_url, it.notes]).lower()
            if q and q not in hay:
                continue

            row = _ItemRow(it)
            self._list_lay.insertWidget(self._list_lay.count() - 1, row)


# Backwards compatible name used by older patches
class LicensesViewerDialog(ThirdPartyLicensesDialog):
    pass
