from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QPlainTextEdit, QScrollArea, QWidget, QDialogButtonBox,
    QFileDialog, QMessageBox
)


class MultiPromptExtendDialog(QDialog):
    """
    Collect one prompt per EXTEND segment.

    Expected meaning:
    - These prompts apply to the *chained* segments only
      (not the initial run you start with the main Generate button).
    - Segment [1] = first extend segment after the initial clip.
    """

    def __init__(
        self,
        extend_count: int,
        base_prompt: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Multi prompt for Extend")
        self.setMinimumWidth(520)

        self._extend_count = max(int(extend_count), 0)
        self._editors: List[QPlainTextEdit] = []
        self._last_json_path: Optional[Path] = None

        root = QVBoxLayout(self)
        root.setSpacing(10)

        info = QLabel(
            "Assign a different prompt for each Extend segment.\n"
            "These prompts apply only to the extra chained segments."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        hint = QLabel(
            "JSON format (examples):\n"
            "1) Simple list:\n"
            "   [\"prompt for [1]\", \"prompt for [2]\"]\n"
            "2) Numbered keys:\n"
            "   {\"1\": \"...\", \"2\": \"...\"}\n"
            "3) Structured:\n"
            "   {\"version\": 1, \"segments\": [{\"prompt\": \"...\"}]}"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888;")
        root.addWidget(hint)

        # Scroll area for editors
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        inner_lay = QVBoxLayout(inner)
        inner_lay.setSpacing(8)

        for i in range(self._extend_count):
            lbl = QLabel(f"[{i+1}] Prompt for extend segment {i+1}:")
            ed = QPlainTextEdit()
            ed.setPlaceholderText("Enter prompt...")
            ed.setFixedHeight(70)

            if i == 0 and base_prompt:
                # Light convenience: seed first box with base prompt
                ed.setPlainText(base_prompt)

            inner_lay.addWidget(lbl)
            inner_lay.addWidget(ed)
            self._editors.append(ed)

        inner_lay.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll, stretch=1)

        # Row: load/save/fill/clear
        tools = QHBoxLayout()
        btn_load = QPushButton("Load JSON")
        btn_save = QPushButton("Save JSON")
        btn_fill = QPushButton("Fill all with base prompt")
        btn_clear = QPushButton("Clear all")

        tools.addWidget(btn_load)
        tools.addWidget(btn_save)
        tools.addSpacing(10)
        tools.addWidget(btn_fill)
        tools.addWidget(btn_clear)
        tools.addStretch(1)
        root.addLayout(tools)

        btn_load.clicked.connect(self._on_load_json)
        btn_save.clicked.connect(self._on_save_json)
        btn_fill.clicked.connect(lambda: self._fill_all(base_prompt))
        btn_clear.clicked.connect(self._clear_all)

        # OK / Cancel
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        root.addWidget(box)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)

    # ---------------------------
    # Public API
    # ---------------------------

    def prompts(self) -> List[str]:
        out: List[str] = []
        for ed in self._editors:
            try:
                out.append((ed.toPlainText() or "").strip())
            except Exception:
                out.append("")
        return out

    def config(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "extend_count": self._extend_count,
            "segments": [{"prompt": p} for p in self.prompts()],
        }

    # ---------------------------
    # Helpers
    # ---------------------------

    def _fill_all(self, text: str):
        if not text:
            return
        for ed in self._editors:
            try:
                ed.setPlainText(text)
            except Exception:
                pass

    def _clear_all(self):
        for ed in self._editors:
            try:
                ed.setPlainText("")
            except Exception:
                pass

    def _apply_prompts_list(self, items: List[str]):
        for i, ed in enumerate(self._editors):
            val = items[i] if i < len(items) else ""
            try:
                ed.setPlainText(val or "")
            except Exception:
                pass

    def _parse_json_payload(self, data: Any) -> List[str]:
        # допустимые формы:
        # - list[str]
        # - dict with numbered keys
        # - dict with "segments": [{"prompt": "..."}]
        if isinstance(data, list):
            return [str(x) for x in data]

        if isinstance(data, dict):
            if "segments" in data and isinstance(data["segments"], list):
                out = []
                for seg in data["segments"]:
                    if isinstance(seg, dict):
                        out.append(str(seg.get("prompt", "") or ""))
                    else:
                        out.append(str(seg))
                return out

            # numbered dict keys as strings or ints
            numbered = []
            for i in range(1, self._extend_count + 1):
                if str(i) in data:
                    numbered.append(str(data[str(i)] or ""))
                elif i in data:
                    numbered.append(str(data[i] or ""))
                else:
                    numbered.append("")
            # If any filled, use it
            if any(x.strip() for x in numbered):
                return numbered

        return []

    # ---------------------------
    # JSON I/O
    # ---------------------------

    def _on_load_json(self):
        fn, _ = QFileDialog.getOpenFileName(
            self,
            "Load multi-prompt JSON",
            "",
            "JSON (*.json);;All files (*.*)",
        )
        if not fn:
            return

        try:
            p = Path(fn)
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Could not load JSON:\n{e}")
            return

        prompts = self._parse_json_payload(data)
        if not prompts:
            QMessageBox.warning(
                self,
                "Unrecognized JSON",
                "This JSON doesn't look like a multi-prompt file.\n"
                "See the format hint in the dialog."
            )
            return

        self._last_json_path = p
        self._apply_prompts_list(prompts)

    def _on_save_json(self):
        default_name = "wan22_multiprompt.json"
        start_dir = ""
        try:
            if self._last_json_path:
                start_dir = str(self._last_json_path.parent)
        except Exception:
            start_dir = ""

        fn, _ = QFileDialog.getSaveFileName(
            self,
            "Save multi-prompt JSON",
            start_dir,
            "JSON (*.json);;All files (*.*)",
        )
        if not fn:
            return
        if not fn.lower().endswith(".json"):
            fn += ".json"

        try:
            payload = self.config()
            Path(fn).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            self._last_json_path = Path(fn)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"Could not save JSON:\n{e}")
