
# helpers/renam.py — enhanced Multi Rename pane (with full tooltips)
from __future__ import annotations
import os, re, csv, json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton,
    QFormLayout, QSpinBox, QCheckBox, QLineEdit, QComboBox, QLabel,
    QFileDialog, QMessageBox, QToolButton
)

IMG_EXTS = {".png",".jpg",".jpeg",".bmp",".tif",".tiff",".webp",".gif"}
VID_EXTS = {".mp4",".mov",".mkv",".webm",".avi",".wmv",".m4v",".mpg",".mpeg"}

class _Collapsible(QWidget):
    """Lightweight collapsible panel used inside RenamPane only."""
    def __init__(self, title: str, parent=None, expanded=False):
        super().__init__(parent)
        self._expanded = bool(expanded)
        self.toggle = QToolButton(self)
        self.toggle.setStyleSheet("QToolButton { border:none; }")
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow if self._expanded else Qt.RightArrow)
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(self._expanded)
        self.toggle.setToolTip(f"Click to {'collapse' if self._expanded else 'expand'} “{title}”.")

        self.content = QWidget(self)
        self.content.setLayout(QVBoxLayout())
        self.content.layout().setContentsMargins(12,6,12,6)
        self.content.setVisible(self._expanded)
        if not self._expanded:
            self.content.setMaximumHeight(0)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.setSpacing(4)
        lay.addWidget(self.toggle)
        lay.addWidget(self.content)

        def _on_toggled(on):
            self._expanded = on
            self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
            self.toggle.setToolTip(f"Click to {'collapse' if on else 'expand'} “{title}”.")
            self.content.setVisible(on)
            self.content.setMaximumHeight(16777215 if on else 0)
        self.toggle.toggled.connect(_on_toggled)

    def setContentLayout(self, layout):
        QWidget().setLayout(self.content.layout())
        self.content.setLayout(layout)
        self.content.setMaximumHeight(16777215 if self._expanded else 0)

class RenamPane(QWidget):
    """
    Self-contained multi-rename tool.
    Adds safety (undo, conflicts), power features (numbering, search/replace, case, tokens), and presets.
    """
    def __init__(self, main=None, parent=None):
        super().__init__(parent)
        self.main = main
        self._paths: List[str] = []
        self._last_plan: List[Tuple[str,str]] = []  # (old_path, new_path)
        self._last_plan_file: Path | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0,0,0,0)
        root.setSpacing(8)

        # --- Top controls: picking, filtering, sorting
        row_top = QHBoxLayout()
        self.btn_pick_folder = QPushButton("Choose folder…")
        self.btn_pick_folder.setToolTip("Pick a folder. All files inside will be listed (filtered by Type).")
        self.btn_pick_files  = QPushButton("Choose files…")
        self.btn_pick_files.setToolTip("Pick specific files to rename. Hold Ctrl/Shift to multi-select.")
        row_top.addWidget(self.btn_pick_folder)
        row_top.addWidget(self.btn_pick_files)

        self.cmb_type = QComboBox(); self.cmb_type.addItems(["All","Images","Videos"])
        self.cmb_type.setToolTip("Filter which files are included: All, Images, or Videos.")
        self.cmb_sort = QComboBox(); self.cmb_sort.addItems(["Name","Date modified","Size"])
        self.cmb_sort.setToolTip("Sort files by name, last modified time, or size.")
        self.cmb_order = QComboBox(); self.cmb_order.addItems(["Ascending","Descending"])
        self.cmb_order.setToolTip("Choose ascending or descending sort order.")
        row_top.addWidget(QLabel("Type:"))
        row_top.addWidget(self.cmb_type)
        row_top.addWidget(QLabel("Sort:"))
        row_top.addWidget(self.cmb_sort)
        row_top.addWidget(self.cmb_order)
        row_top.addStretch(1)
        root.addLayout(row_top)

        # Filter box
        row_filter = QHBoxLayout()
        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("Filter preview (substring or regex: re/your_regex)")
        self.edit_filter.setToolTip("Live filter for the preview below. Type plain text or use regex by starting with re/, e.g., re/^IMG_\\d+/.")
        lbl_f = QLabel("Filter:")
        lbl_f.setToolTip("Live filter for the preview below. Type plain text or use regex by starting with re/.")
        row_filter.addWidget(lbl_f)
        row_filter.addWidget(self.edit_filter)
        root.addLayout(row_filter)

        # Preview list
        self.rename_preview = QListWidget()
        self.rename_preview.setMinimumHeight(200)
        self.rename_preview.setToolTip("Preview of old → new filenames. Red = potential conflict (destination exists or duplicate in plan).")
        root.addWidget(self.rename_preview)

        # --- Basic options section
        sec_basic = _Collapsible("Basic options", expanded=True)
        f_basic = QFormLayout()

        self.rm_start = QSpinBox(); self.rm_start.setRange(0, 999); self.rm_start.setValue(0)
        self.rm_start.setToolTip("Remove N characters from the START of each name (before other operations).")
        self.rm_end   = QSpinBox(); self.rm_end.setRange(0, 999); self.rm_end.setValue(0)
        self.rm_end.setToolTip("Remove N characters from the END of each name (before other operations).")
        self.cb_remove_leading_nums = QCheckBox("Remove leading numbers")
        self.cb_remove_leading_nums.setToolTip("Strip any digits at the very beginning of the name (e.g., '001-File' → 'File').")
        self.ed_prefix = QLineEdit(); self.ed_prefix.setToolTip("Text to add before the filename (applied after date/counter if enabled).")
        self.ed_suffix = QLineEdit(); self.ed_suffix.setToolTip("Text to add after the filename (before the extension).")
        # Date options
        self.cb_add_date = QCheckBox("Add date")
        self.cb_add_date.setToolTip("Add a date token to the prefix. By default it becomes 'DATE_'.")
        self.cmb_date_source = QComboBox(); self.cmb_date_source.addItems(["Today","Modified time"])
        self.cmb_date_source.setToolTip("Choose which date to use: today's date or the file's last modified date.")
        self.ed_date_fmt = QLineEdit("%Y%m%d")
        self.ed_date_fmt.setToolTip("Datetime format using Python strftime, e.g., %Y-%m-%d_%H%M.")

        f_basic.addRow("Remove first N chars", self.rm_start)
        f_basic.addRow("Remove last  N chars", self.rm_end)
        f_basic.addRow(self.cb_remove_leading_nums)
        f_basic.addRow("Add prefix", self.ed_prefix)
        f_basic.addRow("Add suffix", self.ed_suffix)
        row_date = QHBoxLayout()
        row_date.addWidget(self.cb_add_date)
        row_date.addWidget(QLabel("Source:")); row_date.addWidget(self.cmb_date_source)
        row_date.addWidget(QLabel("Format:")); row_date.addWidget(self.ed_date_fmt)
        row_date.addStretch(1)
        f_basic.addRow("Date options", row_date)

        sec_basic.setContentLayout(f_basic)
        sec_basic.toggle.setToolTip("Quick rename settings (always visible). Click to collapse/expand.")
        root.addWidget(sec_basic)

        # --- Advanced section (collapsed by default)
        sec_adv = _Collapsible("Advanced settings", expanded=False)
        f_adv = QFormLayout()

        # Numbering
        row_num = QHBoxLayout()
        self.cb_numbering = QCheckBox("Enable numbering"); self.cb_numbering.setToolTip("Turn on an auto-incrementing counter.")
        self.sp_num_start = QSpinBox(); self.sp_num_start.setRange(0, 999999); self.sp_num_start.setValue(1); self.sp_num_start.setToolTip("First counter value.")
        self.sp_num_step  = QSpinBox(); self.sp_num_step.setRange(1, 999999); self.sp_num_step.setValue(1); self.sp_num_step.setToolTip("How much to add between files.")
        self.sp_num_pad   = QSpinBox(); self.sp_num_pad.setRange(1, 8); self.sp_num_pad.setValue(3); self.sp_num_pad.setToolTip("Zero-padding length (e.g., 3 → 001).")
        self.cmb_num_pos  = QComboBox(); self.cmb_num_pos.addItems(["prefix","suffix","replace filename"]); self.cmb_num_pos.setToolTip("Insert the counter at the start, end, or use it as the whole name.")
        row_num.addWidget(self.cb_numbering)
        row_num.addWidget(QLabel("start")); row_num.addWidget(self.sp_num_start)
        row_num.addWidget(QLabel("step"));  row_num.addWidget(self.sp_num_step)
        row_num.addWidget(QLabel("pad"));   row_num.addWidget(self.sp_num_pad)
        row_num.addWidget(QLabel("position")); row_num.addWidget(self.cmb_num_pos)
        row_num.addStretch(1)
        f_adv.addRow("Numbering", row_num)

        # Search & Replace
        row_sr = QVBoxLayout()
        row1 = QHBoxLayout()
        self.ed_find = QLineEdit(); self.ed_find.setPlaceholderText("Find text or regex"); self.ed_find.setToolTip("Text to search for. Use regex if 'Use regex' is enabled.")
        self.ed_repl = QLineEdit(); self.ed_repl.setPlaceholderText("Replace with…"); self.ed_repl.setToolTip("Replacement text. With regex you can use groups like \\1.")
        row1.addWidget(self.ed_find); row1.addWidget(self.ed_repl)
        row2 = QHBoxLayout()
        self.cb_regex = QCheckBox("Use regex"); self.cb_regex.setToolTip("Interpret 'Find' as a regular expression.")
        self.cb_case  = QCheckBox("Case sensitive"); self.cb_case.setChecked(False); self.cb_case.setToolTip("Match case exactly when searching.")
        self.cb_apply_ext = QCheckBox("Apply to extension"); self.cb_apply_ext.setChecked(False); self.cb_apply_ext.setToolTip("Also apply find/replace to the file extension.")
        row2.addWidget(self.cb_regex); row2.addWidget(self.cb_case); row2.addWidget(self.cb_apply_ext); row2.addStretch(1)
        row_sr.addLayout(row1); row_sr.addLayout(row2)
        f_adv.addRow("Search & Replace", row_sr)

        # Case conversion & extension handling
        self.cmb_case = QComboBox(); self.cmb_case.addItems(["none","lower","UPPER","Title Case"]); self.cmb_case.setToolTip("Transform the filename's letter case.")
        row_ext = QHBoxLayout()
        self.cmb_ext_mode = QComboBox(); self.cmb_ext_mode.addItems(["keep","lowercase","UPPERCASE","change to…"]); self.cmb_ext_mode.setToolTip("How to treat the file extension.")
        self.ed_ext_new = QLineEdit(); self.ed_ext_new.setPlaceholderText("e.g., jpg"); self.ed_ext_new.setFixedWidth(100); self.ed_ext_new.setToolTip("When 'change to…' is selected, type the new extension here (without dot).")
        row_ext.addWidget(self.cmb_ext_mode); row_ext.addWidget(self.ed_ext_new); row_ext.addStretch(1)
        f_adv.addRow("Name case", self.cmb_case)
        f_adv.addRow("Extension", row_ext)

        # Smart trimming
        self.cb_spaces_to_underscore = QCheckBox("Replace spaces with _"); self.cb_spaces_to_underscore.setToolTip("Turn spaces into underscores.")
        self.cb_collapse_spaces = QCheckBox("Collapse whitespace to single space"); self.cb_collapse_spaces.setToolTip("Convert runs of spaces/tabs into a single space.")
        self.cb_collapse_underscores = QCheckBox("Collapse multiple _"); self.cb_collapse_underscores.setToolTip("Convert repeated underscores to a single _.")
        self.cb_remove_double_dots = QCheckBox("Remove double dots .."); self.cb_remove_double_dots.setToolTip("Replace any '..' sequences with a single '.'.")
        smart_box = QVBoxLayout()
        for w in (self.cb_spaces_to_underscore, self.cb_collapse_spaces, self.cb_collapse_underscores, self.cb_remove_double_dots):
            smart_box.addWidget(w)
        f_adv.addRow("Smart trimming", smart_box)

        # Pattern builder
        self.cb_use_pattern = QCheckBox("Use pattern"); self.cb_use_pattern.setToolTip("If enabled, build the new filename from the pattern below.")
        self.ed_pattern = QLineEdit("{name}")
        self.ed_pattern.setToolTip("Tokens: {name}, {ext}, {ext_with_dot}, {date}, {counter}. Example: {date}_{counter}_{name}{ext_with_dot}")
        f_adv.addRow(self.cb_use_pattern, self.ed_pattern)

        sec_adv.setContentLayout(f_adv)
        sec_adv.toggle.setToolTip("Advanced rename options. Closed by default to stay out of your way.")
        root.addWidget(sec_adv)

        # --- Actions
        row_act = QHBoxLayout()
        self.btn_preview = QPushButton("Preview")
        self.btn_preview.setToolTip("Dry run. Builds a full list without renaming, highlights conflicts, and updates the preview.")
        self.btn_export = QPushButton("Export plan (CSV)")
        self.btn_export.setToolTip("Save the current preview plan to CSV with columns old_path,new_path.")
        self.btn_apply   = QPushButton("Rename")
        self.btn_apply.setToolTip("Execute the rename plan. Conflicts are auto-suffixed (_1, _2, …) to avoid overwrites.")
        self.btn_undo    = QPushButton("Undo last")
        self.btn_undo.setToolTip("Revert the most recent rename using the plan stored in memory or .renam_last.json.")
        row_act.addWidget(self.btn_preview)
        row_act.addWidget(self.btn_export)
        row_act.addStretch(1)
        row_act.addWidget(self.btn_undo)
        row_act.addWidget(self.btn_apply)
        root.addLayout(row_act)

        # Wire up
        self.btn_pick_folder.clicked.connect(self._pick_folder)
        self.btn_pick_files.clicked.connect(self._pick_files)
        self.cmb_type.currentIndexChanged.connect(lambda *_: self._refresh_preview_names())
        self.cmb_sort.currentIndexChanged.connect(lambda *_: self._refresh_preview_names())
        self.cmb_order.currentIndexChanged.connect(lambda *_: self._refresh_preview_names())
        self.edit_filter.textChanged.connect(lambda *_: self._preview_rename())
        self.btn_preview.clicked.connect(self._preview_rename)
        self.btn_export.clicked.connect(self._export_plan)
        self.btn_apply.clicked.connect(self._apply_rename)
        self.btn_undo.clicked.connect(self._undo_last)

    # ---------- Helpers ----------
    def _current_base_dir(self) -> Path:
        try:
            if self.main and getattr(self.main, "current_path", None):
                p = getattr(self.main, "current_path", None)
                if isinstance(p, Path):
                    return p.parent
                try:
                    return Path(str(p)).parent
                except Exception:
                    pass
        except Exception:
            pass
        return Path(".").resolve()

    def _filter_paths(self, paths: List[str]) -> List[str]:
        t = self.cmb_type.currentText().lower()
        def _ok(p: str) -> bool:
            ext = Path(p).suffix.lower()
            if t == "images":
                return ext in IMG_EXTS
            if t == "videos":
                return ext in VID_EXTS
            return True
        return [p for p in paths if _ok(p)]

    def _sort_paths(self, paths: List[str]) -> List[str]:
        key = self.cmb_sort.currentText().lower()
        rev = (self.cmb_order.currentIndex() == 1)
        def _k(p: str):
            try:
                st = os.stat(p)
            except Exception:
                st = None
            if key.startswith("date"):
                return (st.st_mtime if st else 0.0, os.path.basename(p).lower())
            if key.startswith("size"):
                return (st.st_size if st else -1, os.path.basename(p).lower())
            return os.path.basename(p).lower()
        return sorted(paths, key=_k, reverse=rev)

    def _apply_filter_text(self, items: List[tuple]) -> List[tuple]:
        f = (self.edit_filter.text() or "").strip()
        if not f:
            return items
        try:
            if f.startswith("re/"):
                pat = re.compile(f[3:])
                return [it for it in items if pat.search(it[0]) or pat.search(it[1])]
            else:
                fl = f.lower()
                return [it for it in items if fl in it[0].lower() or fl in it[1].lower()]
        except Exception:
            return items

    def _refresh_preview_names(self):
        try:
            paths = self._filter_paths(self._paths)
            paths = self._sort_paths(paths)
            self.rename_preview.clear()
            for p in paths:
                self.rename_preview.addItem(os.path.basename(p))
        except Exception:
            pass

    def _pick_folder(self):
        base = str(self._current_base_dir())
        fn = QFileDialog.getExistingDirectory(self, "Choose folder", base)
        if not fn:
            return
        try:
            files = [str((Path(fn)/f)) for f in os.listdir(fn) if os.path.isfile(str(Path(fn)/f))]
        except Exception:
            files = []
        self._paths = files
        self._refresh_preview_names()

    def _pick_files(self):
        base = str(self._current_base_dir())
        files, _ = QFileDialog.getOpenFileNames(self, "Choose files", base, "All files (*.*)")
        if files:
            self._paths = [str(Path(p)) for p in files]
            self._refresh_preview_names()

    # ---------- Core formatting ----------
    def _date_for(self, path: str) -> str:
        src = self.cmb_date_source.currentText().lower()
        fmt = self.ed_date_fmt.text() or "%Y%m%d"
        dt = datetime.now()
        if src.startswith("modified"):
            try:
                dt = datetime.fromtimestamp(os.path.getmtime(path))
            except Exception:
                dt = datetime.now()
        try:
            return dt.strftime(fmt)
        except Exception:
            return dt.strftime("%Y%m%d")

    def _apply_search_replace(self, text: str, ext: str):
        find = self.ed_find.text()
        if not find:
            return text, ext
        repl = self.ed_repl.text() or ""
        flags = 0 if self.cb_case.isChecked() else re.IGNORECASE
        if self.cb_regex.isChecked():
            try:
                pat = re.compile(find, flags)
                text = pat.sub(repl, text)
                if self.cb_apply_ext.isChecked():
                    ext = pat.sub(repl, ext)
                return text, ext
            except Exception:
                return text, ext
        # simple replace
        def _rep(s: str) -> str:
            if self.cb_case.isChecked():
                return s.replace(find, repl)
            return re.sub(re.escape(find), repl, s, flags=re.IGNORECASE)
        text = _rep(text)
        if self.cb_apply_ext.isChecked():
            ext = _rep(ext)
        return text, ext

    def _case_convert(self, s: str) -> str:
        mode = self.cmb_case.currentText().lower()
        if mode == "lower":
            return s.lower()
        if mode == "upper":
            return s.upper()
        if mode.startswith("title"):
            parts = re.split(r"([_\-\s]+)", s)
            return "".join(p.title() if i%2==0 else p for i,p in enumerate(parts))
        return s

    def _smart_trim(self, s: str) -> str:
        if self.cb_collapse_spaces.isChecked():
            s = re.sub(r"\s+", " ", s)
        if self.cb_spaces_to_underscore.isChecked():
            s = s.replace(" ", "_")
        if self.cb_collapse_underscores.isChecked():
            s = re.sub(r"_+", "_", s)
        if self.cb_remove_double_dots.isChecked():
            s = re.sub(r"\.{2,}", ".", s)
        return s

    def _ext_after_mode(self, ext: str) -> str:
        mode = self.cmb_ext_mode.currentText().lower()
        if mode == "lowercase":
            return ext.lower()
        if mode == "uppercase":
            return ext.upper()
        if mode.startswith("change"):
            newe = (self.ed_ext_new.text() or "").lstrip(".")
            return f".{newe}" if newe else ext
        return ext

    def _counter_for_index(self, idx: int) -> str:
        if not self.cb_numbering.isChecked():
            return ""
        start = int(self.sp_num_start.value())
        step  = int(self.sp_num_step.value())
        pad   = int(self.sp_num_pad.value())
        val = start + idx*step
        return str(val).zfill(pad)

    def _build_new_name(self, path: str, idx: int) -> str:
        base = os.path.basename(path)
        stem, ext = os.path.splitext(base)

        # remove segments
        n1 = int(self.rm_start.value())
        n2 = int(self.rm_end.value())
        if n1 > 0: stem = stem[n1:]
        if n2 > 0: stem = stem[:-n2] if n2 < len(stem) else ""

        if self.cb_remove_leading_nums.isChecked():
            stem = re.sub(r"^\d+", "", stem)

        # search & replace
        stem, ext_no_dot = self._apply_search_replace(stem, ext.lstrip("."))
        ext = "." + ext_no_dot if ext_no_dot else ""

        # case + smart trims
        stem = self._case_convert(stem)
        stem = self._smart_trim(stem)
        ext  = self._ext_after_mode(ext)

        # date & counter
        date_txt = self._date_for(path) if self.cb_add_date.isChecked() else ""
        counter_txt = self._counter_for_index(idx)

        # numbering placement (if not full pattern replacement)
        if self.cb_numbering.isChecked() and self.cmb_num_pos.currentIndex() == 2:
            stem_final = counter_txt
        else:
            stem_final = stem
            if self.cb_numbering.isChecked():
                if self.cmb_num_pos.currentText() == "prefix":
                    stem_final = f"{counter_txt}{stem_final}"
                else:
                    stem_final = f"{stem_final}{counter_txt}"

        # prefixes/suffixes + date
        pref = (self.ed_prefix.text() or "")
        suf  = (self.ed_suffix.text() or "")
        if self.cb_add_date.isChecked():
            d = date_txt
            pref = (d + "_" + pref) if pref else (d + "_")

        # pattern builder override
        if self.cb_use_pattern.isChecked():
            pattern = self.ed_pattern.text() or "{name}"
            tokens = {
                "name": stem,
                "ext": ext.lstrip("."),
                "ext_with_dot": ext if ext else "",
                "date": date_txt,
                "counter": counter_txt
            }
            try:
                built = pattern.format(**tokens)
            except Exception:
                built = f"{stem}{ext}"
            if "{ext" in pattern:
                return built
            else:
                return f"{built}{ext}"

        return f"{pref}{stem_final}{suf}{ext}"

    def _build_plan(self) -> List[Tuple[str, str]]:
        paths = self._filter_paths(self._paths)
        paths = self._sort_paths(paths)
        plan: List[Tuple[str,str]] = []
        for i, p in enumerate(paths):
            newname = self._build_new_name(p, i)
            plan.append((p, str(Path(p).with_name(newname))))
        return plan

    def _preview_rename(self):
        plan = self._build_plan()

        display_items = [ (os.path.basename(old), os.path.basename(new)) for old,new in plan ]
        display_items = self._apply_filter_text(display_items)

        # Conflicts
        conflicts = set()
        targets_seen = {}
        for old, new in plan:
            base = str(Path(new))
            targets_seen[base] = targets_seen.get(base, 0) + 1
        for old, new in plan:
            dst = Path(new)
            if targets_seen[str(dst)] > 1 or (dst.exists() and str(dst) != str(old)):
                conflicts.add(old)

        self.rename_preview.clear()
        for old, new in plan:
            b_old = os.path.basename(old); b_new = os.path.basename(new)
            if (b_old, b_new) not in display_items:
                continue
            it = QListWidgetItem(f"{b_old}  →  {b_new}")
            if old in conflicts:
                it.setForeground(QBrush(Qt.red))
                it.setToolTip("Conflict: destination exists or duplicate in plan.")
            else:
                it.setToolTip("No conflict detected.")
            self.rename_preview.addItem(it)

        self._last_plan = plan

    def _export_plan(self):
        if not self._last_plan:
            self._preview_rename()
        if not self._last_plan:
            QMessageBox.information(self, "Export plan", "Nothing to export—build a preview first.")
            return
        fn, _ = QFileDialog.getSaveFileName(self, "Export rename plan (CSV)", "", "CSV files (*.csv)")
        if not fn:
            return
        try:
            with open(fn, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["old_path","new_path"])
                for old, new in self._last_plan:
                    w.writerow([old, new])
            QMessageBox.information(self, "Export plan", f"Saved: {fn}")
        except Exception as e:
            QMessageBox.critical(self, "Export error", str(e))

    def _save_last_plan_log(self, plan: List[Tuple[str,str]]):
        if not plan:
            self._last_plan_file = None
            return
        try:
            first_dir = Path(plan[0][0]).parent
            log_path = first_dir / ".renam_last.json"
            data = [{"old": o, "new": n} for o,n in plan]
            log_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._last_plan_file = log_path
        except Exception:
            self._last_plan_file = None

    def _apply_rename(self):
        plan = self._build_plan()
        if not plan:
            QMessageBox.information(self, "Multi Rename", "Pick files or a folder first.")
            return

        # conflicts warning
        conflicts = []
        for old, new in plan:
            dst = Path(new)
            if (dst.exists() and str(dst) != str(old)):
                conflicts.append((old,new))
        if conflicts:
            msg = f"{len(conflicts)} potential conflict(s) detected. Conflicting items will be suffixed with _1, _2, … to avoid overwriting.\nProceed?"
        else:
            msg = f"Rename {len(plan)} item(s)?"
        ok = QMessageBox.question(self, "Confirm rename", msg)
        if ok != QMessageBox.StandardButton.Yes:
            return

        applied = []
        for old, new in plan:
            try:
                src = Path(old)
                dst = Path(new)
                if src == dst:
                    continue
                if dst.exists():
                    i = 1
                    base_stem = dst.stem
                    base_ext  = dst.suffix
                    cand = dst.with_name(f"{base_stem}_{i}{base_ext}")
                    while cand.exists():
                        i += 1
                        cand = dst.with_name(f"{base_stem}_{i}{base_ext}")
                    dst = cand
                os.rename(str(src), str(dst))
                applied.append((str(src), str(dst)))
            except Exception:
                continue

        if not applied:
            QMessageBox.information(self, "Multi Rename", "No files were renamed.")
            return

        self._last_plan = applied
        self._save_last_plan_log(applied)
        QMessageBox.information(self, "Multi Rename", f"Renamed {len(applied)} item(s).")
        self._paths = [p for p,_ in applied]
        self._refresh_preview_names()

    def _undo_last(self):
        plan = self._last_plan[:]
        if not plan and self._last_plan_file and self._last_plan_file.exists():
            try:
                data = json.loads(self._last_plan_file.read_text(encoding="utf-8"))
                plan = [(d["old"], d["new"]) for d in data]
            except Exception:
                plan = []

        if not plan:
            QMessageBox.information(self, "Undo", "No previous rename plan found.")
            return

        reversed_plan = [(new, old) for (old, new) in plan]
        undone = 0
        for src_path, dst_path in reversed_plan:
            try:
                src = Path(src_path)
                dst = Path(dst_path)
                if not src.exists():
                    continue
                if dst.exists():
                    i = 1
                    cand = dst.with_name(f"{dst.stem}_restore_{i}{dst.suffix}")
                    while cand.exists():
                        i += 1
                        cand = dst.with_name(f"{dst.stem}_restore_{i}{dst.suffix}")
                    dst = cand
                os.rename(str(src), str(dst))
                undone += 1
            except Exception:
                continue

        QMessageBox.information(self, "Undo", f"Restored {undone} item(s).")
        self._refresh_preview_names()
