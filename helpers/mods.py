
from __future__ import annotations
import os, json, subprocess
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
                               QLabel, QPushButton, QFileDialog, QMessageBox)

class ModelsPane(QWidget):
    """Standalone models placeholder panel extracted from the legacy UI.
    It shows the models manifest, lets you pick the models folder, and test the selected model (--help).
    """
    def __init__(self, main, paths: dict, parent=None):
        super().__init__(parent); self.main = main
        self.MODELS_DIR = Path(paths.get('MODELS_DIR'))
        self.MANIFEST_PATH = Path(paths.get('MANIFEST_PATH'))
        self.config = paths.get('config')

        v = QVBoxLayout(self)

        # Top: models folder chooser
        top = QHBoxLayout()
        self.lbl_folder = QLabel(f"Models folder: {self.config.get('models_folder', str(self.MODELS_DIR))}")
        self.btn_pick = QPushButton("Pick folder"); self.btn_pick.clicked.connect(self.pick_folder)
        top.addWidget(self.lbl_folder); top.addWidget(self.btn_pick)
        v.addLayout(top)

        # Table of models
        self.tbl = QTableWidget(0, 4, self)
        self.tbl.setHorizontalHeaderLabels(["Model","Exe","Exists","Last test"])
        v.addWidget(self.tbl)

        # Footer test button
        self.btn_test = QPushButton("Test selected (--help)")
        self.btn_test.clicked.connect(self.test_selected)
        v.addWidget(self.btn_test)

        self.refresh()

    def manifest(self) -> dict:
        try:
            return json.loads(self.MANIFEST_PATH.read_text(encoding='utf-8'))
        except Exception:
            return {}

    def pick_folder(self):
        new_dir = QFileDialog.getExistingDirectory(self, "Pick models folder", self.config.get('models_folder', str(self.MODELS_DIR)))
        if not new_dir: return
        self.config['models_folder'] = new_dir
        try:
            from helpers.framevision_app import save_config
            save_config()
        except Exception:
            pass
        self.lbl_folder.setText(f"Models folder: {new_dir}")
        self.refresh()

    def refresh(self):
        mani = self.manifest(); folder = Path(self.config.get('models_folder', str(self.MODELS_DIR)))
        rows = []
        for name, spec in mani.items():
            exe = spec.get('exe','')
            exists = (folder / exe).exists()
            rows.append((name, exe, exists, '—'))
        self.tbl.setRowCount(len(rows))
        for r,(name,exe,exists,last) in enumerate(rows):
            self.tbl.setItem(r,0,QTableWidgetItem(name))
            self.tbl.setItem(r,1,QTableWidgetItem(exe))
            self.tbl.setItem(r,2,QTableWidgetItem('Yes' if exists else 'No'))
            self.tbl.setItem(r,3,QTableWidgetItem(last))

    def test_selected(self):
        r = self.tbl.currentRow()
        if r < 0: 
            QMessageBox.information(self,"No selection","Select a model row to test."); 
            return
        name = self.tbl.item(r,0).text(); exe = self.tbl.item(r,1).text()
        folder = Path(self.config.get('models_folder', str(self.MODELS_DIR)))
        exe_path = folder / exe
        if not exe_path.exists():
            QMessageBox.warning(self,"Missing executable", f"{exe_path} not found."); return
        try:
            out = subprocess.run([str(exe_path), "--help"], capture_output=True, text=True, timeout=10)
            QMessageBox.information(self, f"{name} --help", out.stdout[:2000] or '(no output)')
        except Exception as e:
            QMessageBox.critical(self, "Test failed", str(e))
