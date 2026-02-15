
import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QTabWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
    QTextEdit,
    QProgressBar,
)

# Base paths (assuming this file is in root/helpers/)
ROOT_DIR = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT_DIR / "presets" / "bin"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output" / "video" / "splitglue"


def get_ffmpeg_path() -> str:
    """Return the ffmpeg executable path from presets/bin."""
    candidates = ["ffmpeg.exe", "ffmpeg"]
    for name in candidates:
        candidate = BIN_DIR / name
        if candidate.exists():
            return str(candidate)
    # Fallback to just 'ffmpeg' in PATH
    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Return the ffprobe executable path from presets/bin."""
    candidates = ["ffprobe.exe", "ffprobe"]
    for name in candidates:
        candidate = BIN_DIR / name
        if candidate.exists():
            return str(candidate)
    return "ffprobe"


class FFmpegBatchWorker(QThread):
    progress = Signal(str, int, int)  # message, current, total
    finished = Signal(bool, str, list)  # success, message, output_files

    def __init__(self, commands, output_files, parent=None):
        super().__init__(parent)
        self.commands = commands
        self.output_files = output_files

    def run(self):
        total = len(self.commands)
        created = []
        for idx, cmd in enumerate(self.commands, start=1):
            self.progress.emit("Running command", idx, total)
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    shell=False,
                )
            except Exception as e:
                self.finished.emit(False, f"Error running ffmpeg: {e}", created)
                return

            if proc.returncode != 0:
                self.finished.emit(
                    False,
                    f"ffmpeg failed (step {idx}/{total}): {proc.stderr.strip()}",
                    created,
                )
                return

            if idx - 1 < len(self.output_files):
                created.append(self.output_files[idx - 1])

            self.progress.emit("Step completed", idx, total)

        self.finished.emit(True, "All operations completed successfully.", created)


def parse_timecode(value: str) -> float:
    """Parse a simple timecode like HH:MM:SS(.ms) or MM:SS into seconds."""
    value = value.strip()
    if not value:
        raise ValueError("Empty timecode")
    parts = value.split(":")
    parts = [p.strip() for p in parts]
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    raise ValueError(f"Invalid timecode format: {value}")


def seconds_to_timecode(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    total_min = total_sec // 60
    m = total_min % 60
    h = total_min // 60
    if ms:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


class SpliglueVideoTool(QWidget):
    """PySide6 tool to split and glue videos using ffmpeg."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Split & Glue Video Tool")
        self.ffmpeg_path = get_ffmpeg_path()
        self.ffprobe_path = get_ffprobe_path()

        self.split_input_path: Path | None = None
        self.split_output_dir: Path = DEFAULT_OUTPUT_DIR
        self.split_duration_seconds: float | None = None

        self.glue_output_dir: Path = DEFAULT_OUTPUT_DIR

        self.worker: FFmpegBatchWorker | None = None

        self._build_ui()

    # --------------------------- UI SETUP ---------------------------

    def _open_results_folder(self, folder: Path):
        """Open a folder in Media Explorer if available, else OS file manager."""
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        main = None
        try:
            main = getattr(self, "main", None)
        except Exception:
            main = None
        if main is None:
            try:
                main = self.window() if hasattr(self, "window") else None
            except Exception:
                main = None

        if main is not None and hasattr(main, "open_media_explorer_folder"):
            try:
                main.open_media_explorer_folder(str(folder), preset="videos", include_subfolders=False)
                return
            except TypeError:
                try:
                    main.open_media_explorer_folder(str(folder))
                    return
                except Exception:
                    pass
            except Exception:
                pass

        # Fallback: OS open
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception:
            pass

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        self._build_split_tab()
        self._build_glue_tab()

        # Simple progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Idle")
        layout.addWidget(self.progress_bar)

        # Log area
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        layout.addWidget(self.log)

    def _build_split_tab(self):
        tab = QWidget(self)
        v = QVBoxLayout(tab)

        # Input file row
        row1 = QHBoxLayout()
        self.split_input_edit = QLineEdit(tab)
        self.split_input_edit.setPlaceholderText("Select input video...")
        browse_in = QPushButton("Browse", tab)
        browse_in.clicked.connect(self._browse_split_input)

        row1.addWidget(QLabel("Input video:", tab))
        row1.addWidget(self.split_input_edit)
        row1.addWidget(browse_in)
        v.addLayout(row1)

        # Duration label
        self.duration_label = QLabel("Duration: unknown", tab)
        v.addWidget(self.duration_label)

        # Output folder row
        row2 = QHBoxLayout()
        self.split_output_edit = QLineEdit(tab)
        self.split_output_edit.setText(str(self.split_output_dir))
        browse_out = QPushButton("Browse", tab)
        browse_out.clicked.connect(self._browse_split_output)
        use_default = QPushButton("Use default", tab)
        use_default.clicked.connect(self._use_default_split_output)
        view_results = QPushButton("View results", tab)
        view_results.setToolTip("Open these results in Media Explorer.")
        view_results.clicked.connect(lambda: self._open_results_folder(Path(self.split_output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR))))

        row2.addWidget(QLabel("Output folder:", tab))
        row2.addWidget(self.split_output_edit)
        row2.addWidget(browse_out)
        row2.addWidget(use_default)
        row2.addWidget(view_results)
        row2.addWidget(view_results)
        v.addLayout(row2)

        # Auto-generate equal segments
        auto_row = QHBoxLayout()
        auto_row.addWidget(QLabel("Auto segments:", tab))
        self.auto_segments_edit = QLineEdit(tab)
        self.auto_segments_edit.setPlaceholderText("Number of equal segments")
        auto_btn = QPushButton("Generate equal segments", tab)
        auto_btn.clicked.connect(self._auto_generate_segments)
        auto_row.addWidget(self.auto_segments_edit)
        auto_row.addWidget(auto_btn)
        auto_row.addStretch()
        v.addLayout(auto_row)

        # Segments table
        self.segment_table = QTableWidget(0, 3, tab)
        self.segment_table.setHorizontalHeaderLabels(
            ["Start (HH:MM:SS)", "End (HH:MM:SS)", "Name suffix"]
        )
        self.segment_table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(QLabel("Define segments (at least 1 row):", tab))
        v.addWidget(self.segment_table)

        # Buttons for segments
        row3 = QHBoxLayout()
        add_row = QPushButton("Add row", tab)
        add_row.clicked.connect(self._add_segment_row)
        remove_row = QPushButton("Remove selected", tab)
        remove_row.clicked.connect(self._remove_segment_row)

        row3.addWidget(add_row)
        row3.addWidget(remove_row)
        row3.addStretch()
        v.addLayout(row3)

        # Split button
        row4 = QHBoxLayout()
        split_btn = QPushButton("Split video", tab)
        split_btn.clicked.connect(self._start_split)
        row4.addStretch()
        row4.addWidget(split_btn)
        v.addLayout(row4)

        self.tabs.addTab(tab, "Split")


    def _build_glue_tab(self):
        tab = QWidget(self)
        v = QVBoxLayout(tab)

        # List of videos
        v.addWidget(QLabel("Videos to glue (in order):", tab))
        self.glue_list = QListWidget(tab)

        v.addWidget(self.glue_list)

        # Controls for the list
        row1 = QHBoxLayout()
        add_btn = QPushButton("Add videos", tab)
        add_btn.clicked.connect(self._glue_add_files)
        remove_btn = QPushButton("Remove selected", tab)
        remove_btn.clicked.connect(self._glue_remove_selected)
        up_btn = QPushButton("Move up", tab)
        up_btn.clicked.connect(self._glue_move_up)
        down_btn = QPushButton("Move down", tab)
        down_btn.clicked.connect(self._glue_move_down)
        clear_btn = QPushButton("Clear", tab)
        clear_btn.clicked.connect(self.glue_list.clear)

        row1.addWidget(add_btn)
        row1.addWidget(remove_btn)
        row1.addWidget(up_btn)
        row1.addWidget(down_btn)
        row1.addWidget(clear_btn)
        v.addLayout(row1)

        # Output folder
        row2 = QHBoxLayout()
        self.glue_output_edit = QLineEdit(tab)
        self.glue_output_edit.setText(str(self.glue_output_dir))
        browse_out = QPushButton("Browse", tab)
        browse_out.clicked.connect(self._browse_glue_output)
        use_default = QPushButton("Use default", tab)
        use_default.clicked.connect(self._use_default_glue_output)
        view_results = QPushButton("View results", tab)
        view_results.setToolTip("Open these results in Media Explorer.")
        view_results.clicked.connect(lambda: self._open_results_folder(Path(self.glue_output_edit.text().strip() or str(DEFAULT_OUTPUT_DIR))))

        row2.addWidget(QLabel("Output folder:", tab))
        row2.addWidget(self.glue_output_edit)
        row2.addWidget(browse_out)
        row2.addWidget(use_default)
        row2.addWidget(view_results)
        v.addLayout(row2)

        # Output file name
        row3 = QHBoxLayout()
        self.glue_output_name_edit = QLineEdit(tab)
        self.glue_output_name_edit.setPlaceholderText("Output file name (without extension)")
        row3.addWidget(QLabel("Output name:", tab))
        row3.addWidget(self.glue_output_name_edit)
        v.addLayout(row3)

        # Glue button
        row4 = QHBoxLayout()
        glue_btn = QPushButton("Glue videos", tab)
        glue_btn.clicked.connect(self._start_glue)
        row4.addStretch()
        row4.addWidget(glue_btn)
        v.addLayout(row4)

        self.tabs.addTab(tab, "Glue")


    # --------------------------- Split logic ---------------------------

    def _browse_split_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select input video",
            str(ROOT_DIR),
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*.*)",
        )
        if path:
            self.split_input_path = Path(path)
            self.split_input_edit.setText(path)
            self._probe_duration()

    def _probe_duration(self):
        """Use ffprobe to get duration of the currently selected split input video."""
        self.split_duration_seconds = None
        if not self.split_input_path or not self.split_input_path.exists():
            self.duration_label.setText("Duration: unknown")
            return

        cmd = [
            self.ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(self.split_input_path),
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
            if proc.returncode != 0:
                self.duration_label.setText("Duration: unknown (ffprobe error)")
                return
            value = proc.stdout.strip()
            duration = float(value)
            self.split_duration_seconds = duration
            self.duration_label.setText(
                f"Duration: {seconds_to_timecode(duration)}  ({duration:.2f} s)"
            )
        except Exception:
            self.duration_label.setText("Duration: unknown (ffprobe error)")

    def _browse_split_output(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self.split_output_dir),
        )
        if path:
            self.split_output_dir = Path(path)
            self.split_output_edit.setText(path)

    def _use_default_split_output(self):
        self.split_output_dir = DEFAULT_OUTPUT_DIR
        self.split_output_edit.setText(str(self.split_output_dir))

    def _auto_generate_segments(self):
        """Auto-generate equal segments for the full duration."""
        if not self.split_input_path:
            QMessageBox.warning(
                self,
                "No input video",
                "Please select an input video first.",
            )
            return

        # Ensure we have duration
        if self.split_duration_seconds is None:
            self._probe_duration()
        if self.split_duration_seconds is None:
            QMessageBox.warning(
                self,
                "No duration",
                "Could not determine video duration with ffprobe.",
            )
            return

        text = self.auto_segments_edit.text().strip()
        if not text:
            QMessageBox.warning(
                self,
                "Missing value",
                "Please enter the number of equal segments.",
            )
            return

        try:
            count = int(text)
        except ValueError:
            QMessageBox.warning(
                self,
                "Invalid value",
                "Number of segments must be an integer.",
            )
            return

        if count <= 0:
            QMessageBox.warning(
                self,
                "Invalid value",
                "Number of segments must be at least 1.",
            )
            return

        duration = self.split_duration_seconds
        part_len = duration / count

        self.segment_table.setRowCount(0)
        for i in range(count):
            start = i * part_len
            if i == count - 1:
                end = duration
            else:
                end = (i + 1) * part_len

            row = self.segment_table.rowCount()
            self.segment_table.insertRow(row)
            start_item = QTableWidgetItem(seconds_to_timecode(start))
            end_item = QTableWidgetItem(seconds_to_timecode(end))
            suffix_item = QTableWidgetItem(f"part_{i+1}")
            self.segment_table.setItem(row, 0, start_item)
            self.segment_table.setItem(row, 1, end_item)
            self.segment_table.setItem(row, 2, suffix_item)

    def _add_segment_row(self):
        row = self.segment_table.rowCount()
        self.segment_table.insertRow(row)
        # Pre-fill suffix for convenience
        suffix_item = QTableWidgetItem(f"part_{row+1}")
        self.segment_table.setItem(row, 2, suffix_item)

    def _remove_segment_row(self):
        row = self.segment_table.currentRow()
        if row >= 0:
            self.segment_table.removeRow(row)

    def _start_split(self):
        if not self.split_input_edit.text().strip():
            QMessageBox.warning(self, "Missing input", "Please select an input video.")
            return

        input_path = Path(self.split_input_edit.text().strip())
        if not input_path.exists():
            QMessageBox.warning(self, "Invalid input", "Input video does not exist.")
            return

        if self.segment_table.rowCount() < 1:
            QMessageBox.warning(
                self,
                "No segments",
                "Please add at least one segment row.",
            )
            return

        out_dir = Path(self.split_output_edit.text().strip()) if self.split_output_edit.text().strip() else DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        self.split_output_dir = out_dir

        commands = []
        outputs = []

        stem = input_path.stem
        ext = input_path.suffix or ".mp4"

        for row in range(self.segment_table.rowCount()):
            start_item = self.segment_table.item(row, 0)
            end_item = self.segment_table.item(row, 1)
            suffix_item = self.segment_table.item(row, 2)

            start_text = start_item.text().strip() if start_item else ""
            end_text = end_item.text().strip() if end_item else ""
            suffix_text = suffix_item.text().strip() if suffix_item else f"part_{row+1}"

            if not start_text or not end_text:
                QMessageBox.warning(
                    self,
                    "Missing data",
                    f"Row {row+1}: please fill both start and end time.",
                )
                return

            try:
                start_seconds = parse_timecode(start_text)
                end_seconds = parse_timecode(end_text)
            except ValueError as e:
                QMessageBox.warning(self, "Invalid timecode", str(e))
                return

            if end_seconds <= start_seconds:
                QMessageBox.warning(
                    self,
                    "Invalid segment",
                    f"Row {row+1}: end must be greater than start.",
                )
                return

            duration = end_seconds - start_seconds
            out_file = out_dir / f"{stem}_{suffix_text}{ext}"
            cmd = [
                self.ffmpeg_path,
                "-y",
                "-ss",
                str(start_seconds),
                "-i",
                str(input_path),
                "-t",
                str(duration),
                "-c",
                "copy",
                str(out_file),
            ]
            commands.append(cmd)
            outputs.append(str(out_file))

        self._run_batch(commands, outputs, operation="split")


    # --------------------------- Glue logic ---------------------------

    def _browse_glue_output(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output folder",
            str(self.glue_output_dir),
        )
        if path:
            self.glue_output_dir = Path(path)
            self.glue_output_edit.setText(path)

    def _use_default_glue_output(self):
        self.glue_output_dir = DEFAULT_OUTPUT_DIR
        self.glue_output_edit.setText(str(self.glue_output_dir))

    def _glue_add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select videos to glue",
            str(ROOT_DIR),
            "Video files (*.mp4 *.mkv *.mov *.avi *.webm);;All files (*.*)",
        )
        for p in paths:
            if p:
                self.glue_list.addItem(p)

    def _glue_remove_selected(self):
        for item in self.glue_list.selectedItems():
            row = self.glue_list.row(item)
            self.glue_list.takeItem(row)

    def _glue_move_up(self):
        row = self.glue_list.currentRow()
        if row > 0:
            item = self.glue_list.takeItem(row)
            self.glue_list.insertItem(row - 1, item)
            self.glue_list.setCurrentRow(row - 1)

    def _glue_move_down(self):
        row = self.glue_list.currentRow()
        if row >= 0 and row < self.glue_list.count() - 1:
            item = self.glue_list.takeItem(row)
            self.glue_list.insertItem(row + 1, item)
            self.glue_list.setCurrentRow(row + 1)

    def _start_glue(self):
        if self.glue_list.count() < 2:
            QMessageBox.warning(
                self,
                "Not enough videos",
                "Please add at least two videos to glue.",
            )
            return

        paths = [self.glue_list.item(i).text().strip() for i in range(self.glue_list.count())]
        for p in paths:
            if not p or not Path(p).exists():
                QMessageBox.warning(
                    self,
                    "Invalid input",
                    "One or more selected videos do not exist.",
                )
                return

        out_dir = Path(self.glue_output_edit.text().strip()) if self.glue_output_edit.text().strip() else DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        self.glue_output_dir = out_dir

        name = self.glue_output_name_edit.text().strip()
        if not name:
            # Use name of first file with _glued suffix
            first = Path(paths[0])
            name = f"{first.stem}_glued"

        # Use extension from the first file
        ext = Path(paths[0]).suffix or ".mp4"
        output_file = out_dir / f"{name}{ext}"

        # Create a temporary concat list file
        concat_list_path = out_dir / f"_splitglue_concat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with concat_list_path.open("w", encoding="utf-8") as f:
            for p in paths:
                # ffmpeg concat demuxer expects paths in single quotes
                f.write(f"file '{Path(p).as_posix()}'\n")  # use forward slashes

        cmd = [
            self.ffmpeg_path,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list_path),
            "-c",
            "copy",
            str(output_file),
        ]

        self._run_batch([cmd], [str(output_file)], operation="glue", extra_files=[concat_list_path])


    # --------------------------- Batch & zip ---------------------------

    def _run_batch(self, commands, outputs, operation="", extra_files=None):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(
                self,
                "Busy",
                "A job is already running. Please wait until it finishes.",
            )
            return

        self._log(f"Starting {operation or 'ffmpeg'} job with {len(commands)} step(s)...")

        # Setup progress bar
        total = max(1, len(commands))
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(0)
        if operation:
            self.progress_bar.setFormat(f"{operation.capitalize()} %p%")
        else:
            self.progress_bar.setFormat("Working %p%")

        self.worker = FFmpegBatchWorker(commands, outputs, self)
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(lambda ok, msg, out: self._on_worker_finished(ok, msg, out, operation, extra_files))
        self.worker.start()

    def _on_worker_progress(self, message: str, current: int, total: int):
        self._log(f"{message}: {current}/{total}")
        # Update progress bar
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def _on_worker_finished(self, success: bool, message: str, output_files: list, operation: str, extra_files=None):
        self._log(message)
        # Reset progress bar to finished state
        if self.progress_bar.maximum() > 0:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_bar.setFormat("Done %p%")
        else:
            self.progress_bar.setFormat("Done")

        if extra_files:
            for p in extra_files:
                try:
                    if Path(p).exists():
                        Path(p).unlink()
                except Exception:
                    pass

        if not success:
            QMessageBox.critical(self, "Error", message)
            return

        if not output_files:
            QMessageBox.information(self, "Done", "No output files were created.")
            return

        # Report results (no zip creation)
        out_dir = Path(output_files[0]).parent
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        names = [Path(p).name for p in output_files]
        max_show = 10
        shown = names[:max_show]
        extra = len(names) - len(shown)

        lines = "\n".join(f"- {n}" for n in shown)
        if extra > 0:
            lines += f"\n... (+{extra} more)"

        self._log(f"Created {len(names)} file(s) in: {out_dir}")
        QMessageBox.information(
            self,
            "Done",
            f"Job finished successfully.\nCreated {len(names)} file(s) in:\n{out_dir}\n\n{lines}",
        )


    def _log(self, text: str):
        self.log.append(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SpliglueVideoTool()
    w.resize(900, 600)
    w.show()
    sys.exit(app.exec())
