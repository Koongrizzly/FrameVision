import os
import sys
import subprocess
import json
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QTextEdit, QFormLayout,
    QDateTimeEdit, QCheckBox, QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, QDateTime

from PIL import Image
import piexif


# --------------------------------------------------------------------------------------
# FFmpeg tool paths
# Looks for ffmpeg / ffprobe / ffplay in: <root folder>/presets/bin/
# where <root folder> is the folder containing this metadata.py
# --------------------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
BIN_DIR = ROOT_DIR / "presets" / "bin"

if os.name == "nt":
    FFMPEG_PATH = BIN_DIR / "ffmpeg.exe"
    FFPROBE_PATH = BIN_DIR / "ffprobe.exe"
    FFPLAY_PATH = BIN_DIR / "ffplay.exe"
else:
    FFMPEG_PATH = BIN_DIR / "ffmpeg"
    FFPROBE_PATH = BIN_DIR / "ffprobe"
    FFPLAY_PATH = BIN_DIR / "ffplay"


def human_readable_datetime(dt_str: str | None) -> QDateTime | None:
    if not dt_str:
        return None
    # EXIF DateTimeOriginal format: "YYYY:MM:DD HH:MM:SS"
    try:
        dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        return QDateTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
    except Exception:
        return None


class ImageMetadataTab(QWidget):
    def __init__(self, log_widget: QTextEdit, parent=None):
        super().__init__(parent)
        self.log_widget = log_widget
        self.current_path: Path | None = None

        main_layout = QVBoxLayout(self)

        # File selector
        file_row = QHBoxLayout()
        label = QLabel("Image file:")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_image)
        self.image_path_edit = QLineEdit()
        file_row.addWidget(label)
        file_row.addWidget(browse_btn)
        file_row.addWidget(self.image_path_edit, 1)
        main_layout.addLayout(file_row)

        # Metadata form
        form = QFormLayout()
        self.author_edit = QLineEdit()
        self.copyright_edit = QLineEdit()
        self.capture_date_edit = QDateTimeEdit()
        self.capture_date_edit.setCalendarPopup(True)
        self.capture_date_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.capture_date_edit.setSpecialValueText("Not set")
        self.capture_date_edit.setMinimumDateTime(QDateTime(1900, 1, 1, 0, 0, 0))

        form.addRow("Author:", self.author_edit)
        form.addRow("Copyright:", self.copyright_edit)
        form.addRow("Capture date:", self.capture_date_edit)
        main_layout.addLayout(form)

        # EXIF preview panel
        main_layout.addWidget(QLabel("EXIF preview (JSON-ish):"))
        self.preview_edit = QTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setPlaceholderText("EXIF metadata will be shown here after loading an image.")
        self.preview_edit.setMinimumHeight(140)
        main_layout.addWidget(self.preview_edit)

        # Overwrite checkbox
        self.overwrite_checkbox = QCheckBox("Overwrite original file (otherwise save as new file)")
        main_layout.addWidget(self.overwrite_checkbox)

        # Buttons
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load metadata")
        clear_exif_btn = QPushButton("Remove ALL EXIF (privacy clean)")
        batch_btn = QPushButton("Batch clean EXIF (folder)")
        save_btn = QPushButton("Save metadata")

        load_btn.clicked.connect(self.load_metadata)
        clear_exif_btn.clicked.connect(self.remove_exif)
        batch_btn.clicked.connect(self.run_batch_clean)
        save_btn.clicked.connect(self.save_metadata)

        btn_row.addWidget(load_btn)
        btn_row.addWidget(clear_exif_btn)
        btn_row.addWidget(batch_btn)
        btn_row.addWidget(save_btn)
        main_layout.addLayout(btn_row)

        main_layout.addStretch()

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select image", "", "Images (*.jpg *.jpeg *.png *.tif *.tiff);;All files (*)"
        )
        if file_path:
            self.image_path_edit.setText(file_path)
            self.current_path = Path(file_path)
            self.load_metadata()

    def log(self, message: str):
        self.log_widget.append(f"[Images] {message}")

    def _load_exif_dict(self, path: Path):
        with Image.open(path) as img:
            exif_bytes = img.info.get("exif")
            if not exif_bytes:
                return None
            try:
                exif_dict = piexif.load(exif_bytes)
                return exif_dict
            except Exception as e:
                self.log(f"Failed to parse EXIF: {e}")
                return None

    def update_preview(self, exif_dict):
        if not exif_dict:
            self.preview_edit.clear()
            self.preview_edit.setPlainText("No EXIF metadata found.")
            return

        def to_serializable(value):
            if isinstance(value, bytes):
                try:
                    return value.decode("utf-8", errors="ignore")
                except Exception:
                    return repr(value)
            if isinstance(value, (int, float, str)) or value is None:
                return value
            return repr(value)

        clean = {}
        for ifd_name, ifd in exif_dict.items():
            if ifd_name == "thumbnail":
                continue
            clean[ifd_name] = {}
            for tag_id, value in ifd.items():
                clean[ifd_name][str(tag_id)] = to_serializable(value)

        try:
            text = json.dumps(clean, indent=2, ensure_ascii=False)
        except Exception:
            text = str(clean)
        self.preview_edit.setPlainText(text)

    def load_metadata(self):
        if not self.image_path_edit.text():
            QMessageBox.warning(self, "No file", "Please select an image file first.")
            return
        path = Path(self.image_path_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "File not found", f"File does not exist:\n{path}")
            return

        self.current_path = path
        exif_dict = self._load_exif_dict(path)
        if exif_dict is None:
            self.log("No EXIF metadata found.")
            self.author_edit.clear()
            self.copyright_edit.clear()
            self.capture_date_edit.clear()
            self.update_preview(None)
            return

        artist = exif_dict.get("0th", {}).get(piexif.ImageIFD.Artist, b"")
        copyright_ = exif_dict.get("0th", {}).get(piexif.ImageIFD.Copyright, b"")
        dt_original = (
            exif_dict.get("Exif", {}).get(piexif.ExifIFD.DateTimeOriginal, b"") or
            exif_dict.get("0th", {}).get(piexif.ImageIFD.DateTime, b"")
        )

        if isinstance(artist, bytes):
            artist = artist.decode("utf-8", errors="ignore")
        if isinstance(copyright_, bytes):
            copyright_ = copyright_.decode("utf-8", errors="ignore")
        if isinstance(dt_original, bytes):
            dt_original = dt_original.decode("utf-8", errors="ignore")

        self.author_edit.setText(artist or "")
        self.copyright_edit.setText(copyright_ or "")

        qdt = human_readable_datetime(dt_original)
        if qdt is not None:
            self.capture_date_edit.setDateTime(qdt)
        else:
            self.capture_date_edit.clear()

        self.update_preview(exif_dict)
        self.log(f"Loaded EXIF metadata from {path.name}")

    def remove_exif(self):
        if not self.image_path_edit.text():
            QMessageBox.warning(self, "No file", "Please select an image file first.")
            return
        path = Path(self.image_path_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "File not found", f"File does not exist:\n{path}")
            return

        if self.overwrite_checkbox.isChecked():
            out_path = path
        else:
            out_path = path.with_name(path.stem + "_noexif" + path.suffix)

        try:
            with Image.open(path) as img:
                # Remove EXIF by saving without exif info
                data = list(img.getdata())
                img_without_exif = Image.new(img.mode, img.size)
                img_without_exif.putdata(data)
                img_without_exif.save(out_path)
            self.log(f"Removed EXIF metadata -> {out_path}")
            self.update_preview(None)
            QMessageBox.information(self, "Done", f"EXIF metadata removed.\nSaved to:\n{out_path}")
        except Exception as e:
            self.log(f"Error removing EXIF: {e}")
            QMessageBox.critical(self, "Error", f"Failed to remove EXIF:\n{e}")

    def save_metadata(self):
        if not self.image_path_edit.text():
            QMessageBox.warning(self, "No file", "Please select an image file first.")
            return
        path = Path(self.image_path_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "File not found", f"File does not exist:\n{path}")
            return

        if self.overwrite_checkbox.isChecked():
            out_path = path
        else:
            out_path = path.with_name(path.stem + "_meta" + path.suffix)

        author = self.author_edit.text().strip()
        copyright_ = self.copyright_edit.text().strip()

        if self.capture_date_edit.dateTime().isValid():
            dt = self.capture_date_edit.dateTime().toPython()
            date_str = dt.strftime("%Y:%m:%d %H:%M:%S")
        else:
            date_str = None

        try:
            with Image.open(path) as img:
                exif_bytes = img.info.get("exif")
                if exif_bytes:
                    exif_dict = piexif.load(exif_bytes)
                else:
                    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

                if author:
                    exif_dict["0th"][piexif.ImageIFD.Artist] = author.encode("utf-8", errors="ignore")
                else:
                    exif_dict["0th"].pop(piexif.ImageIFD.Artist, None)

                if copyright_:
                    exif_dict["0th"][piexif.ImageIFD.Copyright] = copyright_.encode("utf-8", errors="ignore")
                else:
                    exif_dict["0th"].pop(piexif.ImageIFD.Copyright, None)

                if date_str:
                    exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = date_str.encode("ascii")
                    exif_dict["0th"][piexif.ImageIFD.DateTime] = date_str.encode("ascii")
                else:
                    exif_dict["Exif"].pop(piexif.ExifIFD.DateTimeOriginal, None)
                    exif_dict["0th"].pop(piexif.ImageIFD.DateTime, None)

                exif_bytes_out = piexif.dump(exif_dict)
                img.save(out_path, exif=exif_bytes_out)

            # Refresh preview from the new file
            new_exif = self._load_exif_dict(out_path)
            self.update_preview(new_exif)
            self.log(f"Updated EXIF metadata -> {out_path}")
            QMessageBox.information(self, "Done", f"Metadata saved to:\n{out_path}")
        except Exception as e:
            self.log(f"Error saving metadata: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save metadata:\n{e}")

    def run_batch_clean(self):
        """
        Batch-remove EXIF from all supported images in a chosen folder.
        Respects the "Overwrite original file" checkbox.
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select folder for batch EXIF clean",
            "",
        )
        if not folder:
            return
        folder_path = Path(folder)

        patterns = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG",
                    "*.png", "*.PNG",
                    "*.tif", "*.tiff", "*.TIF", "*.TIFF")

        files: list[Path] = []
        for pattern in patterns:
            files.extend(folder_path.rglob(pattern))

        if not files:
            QMessageBox.information(
                self,
                "No images found",
                f"No supported images found in:\n{folder_path}",
            )
            return

        overwrite = self.overwrite_checkbox.isChecked()
        total = len(files)
        cleaned = 0
        errors = 0

        for path in files:
            if overwrite:
                out_path = path
            else:
                out_path = path.with_name(path.stem + "_noexif" + path.suffix)
            try:
                with Image.open(path) as img:
                    data = list(img.getdata())
                    img_without_exif = Image.new(img.mode, img.size)
                    img_without_exif.putdata(data)
                    img_without_exif.save(out_path)
                cleaned += 1
            except Exception as e:
                errors += 1
                self.log(f"[Batch] Error processing {path}: {e}")

        msg = f"Processed {total} images in:\n{folder_path}\n\nCleaned: {cleaned}"
        if errors:
            msg += f"\nErrors: {errors} (see log)."
        QMessageBox.information(self, "Batch EXIF clean done", msg)
        self.log(f"Batch EXIF clean finished: {cleaned} ok, {errors} errors in {folder_path}")


class VideoMetadataTab(QWidget):
    def __init__(self, log_widget: QTextEdit, parent=None):
        super().__init__(parent)
        self.log_widget = log_widget
        self.current_path: Path | None = None

        main_layout = QVBoxLayout(self)

        # File selector
        file_row = QHBoxLayout()
        label = QLabel("Video file:")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_video)
        self.video_path_edit = QLineEdit()
        file_row.addWidget(label)
        file_row.addWidget(browse_btn)
        file_row.addWidget(self.video_path_edit, 1)
        main_layout.addLayout(file_row)

        # Metadata form
        form = QFormLayout()
        self.title_edit = QLineEdit()
        self.comment_edit = QLineEdit()
        self.rotation_combo = QComboBox()
        self.rotation_combo.addItem("Keep as is", userData=None)
        for angle in (0, 90, 180, 270):
            self.rotation_combo.addItem(f"{angle}°", userData=angle)
        self.remove_location_checkbox = QCheckBox("Remove location metadata (if present)")

        form.addRow("Title:", self.title_edit)
        form.addRow("Comment:", self.comment_edit)
        form.addRow("Rotation flag:", self.rotation_combo)
        form.addRow("", self.remove_location_checkbox)
        main_layout.addLayout(form)

        # Metadata preview panel
        main_layout.addWidget(QLabel("Metadata preview (ffprobe JSON):"))
        self.preview_edit = QTextEdit()
        self.preview_edit.setReadOnly(True)
        self.preview_edit.setPlaceholderText("ffprobe metadata will be shown here after loading a video.")
        self.preview_edit.setMinimumHeight(140)
        main_layout.addWidget(self.preview_edit)

        # Overwrite checkbox
        self.overwrite_checkbox = QCheckBox("Overwrite original file (otherwise save as new file)")
        main_layout.addWidget(self.overwrite_checkbox)

        # Buttons
        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load metadata")
        save_btn = QPushButton("Save metadata")
        load_btn.clicked.connect(self.load_metadata)
        save_btn.clicked.connect(self.save_metadata)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(save_btn)
        main_layout.addLayout(btn_row)

        main_layout.addStretch()

    def log(self, message: str):
        self.log_widget.append(f"[Videos] {message}")

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Videos (*.mp4 *.mov *.mkv *.avi *.m4v *.webm);;All files (*)"
        )
        if file_path:
            self.video_path_edit.setText(file_path)
            self.current_path = Path(file_path)
            self.load_metadata()

    def _run_ffprobe(self, path: Path):
        if not FFPROBE_PATH.exists():
            QMessageBox.critical(
                self,
                "ffprobe not found",
                f"ffprobe was not found in:\n{FFPROBE_PATH}\n"
                f"Place ffprobe in presets/bin next to this app."
            )
            return None

        cmd = [
            str(FFPROBE_PATH), "-v", "quiet", "-print_format", "json",
            "-show_format", str(path)
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            data = json.loads(out.decode("utf-8", errors="ignore"))
            return data
        except subprocess.CalledProcessError as e:
            self.log(f"ffprobe error: {e.output.decode('utf-8', errors='ignore')}")
        except Exception as e:
            self.log(f"ffprobe failed: {e}")
        return None

    def load_metadata(self):
        if not self.video_path_edit.text():
            QMessageBox.warning(self, "No file", "Please select a video file first.")
            return
        path = Path(self.video_path_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "File not found", f"File does not exist:\n{path}")
            return

        self.current_path = path
        data = self._run_ffprobe(path)
        if not data:
            self.preview_edit.clear()
            self.preview_edit.setPlainText("No metadata found or ffprobe failed.")
            return

        # Fill basic fields
        tags = data.get("format", {}).get("tags", {})
        title = tags.get("title", "")
        comment = tags.get("comment", "") or tags.get("description", "")

        # Rotation can be in tags; we try tags first
        rotate_tag = tags.get("rotate")
        rotation_val = None
        try:
            if rotate_tag is not None:
                rotation_val = int(rotate_tag)
        except ValueError:
            rotation_val = None

        self.title_edit.setText(title)
        self.comment_edit.setText(comment)

        # set combo box
        if rotation_val in (0, 90, 180, 270):
            idx = self.rotation_combo.findData(rotation_val)
            if idx >= 0:
                self.rotation_combo.setCurrentIndex(idx)
            else:
                self.rotation_combo.setCurrentIndex(0)
        else:
            self.rotation_combo.setCurrentIndex(0)

        # Update preview panel with raw JSON
        try:
            preview_text = json.dumps(data, indent=2, ensure_ascii=False)
        except Exception:
            preview_text = str(data)
        self.preview_edit.setPlainText(preview_text)

        self.log(f"Loaded metadata from {path.name}")

    def _run_ffmpeg(self, args: list[str]):
        if not FFMPEG_PATH.exists():
            QMessageBox.critical(
                self,
                "ffmpeg not found",
                f"ffmpeg was not found in:\n{FFMPEG_PATH}\n"
                f"Place ffmpeg in presets/bin next to this app."
            )
            return False
        cmd = [str(FFMPEG_PATH)] + args
        try:
            subprocess.check_call(cmd)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"ffmpeg error: {e}")
            QMessageBox.critical(self, "ffmpeg error", f"ffmpeg failed with code {e.returncode}.")
        except Exception as e:
            self.log(f"ffmpeg failed: {e}")
            QMessageBox.critical(self, "Error", f"ffmpeg failed:\n{e}")
        return False

    def save_metadata(self):
        if not self.video_path_edit.text():
            QMessageBox.warning(self, "No file", "Please select a video file first.")
            return
        path = Path(self.video_path_edit.text())
        if not path.exists():
            QMessageBox.warning(self, "File not found", f"File does not exist:\n{path}")
            return

        if self.overwrite_checkbox.isChecked():
            out_path = path.with_suffix(path.suffix + ".tmp_meta")
            overwrite = True
        else:
            out_path = path.with_name(path.stem + "_meta" + path.suffix)
            overwrite = False

        title = self.title_edit.text().strip()
        comment = self.comment_edit.text().strip()
        removal_loc = self.remove_location_checkbox.isChecked()
        rotation_angle = self.rotation_combo.currentData()

        args = ["-y", "-i", str(path), "-map", "0", "-c", "copy"]

        # Metadata writes
        if title:
            args += ["-metadata", f"title={title}"]
        else:
            args += ["-metadata", "title="]

        if comment:
            args += ["-metadata", f"comment={comment}"]
        else:
            args += ["-metadata", "comment="]

        # rotation (for first video stream)
        if rotation_angle is not None:
            args += ["-metadata:s:v:0", f"rotate={rotation_angle}"]

        # location metadata removal: blank a few common fields
        if removal_loc:
            for key in ("location", "location-eng", "com.apple.quicktime.location.ISO6709"):
                args += ["-metadata", f"{key}="]

        args.append(str(out_path))

        self.log(f"Running ffmpeg to update metadata -> {out_path.name}")
        if not self._run_ffmpeg(args):
            return

        if overwrite:
            # replace original file
            try:
                Path(path).unlink()
                Path(out_path).rename(path)
                final_path = path
            except Exception as e:
                self.log(f"Failed to overwrite original: {e}")
                QMessageBox.warning(
                    self, "Overwrite failed",
                    f"Metadata updated but could not overwrite original.\n"
                    f"Updated file is at:\n{out_path}"
                )
                final_path = out_path
        else:
            final_path = out_path

        # Refresh preview using fresh ffprobe
        data = self._run_ffprobe(final_path)
        if data:
            try:
                preview_text = json.dumps(data, indent=2, ensure_ascii=False)
            except Exception:
                preview_text = str(data)
            self.preview_edit.setPlainText(preview_text)

        self.log(f"Updated video metadata -> {final_path}")
        QMessageBox.information(self, "Done", f"Metadata saved to:\n{final_path}")


class MetadataEditorWidget(QWidget):
    """
    Container widget that holds both the Image and Video metadata editors,
    plus a shared log panel. This is what you embed into your main app tab.
    """
    def __init__(self, parent=None, external_log_widget: QTextEdit | None = None):
        super().__init__(parent)

        main_layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.log_widget = external_log_widget or QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setPlaceholderText("Log output...")

        self.image_tab = ImageMetadataTab(self.log_widget, parent=self)
        self.video_tab = VideoMetadataTab(self.log_widget, parent=self)

        self.tabs.addTab(self.image_tab, "Images")
        self.tabs.addTab(self.video_tab, "Videos")


        # Optional: load date_changer.py as an extra tab.
        # Note: Qt uses '&' for mnemonics; '&&' renders a literal '&' in the label.
        date_widget = self._try_load_date_changer_tab()
        if date_widget is not None:
            self.tabs.addTab(date_widget, "change date && time")

        main_layout.addWidget(self.tabs)
        main_layout.addWidget(QLabel("Log:"))
        main_layout.addWidget(self.log_widget, stretch=1)
    def _try_load_date_changer_tab(self) -> QWidget | None:
        """
        Loads date_changer.py (sitting next to this metadata file) and embeds its UI as a QWidget tab.
        We support:
          - DateChangerWidget / DateChangerTab / DateChangerPane (QWidget)
          - build_widget() factory returning QWidget
          - MainWindow (QMainWindow) hosted as a child widget
        """
        try:
            candidates = [
                ROOT_DIR / "date_changer.py",
                ROOT_DIR.parent / "date_changer.py",
            ]
            candidate = next((p for p in candidates if p.exists()), None)
            if candidate is None:
                raise FileNotFoundError("Missing file: date_changer.py (looked in helpers/ and project root)")

            spec = importlib.util.spec_from_file_location("date_changer_embedded", str(candidate))
            if spec is None or spec.loader is None:
                raise ImportError("Could not create import spec for date_changer.py")

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore

            # 1) Preferred: a dedicated QWidget class
            for cls_name in ("DateChangerWidget", "DateChangerTab", "DateChangerPane"):
                if hasattr(mod, cls_name):
                    cls = getattr(mod, cls_name)
                    w = cls()  # noqa
                    if isinstance(w, QWidget):
                        return w

            # 2) Factory function
            if hasattr(mod, "build_widget"):
                w = mod.build_widget()  # type: ignore
                if isinstance(w, QWidget):
                    return w

            # 3) Fallback: host the provided MainWindow as a child widget
            if hasattr(mod, "MainWindow"):
                mw = mod.MainWindow()  # type: ignore
                if isinstance(mw, QWidget):
                    mw.setWindowFlags(Qt.Widget)
                    container = QWidget(self)
                    lay = QVBoxLayout(container)
                    lay.setContentsMargins(0, 0, 0, 0)
                    lay.addWidget(mw)

                    # Keep a ref so it doesn't get GC'd
                    self._date_changer_window = mw  # type: ignore[attr-defined]
                    return container

            raise ImportError("date_changer.py loaded, but no embeddable widget was found.")

        except Exception:
            tb = traceback.format_exc()
            try:
                self.log_widget.append("---- date_changer tab load error ----")
                self.log_widget.append(tb)
            except Exception:
                pass

            w = QWidget(self)
            lay = QVBoxLayout(w)
            lay.setContentsMargins(16, 16, 16, 16)
            msg = QLabel("❌ Failed to load date_changer.py (missing file or import error).\n\nSee Log for details.")
            msg.setWordWrap(True)
            lay.addWidget(msg)
            lay.addStretch(1)
            return w


class MetadataEditorMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metadata Editor (Images & Videos)")
        self.resize(900, 700)

        central = MetadataEditorWidget(self)
        self.setCentralWidget(central)


def main():
    app = QApplication(sys.argv)
    win = MetadataEditorMainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
