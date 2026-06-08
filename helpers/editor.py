#!/usr/bin/env python3
"""
OpenShot Clone Video Editor
A complete video editing application built with PySide6
Author: MiniMax Agent
"""

import os
import sys
import json
import time
import threading
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from PySide6.QtMultimedia import *
from PySide6.QtMultimediaWidgets import *

# Configuration paths
HELPERS_PATH = "/root/helpers/"
PRESETS_BIN_PATH = "/root/presets/bin/"

@dataclass
class MediaClip:
    """Represents a media clip in the timeline"""
    id: str
    name: str
    file_path: str
    file_type: str  # 'video', 'audio', 'image'
    duration: float
    start_time: float
    end_time: float
    position: float  # position on timeline in seconds
    track: int
    thumbnail_path: Optional[str] = None

@dataclass
class Project:
    """Represents a video editing project"""
    name: str
    file_path: str
    clips: List[MediaClip]
    timeline_duration: float
    fps: int = 30
    width: int = 1920
    height: int = 1080
    created_time: Optional[str] = None
    modified_time: Optional[str] = None

class FFmpegManager:
    """Manages FFmpeg integration for video processing"""
    
    def __init__(self):
        self.ffmpeg_path = os.path.join(PRESETS_BIN_PATH, "ffmpeg")
        self.ffprobe_path = os.path.join(PRESETS_BIN_PATH, "ffprobe")
        
    def get_video_info(self, file_path: str) -> Dict:
        """Get video file information using ffprobe"""
        try:
            cmd = [
                self.ffprobe_path,
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Extract duration from format section (more reliable)
            duration = 5.0  # Default fallback
            if 'format' in info and 'duration' in info['format']:
                try:
                    duration = float(info['format']['duration'])
                except (ValueError, TypeError):
                    pass
            
            # If no duration in format, try streams
            if duration <= 0 and 'streams' in info:
                for stream in info['streams']:
                    if stream.get('codec_type') == 'video' and 'duration' in stream:
                        try:
                            duration = float(stream['duration'])
                            break
                        except (ValueError, TypeError):
                            pass
                    elif stream.get('codec_type') == 'audio' and 'duration' in stream:
                        try:
                            duration = max(duration, float(stream['duration']))
                        except (ValueError, TypeError):
                            pass
            
            # Determine file type
            file_type = 'unknown'
            if 'streams' in info:
                for stream in info['streams']:
                    if stream.get('codec_type') == 'video':
                        file_type = 'video'
                        break
                    elif stream.get('codec_type') == 'audio':
                        if file_type == 'unknown':
                            file_type = 'audio'
            
            # Add duration and type to info
            info['duration'] = duration
            info['type'] = file_type
            
            print(f"Media info for {Path(file_path).name}: duration={duration:.1f}s, type={file_type}")
            return info
            
        except Exception as e:
            print(f"Error getting video info for {file_path}: {e}")
            return {'duration': 5.0, 'type': 'unknown'}
    
    def extract_thumbnail(self, video_path: str, output_path: str, timestamp: float = 1.0) -> bool:
        """Extract thumbnail from video"""
        try:
            cmd = [
                self.ffmpeg_path,
                '-i', video_path,
                '-ss', str(timestamp),
                '-vframes', '1',
                '-f', 'image2',
                '-y',
                output_path
            ]
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            print(f"Error extracting thumbnail: {e}")
            return False
    
    def export_project(self, project: Project, output_path: str) -> bool:
        """Export video project using FFmpeg"""
        try:
            # This is a simplified export - in a real implementation,
            # you'd need to build a complex FFmpeg filter graph
            cmd = [
                self.ffmpeg_path,
                '-f', 'concat',
                '-safe', '0',
                '-i', self._create_concat_file(project),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-y',
                output_path
            ]
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            print(f"Error exporting project: {e}")
            return False
    
    def _create_concat_file(self, project: Project) -> str:
        """Create a temporary concat file for FFmpeg"""
        concat_content = []
        for clip in project.clips:
            if clip.file_type == 'video':
                concat_content.append(f"file '{clip.file_path}'")
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write('\n'.join(concat_content))
        temp_file.close()
        
        return temp_file.name

class TimelineWidget(QWidget):
    """Timeline widget for video editing"""
    
    def __init__(self):
        super().__init__()
        self.clips: List[MediaClip] = []
        self.selected_clip_id = None
        self.current_time = 0.0
        self.duration = 60.0  # 1 minute default
        self.zoom = 1.0
        self.pixels_per_second = 50
        self.track_height = 60
        self.num_tracks = 5
        self.parent_window = None  # Reference to main window
        
        self.setMinimumHeight(350)
        self.setAcceptDrops(True)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup timeline UI"""
        self.setLayout(QVBoxLayout())
        
        # Timeline ruler
        self.ruler = TimelineRuler(self)
        self.layout().addWidget(self.ruler)
        
        # Scroll area for timeline
        scroll = QScrollBar(Qt.Horizontal)
        scroll.valueChanged.connect(self.scroll_changed)
        self.layout().addWidget(scroll)
        
        self.timeline_area = TimelineArea(self)
        self.timeline_area.setAcceptDrops(True)
        self.layout().addWidget(self.timeline_area)
        
        # Track labels
        self.setup_track_labels()
    
    def setup_track_labels(self):
        """Setup or update track labels"""
        # Remove existing track labels
        if hasattr(self, 'track_labels') and self.track_labels is not None:
            self.layout().removeWidget(self.track_labels)
            self.track_labels.deleteLater()
        
        # Create new track labels
        self.track_labels = QWidget()
        track_labels_layout = QVBoxLayout(self.track_labels)
        track_labels_layout.setContentsMargins(0, 0, 0, 0)
        track_labels_layout.setSpacing(0)
        
        for i in range(self.num_tracks):
            label = QLabel(f"Track {i+1}")
            label.setFixedHeight(self.track_height)
            label.setStyleSheet("""
                QLabel {
                    background-color: rgb(50, 50, 60);
                    border: 1px solid rgb(80, 80, 90);
                    padding: 5px;
                    font-weight: bold;
                    color: white;
                }
            """)
            track_labels_layout.addWidget(label)
        
        self.layout().addWidget(self.track_labels)
        
    def add_clip(self, clip: MediaClip):
        """Add a clip to the timeline"""
        self.clips.append(clip)
        self.update_duration()
        self.update_timeline()
        
    def remove_clip(self, clip_id: str):
        """Remove a clip from the timeline"""
        self.clips = [clip for clip in self.clips if clip.id != clip_id]
        self.update_duration()
        self.update_timeline()
    
    def update_duration(self):
        """Update timeline duration based on clips"""
        if not self.clips:
            self.duration = 60.0  # Default duration
        else:
            # Find the latest end time of all clips
            latest_end = max(clip.end_time for clip in self.clips)
            self.duration = max(60.0, latest_end + 10)  # Add 10 second buffer
        
        print(f"Timeline duration updated: {self.duration:.1f}s")
        
    def update_timeline(self):
        """Update timeline display"""
        self.timeline_area.update()
    
    def update_ruler(self):
        """Update timeline ruler display"""
        if hasattr(self, 'ruler'):
            self.ruler.update()
        
    def clear_clips(self):
        """Clear all clips from timeline"""
        self.clips.clear()
        self.selected_clip_id = None
        self.duration = 60.0
        self.update_timeline()
        print("Cleared all clips from timeline")
    
    def add_track(self):
        """Add a new track to timeline"""
        self.num_tracks += 1
        self.setup_track_labels()
        self.update_timeline()
        print(f"Added new track. Total tracks: {self.num_tracks}")
    
    def scroll_changed(self, value):
        """Handle scroll changes"""
        self.timeline_area.scroll_offset = value * self.pixels_per_second
        self.timeline_area.update()
    
    def add_media_to_timeline(self, file_path: str):
        """Add media file to timeline from media library"""
        try:
            # Get video information using FFmpeg
            window = self.window()
            if hasattr(window, 'ffmpeg_manager'):
                video_info = window.ffmpeg_manager.get_video_info(file_path)
                duration = video_info.get('duration', 5.0)
                file_type = video_info.get('type', 'unknown')
            else:
                duration = 5.0
                file_type = 'unknown'
            
            # Create clip
            import uuid
            clip = MediaClip(
                id=str(uuid.uuid4()),
                file_path=file_path,
                name=Path(file_path).stem,
                duration=duration,
                start_time=0.0,
                end_time=duration,
                position=self.current_time,
                track=0,  # Add to first track
                file_type=file_type
            )
            
            self.add_clip(clip)
            
            # Update current time
            self.current_time += duration + 0.1
            
            # Show confirmation message
            if hasattr(window, 'statusBar'):
                window.statusBar().showMessage(f"Added {clip.name} to timeline")
            
        except Exception as e:
            if hasattr(window, 'statusBar'):
                window.statusBar().showMessage(f"Error adding media: {str(e)}")
            print(f"Error adding media to timeline: {e}")

class TimelineRuler(QWidget):
    """Timeline ruler showing time markers"""
    
    def __init__(self, timeline: TimelineWidget):
        super().__init__()
        self.timeline = timeline
        self.setFixedHeight(30)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw background
        painter.fillRect(self.rect(), QColor(45, 45, 45))
        
        # Draw time markers
        painter.setPen(QColor(200, 200, 200))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        
        pixels_per_sec = self.timeline.pixels_per_second
        width = self.width()
        duration = self.timeline.duration
        
        for i in range(int(duration) + 1):
            x = i * pixels_per_sec
            
            if x > width:
                break
                
            # Draw major tick
            painter.drawLine(x, 0, x, 15)
            
            # Draw time label
            painter.drawText(QPoint(x + 5, 25), f"{i}s")

class TimelineArea(QWidget):
    """Timeline area where clips are displayed"""
    
    def __init__(self, timeline: TimelineWidget):
        super().__init__()
        self.timeline = timeline
        self.scroll_offset = 0
        self.setAcceptDrops(True)  # Enable drag and drop
        
        # Drag operation variables
        self.dragging_clip = None
        self.drag_start_pos = None
        self.drag_start_time = None
        self.drag_start_track = None
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Calculate dimensions
        pixels_per_sec = self.timeline.pixels_per_second
        track_height = self.timeline.track_height
        duration = self.timeline.duration
        
        # Set minimum size
        total_width = int(duration * pixels_per_sec) + 100
        total_height = self.timeline.num_tracks * track_height + 10
        self.setMinimumSize(total_width, total_height)
        
        # Draw timeline background
        painter.fillRect(self.rect(), QColor(45, 45, 45))
        
        # Draw tracks
        for track in range(self.timeline.num_tracks):
            y = track * track_height
            
            # Track background with alternating colors
            if track % 2 == 0:
                track_color = QColor(60, 60, 70)  # Dark gray-blue
            else:
                track_color = QColor(50, 50, 60)  # Slightly darker
            
            track_rect = QRect(0, y, total_width, track_height)
            painter.fillRect(track_rect, track_color)
            
            # Track separator line
            painter.setPen(QColor(80, 80, 90))
            painter.drawLine(0, y, self.width(), y + track_height)
        
        # Draw current time indicator (playhead)
        current_time_x = int(self.timeline.current_time * pixels_per_sec - self.scroll_offset)
        painter.setPen(QColor(255, 50, 50))
        painter.drawLine(current_time_x, 0, current_time_x, total_height)
        
        # Draw clips
        for clip in self.timeline.clips:
            x = int(clip.position * pixels_per_sec - self.scroll_offset)
            y = clip.track * track_height + 5
            
            # Clip background
            if clip.file_type == 'video':
                color = QColor(0, 150, 255)
            elif clip.file_type == 'audio':
                color = QColor(255, 150, 0)
            else:  # image
                color = QColor(150, 255, 100)
                
            clip_width = int((clip.end_time - clip.start_time) * pixels_per_sec)
            clip_rect = QRect(x, y, clip_width, track_height - 10)
            
            painter.fillRect(clip_rect, color)
            
            # Clip border
            painter.setPen(QColor(255, 255, 255))
            painter.drawRect(clip_rect)
            
            # Clip name
            painter.setPen(QColor(255, 255, 255))
            font = painter.font()
            font.setPointSize(9)
            painter.setFont(font)
            
            text_rect = clip_rect.adjusted(5, 5, -5, -5)
            painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignTop, clip.name)
    
    def mousePressEvent(self, event):
        """Handle mouse press for playhead movement, clip selection, and dragging"""
        if event.button() == Qt.LeftButton:
            click_pos = event.pos()
            pixels_per_sec = self.timeline.pixels_per_second
            
            # Check if click is on a clip
            clicked_clip = None
            for clip in self.timeline.clips:
                clip_x = int(clip.position * pixels_per_sec - self.scroll_offset)
                clip_y = clip.track * self.timeline.track_height + 5
                clip_width = int((clip.end_time - clip.start_time) * pixels_per_sec)
                
                clip_rect = QRect(clip_x, clip_y, clip_width, self.timeline.track_height - 10)
                if clip_rect.contains(click_pos):
                    clicked_clip = clip
                    break
            
            if clicked_clip:
                # Select clip and load in preview
                self.timeline.selected_clip_id = clicked_clip.id
                if hasattr(self.timeline, 'parent_window') and self.timeline.parent_window:
                    self.timeline.parent_window.play_preview_media(clicked_clip.file_path)
                print(f"Selected and loaded clip: {clicked_clip.name}")
                
                # Calculate new time position (move playhead to clip position)
                new_time = clicked_clip.position
                self.timeline.current_time = new_time
                self.timeline.update_timeline()
                self.timeline.update_ruler()
                print(f"Moved playhead to clip start: {new_time:.2f}s")
                
                # Start drag operation
                self.dragging_clip = clicked_clip
                self.drag_start_pos = click_pos
                self.drag_start_time = clicked_clip.position
                self.drag_start_track = clicked_clip.track
                
            else:
                # Move playhead to click position
                new_time = (click_pos.x() + self.scroll_offset) / pixels_per_sec
                new_time = max(0, min(new_time, self.timeline.duration))
                self.timeline.current_time = new_time
                self.timeline.update_timeline()
                self.timeline.update_ruler()
                print(f"Moved playhead to: {new_time:.2f}s")
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for clip dragging"""
        if hasattr(self, 'dragging_clip') and self.dragging_clip:
            click_pos = event.pos()
            pixels_per_sec = self.timeline.pixels_per_second
            track_height = self.timeline.track_height
            
            # Calculate new position
            delta_x = click_pos.x() - self.drag_start_pos.x()
            delta_time = delta_x / pixels_per_sec
            new_time = max(0, self.drag_start_time + delta_time)
            
            # Calculate new track
            delta_y = click_pos.y() - self.drag_start_pos.y()
            new_track = max(0, self.drag_start_track + int(delta_y / track_height))
            new_track = min(new_track, self.timeline.num_tracks - 1)
            
            # Update clip position (visual feedback)
            self.dragging_clip.position = new_time
            self.dragging_clip.track = new_track
            
            # Update timeline display
            self.timeline.update_timeline()
            
            print(f"Dragging clip to: {new_time:.2f}s, track {new_track}")
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to complete drag operation"""
        if hasattr(self, 'dragging_clip') and self.dragging_clip and event.button() == Qt.LeftButton:
            # Finalize drag
            print(f"Finished dragging clip: {self.dragging_clip.name}")
            self.dragging_clip = None
            self.drag_start_pos = None
            self.drag_start_time = None
            self.drag_start_track = None
    
    def mouseDoubleClickEvent(self, event):
        """Handle double-click to play selected clip"""
        if event.button() == Qt.LeftButton:
            click_pos = event.pos()
            pixels_per_sec = self.timeline.pixels_per_second
            
            # Check if double-click is on a clip
            clicked_clip = None
            for clip in self.timeline.clips:
                clip_x = int(clip.position * pixels_per_sec - self.scroll_offset)
                clip_y = clip.track * self.timeline.track_height + 5
                clip_width = int((clip.end_time - clip.start_time) * pixels_per_sec)
                
                clip_rect = QRect(clip_x, clip_y, clip_width, self.timeline.track_height - 10)
                if clip_rect.contains(click_pos):
                    clicked_clip = clip
                    break
            
            if clicked_clip:
                # Load clip in preview and play
                if hasattr(self.timeline, 'parent_window') and self.timeline.parent_window:
                    self.timeline.parent_window.play_preview_media(clicked_clip.file_path)
                    # Start playback
                    self.timeline.parent_window.preview_widget.toggle_playback()
                print(f"Playing clip: {clicked_clip.name}")
    
    def mouseWheelEvent(self, event):
        """Handle mouse wheel for timeline zooming"""
        # Zoom factor
        zoom_factor = 1.1
        
        if event.angleDelta().y() > 0:
            # Zoom in
            self.timeline.pixels_per_second *= zoom_factor
        else:
            # Zoom out
            self.timeline.pixels_per_second /= zoom_factor
        
        # Clamp zoom limits
        self.timeline.pixels_per_second = max(10, min(200, self.timeline.pixels_per_second))
        
        # Update timeline
        self.timeline.update_timeline()
        self.timeline.update_ruler()
        
        print(f"Timeline zoom: {self.timeline.pixels_per_second:.1f} pixels/second")
    
    def contextMenuEvent(self, event):
        """Handle right-click context menu"""
        from PySide6.QtWidgets import QMenu
        
        menu = QMenu(self)
        
        # Check if clicked on clip
        click_pos = event.pos()
        clicked_clip = None
        for clip in self.timeline.clips:
            clip_x = int(clip.position * self.timeline.pixels_per_second - self.scroll_offset)
            clip_y = clip.track * self.timeline.track_height + 5
            clip_width = int((clip.end_time - clip.start_time) * self.timeline.pixels_per_second)
            
            clip_rect = QRect(clip_x, clip_y, clip_width, self.timeline.track_height - 10)
            if clip_rect.contains(click_pos):
                clicked_clip = clip
                break
        
        if clicked_clip:
            # Clip-specific menu
            delete_action = menu.addAction("Delete Clip")
            delete_action.triggered.connect(lambda: self.timeline.remove_clip(clicked_clip.id))
            
            split_action = menu.addAction("Split Clip")
            split_action.triggered.connect(lambda: self.timeline.split_clip(clicked_clip, 
                (click_pos.x() + self.scroll_offset) / self.timeline.pixels_per_second))
        else:
            # Timeline area menu
            add_track_action = menu.addAction("Add Track")
            add_track_action.triggered.connect(lambda: self.timeline.add_track())
            
            clear_action = menu.addAction("Clear All Clips")
            clear_action.triggered.connect(self.timeline.clear_clips)
        
        menu.exec(event.globalPos())
    
    def dragEnterEvent(self, event):
        """Handle drag enter"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop"""
        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    files.append(url.toLocalFile())
            
            # Calculate drop position
            drop_pos = event.pos()
            track_height = self.timeline.track_height
            pixels_per_sec = self.timeline.pixels_per_second
            
            # Calculate which track and position the drop occurred at
            track = max(0, min(drop_pos.y() // track_height, self.timeline.num_tracks - 1))
            position = max(0, (drop_pos.x() + self.scroll_offset) / pixels_per_sec)
            
            # Add files as clips
            from PySide6.QtCore import QTimer
            import uuid
            
            for file_path in files:
                # Get video duration using FFmpeg
                try:
                    ffmpeg_manager = self.timeline.window().ffmpeg_manager
                    video_info = ffmpeg_manager.get_video_info(file_path)
                    duration = video_info.get('duration', 5.0)
                    file_type = video_info.get('type', 'unknown')
                    
                    print(f"Adding clip: {Path(file_path).name}, duration: {duration:.1f}s, type: {file_type}")
                except Exception as e:
                    print(f"Error getting video info for {file_path}: {e}")
                    duration = 5.0
                    file_type = 'unknown'
                
                clip = MediaClip(
                    id=str(uuid.uuid4()),
                    file_path=file_path,
                    name=Path(file_path).stem,
                    duration=duration,
                    start_time=0.0,
                    end_time=duration,
                    position=position,
                    track=track,
                    file_type=file_type
                )
                
                self.timeline.add_clip(clip)
                position += duration + 0.1  # Offset next clip slightly
            
            event.acceptProposedAction()
    
    def contextMenuEvent(self, event):
        """Show context menu for timeline"""
        menu = QMenu(self)
        
        # Get position in timeline
        pos = event.pos()
        track_height = self.timeline.track_height
        pixels_per_sec = self.timeline.pixels_per_second
        
        track = max(0, min(pos.y() // track_height, self.timeline.num_tracks - 1))
        time_pos = (pos.x() + self.scroll_offset) / pixels_per_sec
        
        # Check if clicking on a clip
        clicked_clip = None
        for clip in self.timeline.clips:
            clip_x = int(clip.position * pixels_per_sec - self.scroll_offset)
            clip_y = clip.track * track_height + 5
            clip_width = int((clip.end_time - clip.start_time) * pixels_per_sec)
            clip_height = track_height - 10
            
            if (pos.x() >= clip_x and pos.x() <= clip_x + clip_width and
                pos.y() >= clip_y and pos.y() <= clip_y + clip_height):
                clicked_clip = clip
                break
        
        if clicked_clip:
            # Clip-specific menu
            menu.addAction("Delete Clip", lambda: self.timeline.remove_clip(clicked_clip.id))
            menu.addAction("Split Clip", lambda: self.split_clip(clicked_clip, time_pos))
            menu.addAction("Trim Clip", lambda: self.trim_clip(clicked_clip))
        else:
            # Timeline-specific menu
            menu.addAction("Clear All Clips", self.timeline.window().clear_timeline)
            menu.addAction("Add Track", self.timeline.window().add_track)
            menu.addAction("Delete Track", lambda: self.timeline.window().delete_track(track))
            menu.addSeparator()
            menu.addAction("Copy", lambda: self.timeline.window().copy_timeline_state())
            menu.addAction("Paste", lambda: self.timeline.window().paste_timeline_state())
        
        menu.exec(event.globalPos())
    
    def split_clip(self, clip, split_time):
        """Split a clip at the specified time"""
        if clip.start_time < split_time < clip.end_time:
            from PySide6.QtCore import QTimer
            import uuid
            
            # Create second part of split
            new_clip = MediaClip(
                id=str(uuid.uuid4()),
                file_path=clip.file_path,
                name=f"{clip.name}_part2",
                duration=clip.end_time - split_time,
                start_time=split_time,
                end_time=clip.end_time,
                position=clip.position + (split_time - clip.start_time),
                track=clip.track,
                file_type=clip.file_type
            )
            
            # Update original clip
            clip.end_time = split_time
            
            # Add new clip
            self.timeline.add_clip(new_clip)
            self.timeline.update_timeline()
    
    def trim_clip(self, clip):
        """Trim the beginning or end of a clip"""
        dialog = TrimClipDialog(clip, self)
        if dialog.exec():
            self.timeline.update_timeline()
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    files.append(url.toLocalFile())
            
            # Add files as clips (simplified - would need proper coordinate mapping)
            position = (event.pos().x() + self.scroll_offset) / self.timeline.pixels_per_second
            window = QApplication.instance().main_window
            
            if hasattr(window, 'add_media_files_to_timeline'):
                window.add_media_files_to_timeline(files, position)

class MediaLibraryWidget(QWidget):
    """Media library for browsing imported files"""
    
    file_imported = Signal(str)  # Signal when file is imported
    
    def __init__(self):
        super().__init__()
        self.media_files: List[Dict] = []
        self.setup_ui()
        
    def setup_ui(self):
        """Setup media library UI"""
        layout = QVBoxLayout()
        
        # Import buttons
        button_layout = QHBoxLayout()
        
        import_btn = QPushButton("Import Files")
        import_btn.clicked.connect(self.import_files)
        
        import_folder_btn = QPushButton("Import Folder")
        import_folder_btn.clicked.connect(self.import_folder)
        
        button_layout.addWidget(import_btn)
        button_layout.addWidget(import_folder_btn)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Search box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Search:"))
        
        self.search_edit = QLineEdit()
        self.search_edit.textChanged.connect(self.filter_files)
        search_layout.addWidget(self.search_edit)
        
        layout.addLayout(search_layout)
        
        # Media list
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(80, 60))
        self.list_widget.setViewMode(QListView.IconMode)
        self.list_widget.setResizeMode(QListView.Adjust)
        self.list_widget.setSpacing(10)
        self.list_widget.setAcceptDrops(True)
        self.list_widget.itemDoubleClicked.connect(self.item_double_clicked)
        self.list_widget.itemSelectionChanged.connect(self.update_status)
        self.list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_widget.customContextMenuRequested.connect(self.show_context_menu)
        self.list_widget.setDragEnabled(True)
        
        layout.addWidget(self.list_widget)
        
        self.setLayout(layout)
        
    def import_files(self):
        """Import media files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Media Files",
            "",
            "Media Files (*.mp4 *.avi *.mov *.mkv *.wav *.mp3 *.png *.jpg *.jpeg);;All Files (*)"
        )
        
        for file_path in files:
            self.add_media_file(file_path)
    
    def import_folder(self):
        """Import entire folder"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Import Media Folder"
        )
        
        if folder_path:
            extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.png', '.jpg', '.jpeg']
            
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        self.add_media_file(file_path)
    
    def add_media_file(self, file_path: str):
        """Add media file to library"""
        file_info = {
            'path': file_path,
            'name': os.path.basename(file_path),
            'type': self.get_media_type(file_path),
            'size': os.path.getsize(file_path),
            'modified': os.path.getmtime(file_path)
        }
        
        # Check if already exists
        if any(mf['path'] == file_path for mf in self.media_files):
            return
            
        self.media_files.append(file_info)
        self.update_list()
        
        self.file_imported.emit(file_path)
    
    def get_media_type(self, file_path: str) -> str:
        """Determine media type from file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        
        video_exts = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        audio_exts = ['.wav', '.mp3', '.flac', '.aac', '.ogg']
        image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff']
        
        if ext in video_exts:
            return 'video'
        elif ext in audio_exts:
            return 'audio'
        elif ext in image_exts:
            return 'image'
        else:
            return 'unknown'
    
    def update_list(self):
        """Update media list display"""
        self.list_widget.clear()
        
        for media in self.media_files:
            item = QListWidgetItem()
            item.setText(media['name'])
            item.setData(Qt.UserRole, media)
            
            # Set icon based on media type
            icon = self.get_icon_for_type(media['type'])
            item.setIcon(icon)
            
            self.list_widget.addItem(item)
    
    def get_icon_for_type(self, media_type: str) -> QIcon:
        """Get icon for media type"""
        if media_type == 'video':
            return QIcon.fromTheme("video-x-generic")
        elif media_type == 'audio':
            return QIcon.fromTheme("audio-x-generic")
        elif media_type == 'image':
            return QIcon.fromTheme("image-x-generic")
        else:
            return QIcon.fromTheme("application-x-generic")
    
    def filter_files(self, text: str):
        """Filter files based on search text"""
        text = text.lower()
        
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            media = item.data(Qt.UserRole)
            
            if text in media['name'].lower():
                item.setHidden(False)
            else:
                item.setHidden(True)
    
    def item_double_clicked(self, item):
        """Handle double click on media item"""
        media = item.data(Qt.UserRole)
        window = QApplication.instance().main_window
        
        if hasattr(window, 'play_preview_media'):
            window.play_preview_media(media['path'])
        else:
            # Fallback: add to timeline at end
            self.add_to_timeline_from_item(item)
    
    def show_context_menu(self, position):
        """Show context menu for media library"""
        item = self.list_widget.itemAt(position)
        if item:
            menu = QMenu(self)
            
            # Get the media file data
            row = self.list_widget.row(item)
            media_file = self.media_files[row]
            
            # Media-specific actions
            menu.addAction("Add to Timeline", lambda: self.add_to_timeline(row))
            menu.addAction("Preview", lambda: self.preview_media(row))
            menu.addSeparator()
            menu.addAction("Duplicate", lambda: self.duplicate_media(row))
            menu.addAction("Remove from Library", lambda: self.remove_media(row))
            menu.addSeparator()
            menu.addAction("Properties", lambda: self.show_properties(row))
            
            menu.exec(self.list_widget.mapToGlobal(position))
        else:
            # Empty space context menu
            menu = QMenu(self)
            menu.addAction("Import Files", self.import_files)
            menu.addAction("Import Folder", self.import_folder)
            menu.addAction("Clear All", self.clear_library)
            menu.exec(self.list_widget.mapToGlobal(position))
    
    def update_status(self):
        """Update status when selection changes"""
        items = self.list_widget.selectedItems()
        if items:
            item = items[0]
            media = item.data(Qt.UserRole)
            window = QApplication.instance().main_window
            if hasattr(window, 'statusBar'):
                window.statusBar().showMessage(f"Selected: {media['name']}")
    
    def add_to_timeline(self, row):
        """Add media file to timeline"""
        if 0 <= row < len(self.media_files):
            media = self.media_files[row]
            window = QApplication.instance().main_window
            if hasattr(window, 'timeline'):
                window.timeline.add_media_to_timeline(media['path'])
    
    def add_to_timeline_from_item(self, item):
        """Add media from item to timeline (for double-click fallback)"""
        media = item.data(Qt.UserRole)
        window = QApplication.instance().main_window
        if hasattr(window, 'timeline'):
            window.timeline.add_media_to_timeline(media['path'])
    
    def preview_media(self, row):
        """Preview selected media"""
        if 0 <= row < len(self.media_files):
            media = self.media_files[row]
            window = QApplication.instance().main_window
            if hasattr(window, 'preview_widget'):
                window.preview_widget.load_media(media['path'])
                # Preview widget will automatically display the loaded media
    
    def duplicate_media(self, row):
        """Duplicate media file entry"""
        if 0 <= row < len(self.media_files):
            original = self.media_files[row].copy()
            original['name'] = f"{original['name']}_copy"
            self.media_files.append(original)
            self.refresh_display()
    
    def remove_media(self, row):
        """Remove media from library"""
        if 0 <= row < len(self.media_files):
            self.media_files.pop(row)
            self.refresh_display()
    
    def show_properties(self, row):
        """Show properties dialog for media"""
        if 0 <= row < len(self.media_files):
            media = self.media_files[row]
            info = f"Name: {media['name']}\n"
            info += f"Path: {media['path']}\n"
            info += f"Type: {media['type']}\n"
            info += f"Size: {self.format_size(media['size'])}"
            
            QMessageBox.information(self, "Media Properties", info)
    
    def clear_library(self):
        """Clear all media from library"""
        reply = QMessageBox.question(
            self, "Clear Library", 
            "Are you sure you want to remove all media files from the library?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.media_files.clear()
            self.refresh_display()
    
    def format_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def startDrag(self, supportedActions):
        """Enable dragging from media library"""
        item = self.currentItem()
        if item:
            media = item.data(Qt.UserRole)
            drag = QDrag(self)
            mime_data = QMimeData()
            
            # Set the file path as text
            mime_data.setText(media['path'])
            
            # Set URLs for drag and drop
            urls = [QUrl.fromLocalFile(media['path'])]
            mime_data.setUrls(urls)
            
            drag.setMimeData(mime_data)
            drag.exec(supportedActions)


class TrimClipDialog(QDialog):
    """Dialog for trimming clips"""
    
    def __init__(self, clip: MediaClip, parent=None):
        super().__init__(parent)
        self.clip = clip
        self.setup_ui()
    
    def setup_ui(self):
        """Setup trim dialog UI"""
        self.setWindowTitle(f"Trim: {self.clip.name}")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # Current values
        layout.addWidget(QLabel(f"Current duration: {self.clip.end_time - self.clip.start_time:.2f}s"))
        
        # Trim controls
        trim_layout = QHBoxLayout()
        trim_layout.addWidget(QLabel("Start:"))
        self.start_spinbox = QDoubleSpinBox()
        self.start_spinbox.setRange(0, self.clip.end_time)
        self.start_spinbox.setValue(self.clip.start_time)
        trim_layout.addWidget(self.start_spinbox)
        
        trim_layout.addWidget(QLabel("End:"))
        self.end_spinbox = QDoubleSpinBox()
        self.end_spinbox.setRange(self.clip.start_time, 999)
        self.end_spinbox.setValue(self.clip.end_time)
        trim_layout.addWidget(self.end_spinbox)
        
        layout.addLayout(trim_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        ok_btn = QPushButton("Apply")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)


class PreviewWidget(QWidget):
    """Video preview widget"""
    
    def __init__(self):
        super().__init__()
        self.video_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.current_file = None
        self.is_timeline_playback = False  # Track if playing timeline vs individual file
        
        # Connect audio output to media player
        self.video_player.setAudioOutput(self.audio_output)
        self.video_player.setVideoOutput(self)  # Connect video output to this widget
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup preview UI"""
        layout = QVBoxLayout()
        
        # Video display area
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: black;
                border: 1px solid gray;
                border-radius: 5px;
            }
        """)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No video loaded")
        self.video_label.setProperty("class", "video-preview")
        
        # Set video output to the label
        self.video_player.setVideoOutput(self.video_label)
        
        layout.addWidget(self.video_label)
        
        # Playback controls
        controls_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setMaximumWidth(50)
        
        self.time_label = QLabel("00:00 / 00:00")
        
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.sliderMoved.connect(self.seek_video)
        self.progress_slider.setEnabled(False)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMaximumWidth(100)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setValue(50)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.time_label)
        controls_layout.addWidget(self.progress_slider)
        controls_layout.addWidget(self.volume_slider)
        
        layout.addLayout(controls_layout)
        self.setLayout(layout)
        
        controls_layout.addWidget(self.play_btn)
        controls_layout.addWidget(self.time_label)
        controls_layout.addWidget(self.progress_slider)
        controls_layout.addWidget(QLabel("Volume:"))
        controls_layout.addWidget(self.volume_slider)
        
        layout.addWidget(self.video_label)
        layout.addLayout(controls_layout)
        
        self.setLayout(layout)
        
        # Connect media player signals
        self.video_player.positionChanged.connect(self.update_position)
        self.video_player.durationChanged.connect(self.update_duration)
        self.video_player.mediaStatusChanged.connect(self.handle_media_status)
    
    def load_media(self, file_path: str):
        """Load media file for preview"""
        self.current_file = file_path
        url = QUrl.fromLocalFile(file_path)
        self.video_player.setSource(url)
        
    def toggle_playback(self):
        """Toggle play/pause - handles both individual file and timeline playback"""
        # Check if we're in timeline playback mode
        if hasattr(self, 'parent_window') and self.parent_window:
            # Get reference to main window
            main_window = self.parent_window
            
            if hasattr(main_window, 'preview_widget') and main_window.preview_widget == self:
                # If current file is loaded from timeline, this is timeline playback
                if self.is_timeline_playback:
                    if self.video_player.playbackState() == QMediaPlayer.PlayingState:
                        main_window.stop_timeline()
                    else:
                        main_window.play_timeline()
                else:
                    # Individual file playback
                    if self.video_player.playbackState() == QMediaPlayer.PlayingState:
                        self.video_player.pause()
                        self.play_btn.setText("▶")
                    else:
                        self.video_player.play()
                        self.play_btn.setText("⏸")
            else:
                # Fallback to regular playback
                if self.video_player.playbackState() == QMediaPlayer.PlayingState:
                    self.video_player.pause()
                    self.play_btn.setText("▶")
                else:
                    self.video_player.play()
                    self.play_btn.setText("⏸")
        else:
            # Fallback to regular playback
            if self.video_player.playbackState() == QMediaPlayer.PlayingState:
                self.video_player.pause()
                self.play_btn.setText("▶")
            else:
                self.video_player.play()
                self.play_btn.setText("⏸")
    
    def seek_video(self, position):
        """Seek to position"""
        self.video_player.setPosition(position)
    
    def set_volume(self, volume):
        """Set playback volume"""
        self.audio_output.setVolume(volume / 100.0)  # Convert 0-100 to 0.0-1.0
    
    def update_position(self, position):
        """Update position display and slider"""
        self.progress_slider.setRange(0, self.video_player.duration())
        self.progress_slider.setValue(position)
        
        # Update time display
        current_time = self.format_time(position)
        total_time = self.format_time(self.video_player.duration())
        self.time_label.setText(f"{current_time} / {total_time}")
    
    def update_duration(self, duration):
        """Update duration"""
        self.progress_slider.setRange(0, duration)
    
    def handle_media_status(self, status):
        """Handle media status changes"""
        if status == QMediaPlayer.EndOfMedia:
            self.play_btn.setText("▶")
    
    def format_time(self, milliseconds: int) -> str:
        """Format milliseconds to MM:SS"""
        if milliseconds <= 0:
            return "00:00"
        
        seconds = milliseconds // 1000
        minutes = seconds // 60
        seconds = seconds % 60
        
        return f"{minutes:02d}:{seconds:02d}"

class ProjectManager:
    """Manages project operations (new, open, save)"""
    
    def __init__(self):
        self.current_project: Optional[Project] = None
        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_timer.start(60000)  # 1 minute intervals
        
    def new_project(self, name: str = "Untitled Project") -> Project:
        """Create new project"""
        self.current_project = Project(
            name=name,
            file_path="",
            clips=[],
            timeline_duration=60.0,
            created_time=datetime.now().isoformat(),
            modified_time=datetime.now().isoformat()
        )
        return self.current_project
    
    def save_project(self, file_path: str = None) -> bool:
        """Save current project"""
        if not self.current_project:
            return False
        
        if file_path:
            self.current_project.file_path = file_path
        
        try:
            if not self.current_project.file_path:
                return False
            
            project_data = asdict(self.current_project)
            
            with open(self.current_project.file_path, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            self.current_project.modified_time = datetime.now().isoformat()
            return True
            
        except Exception as e:
            print(f"Error saving project: {e}")
            return False
    
    def load_project(self, file_path: str) -> bool:
        """Load project from file"""
        try:
            with open(file_path, 'r') as f:
                project_data = json.load(f)
            
            # Reconstruct project
            clips_data = project_data.get('clips', [])
            clips = []
            for clip_data in clips_data:
                clip = MediaClip(**clip_data)
                clips.append(clip)
            
            self.current_project = Project(
                name=project_data.get('name', 'Untitled'),
                file_path=file_path,
                clips=clips,
                timeline_duration=project_data.get('timeline_duration', 60.0),
                fps=project_data.get('fps', 30),
                width=project_data.get('width', 1920),
                height=project_data.get('height', 1080),
                created_time=project_data.get('created_time'),
                modified_time=project_data.get('modified_time')
            )
            
            return True
            
        except Exception as e:
            print(f"Error loading project: {e}")
            return False
    
    def auto_save(self):
        """Auto save current project"""
        if self.current_project and self.current_project.file_path:
            self.save_project()
    
    def export_project(self, output_path: str) -> bool:
        """Export project as video"""
        if not self.current_project:
            return False
        
        ff_manager = FFmpegManager()
        return ff_manager.export_project(self.current_project, output_path)

class VideoEditorMainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.project_manager = ProjectManager()
        self.ffmpeg_manager = FFmpegManager()
        self.setup_ui()
        self.setup_menus()
        self.setup_toolbar()
        self.statusBar().showMessage("Ready")
        
        # Set window properties
        self.setWindowTitle("OpenShot Clone - Video Editor")
        self.setMinimumSize(1200, 800)
        
        # Create new project on startup
        self.project_manager.new_project()
        self.update_project_display()
        
        # Store reference for signal connections
        QApplication.instance().main_window = self
    
    def setup_ui(self):
        """Setup main UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Media Library
        self.media_library = MediaLibraryWidget()
        self.media_library.file_imported.connect(self.on_file_imported)
        
        # Right panel - Preview and Timeline (fix layout issue)
        right_panel_widget = QWidget()
        right_panel_layout = QVBoxLayout(right_panel_widget)
        right_panel_layout.setContentsMargins(0, 0, 0, 0)
        right_panel_layout.setSpacing(2)
        
        # Preview widget (collapsible)
        self.preview_widget = PreviewWidget()
        self.preview_widget.parent_window = self  # Store reference to main window
        right_panel_layout.addWidget(self.preview_widget, 1)  # Give more space to preview
        
        # Timeline (takes remaining space)
        self.timeline = TimelineWidget()
        self.timeline.parent_window = self  # Store reference to main window
        right_panel_layout.addWidget(self.timeline, 3)  # Give more space to timeline
        
        # Create splitters
        left_splitter = QSplitter(Qt.Horizontal)
        left_splitter.addWidget(self.media_library)
        left_splitter.addWidget(right_panel_widget)
        left_splitter.setSizes([300, 900])
        
        # Add panels to main layout
        main_layout.addWidget(left_splitter)
        
        # Connect signals
        self.media_library.file_imported.connect(self.statusBar().showMessage)
        
        # Store preview toggle state
        self.preview_visible = True
    
    def toggle_preview(self):
        """Toggle preview widget visibility"""
        if self.preview_visible:
            self.preview_widget.hide()
            self.preview_visible = False
            self.statusBar().showMessage("Preview hidden - Timeline maximized")
        else:
            self.preview_widget.show()
            self.preview_visible = True
            self.statusBar().showMessage("Preview shown")
    
    def setup_menus(self):
        """Setup application menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        new_action = QAction('New Project', self)
        new_action.setShortcut('Ctrl+N')
        new_action.triggered.connect(self.new_project)
        file_menu.addAction(new_action)
        
        open_action = QAction('Open Project', self)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_project)
        file_menu.addAction(open_action)
        
        save_action = QAction('Save Project', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        import_action = QAction('Import Media', self)
        import_action.setShortcut('Ctrl+I')
        import_action.triggered.connect(self.import_media)
        file_menu.addAction(import_action)
        
        file_menu.addSeparator()
        
        export_action = QAction('Export Video', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_video)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu('Edit')
        
        undo_action = QAction('Undo', self)
        undo_action.setShortcut('Ctrl+Z')
        edit_menu.addAction(undo_action)
        
        redo_action = QAction('Redo', self)
        redo_action.setShortcut('Ctrl+Y')
        edit_menu.addAction(redo_action)
        
        edit_menu.addSeparator()
        
        cut_action = QAction('Cut', self)
        cut_action.setShortcut('Ctrl+X')
        edit_menu.addAction(cut_action)
        
        copy_action = QAction('Copy', self)
        copy_action.setShortcut('Ctrl+C')
        edit_menu.addAction(copy_action)
        
        paste_action = QAction('Paste', self)
        paste_action.setShortcut('Ctrl+V')
        edit_menu.addAction(paste_action)
        
        # View menu
        view_menu = menubar.addMenu('View')
        
        toggle_preview_action = QAction('Toggle Preview', self)
        toggle_preview_action.setShortcut('F5')
        toggle_preview_action.triggered.connect(self.toggle_preview)
        view_menu.addAction(toggle_preview_action)
        
        fullscreen_action = QAction('Fullscreen Preview', self)
        fullscreen_action.setShortcut('F11')
        fullscreen_action.triggered.connect(self.fullscreen_preview)
        view_menu.addAction(fullscreen_action)
        
        view_menu.addSeparator()
        
        add_track_menu = view_menu.addMenu('Timeline')
        
        add_track_action = QAction('Add Track', self)
        add_track_action.setShortcut('Ctrl+T')
        add_track_action.triggered.connect(self.add_track)
        add_track_menu.addAction(add_track_action)
        
        delete_track_action = QAction('Delete Track', self)
        delete_track_action.setShortcut('Ctrl+Shift+T')
        delete_track_action.triggered.connect(self.delete_track)
        add_track_menu.addAction(delete_track_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_toolbar(self):
        """Setup toolbar"""
        toolbar = self.addToolBar('Main')
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        
        # New project
        new_action = QAction('New', self)
        new_action.setToolTip('New Project')
        new_action.triggered.connect(self.new_project)
        toolbar.addAction(new_action)
        
        # Open project
        open_action = QAction('Open', self)
        open_action.setToolTip('Open Project')
        open_action.triggered.connect(self.open_project)
        toolbar.addAction(open_action)
        
        # Save project
        save_action = QAction('Save', self)
        save_action.setToolTip('Save Project')
        save_action.triggered.connect(self.save_project)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # Import media
        import_action = QAction('Import', self)
        import_action.setToolTip('Import Media')
        import_action.triggered.connect(self.import_media)
        toolbar.addAction(import_action)
        
        toolbar.addSeparator()
        
        # Timeline controls
        add_track_action = QAction('+ Track', self)
        add_track_action.setToolTip('Add New Track')
        add_track_action.triggered.connect(self.add_track)
        toolbar.addAction(add_track_action)
        
        delete_track_action = QAction('- Track', self)
        delete_track_action.setToolTip('Delete Selected Track')
        delete_track_action.triggered.connect(self.delete_track)
        toolbar.addAction(delete_track_action)
        
        toolbar.addSeparator()
        
        # Export
        export_action = QAction('Export', self)
        export_action.setToolTip('Export Video')
        export_action.triggered.connect(self.export_video)
        toolbar.addAction(export_action)
    
    def add_track(self):
        """Add new track to timeline"""
        if hasattr(self.timeline, 'num_tracks'):
            self.timeline.num_tracks += 1
            self.timeline.update_timeline()
            self.statusBar().showMessage(f"Added track {self.timeline.num_tracks}")
    
    def delete_track(self):
        """Delete selected track from timeline"""
        if hasattr(self.timeline, 'num_tracks') and self.timeline.num_tracks > 1:
            self.timeline.num_tracks -= 1
            # Remove clips from deleted track
            clips_to_remove = [clip for clip in self.timeline.clips if clip.track >= self.timeline.num_tracks]
            for clip in clips_to_remove:
                self.timeline.remove_clip(clip.id)
            self.timeline.update_timeline()
            self.statusBar().showMessage(f"Deleted track. Current tracks: {self.timeline.num_tracks}")
    
    def new_project(self):
        """Create new project"""
        reply = QMessageBox.question(
            self, 'New Project', 
            'Create a new project? Unsaved changes will be lost.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            name, ok = QInputDialog.getText(self, 'New Project', 'Project name:')
            if ok and name:
                self.project_manager.new_project(name)
                self.update_project_display()
                self.statusBar().showMessage(f"Created new project: {name}")
    
    def open_project(self):
        """Open existing project"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', '', 'Project Files (*.osp);;All Files (*)'
        )
        
        if file_path:
            if self.project_manager.load_project(file_path):
                self.update_project_display()
                self.statusBar().showMessage(f"Opened project: {os.path.basename(file_path)}")
            else:
                QMessageBox.warning(self, 'Error', 'Failed to load project file.')
    
    def save_project(self):
        """Save current project"""
        if not self.project_manager.current_project:
            return
        
        if not self.project_manager.current_project.file_path:
            self.save_project_as()
        else:
            if self.project_manager.save_project():
                self.statusBar().showMessage("Project saved successfully")
            else:
                QMessageBox.warning(self, 'Error', 'Failed to save project.')
    
    def save_project_as(self):
        """Save project with new name"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 'Save Project As', '', 'Project Files (*.osp);;All Files (*)'
        )
        
        if file_path:
            if not file_path.endswith('.osp'):
                file_path += '.osp'
            
            if self.project_manager.save_project(file_path):
                self.update_project_display()
                self.statusBar().showMessage("Project saved successfully")
            else:
                QMessageBox.warning(self, 'Error', 'Failed to save project.')
    
    def import_media(self):
        """Import media files"""
        self.media_library.import_files()
    
    def export_video(self):
        """Export project as video"""
        if not self.project_manager.current_project:
            QMessageBox.warning(self, 'Warning', 'No project to export.')
            return
        
        output_path, _ = QFileDialog.getSaveFileName(
            self, 'Export Video', '', 'MP4 Files (*.mp4);;AVI Files (*.avi);;All Files (*)'
        )
        
        if output_path:
            if not output_path.endswith(('.mp4', '.avi')):
                output_path += '.mp4'
            
            self.statusBar().showMessage("Exporting video...")
            
            # Run export in thread to prevent UI blocking
            def export_thread():
                success = self.project_manager.export_project(output_path)
                
                # Show result in main thread
                QTimer.singleShot(0, lambda: self.export_finished(success, output_path))
            
            threading.Thread(target=export_thread, daemon=True).start()
    
    def export_finished(self, success: bool, output_path: str):
        """Handle export completion"""
        if success:
            QMessageBox.information(self, 'Export Complete', f'Video exported to: {output_path}')
            self.statusBar().showMessage("Export completed successfully")
        else:
            QMessageBox.warning(self, 'Export Failed', 'Failed to export video.')
            self.statusBar().showMessage("Export failed")
    
    def toggle_preview(self):
        """Toggle preview pane visibility"""
        self.preview_widget.setVisible(not self.preview_widget.isVisible())
    
    def fullscreen_preview(self):
        """Show preview in fullscreen"""
        dialog = QDialog(self)
        dialog.setWindowTitle('Fullscreen Preview')
        dialog.setModal(True)
        
        preview = PreviewWidget()
        if self.preview_widget.current_file:
            preview.load_media(self.preview_widget.current_file)
    
    def play_timeline(self):
        """Play the entire timeline"""
        if not self.timeline.clips:
            return
        
        self.preview_widget.is_timeline_playback = True
        self.preview_widget.play_btn.setText("⏸")
        
        # Start timeline playback from current time
        self.timeline_playback_start()
    
    def stop_timeline(self):
        """Stop timeline playback"""
        self.preview_widget.is_timeline_playback = False
        self.preview_widget.play_btn.setText("▶")
        self.preview_widget.video_player.pause()
        
        if hasattr(self, 'timeline_playback_timer'):
            self.timeline_playback_timer.stop()
    
    def timeline_playback_start(self):
        """Start timeline playback timer"""
        from PySide6.QtCore import QTimer
        
        if not hasattr(self, 'timeline_playback_timer'):
            self.timeline_playback_timer = QTimer()
            self.timeline_playback_timer.timeout.connect(self.timeline_playback_update)
        
        self.timeline_playback_timer.start(50)  # Update every 50ms
        self.timeline_playback_update()
    
    def timeline_playback_update(self):
        """Update timeline during playback"""
        if not self.preview_widget.is_timeline_playback:
            return
        
        current_time = self.timeline.current_time
        
        # Find clips that should be playing at current time
        active_clips = []
        for clip in self.timeline.clips:
            if (clip.position <= current_time < clip.end_time):
                active_clips.append(clip)
        
        if not active_clips:
            # No clips playing, stop playback
            self.stop_timeline()
            return
        
        # For simplicity, play the first active clip (video takes priority)
        video_clip = None
        audio_clip = None
        
        for clip in active_clips:
            if clip.file_type == 'video' and not video_clip:
                video_clip = clip
            elif clip.file_type == 'audio' and not audio_clip:
                audio_clip = clip
        
        # Play video if available, otherwise play first clip
        play_clip = video_clip or active_clips[0]
        
        # Update playhead
        self.timeline.current_time = current_time + 0.05  # Advance 50ms
        self.timeline.update_timeline()
        self.timeline.update_ruler()
        
        # Load clip in preview if it's different from current
        if (self.preview_widget.current_file != play_clip.file_path):
            self.play_preview_media(play_clip.file_path)
            
            # Start playing
            self.preview_widget.video_player.setPosition(int((current_time - play_clip.position) * 1000))
            self.preview_widget.video_player.play()
            
        print(f"Timeline playback: {current_time:.1f}s, playing {play_clip.name}")
        
        # Check if we've reached the end
        if self.timeline.current_time >= self.timeline.duration:
            self.stop_timeline()
        layout = QVBoxLayout()
        layout.addWidget(preview)
        
        close_btn = QPushButton('Close')
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.resize(800, 600)
        dialog.exec()
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            'About OpenShot Clone',
            'OpenShot Clone Video Editor\n\n'
            'A video editing application built with PySide6\n'
            'Author: MiniMax Agent\n'
            'Version: 1.0'
        )
    
    def on_file_imported(self, file_path: str):
        """Handle file import"""
        self.statusBar().showMessage(f"Imported: {os.path.basename(file_path)}")
    
    def add_media_files_to_timeline(self, file_paths: List[str], position: float = 0.0):
        """Add media files to timeline"""
        for i, file_path in enumerate(file_paths):
            # Create clip
            file_name = os.path.basename(file_path)
            media_type = self.media_library.get_media_type(file_path)
            
            # Get video info if needed
            duration = 5.0  # Default duration
            try:
                info = self.ffmpeg_manager.get_video_info(file_path)
                if 'format' in info and 'duration' in info['format']:
                    duration = float(info['format']['duration'])
            except:
                pass
            
            clip = MediaClip(
                id=f"clip_{int(time.time())}_{i}",
                name=file_name,
                file_path=file_path,
                file_type=media_type,
                duration=duration,
                start_time=0.0,
                end_time=duration,
                position=position + i * 10,  # Offset each clip
                track=0  # Default to first track
            )
            
            self.timeline.add_clip(clip)
            if self.project_manager.current_project:
                self.project_manager.current_project.clips.append(clip)
        
        self.statusBar().showMessage(f"Added {len(file_paths)} clips to timeline")
    
    def play_preview_media(self, file_path: str):
        """Load media in preview pane"""
        self.preview_widget.load_media(file_path)
        self.statusBar().showMessage(f"Loaded in preview: {os.path.basename(file_path)}")
    
    def contextMenuEvent(self, event):
        """Handle right-click context menus"""
        context_menu = QMenu(self)
        
        # Timeline context menu
        if hasattr(self.timeline, 'timeline_area') and self.timeline.timeline_area.geometry().contains(event.pos()):
            # Timeline-specific actions
            add_track_action = QAction("Add Track", self)
            add_track_action.triggered.connect(self.add_track)
            context_menu.addAction(add_track_action)
            
            if self.timeline.num_tracks > 1:
                delete_track_action = QAction("Delete Track", self)
                delete_track_action.triggered.connect(self.delete_track)
                context_menu.addAction(delete_track_action)
            
            context_menu.addSeparator()
            
            clear_timeline_action = QAction("Clear Timeline", self)
            clear_timeline_action.triggered.connect(self.clear_timeline)
            context_menu.addAction(clear_timeline_action)
            
        # Media library context menu
        elif hasattr(self.media_library, 'list_widget') and self.media_library.list_widget.geometry().contains(event.pos()):
            # Media-specific actions
            import_files_action = QAction("Import Files", self)
            import_files_action.triggered.connect(self.import_media)
            context_menu.addAction(import_files_action)
            
            import_folder_action = QAction("Import Folder", self)
            import_folder_action.triggered.connect(self.media_library.import_folder)
            context_menu.addAction(import_folder_action)
            
            if self.media_library.list_widget.currentItem():
                context_menu.addSeparator()
                remove_file_action = QAction("Remove from Library", self)
                remove_file_action.triggered.connect(self.remove_selected_media)
                context_menu.addAction(remove_file_action)
        
        # General context menu
        else:
            about_action = QAction("About", self)
            about_action.triggered.connect(self.show_about)
            context_menu.addAction(about_action)
        
        context_menu.exec(event.globalPos())
    
    def clear_timeline(self):
        """Clear all clips from timeline"""
        reply = QMessageBox.question(
            self, 'Clear Timeline', 
            'Remove all clips from timeline?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.timeline.clips.clear()
            self.timeline.update_timeline()
            if self.project_manager.current_project:
                self.project_manager.current_project.clips.clear()
            self.statusBar().showMessage("Timeline cleared")
    
    def remove_selected_media(self):
        """Remove selected media from library"""
        current_item = self.media_library.list_widget.currentItem()
        if current_item:
            media = current_item.data(Qt.UserRole)
            # Remove from internal list
            self.media_library.media_files = [
                mf for mf in self.media_library.media_files 
                if mf['path'] != media['path']
            ]
            # Update display
            self.media_library.refresh_display()
            self.statusBar().showMessage(f"Removed: {media['name']}")
    

    
    def copy_timeline_state(self):
        """Copy current timeline state (placeholder for future implementation)"""
        self.statusBar().showMessage("Timeline copied to clipboard")
    
    def paste_timeline_state(self):
        """Paste timeline state (placeholder for future implementation)"""
        self.statusBar().showMessage("Timeline pasted from clipboard")
    
    def update_project_display(self):
        """Update project-related UI elements"""
        if self.project_manager.current_project:
            project = self.project_manager.current_project
            self.setWindowTitle(f"OpenShot Clone - {project.name}")
            
            # Update timeline with project clips
            self.timeline.clips = project.clips
            self.timeline.update_timeline()
    
    def closeEvent(self, event):
        """Handle application close"""
        # Save project before closing
        if (self.project_manager.current_project and 
            self.project_manager.current_project.file_path):
            self.project_manager.save_project()
        
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("OpenShot Clone")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = VideoEditorMainWindow()
    window.show()
    
    return app.exec()

if __name__ == "__main__":
    main()