from __future__ import annotations

"""
Standalone help window for the FrameVision Timeline Editor.

This file is intentionally separate from timeline_editor.py so the help text can be
expanded later without touching playback, timeline, transition, or export logic.
"""

from html import escape
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


HELP_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 13px;
        line-height: 1.45;
        color: #202124;
        background: #ffffff;
        margin: 18px;
    }
    h1 {
        font-size: 24px;
        margin: 0 0 8px 0;
    }
    h2 {
        font-size: 18px;
        margin: 24px 0 8px 0;
        padding-top: 8px;
        border-top: 1px solid #ddd;
    }
    h3 {
        font-size: 15px;
        margin: 16px 0 6px 0;
    }
    p {
        margin: 6px 0 10px 0;
    }
    ul, ol {
        margin-top: 6px;
        margin-bottom: 12px;
        padding-left: 24px;
    }
    li {
        margin-bottom: 4px;
    }
    .tip {
        background: #f3f7ff;
        border-left: 4px solid #5b8def;
        padding: 8px 10px;
        margin: 10px 0;
    }
    .warn {
        background: #fff7e6;
        border-left: 4px solid #e5a100;
        padding: 8px 10px;
        margin: 10px 0;
    }
    .small {
        color: #555;
        font-size: 12px;
    }
    .kbd {
        display: inline-block;
        border: 1px solid #bbb;
        border-bottom-width: 2px;
        border-radius: 3px;
        padding: 0 5px;
        background: #f7f7f7;
        font-family: Consolas, monospace;
        font-size: 12px;
    }
</style>
</head>
<body>

<h1>FrameVision Timeline Editor Help</h1>
<p>
This editor is a local/offline timeline editor for building a video from clips, images,
audio, transitions, fades, transforms, and a final MP4 export.
</p>

<div class="tip">
<b>Best beginner workflow:</b> add media, drag clips to the timeline, trim or move them,
preview the result, add transitions/fades if needed, then export the timeline.
</div>

<h2>Quick Start</h2>
<ol>
    <li>Click <b>Add Media</b> to load videos, images, or audio into the Media Bin.</li>
    <li>Choose the <b>Target track</b> or drag media directly onto the timeline.</li>
    <li>Click <b>Text</b> on the timeline toolbar when you want to create titles, captions, credits, or animated text overlays.</li>
    <li>Use <b>Play</b>, <b>Pause</b>, <b>Stop</b>, <b>Start</b>, and <b>End</b> at the bottom to preview your edit.</li>
    <li>Drag clip edges to trim. Drag the clip body to move clips along the timeline or between tracks.</li>
    <li>Overlap two visual clips and apply a transition from the Transitions panel.</li>
    <li>Use right-click menus for clip options such as split, delete, speed, volume, fades, and transforms.</li>
    <li>Click <b>Export Video</b> when the timeline looks right.</li>
</ol>

<h2>Top Toolbar</h2>
<h3>New Project</h3>
<p>Starts a clean project. Use this when you want to clear the current timeline and begin again.</p>

<h3>Load Project / Save Project</h3>
<p>
Loads or saves the editor project JSON. This stores the edit structure, clip placement,
tracks, transitions, fades, transforms, Text clips, multi-selection state, and other project settings.
It does not create the final video; use <b>Export Video</b> for that.
</p>

<h3>Autosave and Recovery</h3>
<p>
Automatic project saving helps recover work after an accidental close or crash.
Autosave runs silently in the background about every <b>60 seconds</b>, but only when the project has changes.
</p>
<ul>
    <li>If you are working in an already saved project JSON, autosave updates that project file.</li>
    <li>If the project has not been manually saved yet, autosave writes a recovery file in <b>/temp/timeline_editor_autosave.json</b>.</li>
    <li>When the editor starts and finds a recovery autosave, it asks whether to <b>Restore</b>, <b>Delete</b>, or <b>Ignore</b> it.</li>
    <li><b>Restore</b> loads the recovery project and keeps it marked as unsaved, so you can review it and save it where you want.</li>
    <li><b>Delete</b> removes the recovery file. Use this only when you are sure you do not need it.</li>
    <li><b>Ignore</b> leaves the recovery file in place so you can decide later.</li>
</ul>
<div class="tip">
Autosave is a safety net, not a replacement for normal project saving. For important edits, still use <b>Save Project</b> and give the project a clear name.
</div>
<div class="warn">
If you intentionally want to keep different versions of a project, use separate saved project files. Autosave is designed to protect the current work, not to manage version history.
</div>

<h3>Export Video</h3>
<p>
Creates the final MP4 from the full timeline using FFmpeg. Export is the real result,
so if live preview is a bit slow around heavy transitions, check the exported MP4 before assuming the edit is broken.
</p>

<h3>Project Settings</h3>
<p>
Controls project canvas settings. The canvas is the final composition size used for preview and export.
Use this when you want a different project format, such as landscape, portrait, or square.
</p>

<h3>Master Volume and Master Mute</h3>
<p>
Controls the overall audio level for the full project. Master mute silences everything without changing
the individual clip or track volumes.
</p>

<h3>Layout and Reset Layout</h3>
<p>
Changes the workspace layout. Reset layout restores a safe default if panels become awkward after resizing.
</p>

<h3>Font</h3>
<p>
Changes the editor UI font size. This is useful on high-resolution screens or when the timeline gets crowded.
</p>

<h3>Preview Quality</h3>
<p>
Controls live preview quality only. Lower preview quality can make editing smoother. It does not reduce final export quality.
</p>

<h3>Text</h3>
<p>
Creates a reusable Text/Title asset and adds it to the timeline. Use this for titles, captions, lower-thirds, end credits, labels, or animated text overlays.
</p>

<h2>Media Bin</h2>
<p>
The Media Bin lists files that are available for your project. It shows favorites, media-type colors,
preview thumbnails, display name, type, duration, and source path.
</p>
<ul>
    <li><b>Add Media</b>: import video, image, audio, or Text assets.</li>
    <li><b>Preview thumbnail</b>: shows a small preview when available. Images and Text assets can show their own preview; videos can reuse already generated timeline thumbnails.</li>
    <li><b>Colored dot</b>: shows the media type at a glance: video, audio, image, or Text.</li>
    <li><b>Favorites</b>: mark important media with a star. Favorites stay grouped above normal items so they are easier to find.</li>
    <li><b>Sorting</b>: click the Media Bin column headers to sort by name, type, duration, path, or favorite state.</li>
    <li><b>Remove</b>: remove selected files from the Media Bin.</li>
    <li><b>Add</b>: add the selected Media Bin item to the selected Target track.</li>
    <li><b>Target track</b>: decides where the Add button places the selected media.</li>
</ul>
<div class="tip">
You can also drag media directly from the Media Bin onto any timeline track. This is usually faster than using the Add button.
</div>
<p>
Text assets also appear in the Media Bin as type <b>Text</b>. You can drag them onto the timeline like normal media.
Right-click a Text asset in the Media Bin and choose <b>Edit Text Asset</b> to change the reusable saved text style.
</p>
<p>
Right-click a Media Bin item for quick actions: <b>Preview source</b>, <b>Add to timeline at playhead</b>,
<b>Rename in Media Bin</b>, <b>Open Folder</b>, <b>Favorite</b>, <b>Remove from media bin</b>,
<b>Media info</b>, or <b>Copy media info</b>. Rename in Media Bin is editor-only: it changes the name shown in the project,
but it does not rename or move the real source file.
</p>

<h2>Preview Pane</h2>
<p>
The Preview pane shows the current frame from the timeline or selected media. It is also used for checking visual
position, scale, rotation, opacity, transitions, and fine details inside a media file.
</p>
<p>
The Preview pane itself supports pan and zoom when the mouse is hovering over it. Use the mouse scroll wheel up/down
to zoom in and out. You can zoom very far into the media, up to about <b>50x</b>, and then pan around to inspect details.
</p>
<div class="tip">
For very deep zoom levels, especially around 50x, pause video playback before zooming or panning. This avoids unnecessary
preview stutter while the editor is trying to play video and redraw a heavily zoomed preview at the same time.
</div>
<p>
Preview-related controls such as <b>Viewer Fit</b>, <b>Fullscreen</b>, <b>Reset Transform</b>, and visual <b>Keyframe</b> buttons may be shown near
the preview area depending on the current layout. Use <b>Reset Transform</b> if a selected visual clip becomes misplaced,
resized, rotated, or transparent by accident.
</p>
<p>
The preview header can also show transform information such as <b>X</b> and <b>Y</b> position. X/Y values describe how far the selected visual clip is moved from its normal centered/default placement.
A visible default-size border may be shown in the Preview pane so you can compare the current moved/scaled/rotated clip against its original/default placement.
</p>

<h2>Timeline Basics</h2>
<p>
The timeline is where the actual edit happens. Time runs from left to right. Tracks are stacked vertically.
The playhead shows the current time.
</p>
<ul>
    <li><b>Zoom - / Zoom +</b>: zoom out or in on the timeline.</li>
    <li><b>Zoom @ Playhead</b>: zooms while keeping focus around the current playhead position.</li>
    <li><b>Fit Timeline</b>: zooms so the project fits in view.</li>
    <li><b>Split</b>: cuts the selected clip at the playhead.</li>
    <li><b>Delete</b>: removes the selected clip or item.</li>
    <li><b>Add Track</b>: creates another track.</li>
    <li><b>Snap</b>: helps clips snap to nearby times, edges, or useful positions.</li>
</ul>

<h2>Bottom Playback Controls</h2>
<ul>
    <li><b>Start</b>: moves the playhead to the beginning of all timeline media.</li>
    <li><b>Play</b>: plays the timeline from the current playhead position.</li>
    <li><b>Pause</b>: pauses playback.</li>
    <li><b>Stop</b>: stops playback.</li>
    <li><b>End</b>: moves the playhead to the end of all timeline media.</li>
    <li><b>Position slider</b>: scrub through the timeline.</li>
</ul>

<h2>Tracks</h2>
<p>
Tracks hold clips. Higher visual tracks can appear above lower visual tracks when clips overlap.
Audio tracks can be muted, soloed, and mixed.
</p>
<ul>
    <li><b>Eye</b>: show or hide a visual track.</li>
    <li><b>M</b>: mute a track.</li>
    <li><b>S</b>: solo a track.</li>
    <li><b>Lock</b>: prevent accidental changes to that track.</li>
</ul>

<h2>Clip Editing</h2>
<p>
Select a clip on the timeline to edit it. Most deeper options are in the right-click menu.
</p>
<ul>
    <li><b>Move</b>: drag the clip body left/right or to another compatible track.</li>
    <li><b>Trim</b>: drag the left or right edge of a clip.</li>
    <li><b>Split</b>: place the playhead where you want the cut and click Split.</li>
    <li><b>Copy / Paste / Duplicate</b>: useful for repeating clips or reusing effects.</li>
    <li><b>Change Speed</b>: changes video speed when available.</li>
    <li><b>Playback &gt; Play backwards</b>: reverses a video clip non-destructively. The original source file is not changed, and the clip keeps its normal place and length on the timeline.</li>
    <li><b>Detach Audio</b>: creates an audio-only clip from a video source.</li>
</ul>


<h2>Multi-Selection</h2>
<p>
Multi-selection lets you select more than one timeline clip at the same time. This is useful when you want to move, delete, copy, paste, duplicate, or apply compatible operations to several clips together.
</p>
<ul>
    <li>Hold <span class="kbd">Shift</span> and click timeline clips to add or remove them from the current selection.</li>
    <li>Drag selected clips to move them together while keeping their relative timing.</li>
    <li>Use copy, paste, duplicate, or delete when you want to work on a group instead of one clip.</li>
    <li>Some actions only apply to compatible clips. For example, visual transforms apply to visual clips, while audio options apply to clips with audio.</li>
</ul>
<div class="tip">
Multi-select is best for moving a whole section of an edit, deleting a group of clips, or copying a repeated layout. If an action seems unavailable, check whether the selected clips are mixed types such as video, audio, image, and Text.
</div>

<h2>Visual Transform Tools</h2>
<p>
Visual clips can be moved, resized, rotated, fitted, mirrored, flipped, or made transparent. These changes affect preview and export.
</p>
<ul>
    <li>Use transform controls or preview interactions to adjust a selected visual clip.</li>
    <li>Use <b>Right-click visual clip &gt; Transform</b> for fit, fill, stretch, reset, rotation, opacity, scale, position, and manual rotation controls.</li>
    <li>Use <b>Transform &gt; Mirror / Flip</b> to mirror a video or image horizontally, vertically, both directions, or reset the mirror state.</li>
    <li>Use <b>Reset Transform</b> if the clip becomes misplaced or you want to return to the default fit.</li>
    <li>Some transform actions can apply to multiple selected visual clips.</li>
    <li>Use clip protection when you want to preview without accidentally moving or resizing a clip.</li>
</ul>
<p>
The Preview pane can be used as a direct visual editor: select a video or image clip, move it around, resize it, rotate it, and check the live result in the viewer. The <b>X/Y position readout</b> helps you see exactly how far the clip has moved from default center.
</p>
<div class="tip">
Use the default-size border as your visual anchor. It makes it much easier to create controlled zooms, pans, rotations, picture-in-picture moves, and before/after style effects without losing the original position.
</div>

<h2>Visual Keyframes</h2>
<p>
Visual keyframes let you animate normal video and image clips over time. A keyframe stores the selected clip's visual transform at the current playhead position: position X/Y, scale/size, rotation, opacity, and fit mode. When you add multiple keyframes on the same clip, the editor interpolates between them, creating smooth movement during playback and export.
</p>
<ol>
    <li>Select a video or image clip on the timeline.</li>
    <li>Move the playhead to the time where the effect should start.</li>
    <li>Move, resize, rotate, or adjust the clip in the Preview pane.</li>
    <li>Click <b>Add/Update Keyframe</b> to store that position.</li>
    <li>Move the playhead later in the clip, change the transform again, and click <b>Add/Update Keyframe</b> again.</li>
    <li>Play the clip to preview the animated movement.</li>
</ol>
<ul>
    <li><b>Add/Update Keyframe</b>: saves the current visual transform at the playhead. If a keyframe already exists near that time, it updates it.</li>
    <li><b>Add Default Keyframe</b>: saves the true default placement at the playhead: X/Y 0, scale 1, rotation 0, opacity 1, fit mode fit.</li>
    <li><b>Delete Keyframe</b>: removes the nearest transform keyframe around the current playhead position.</li>
    <li><b>Clear Keyframes</b>: removes all visual transform keyframes from the selected clip.</li>
    <li><b>Keyframes: 0 / 1 / 2...</b>: shows how many transform keyframes the selected visual clip currently has.</li>
    <li><b>Reset Transform</b>: with no keyframes, resets the clip back to default. With keyframes, it can be used to store/reset the current keyframed position back to default.</li>
</ul>
<div class="tip">
Simple two-keyframe ideas are often enough: start normal, then slowly zoom in; start left, then move right; start small, then grow full screen; or rotate slightly during a beat/drop.
</div>
<div class="warn">
Keyframes are based on the selected clip and the current playhead position inside that clip. If a keyframe button seems to do nothing, check that a video/image clip is selected and that the playhead is actually over that clip.
</div>
<p>
Because multiple keyframes can be added to the same clip, this can be used for much more than basic zooms. You can build custom camera moves, fake handheld motion, picture-in-picture layouts, animated crops, rotating overlays, slow reveal effects, or complex movement patterns by adding several keyframes across the clip.
</p>

<h2>Text / Title Engine</h2>
<p>
The Text tool creates reusable title assets that are saved locally under <b>/assets/text</b>. A Text asset contains the words, styling, preview image, and default duration/position. Once created, it appears in the Media Bin and can be reused on the timeline.
</p>
<ol>
    <li>Click <b>Text</b> on the timeline toolbar.</li>
    <li>Enter the text, asset name, font size, color, opacity, duration, and position.</li>
    <li>Optional: enable outline, shadow, background box, and fade in/out.</li>
    <li>Click <b>Create &amp; Add</b> to save the Text asset and place it on the timeline.</li>
    <li>Drag the Text clip to the right time and track. Put it on a higher visual track when it should appear above video/images.</li>
</ol>

<h3>Editing Text</h3>
<ul>
    <li><b>Right-click a Text clip on the timeline &gt; Edit Text</b> to edit the shared Text asset used by that clip.</li>
    <li><b>Right-click a Text asset in the Media Bin &gt; Edit Text Asset</b> to edit it from the bin.</li>
    <li>Because Text assets are reusable, editing the asset can update other timeline clips that use the same Text asset.</li>
</ul>

<h3>Moving Text on the Preview Pane</h3>
<p>
Selected Text clips can be moved directly inside the Preview pane. This saves a clip-specific position, so one Text asset can be reused in different places on different clips.
</p>
<ul>
    <li>Select the Text clip on the timeline.</li>
    <li>Move it in the Preview pane to the position you want.</li>
    <li>Use <b>Right-click Text clip &gt; Reset Text Position</b> if you want that clip to return to the Text asset/default position.</li>
</ul>

<h3>Text Animation</h3>
<p>
Text animation is clip-specific. This means you can reuse one Text asset many times, but give each timeline clip a different animation.
</p>
<ul>
    <li><b>Right-click a Text clip &gt; Text Animation</b> to open the animation settings.</li>
    <li><b>In / Out animation:</b> slide in/out from left, right, top, or bottom, or zoom in/out.</li>
    <li><b>Scroll:</b> scroll up/down/left/right for credits, crawls, or moving captions.</li>
    <li><b>Static rotation:</b> rotate the Text clip by a fixed number of degrees.</li>
    <li><b>Text reveal:</b> typewriter/reveal characters over time.</li>
    <li><b>Rotate in/out:</b> rotate clockwise or counterclockwise during the start or end of the clip.</li>
    <li><b>Pop/Bounce in/out:</b> make text appear or leave with a stronger motion effect.</li>
    <li><b>Easing:</b> controls how smooth or sharp the movement feels.</li>
</ul>
<div class="warn">
When <b>Scroll</b> is active, it owns the text movement. Slide/zoom, rotate-in/out, and pop/bounce are disabled so animations do not fight each other. Fade, typewriter reveal, static rotation, and easing can still be used.
</div>
<div class="tip">
For title cards, use a short fade or pop. For subtitles or labels, keep animation simple. For credits, use Scroll Up with a longer Text clip duration.
</div>

<h2>Transitions</h2>
<p>
Transitions use mask images from the transitions folder. They are applied to overlapping visual clips.
</p>
<ol>
    <li>Place two visual clips so they overlap on the timeline.</li>
    <li>Select or drag a transition from the Transitions panel.</li>
    <li>The transition block appears over the overlap area.</li>
    <li>Preview around the overlap to check the result.</li>
</ol>
<div class="tip">
For smoother live preview, use optimized PNG transition masks. Large or overly heavy transition images can make preview stutter,
even if the final exported MP4 is fine.
</div>

<h3>Custom PNG Transition Files</h3>
<p>
You can create your own PNG transition mask files and drop them into <b>/assets/transitions</b>.
After adding new files, use <b>Refresh</b> in the Transitions panel or restart the editor if needed.
</p>
<ul>
    <li>Best format: <b>720p grayscale PNG</b>.</li>
    <li>Good mask depth: about <b>64 grayscale layers/levels</b>.</li>
    <li>Recommended file size: keep each PNG under <b>500 KB</b> when possible.</li>
    <li>White/bright areas reveal one clip sooner; black/dark areas reveal later.</li>
    <li>Avoid huge or overly detailed masks if live preview becomes slow.</li>
</ul>
<div class="tip">
A 720p grayscale PNG with around 64 levels and a file size under 500 KB is a good balance: smooth enough for nice transitions, but still light enough for live preview.
</div>

<h2>Reverse Playback</h2>
<p>
Reverse playback is available for video clips from <b>Right-click clip &gt; Playback &gt; Play backwards</b>.
It plays the selected video clip backwards while keeping the timeline position, trim, speed, and project structure normal.
The source file is not changed.
</p>
<ul>
    <li>Reverse playback is video-only. Image, audio-only, and Text clips are ignored.</li>
    <li>Sound from the reversed video is included when the video has usable embedded audio and clip audio is enabled.</li>
    <li>The setting can be toggled on or off from the same menu.</li>
    <li>Live preview may stutter when reverse playback is combined with transitions, heavy transforms, or other demanding effects.</li>
    <li>Export to MP4 renders the reverse playback normally, so judge the final result from an exported test if the preview struggles.</li>
</ul>

<h2>Audio Tools</h2>
<p>
Audio can come from video clips or separate audio tracks. The editor can mix multiple audio sources.
</p>
<ul>
    <li>Use clip volume for one clip.</li>
    <li>Use track volume/mute/solo for a whole track.</li>
    <li>Use master volume for the full project.</li>
    <li>Use audio fades for smooth starts and endings.</li>
    <li>Use volume automation/keyframes when you need volume changes inside a clip.</li>
</ul>

<h2>Audio Fades vs Video Fades</h2>
<p>
Audio fades and video fades are different tools:
</p>
<ul>
    <li><b>Audio fade in/out</b>: fades sound volume up or down.</li>
    <li><b>Video fade in/out</b>: fades the image visually, usually from or to black/transparent.</li>
</ul>
<p>
Use audio fades to avoid harsh sound cuts. Use video fades to create visual openings, endings, or softer cuts.
</p>

<h2>Export Tips</h2>
<ul>
    <li>Save the project before export, especially for longer timelines.</li>
    <li>If preview stutters during heavy transitions, reverse playback, or stacked effects, export a short test before changing the edit.</li>
    <li>Use a higher bitrate for cleaner video and a lower bitrate for smaller files.</li>
    <li>If audio sounds wrong in live preview, check the exported MP4 too. Live preview can be more demanding than final render.</li>
</ul>

<h2>Tips for Smooth Editing</h2>
<ul>
    <li>Lower <b>Preview Quality</b> while editing complex timelines.</li>
    <li>Use <b>Fit Timeline</b> when you get lost.</li>
    <li>Use <b>Reset Layout</b> if panels look wrong after resizing.</li>
    <li>Keep transition assets reasonably optimized for live preview.</li>
    <li>Use <b>Save Project</b> often before testing big edits. Autosave helps, but named project saves are still the safest habit.</li>
    <li>When something looks strange, test a short export before spending time fixing a preview-only issue.</li>
    <li>Use reverse playback carefully around complex transitions. The MP4 export can still be fine even when live preview has to catch up.</li>
</ul>

<h2>Common Beginner Mistakes</h2>
<ul>
    <li><b>Clips are added to the wrong track:</b> check the Target track, or drag directly to the track you want.</li>
    <li><b>A transition does not appear:</b> make sure two visual clips overlap.</li>
    <li><b>No sound:</b> check clip volume, track mute, solo buttons, master mute, and the audio mode.</li>
    <li><b>Preview is slow:</b> lower Preview Quality, use lighter transition masks, or test-export if the slowdown happens around reverse playback or heavy effects.</li>
    <li><b>Clip looks zoomed, moved, mirrored, or flipped:</b> select it and use Reset Transform or Transform &gt; Mirror / Flip &gt; Reset mirror.</li>
    <li><b>Keyframe movement is not happening:</b> make sure the clip has at least two visual keyframes at different times, and that the playhead is inside that clip.</li>
    <li><b>Keyframe button does nothing:</b> select a video or image clip first. Audio-only clips and empty timeline space cannot use visual transform keyframes.</li>
    <li><b>Text changed in more places than expected:</b> remember that Text assets are reusable. Use separate Text assets when you need different wording or styling.</li>
    <li><b>Text animation options are disabled:</b> check whether Scroll is active. Scroll disables other movement animations so they do not conflict.</li>
    <li><b>Group edit affects the wrong clips:</b> check your multi-selection before moving, deleting, copying, or pasting.</li>
    <li><b>Export does not match expectation:</b> check track visibility/mute/solo and whether the correct clips are on top.</li>
</ul>

<h2>Good Habits</h2>
<ul>
    <li>Build the rough cut first, then add transitions and fades.</li>
    <li>Keep audio cleanup near the end of the edit.</li>
    <li>Use short test exports when adding new transition styles.</li>
    <li>Create separate Text assets for different title styles so edits stay predictable.</li>
    <li>For keyframed effects, start with two keyframes first. Add extra keyframes only when the basic movement already works.</li>
    <li>Use <b>Add Default Keyframe</b> as a clean anchor when you want a clip to return to its original/default position mid-effect.</li>
    <li>Name/save project versions when trying risky changes, because autosave follows the current project instead of creating a full version history.</li>
</ul>

<p class="small">
This help window is local and offline. The help text lives in <b>helpers/editor_help.py</b>,
so it can be expanded later without changing timeline playback or export code.
</p>

</body>
</html>
"""


class TimelineEditorHelpWindow(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Timeline Editor Help")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(980, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header_row = QHBoxLayout()
        title = QLabel("Timeline Editor Help", self)
        title_font = QFont("Segoe UI", 14)
        title_font.setBold(True)
        title.setFont(title_font)

        close_btn = QPushButton("Close", self)
        close_btn.clicked.connect(self.close)

        header_row.addWidget(title)
        header_row.addStretch(1)
        header_row.addWidget(close_btn)

        self.browser = QTextBrowser(self)
        self.browser.setOpenExternalLinks(False)
        self.browser.setHtml(HELP_HTML)
        self.browser.setReadOnly(True)

        layout.addLayout(header_row)
        layout.addWidget(self.browser, 1)

        QShortcut(QKeySequence("Esc"), self, activated=self.close)
        QShortcut(QKeySequence("Ctrl+W"), self, activated=self.close)

    def show_and_raise(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()


def show_timeline_editor_help(parent: Optional[QWidget] = None) -> TimelineEditorHelpWindow:
    existing = getattr(parent, "_timeline_editor_help_window", None) if parent is not None else None
    if isinstance(existing, TimelineEditorHelpWindow):
        existing.show_and_raise()
        return existing

    window = TimelineEditorHelpWindow(parent)
    if parent is not None:
        setattr(parent, "_timeline_editor_help_window", window)
    window.show_and_raise()
    return window
