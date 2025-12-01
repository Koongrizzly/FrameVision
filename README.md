### 🌟 FrameVision 2.0 
**All-in-one Video & Photo Upscaler/Editor**  
✨ *Upscale, edit, describe, and create — in one streamlined app.*

December 1, 2025 V2.0 is finally ready
### 🌟 Release Notes
---
**🧩 NEW Features**
_Music Video Clip Creator_
- [ ] Works with every CPU or Gpu.
- [ ] Load one or multiple clips and/or images, add your music track, and create a complete music video in a couple of minutes*.
- [ ] Three settings for FX intensity or FX OFF (only hardcuts).
- [ ] One-click presets – apply 20+ settings and generate a full video in one click.
- [ ] Random seed toggle so you never get the same video. (or the same with your new edits when turned off)
- [ ] fade in/out with on/off toggle
- [ ] Built-in detector for auto selection of segments and energy peaks.
- [ ] Transitions with selection and randomization of favorites.
- [ ] Auto Ken Burns–style motion for images (+ extra settings for video).
- [ ] Global brightness/contrast/saturation normalization for smooth flow.
- [ ] Aspect ratios: 16:9 or 9:16 with Source, Fill (auto-crop), or Stretch modes.
- [ ] Multiple output resolutions supported.(source/lowest/highest + fixed 480/720/1080)
- [ ] Beat-synced Music visuals (this setting more then doubles generation time)
- choose 1 for the whole clip
- a new (random) one for every segment
- One per Section (intro verse chorus break drop outro) with user selection (200+ visuals)
- Transparency slider 0-100 (100 will only show beat synced visuals)
- [ ] Timeline Editor Tab
- Shows a detailed view of the entire track
- Visual preview of each segment with zoom and pan and right moiuse button options
- Rename added clips/images.
- Add clips or images directly to segments.
- (beta) Save or load JSON timeline files for re-use *Lowering beats per segment, long tracks, and/or adding beat-synced visuals will make generation time longer. 

_Wan 2.2 – Text & Image to Video creation_

- [ ] Optional install (~30 GB on disk).
- [ ] Minimum 16 GB VRAM (RTX Nvidea) advised for stable performance.
- [ ] (Hard-coded by model) output at 1280×708, 24 fps.
- [ ] Choose between Text to Video, Image to Video, or Video to Video modes.
- [ ] Option to use the current frame of a (playing) video as the starting frame.
- [ ] Extend generated videos by re-using the last frame to make longer clips without extra VRAM.
- [ ] Includes (random) Seed, Guidance Scale, Loras, and other fine-tuning parameters.
- [ ] Batch support for:
- [ ] TXT2VID –> multiple videos from one text prompt.
- [ ] IMG2VID –> one image with multiple random seeds or several images with same/random seed.
- [ ] Recent results view with adjustable thumbnails for quick preview and management.

_Z-Image Turbo – Enhanced Image Generator_

- [ ] Creates more realistic images, with improved spelling accuracy and better prompt adherence.
- [ ] Optional install (~30–35 GB on disk). 16 gig vram (RTX Nvidea) advised for 720p and above
- [ ] When installed, it merges seamlessly with the default SDXL Image Creator tab and adds a new Engine Loader dropdown.
- [ ] Auto-switches settings (CFG, Steps, Output Name, etc.) per engine for best results.

_Ace Music creation_

- [ ] Optional install (~6 GB on disk).
- [ ] Genre preset system with favorites for instant style composition.*
 (* ace is still hit and miss, sometimes it's good result many times it is not good result)
 First run may download extra requirement files and may look stuck in the queue for a while.
- [ ] Fast generation, only 6–8 GB VRAM required.
- [ ] Reference track with slider for style matching. (beta)
- [ ] Negatives to exclude undesired elements.
- [ ] Recents list with Play/Delete options.
- [ ] Various fine-tuning controls.


### Audio Edit – mini Audacity-style editor

- [ ] Basic tool to edit sound files
- [ ] Crop MP3/WAV
- [ ] Convert formats
- [ ] Edit metadata
- [ ] Zoom & panning
- [ ] (beta) Right-click menu: Cut / Copy / Paste / Clear / Zoom to Selection / Show Full Waveform


_Video Split & Joiner_ – combine multiple clips or split a video into parts.

_Frame Extractor_ – now has extra option to rejoin frames into a video.

_Metadata Tool_ – edit or delete metadata from videos and images (with batch support.)

**🎨 UI Improvements**

- [ ] Executable launch option – start with FrameVision.exe (works both to install or run the app) instead of start.bat (.bat remains for debugging).
- [ ] Installer Options Menu – choose optional extras (Ace-Step Music, Wan 2.2, Z-Image Turbo).
- [ ] Recent results panel – sorting (date/size/alphabet) + delete function.
- [ ] Right-click menu in Recent Results – options to: Open Folder / Rename / Show Info / Delete from Drive.
- [ ] Emoji pack with on/off toggle + hide tab labels (only show the emoji instead of full tab name) option
- [ ] Colorful tab banners – toggle for on/off and color options.
- [ ] Panel swap toggle – media player left/right.
- [ ] Sticky bottom buttons for each tab.
- [ ] Important Buttons now change background color on hover.
- [ ] All themes now have color-changing edges on hover.

**🐞 Bugfixes**

- [ ] Fixed Z-Image Turbo batch/enqueue problems.
- [ ] Ace Music AI reference track fix – stable alignment and output.
- [ ] RIFE fixed for systems without FFmpeg on PATH.
- [ ] Frame Extractor fix – proper saving for single and batch modes.

---
---

FrameVision All in one sound/image/video tool FULL feauture list
---
### 🧪 Upscaler with model loader (15+ open source upscale models for photo & video)
### ⏱️ RIFE Interpolation
### 📝 Describe anything with Qwen3 VL 2B
### 🖌️ TXT → IMG (SD15/SDXL/Z-Image)
### 🎬 WAN 2.2 5B txt/img/video to video with video extender
### 🎵 Ace Music creation
### 🖌 Remove Background + basic (sd15) inpainting
### 👽 Ask Framie, little offline chatbot, works with Qwen3VL 2B
### 🧰 Tools
- Change speed for videos (with audio sync).
- Prompt enhancer
- Thumbnail/meme creator
- Sound edit (mini audacity)
- sound mixer
- (music) videoclip creator
- Resize image/video 
- Create Gif files
- Extract frames, add frames together
- Trim videos
- Crop videos/images
- Multi rename/replace
- Metadata editor/remover
### ⏳ Batch tools for almost all tools
### 📋 Advanced Queue with seperate worker for most tools
### 🧮 System Monitor with Hud
### 🕹️ EASTER EGGS : use the app to unlock them
### 📺 in app resizable Multi media player
### 🎶 Music player with playlist & 200+ bneat synced visuals
### All kinds of extra's i forgot to mention such as random 'open app' screens with visual overlays or 20+ themes.
---

## 🛠 Installation
Run **`framevision.exe`** — it automatically:
1. Opens the installer menu if setup is incomplete.
2. Repairs missing components (like `psutil`).
3. Starts the app and background worker.

### Installer Menu Options
- 1 **check requirements** + basic 'first time ai' installer.
- 2 **Core Install:** App only (minimal).
- 3 **Full CPU Install:** Torch CPU + models/binaries.
- 4 **Full CUDA Install:** Torch CUDA + models/binaries.
- 5 **optional install** list with extra tools that have their own environment (wan 2.2, z-image,...
    These kan be skipped, or when installed, deleting their model folder and environment folder wikk safely remove them again from the app
---
Portable & offline
(install must go to c:\framevision-main, once everything is installed no more internet is needed and you can move it to any other location on the drive
One click install tries to make everything work out of the box, even for 'new to ai' users
Tooltips everywhere (can be turned of in settings tab)
Draggable middle splitter & swap ui to make the app look the way user wants
---

## 🤝 Contributing
Pull requests and feature suggestions are welcome!  
If you encounter bugs, please open an issue or use the **Bug Report** button in the app's Settings tab.

---

## 📜 License
[MIT License](LICENSE)

---

> Built by **Contrinsan**  
> installer help by **ChatGPT**  
> *This project is a living work — details evolve with releases.*

